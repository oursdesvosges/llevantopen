//! First hits: where each ray first enters a visible cell (for the 3D
//! viewer).
//!
//! The same machinery as [`Model::trace`], stopping at the first visible
//! segment: the line is searched in windows of doubling length from its
//! start, universes are only entered while nothing visible has been found,
//! and never through a hidden cell. So `first_hit` gives the start of the
//! first visible segment `trace` would give (up to the tolerance, where a
//! crossing falls within it of a window's end).

use rayon::prelude::*;

use crate::classify::{decode, Label, Status};
use crate::error::{Error, Result};
use crate::intern;
use crate::trace::{Line, Tracer, NO_SURFACE};
use crate::vec3::Vec3;
use crate::Model;

/// Which segments count as visible.
///
/// A segment inside a cell is visible when every cell on its path is: a
/// hidden filled cell hides everything in it. Segments that are not inside a
/// cell are visible as asked: gaps with `gaps`; overlaps, unsupported cells
/// and too-deep nesting with `problems`. On-surface stretches (thinner than
/// the tolerance, or along a surface) are never visible.
///
/// With `level`, rays are traced down to that universe level only
/// ([`Depth::Level`](crate::Depth::Level)) and, as MCNP's plotter shows `LEVEL n`, only
/// cells at that level are seen: a filled cell there is solid, and the
/// cells above it that are not filled show nothing (gaps and problems
/// show at any level, as asked).
///
/// Ghost cells are seen as shells over what lies behind them, never as
/// solids: a ray goes on through them, and [`Hits::ghosts`] counts the
/// ghost boundaries it crossed before its hit. A filled ghost cell's
/// contents show as `cells` says; a hidden ghost cell hides as any other.
///
/// With `veil`, gaps and overlaps that are shown are seen as veils rather
/// than solids: a ray goes on through them, and [`Hits::veils`] says per
/// ray which kinds it went through before its hit. Unsupported cells and
/// too-deep nesting still stop it.
///
/// With `problem_cells`, gaps and problems show only where they belong to
/// one of its cells ([`Visibility::owns_problem`]): those of a part looked
/// at alone, not every overlap elsewhere in the model.
#[derive(Clone, Debug)]
pub struct Visibility {
    /// By cell index ([`Model::cells`]).
    pub cells: Vec<bool>,
    pub gaps: bool,
    pub problems: bool,
    /// The universe level shown (0 is the top); `None` shows the deepest
    /// cell of every path.
    pub level: Option<usize>,
    /// By cell index, or empty for no ghosts.
    pub ghosts: Vec<bool>,
    /// Gaps and overlaps seen through, as veils.
    pub veil: bool,
    /// By cell index, the cells whose gaps and problems show; empty for
    /// those of every cell.
    pub problem_cells: Vec<bool>,
}

impl Visibility {
    /// Every cell visible, gaps and problems too.
    pub fn all(m: &Model) -> Visibility {
        Visibility {
            cells: vec![true; m.cells.len()],
            gaps: true,
            problems: true,
            level: None,
            ghosts: Vec::new(),
            veil: false,
            problem_cells: Vec::new(),
        }
    }

    /// Cells with material visible (and filled cells, so their contents can
    /// be); void cells hidden. Gaps and problems visible.
    pub fn solid(m: &Model) -> Visibility {
        Visibility {
            cells: m
                .cells
                .iter()
                .map(|c| c.material != 0 || c.fill.is_some())
                .collect(),
            gaps: true,
            problems: true,
            level: None,
            ghosts: Vec::new(),
            veil: false,
            problem_cells: Vec::new(),
        }
    }

    /// Hide the cells with these MCNP numbers (and so what fills them).
    /// [`Error::NoCell`] for the first number that is not a cell.
    pub fn hide(&mut self, m: &Model, ids: &[i64]) -> Result<()> {
        for &id in ids {
            let c = m.cell_index(id).ok_or(Error::NoCell(id))?;
            self.cells[c as usize] = false;
        }
        Ok(())
    }

    /// Hide every cell of universe `u` (MCNP number).
    /// [`Error::NoUniverse`] if there is no such universe.
    pub fn hide_universe(&mut self, m: &Model, u: i64) -> Result<()> {
        let unit = m
            .units()
            .iter()
            .find(|x| x.universe == u)
            .ok_or(Error::NoUniverse(u))?;
        for &c in &unit.cells {
            self.cells[c as usize] = false;
        }
        Ok(())
    }

    /// Make the cells with these MCNP numbers ghosts.
    /// [`Error::NoCell`] for the first number that is not a cell.
    pub fn ghost(&mut self, m: &Model, ids: &[i64]) -> Result<()> {
        for &id in ids {
            let c = m.cell_index(id).ok_or(Error::NoCell(id))?;
            if self.ghosts.is_empty() {
                self.ghosts = vec![false; m.cells.len()];
            }
            self.ghosts[c as usize] = true;
        }
        Ok(())
    }

    /// Whether cell `c` (index into [`Model::cells`]) is a ghost.
    #[inline]
    pub fn is_ghost(&self, c: u32) -> bool {
        self.ghosts.get(c as usize).copied().unwrap_or(false)
    }

    /// Whether any cell is a ghost.
    pub fn has_ghosts(&self) -> bool {
        self.ghosts.iter().any(|&g| g)
    }

    /// Whether a segment of `status` is seen as a veil.
    pub(crate) fn veils(&self, status: Status) -> bool {
        self.veil && matches!(status, Status::Gap | Status::Overlap) && self.shows(status)
    }

    /// [`Error::Invalid`] unless the flags fit a model of `n` cells.
    pub(crate) fn check(&self, n: usize) -> Result<()> {
        if self.cells.len() != n {
            return Err(Error::Invalid(format!(
                "visibility has {} cell flags for a model of {} cells",
                self.cells.len(),
                n
            )));
        }
        if !self.ghosts.is_empty() && self.ghosts.len() != n {
            return Err(Error::Invalid(format!(
                "visibility has {} ghost flags for a model of {} cells",
                self.ghosts.len(),
                n
            )));
        }
        if !self.problem_cells.is_empty() && self.problem_cells.len() != n {
            return Err(Error::Invalid(format!(
                "visibility has {} problem cell flags for a model of {} cells",
                self.problem_cells.len(),
                n
            )));
        }
        Ok(())
    }

    /// Whether a label that is not inside one cell belongs to a cell of
    /// [`Visibility::problem_cells`] (always, when it is empty): an overlap
    /// or unsupported point to one of the cells involved, a gap or too deep
    /// nesting to the filled cell its path ends at (none at level 0). True
    /// for labels inside a cell.
    pub fn owns_problem(&self, l: &Label) -> bool {
        self.owns(l.status, l.path.last().map(|s| s.cell), &l.involved)
    }

    /// [`Visibility::owns_problem`] of a label of `status`, whose path ends
    /// at `holder`, with `cells` involved.
    pub(crate) fn owns(&self, status: Status, holder: Option<u32>, cells: &[u32]) -> bool {
        let mask = |c: u32| self.problem_cells.get(c as usize).copied().unwrap_or(false);
        match status {
            _ if self.problem_cells.is_empty() => true,
            Status::Inside | Status::OnSurface => true,
            Status::Overlap | Status::Unsupported => cells.iter().any(|&c| mask(c)),
            Status::Gap | Status::TooDeep => holder.is_some_and(mask),
        }
    }

    pub(crate) fn shows(&self, status: Status) -> bool {
        match status {
            Status::Inside => true,
            Status::Gap => self.gaps,
            Status::Overlap | Status::Unsupported | Status::TooDeep => self.problems,
            Status::OnSurface => false,
        }
    }
}

/// Label id of a ray that hits nothing.
pub const NO_HIT: u32 = u32::MAX;

/// Cell of a [`Ghost`] when the ray crossed no ghost boundary.
pub const NO_CELL: u32 = u32::MAX;

/// What a ray went through, seen as veils ([`Visibility::veil`]), before
/// its hit (or along all of it, when it hits nothing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Veils {
    pub gap: bool,
    /// An overlap (which the GPU renderer, not looking for overlaps, never
    /// sees).
    pub overlap: bool,
}

impl Veils {
    pub fn any(self) -> bool {
        self.gap || self.overlap
    }

    /// This ray's veils with those of a later part of it.
    pub fn or(self, later: Veils) -> Veils {
        Veils {
            gap: self.gap || later.gap,
            overlap: self.overlap || later.overlap,
        }
    }
}

/// The ghost boundaries a ray crossed before its hit (or all of them, when
/// it hits nothing): how many, and the first one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ghost {
    /// Boundaries crossed; crossings within the tolerance of each other
    /// count once.
    pub count: u32,
    /// The ghost cell (index into [`Model::cells`]) the ray enters or
    /// leaves at the first, or [`NO_CELL`].
    pub cell: u32,
    /// Where along the ray the first is (`INFINITY` without one).
    pub t: f64,
    /// Unit normal there in the global frame, facing the ray.
    pub normal: Vec3,
}

impl Default for Ghost {
    fn default() -> Ghost {
        Ghost {
            count: 0,
            cell: NO_CELL,
            t: f64::INFINITY,
            normal: [0.0; 3],
        }
    }
}

impl Ghost {
    /// This ray's ghosts followed by those of a later part of it.
    pub fn then(self, later: Ghost) -> Ghost {
        Ghost {
            count: self.count + later.count,
            ..if self.count > 0 { self } else { later }
        }
    }
}

/// First hits of a batch of rays: per ray, the `t` where it enters its first
/// visible segment (`INFINITY` if none), that segment's label id (or
/// [`NO_HIT`]), the surface it enters through ([`NO_SURFACE`] when the ray
/// starts inside), and that surface's unit normal in the global frame,
/// facing the ray (zero without a surface).
#[derive(Clone, Debug, Default)]
pub struct Hits {
    pub t: Vec<f64>,
    pub ids: Vec<u32>,
    pub surface: Vec<u32>,
    pub normal: Vec<Vec3>,
    pub labels: Vec<Label>,
    /// Per ray, the ghost boundaries crossed before the hit; empty when the
    /// visibility has no ghosts.
    pub ghosts: Vec<Ghost>,
    /// Per ray, the veils gone through before the hit; empty without
    /// [`Visibility::veil`].
    pub veils: Vec<Veils>,
}

#[derive(Default)]
struct Part {
    chunk: intern::Chunk,
    t: Vec<f64>,
    hit: Vec<bool>,
    surface: Vec<u32>,
    normal: Vec<Vec3>,
    ghosts: Vec<Ghost>,
    veils: Vec<Veils>,
}

impl Model {
    /// First visible hit of each ray (lines in the global frame). Results
    /// are identical for any number of threads. Rays need what
    /// [`Model::trace`] needs of lines, and `vis` one flag per cell.
    ///
    /// With `skip` ([`Model::skip`] of `vis`) and a visibility that shows no
    /// gaps and no problems, only where something visible may be is
    /// searched, with the same results (`skip` is ignored otherwise, as
    /// everything is searched for gaps and problems). Compute `skip` once
    /// per visibility (~40 ms on the reference model).
    pub fn first_hits(
        &self,
        rays: &[Line],
        vis: &Visibility,
        skip: Option<&crate::Skip>,
    ) -> Result<Hits> {
        if skip.is_some_and(|s| s.worth.len() != self.cells.len()) {
            return Err(Error::Invalid("a skip of another model".into()));
        }
        let skip = skip.filter(|_| !vis.gaps && !vis.problems);
        vis.check(self.cells.len())?;
        crate::trace::check_lines(rays)?;
        let exact_nodes = if skip.is_none() && rays.len() >= 64 {
            self.exact_first_hit_nodes(vis)
        } else {
            None
        };
        let exact_ref = exact_nodes.as_deref();
        let ghosts = vis.has_ghosts();
        let parts: Vec<Part> = rays
            .par_chunks(crate::trace::chunk_len(rays.len()))
            .map_init(
                || {
                    let mut tr = Tracer::new(self, false);
                    tr.set_level(vis.level);
                    tr.hit = Some(crate::trace::HitSearch {
                        vis,
                        skip,
                        exact_nodes: exact_ref,
                        found: None,
                        ghosts: ghosts.then(Vec::new),
                        veils: Default::default(),
                    });
                    tr
                },
                |tr, chunk| {
                    let mut part = Part::default();
                    for r in chunk {
                        let found = tr.first_hit(r);
                        if ghosts {
                            let t = found.as_ref().map_or(f64::INFINITY, |h| h.t);
                            part.ghosts.push(tr.ghost(t));
                        }
                        if vis.veil {
                            part.veils.push(tr.veils());
                        }
                        match found {
                            Some(h) => {
                                part.chunk.push(&tr.keys()[h.key.0..h.key.1]);
                                part.t.push(h.t);
                                part.hit.push(true);
                                part.surface.push(h.surface);
                                part.normal.push(h.normal);
                            }
                            None => {
                                part.t.push(f64::INFINITY);
                                part.hit.push(false);
                                part.surface.push(NO_SURFACE);
                                part.normal.push([0.0; 3]);
                            }
                        }
                    }
                    part
                },
            )
            .collect();
        let mut out = Hits::default();
        let mut chunks = Vec::with_capacity(parts.len());
        let mut hit = Vec::with_capacity(rays.len());
        for p in parts {
            out.t.extend(p.t);
            out.surface.extend(p.surface);
            out.normal.extend(p.normal);
            out.ghosts.extend(p.ghosts);
            out.veils.extend(p.veils);
            hit.extend(p.hit);
            chunks.push(p.chunk);
        }
        let (ids, keys) = intern::merge(&chunks);
        let mut next = ids.into_iter();
        out.ids = hit
            .iter()
            .map(|&h| if h { next.next().unwrap() } else { NO_HIT })
            .collect();
        out.labels = keys.par_iter().map(|k| decode(k)).collect();
        Ok(out)
    }

    /// Marks which octree nodes of unit 0 can contain anything observable even
    /// when `vis.gaps` / `vis.problems` are enabled: a leaf whose sole cell has
    /// residual `PROG_TRUE`, is hidden (`!vis.cells[c] && !vis.is_ghost(c)`),
    /// unfilled, supported, and disjoint from every `u0.always` cell cannot
    /// hold any hit, ghost boundary, gap, or problem.
    fn exact_first_hit_nodes(&self, vis: &Visibility) -> Option<Vec<bool>> {
        let u0 = &self.units[0];
        let g = u0.grid.as_ref()?;
        if g.progs.is_empty() || u0.always.len() > 8 {
            return None;
        }
        for &a in &u0.always {
            let ca = &self.cells[a as usize];
            if ca.unsupported.is_some() {
                return None;
            }
        }

        let tol = self.tolerance;
        let leaf_interesting = |n: usize, bx: &crate::bbox::Aabb| -> bool {
            let node = g.nodes[n];
            if node.len != 1 {
                return true;
            }
            let first = node.first as usize;
            if g.progs.get(first) != Some(&crate::grid::PROG_TRUE) {
                return true;
            }
            let c = g.items[first];
            let cell = &self.cells[c as usize];
            if vis.cells[c as usize]
                || vis.is_ghost(c)
                || cell.fill.is_some()
                || cell.unsupported.is_some()
            {
                return true;
            }
            for &a in &u0.always {
                let v = crate::expr::eval_box(
                    self.code(a),
                    bx,
                    self.node_boxes_of(a),
                    &mut |s, want_pos| match (self.shapes[s as usize].side_over(bx, tol), want_pos)
                    {
                        (crate::surface::Side::On, _) => crate::expr::Tri::Unknown,
                        (crate::surface::Side::Pos, true) | (crate::surface::Side::Neg, false) => {
                            crate::expr::Tri::True
                        }
                        _ => crate::expr::Tri::False,
                    },
                );
                if v != crate::expr::Tri::False {
                    return true;
                }
            }
            false
        };

        let mut flags = vec![true; g.nodes.len()];
        let mut n_leaves = 0usize;
        let mut n_interesting = 0usize;
        fn walk(
            g: &crate::grid::Octree,
            n: usize,
            bx: crate::bbox::Aabb,
            flags: &mut [bool],
            n_leaves: &mut usize,
            n_interesting: &mut usize,
            leaf_interesting: &impl Fn(usize, &crate::bbox::Aabb) -> bool,
        ) -> bool {
            let node = g.nodes[n];
            if node.len != crate::grid::INNER {
                *n_leaves += 1;
                let keep = leaf_interesting(n, &bx);
                flags[n] = keep;
                if keep {
                    *n_interesting += 1;
                }
                return keep;
            }
            let mut any = false;
            for k in 0..8 {
                let child = node.first as usize + k;
                let cb = crate::grid::child_box(&bx, k);
                if walk(
                    g,
                    child,
                    cb,
                    flags,
                    n_leaves,
                    n_interesting,
                    leaf_interesting,
                ) {
                    any = true;
                }
            }
            flags[n] = any;
            any
        }
        walk(
            g,
            0,
            g.root,
            &mut flags,
            &mut n_leaves,
            &mut n_interesting,
            &leaf_interesting,
        );
        if n_leaves == 0 || (n_interesting as f64) > 0.75 * (n_leaves as f64) {
            return None;
        }
        Some(flags)
    }
}
