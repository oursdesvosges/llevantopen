//! Batch point classification through universes.

use std::mem;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::bbox::Aabb;
use crate::expr::{self, Tri};
use crate::intern;
use crate::lattice::Element;
use crate::model::Fill;
use crate::model::Unit;
use crate::surface::Side;
use crate::vec3::Vec3;
use crate::work::Work;
use crate::Model;

/// How a point was classified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// Inside exactly one cell at every level.
    Inside,
    /// No cell of the universe at the last level contains the point.
    Gap,
    /// Several cells of the universe at the last level contain it.
    Overlap,
    /// On a cell boundary: within the tolerance of a surface that decides it.
    OnSurface,
    /// No supported cell contains the point, and a cell that might uses an
    /// unsupported feature.
    Unsupported,
    /// The universe nesting is deeper than the limit (a cycle, or absurd input).
    TooDeep,
}

impl Status {
    /// Short name: `inside`, `gap`, `overlap`, `on-surface`, `unsupported`
    /// or `too-deep`.
    pub fn name(self) -> &'static str {
        match self {
            Status::Inside => "inside",
            Status::Gap => "gap",
            Status::Overlap => "overlap",
            Status::OnSurface => "on-surface",
            Status::Unsupported => "unsupported",
            Status::TooDeep => "too-deep",
        }
    }

    /// Overlaps, unsupported cells and too-deep nesting: what a check
    /// reports besides gaps.
    pub fn is_problem(self) -> bool {
        matches!(
            self,
            Status::Overlap | Status::Unsupported | Status::TooDeep
        )
    }
}

/// One level of a path: a cell, and the lattice element when the cell is a
/// lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Step {
    /// Index into [`Model::cells`].
    pub cell: u32,
    pub lattice: Option<[i32; 3]>,
}

/// The full description of where a point is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Label {
    pub status: Status,
    /// Cells from level 0 down. For [`Status::Inside`] the last step is the
    /// cell that holds the point (a lattice cell, with its element, when the
    /// element is filled by the lattice cell itself); otherwise the path ends
    /// at the filled cell whose universe has the problem (empty at level 0),
    /// or at a lattice element outside its fill array ([`Status::Gap`]).
    pub path: Vec<Step>,
    /// For overlaps, on-surface and unsupported points: the cells involved
    /// at the last level.
    pub involved: Vec<u32>,
}

/// Classification of a batch of points: one label id per point, into `labels`.
#[derive(Clone, Debug, Default)]
pub struct Classification {
    pub ids: Vec<u32>,
    pub labels: Vec<Label>,
}

pub(crate) const MAX_DEPTH: usize = 64;

/// A candidate found without an octree leaf: its whole program counts.
pub(crate) const NO_ITEM: u32 = u32::MAX;

const CHUNK: usize = 2048;

/// Per-thread buffers.
#[derive(Default)]
struct Scratch {
    candidates: Vec<u32>,
    /// Per candidate: its octree item (for its program specialized to the
    /// leaf), or `NO_ITEM`.
    items: Vec<u32>,
    inside: Vec<u32>,
    unknown: Vec<u32>,
    unsupported: Vec<u32>,
    key: Vec<u32>,
    /// Counters, when the work is wanted (`Model::work_points`).
    work: Option<Work>,
    /// Deepest level classified ([`Depth::Level`]); `None` for every level.
    level: Option<usize>,
}

/// How deep labels go: down to the cells that are not filled, or to a
/// universe level at most, as MCNP's plotter shows `LEVEL n`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Depth {
    /// Every level: paths end at a cell that is not filled.
    #[default]
    All,
    /// Down to universe level `n` (0 is the top): a filled cell at that
    /// level is a cell like any other (`Inside`, its path ending with it,
    /// and with the lattice element for a lattice cell), and nothing below
    /// it is looked at, so gaps and problems below are not seen.
    Level(usize),
}

impl Depth {
    /// The deepest level, `None` for every level.
    pub fn level(self) -> Option<usize> {
        match self {
            Depth::All => None,
            Depth::Level(n) => Some(n),
        }
    }
}

impl From<Option<usize>> for Depth {
    /// `None` is every level.
    fn from(level: Option<usize>) -> Depth {
        level.map_or(Depth::All, Depth::Level)
    }
}

impl Model {
    /// Classify points given in the global frame, down to `depth`: with
    /// [`Depth::Level`], labels are the full ones cut to `level + 1` steps
    /// (as [`Model::trace`]).
    ///
    /// Results are identical for any number of threads: label ids are
    /// assigned in point order.
    pub fn classify(&self, points: &[Vec3], depth: Depth) -> Classification {
        self.classify_impl(points, false, depth.level())
    }

    /// [`Model::classify`], or with `brute` the reference
    /// ([`crate::reference::classify`]).
    pub(crate) fn classify_impl(
        &self,
        points: &[Vec3],
        brute: bool,
        level: Option<usize>,
    ) -> Classification {
        if points.len() <= CHUNK {
            let mut s = Scratch {
                level,
                ..Scratch::default()
            };
            let mut out = intern::Chunk::default();
            out.ids.reserve(points.len());
            for &p in points {
                self.classify_into(p, &mut s, brute);
                out.push(&s.key);
            }
            let labels = (0..out.len()).map(|i| decode(out.key(i))).collect();
            return Classification {
                ids: out.ids,
                labels,
            };
        }
        // Each worker reuses one per-thread table across chunks so label
        // interning avoids chunk-sized hash tables, then renumbers ids in
        // exact point order.
        let threads = rayon::current_num_threads();
        let tables: Vec<Mutex<intern::Chunk>> = (0..threads).map(|_| Mutex::default()).collect();
        let spans: Vec<(usize, usize, usize)> = points
            .par_chunks(CHUNK)
            .map_init(
                || Scratch {
                    level,
                    ..Scratch::default()
                },
                |s, chunk| {
                    let t = rayon::current_thread_index().unwrap_or(0) % threads;
                    let mut table = {
                        let mut guard = tables[t].lock().unwrap_or_else(|e| e.into_inner());
                        mem::take(&mut *guard)
                    };
                    let start = table.ids.len();
                    for &p in chunk {
                        self.classify_into(p, s, brute);
                        table.push(&s.key);
                    }
                    let end = table.ids.len();
                    *tables[t].lock().unwrap_or_else(|e| e.into_inner()) = table;
                    (t, start, end)
                },
            )
            .collect();
        let tables: Vec<intern::Chunk> = tables
            .into_iter()
            .map(|t| t.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let (canon, keys) = intern::merge(&tables);
        let mut base = Vec::with_capacity(tables.len());
        let mut at = 0;
        for t in &tables {
            base.push(at);
            at += t.ids.len();
        }
        let mut number = vec![u32::MAX; keys.len()];
        let mut order: Vec<u32> = Vec::with_capacity(keys.len());
        let mut ids = Vec::with_capacity(canon.len());
        for &(t, start, end) in &spans {
            for &c in &canon[base[t] + start..base[t] + end] {
                let n = &mut number[c as usize];
                if *n == u32::MAX {
                    *n = order.len() as u32;
                    order.push(c);
                }
                ids.push(*n);
            }
        }
        let labels = order
            .par_iter()
            .map(|&c| decode(keys[c as usize]))
            .collect();
        Classification { ids, labels }
    }

    /// Classify one point (convenience for tests and small jobs).
    pub fn classify_point(&self, p: Vec3) -> Label {
        let mut s = Scratch::default();
        self.classify_into(p, &mut s, false);
        decode(&s.key)
    }

    /// Point `p`, whose label is `label`, in the frame of the universe that
    /// holds step `level` of its path (through the fills and lattice
    /// elements above it, as classification goes down); `p` itself for
    /// level 0. `None` if the path is not that deep.
    pub fn point_at_level(&self, label: &Label, level: usize, p: Vec3) -> Option<Vec3> {
        if level >= label.path.len().max(1) {
            return None;
        }
        let mut q = p;
        for s in &label.path[..level] {
            match self.descend(s.cell, q) {
                Descend::Into { point, .. } => q = point,
                _ => return None,
            }
        }
        Some(q)
    }

    /// Classify `p` as [`Model::classify`] does, adding its work to `w`.
    #[cfg(feature = "bench")]
    pub(crate) fn classify_counted(&self, p: Vec3, w: &mut Work) {
        let mut s = Scratch {
            work: Some(std::mem::take(w)),
            ..Scratch::default()
        };
        self.classify_into(p, &mut s, false);
        *w = s.work.take().unwrap();
    }

    /// Classify `p`, leaving its encoded label in `s.key`:
    /// `[status, nsteps, (cell, has_lattice, i, j, k)*, cells*]`.
    fn classify_into(&self, p: Vec3, s: &mut Scratch, brute: bool) {
        s.key.clear();
        s.key.push(0);
        s.key.push(0);
        let mut q = p;
        let mut unit = 0u32;
        for depth in 0..MAX_DEPTH {
            let u = &self.units[unit as usize];
            s.candidates.clear();
            s.items.clear();
            // Node boxes are only exact inside the region they were built for.
            let in_roi = u.roi.contains(q);
            let use_boxes = !brute && in_roi;
            if !brute && in_roi {
                self.candidates(u, q, &mut s.candidates, &mut s.items);
            } else {
                // Outside the region the boxes were clipped to (only possible
                // at level 0): every cell, or beyond the problem box only its
                // `IMP:N=0` cells (brute force follows the same rule).
                let cells = if in_roi { &u.cells } else { u.outside() };
                s.candidates.extend_from_slice(cells);
                s.items.resize(s.candidates.len(), NO_ITEM);
            }
            s.inside.clear();
            s.unknown.clear();
            s.unsupported.clear();
            if let Some(w) = s.work.as_mut() {
                let l = w.level(depth);
                l.visits += 1;
                l.candidates += s.candidates.len() as u64;
            }
            for (&c, &item) in s.candidates.iter().zip(&s.items) {
                let cell = &self.cells[c as usize];
                if cell.unsupported.is_some() {
                    s.unsupported.push(c);
                    continue;
                }
                if self.is_lattice(c) {
                    // A lattice fills all of its universe.
                    s.inside.push(c);
                    continue;
                }
                // The program specialized to the octree leaf holding `q`:
                // none when the cell is true over the whole leaf.
                let code = match (item, &u.grid) {
                    (NO_ITEM, _) | (_, None) => Some(self.code(c)),
                    (k, Some(g)) => g.prog(k as usize, self.code(c)),
                };
                let boxes = use_boxes.then(|| self.node_boxes_of(c));
                let v = match (code, s.work.as_mut()) {
                    (None, _) => Tri::True,
                    (Some(code), None) => self.eval_code(code, q, boxes),
                    (Some(code), Some(w)) => {
                        let mut n = 0;
                        let v = self.eval_code_counted(code, q, boxes, &mut n);
                        w.level(depth).surface_evals += n;
                        v
                    }
                };
                match v {
                    Tri::True => s.inside.push(c),
                    Tri::Unknown => s.unknown.push(c),
                    Tri::False => {}
                }
            }
            // Unsupported cells cannot be evaluated: they only decide points
            // that no supported cell claims (in a valid model a point claimed
            // by a supported cell cannot also be in them).
            let status = if s.inside.len() > 1 {
                Some((Status::Overlap, &mut s.inside))
            } else if s.inside.len() == 1 {
                None
            } else if !s.unsupported.is_empty() {
                Some((Status::Unsupported, &mut s.unsupported))
            } else if !s.unknown.is_empty() {
                Some((Status::OnSurface, &mut s.unknown))
            } else {
                Some((Status::Gap, &mut s.unknown))
            };
            if let Some((status, cells)) = status {
                cells.sort_unstable();
                s.key[0] = status as u32;
                s.key.extend_from_slice(cells);
                return;
            }
            let c = s.inside[0];
            s.key[1] += 1;
            s.key.push(c);
            match self.descend(c, q) {
                Descend::Stop => {
                    s.key.extend_from_slice(&[0, 0, 0, 0]);
                    s.key[0] = Status::Inside as u32;
                    return;
                }
                Descend::Element(ix, status) => {
                    // A lattice element filled by the lattice cell itself,
                    // or one outside the fill array.
                    s.key
                        .extend_from_slice(&[1, ix[0] as u32, ix[1] as u32, ix[2] as u32]);
                    s.key[0] = status as u32;
                    return;
                }
                Descend::Into {
                    unit: u,
                    point,
                    lattice,
                } => {
                    match lattice {
                        Some(ix) => {
                            s.key
                                .extend_from_slice(&[1, ix[0] as u32, ix[1] as u32, ix[2] as u32])
                        }
                        None => s.key.extend_from_slice(&[0, 0, 0, 0]),
                    }
                    // The deepest level asked for: the filled cell is a cell.
                    if s.level.is_some_and(|k| depth >= k) {
                        s.key[0] = Status::Inside as u32;
                        return;
                    }
                    q = point;
                    unit = u;
                }
            }
        }
        s.key[0] = Status::TooDeep as u32;
    }
}

impl Model {
    /// Value of a cell's program (whole, or specialized to an octree leaf
    /// holding `q`) at `q`.
    #[inline]
    pub(crate) fn eval_code(&self, code: &[u32], q: Vec3, boxes: Option<&[[Aabb; 2]]>) -> Tri {
        let tol = self.tolerance;
        expr::eval(code, q, boxes, &mut |surf, want_pos| match (
            self.shapes[surf as usize].side(q, tol),
            want_pos,
        ) {
            (Side::On, _) => Tri::Unknown,
            (Side::Pos, true) | (Side::Neg, false) => Tri::True,
            _ => Tri::False,
        })
    }

    /// [`Model::eval_code`], adding the surfaces it tests to `n`.
    pub(crate) fn eval_code_counted(
        &self,
        code: &[u32],
        q: Vec3,
        boxes: Option<&[[Aabb; 2]]>,
        n: &mut u64,
    ) -> Tri {
        let tol = self.tolerance;
        expr::eval(code, q, boxes, &mut |surf, want_pos| {
            *n += 1;
            match (self.shapes[surf as usize].side(q, tol), want_pos) {
                (Side::On, _) => Tri::Unknown,
                (Side::Pos, true) | (Side::Neg, false) => Tri::True,
                _ => Tri::False,
            }
        })
    }

    /// Cells of `u` that may hold `q` (in the unit's region): the octree
    /// leaf or the BVH, plus the cells checked everywhere; with, for each,
    /// its octree item or `NO_ITEM`.
    #[inline]
    fn candidates(&self, u: &Unit, q: Vec3, out: &mut Vec<u32>, items: &mut Vec<u32>) {
        match u.grid.as_ref().and_then(|g| g.leaf_span(q).map(|s| (g, s))) {
            Some((g, (first, n))) => {
                for k in first..first + n {
                    let c = g.items[k];
                    if self.cells[c as usize].bbox.contains(q) {
                        out.push(c);
                        items.push(k as u32);
                    }
                }
            }
            None => u.bvh.query(q, |c| out.push(c)),
        }
        out.extend_from_slice(&u.always);
        items.resize(out.len(), NO_ITEM);
    }
}

/// Where a point goes after the cell holding it.
enum Descend {
    /// The cell is not filled.
    Stop,
    /// A lattice element that ends the path, with the status to report.
    Element([i32; 3], Status),
    Into {
        unit: u32,
        point: Vec3,
        lattice: Option<[i32; 3]>,
    },
}

impl Model {
    #[inline]
    fn descend(&self, c: u32, q: Vec3) -> Descend {
        match &self.cells[c as usize].fill {
            None => Descend::Stop,
            Some(Fill::Universe(t)) => Descend::Into {
                unit: t.unit,
                point: t.apply(q),
                lattice: None,
            },
            Some(Fill::Lattice(l)) => {
                let lat = &self.lattices[*l as usize];
                let ix = lat.index(q);
                match lat.element(ix) {
                    Element::Fill(t) => Descend::Into {
                        unit: t.unit,
                        point: t.apply(lat.to_element(q, ix)),
                        lattice: Some(ix),
                    },
                    Element::Own => Descend::Element(ix, Status::Inside),
                    Element::Missing => Descend::Element(ix, Status::Gap),
                }
            }
        }
    }
}

pub(crate) fn decode(k: &[u32]) -> Label {
    let status = match k[0] {
        0 => Status::Inside,
        1 => Status::Gap,
        2 => Status::Overlap,
        3 => Status::OnSurface,
        4 => Status::Unsupported,
        _ => Status::TooDeep,
    };
    let n = k[1] as usize;
    let path = (0..n)
        .map(|i| {
            let w = &k[2 + 5 * i..7 + 5 * i];
            Step {
                cell: w[0],
                lattice: (w[1] == 1).then_some([w[2] as i32, w[3] as i32, w[4] as i32]),
            }
        })
        .collect();
    Label {
        status,
        path,
        involved: k[2 + 5 * n..].to_vec(),
    }
}

impl Label {
    /// The cell holding the point, for [`Status::Inside`].
    pub fn cell(&self) -> Option<u32> {
        (self.status == Status::Inside)
            .then(|| self.path.last().map(|s| s.cell))
            .flatten()
    }
}
