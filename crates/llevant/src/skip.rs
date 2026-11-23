//! Where a ray can meet something visible: the parts of each universe that
//! hold a cell worth entering, so that first hits can skip the rest.
//!
//! A cell is *worth entering* under a [`Visibility`] when it is visible and
//! something visible may be reached through it: it is not filled, or what
//! fills it (a universe, or some element of a lattice) has a cell worth
//! entering. A ghost cell always is (its shell shows). With a level shown,
//! a cell at that level is (filled or not), one above it only if what fills
//! it is: a universe reached at several levels counts at each of them. A point in no such cell is either in a cell that shows nothing
//! (hidden, or holding only hidden cells) or in a gap or a problem. So where
//! gaps and problems are not shown, nothing visible lies beyond the boxes of
//! the cells worth entering, nor in octree nodes whose leaves hold none.
//!
//! The GPU renderer always skips there (it is approximate by design and
//! reports gaps only where it does not skip); [`Model::first_hits`] only
//! when gaps and problems are hidden, where skipping is exact.

use rayon::prelude::*;

use crate::bbox::Aabb;
use crate::grid::INNER;
use crate::hit::Visibility;
use crate::lattice::LatticeFill;
use crate::model::Fill;
use crate::Model;

/// A box holding everything (finite, so that slab tests stay finite too).
const EVERYWHERE: Aabb = Aabb {
    lo: [-1e300; 3],
    hi: [1e300; 3],
};

/// What [`Model::skip`] finds for one visibility.
#[derive(Clone, Debug)]
pub struct Skip {
    /// Per cell: worth entering.
    pub(crate) worth: Vec<bool>,
    /// Per unit: the box (in its frame) of its cells worth entering, empty
    /// when it has none.
    pub(crate) boxes: Vec<Aabb>,
    /// Per unit with an octree: per node, whether some leaf below holds a
    /// candidate worth entering. `None` without an octree, or when a cell
    /// checked everywhere (not in the leaves) is worth entering: then no
    /// node can be skipped.
    pub(crate) nodes: Vec<Option<Vec<bool>>>,
    /// Per unit: whether few enough of its octree leaves hold a cell worth
    /// entering for walking them along a line to pay ([`Skip::spans`]);
    /// otherwise its box alone is used.
    pub(crate) walk: Vec<bool>,
}

/// Largest share of an octree's leaves holding a cell worth entering for
/// [`Skip::spans`] to walk them: in views where most of a model shows, the
/// walk cost more than it skipped (solid views of the reference model
/// 10-40 % slower on the CPU).
const WALK_SHARE: f64 = 0.25;

impl Skip {
    /// Whether cell `c` (index into [`Model::cells`]) is worth entering.
    pub fn worth(&self, c: u32) -> bool {
        self.worth[c as usize]
    }

    /// Share of universe 0's cells worth entering: how much of a view can
    /// show (small when one part is isolated).
    pub fn shown_share(&self, m: &Model) -> f64 {
        let cells = &m.units[0].cells;
        let n = cells.iter().filter(|&&c| self.worth[c as usize]).count();
        n as f64 / cells.len().max(1) as f64
    }

    /// Box of unit `u`'s cells worth entering (empty if none).
    pub fn unit_box(&self, u: u32) -> &Aabb {
        &self.boxes[u as usize]
    }

    /// Per node of unit `u`'s octree, whether it may hold a cell worth
    /// entering; `None` when nothing in the unit can be skipped by node.
    pub fn nodes(&self, u: u32) -> Option<&[bool]> {
        self.nodes[u as usize].as_deref()
    }

    /// Same as [`Skip::spans`], reusing caller buffers `raw` and `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spans_into(
        &self,
        m: &Model,
        u: u32,
        o: crate::vec3::Vec3,
        d: crate::vec3::Vec3,
        range: (f64, f64),
        pad: f64,
        raw: &mut Vec<(f64, f64)>,
        out: &mut Vec<(f64, f64)>,
    ) {
        let grid = m.units[u as usize]
            .grid
            .as_deref()
            .filter(|_| self.walk[u as usize]);
        octree_spans_into(
            &self.boxes[u as usize],
            grid,
            self.nodes(u),
            o,
            d,
            range,
            pad,
            raw,
            out,
        );
    }

    /// The parts of `[lo, hi]` of the line `o + t·d` in unit `u` (in its
    /// frame) where a cell worth entering may be, each grown by `pad` on
    /// both sides, in order and merged: within the unit's box of such
    /// cells, less the octree nodes that hold none (leaf boxes widened as
    /// cell boxes are).
    #[allow(dead_code)]
    pub(crate) fn spans(
        &self,
        m: &Model,
        u: u32,
        o: crate::vec3::Vec3,
        d: crate::vec3::Vec3,
        range: (f64, f64),
        pad: f64,
    ) -> Vec<(f64, f64)> {
        let mut raw = Vec::new();
        let mut out = Vec::new();
        self.spans_into(m, u, o, d, range, pad, &mut raw, &mut out);
        out
    }
}

/// Compute ordered, `pad`-widened, merged `t` spans within `outer` where `nodes`
/// indicates an octree leaf may hold something relevant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn octree_spans_into(
    outer: &Aabb,
    grid: Option<&crate::grid::Octree>,
    nodes: Option<&[bool]>,
    o: crate::vec3::Vec3,
    d: crate::vec3::Vec3,
    (lo, hi): (f64, f64),
    pad: f64,
    raw: &mut Vec<(f64, f64)>,
    out: &mut Vec<(f64, f64)>,
) {
    raw.clear();
    out.clear();
    let inv = Aabb::ray_inv(d);
    let Some((a, b)) = outer.widen().ray_range_inv(o, d, inv, lo, hi) else {
        return;
    };
    match (grid, nodes) {
        (Some(g), Some(flags)) => {
            // Outside the octree's root, keep the span unchanged.
            match g.root.widen().ray_range_inv(o, d, inv, a, b) {
                None => raw.push((a, b)),
                Some((r0, r1)) => {
                    if a < r0 {
                        raw.push((a, r0));
                    }
                    if r1 < b {
                        raw.push((r1, b));
                    }
                    if flags[0] {
                        let mut stack = [(0usize, Aabb::EMPTY, 0.0f64, 0.0f64); 192];
                        let mut sp = 1usize;
                        stack[0] = (0, g.root, r0, r1);
                        let mut overflow: Vec<(usize, Aabb, f64, f64)> = Vec::new();
                        while sp > 0 || !overflow.is_empty() {
                            let (n, bx, ra, rb) = if let Some(top) = overflow.pop() {
                                top
                            } else {
                                sp -= 1;
                                stack[sp]
                            };
                            let node = g.nodes[n];
                            if node.len != INNER {
                                raw.push((ra, rb));
                                continue;
                            }
                            for k in 0..8 {
                                let child = node.first as usize + k;
                                if !flags[child] {
                                    continue;
                                }
                                let cb = crate::grid::child_box(&bx, k);
                                if let Some((ca, cb_t)) =
                                    cb.widen().ray_range_inv(o, d, inv, r0, r1)
                                {
                                    let entry = (child, cb, ca, cb_t);
                                    if sp < stack.len() {
                                        stack[sp] = entry;
                                        sp += 1;
                                    } else {
                                        overflow.push(entry);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => raw.push((a, b)),
    }
    raw.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
    for &(s, e) in raw.iter() {
        let (s, e) = ((s - pad).max(lo), (e + pad).min(hi));
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
}

impl Model {
    /// Where `vis` leaves something to see (see the module documentation).
    /// [`crate::Error::Invalid`] if `vis` does not have one flag per cell.
    pub fn skip(&self, vis: &Visibility) -> crate::Result<Skip> {
        vis.check(self.cells.len())?;
        // Units in an order where every unit comes after those it fills
        // (the fill graph has no cycles), then cells and units bottom-up.
        let order = self.fill_order();
        let n_units = self.units.len();
        // With a level, the levels each unit is reached at (a bit each, up
        // to the level shown), from the top down.
        let mut levels = vec![0u64; n_units];
        if let Some(k) = vis.level {
            let above = (1u64 << k.min(63)) - 1;
            levels[0] = 1;
            for &u in order.iter().rev() {
                let down = (levels[u as usize] & above) << 1;
                if down != 0 {
                    for v in self.unit_children(u as usize) {
                        levels[v as usize] |= down;
                    }
                }
            }
        }
        let mut unit_worth: Vec<Option<bool>> = vec![None; n_units];
        let mut worth = vec![false; self.cells.len()];
        for &u in &order {
            // Whether the unit is reached at the level shown, and above it.
            let (at_level, above) = match vis.level {
                Some(k) => {
                    let (k, l) = (k.min(63), levels[u as usize]);
                    ((l >> k) & 1 == 1, l & ((1u64 << k) - 1) != 0)
                }
                None => (true, true),
            };
            let fills = |t: &crate::lattice::Target| unit_worth[t.unit as usize].unwrap_or(true);
            let mut any = false;
            for &c in &self.units[u as usize].cells {
                let cell = &self.cells[c as usize];
                // Whether something visible may be reached through what
                // fills the cell (`None` when it is not filled).
                let contents = match &cell.fill {
                    None => None,
                    Some(Fill::Universe(t)) => Some(fills(t)),
                    Some(Fill::Lattice(l)) => Some(match &self.lattices[*l as usize].fill {
                        LatticeFill::Uniform(t) => fills(t),
                        LatticeFill::Array { elements, .. } => elements.iter().any(|e| match e {
                            // The lattice cell's own material, which shows
                            // above the level shown only without one.
                            None => vis.level.is_none(),
                            Some(t) => fills(t),
                        }),
                    }),
                };
                let w = vis.cells[c as usize]
                    && ((vis.is_ghost(c) && (at_level || above))
                        || match (vis.level, contents) {
                            (None, None) => true,
                            (None, Some(inner)) => inner,
                            (Some(_), None) => at_level,
                            (Some(_), Some(inner)) => at_level || (above && inner),
                        });
                worth[c as usize] = w;
                any |= w;
            }
            unit_worth[u as usize] = Some(any);
        }
        let boxes: Vec<Aabb> = self
            .units
            .iter()
            .enumerate()
            .map(|(k, u)| {
                // Universe 0's boxes stop at its region; beyond it, the cells
                // checked there hold every point.
                if k == 0 && u.outside().iter().any(|&c| worth[c as usize]) {
                    return EVERYWHERE;
                }
                u.cells
                    .iter()
                    .filter(|&&c| worth[c as usize])
                    .fold(Aabb::EMPTY, |b, &c| b.union(&self.cells[c as usize].bbox))
            })
            .collect();
        let nodes: Vec<Option<Vec<bool>>> = self
            .units
            .par_iter()
            .map(|u| {
                let g = u.grid.as_ref()?;
                if u.always.iter().any(|&c| worth[c as usize]) {
                    return None;
                }
                let mut flag: Vec<bool> = g
                    .nodes
                    .par_iter()
                    .map(|n| {
                        n.len != INNER
                            && g.items[n.first as usize..(n.first + n.len) as usize]
                                .iter()
                                .any(|&c| worth[c as usize])
                    })
                    .collect();
                // Children come after their parent: fill inner nodes from
                // the back.
                for i in (0..g.nodes.len()).rev() {
                    let n = g.nodes[i];
                    if n.len == INNER {
                        let k = n.first as usize;
                        flag[i] = flag[k..k + 8].iter().any(|&f| f);
                    }
                }
                Some(flag)
            })
            .collect();
        let walk: Vec<bool> = self
            .units
            .iter()
            .zip(&nodes)
            .map(|(u, f)| match (&u.grid, f) {
                (Some(g), Some(f)) => {
                    let (all, some) = g
                        .nodes
                        .par_iter()
                        .zip(f.par_iter())
                        .filter(|(n, _)| n.len != INNER)
                        .map(|(_, &w)| (1usize, w as usize))
                        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
                    (some as f64) <= WALK_SHARE * all as f64
                }
                _ => false,
            })
            .collect();
        Ok(Skip {
            worth,
            boxes,
            nodes,
            walk,
        })
    }

    /// How many universe levels the model has: 1 without fills, plus one
    /// per step of its longest chain of fills (the levels 0 to n - 1 that
    /// [`Visibility::level`] can show).
    pub fn levels(&self) -> usize {
        // Bottom-up: each unit after those it fills.
        let mut below = vec![0usize; self.units.len()];
        for u in self.fill_order() {
            let deepest = self
                .unit_children(u as usize)
                .iter()
                .map(|&v| below[v as usize] + 1)
                .max()
                .unwrap_or(0);
            below[u as usize] = deepest;
        }
        1 + below.first().copied().unwrap_or(0)
    }

    /// The units unit `u`'s cells fill, once each.
    fn unit_children(&self, u: usize) -> Vec<u32> {
        let mut v = Vec::new();
        for &c in &self.units[u].cells {
            match &self.cells[c as usize].fill {
                None => {}
                Some(Fill::Universe(t)) => v.push(t.unit),
                Some(Fill::Lattice(l)) => match &self.lattices[*l as usize].fill {
                    LatticeFill::Uniform(t) => v.push(t.unit),
                    LatticeFill::Array { elements, .. } => {
                        v.extend(elements.iter().flatten().map(|t| t.unit))
                    }
                },
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Unit indices, each after every unit its cells fill.
    fn fill_order(&self) -> Vec<u32> {
        let n = self.units.len();
        let mut state = vec![0u8; n];
        let mut out = Vec::with_capacity(n);
        let children = |u: usize| self.unit_children(u);
        for root in 0..n {
            if state[root] != 0 {
                continue;
            }
            // Iterative post-order.
            let mut stack = vec![(root, children(root), 0usize)];
            state[root] = 1;
            while let Some((u, kids, k)) = stack.last_mut() {
                if *k < kids.len() {
                    let v = kids[*k] as usize;
                    *k += 1;
                    if state[v] == 0 {
                        state[v] = 1;
                        let kv = children(v);
                        stack.push((v, kv, 0));
                    }
                } else {
                    state[*u] = 2;
                    out.push(*u as u32);
                    stack.pop();
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GridMode, Options};

    /// Two boxes filled with universe 2 (a sphere in void), one box of
    /// material, the void around them and the graveyard.
    const TWO: &str = "\
t
1 0 -1 fill=2 imp:n=1
2 0 -2 fill=2 (40 0 0) imp:n=1
3 1 -1.0 -3 imp:n=1
4 0 1 2 3 -9 imp:n=1
9 0 9 imp:n=0
21 1 -1.0 -21 u=2 imp:n=1
22 0 21 u=2 imp:n=1

1 rpp -10 10 -10 10 -10 10
2 rpp 30 50 -10 10 -10 10
3 rpp -10 10 30 50 -10 10
9 so 200
21 so 3

m1 1001.80c 1
";

    fn ids(m: &Model, s: &Skip) -> Vec<i64> {
        (0..m.cells.len() as u32)
            .filter(|&c| s.worth(c))
            .map(|c| m.cells[c as usize].id)
            .collect()
    }

    #[test]
    fn worth_entering_follows_fills_and_visibility() {
        let m = Model::from_text(TWO, &Options::default().with_grid(GridMode::Always));
        let solid = Visibility::solid(&m);
        let s = m.skip(&solid).unwrap();
        assert_eq!(ids(&m, &s), vec![1, 2, 3, 21]);
        // Hiding the sphere leaves nothing to see in the boxes that hold it.
        let mut vis = solid.clone();
        vis.hide(&m, &[21]).unwrap();
        let s = m.skip(&vis).unwrap();
        assert_eq!(ids(&m, &s), vec![3]);
        let u0 = &s.boxes[0];
        assert!(u0.lo[1] >= 30.0 - 1e-6 && u0.hi[0] <= 10.0 + 1e-6, "{u0:?}");
        // Hiding one holder: only the other one and the material box.
        let mut vis = solid.clone();
        vis.hide(&m, &[1]).unwrap();
        let s = m.skip(&vis).unwrap();
        assert_eq!(ids(&m, &s), vec![2, 3, 21]);
        let u2 = m.units.iter().position(|u| u.universe == 2).unwrap();
        assert!(!s.unit_box(u2 as u32).is_empty());
    }

    #[test]
    fn levels_count_the_longest_chain_of_fills() {
        let m = Model::from_text(TWO, &Options::default());
        assert_eq!(m.levels(), 2);
    }

    #[test]
    fn octree_nodes_hold_their_leaves() {
        let m = Model::from_text(TWO, &Options::default().with_grid(GridMode::Always));
        let mut vis = Visibility::solid(&m);
        vis.hide(&m, &[1, 2]).unwrap();
        let s = m.skip(&vis).unwrap();
        let g = m.units[0].grid.as_ref().expect("an octree at level 0");
        let flags = s.nodes(0).expect("node flags");
        assert_eq!(flags.len(), g.nodes.len());
        // The root holds cell 3; some leaf holds nothing worth entering.
        assert!(flags[0]);
        assert!(flags.iter().any(|&f| !f));
        for (i, n) in g.nodes.iter().enumerate() {
            let below = if n.len == INNER {
                (0..8).any(|k| flags[n.first as usize + k])
            } else {
                g.items[n.first as usize..(n.first + n.len) as usize]
                    .iter()
                    .any(|&c| s.worth(c))
            };
            assert_eq!(flags[i], below, "node {i}");
        }
        // Everything hidden: nothing anywhere.
        let none = Visibility {
            cells: vec![false; m.cells.len()],
            gaps: false,
            problems: false,
            level: None,
            ghosts: Vec::new(),
            veil: false,
            problem_cells: Vec::new(),
        };
        let s = m.skip(&none).unwrap();
        assert!(s.boxes.iter().all(Aabb::is_empty));
        assert!(!s.nodes(0).unwrap()[0]);
    }
}
