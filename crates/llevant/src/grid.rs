//! Octree of candidate cells, for universes whose cell boxes overlap a lot.
//!
//! Curved cells (sectors of a torus, say) have boxes much larger than the
//! cells, so many boxes contain any given point: ~93 at level 0 of the
//! reference model, of which one holds the point. An octree leaf keeps only
//! the cells that may hold some point of the leaf: a cell is dropped when its
//! program is false over the whole leaf box (`expr::eval_box` with
//! `Shape::side_over`), so the fast path still gives exactly the brute-force
//! labels.
//!
//! The tree is refined best-first while splitting still cuts the candidate
//! lists, down to a size set by the cells, and within a memory budget.
//!
//! Each candidate of a leaf also has its program specialized to the leaf
//! box (`expr::specialize`): within a leaf only a few surfaces of a long
//! program can change its value. Every use looks up the leaf holding the
//! exact point, so the residual is evaluated only where it is valid.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};

use crate::bbox::Aabb;
use crate::vec3::Vec3;

/// `Node::len` of an inner node.
pub(crate) const INNER: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Node {
    /// Inner node: index of the first of 8 children (bit `a` of the child
    /// number set: upper half along axis `a`). Leaf: first candidate.
    pub(crate) first: u32,
    /// Leaf: number of candidates. Inner: `INNER`.
    pub(crate) len: u32,
}

/// `Octree::progs` entry of a candidate evaluated with its whole program.
pub(crate) const PROG_FULL: u32 = u32::MAX;
/// `Octree::progs` entry of a candidate true over its whole leaf.
pub(crate) const PROG_TRUE: u32 = u32::MAX - 1;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Octree {
    pub root: Aabb,
    pub(crate) nodes: Vec<Node>,
    pub(crate) items: Vec<u32>,
    /// Per item: where its program specialized to the leaf starts in
    /// `arena` (see `expr::program_at`), or `PROG_FULL` / `PROG_TRUE`.
    /// Empty until [`Octree::specialize`].
    pub(crate) progs: Vec<u32>,
    pub(crate) arena: Vec<u32>,
}

/// What [`Octree::specialize`]'s callback made of a candidate in a leaf.
pub(crate) enum Prog {
    /// Appended to the buffer.
    Code,
    /// True over the whole leaf.
    True,
    /// Keep the whole program.
    Full,
}

/// Leaves specialized per parallel task.
const SPECIALIZE_CHUNK: usize = 1024;

/// Limits for [`Octree::build`].
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Leaves with at most this many candidates are not split.
    pub leaf_target: usize,
    /// Leaves whose smallest edge is below this are not split.
    pub min_edge: f64,
    pub max_depth: usize,
    /// Budget for `items + 2 · nodes` (in u32 words).
    pub max_words: usize,
    /// Split only when the children hold on average at most this fraction
    /// of the parent's candidates (1 splits whatever the gain).
    pub min_gain: f64,
}

/// Split anyway above this many candidates, even when one level gains
/// little: the gain often comes a level or two further down.
const FORCE_SPLIT: usize = 32;
/// Candidate entries to test per parallel batch of splits.
const BATCH_WORK: usize = 1 << 16;

/// A leaf that may still be split, ordered by volume × candidates (ties by
/// node index, so the order is total and deterministic).
struct Pending {
    cost: f64,
    node: u32,
    bbox: Aabb,
    list: Vec<u32>,
    depth: usize,
}

impl PartialEq for Pending {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Pending {
    fn cmp(&self, o: &Self) -> Ordering {
        self.cost
            .total_cmp(&o.cost)
            .then_with(|| o.node.cmp(&self.node))
    }
}

#[inline]
pub(crate) fn child_box(b: &Aabb, k: usize) -> Aabb {
    let c = b.center();
    let mut r = *b;
    for a in 0..3 {
        if k >> a & 1 == 1 {
            r.lo[a] = c[a];
        } else {
            r.hi[a] = c[a];
        }
    }
    r
}

impl Octree {
    /// Build over `root` from `cands`; `keep(cell, box)` must be true
    /// whenever the cell may hold a point of the box.
    ///
    /// Refinement is best-first: the leaf with the largest volume × candidate
    /// count is split next, so the budget goes where a uniformly spread query
    /// pays most. Batches are split in parallel and accepted in a fixed
    /// order, so the result does not depend on the thread count.
    pub fn build(
        root: Aabb,
        cands: &[u32],
        keep: impl Fn(u32, &Aabb) -> bool + Sync,
        p: &Params,
    ) -> Octree {
        let first: Vec<u32> = cands
            .par_iter()
            .with_min_len(64)
            .copied()
            .filter(|&c| keep(c, &root))
            .collect();
        let mut t = Octree {
            root,
            nodes: vec![Node { first: 0, len: 0 }],
            items: Vec::new(),
            progs: Vec::new(),
            arena: Vec::new(),
        };
        let mut words = first.len() + 2;
        let mut heap = BinaryHeap::new();
        t.settle(&mut heap, p, 0, root, first, 0);
        'refine: while !heap.is_empty() {
            // A batch with enough work to keep the threads busy.
            let mut batch = Vec::new();
            let mut work = 0;
            while let Some(e) = heap.pop() {
                work += e.list.len();
                batch.push(e);
                if work >= BATCH_WORK {
                    break;
                }
            }
            let kids: Vec<Vec<u32>> = batch
                .par_iter()
                .flat_map_iter(|e| (0..8).map(move |k| (e, k)))
                .map(|(e, k)| {
                    let cb = child_box(&e.bbox, k);
                    e.list
                        .par_iter()
                        .with_min_len(64)
                        .copied()
                        .filter(|&c| keep(c, &cb))
                        .collect()
                })
                .collect();
            let mut kids = kids.into_iter();
            let mut out_of_budget = false;
            for e in batch {
                let k: Vec<Vec<u32>> = kids.by_ref().take(8).collect();
                let total: usize = k.iter().map(Vec::len).sum();
                let gain = total as f64 / 8.0 <= p.min_gain * e.list.len() as f64;
                let delta = total as isize + 16 - e.list.len() as isize;
                out_of_budget |= words as isize + delta > p.max_words as isize;
                if out_of_budget || !(e.list.len() > FORCE_SPLIT || gain) {
                    t.make_leaf(e.node, e.list);
                    continue;
                }
                words = (words as isize + delta) as usize;
                let first = t.nodes.len() as u32;
                t.nodes[e.node as usize] = Node { first, len: INNER };
                t.nodes.extend((0..8).map(|_| Node { first: 0, len: 0 }));
                for (i, kl) in k.into_iter().enumerate() {
                    let cb = child_box(&e.bbox, i);
                    t.settle(&mut heap, p, first + i as u32, cb, kl, e.depth + 1);
                }
            }
            if out_of_budget {
                break 'refine;
            }
        }
        for e in heap.into_vec() {
            t.make_leaf(e.node, e.list);
        }
        t.items.shrink_to_fit();
        t
    }

    /// Queue a new node for splitting, or make it a leaf now.
    fn settle(
        &mut self,
        heap: &mut BinaryHeap<Pending>,
        p: &Params,
        node: u32,
        bbox: Aabb,
        list: Vec<u32>,
        depth: usize,
    ) {
        let edge = (0..3)
            .map(|a| bbox.hi[a] - bbox.lo[a])
            .fold(f64::INFINITY, f64::min);
        if list.len() > p.leaf_target && depth < p.max_depth && edge >= 2.0 * p.min_edge {
            heap.push(Pending {
                cost: bbox.volume() * list.len() as f64,
                node,
                bbox,
                list,
                depth,
            });
        } else {
            self.make_leaf(node, list);
        }
    }

    fn make_leaf(&mut self, node: u32, list: Vec<u32>) {
        self.nodes[node as usize] = Node {
            first: self.items.len() as u32,
            len: list.len() as u32,
        };
        self.items.extend(list);
    }

    /// Specialize every candidate's program to its leaf: `f(cell, box, out)`
    /// appends the residual to `out` or says why not. In parallel, with the
    /// same result for any thread count.
    pub(crate) fn specialize(&mut self, f: impl Fn(u32, &Aabb, &mut Vec<u32>) -> Prog + Sync) {
        let leaves = self.leaves();
        // Per chunk of leaves: its arena, and `(item, start in it)`.
        type Part = (Vec<u32>, Vec<(u32, u32)>);
        let parts: Vec<Part> = leaves
            .par_chunks(SPECIALIZE_CHUNK)
            .map(|chunk| {
                let mut arena = Vec::new();
                let mut progs = Vec::new();
                // Neighbouring leaves often leave a cell the same residual
                // (and different cells can share one): store each once.
                let mut seen: FxHashMap<u64, u32> = FxHashMap::default();
                for &(node, b) in chunk {
                    let n = self.nodes[node as usize];
                    for k in n.first..n.first + n.len {
                        let start = arena.len();
                        let p = match f(self.items[k as usize], &b, &mut arena) {
                            Prog::Code => {
                                let code = &arena[start..];
                                let mut h = FxHasher::default();
                                code.hash(&mut h);
                                let key = h.finish();
                                match seen.get(&key) {
                                    Some(&old)
                                        if crate::expr::program_at(&arena, old as usize)
                                            == code =>
                                    {
                                        arena.truncate(start);
                                        old
                                    }
                                    _ => {
                                        seen.insert(key, start as u32);
                                        start as u32
                                    }
                                }
                            }
                            Prog::True => PROG_TRUE,
                            Prog::Full => PROG_FULL,
                        };
                        progs.push((k, p));
                    }
                }
                (arena, progs)
            })
            .collect();
        let mut progs = vec![PROG_FULL; self.items.len()];
        let mut arena = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum());
        for (part, entries) in parts {
            let base = arena.len() as u32;
            for (k, p) in entries {
                progs[k as usize] = if p < PROG_TRUE { base + p } else { p };
            }
            arena.extend(part);
        }
        assert!(
            arena.len() < PROG_TRUE as usize,
            "residual programs too large"
        );
        self.progs = progs;
        self.arena = arena;
    }

    /// Every leaf, with its box (as `leaf` and `walk` compute it), in node
    /// order.
    pub(crate) fn leaves(&self) -> Vec<(u32, Aabb)> {
        let mut out = Vec::new();
        let mut stack = vec![(0u32, self.root)];
        while let Some((i, b)) = stack.pop() {
            let n = self.nodes[i as usize];
            if n.len != INNER {
                out.push((i, b));
                continue;
            }
            for k in 0..8 {
                stack.push((n.first + k as u32, child_box(&b, k)));
            }
        }
        out.sort_unstable_by_key(|l| l.0);
        out
    }

    /// The program of item `k` in its leaf: `None` when the cell is true
    /// over the whole leaf, else its residual, or `full` without one.
    #[inline]
    pub(crate) fn prog<'a>(&'a self, k: usize, full: &'a [u32]) -> Option<&'a [u32]> {
        match self.progs.get(k).copied().unwrap_or(PROG_FULL) {
            PROG_FULL => Some(full),
            PROG_TRUE => None,
            p => Some(crate::expr::program_at(&self.arena, p as usize)),
        }
    }

    /// Index of the first candidate of the leaf holding `p` and their
    /// number, or `None` outside the root.
    #[inline]
    pub(crate) fn leaf_span(&self, p: Vec3) -> Option<(usize, usize)> {
        if !self.root.contains(p) {
            return None;
        }
        let mut b = self.root;
        let mut n = self.nodes[0];
        while n.len == INNER {
            // Same arithmetic as `child_box`, so build and query agree on
            // every boundary.
            let c = b.center();
            let mut k = 0;
            for a in 0..3 {
                if p[a] >= c[a] {
                    k |= 1 << a;
                    b.lo[a] = c[a];
                } else {
                    b.hi[a] = c[a];
                }
            }
            n = self.nodes[(n.first + k) as usize];
        }
        Some((n.first as usize, n.len as usize))
    }

    /// Candidates of the leaf holding `p`, or `None` outside the root.
    #[inline]
    pub fn leaf(&self, p: Vec3) -> Option<&[u32]> {
        self.leaf_span(p)
            .map(|(first, n)| &self.items[first..first + n])
    }

    /// Call `f(first, candidates, t0, t1)` for every leaf the line `o + t·d`
    /// crosses within `[lo, hi]`, `first` being the index of its first
    /// candidate. Leaf boxes are widened like cell boxes, so rounding cannot
    /// open a gap between neighbouring leaves.
    pub fn walk(
        &self,
        o: Vec3,
        d: Vec3,
        lo: f64,
        hi: f64,
        mut f: impl FnMut(usize, &[u32], f64, f64),
    ) {
        let inv = Aabb::ray_inv(d);
        let range = |b: &Aabb| b.widen().ray_range_inv(o, d, inv, lo, hi);
        let Some((ra, rz)) = range(&self.root) else {
            return;
        };
        let mut stack = [(0u32, Aabb::EMPTY, 0.0f64, 0.0f64); 192];
        let mut sp = 1usize;
        stack[0] = (0, self.root, ra, rz);
        let mut overflow: Vec<(u32, Aabb, f64, f64)> = Vec::new();
        while sp > 0 || !overflow.is_empty() {
            let (n, b, a, z) = if let Some(top) = overflow.pop() {
                top
            } else {
                sp -= 1;
                stack[sp]
            };
            let node = self.nodes[n as usize];
            if node.len != INNER {
                f(
                    node.first as usize,
                    &self.items[node.first as usize..(node.first + node.len) as usize],
                    a,
                    z,
                );
                continue;
            }
            for k in 0..8 {
                let cb = child_box(&b, k);
                if let Some((ca, cz)) = range(&cb) {
                    let entry = (node.first + k as u32, cb, ca, cz);
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

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Heap memory in bytes.
    pub fn bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<Node>()
            + (self.items.len() + self.progs.len() + self.arena.len()) * 4
    }

    pub fn depth(&self) -> usize {
        fn d(t: &Octree, i: usize) -> usize {
            let n = t.nodes[i];
            if n.len != INNER {
                1
            } else {
                1 + (0..8).map(|k| d(t, n.first as usize + k)).max().unwrap()
            }
        }
        d(self, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spheres; `keep` drops a sphere when the box is entirely outside it.
    #[test]
    fn leaves_hold_every_sphere_that_contains_the_point() {
        let mut spheres = Vec::new();
        for i in 0..300 {
            let x = (i as f64 * 0.731).sin() * 40.0;
            let y = (i as f64 * 0.377).cos() * 40.0;
            let z = (i as f64 * 0.113).sin() * 40.0;
            spheres.push(([x, y, z], 2.0 + (i % 5) as f64 * 3.0));
        }
        let keep = |c: u32, b: &Aabb| {
            let (ctr, r) = spheres[c as usize];
            let d2: f64 = (0..3)
                .map(|a| (ctr[a] - ctr[a].clamp(b.lo[a], b.hi[a])).powi(2))
                .sum();
            d2 <= r * r
        };
        let ids: Vec<u32> = (0..300).collect();
        let root = Aabb {
            lo: [-60.0; 3],
            hi: [60.0; 3],
        };
        let t = Octree::build(
            root,
            &ids,
            keep,
            &Params {
                leaf_target: 2,
                min_edge: 0.5,
                max_depth: 20,
                max_words: 1 << 20,
                min_gain: 0.75,
            },
        );
        assert!(t.node_count() > 1);
        for k in 0..5000 {
            let p = [
                (k as f64 * 0.17).sin() * 59.0,
                (k as f64 * 0.29).cos() * 59.0,
                (k as f64 * 0.41).sin() * 59.0,
            ];
            let leaf = t.leaf(p).unwrap();
            for (i, (ctr, r)) in spheres.iter().enumerate() {
                let d2: f64 = (0..3).map(|a| (p[a] - ctr[a]).powi(2)).sum();
                if d2 <= r * r {
                    assert!(leaf.contains(&(i as u32)), "{p:?} sphere {i}");
                }
            }
        }
        assert!(t.leaf([100.0, 0.0, 0.0]).is_none());
    }
}
