//! Flat bounding volume hierarchy over cell boxes (binned SAH build).

use crate::bbox::Aabb;
use crate::vec3::Vec3;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub bbox: Aabb,
    /// Leaf: first item. Inner: index of the left child (right is `+ 1`).
    pub start: u32,
    /// Leaf: number of items. Inner: 0.
    pub count: u32,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    /// Item ids, in leaf order.
    pub items: Vec<u32>,
    /// Item boxes, in leaf order.
    pub boxes: Vec<Aabb>,
}

const LEAF_MAX: usize = 4;
const BINS: usize = 16;
/// Beyond this depth splits are by count, so depth ≤ 64 + log2(n) < QUERY_STACK.
const MAX_SAH_DEPTH: usize = 64;
const QUERY_STACK: usize = 160;

impl Bvh {
    /// Build over `(id, box)` pairs; every box must be finite.
    pub fn build(prims: &[(u32, Aabb)]) -> Bvh {
        let mut bvh = Bvh::default();
        if prims.is_empty() {
            return bvh;
        }
        let mut work: Vec<(u32, Aabb, Vec3)> =
            prims.iter().map(|&(i, b)| (i, b, b.center())).collect();
        bvh.nodes.push(Node {
            bbox: Aabb::EMPTY,
            start: 0,
            count: 0,
        });
        let mut stack = vec![(0usize, 0usize, work.len(), 0usize)];
        while let Some((node, lo, hi, depth)) = stack.pop() {
            let slice = &mut work[lo..hi];
            let mut bbox = Aabb::EMPTY;
            let mut cbox = Aabb::EMPTY;
            for (_, b, c) in slice.iter() {
                bbox = bbox.union(b);
                cbox.grow_point(*c);
            }
            let n = slice.len();
            let split = if n <= LEAF_MAX {
                None
            } else if depth >= MAX_SAH_DEPTH {
                // Keep the tree shallow enough for the fixed query stack.
                Some(n / 2)
            } else {
                sah_split(slice, &bbox, &cbox)
            };
            match split {
                None => {
                    bvh.nodes[node] = Node {
                        bbox,
                        start: lo as u32,
                        count: n as u32,
                    };
                }
                Some(mid) => {
                    let left = bvh.nodes.len();
                    bvh.nodes.push(Node {
                        bbox: Aabb::EMPTY,
                        start: 0,
                        count: 0,
                    });
                    bvh.nodes.push(Node {
                        bbox: Aabb::EMPTY,
                        start: 0,
                        count: 0,
                    });
                    bvh.nodes[node] = Node {
                        bbox,
                        start: left as u32,
                        count: 0,
                    };
                    stack.push((left, lo, lo + mid, depth + 1));
                    stack.push((left + 1, lo + mid, hi, depth + 1));
                }
            }
        }
        bvh.items = work.iter().map(|w| w.0).collect();
        bvh.boxes = work.iter().map(|w| w.1).collect();
        bvh
    }

    /// The node boxes computed again from the item boxes (`boxes`, changed
    /// in place by the caller), keeping the tree: still a valid BVH over
    /// the new boxes, only less tight than a new build. Children come after
    /// their parent in `nodes`, so one pass from the end does it.
    pub fn refit(&mut self) {
        for i in (0..self.nodes.len()).rev() {
            let n = self.nodes[i];
            let b = if n.count > 0 {
                let s = n.start as usize;
                self.boxes[s..s + n.count as usize]
                    .iter()
                    .fold(Aabb::EMPTY, |a, b| a.union(b))
            } else {
                let l = n.start as usize;
                self.nodes[l].bbox.union(&self.nodes[l + 1].bbox)
            };
            self.nodes[i].bbox = b;
        }
    }

    /// Call `f` for every item whose box contains `p`.
    #[inline]
    pub fn query(&self, p: Vec3, mut f: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = [0u32; QUERY_STACK];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !node.bbox.contains(p) {
                continue;
            }
            if node.count > 0 {
                let s = node.start as usize;
                for k in s..s + node.count as usize {
                    if self.boxes[k].contains(p) {
                        f(self.items[k]);
                    }
                }
            } else {
                stack[sp] = node.start;
                stack[sp + 1] = node.start + 1;
                sp += 2;
            }
        }
    }

    /// Call `f(item, t0, t1)` for every item whose box the line `o + t·d`
    /// crosses within `[lo, hi]`, with the range of `t` inside the box.
    pub fn query_ray(&self, o: Vec3, d: Vec3, lo: f64, hi: f64, mut f: impl FnMut(u32, f64, f64)) {
        if self.nodes.is_empty() {
            return;
        }
        let inv = Aabb::ray_inv(d);
        if self.nodes[0]
            .bbox
            .ray_range_inv(o, d, inv, lo, hi)
            .is_none()
        {
            return;
        }
        let mut stack = [0u32; QUERY_STACK];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if node.count > 0 {
                let s = node.start as usize;
                for k in s..s + node.count as usize {
                    if let Some((a, b)) = self.boxes[k].ray_range_inv(o, d, inv, lo, hi) {
                        f(self.items[k], a, b);
                    }
                }
            } else {
                let left = node.start;
                let right = node.start + 1;
                if self.nodes[left as usize]
                    .bbox
                    .ray_range_inv(o, d, inv, lo, hi)
                    .is_some()
                {
                    stack[sp] = left;
                    sp += 1;
                }
                if self.nodes[right as usize]
                    .bbox
                    .ray_range_inv(o, d, inv, lo, hi)
                    .is_some()
                {
                    stack[sp] = right;
                    sp += 1;
                }
            }
        }
    }

    pub fn depth(&self) -> usize {
        fn d(b: &Bvh, i: usize) -> usize {
            let n = &b.nodes[i];
            if n.count > 0 {
                1
            } else {
                1 + d(b, n.start as usize).max(d(b, n.start as usize + 1))
            }
        }
        if self.nodes.is_empty() {
            0
        } else {
            d(self, 0)
        }
    }
}

/// Partition `slice` by the best binned-SAH plane; returns the split point,
/// or `None` if a leaf is cheaper.
fn sah_split(slice: &mut [(u32, Aabb, Vec3)], bbox: &Aabb, cbox: &Aabb) -> Option<usize> {
    let n = slice.len();
    let mut best: Option<(f64, usize, f64)> = None; // (cost, axis, split position)
    for axis in 0..3 {
        let (lo, hi) = (cbox.lo[axis], cbox.hi[axis]);
        if !(hi > lo) {
            continue;
        }
        let scale = BINS as f64 / (hi - lo);
        let mut counts = [0usize; BINS];
        let mut boxes = [Aabb::EMPTY; BINS];
        for (_, b, c) in slice.iter() {
            let k = (((c[axis] - lo) * scale) as usize).min(BINS - 1);
            counts[k] += 1;
            boxes[k] = boxes[k].union(b);
        }
        let mut right_area = [0.0; BINS];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        let mut right_cnt = [0usize; BINS];
        for k in (1..BINS).rev() {
            acc = acc.union(&boxes[k]);
            cnt += counts[k];
            right_area[k] = if cnt > 0 { acc.surface_area() } else { 0.0 };
            right_cnt[k] = cnt;
        }
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        for k in 0..BINS - 1 {
            acc = acc.union(&boxes[k]);
            cnt += counts[k];
            if cnt == 0 || right_cnt[k + 1] == 0 {
                continue;
            }
            let cost =
                acc.surface_area() * cnt as f64 + right_area[k + 1] * right_cnt[k + 1] as f64;
            if best.is_none_or(|b| cost < b.0) {
                best = Some((cost, axis, lo + (k + 1) as f64 / scale));
            }
        }
    }
    let leaf_cost = bbox.surface_area() * n as f64;
    match best {
        Some((cost, axis, pos)) if cost < leaf_cost || n > 4 * LEAF_MAX => {
            let mid = partition(slice, |w| w.2[axis] < pos);
            if mid == 0 || mid == n {
                Some(n / 2)
            } else {
                Some(mid)
            }
        }
        _ if n > 4 * LEAF_MAX => {
            // All centroids coincide: split in the middle anyway.
            Some(n / 2)
        }
        _ => None,
    }
}

fn partition<T>(s: &mut [T], pred: impl Fn(&T) -> bool) -> usize {
    let mut i = 0;
    for j in 0..s.len() {
        if pred(&s[j]) {
            s.swap(i, j);
            i += 1;
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_matches_brute_force() {
        let mut prims = Vec::new();
        for i in 0..500u32 {
            let x = (i as f64 * 0.731).sin() * 50.0;
            let y = (i as f64 * 0.377).cos() * 50.0;
            let z = (i as f64 * 0.113).sin() * 50.0;
            let r = 1.0 + (i % 7) as f64;
            prims.push((
                i,
                Aabb {
                    lo: [x - r, y - r, z - r],
                    hi: [x + r, y + r, z + r],
                },
            ));
        }
        let bvh = Bvh::build(&prims);
        for k in 0..2000 {
            let p = [
                (k as f64 * 0.17).sin() * 55.0,
                (k as f64 * 0.29).cos() * 55.0,
                (k as f64 * 0.41).sin() * 55.0,
            ];
            let mut got = Vec::new();
            bvh.query(p, |i| got.push(i));
            got.sort();
            let want: Vec<u32> = prims
                .iter()
                .filter(|(_, b)| b.contains(p))
                .map(|(i, _)| *i)
                .collect();
            assert_eq!(got, want);
        }
    }
}
