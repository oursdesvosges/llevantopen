//! Grids of lines: probes of random direction and offset, and the pieces
//! of their lines over a union of boxes.

use super::*;

/// Probes of [`VolumeGrid::new`] and [`VolumeGrid::seeded`].
pub const DEFAULT_PROBES: usize = 12;

/// Lines over a region, in `probes` independent lattices of parallel lines.
/// `h` sets the cost: as many lines as three families parallel to x, y and
/// z with spacing `h`, so each probe's lines are
/// [`spacing`](VolumeGrid::spacing) `= h·√(probes / 3)` apart.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeGrid {
    pub region: Aabb,
    pub h: f64,
    /// Draws every probe's direction, rotation and offset: the same seed
    /// gives the same lines.
    pub seed: u64,
    /// Number of independent lattices, at least 2.
    pub probes: usize,
}

/// Cells per side of the rasters [`spacing_within`] projects boxes on.
pub(super) const SPACING_RASTER: usize = 128;

/// The spacing `h` that gives about `lines` lines through the union of
/// `boxes` within `region` (as [`VolumeGrid::h`] counts them: three
/// axis-parallel families `h` apart), for [`Model::volumes_within`]. The
/// union's area seen along each axis is measured on a raster of its
/// projection, so overlapping boxes count about once. `None` when nothing
/// of the boxes is within `region`.
pub fn spacing_within(region: &Aabb, boxes: &[Aabb], lines: f64) -> Option<f64> {
    let boxes: Vec<Aabb> = boxes
        .iter()
        .map(|b| b.intersect(region))
        .filter(|b| !b.is_empty())
        .collect();
    if boxes.is_empty() || !region.is_finite() {
        return None;
    }
    let n = SPACING_RASTER;
    let mut area = 0.0;
    for a in 0..3 {
        let (u, v) = ((a + 1) % 3, (a + 2) % 3);
        let du = (region.hi[u] - region.lo[u]).max(1e-12) / n as f64;
        let dv = (region.hi[v] - region.lo[v]).max(1e-12) / n as f64;
        let cell = |x: f64, lo: f64, d: f64| (((x - lo) / d).floor().max(0.0) as usize).min(n - 1);
        // Per raster cell, the largest share of it one box covers: exact
        // for one box, for boxes repeated and for boxes thinner than a cell;
        // where boxes overlap in part, between the largest and their union.
        let mut share = vec![0.0f64; n * n];
        for b in &boxes {
            let (u0, u1) = (
                cell(b.lo[u], region.lo[u], du),
                cell(b.hi[u], region.lo[u], du),
            );
            let (v0, v1) = (
                cell(b.lo[v], region.lo[v], dv),
                cell(b.hi[v], region.lo[v], dv),
            );
            for i in u0..=u1 {
                let lo = region.lo[u] + i as f64 * du;
                let fu = ((b.hi[u].min(lo + du) - b.lo[u].max(lo)) / du).clamp(0.0, 1.0);
                for j in v0..=v1 {
                    let lo = region.lo[v] + j as f64 * dv;
                    let fv = ((b.hi[v].min(lo + dv) - b.lo[v].max(lo)) / dv).clamp(0.0, 1.0);
                    let s = &mut share[i * n + j];
                    *s = s.max(fu * fv);
                }
            }
        }
        area += share.iter().sum::<f64>() * du * dv;
    }
    Some((area.max(1e-18) / lines.max(1.0)).sqrt())
}

/// One probe: lines along `dir` through `offset + k·spacing` along `a` and
/// `b` (integer `k`), a lattice anchored at the origin.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lattice {
    pub(super) dir: Vec3,
    pub(super) a: Vec3,
    pub(super) b: Vec3,
    pub(super) offset: [f64; 2],
    /// Integer ranges `[k0, k1)` along `a` and `b` that cover the region.
    pub(super) ka: (i64, i64),
    pub(super) kb: (i64, i64),
}

impl Lattice {
    pub(super) fn count(&self) -> usize {
        ((self.ka.1 - self.ka.0).max(0) * (self.kb.1 - self.kb.0).max(0)) as usize
    }
}

/// splitmix64, as uniform numbers in `[0, 1)`.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl VolumeGrid {
    /// A grid of [`DEFAULT_PROBES`] probes drawn from seed 1.
    pub fn new(region: Aabb, h: f64) -> VolumeGrid {
        VolumeGrid::seeded(region, h, 1)
    }

    /// A grid of [`DEFAULT_PROBES`] probes drawn from `seed`.
    pub fn seeded(region: Aabb, h: f64, seed: u64) -> VolumeGrid {
        VolumeGrid {
            region,
            h,
            seed,
            probes: DEFAULT_PROBES,
        }
    }

    /// Distance between neighbouring lines of one probe.
    pub fn spacing(&self) -> f64 {
        self.h * (self.probes as f64 / 3.0).sqrt()
    }

    /// Probe `p`: a uniformly random rotation (Shoemake) and offset.
    pub(super) fn lattice(&self, p: usize) -> Lattice {
        self.lattice_in(p, &self.region)
    }

    /// Probe `p`, its ranges covering `region` (the lines themselves do not
    /// depend on it: every region gets the same lattice).
    pub(super) fn lattice_in(&self, p: usize, region: &Aabb) -> Lattice {
        let mut rng = Rng(self.seed ^ (p as u64 + 1).wrapping_mul(0xd1b54a32d192ed03));
        let (u1, u2, u3) = (rng.next(), rng.next(), rng.next());
        let tau = std::f64::consts::TAU;
        let (r1, r2) = ((1.0 - u1).sqrt(), u1.sqrt());
        let (x, y, z, w) = (
            r1 * (tau * u2).sin(),
            r1 * (tau * u2).cos(),
            r2 * (tau * u3).sin(),
            r2 * (tau * u3).cos(),
        );
        // Columns of the rotation matrix of the quaternion.
        let a = [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ];
        let b = [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ];
        let dir = [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ];
        let s = self.spacing();
        let offset = [rng.next() * s, rng.next() * s];
        let l = Lattice {
            dir,
            a,
            b,
            offset,
            ka: (0, 0),
            kb: (0, 0),
        };
        self.lattice_over(&l, region)
    }

    /// Lattice `l` (any of its probe's), its ranges covering `region`:
    /// cheap, so that only boxes need keeping, not a lattice per box.
    pub(super) fn lattice_over(&self, l: &Lattice, region: &Aabb) -> Lattice {
        let s = self.spacing();
        let c = region.center();
        let half = [0, 1, 2].map(|i| 0.5 * (region.hi[i] - region.lo[i]));
        let range = |e: Vec3, o: f64| {
            let mid = dot(c, e);
            let r: f64 = (0..3).map(|i| e[i].abs() * half[i]).sum();
            let k = |x: f64| ((x - o) / s).ceil() as i64;
            (k(mid - r), k(mid + r))
        };
        Lattice {
            ka: range(l.a, l.offset[0]),
            kb: range(l.b, l.offset[1]),
            ..*l
        }
    }

    /// Number of lines of all probes; those that only touch a corner of the
    /// region outside it are counted but not traced.
    pub fn line_count(&self) -> usize {
        (0..self.probes).map(|p| self.lattice(p).count()).sum()
    }

    /// A point of line number `n` of lattice `l` (the line is that point
    /// plus multiples of `l.dir`).
    #[cfg(test)]
    pub(super) fn line_point(&self, l: &Lattice, n: usize) -> Vec3 {
        let nb = (l.kb.1 - l.kb.0) as usize;
        let ka = l.ka.0 + (n / nb) as i64;
        let kb = l.kb.0 + (n % nb) as i64;
        let s = self.spacing();
        add(
            scale(l.a, l.offset[0] + ka as f64 * s),
            scale(l.b, l.offset[1] + kb as f64 * s),
        )
    }
}

/// The pieces of lines over the union of `boxes` as they were made before
/// [`union_pieces`], kept as its reference in tests: each box gets the
/// lattice lines that cross it; along each line, every stretch of the union
/// is traced once, by the box it starts in (found with a BVH of the boxes).
#[cfg(test)]
pub(super) struct Patches<'a> {
    pub(super) boxes: &'a [Aabb],
    /// Over the boxes, to find the others along a line (none for one box).
    pub(super) bvh: Option<Bvh>,
}

#[cfg(test)]
impl<'a> Patches<'a> {
    pub(super) fn new(boxes: &'a [Aabb]) -> Patches<'a> {
        let bvh = (boxes.len() > 1).then(|| {
            let prims: Vec<(u32, Aabb)> = boxes
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u32, *b))
                .collect();
            Bvh::build(&prims)
        });
        Patches { boxes, bvh }
    }

    /// Range of the line through `p` along `dir` in box `i`, if any.
    pub(super) fn range(&self, i: usize, p: Vec3, dir: Vec3) -> Option<(f64, f64)> {
        self.boxes[i]
            .ray_range(p, dir, f64::NEG_INFINITY, f64::INFINITY)
            .filter(|(a, b)| b > a)
    }

    /// The stretch of the union along the line through `p` along `dir`
    /// that starts in box `i`, if one does (no other box holds that start,
    /// short of one starting there too with a lower index), as a line.
    pub(super) fn piece(&self, i: usize, p: Vec3, dir: Vec3, out: &mut Vec<Line>) {
        let Some((a, b)) = self.range(i, p, dir) else {
            return;
        };
        let mut end = b;
        if let Some(bvh) = &self.bvh {
            let mut owner = true;
            bvh.query_ray(p, dir, a, a, |j, _, _| {
                let j = j as usize;
                if owner && j != i {
                    if let Some((aj, _)) = self.range(j, p, dir) {
                        owner = !(aj < a || (aj == a && j < i));
                    }
                }
            });
            if !owner {
                return;
            }
            loop {
                let mut next = end;
                bvh.query_ray(p, dir, end, end, |j, _, _| {
                    if let Some((aj, bj)) = self.range(j as usize, p, dir) {
                        if aj <= end && bj > next {
                            next = bj;
                        }
                    }
                });
                if next <= end {
                    break;
                }
                end = next;
            }
        }
        out.push(Line::new(add(p, scale(dir, a)), dir, 0.0, end - a));
    }
}

/// Lines over boxes per chunk ([`Model::volumes_in`]), at least and at
/// most: about [`CHUNKS`] chunks in all between those.
pub(super) const BAND_MIN: usize = 4096;
pub(super) const BAND_MAX: usize = 1 << 20;

/// A chunk of [`Model::volumes_in`]: rows `rows.0..rows.1` of a probe's
/// lattice over the boxes that reach them.
pub(super) struct Band {
    pub(super) probe: usize,
    pub(super) rows: (i64, i64),
    pub(super) boxes: Vec<u32>,
}

/// The stretches of the union of boxes `ids` (of `boxes`) along the lines of
/// rows `rows` of lattice `l`, each traced once: on each line, the ranges of
/// the boxes it crosses, sorted by entry, merged while one starts before
/// (or where) the stretch so far ends. Overlapping boxes count once and
/// abutting ones make one piece, so the union is sampled exactly as one
/// region would be, with no more pieces than it needs. Lines in row, then
/// column order.
pub(super) fn union_pieces(
    grid: &VolumeGrid,
    l: &Lattice,
    boxes: &[Aabb],
    ids: &[u32],
    rows: (i64, i64),
    out: &mut Vec<Line>,
) {
    let s = grid.spacing();
    let inv = Aabb::ray_inv(l.dir);
    let point = |ka: i64, kb: i64| {
        add(
            scale(l.a, l.offset[0] + ka as f64 * s),
            scale(l.b, l.offset[1] + kb as f64 * s),
        )
    };
    // (row, column, entry, exit) of each line over each box.
    let mut spans: Vec<(i64, i64, f64, f64)> = Vec::new();
    for &i in ids {
        let b = &boxes[i as usize];
        let o = grid.lattice_over(l, b);
        for ka in o.ka.0.max(rows.0)..o.ka.1.min(rows.1) {
            for kb in o.kb.0..o.kb.1 {
                let range =
                    b.ray_range_inv(point(ka, kb), l.dir, inv, f64::NEG_INFINITY, f64::INFINITY);
                if let Some((t0, t1)) = range.filter(|(t0, t1)| t1 > t0) {
                    spans.push((ka, kb, t0, t1));
                }
            }
        }
    }
    spans.sort_unstable_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)).then(x.2.total_cmp(&y.2)));
    let mut k = 0;
    while k < spans.len() {
        let (ka, kb, start, mut end) = spans[k];
        k += 1;
        while k < spans.len() && (spans[k].0, spans[k].1) == (ka, kb) && spans[k].2 <= end {
            end = end.max(spans[k].3);
            k += 1;
        }
        let q = point(ka, kb);
        out.push(Line::new(
            add(q, scale(l.dir, start)),
            l.dir,
            0.0,
            end - start,
        ));
    }
}
