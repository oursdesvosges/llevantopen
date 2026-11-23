//! Conservative axis-aligned bounding boxes of cells.
//!
//! A box is shrunk by each leaf constraint of the cell program with interval
//! constraint propagation: for a quadric (or plane) `g(x) ≤ 0` and each axis,
//! the other coordinates are bounded by intervals and the resulting interval
//! quadratic inequality is solved for that axis. AND nodes repeat this over
//! their children a few rounds; OR nodes take the hull of their children.
//! Every step rounds outwards, so the result always contains the cell.

#[cfg(test)]
use crate::expr::NO_BOX;
use crate::expr::{MASK, NODE, OR, POS};
use crate::lp::{self, Constraint};
use crate::surface::{Quadric, Shape};
use crate::transform::Affine;
use crate::vec3::{dot, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Aabb {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl Aabb {
    pub const INFINITE: Aabb = Aabb {
        lo: [f64::NEG_INFINITY; 3],
        hi: [f64::INFINITY; 3],
    };
    pub const EMPTY: Aabb = Aabb {
        lo: [f64::INFINITY; 3],
        hi: [f64::NEG_INFINITY; 3],
    };

    #[inline]
    pub fn is_empty(&self) -> bool {
        (0..3).any(|i| !(self.lo[i] <= self.hi[i]))
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        !self.is_empty() && (0..3).all(|i| self.lo[i].is_finite() && self.hi[i].is_finite())
    }

    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        (p[0] >= self.lo[0])
            & (p[0] <= self.hi[0])
            & (p[1] >= self.lo[1])
            & (p[1] <= self.hi[1])
            & (p[2] >= self.lo[2])
            & (p[2] <= self.hi[2])
    }

    /// Whether the closed boxes share a point.
    #[inline]
    pub fn overlaps(&self, o: &Aabb) -> bool {
        (0..3).all(|i| self.lo[i] <= o.hi[i] && o.lo[i] <= self.hi[i])
    }

    pub fn intersect(&self, o: &Aabb) -> Aabb {
        let mut r = *self;
        for i in 0..3 {
            r.lo[i] = r.lo[i].max(o.lo[i]);
            r.hi[i] = r.hi[i].min(o.hi[i]);
        }
        r
    }

    pub fn union(&self, o: &Aabb) -> Aabb {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        let mut r = *self;
        for i in 0..3 {
            r.lo[i] = r.lo[i].min(o.lo[i]);
            r.hi[i] = r.hi[i].max(o.hi[i]);
        }
        r
    }

    pub fn grow_point(&mut self, p: Vec3) {
        for i in 0..3 {
            self.lo[i] = self.lo[i].min(p[i]);
            self.hi[i] = self.hi[i].max(p[i]);
        }
    }

    pub fn center(&self) -> Vec3 {
        [
            0.5 * (self.lo[0] + self.hi[0]),
            0.5 * (self.lo[1] + self.hi[1]),
            0.5 * (self.lo[2] + self.hi[2]),
        ]
    }

    pub fn surface_area(&self) -> f64 {
        let d = [
            self.hi[0] - self.lo[0],
            self.hi[1] - self.lo[1],
            self.hi[2] - self.lo[2],
        ];
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    pub fn volume(&self) -> f64 {
        (0..3).map(|i| self.hi[i] - self.lo[i]).product()
    }

    /// The box of this box's image under `t` (through its 8 corners), widened
    /// for rounding. Infinite boxes stay infinite unless `t` keeps the axes.
    pub fn transformed(&self, t: &Affine) -> Aabb {
        if self.is_empty() {
            return *self;
        }
        if !self.is_finite() {
            return self.transformed_infinite(t);
        }
        let mut out = Aabb::EMPTY;
        for k in 0..8 {
            let c = [
                if k & 1 == 0 { self.lo[0] } else { self.hi[0] },
                if k & 2 == 0 { self.lo[1] } else { self.hi[1] },
                if k & 4 == 0 { self.lo[2] } else { self.hi[2] },
            ];
            out.grow_point(t.apply(c));
        }
        out.widen()
    }

    /// Interval-arithmetic image: exact for infinite extents along axes the
    /// rotation does not mix.
    fn transformed_infinite(&self, t: &Affine) -> Aabb {
        let mut out = Aabb::INFINITE;
        for i in 0..3 {
            let mut iv = Iv::point(t.disp[i]);
            for j in 0..3 {
                iv = iv.add(
                    Iv {
                        lo: self.lo[j],
                        hi: self.hi[j],
                    }
                    .scale(t.rot[i][j]),
                );
            }
            out.lo[i] = iv.lo;
            out.hi[i] = iv.hi;
        }
        out.widen()
    }

    /// Precomputed reciprocals of a ray direction for [`Aabb::ray_range_inv`].
    #[inline]
    pub fn ray_inv(d: Vec3) -> Vec3 {
        [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]]
    }

    /// Range of `t` in `[lo, hi]` over which `o + t·d` is in the box, using
    /// precomputed `inv = Self::ray_inv(d)`.
    #[inline]
    pub fn ray_range_inv(
        &self,
        o: Vec3,
        d: Vec3,
        inv: Vec3,
        lo: f64,
        hi: f64,
    ) -> Option<(f64, f64)> {
        let (mut a, mut b) = (lo, hi);
        for i in 0..3 {
            if d[i] == 0.0 {
                if !(o[i] >= self.lo[i] && o[i] <= self.hi[i]) {
                    return None;
                }
                continue;
            }
            let (t1, t2) = ((self.lo[i] - o[i]) * inv[i], (self.hi[i] - o[i]) * inv[i]);
            let (t1, t2) = if inv[i] < 0.0 { (t2, t1) } else { (t1, t2) };
            a = a.max(t1);
            b = b.min(t2);
        }
        (a <= b).then_some((a, b))
    }

    /// Range of `t` in `[lo, hi]` over which `o + t·d` is in the box, if any.
    #[inline]
    pub fn ray_range(&self, o: Vec3, d: Vec3, lo: f64, hi: f64) -> Option<(f64, f64)> {
        self.ray_range_inv(o, d, Self::ray_inv(d), lo, hi)
    }

    pub fn widen(&self) -> Aabb {
        let mut r = *self;
        for i in 0..3 {
            r.lo[i] = down(r.lo[i]);
            r.hi[i] = up(r.hi[i]);
        }
        r
    }
}

const ABS_PAD: f64 = 1e-7;
const REL_PAD: f64 = 1e-12;

#[inline]
fn down(x: f64) -> f64 {
    x - ABS_PAD - REL_PAD * x.abs()
}

#[inline]
fn up(x: f64) -> f64 {
    x + ABS_PAD + REL_PAD * x.abs()
}

/// A closed interval; endpoints may be infinite.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Iv {
    lo: f64,
    hi: f64,
}

impl Iv {
    fn point(x: f64) -> Iv {
        Iv { lo: x, hi: x }
    }
    fn add(self, o: Iv) -> Iv {
        Iv {
            lo: self.lo + o.lo,
            hi: self.hi + o.hi,
        }
    }
    fn scale(self, k: f64) -> Iv {
        if k == 0.0 {
            Iv::point(0.0)
        } else if k > 0.0 {
            Iv {
                lo: self.lo * k,
                hi: self.hi * k,
            }
        } else {
            Iv {
                lo: self.hi * k,
                hi: self.lo * k,
            }
        }
    }
    fn mul(self, o: Iv) -> Iv {
        let ps = [
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        ];
        // 0·∞ is NaN in IEEE; as a bound it is 0.
        let ps = ps.map(|v| if v.is_nan() { 0.0 } else { v });
        Iv {
            lo: ps.iter().copied().fold(f64::INFINITY, f64::min),
            hi: ps.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    fn sq(self) -> Iv {
        let (a, b) = (self.lo * self.lo, self.hi * self.hi);
        if self.lo <= 0.0 && self.hi >= 0.0 {
            Iv {
                lo: 0.0,
                hi: a.max(b),
            }
        } else {
            Iv {
                lo: a.min(b),
                hi: a.max(b),
            }
        }
    }
}

/// Shrink `b` to the part where `sign · q(x) ≤ 0` can hold.
fn contract_quadric(q: &Quadric, sign: f64, b: Aabb) -> Aabb {
    let m = q.matrix();
    // Work relative to the quadric's origin.
    let mut y = [Iv::point(0.0); 3];
    for i in 0..3 {
        y[i] = Iv {
            lo: b.lo[i] - q.origin[i],
            hi: b.hi[i] - q.origin[i],
        };
    }
    let mut out = b;
    for i in 0..3 {
        let a = sign * m[i][i];
        let mut beta = Iv::point(sign * q.b[i]);
        let mut gamma = Iv::point(sign * q.c);
        for j in 0..3 {
            if j == i {
                continue;
            }
            beta = beta.add(y[j].scale(2.0 * sign * m[i][j]));
            gamma = gamma.add(y[j].sq().scale(sign * m[j][j]));
            gamma = gamma.add(y[j].scale(sign * q.b[j]));
        }
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        gamma = gamma.add(y[j].mul(y[k]).scale(2.0 * sign * m[j][k]));
        if gamma.lo.is_nan() || gamma.lo == f64::NEG_INFINITY {
            continue;
        }
        let range = y[i];
        let mut hull = Iv {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        };
        // y ≥ 0: a y² + β y + γ ≥ a y² + β.lo y + γ.lo; y ≤ 0 uses β.hi.
        let pos = Iv {
            lo: range.lo.max(0.0),
            hi: range.hi,
        };
        if pos.lo <= pos.hi {
            let part = if beta.lo == f64::NEG_INFINITY {
                Some(pos)
            } else {
                solve_le(a, beta.lo, gamma.lo, pos)
            };
            if let Some(p) = part {
                hull = Iv {
                    lo: hull.lo.min(p.lo),
                    hi: hull.hi.max(p.hi),
                };
            }
        }
        let neg = Iv {
            lo: range.lo,
            hi: range.hi.min(0.0),
        };
        if neg.lo <= neg.hi {
            let part = if beta.hi == f64::INFINITY {
                Some(neg)
            } else {
                solve_le(a, beta.hi, gamma.lo, neg)
            };
            if let Some(p) = part {
                hull = Iv {
                    lo: hull.lo.min(p.lo),
                    hi: hull.hi.max(p.hi),
                };
            }
        }
        if hull.lo > hull.hi {
            return Aabb::EMPTY;
        }
        out.lo[i] = out.lo[i].max(down(hull.lo + q.origin[i]));
        out.hi[i] = out.hi[i].min(up(hull.hi + q.origin[i]));
        y[i] = Iv {
            lo: out.lo[i] - q.origin[i],
            hi: out.hi[i] - q.origin[i],
        };
    }
    out
}

/// Hull of `{y ∈ r : a y² + b y + c ≤ 0}`, or `None` if empty.
fn solve_le(a: f64, b: f64, c: f64, r: Iv) -> Option<Iv> {
    let clip = |lo: f64, hi: f64| {
        let iv = Iv {
            lo: lo.max(r.lo),
            hi: hi.min(r.hi),
        };
        (iv.lo <= iv.hi).then_some(iv)
    };
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 || !scale.is_finite() {
        return Some(r);
    }
    if a.abs() <= 1e-14 * scale {
        // Linear: b y + c ≤ 0.
        if b.abs() <= 1e-14 * scale {
            return if c <= 0.0 { Some(r) } else { None };
        }
        let y0 = -c / b;
        return if b > 0.0 {
            clip(f64::NEG_INFINITY, y0)
        } else {
            clip(y0, f64::INFINITY)
        };
    }
    let disc = b * b - 4.0 * a * c;
    if a > 0.0 {
        if disc < 0.0 {
            return None;
        }
        let s = disc.sqrt();
        let qq = -0.5 * (b + b.signum() * s);
        let (r1, r2) = if qq == 0.0 {
            (0.0, 0.0)
        } else {
            let (x1, x2) = (qq / a, c / qq);
            (x1.min(x2), x1.max(x2))
        };
        clip(r1, r2)
    } else {
        if disc <= 0.0 {
            return Some(r);
        }
        // Feasible outside the roots: (-∞, r1] ∪ [r2, ∞).
        let s = disc.sqrt();
        let qq = -0.5 * (b + b.signum() * s);
        let (x1, x2) = (qq / a, c / qq);
        let (r1, r2) = (x1.min(x2), x1.max(x2));
        let left = clip(f64::NEG_INFINITY, r1);
        let right = clip(r2, f64::INFINITY);
        match (left, right) {
            (Some(l), Some(rr)) => Some(Iv {
                lo: l.lo,
                hi: rr.hi,
            }),
            (Some(l), None) => Some(l),
            (None, Some(rr)) => Some(rr),
            (None, None) => None,
        }
    }
}

/// Eigen-structure of a quadric, for bounding it in any orientation:
/// `q = Σ_stiff λ_k z_k² + Σ_soft (λ_k y_k² + β_k y_k) + k` with
/// `z_k = y_k = u_k·(p - center)`.
///
/// Directions with a relatively tiny eigenvalue are "soft": GEOUNED writes
/// cylinders as GQ with ~7 digits, so their axis eigenvalue is ±1e-9 rather
/// than 0. Completing the square there would shift the centre by ~1e9 and
/// cancel catastrophically, so soft directions keep their raw linear form and
/// are bounded by the range of `y_k` inside the current box.
#[derive(Clone, Copy, Debug)]
pub struct QuadInfo {
    pub center: Vec3,
    pub u: [Vec3; 3],
    pub lam: [f64; 3],
    /// Linear coefficients along soft directions (0 for stiff ones).
    pub beta: [f64; 3],
    pub stiff: [bool; 3],
    pub k: f64,
}

/// Eigenvalues below this fraction of the largest are soft.
const SOFT: f64 = 1e-6;

/// Analyse a quadric; `None` if it is degenerate (all eigenvalues ~0).
pub fn analyse(q: &Quadric) -> Option<QuadInfo> {
    let (lam, u) = eigen_sym(q.matrix());
    let scale = lam.iter().fold(0.0f64, |m, l| m.max(l.abs()));
    if scale == 0.0 {
        return None;
    }
    let mut center = q.origin;
    let mut k = q.c;
    let mut stiff = [false; 3];
    let mut beta = [0.0; 3];
    for i in 0..3 {
        let bi = dot(u[i], q.b);
        if lam[i].abs() > SOFT * scale {
            stiff[i] = true;
            let s = -bi / (2.0 * lam[i]);
            for c in 0..3 {
                center[c] += s * u[i][c];
            }
            k -= bi * bi / (4.0 * lam[i]);
        } else {
            beta[i] = bi;
        }
    }
    Some(QuadInfo {
        center,
        u,
        lam,
        beta,
        stiff,
        k,
    })
}

/// Eigenvalues and orthonormal eigenvectors (rows) of a symmetric 3×3 matrix
/// by cyclic Jacobi rotations.
fn eigen_sym(m: [Vec3; 3]) -> ([f64; 3], [Vec3; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let diag = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-15 * diag || off == 0.0 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A ← Jᵀ A J
            for r in 0..3 {
                let (arp, arq) = (a[r][p], a[r][q]);
                a[r][p] = c * arp - s * arq;
                a[r][q] = s * arp + c * arq;
            }
            for r in 0..3 {
                let (apr, aqr) = (a[p][r], a[q][r]);
                a[p][r] = c * apr - s * aqr;
                a[q][r] = s * apr + c * aqr;
            }
            for row in v.iter_mut() {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let lam = [a[0][0], a[1][1], a[2][2]];
    // Columns of v are eigenvectors; return them as rows.
    let u = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    (lam, u)
}

/// One side of an analysed quadric as `Σ_kept μ_k z_k² ≤ s`, where the other
/// directions have been bounded over the box and folded into `s`, and `free`
/// lists the directions whose coordinate is only known to lie in `ranges`.
struct Region {
    mu: [f64; 3],
    s: f64,
    kept: Dirs,
    free: Dirs,
    ranges: [Iv; 3],
}

/// Up to three eigen directions, without allocating (regions are built twice
/// per quadric leaf and contraction, millions of times while bounding).
#[derive(Default)]
struct Dirs {
    n: usize,
    k: [usize; 3],
}

impl Dirs {
    fn push(&mut self, k: usize) {
        self.k[self.n] = k;
        self.n += 1;
    }
}

impl std::ops::Deref for Dirs {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.k[..self.n]
    }
}

impl std::ops::DerefMut for Dirs {
    fn deref_mut(&mut self) -> &mut [usize] {
        &mut self.k[..self.n]
    }
}

/// Coordinate ranges `u_k·(p - center)` over the box.
fn ranges(info: &QuadInfo, b: &Aabb) -> [Iv; 3] {
    let mut out = [Iv::point(0.0); 3];
    for (k, r) in out.iter_mut().enumerate() {
        for i in 0..3 {
            let side = Iv {
                lo: b.lo[i] - info.center[i],
                hi: b.hi[i] - info.center[i],
            };
            *r = r.add(side.scale(info.u[k][i]));
        }
    }
    out
}

/// Minimum of `a y² + c y` over `r` (−∞ if unbounded below).
fn min_quad(a: f64, c: f64, r: Iv) -> f64 {
    let f = |y: f64| {
        if y.is_infinite() {
            if a > 0.0 {
                f64::INFINITY
            } else if a < 0.0 {
                f64::NEG_INFINITY
            } else if c == 0.0 {
                0.0
            } else if (c > 0.0) == (y > 0.0) {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            a * y * y + c * y
        }
    };
    let mut m = f(r.lo).min(f(r.hi));
    if a > 0.0 {
        let v = -c / (2.0 * a);
        if v >= r.lo && v <= r.hi {
            m = m.min(f(v));
        }
    }
    m
}

/// Build the region for one side, keeping the (at most `max_kept`) stiffest
/// positive directions and folding the others in using the box.
fn region(info: &QuadInfo, want_pos: bool, b: &Aabb, max_kept: usize) -> Option<Region> {
    let sign = if want_pos { -1.0 } else { 1.0 };
    let mu = info.lam.map(|l| sign * l);
    let nu = info.beta.map(|l| sign * l);
    let rg = ranges(info, b);
    let mut s = -sign * info.k;
    let mut pos = Dirs::default();
    for k in (0..3).filter(|&k| info.stiff[k] && mu[k] > 0.0) {
        pos.push(k);
    }
    pos.sort_by(|&a, &c| mu[c].total_cmp(&mu[a]));
    if pos.is_empty() {
        return None;
    }
    let mut kept = Dirs::default();
    for &k in pos.iter().take(max_kept) {
        kept.push(k);
    }
    let mut free = Dirs::default();
    for k in 0..3 {
        if kept.contains(&k) {
            continue;
        }
        free.push(k);
        // Move this direction's term to the right-hand side, at its least
        // favourable value over the box.
        let m = if info.stiff[k] {
            min_quad(mu[k], 0.0, rg[k])
        } else {
            min_quad(mu[k], nu[k], rg[k])
        };
        s -= m;
    }
    if s.is_nan() || s == f64::INFINITY {
        return None;
    }
    Some(Region {
        mu,
        s,
        kept,
        free,
        ranges: rg,
    })
}

/// Shrink `b` with the eigen form of a quadric side.
fn contract_eigen(info: &QuadInfo, want_pos: bool, b: Aabb) -> Aabb {
    let mut out = b;
    for max_kept in [3, 2] {
        let Some(r) = region(info, want_pos, &out, max_kept) else {
            return out;
        };
        if r.s < 0.0 {
            if r.s < -1e-9 * (1.0 + info.k.abs()) {
                return Aabb::EMPTY;
            }
            continue;
        }
        for i in 0..3 {
            let mut iv = Iv::point(info.center[i]);
            for &k in r.free.iter() {
                iv = iv.add(r.ranges[k].scale(info.u[k][i]));
            }
            if !(iv.lo.is_finite() && iv.hi.is_finite()) {
                continue;
            }
            let rad2: f64 = r
                .kept
                .iter()
                .map(|&k| info.u[k][i] * info.u[k][i] / r.mu[k])
                .sum();
            let rad = (r.s * rad2).sqrt();
            out.lo[i] = out.lo[i].max(down(iv.lo - rad));
            out.hi[i] = out.hi[i].min(up(iv.hi + rad));
        }
        if out.is_empty() {
            return out;
        }
    }
    out
}

/// Bounds cell programs, given the model's shapes.
pub struct Bounder<'a> {
    pub shapes: &'a [Shape],
    pub info: Vec<Option<QuadInfo>>,
}

impl<'a> Bounder<'a> {
    pub fn new(shapes: &'a [Shape]) -> Self {
        use rayon::prelude::*;
        let info = shapes
            .par_iter()
            .map(|s| match s {
                Shape::Quadric(q) => analyse(q),
                Shape::Cone(c) => analyse(&c.q),
                _ => None,
            })
            .collect();
        Bounder { shapes, info }
    }

    /// Shrink `b` to the part where a leaf can be true.
    pub fn contract_leaf(&self, s: u32, want_pos: bool, b: Aabb) -> Aabb {
        let shape = &self.shapes[s as usize];
        let mut b = contract_leaf(shape, want_pos, b);
        if b.is_empty() {
            return b;
        }
        if let Some(info) = &self.info[s as usize] {
            let cone_pos = matches!(shape, Shape::Cone(_)) && want_pos;
            if !cone_pos {
                b = contract_eigen(info, want_pos, b);
            }
        }
        b
    }

    /// Linear outer approximation of a leaf, appended to `out`.
    fn linear(&self, s: u32, want_pos: bool, b: &Aabb, out: &mut Vec<Constraint>) {
        let shape = &self.shapes[s as usize];
        match shape {
            Shape::Plane { n, d } => {
                if want_pos {
                    out.push(([-n[0], -n[1], -n[2]], -d));
                } else {
                    out.push((*n, *d));
                }
            }
            Shape::Quadric(_) => {
                let Some(info) = &self.info[s as usize] else {
                    return;
                };
                // Cross-section of the two stiffest directions, with the third
                // folded in over the box, circumscribed by an octagon.
                let Some(r) = region(info, want_pos, b, 2) else {
                    return;
                };
                if r.kept.len() != 2 || r.s < 0.0 {
                    return;
                }
                let (e1, e2) = (info.u[r.kept[0]], info.u[r.kept[1]]);
                let (m1, m2) = (r.mu[r.kept[0]], r.mu[r.kept[1]]);
                for j in 0..8 {
                    let phi = j as f64 * std::f64::consts::FRAC_PI_4;
                    let (c, sn) = (phi.cos(), phi.sin());
                    let w = [
                        c * e1[0] + sn * e2[0],
                        c * e1[1] + sn * e2[1],
                        c * e1[2] + sn * e2[2],
                    ];
                    let h = (r.s * (c * c / m1 + sn * sn / m2)).sqrt();
                    out.push((w, up(dot(w, info.center) + h)));
                }
            }
            Shape::Cone(cone) if !want_pos => {
                let Some(info) = &self.info[s as usize] else {
                    return;
                };
                if !info.stiff.iter().all(|&x| x) {
                    return;
                }
                let mu = info.lam;
                let mut neg = Dirs::default();
                for k in (0..3).filter(|&k| mu[k] < 0.0) {
                    neg.push(k);
                }
                let [a] = &*neg else { return };
                let r = Region {
                    mu,
                    s: -info.k,
                    kept: Dirs::default(),
                    free: Dirs::default(),
                    ranges: [Iv::point(0.0); 3],
                };
                let mut others = Dirs::default();
                for k in (0..3).filter(|k| k != a) {
                    others.push(k);
                }
                let ua = info.u[*a];
                let sigma = dot(ua, cone.axis).signum();
                let apex = cone.apex;
                // Nappe half-space and a circumscribed octagonal pyramid.
                let ax = cone.axis;
                out.push(([-ax[0], -ax[1], -ax[2]], -dot(ax, apex) + 1e-9));
                for j in 0..8 {
                    let phi = j as f64 * std::f64::consts::FRAC_PI_4;
                    let (c, sn) = (phi.cos(), phi.sin());
                    let (e1, e2) = (info.u[others[0]], info.u[others[1]]);
                    let w = [
                        c * e1[0] + sn * e2[0],
                        c * e1[1] + sn * e2[1],
                        c * e1[2] + sn * e2[2],
                    ];
                    let g = (r.mu[*a].abs()
                        * (c * c / r.mu[others[0]] + sn * sn / r.mu[others[1]]))
                        .sqrt()
                        * (1.0 + 1e-9);
                    let nrm = [
                        w[0] - g * sigma * ua[0],
                        w[1] - g * sigma * ua[1],
                        w[2] - g * sigma * ua[2],
                    ];
                    out.push((nrm, dot(nrm, apex) + 1e-9));
                }
            }
            _ => {}
        }
    }

    /// Bound a cell program inside `outer`.
    pub fn bound(&self, code: &[u32], outer: Aabb) -> Aabb {
        if code.is_empty() {
            return Aabb::EMPTY;
        }
        self.bound_at(code, 0, outer, false, &mut Vec::new()).0
    }

    /// Boxes for the large nodes of a program, for pruning during
    /// evaluation: `(position, may-be-true box, may-be-false box)`.
    ///
    /// Both are computed with the same outer box as in [`Bounder::bound`],
    /// so pruning is exact for points inside `outer` (see `expr::eval`).
    pub fn node_boxes(&self, code: &[u32], outer: Aabb, min_len: u32) -> Vec<(usize, Aabb, Aabb)> {
        let mut out = Vec::new();
        if !code.is_empty() {
            self.annotate(code, 0, outer, min_len, &mut out, &mut Vec::new());
        }
        out
    }

    fn annotate(
        &self,
        code: &[u32],
        pos: usize,
        outer: Aabb,
        min_len: u32,
        out: &mut Vec<(usize, Aabb, Aabb)>,
        lin: &mut Vec<Constraint>,
    ) {
        let w = code[pos];
        if w & NODE == 0 || w & MASK < min_len {
            return;
        }
        let (t, end) = self.bound_at(code, pos, outer, false, lin);
        let (f, _) = self.bound_at(code, pos, outer, true, lin);
        out.push((pos, t, f));
        // Children are bounded inside the node's box (AND) or its outer (OR).
        let child_outer = if w & OR != 0 { outer } else { t };
        let mut p = pos + 2;
        while p < end {
            let cw = code[p];
            self.annotate(code, p, child_outer, min_len, out, lin);
            p += if cw & NODE == 0 {
                1
            } else {
                1 + (cw & MASK) as usize
            };
        }
    }

    /// Returns the box and the position after the node. With `neg`, bounds
    /// the complement of the node instead. `lin` is scratch space, shared by
    /// the whole recursion so that bounding does not allocate per node.
    fn bound_at(
        &self,
        code: &[u32],
        pos: usize,
        outer: Aabb,
        neg: bool,
        lin: &mut Vec<Constraint>,
    ) -> (Aabb, usize) {
        let w = code[pos];
        if w & NODE == 0 {
            if outer.is_empty() {
                return (outer, pos + 1);
            }
            return (
                self.contract_leaf(w & MASK, (w & POS != 0) != neg, outer),
                pos + 1,
            );
        }
        // Skip the header and its box word.
        let start = pos + 2;
        let end = pos + 1 + (w & MASK) as usize;
        if (w & OR != 0) != neg {
            let mut hull = Aabb::EMPTY;
            let mut p = start;
            while p < end {
                let (b, next) = self.bound_at(code, p, outer, neg, lin);
                hull = hull.union(&b);
                p = next;
            }
            return (hull.intersect(&outer), end);
        }
        let mut b = outer;
        for _round in 0..4 {
            let before = b;
            let mut p = start;
            while p < end {
                let (nb, next) = self.bound_at(code, p, b, neg, lin);
                b = nb;
                if b.is_empty() {
                    return (b, end);
                }
                p = next;
            }
            // Linear outer approximations of the direct leaves, for the LP
            // (the children are done with `lin`).
            lin.clear();
            let mut general = false;
            let mut p = start;
            while p < end {
                let w = code[p];
                if w & NODE == 0 {
                    let first = lin.len();
                    self.linear(w & MASK, (w & POS != 0) != neg, &b, lin);
                    general |= lin[first..]
                        .iter()
                        .any(|(n, _)| n.iter().filter(|x| **x != 0.0).count() > 1);
                    p += 1;
                } else {
                    p += 1 + (w & MASK) as usize;
                }
            }
            if general {
                b = lp_box(lin, b);
                if b.is_empty() {
                    return (b, end);
                }
            }
            if !shrunk_enough(&before, &b) {
                break;
            }
        }
        (b, end)
    }
}

/// Tighten `b` by minimising and maximising each coordinate over `lin` ∩ `b`
/// (adds the box's faces to `lin` and reorders it).
fn lp_box(lin: &mut Vec<Constraint>, b: Aabb) -> Aabb {
    let cons = lin;
    for i in 0..3 {
        let mut e = [0.0; 3];
        e[i] = 1.0;
        if b.hi[i].is_finite() {
            cons.push((e, b.hi[i]));
        }
        if b.lo[i].is_finite() {
            cons.push(([-e[0], -e[1], -e[2]], -b.lo[i]));
        }
    }
    lp::shuffle(cons);
    let mut out = b;
    for i in 0..3 {
        let mut c = [0.0; 3];
        c[i] = 1.0;
        let Some(lo) = lp::minimize(cons, c) else {
            // Infeasible within rounding: keep the box rather than trust it.
            return b;
        };
        c[i] = -1.0;
        let Some(hi) = lp::minimize(cons, c) else {
            return b;
        };
        if lo[i] > -0.99 * lp::BIG {
            out.lo[i] = out.lo[i].max(down(lo[i]) - 1e-6);
        }
        if hi[i] < 0.99 * lp::BIG {
            out.hi[i] = out.hi[i].min(up(hi[i]) + 1e-6);
        }
    }
    out
}

/// Shrink `b` to the part of it where a leaf can be true (per-axis interval
/// contraction only).
pub fn contract_leaf(shape: &Shape, want_pos: bool, b: Aabb) -> Aabb {
    let sign = if want_pos { -1.0 } else { 1.0 };
    match shape {
        Shape::Plane { n, d } => {
            let q = Quadric {
                origin: [0.0; 3],
                a: [0.0; 6],
                b: *n,
                c: -d,
            };
            contract_quadric(&q, sign, b)
        }
        Shape::Quadric(q) => contract_quadric(q, sign, b),
        Shape::Cone(c) => {
            if want_pos {
                return b;
            }
            let b = contract_quadric(&c.q, 1.0, b);
            let half = Quadric {
                origin: c.apex,
                a: [0.0; 6],
                b: c.axis,
                c: 0.0,
            };
            // On the nappe side: axis·(p - apex) ≥ 0.
            contract_quadric(&half, -1.0, b)
        }
        Shape::Torus(t) => {
            if want_pos {
                return b;
            }
            let r = t.a + t.c;
            let mut local = Aabb {
                lo: [-r; 3],
                hi: [r; 3],
            };
            local.lo[t.axis as usize] = -t.b;
            local.hi[t.axis as usize] = t.b;
            b.intersect(&local.transformed(&t.frame.inverse()))
        }
    }
}

/// Whether another round is worth it: some side moved by more than 1% of
/// the extent (or became finite).
fn shrunk_enough(before: &Aabb, after: &Aabb) -> bool {
    (0..3).any(|i| {
        let ext = after.hi[i] - after.lo[i];
        let dl = after.lo[i] - before.lo[i];
        let dh = before.hi[i] - after.hi[i];
        if !ext.is_finite() {
            return false;
        }
        !dl.is_finite() || !dh.is_finite() || dl > 0.01 * ext || dh > 0.01 * ext
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::build;

    fn leaf_box(kind: &str, v: &[f64], want_pos: bool) -> Aabb {
        contract_leaf(&build(kind, v).unwrap()[0], want_pos, Aabb::INFINITE)
    }

    #[test]
    fn sphere_and_cylinder_boxes() {
        let b = leaf_box("S", &[1.0, 2.0, 3.0, 2.0], false);
        assert!((b.lo[0] + 1.0).abs() < 1e-6 && (b.hi[2] - 5.0).abs() < 1e-6);
        let b = leaf_box("C/Z", &[1.0, 2.0, 0.5], false);
        assert!((b.lo[0] - 0.5).abs() < 1e-6 && (b.hi[1] - 2.5).abs() < 1e-6);
        assert!(b.lo[2] == f64::NEG_INFINITY);
        let b = leaf_box("SO", &[1.0], true);
        assert_eq!(b, Aabb::INFINITE);
        let b = leaf_box("TZ", &[0.0, 0.0, 1.0, 5.0, 1.0, 2.0], false);
        assert!((b.hi[0] - 7.0).abs() < 1e-6 && (b.lo[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn planes_close_a_tetrahedron() {
        // x > 0, y > 0, z > 0, x + y + z < 1
        let shapes = [
            build("PX", &[0.0]).unwrap().remove(0),
            build("PY", &[0.0]).unwrap().remove(0),
            build("PZ", &[0.0]).unwrap().remove(0),
            build("P", &[1.0, 1.0, 1.0, 1.0]).unwrap().remove(0),
        ];
        let code = [NODE | 5, NO_BOX, POS, 1 | POS, 2 | POS, 3];
        let b = Bounder::new(&shapes).bound(&code, Aabb::INFINITE);
        for i in 0..3 {
            assert!(b.lo[i] > -1e-6 && b.lo[i] < 1e-6);
            assert!((b.hi[i] - 1.0).abs() < 1e-6, "{b:?}");
        }
    }

    #[test]
    fn rotated_cylinder_in_a_box() {
        // Cylinder of radius 1 along (1,1,0)/√2: z is bounded by the radius.
        let shapes = [
            build("GQ", &[0.5, 0.5, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                .unwrap()
                .remove(0),
        ];
        let outer = Aabb {
            lo: [-10.0; 3],
            hi: [10.0; 3],
        };
        let b = Bounder::new(&shapes).bound(&[0], outer);
        assert!(b.hi[2] < 1.0 + 1e-6 && b.lo[2] > -1.0 - 1e-6, "{b:?}");
    }

    #[test]
    fn rotated_cylinder_between_planes() {
        // Radius 1 along (1,1,0)/√2, cut at ±5 along the axis: the LP with the
        // circumscribed prism bounds x and y from an infinite start.
        let s2 = 0.5f64.sqrt();
        let shapes = [
            build("GQ", &[0.5, 0.5, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                .unwrap()
                .remove(0),
            build("P", &[s2, s2, 0.0, 5.0]).unwrap().remove(0),
            build("P", &[s2, s2, 0.0, -5.0]).unwrap().remove(0),
        ];
        let code = [NODE | 4, NO_BOX, 0, 1, 2 | POS];
        let b = Bounder::new(&shapes).bound(&code, Aabb::INFINITE);
        assert!(b.is_finite(), "{b:?}");
        assert!(
            b.hi[0] < 5.0 * s2 + s2 + 0.2 && b.hi[2] < 1.0 + 1e-6,
            "{b:?}"
        );
    }

    #[test]
    fn rotated_cube_from_general_planes() {
        let s2 = 0.5f64.sqrt();
        let p = |a: f64, b: f64, c: f64, d: f64| build("P", &[a, b, c, d]).unwrap().remove(0);
        let shapes = [
            p(s2, s2, 0.0, 1.0),
            p(s2, s2, 0.0, -1.0),
            p(s2, -s2, 0.0, 1.0),
            p(s2, -s2, 0.0, -1.0),
            p(0.0, 0.0, 1.0, 1.0),
            p(0.0, 0.0, 1.0, -1.0),
        ];
        let code = [NODE | 7, NO_BOX, 0, 1 | POS, 2, 3 | POS, 4, 5 | POS];
        let b = Bounder::new(&shapes).bound(&code, Aabb::INFINITE);
        assert!(
            (b.hi[0] - 2f64.sqrt()).abs() < 1e-5 && (b.lo[1] + 2f64.sqrt()).abs() < 1e-5,
            "{b:?}"
        );
    }

    #[test]
    fn eigen_bounds_rotated_ellipsoid_and_cone() {
        let t = crate::transform::from_mcnp(
            &[
                1.0, 2.0, 3.0, 30.0, 60.0, 90.0, 120.0, 30.0, 90.0, 90.0, 90.0, 0.0,
            ],
            true,
        )
        .unwrap();
        // Sphere of radius 2 written as a rotated GQ: its box is exact.
        let s = build("SO", &[2.0]).unwrap().remove(0).to_parent(&t);
        let b = Bounder::new(std::slice::from_ref(&s)).bound(&[0], Aabb::INFINITE);
        let c = t.inverse().apply([0.0; 3]);
        for i in 0..3 {
            assert!((b.hi[i] - c[i] - 2.0).abs() < 1e-5, "{b:?}");
        }
        // One-sheet cone, cut by a plane: finite.
        let shapes = [
            build("KZ", &[0.0, 0.25, 1.0])
                .unwrap()
                .remove(0)
                .to_parent(&t),
            build("PZ", &[4.0]).unwrap().remove(0).to_parent(&t),
        ];
        let b = Bounder::new(&shapes).bound(&[NODE | 3, NO_BOX, 0, 1], Aabb::INFINITE);
        assert!(b.is_finite(), "{b:?}");
    }
}
