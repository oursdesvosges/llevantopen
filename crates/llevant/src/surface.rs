//! Compiled surfaces: MCNP mnemonics and macrobodies → planes, quadrics,
//! one-sheet cones and tori, all in the global frame of their universe.

use crate::bbox::Aabb;
use crate::transform::Affine;
use crate::vec3::{cross, dot, norm, normalize, scale, sub, Vec3};

/// Which side of a surface a point is on, with an undecided band of width
/// `tol` around the surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Neg,
    On,
    Pos,
}

/// `f(p) = (p-o)ᵀ A (p-o) + b·(p-o) + c` with symmetric `A` stored as
/// `[xx, yy, zz, xy, yz, zx]`. Keeping a local origin `o` (the centre or apex
/// of the surface as written) avoids cancellation far from the global origin.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Quadric {
    pub origin: Vec3,
    pub a: [f64; 6],
    pub b: Vec3,
    pub c: f64,
}

impl Quadric {
    #[inline]
    pub fn matrix(&self) -> [Vec3; 3] {
        let a = &self.a;
        [[a[0], a[3], a[5]], [a[3], a[1], a[4]], [a[5], a[4], a[2]]]
    }

    /// Value and gradient length at `p`.
    #[inline]
    pub fn eval(&self, p: Vec3) -> (f64, f64) {
        let d = sub(p, self.origin);
        let a = &self.a;
        let ad = [
            a[0] * d[0] + a[3] * d[1] + a[5] * d[2],
            a[3] * d[0] + a[1] * d[1] + a[4] * d[2],
            a[5] * d[0] + a[4] * d[1] + a[2] * d[2],
        ];
        let f = dot(d, ad) + dot(self.b, d) + self.c;
        let g = [
            2.0 * ad[0] + self.b[0],
            2.0 * ad[1] + self.b[1],
            2.0 * ad[2] + self.b[2],
        ];
        (f, norm(g))
    }

    /// The same quadric written in the parent frame, given `t: parent → local`.
    pub fn to_parent(self, t: &Affine) -> Quadric {
        let r = t.rot;
        let m = self.matrix();
        // A' = Rᵀ A R
        let mut ar = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                ar[i][j] = (0..3).map(|k| m[i][k] * r[k][j]).sum();
            }
        }
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = (0..3).map(|k| r[k][i] * ar[k][j]).sum();
            }
        }
        Quadric {
            origin: t.inverse().apply(self.origin),
            a: [
                out[0][0],
                out[1][1],
                out[2][2],
                0.5 * (out[0][1] + out[1][0]),
                0.5 * (out[1][2] + out[2][1]),
                0.5 * (out[2][0] + out[0][2]),
            ],
            b: t.apply_dir_inv(self.b),
            c: self.c,
        }
    }

    fn sphere(center: Vec3, r: f64) -> Quadric {
        Quadric {
            origin: center,
            a: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            b: [0.0; 3],
            c: -r * r,
        }
    }

    /// Cylinder (`t2 = 0`) or double cone about an axis parallel to x, y or z.
    fn axial(axis: usize, origin: Vec3, r2: f64, t2: f64) -> Quadric {
        let mut a = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        a[axis] = -t2;
        Quadric {
            origin,
            a,
            b: [0.0; 3],
            c: -r2,
        }
    }

    /// Cylinder or double cone about a general unit axis `u` through `origin`:
    /// `|d|² - (1 + t2)(d·u)² - r2`.
    fn general_axial(origin: Vec3, u: Vec3, r2: f64, t2: f64) -> Quadric {
        let k = 1.0 + t2;
        Quadric {
            origin,
            a: [
                1.0 - k * u[0] * u[0],
                1.0 - k * u[1] * u[1],
                1.0 - k * u[2] * u[2],
                -k * u[0] * u[1],
                -k * u[1] * u[2],
                -k * u[2] * u[0],
            ],
            b: [0.0; 3],
            c: -r2,
        }
    }
}

/// A one-sheet cone: the quadric's negative side, restricted to the nappe on
/// the `axis` side of `apex`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cone {
    pub q: Quadric,
    pub apex: Vec3,
    /// Unit vector pointing into the selected nappe.
    pub axis: Vec3,
}

/// `(w/b)² + ((ρ - a)/c)² - 1` in a local frame where the torus axis is
/// coordinate `axis` and its centre is the origin.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Torus {
    pub frame: Affine,
    pub axis: u8,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Torus {
    #[inline]
    fn eval(&self, p: Vec3) -> (f64, f64) {
        let q = self.frame.apply(p);
        let ax = self.axis as usize;
        let w = q[ax];
        let (u, v) = (q[(ax + 1) % 3], q[(ax + 2) % 3]);
        let rho = (u * u + v * v).sqrt();
        let dw = w / self.b;
        let dr = (rho - self.a) / self.c;
        let f = dw * dw + dr * dr - 1.0;
        let g = ((2.0 * dw / self.b).powi(2) + (2.0 * dr / self.c).powi(2)).sqrt();
        (f, g)
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Shape {
    /// `n·p - d` with unit `n`.
    Plane {
        n: Vec3,
        d: f64,
    },
    Quadric(Quadric),
    Cone(Box<Cone>),
    Torus(Box<Torus>),
}

impl Shape {
    pub(crate) fn plane(n: Vec3, d: f64) -> Shape {
        let l = norm(n);
        Shape::Plane {
            n: scale(n, 1.0 / l),
            d: d / l,
        }
    }

    #[inline]
    pub fn side(&self, p: Vec3, tol: f64) -> Side {
        let (f, g) = match self {
            Shape::Plane { n, d } => (dot(*n, p) - d, 1.0),
            Shape::Quadric(q) => q.eval(p),
            Shape::Torus(t) => t.eval(p),
            Shape::Cone(c) => {
                let (f, g) = c.q.eval(p);
                let s = dot(c.axis, sub(p, c.apex));
                let inside = side_of(f, g, tol);
                // Negative region = inside the double cone AND on the nappe side.
                let nappe = if s > tol {
                    Side::Neg
                } else if s < -tol {
                    Side::Pos
                } else {
                    Side::On
                };
                return match (inside, nappe) {
                    (Side::Pos, _) | (_, Side::Pos) => Side::Pos,
                    (Side::Neg, Side::Neg) => Side::Neg,
                    _ => Side::On,
                };
            }
        };
        side_of(f, g, tol)
    }

    /// The shape as seen from the parent frame, given `t: parent → local`.
    pub fn to_parent(&self, t: &Affine) -> Shape {
        match self {
            Shape::Plane { n, d } => {
                // n·(R p + t) - d
                let nn = t.apply_dir_inv(*n);
                Shape::Plane {
                    n: nn,
                    d: d - dot(*n, t.disp),
                }
            }
            Shape::Quadric(q) => Shape::Quadric(q.to_parent(t)),
            Shape::Cone(c) => Shape::Cone(Box::new(Cone {
                q: c.q.to_parent(t),
                apex: t.inverse().apply(c.apex),
                axis: t.apply_dir_inv(c.axis),
            })),
            Shape::Torus(tor) => Shape::Torus(Box::new(Torus {
                frame: tor.frame.compose(t),
                ..**tor
            })),
        }
    }
}

impl Shape {
    /// The side of the whole box `b`: `Pos` or `Neg` only if [`Shape::side`]
    /// gives that answer at every point of `b`, rounding included; `On` when
    /// undecided. Tori are always undecided.
    pub fn side_over(&self, b: &Aabb, tol: f64) -> Side {
        if !b.is_finite() {
            return Side::On;
        }
        let (c, h) = center_half(b);
        match self {
            Shape::Plane { n, d } => {
                let f = dot(*n, c) - d;
                let r: f64 = (0..3).map(|i| n[i].abs() * h[i]).sum();
                let mag: f64 = (0..3)
                    .map(|i| n[i].abs() * (c[i].abs() + h[i]))
                    .sum::<f64>()
                    + d.abs();
                decide(f - r, f + r, tol, mag)
            }
            Shape::Quadric(q) => q.side_over(c, h, tol),
            Shape::Cone(k) => {
                let inside = k.q.side_over(c, h, tol);
                // Same combination as `side`: s > tol is the nappe (Neg).
                let s = dot(k.axis, sub(c, k.apex));
                let r: f64 = (0..3).map(|i| k.axis[i].abs() * h[i]).sum();
                let mag: f64 = (0..3)
                    .map(|i| k.axis[i].abs() * (c[i].abs() + k.apex[i].abs() + h[i]))
                    .sum();
                let nappe = match decide(s - r, s + r, tol, mag) {
                    Side::Pos => Side::Neg,
                    Side::Neg => Side::Pos,
                    Side::On => Side::On,
                };
                match (inside, nappe) {
                    (Side::Pos, _) | (_, Side::Pos) => Side::Pos,
                    (Side::Neg, Side::Neg) => Side::Neg,
                    _ => Side::On,
                }
            }
            Shape::Torus(_) => Side::On,
        }
    }
}

impl Shape {
    /// Gradient of the surface function at `p` (not normalised; may be zero
    /// at singular points such as a cone's apex). For a one-sheet cone, that
    /// of its quadric.
    pub fn gradient(&self, p: Vec3) -> Vec3 {
        let quadric = |q: &Quadric| {
            let d = sub(p, q.origin);
            let m = q.matrix();
            [0, 1, 2].map(|i| 2.0 * dot(m[i], d) + q.b[i])
        };
        match self {
            Shape::Plane { n, .. } => *n,
            Shape::Quadric(q) => quadric(q),
            Shape::Cone(k) => quadric(&k.q),
            Shape::Torus(t) => {
                let q = t.frame.apply(p);
                let ax = t.axis as usize;
                let (iu, iv) = ((ax + 1) % 3, (ax + 2) % 3);
                let rho = (q[iu] * q[iu] + q[iv] * q[iv]).sqrt();
                let mut g = [0.0; 3];
                g[ax] = 2.0 * q[ax] / (t.b * t.b);
                if rho > 0.0 {
                    let k = 2.0 * (rho - t.a) / (t.c * t.c * rho);
                    g[iu] = k * q[iu];
                    g[iv] = k * q[iv];
                }
                t.frame.apply_dir_inv(g)
            }
        }
    }

    /// Parameters `t` in `[lo, hi]` where the line `o + t·d` (unit `d`) may
    /// change side, appended to `out` unsorted. Every sign change of the
    /// surface function along the line is reported; a few extra values
    /// (tangent touches, a cone's apex plane away from the cone, the mirror
    /// sheet of a squared torus equation) are harmless, since callers decide
    /// sides between consecutive values.
    pub fn crossings(&self, o: Vec3, d: Vec3, lo: f64, hi: f64, out: &mut Vec<f64>) {
        let mut push = |t: f64| {
            if t >= lo && t <= hi {
                out.push(t);
            }
        };
        match self {
            Shape::Plane { n, d: off } => {
                if let Some(t) = plane_crossing(*n, *off, o, d) {
                    push(t);
                }
            }
            Shape::Quadric(q) => q.crossings(o, d, &mut push),
            Shape::Cone(k) => {
                k.q.crossings(o, d, &mut push);
                // The nappe side flips on the plane through the apex.
                if let Some(t) = plane_crossing(k.axis, dot(k.axis, k.apex), o, d) {
                    push(t);
                }
            }
            Shape::Torus(t) => t.crossings(o, d, lo, hi, &mut push),
        }
    }
}

#[inline]
fn plane_crossing(n: Vec3, off: f64, o: Vec3, d: Vec3) -> Option<f64> {
    let nd = dot(n, d);
    (nd != 0.0).then(|| (off - dot(n, o)) / nd)
}

impl Quadric {
    /// Roots of the quadric along `o + t·d`, written about the point of the
    /// line closest to the quadric's origin so that coefficients stay small.
    fn crossings(&self, o: Vec3, d: Vec3, push: &mut impl FnMut(f64)) {
        let w = sub(o, self.origin);
        let tc = -dot(w, d);
        let e = [0, 1, 2].map(|i| w[i] + tc * d[i]);
        let m = self.matrix();
        let ad = [dot(m[0], d), dot(m[1], d), dot(m[2], d)];
        let a = dot(d, ad);
        let b = 2.0 * dot(e, ad) + dot(self.b, d);
        let ae = [dot(m[0], e), dot(m[1], e), dot(m[2], e)];
        let c = dot(e, ae) + dot(self.b, e) + self.c;
        quadratic_roots(a, b, c, &mut |s| push(tc + s));
    }
}

/// Real roots of `a s² + b s + c` without cancellation; a tiny `a` gives one
/// root near `-c/b` and one far away.
fn quadratic_roots(a: f64, b: f64, c: f64, push: &mut impl FnMut(f64)) {
    if a == 0.0 {
        if b != 0.0 {
            push(-c / b);
        }
        return;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return;
    }
    let q = -0.5 * (b + b.signum() * disc.sqrt());
    if q == 0.0 {
        push(0.0);
        return;
    }
    push(q / a);
    push(c / q);
}

impl Torus {
    /// Roots along `o + t·d` within `[lo, hi]`, from the quartic
    /// `((ρ-a)² - K)((ρ+a)² - K)` with `K = c²(1 - w²/b²)`: its first factor
    /// has the sign of the torus function, the second only adds roots.
    fn crossings(&self, o: Vec3, d: Vec3, lo: f64, hi: f64, push: &mut impl FnMut(f64)) {
        let q0 = self.frame.apply(o);
        let dq = self.frame.apply_dir(d);
        // Work about the point closest to the centre, within the bounding
        // sphere.
        let tc = -dot(q0, dq);
        let p = [0, 1, 2].map(|i| q0[i] + tc * dq[i]);
        let rb = ((self.a + self.c).powi(2) + self.b * self.b).sqrt() * (1.0 + 1e-9) + 1e-9;
        let h2 = dot(p, p);
        if h2 > rb * rb {
            return;
        }
        let half = (rb * rb - h2).sqrt();
        let (slo, shi) = ((lo - tc).max(-half), (hi - tc).min(half));
        if !(slo <= shi) {
            return;
        }
        let ax = self.axis as usize;
        let (iu, iv) = ((ax + 1) % 3, (ax + 2) % 3);
        let r = [
            p[iu] * p[iu] + p[iv] * p[iv],
            2.0 * (p[iu] * dq[iu] + p[iv] * dq[iv]),
            dq[iu] * dq[iu] + dq[iv] * dq[iv],
        ];
        let w = [p[ax] * p[ax], 2.0 * p[ax] * dq[ax], dq[ax] * dq[ax]];
        let k = (self.c / self.b).powi(2);
        let a2 = self.a * self.a;
        // G = ρ² + a² - K = R + k·W + a² - c².
        let g = [
            r[0] + k * w[0] + a2 - self.c * self.c,
            r[1] + k * w[1],
            r[2] + k * w[2],
        ];
        let mut roots = [0.0f64; 8];
        let mut n_roots = 0usize;
        if self.a == 0.0 {
            // ρ² = K: the quartic would be G², without sign changes.
            poly_roots(&g, slo, shi, &mut roots, &mut n_roots);
        } else {
            let quartic = [
                g[0] * g[0] - 4.0 * a2 * r[0],
                2.0 * g[1] * g[0] - 4.0 * a2 * r[1],
                g[1] * g[1] + 2.0 * g[2] * g[0] - 4.0 * a2 * r[2],
                2.0 * g[2] * g[1],
                g[2] * g[2],
            ];
            poly_roots(&quartic, slo, shi, &mut roots, &mut n_roots);
        }
        for &s in &roots[..n_roots] {
            push(tc + s);
        }
    }
}

#[inline]
fn poly_eval(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
}

#[inline]
fn poly_eval_deriv(c: &[f64], x: f64) -> (f64, f64) {
    let mut f = 0.0;
    let mut df = 0.0;
    for &k in c.iter().rev() {
        df = df * x + f;
        f = f * x + k;
    }
    (f, df)
}

/// Sign changes of the polynomial `Σ c[i] xⁱ` on `[lo, hi]`: linear and
/// quadratic equations use closed form; cubic and quartic split `[lo, hi]`
/// at the derivative's roots and refine each bracket with safeguarded
/// Newton-Raphson + bisection, with zero heap allocations.
fn poly_roots(c: &[f64], lo: f64, hi: f64, out: &mut [f64; 8], out_len: &mut usize) {
    let mut n = c.len();
    while n > 0 && c[n - 1] == 0.0 {
        n -= 1;
    }
    let c = &c[..n];
    let mut emit = |r: f64| {
        if *out_len < out.len() {
            out[*out_len] = r;
            *out_len += 1;
        }
    };
    if n <= 1 {
        return;
    }
    if n == 2 {
        let r = -c[0] / c[1];
        if r >= lo && r <= hi {
            emit(r);
        }
        return;
    }
    if n == 3 {
        quadratic_roots(c[2], c[1], c[0], &mut |r| {
            if r >= lo && r <= hi {
                emit(r);
            }
        });
        return;
    }
    let mut dc = [0.0f64; 4];
    for i in 1..n {
        dc[i - 1] = i as f64 * c[i];
    }
    let mut crit = [0.0f64; 8];
    let mut n_crit = 0usize;
    poly_roots(&dc[..n - 1], lo, hi, &mut crit, &mut n_crit);
    let crit_slice = &mut crit[..n_crit];
    crit_slice.sort_unstable_by(f64::total_cmp);

    let mut cuts = [0.0f64; 10];
    cuts[0] = lo;
    let mut n_cuts = 1usize;
    for &x in crit_slice.iter() {
        if n_cuts < cuts.len() - 1 {
            cuts[n_cuts] = x;
            n_cuts += 1;
        }
    }
    cuts[n_cuts] = hi;
    n_cuts += 1;

    for win in cuts[..n_cuts].windows(2) {
        let (mut a, mut b) = (win[0], win[1]);
        let (fa, fb) = (poly_eval(c, a), poly_eval(c, b));
        if fa == 0.0 {
            emit(a);
            continue;
        }
        if fa.signum() == fb.signum() || fb == 0.0 {
            continue;
        }
        // f(a) and f(b) have opposite signs: safeguarded Newton + bisection.
        let mut x = 0.5 * (a + b);
        for _ in 0..100 {
            let (fx, dfx) = poly_eval_deriv(c, x);
            if fx == 0.0 {
                a = x;
                b = x;
                break;
            }
            if fx.signum() == fa.signum() {
                a = x;
            } else {
                b = x;
            }
            let m = 0.5 * (a + b);
            if m <= a || m >= b {
                break;
            }
            if dfx != 0.0 {
                let nx = x - fx / dfx;
                if nx > a && nx < b {
                    x = nx;
                    continue;
                }
            }
            x = m;
        }
        emit(0.5 * (a + b));
    }
    if poly_eval(c, hi) == 0.0 {
        emit(hi);
    }
}

impl Quadric {
    /// Side over the box with centre `c` and half-sizes `h` (centred form:
    /// `f(c + e) = f(c) + ∇f(c)·e + eᵀAe`).
    fn side_over(&self, c: Vec3, h: Vec3, tol: f64) -> Side {
        let d = sub(c, self.origin);
        // Cover the rounding of `c - origin`.
        let h: Vec3 =
            [0, 1, 2].map(|i| h[i] + 4.0 * f64::EPSILON * (c[i].abs() + self.origin[i].abs()));
        let m = self.matrix();
        let ad = [dot(m[0], d), dot(m[1], d), dot(m[2], d)];
        let f = dot(d, ad) + dot(self.b, d) + self.c;
        let g = [0, 1, 2].map(|i| 2.0 * ad[i] + self.b[i]);
        // f(c + u) = f(c) + Σ gᵢuᵢ + Σ mᵢᵢuᵢ² + 2 Σ mᵢⱼuᵢuⱼ over |uᵢ| ≤ hᵢ.
        let (lo, hi) = if m[0][1] == 0.0 && m[1][2] == 0.0 && m[0][2] == 0.0 {
            // No cross terms (spheres, axis-aligned cylinders, ellipsoids,
            // SQ): a sum of parabolas in one variable each, whose exact
            // range is the sum of theirs. The centred form below misses
            // the curvature: a box well away from a sphere stayed undecided,
            // so the outside cell of sparse.i kept hundreds of spheres in
            // its residual programs over the large octree leaves around them.
            let (mut lo, mut hi) = (0.0, 0.0);
            for i in 0..3 {
                let (a, gi, hi_) = (m[i][i], g[i], h[i]);
                let at = |u: f64| gi * u + a * u * u;
                let (e0, e1) = (at(-hi_), at(hi_));
                let (mut l, mut u) = (e0.min(e1), e0.max(e1));
                if a != 0.0 {
                    let v = -gi / (2.0 * a);
                    if v.abs() < hi_ {
                        let fv = at(v);
                        l = l.min(fv);
                        u = u.max(fv);
                    }
                }
                lo += l;
                hi += u;
            }
            (lo, hi)
        } else {
            let lin: f64 = (0..3).map(|i| g[i].abs() * h[i]).sum();
            let (mut qlo, mut qhi) = (0.0, 0.0);
            for i in 0..3 {
                let t = m[i][i] * h[i] * h[i];
                if t < 0.0 {
                    qlo += t;
                } else {
                    qhi += t;
                }
            }
            let off = 2.0
                * (m[0][1].abs() * h[0] * h[1]
                    + m[1][2].abs() * h[1] * h[2]
                    + m[0][2].abs() * h[0] * h[2]);
            (-lin + qlo - off, lin + qhi + off)
        };
        // Largest gradient length over the box, for the `tol · |∇f|` band.
        let gmax = norm(
            [0, 1, 2].map(|i| g[i].abs() + 2.0 * (0..3).map(|j| m[i][j].abs() * h[j]).sum::<f64>()),
        );
        let r = [0, 1, 2].map(|i| d[i].abs() + h[i]);
        let mag = (0..3)
            .map(|i| {
                (0..3).map(|j| m[i][j].abs() * r[i] * r[j]).sum::<f64>() + self.b[i].abs() * r[i]
            })
            .sum::<f64>()
            + self.c.abs();
        decide(f + lo, f + hi, tol * gmax, mag)
    }
}

/// Centre and half-sizes of a finite box, the half-sizes rounded up so that
/// `[c - h, c + h]` covers it.
fn center_half(b: &Aabb) -> (Vec3, Vec3) {
    let c = b.center();
    let h = [0, 1, 2].map(|i| {
        (b.hi[i] - c[i]).max(c[i] - b.lo[i]) + 4.0 * f64::EPSILON * (b.lo[i].abs() + b.hi[i].abs())
    });
    (c, h)
}

/// `Pos`/`Neg` when `[fmin, fmax]` clears the band `±tolg` with a margin for
/// the rounding of values of size `mag`.
fn decide(fmin: f64, fmax: f64, tolg: f64, mag: f64) -> Side {
    let band = tolg * (1.0 + 1e-6) + 1e-12 * mag;
    if fmin > band {
        Side::Pos
    } else if fmax < -band {
        Side::Neg
    } else {
        Side::On
    }
}

#[inline]
fn side_of(f: f64, g: f64, tol: f64) -> Side {
    if f > tol * g {
        Side::Pos
    } else if f < -tol * g {
        Side::Neg
    } else {
        Side::On
    }
}

/// Number of facets of a macrobody mnemonic, or `None` for an ordinary surface.
pub fn macrobody_facets(kind: &str) -> Option<usize> {
    match kind {
        "RPP" | "BOX" => Some(6),
        "SPH" => Some(1),
        "RCC" | "TRC" => Some(3),
        "RHP" | "HEX" => Some(8),
        "REC" => Some(3),
        "ELL" => Some(1),
        "WED" => Some(5),
        "ARB" => Some(6),
        _ => None,
    }
}

/// Build the shapes of a surface card in its own (possibly auxiliary) frame:
/// one shape for an ordinary surface, the facets in MCNP order for a
/// macrobody. Every shape is negative inside the body.
pub fn build(kind: &str, v: &[f64]) -> Result<Vec<Shape>, String> {
    let k = kind.to_ascii_uppercase();
    let need = |n: usize| -> Result<(), String> {
        if v.len() < n {
            Err(format!("{k} needs {n} coefficients, got {}", v.len()))
        } else {
            Ok(())
        }
    };
    let exact = |ns: &[usize]| -> Result<(), String> {
        if ns.contains(&v.len()) {
            Ok(())
        } else {
            Err(format!(
                "{k} with {} coefficients is not supported",
                v.len()
            ))
        }
    };
    let axis_of = |c: u8| (c - b'X') as usize;
    let unit = |i: usize| {
        let mut e = [0.0; 3];
        e[i] = 1.0;
        e
    };
    let shape = match k.as_str() {
        "P" => match v.len() {
            4 => Shape::plane([v[0], v[1], v[2]], v[3]),
            9 => plane_by_points([v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]])?,
            n => return Err(format!("P with {n} coefficients is not supported")),
        },
        "PX" | "PY" | "PZ" => {
            exact(&[1])?;
            Shape::plane(unit(axis_of(k.as_bytes()[1])), v[0])
        }
        "SO" => {
            exact(&[1])?;
            Shape::Quadric(Quadric::sphere([0.0; 3], v[0]))
        }
        "S" => {
            exact(&[4])?;
            Shape::Quadric(Quadric::sphere([v[0], v[1], v[2]], v[3]))
        }
        "SX" | "SY" | "SZ" => {
            exact(&[2])?;
            let mut c = [0.0; 3];
            c[axis_of(k.as_bytes()[1])] = v[0];
            Shape::Quadric(Quadric::sphere(c, v[1]))
        }
        "C/X" | "C/Y" | "C/Z" => {
            exact(&[3])?;
            let ax = axis_of(k.as_bytes()[2]);
            let mut o = [0.0; 3];
            o[(ax + 1) % 3] = v[0];
            o[(ax + 2) % 3] = v[1];
            if ax == 1 {
                // C/Y is given as `x z`.
                o = [v[0], 0.0, v[1]];
            }
            Shape::Quadric(Quadric::axial(ax, o, v[2] * v[2], 0.0))
        }
        "CX" | "CY" | "CZ" => {
            exact(&[1])?;
            Shape::Quadric(Quadric::axial(
                axis_of(k.as_bytes()[1]),
                [0.0; 3],
                v[0] * v[0],
                0.0,
            ))
        }
        "K/X" | "K/Y" | "K/Z" => {
            exact(&[4, 5])?;
            let ax = axis_of(k.as_bytes()[2]);
            cone(ax, [v[0], v[1], v[2]], v[3], v.get(4).copied())
        }
        "KX" | "KY" | "KZ" => {
            exact(&[2, 3])?;
            let ax = axis_of(k.as_bytes()[1]);
            let mut apex = [0.0; 3];
            apex[ax] = v[0];
            cone(ax, apex, v[1], v.get(2).copied())
        }
        "SQ" => {
            exact(&[10])?;
            Shape::Quadric(Quadric {
                origin: [v[7], v[8], v[9]],
                a: [v[0], v[1], v[2], 0.0, 0.0, 0.0],
                b: [2.0 * v[3], 2.0 * v[4], 2.0 * v[5]],
                c: v[6],
            })
        }
        "GQ" => {
            exact(&[10])?;
            Shape::Quadric(Quadric {
                origin: [0.0; 3],
                a: [v[0], v[1], v[2], 0.5 * v[3], 0.5 * v[4], 0.5 * v[5]],
                b: [v[6], v[7], v[8]],
                c: v[9],
            })
        }
        "TX" | "TY" | "TZ" => {
            exact(&[6])?;
            let ax = axis_of(k.as_bytes()[1]);
            if v[3] < 0.0 || v[4] <= 0.0 || v[5] <= 0.0 {
                return Err(format!("{k} with negative or zero radii"));
            }
            Shape::Torus(Box::new(Torus {
                frame: Affine::translation([-v[0], -v[1], -v[2]]),
                axis: ax as u8,
                a: v[3],
                b: v[4],
                c: v[5],
            }))
        }
        "X" | "Y" | "Z" => axisymmetric(axis_of(k.as_bytes()[0]), v)?,
        "RPP" => {
            exact(&[6])?;
            if v[0] > v[1] || v[2] > v[3] || v[4] > v[5] {
                return Err("RPP with min > max".into());
            }
            let mut out = Vec::with_capacity(6);
            for ax in 0..3 {
                let e = unit(ax);
                out.push(Shape::plane(e, v[2 * ax + 1]));
                out.push(Shape::plane(scale(e, -1.0), -v[2 * ax]));
            }
            return Ok(out);
        }
        "BOX" => {
            exact(&[12])?;
            let corner = [v[0], v[1], v[2]];
            let mut out = Vec::with_capacity(6);
            for i in 0..3 {
                let a = [v[3 + 3 * i], v[4 + 3 * i], v[5 + 3 * i]];
                let n = normalize(a);
                out.push(Shape::plane(n, dot(n, corner) + norm(a)));
                out.push(Shape::plane(scale(n, -1.0), -dot(n, corner)));
            }
            return Ok(out);
        }
        "SPH" => {
            exact(&[4])?;
            Shape::Quadric(Quadric::sphere([v[0], v[1], v[2]], v[3]))
        }
        "RCC" | "TRC" => {
            need(if k == "RCC" { 7 } else { 8 })?;
            exact(if k == "RCC" { &[7] } else { &[8] })?;
            let base = [v[0], v[1], v[2]];
            let h = [v[3], v[4], v[5]];
            let len = norm(h);
            let u = scale(h, 1.0 / len);
            let (r1, r2) = if k == "RCC" {
                (v[6], v[6])
            } else {
                (v[6], v[7])
            };
            let lateral = if r1 == r2 {
                Shape::Quadric(Quadric::general_axial(base, u, r1 * r1, 0.0))
            } else {
                let s0 = r1 / (r1 - r2) * len;
                let apex = crate::vec3::add(base, scale(u, s0));
                let t = (r2 - r1) / len;
                let toward_base = if s0 > 0.0 { scale(u, -1.0) } else { u };
                Shape::Cone(Box::new(Cone {
                    q: Quadric::general_axial(apex, u, 0.0, t * t),
                    apex,
                    axis: toward_base,
                }))
            };
            return Ok(vec![
                lateral,
                Shape::plane(u, dot(u, base) + len),
                Shape::plane(scale(u, -1.0), -dot(u, base)),
            ]);
        }
        "RHP" | "HEX" => return rhp(v),
        "REC" => return rec(v),
        "ELL" => {
            exact(&[7])?;
            return ell(v).map(|s| vec![s]);
        }
        "WED" => {
            exact(&[12])?;
            return wed(v);
        }
        "ARB" => {
            exact(&[30])?;
            return arb(v);
        }
        other => return Err(format!("surface type {other} is not supported")),
    };
    Ok(vec![shape])
}

fn v3(v: &[f64], i: usize) -> Vec3 {
    [v[i], v[i + 1], v[i + 2]]
}

/// A plane through `p` with outward normal `n` (negative on the inside).
fn plane_at(n: Vec3, p: Vec3) -> Shape {
    let n = normalize(n);
    Shape::Plane { n, d: dot(n, p) }
}

/// Rotate `r` about unit axis `u` by `deg` degrees (right-handed).
fn rotate(r: Vec3, u: Vec3, deg: f64) -> Vec3 {
    let (s, c) = deg.to_radians().sin_cos();
    let k = cross(u, r);
    let d = dot(u, r);
    [0, 1, 2].map(|i| r[i] * c + k[i] * s + u[i] * d * (1.0 - c))
}

/// `RHP`/`HEX v h r [s t]`: hexagonal prism. Facets: `±r`, `±s`, `±t`
/// (each pair: the side the vector points to, then the opposite side), then
/// top and bottom. Without `s` and `t` the hexagon is regular.
fn rhp(v: &[f64]) -> Result<Vec<Shape>, String> {
    if v.len() != 9 && v.len() != 15 {
        return Err(format!(
            "RHP with {} coefficients is not supported",
            v.len()
        ));
    }
    let (base, h, r) = (v3(v, 0), v3(v, 3), v3(v, 6));
    let u = normalize(h);
    check_orthogonal(&[r], u, "RHP")?;
    let (s, t) = if v.len() == 15 {
        (v3(v, 9), v3(v, 12))
    } else {
        (rotate(r, u, 60.0), rotate(r, u, 120.0))
    };
    let mut out = Vec::with_capacity(8);
    for w in [r, s, t] {
        out.push(plane_at(w, crate::vec3::add(base, w)));
        out.push(plane_at(scale(w, -1.0), sub(base, w)));
    }
    out.push(plane_at(u, crate::vec3::add(base, h)));
    out.push(plane_at(scale(u, -1.0), base));
    Ok(out)
}

/// `REC v h a1 (a2 | b)`: right elliptical cylinder. Facets: lateral, top,
/// bottom.
fn rec(v: &[f64]) -> Result<Vec<Shape>, String> {
    if v.len() != 10 && v.len() != 12 {
        return Err(format!(
            "REC with {} coefficients is not supported",
            v.len()
        ));
    }
    let (base, h, a1) = (v3(v, 0), v3(v, 3), v3(v, 6));
    let u = normalize(h);
    let a2 = if v.len() == 12 {
        v3(v, 9)
    } else {
        scale(normalize(cross(u, a1)), v[9])
    };
    check_orthogonal(&[a1, a2], u, "REC")?;
    if dot(normalize(a1), normalize(a2)).abs() > 1e-6 {
        return Err("REC axis vectors are not orthogonal".into());
    }
    // (d·â1)²/|a1|² + (d·â2)²/|a2|² - 1, as a quadric about `base`.
    let (l1, l2) = (norm(a1), norm(a2));
    let (e1, e2) = (scale(a1, 1.0 / (l1 * l1)), scale(a2, 1.0 / (l2 * l2)));
    let q = outer_sum(&[(e1, 1.0), (e2, 1.0)]);
    Ok(vec![
        Shape::Quadric(Quadric {
            origin: base,
            a: q,
            b: [0.0; 3],
            c: -1.0,
        }),
        plane_at(u, crate::vec3::add(base, h)),
        plane_at(scale(u, -1.0), base),
    ])
}

/// MCNP rejects macrobodies whose defining vectors are not orthogonal to
/// their height vector.
fn check_orthogonal(vs: &[Vec3], u: Vec3, what: &str) -> Result<(), String> {
    if vs.iter().any(|w| dot(normalize(*w), u).abs() > 1e-6) {
        return Err(format!(
            "{what} vectors are not orthogonal to the height vector"
        ));
    }
    Ok(())
}

/// `Σ w_k (e_k e_kᵀ)` in `[xx, yy, zz, xy, yz, zx]` form.
fn outer_sum(terms: &[(Vec3, f64)]) -> [f64; 6] {
    let mut a = [0.0; 6];
    for (e, w) in terms {
        a[0] += w * e[0] * e[0];
        a[1] += w * e[1] * e[1];
        a[2] += w * e[2] * e[2];
        a[3] += w * e[0] * e[1];
        a[4] += w * e[1] * e[2];
        a[5] += w * e[2] * e[0];
    }
    a
}

/// `ELL v1 v2 rm`: ellipsoid of revolution. With `rm > 0`, `v1` and `v2` are
/// the foci and `rm` the major radius; with `rm < 0`, `v1` is the centre,
/// `v2` the major-axis vector and `|rm|` the minor radius.
fn ell(v: &[f64]) -> Result<Shape, String> {
    let (p1, p2, rm) = (v3(v, 0), v3(v, 3), v[6]);
    let (center, axis, a, b) = if rm > 0.0 {
        let c = 0.5 * norm(sub(p2, p1));
        if rm <= c {
            return Err("ELL major radius not larger than the focal distance".into());
        }
        let axis = if c > 0.0 {
            normalize(sub(p2, p1))
        } else {
            [0.0, 0.0, 1.0]
        };
        (
            scale(crate::vec3::add(p1, p2), 0.5),
            axis,
            rm,
            (rm * rm - c * c).sqrt(),
        )
    } else if rm < 0.0 {
        (p1, normalize(p2), norm(p2), -rm)
    } else {
        return Err("ELL with zero radius".into());
    };
    // |d|²/b² + (d·u)²(1/a² - 1/b²) - 1
    let iso = 1.0 / (b * b);
    let mut q = outer_sum(&[(axis, 1.0 / (a * a) - iso)]);
    q[0] += iso;
    q[1] += iso;
    q[2] += iso;
    Ok(Shape::Quadric(Quadric {
        origin: center,
        a: q,
        b: [0.0; 3],
        c: -1.0,
    }))
}

/// `WED v a1 a2 h`: right-angle wedge. Facets: the slanted face, the face
/// containing `a2`, the face containing `a1`, top, bottom (checked against
/// MCNP).
fn wed(v: &[f64]) -> Result<Vec<Shape>, String> {
    let (p, a1, a2, h) = (v3(v, 0), v3(v, 3), v3(v, 6), v3(v, 9));
    let u = normalize(h);
    check_orthogonal(&[a1, a2], u, "WED")?;
    if dot(normalize(a1), normalize(a2)).abs() > 1e-6 {
        return Err("WED legs are not orthogonal".into());
    }
    let tip1 = crate::vec3::add(p, a1);
    // Slant: through v+a1 and v+a2, parallel to h, facing away from v.
    let along = sub(a2, a1);
    let mut n = cross(along, u);
    if dot(n, sub(p, tip1)) > 0.0 {
        n = scale(n, -1.0);
    }
    // Side faces: each contains one leg and h; outward is away from the other leg.
    let side = |leg: Vec3, other: Vec3| {
        let mut n = cross(leg, u);
        if dot(n, other) > 0.0 {
            n = scale(n, -1.0);
        }
        plane_at(n, p)
    };
    Ok(vec![
        plane_at(n, tip1),
        side(a2, a1),
        side(a1, a2),
        plane_at(u, crate::vec3::add(p, h)),
        plane_at(scale(u, -1.0), p),
    ])
}

/// `ARB`: eight corners, then six four-digit facet descriptors naming the
/// corners of each face (`0` for no face). Each face is the plane through
/// its first three corners, negative on the side of the body's centre.
fn arb(v: &[f64]) -> Result<Vec<Shape>, String> {
    let corners: Vec<Vec3> = (0..8).map(|i| v3(v, 3 * i)).collect();
    let mut used = Vec::new();
    let mut faces = Vec::new();
    for &f in &v[24..30] {
        if f == 0.0 {
            faces.push(None);
            continue;
        }
        // Zeros pad a three-corner face (`1230`).
        let digits: Vec<usize> = format!("{}", f.round() as i64)
            .bytes()
            .map(|b| (b - b'0') as usize)
            .filter(|&d| d != 0)
            .collect();
        if digits.len() < 3 || digits.iter().any(|&d| d > 8) {
            return Err(format!("ARB facet `{f}` is not valid"));
        }
        used.extend(digits.iter().copied());
        faces.push(Some(digits));
    }
    used.sort_unstable();
    used.dedup();
    let mut centre = [0.0; 3];
    for &i in &used {
        centre = crate::vec3::add(centre, corners[i - 1]);
    }
    let centre = scale(centre, 1.0 / used.len() as f64);
    faces
        .into_iter()
        .map(|f| match f {
            // An absent face constrains nothing: always negative.
            None => Ok(Shape::Plane {
                n: [0.0; 3],
                d: 1.0,
            }),
            Some(d) => {
                let (a, b, c) = (corners[d[0] - 1], corners[d[1] - 1], corners[d[2] - 1]);
                let mut n = cross(sub(b, a), sub(c, a));
                if norm(n) == 0.0 {
                    return Err("ARB facet through collinear corners".into());
                }
                if dot(n, sub(centre, a)) > 0.0 {
                    n = scale(n, -1.0);
                }
                Ok(plane_at(n, a))
            }
        })
        .collect()
}

fn cone(ax: usize, apex: Vec3, t2: f64, sheet: Option<f64>) -> Shape {
    let q = Quadric::axial(ax, apex, 0.0, t2);
    match sheet {
        Some(s) if s != 0.0 => {
            let mut axis = [0.0; 3];
            axis[ax] = s.signum();
            Shape::Cone(Box::new(Cone { q, apex, axis }))
        }
        _ => Shape::Quadric(q),
    }
}

/// `X`, `Y`, `Z` defined by two (s, r) points: a plane, a cylinder or a
/// one-sheet cone about the axis.
fn axisymmetric(ax: usize, v: &[f64]) -> Result<Shape, String> {
    if v.len() != 4 {
        return Err(format!(
            "axisymmetric surface with {} coefficients is not supported",
            v.len()
        ));
    }
    let (s1, r1, s2, r2) = (v[0], v[1], v[2], v[3]);
    let mut e = [0.0; 3];
    e[ax] = 1.0;
    if s1 == s2 {
        return Ok(Shape::plane(e, s1));
    }
    if r1 == r2 {
        return Ok(Shape::Quadric(Quadric::axial(ax, [0.0; 3], r1 * r1, 0.0)));
    }
    let t = (r2 - r1) / (s2 - s1);
    let s0 = s1 - r1 / t;
    let mut apex = [0.0; 3];
    apex[ax] = s0;
    let mut axis = [0.0; 3];
    axis[ax] = if s1 > s0 { 1.0 } else { -1.0 };
    Ok(Shape::Cone(Box::new(Cone {
        q: Quadric::axial(ax, apex, 0.0, t * t),
        apex,
        axis,
    })))
}

/// `P` by three points. The origin is on the negative side; for a plane
/// through the origin, (0,0,∞) is positive, then (0,∞,0), then (∞,0,0).
fn plane_by_points(p1: Vec3, p2: Vec3, p3: Vec3) -> Result<Shape, String> {
    let n = cross(sub(p2, p1), sub(p3, p1));
    let l = norm(n);
    if l == 0.0 {
        return Err("P by three collinear points".into());
    }
    let n = scale(n, 1.0 / l);
    let d = dot(n, p1);
    let scale_ref = norm(p1).max(norm(p2)).max(norm(p3)).max(1.0);
    let flip = if d.abs() > 1e-12 * scale_ref {
        d < 0.0
    } else if n[2].abs() > 1e-12 {
        n[2] < 0.0
    } else if n[1].abs() > 1e-12 {
        n[1] < 0.0
    } else {
        n[0] < 0.0
    };
    Ok(if flip {
        Shape::Plane {
            n: scale(n, -1.0),
            d: -d,
        }
    } else {
        Shape::Plane { n, d }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(kind: &str, v: &[f64], p: Vec3) -> Side {
        build(kind, v).unwrap()[0].side(p, 1e-9)
    }

    #[test]
    fn side_over_agrees_with_every_point() {
        let mut r = 0x2545f4914f6cdd1du64;
        let mut f = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            (r >> 11) as f64 / (1u64 << 53) as f64
        };
        let shapes: Vec<Shape> = [
            ("P", vec![0.3, -0.5, 0.8, 1.2]),
            ("PZ", vec![2.0]),
            ("S", vec![1.0, 2.0, -1.0, 3.0]),
            ("C/Y", vec![5.0, 3.0, 1.5]),
            (
                "GQ",
                vec![1.0, 0.8, 0.3, 0.2, -0.1, 0.05, -2.0, 1.0, 0.5, -4.0],
            ),
            ("K/Z", vec![1.0, -1.0, 0.5, 0.7, 1.0]),
            ("K/X", vec![0.0, 0.0, 0.0, 0.3, -1.0]),
            ("KZ", vec![0.0, 2.0]),
            // Separable, mixed signs: the parabola vertex inside boxes.
            (
                "SQ",
                vec![1.0, -0.5, 2.0, 0.3, -0.2, 0.1, -1.0, 0.5, -0.5, 1.0],
            ),
            ("C/Z", vec![-1.0, 2.0, 0.7]),
        ]
        .iter()
        .flat_map(|(k, v)| build(k, v).unwrap())
        .collect();
        let tol = 1e-8;
        let mut decided = 0;
        for shape in &shapes {
            for _ in 0..4000 {
                let size = 10f64.powf(-6.0 + 7.0 * f());
                let lo = [0, 1, 2].map(|_| -6.0 + 12.0 * f());
                let b = Aabb {
                    lo,
                    hi: [0, 1, 2].map(|i| lo[i] + size * (0.2 + f())),
                };
                let s = shape.side_over(&b, tol);
                if s == Side::On {
                    continue;
                }
                decided += 1;
                for k in 0..40 {
                    let p: Vec3 = if k < 8 {
                        [0, 1, 2].map(|i| if k >> i & 1 == 0 { b.lo[i] } else { b.hi[i] })
                    } else {
                        [0, 1, 2].map(|i| b.lo[i] + f() * (b.hi[i] - b.lo[i]))
                    };
                    assert_eq!(shape.side(p, tol), s, "{shape:?} {b:?} {p:?}");
                }
            }
        }
        assert!(decided > shapes.len() * 2000, "{decided}");
    }

    /// A box clear of a sphere is decided whatever its size next to the
    /// distance: the outside of `sparse.i`'s spheres over large octree leaves.
    #[test]
    fn side_over_decides_boxes_clear_of_a_sphere() {
        let s = &build("S", &[0.0, 0.0, 0.0, 1.0]).unwrap()[0];
        for (lo, hi) in [
            ([1.5, -40.0, -40.0], [80.0, 40.0, 40.0]),
            ([0.8, 0.8, 0.8], [3.0, 3.0, 3.0]),
            ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
        ] {
            let b = Aabb { lo, hi };
            let want = if lo[0] > 0.0 { Side::Pos } else { Side::Neg };
            assert_eq!(s.side_over(&b, 1e-8), want, "{b:?}");
        }
    }

    /// Between consecutive crossings the side never changes: checked at
    /// many points of random lines through every kind of surface.
    #[test]
    fn crossings_bound_every_side_change() {
        let mut r = 0x9e3779b97f4a7c15u64;
        let mut f = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            (r >> 11) as f64 / (1u64 << 53) as f64
        };
        let shapes: Vec<Shape> = [
            ("P", vec![0.3, -0.5, 0.8, 1.2]),
            ("S", vec![1.0, 2.0, -1.0, 3.0]),
            ("C/Y", vec![1.0, 0.5, 1.5]),
            (
                "GQ",
                vec![1.0, 0.8, 0.3, 0.2, -0.1, 0.05, -2.0, 1.0, 0.5, -4.0],
            ),
            ("K/Z", vec![1.0, -1.0, 0.5, 0.7, 1.0]),
            ("KX", vec![0.5, 0.3]),
            ("TZ", vec![0.5, -0.3, 0.2, 3.0, 1.0, 0.7]),
            ("TX", vec![0.0, 0.0, 0.0, 2.0, 0.5, 1.5]),
            ("TY", vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]),
            ("TZ", vec![0.0, 0.0, 0.0, 1.0, 2.0, 1.5]),
        ]
        .iter()
        .flat_map(|(k, v)| build(k, v).unwrap())
        .collect();
        let tol = 1e-12;
        for shape in &shapes {
            let mut found = 0;
            for _ in 0..3000 {
                let o = [0, 1, 2].map(|_| -6.0 + 12.0 * f());
                let d = normalize([0, 1, 2].map(|_| f() - 0.5));
                let mut ts = Vec::new();
                shape.crossings(o, d, -10.0, 10.0, &mut ts);
                ts.sort_by(f64::total_cmp);
                found += ts.len();
                ts.insert(0, -10.0);
                ts.push(10.0);
                for w in ts.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if b - a < 1e-6 {
                        continue;
                    }
                    let side_at = |t: f64| shape.side(crate::vec3::add(o, scale(d, t)), tol);
                    let want = side_at(0.5 * (a + b));
                    for k in 1..20 {
                        let t = a + (b - a) * (k as f64 / 20.0);
                        let s = side_at(t);
                        assert!(
                            s == want || s == Side::On || (t - a).min(b - t) < 1e-6,
                            "{shape:?} o={o:?} d={d:?} [{a}, {b}] t={t}"
                        );
                    }
                }
            }
            assert!(found > 100, "{shape:?}: {found}");
        }
    }

    #[test]
    fn crossings_of_simple_shapes() {
        let x = [1.0, 0.0, 0.0];
        let mut ts = Vec::new();
        build("SO", &[2.0]).unwrap()[0].crossings([-5.0, 0.0, 0.0], x, -100.0, 100.0, &mut ts);
        ts.sort_by(f64::total_cmp);
        assert!(
            (ts[0] - 3.0).abs() < 1e-14 && (ts[1] - 7.0).abs() < 1e-14,
            "{ts:?}"
        );
        // Along a torus axis plane: four roots at ρ = a ± c.
        ts.clear();
        build("TZ", &[0.0, 0.0, 0.0, 5.0, 1.0, 1.0]).unwrap()[0].crossings(
            [-10.0, 0.0, 0.0],
            x,
            -100.0,
            100.0,
            &mut ts,
        );
        ts.sort_by(f64::total_cmp);
        let want = [4.0, 6.0, 14.0, 16.0];
        assert_eq!(ts.len(), 4, "{ts:?}");
        for (a, b) in ts.iter().zip(want) {
            assert!((a - b).abs() < 1e-9, "{ts:?}");
        }
        // Parallel to a cylinder axis: no roots.
        ts.clear();
        build("CZ", &[1.0]).unwrap()[0].crossings(
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            -1e9,
            1e9,
            &mut ts,
        );
        assert!(ts.is_empty());
    }

    #[test]
    fn basic_senses() {
        assert_eq!(side("PX", &[1.0], [0.0; 3]), Side::Neg);
        assert_eq!(side("SO", &[1.0], [0.0; 3]), Side::Neg);
        assert_eq!(side("S", &[5.0, 0.0, 0.0, 1.0], [0.0; 3]), Side::Pos);
        assert_eq!(side("C/Y", &[5.0, 3.0, 1.0], [5.0, 100.0, 3.5]), Side::Neg);
        assert_eq!(side("C/Y", &[5.0, 3.0, 1.0], [3.5, 0.0, 5.0]), Side::Pos);
        assert_eq!(side("CZ", &[2.0], [1.0, 1.0, -50.0]), Side::Neg);
        assert_eq!(
            side("TZ", &[0.0, 0.0, 0.0, 5.0, 1.0, 1.0], [5.0, 0.0, 0.5]),
            Side::Neg
        );
        assert_eq!(
            side("TZ", &[0.0, 0.0, 0.0, 5.0, 1.0, 1.0], [0.0; 3]),
            Side::Pos
        );
        assert_eq!(side("PZ", &[0.0], [1.0, 1.0, 0.0]), Side::On);
    }

    #[test]
    fn one_sheet_cone() {
        // Cone along +z from the origin, slope 1.
        assert_eq!(side("KZ", &[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]), Side::Neg);
        assert_eq!(side("KZ", &[0.0, 1.0, 1.0], [0.0, 0.0, -1.0]), Side::Pos);
        assert_eq!(side("KZ", &[0.0, 1.0], [0.0, 0.0, -1.0]), Side::Neg);
        assert_eq!(side("KZ", &[0.0, 1.0, -1.0], [0.0, 0.0, -1.0]), Side::Neg);
    }

    #[test]
    fn macrobody_facets_rpp_rcc() {
        let f = build("RPP", &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0]).unwrap();
        let p = [0.5, 1.0, 1.5];
        assert!(f.iter().all(|s| s.side(p, 1e-9) == Side::Neg));
        assert_eq!(f[0].side([1.5, 1.0, 1.5], 1e-9), Side::Pos);
        assert_eq!(f[1].side([-0.5, 1.0, 1.5], 1e-9), Side::Pos);
        let f = build("RCC", &[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0]).unwrap();
        assert_eq!(f[0].side([1.0, 1.0, 50.0], 1e-9), Side::Neg);
        assert_eq!(f[1].side([0.0, 0.0, 11.0], 1e-9), Side::Pos);
        assert_eq!(f[2].side([0.0, 0.0, -1.0], 1e-9), Side::Pos);
        let f = build("TRC", &[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0, 1.0]).unwrap();
        assert_eq!(f[0].side([1.9, 0.0, 0.0], 1e-9), Side::Neg);
        assert_eq!(f[0].side([1.9, 0.0, 10.0], 1e-9), Side::Pos);
        assert_eq!(f[0].side([0.0, 0.0, 25.0], 1e-9), Side::Pos);
    }

    #[test]
    fn transformed_quadric_matches_point_transform() {
        let t = crate::transform::from_mcnp(
            &[
                3.0, -1.0, 2.0, 30.0, 60.0, 90.0, 120.0, 30.0, 90.0, 90.0, 90.0, 0.0,
            ],
            true,
        )
        .unwrap();
        for (kind, v) in [
            ("C/Z", vec![1.0, 2.0, 0.7]),
            ("K/X", vec![1.0, 2.0, 3.0, 0.3, 1.0]),
            ("TY", vec![1.0, 0.0, 0.0, 3.0, 0.5, 0.8]),
            ("P", vec![1.0, 2.0, 3.0, 4.0]),
        ] {
            let local = build(kind, &v).unwrap().remove(0);
            let global = local.to_parent(&t);
            for i in 0..2000 {
                let x = i as f64;
                let p = [
                    (x * 0.37).sin() * 5.0,
                    (x * 0.11).cos() * 5.0,
                    (x * 0.23).sin() * 5.0,
                ];
                assert_eq!(
                    local.side(t.apply(p), 1e-9),
                    global.side(p, 1e-9),
                    "{kind} at {p:?}"
                );
            }
        }
    }
}
