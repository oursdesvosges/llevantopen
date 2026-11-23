//! Repeated structures: hexahedral (`LAT=1`) and hexagonal (`LAT=2`) lattices.

use crate::bbox::Aabb;
use crate::expr;
use crate::surface::Shape;
use crate::transform::Affine;
use crate::vec3::{add, cross, dot, norm, scale, sub, Vec3};

/// What fills a cell or a lattice element: a universe, seen through a
/// transform from the filled cell's frame.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Target {
    /// Index into [`crate::Model::units`].
    pub(crate) unit: u32,
    pub(crate) transform: Option<Affine>,
}

impl Target {
    /// Index into [`Model::units`](crate::Model::units) of the universe.
    pub fn unit(&self) -> u32 {
        self.unit
    }

    /// The transform from the filled cell's frame to the universe's, when
    /// not the identity.
    pub fn transform(&self) -> Option<&Affine> {
        self.transform.as_ref()
    }

    /// A point of the filled cell's frame in the universe's frame.
    #[inline]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        match &self.transform {
            Some(t) => t.apply(p),
            None => p,
        }
    }
}

/// How the elements of a lattice are filled.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum LatticeFill {
    /// Every element holds the same universe.
    Uniform(Target),
    /// A fill array over `lo ..= lo + size - 1`, `i` varying fastest. `None`
    /// is an element filled by the lattice cell itself (its own universe
    /// number in the array). Elements outside the range do not exist.
    Array {
        lo: [i32; 3],
        size: [i32; 3],
        elements: Vec<Option<Target>>,
    },
}

/// What lies in one lattice element.
#[derive(Clone, Copy, Debug)]
pub enum Element<'a> {
    Fill(&'a Target),
    /// Filled by the lattice cell itself (its material).
    Own,
    /// Outside the fill array: the element does not exist.
    Missing,
}

/// A lattice (`LAT=1` or `LAT=2`): element `[i,j,k]` is element `[0,0,0]` shifted by
/// `i·a + j·b + k·c`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Lattice {
    pub(crate) hexagonal: bool,
    /// Centre of element `[0,0,0]`.
    pub(crate) center: Vec3,
    /// `a`, `b`, `c`; for a two-dimensional lattice `c` is the unit axis.
    pub(crate) basis: [Vec3; 3],
    /// `dual[i] · basis[j] = δ_ij`.
    pub(crate) dual: [Vec3; 3],
    /// False for the axis of a two-dimensional lattice.
    pub(crate) bounded: [bool; 3],
    /// Outward side planes of element `[0,0,0]` (`n·x ≤ d` inside), used to
    /// pick the hexagon a point falls in. The first of `faces`.
    pub(crate) sides: Vec<(Vec3, f64)>,
    /// Every outward plane of element `[0,0,0]` (sides, then the bases of a
    /// three-dimensional lattice) and the compiled surface of each. The
    /// planes tile space exactly and are where [`Lattice::index`] changes
    /// element (`tiling_faces`), not the input's rounded planes.
    pub(crate) faces: Vec<(Vec3, f64)>,
    pub(crate) face_surfaces: Vec<u32>,
    pub(crate) fill: LatticeFill,
}

/// Ray direction projected onto a [`Lattice`]'s faces, for fast stepping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LatticeRay {
    nd: [f64; 8],
    n_faces: usize,
}

impl Lattice {
    /// Whether the lattice is hexagonal (`LAT=2`).
    pub fn hexagonal(&self) -> bool {
        self.hexagonal
    }

    /// Index of the element containing `p` (in the lattice cell's frame).
    #[inline]
    pub fn index(&self, p: Vec3) -> [i32; 3] {
        let d = sub(p, self.center);
        let mut ix = [0i32; 3];
        for i in 0..3 {
            if self.bounded[i] {
                ix[i] = (dot(self.dual[i], d) + 0.5).floor() as i32;
            }
        }
        if self.hexagonal {
            // The parallelogram cell is not the hexagon: of the nearby
            // elements, take the one whose hexagon holds `p` most deeply.
            let (i0, j0) = (ix[0], ix[1]);
            let mut best = (f64::NEG_INFINITY, i0, j0);
            for di in -1..=1 {
                for dj in -1..=1 {
                    let (i, j) = (i0 + di, j0 + dj);
                    let q = sub(
                        sub(p, scale(self.basis[0], i as f64)),
                        scale(self.basis[1], j as f64),
                    );
                    let margin = self
                        .sides
                        .iter()
                        .map(|(n, dd)| dd - dot(*n, q))
                        .fold(f64::INFINITY, f64::min);
                    if margin > best.0 {
                        best = (margin, i, j);
                    }
                }
            }
            ix[0] = best.1;
            ix[1] = best.2;
        }
        ix
    }

    /// `p` moved into element `[0,0,0]`.
    #[inline]
    pub(crate) fn to_element(&self, p: Vec3, ix: [i32; 3]) -> Vec3 {
        let mut q = p;
        for i in 0..3 {
            if ix[i] != 0 {
                q = sub(q, scale(self.basis[i], ix[i] as f64));
            }
        }
        q
    }

    /// Box around element `ix` in the lattice cell's frame, padded outwards;
    /// unbounded along the world axes the axis of a two-dimensional lattice
    /// leans on. Corners of the element's planes, taken three at a time.
    pub(crate) fn element_box(&self, ix: [i32; 3]) -> Aabb {
        // Two planes across each unbounded axis, its extent dropped after.
        let mut planes = self.faces.clone();
        for i in 0..3 {
            if !self.bounded[i] {
                let a = self.basis[i];
                planes.push((a, dot(a, self.center) + 1.0));
                planes.push((scale(a, -1.0), 1.0 - dot(a, self.center)));
            }
        }
        let scale_len = planes
            .iter()
            .map(|(_, d)| d.abs())
            .fold(norm(self.center), f64::max)
            .max(1.0);
        let tol = 1e-9 * scale_len;
        let mut b = Aabb::EMPTY;
        let n = planes.len();
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let Ok(p) = meet(planes[i], planes[j], planes[k]) else {
                        continue;
                    };
                    if planes.iter().all(|(m, d)| dot(*m, p) <= d + tol) {
                        b.grow_point(p);
                    }
                }
            }
        }
        if b.is_empty() {
            return Aabb::INFINITE;
        }
        let pad = 1e-6 * scale_len;
        let shift = (0..3).fold([0.0; 3], |s, i| add(s, scale(self.basis[i], ix[i] as f64)));
        let mut out = Aabb {
            lo: [0, 1, 2].map(|k| b.lo[k] - pad + shift[k]),
            hi: [0, 1, 2].map(|k| b.hi[k] + pad + shift[k]),
        };
        for i in 0..3 {
            if !self.bounded[i] {
                for k in 0..3 {
                    if self.basis[i][k].abs() > 1e-12 {
                        out.lo[k] = f64::NEG_INFINITY;
                        out.hi[k] = f64::INFINITY;
                    }
                }
            }
        }
        out
    }

    /// Precomputed ray direction projections onto this lattice's faces.
    #[inline]
    pub(crate) fn prepare_ray(&self, d: Vec3) -> LatticeRay {
        debug_assert!(self.faces.len() <= 8);
        let mut nd = [0.0; 8];
        for (k, &(n, _)) in self.faces.iter().enumerate() {
            nd[k] = dot(n, d);
        }
        LatticeRay {
            nd,
            n_faces: self.faces.len(),
        }
    }

    /// Range of `t` over which the line `o + t·d` lies in element `ix`, and
    /// the face it leaves through (an index into `faces`, `None` when it
    /// never leaves), using precomputed ray face projections.
    #[inline]
    pub(crate) fn element_range_ray(
        &self,
        o: Vec3,
        ray: &LatticeRay,
        ix: [i32; 3],
    ) -> (f64, f64, Option<usize>) {
        let q = self.to_element(o, ix);
        let (mut lo, mut hi, mut exit) = (f64::NEG_INFINITY, f64::INFINITY, None);
        for k in 0..ray.n_faces {
            let (n, off) = self.faces[k];
            let nd = ray.nd[k];
            let gap = off - dot(n, q);
            if nd == 0.0 {
                if gap < 0.0 {
                    return (f64::INFINITY, f64::NEG_INFINITY, None);
                }
                continue;
            }
            let t = gap / nd;
            if nd > 0.0 {
                if t < hi {
                    hi = t;
                    exit = Some(k);
                }
            } else if t > lo {
                lo = t;
            }
        }
        (lo, hi, exit)
    }

    /// Index of the neighbouring element directly across face `face`.
    #[inline]
    pub(crate) fn step_across(&self, ix: [i32; 3], face: usize) -> [i32; 3] {
        let [dx, dy, dz] = match (self.hexagonal, face) {
            (_, 0) => [1, 0, 0],
            (_, 1) => [-1, 0, 0],
            (_, 2) => [0, 1, 0],
            (_, 3) => [0, -1, 0],
            (true, 4) => [-1, 1, 0],
            (true, 5) => [1, -1, 0],
            (true, 6) | (false, 4) => [0, 0, 1],
            (true, 7) | (false, 5) => [0, 0, -1],
            _ => [0, 0, 0],
        };
        [ix[0] + dx, ix[1] + dy, ix[2] + dz]
    }

    /// Elements around `ix` that share at least a corner with it.
    pub(crate) fn neighbours(&self, ix: [i32; 3]) -> impl Iterator<Item = [i32; 3]> + '_ {
        let k = if self.bounded[2] { 1 } else { 0 };
        (-1..=1).flat_map(move |di| {
            (-1..=1).flat_map(move |dj| {
                (-k..=k)
                    .filter(move |&dk| (di, dj, dk) != (0, 0, 0))
                    .map(move |dk| [ix[0] + di, ix[1] + dj, ix[2] + dk])
            })
        })
    }

    /// The fill array's lowest index and size along each axis, `None` for
    /// a lattice filled with one universe everywhere.
    pub fn fill_array(&self) -> Option<([i32; 3], [i32; 3])> {
        match &self.fill {
            LatticeFill::Uniform(_) => None,
            LatticeFill::Array { lo, size, .. } => Some((*lo, *size)),
        }
    }

    /// What lies in element `ix`.
    pub fn element(&self, ix: [i32; 3]) -> Element<'_> {
        match &self.fill {
            LatticeFill::Uniform(t) => Element::Fill(t),
            LatticeFill::Array { lo, size, elements } => {
                let mut flat = 0usize;
                for i in (0..3).rev() {
                    let k = ix[i] - lo[i];
                    if k < 0 || k >= size[i] {
                        return Element::Missing;
                    }
                    flat = flat * size[i] as usize + k as usize;
                }
                match &elements[flat] {
                    Some(t) => Element::Fill(t),
                    None => Element::Own,
                }
            }
        }
    }

    /// Every universe the lattice can hand points to, once per element
    /// that holds it.
    pub fn targets(&self) -> impl Iterator<Item = &Target> + '_ {
        let (one, array) = match &self.fill {
            LatticeFill::Uniform(t) => (Some(t), None),
            LatticeFill::Array { elements, .. } => (None, Some(elements)),
        };
        one.into_iter().chain(array.into_iter().flatten().flatten())
    }
}

/// Outward planes of a lattice element program, in MCNP surface order.
fn outward_planes(code: &[u32], shapes: &[Shape]) -> Result<Vec<(Vec3, f64)>, String> {
    if code
        .first()
        .is_some_and(|w| w & expr::NODE != 0 && w & expr::OR != 0)
    {
        return Err("lattice element must be an intersection of planes".into());
    }
    expr::leaves(code)
        .map(|(s, want_pos)| match shapes[s as usize] {
            Shape::Plane { n, d } if want_pos => Ok((scale(n, -1.0), -d)),
            Shape::Plane { n, d } => Ok((n, d)),
            _ => Err("lattice elements must be bounded by planes".to_string()),
        })
        .collect()
}

/// Point where three planes `n·x = d` meet.
fn meet(a: (Vec3, f64), b: (Vec3, f64), c: (Vec3, f64)) -> Result<Vec3, String> {
    let det = dot(a.0, cross(b.0, c.0));
    if det.abs() < 1e-12 {
        return Err("degenerate lattice element".into());
    }
    let v = add(
        add(scale(cross(b.0, c.0), a.1), scale(cross(c.0, a.0), b.1)),
        scale(cross(a.0, b.0), c.1),
    );
    Ok(scale(v, 1.0 / det))
}

fn dual_of(basis: &[Vec3; 3]) -> Result<[Vec3; 3], String> {
    let det = dot(basis[0], cross(basis[1], basis[2]));
    if det.abs() < 1e-12 {
        return Err("degenerate lattice element".into());
    }
    Ok([
        scale(cross(basis[1], basis[2]), 1.0 / det),
        scale(cross(basis[2], basis[0]), 1.0 / det),
        scale(cross(basis[0], basis[1]), 1.0 / det),
    ])
}

/// Mid-plane of two opposite outward planes: `n₁·x = (d₁ - d₂)/2`.
fn mid(p: (Vec3, f64), q: (Vec3, f64)) -> Result<(Vec3, f64), String> {
    if dot(p.0, q.0) > -1.0 + 1e-9 {
        return Err("lattice element sides must come in parallel opposite pairs".into());
    }
    Ok((p.0, 0.5 * (p.1 - q.1)))
}

/// Build a lattice from its element program. MCNP order: for `LAT=1` the
/// surfaces bound `[1,0,0]`, `[-1,0,0]`, `[0,1,0]`, `[0,-1,0]`, `[0,0,1]`,
/// `[0,0,-1]`; for `LAT=2`, `[1,0,0]`, `[-1,0,0]`, `[0,1,0]`, `[0,-1,0]`,
/// `[-1,1,0]`, `[1,-1,0]`, then `[0,0,1]`, `[0,0,-1]`. The last pair may be
/// missing (a two-dimensional lattice).
pub(crate) fn build(
    lat: u8,
    code: &[u32],
    shapes: &[Shape],
    fill: LatticeFill,
) -> Result<Lattice, String> {
    let planes = outward_planes(code, shapes)?;
    let face_surfaces: Vec<u32> = expr::leaves(code).map(|(s, _)| s).collect();
    let (sides, dims) = match (lat, planes.len()) {
        (1, 6) => (4, 3),
        (1, 4) => (4, 2),
        (2, 8) => (6, 3),
        (2, 6) => (6, 2),
        (_, n) => {
            return Err(format!(
                "LAT={lat} element with {n} surfaces is not supported"
            ))
        }
    };
    let m0 = mid(planes[0], planes[1])?;
    let m1 = mid(planes[2], planes[3])?;
    let axis = {
        let e = cross(planes[0].0, planes[2].0);
        scale(e, 1.0 / norm(e))
    };
    let (m2, axial) = if dims == 3 {
        let (top, bottom) = (planes[sides], planes[sides + 1]);
        (mid(top, bottom)?, Some((top, bottom)))
    } else {
        ((axis, 0.0), None)
    };
    let center = meet(m0, m1, m2)?;
    let step = |p: (Vec3, f64)| scale(p.0, 2.0 * (p.1 - dot(p.0, center)));
    let a = step(planes[0]);
    let b = step(planes[2]);
    let c = match axial {
        Some((top, bottom)) => {
            // Along the sides (the hexagon or rectangle axis), across the bases.
            let e = if dot(axis, top.0) < 0.0 {
                scale(axis, -1.0)
            } else {
                axis
            };
            scale(e, (top.1 + bottom.1) / dot(e, top.0))
        }
        None => axis,
    };
    let (hexagonal, center, basis, bounded) = if lat == 1 && dims == 3 {
        // A hexahedral element need not have its third translation along the
        // side axis: it is the edge vector between the two bases.
        let corner = meet(planes[1], planes[3], planes[5])?;
        let next = |k: usize| -> Result<Vec3, String> {
            let mut ps = [planes[1], planes[3], planes[5]];
            ps[k] = planes[2 * k];
            Ok(sub(meet(ps[0], ps[1], ps[2])?, corner))
        };
        let basis = [next(0)?, next(1)?, next(2)?];
        let center = add(corner, scale(add(add(basis[0], basis[1]), basis[2]), 0.5));
        (false, center, basis, [true, true, true])
    } else if lat == 1 {
        // Parallelogram prism: translations are its edges.
        let corner = meet(planes[1], planes[3], (axis, 0.0))?;
        let ea = sub(meet(planes[0], planes[3], (axis, 0.0))?, corner);
        let eb = sub(meet(planes[1], planes[2], (axis, 0.0))?, corner);
        let center = add(corner, scale(add(ea, eb), 0.5));
        (false, center, [ea, eb, axis], [true, true, false])
    } else {
        (true, center, [a, b, c], [true, true, dims == 3])
    };
    let dual = dual_of(&basis)?;
    let faces = tiling_faces(hexagonal, center, &basis, &dual, &planes);
    Ok(Lattice {
        hexagonal,
        center,
        dual,
        basis,
        bounded,
        sides: faces[..sides].to_vec(),
        faces,
        face_surfaces,
        fill,
    })
}

/// The element's faces as planes that tile space exactly, in the order of
/// `planes` (the input's, MCNP order). Element `[0,0,0]` translated by the
/// basis must leave no gap or overlap, and its faces must be where
/// [`Lattice::index`] changes element, or a line walked face to face (as
/// tracing does) and a point located on it can disagree near an edge. The
/// input's planes do neither exactly: their normals are rounded (a hexagon's
/// third pair of sides is not quite at 60° to the others). So each face is
/// rebuilt from the basis: across a translation `v` of a hexagon's sides,
/// the plane halfway to the neighbour's centre (`n = v/|v|`); across a
/// translation the index takes from the dual basis (hexahedral elements, the
/// bases of a hexagonal prism), the plane where that index changes. The
/// faces move by the rounding of the input, far below the tolerance.
fn tiling_faces(
    hexagonal: bool,
    center: Vec3,
    basis: &[Vec3; 3],
    dual: &[Vec3; 3],
    planes: &[(Vec3, f64)],
) -> Vec<(Vec3, f64)> {
    let halfway = |v: Vec3| {
        let n = scale(v, 1.0 / norm(v));
        (n, dot(n, center) + 0.5 * norm(v))
    };
    let across_dual = |k: usize, sign: f64| {
        let len = norm(dual[k]);
        let n = scale(dual[k], sign / len);
        (n, dot(n, center) + 0.5 / len)
    };
    let neg = |v: Vec3| scale(v, -1.0);
    let mut faces = Vec::with_capacity(planes.len());
    if hexagonal {
        let (a, b) = (basis[0], basis[1]);
        let third = sub(b, a);
        for v in [a, neg(a), b, neg(b), third, neg(third)] {
            faces.push(halfway(v));
        }
        if planes.len() == 8 {
            faces.push(across_dual(2, 1.0));
            faces.push(across_dual(2, -1.0));
        }
    } else {
        for k in 0..planes.len() / 2 {
            faces.push(across_dual(k, 1.0));
            faces.push(across_dual(k, -1.0));
        }
    }
    faces
}
