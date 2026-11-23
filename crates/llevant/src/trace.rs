//! Batch line tracing through universes.
//!
//! For one line in one universe (a "level"):
//!
//! 1. Candidates: the cells whose boxes the line crosses, each with the range
//!    of `t` over which it may hold the line (octree leaves or BVH; every
//!    cell beyond the unit's region, as for points).
//! 2. Runs: each candidate's surfaces are intersected with the line (once
//!    per surface and line), and the cell is evaluated, with the same
//!    function as point classification, at the midpoint between consecutive
//!    crossings. Consecutive intervals with the same value form a run.
//! 3. Pieces: run ends closer than the tolerance are snapped together, so
//!    coincident surfaces leave no slivers, then a sweep gives each stretch
//!    the status point classification would: inside one cell, overlap,
//!    on-surface, unsupported or gap.
//! 4. Pieces inside filled cells are traced again in the filling universe
//!    (lattices element by element). Transforms are rigid, so `t` is the
//!    same at every level.

use std::mem;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::bbox::Aabb;
use crate::classify::{decode, Depth, Label, Status, MAX_DEPTH};
use crate::error::{Error, Result};
use crate::expr::{self, Tri};
use crate::hit::Visibility;
use crate::intern;
use crate::lattice::{Element, Lattice};
use crate::model::Fill;
use crate::surface::Side;
use crate::vec3::{add, dot, norm, scale, Vec3};
use crate::work::Work;
use crate::Model;

/// A line `origin + t·dir` for `t` in `[t_min, t_max]`. `dir` must be a unit
/// vector and the range finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub origin: Vec3,
    pub dir: Vec3,
    pub t_min: f64,
    pub t_max: f64,
}

impl Line {
    /// The line `origin + t·dir`, `t` in `[t_min, t_max]`; `dir` is
    /// normalised (so `t` is in cm).
    pub fn new(origin: Vec3, dir: Vec3, t_min: f64, t_max: f64) -> Line {
        let n = norm(dir);
        Line {
            origin,
            dir: scale(dir, 1.0 / n),
            t_min,
            t_max,
        }
    }

    /// The segment from `a` to `b` (`t` from 0 to their distance).
    pub fn between(a: Vec3, b: Vec3) -> Line {
        let d = crate::vec3::sub(b, a);
        Line::new(a, d, 0.0, norm(d))
    }

    /// The point at `t`.
    pub fn at(&self, t: f64) -> Vec3 {
        add(self.origin, scale(self.dir, t))
    }
}

/// `surface_in` / `surface_out` of a segment end that is not on a surface
/// (the ends of the line).
pub const NO_SURFACE: u32 = u32::MAX;

/// Segments of a batch of lines, in ragged form: line `i` has segments
/// `offsets[i] .. offsets[i + 1]`, in order along the line, covering
/// `[t_min, t_max]` without gaps (uncovered stretches are segments with
/// status [`Status::Gap`]).
#[derive(Clone, Debug, Default)]
pub struct Tracing {
    pub offsets: Vec<usize>,
    pub t0: Vec<f64>,
    pub t1: Vec<f64>,
    /// Label id of each segment, into `labels`, numbered by first occurrence.
    pub ids: Vec<u32>,
    /// Compiled surface (index of a compiled surface; [`Model::surface_id`] gives its number) at each end of each
    /// segment, or [`NO_SURFACE`].
    pub surface_in: Vec<u32>,
    pub surface_out: Vec<u32>,
    pub labels: Vec<Label>,
}

/// One segment, for [`Model::trace_line`].
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub label: Label,
    pub surface_in: Option<u32>,
    pub surface_out: Option<u32>,
}

impl Tracing {
    pub fn line_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Segment indices of line `i`.
    pub fn segments(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i]..self.offsets[i + 1]
    }

    pub fn segment(&self, k: usize) -> Segment {
        let s = |x: u32| (x != NO_SURFACE).then_some(x);
        Segment {
            t0: self.t0[k],
            t1: self.t1[k],
            label: self.labels[self.ids[k] as usize].clone(),
            surface_in: s(self.surface_in[k]),
            surface_out: s(self.surface_out[k]),
        }
    }
}

/// Lines per parallel task, at most.
const LINE_CHUNK: usize = 64;

/// Lines per parallel task for a batch of `n`: small batches (an image's
/// rows) still give every thread several tasks. Results do not depend on it.
pub(crate) fn chunk_len(n: usize) -> usize {
    (n / (8 * rayon::current_num_threads())).clamp(1, LINE_CHUNK)
}

/// Length of the first window `first_hit` searches; each next window is
/// twice as long.
const HIT_WINDOW: f64 = 16.0;

/// How far (cm) each span of a first-hit search with a skip reaches past
/// where something visible may start or end, so that a visible cell whose
/// surface lies on a span's end is still entered through it.
const SPAN_PAD: f64 = 1e-3;

/// Smallest step the lattice walk takes past an element boundary.
const LATTICE_STEP: f64 = 1e-9;

/// Inside the tracer a segment end is a "bound": the compiled surface in the
/// low bits and, above `DEPTH_SHIFT`, the depth of the universe whose frame
/// it is in (needed for normals). `NO_SURFACE` is its own bound.
const DEPTH_SHIFT: u32 = 26;

/// Most compiled surfaces a model can have (bounds keep their depth above
/// `DEPTH_SHIFT`).
pub(crate) const MAX_SURFACES: usize = 1 << DEPTH_SHIFT;

#[inline]
fn bound(surface: u32, depth: usize) -> u32 {
    surface | (depth as u32) << DEPTH_SHIFT
}

#[inline]
pub(crate) fn surface_of(b: u32) -> u32 {
    if b == NO_SURFACE {
        b
    } else {
        b & ((1 << DEPTH_SHIFT) - 1)
    }
}

#[inline]
fn depth_of(b: u32) -> usize {
    (b >> DEPTH_SHIFT) as usize
}

/// Columns of the identity: a level's directions in global coordinates.
pub(crate) const IDENTITY: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `m · rotᵀ`: the local → global rotation of a universe entered through a
/// transform whose rotation is `rot` (parent → local), from its parent's `m`.
fn child_rotation(m: &[Vec3; 3], rot: &[Vec3; 3]) -> [Vec3; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| m[i][k] * rot[j][k]).sum();
        }
    }
    out
}

/// On-surface segments up to this long (cm) are slivers that
/// [`reference::traces_agree`](crate::reference::traces_agree) passes over.
const SLIVER: f64 = 1e-6;

impl Model {
    /// Whether two segments of `line` agree as fast and brute-force tracing
    /// must: the same label, ends within the tolerance (plus `slack`), and
    /// at each end the same surface, a copy of it (TRCL), an identical one,
    /// or another one that also passes within the tolerance of that end
    /// (crossings are only collected from surfaces that can bound a cell
    /// there, so of nearly coincident surfaces either may be named).
    fn segment_agrees(&self, line: &Line, a: &Segment, b: &Segment, slack: [f64; 2]) -> bool {
        let tol = 2.0 * self.tolerance;
        let close = |x: f64, y: f64, s: f64| (x - y).abs() <= tol + s;
        let same = |x: Option<u32>, y: Option<u32>, t: f64, s: f64| match (x, y) {
            (Some(x), Some(y)) => {
                // A surface is in the frame of its universe: within the
                // tolerance of the end in the frame of some level of the path.
                let frames = self.path_frames(&a.label, line.at(t));
                let near = |k: u32| {
                    frames
                        .iter()
                        .any(|&q| self.shapes[k as usize].side(q, 2.0 * tol + s) == Side::On)
                };
                self.surface_ids[x as usize] == self.surface_ids[y as usize]
                    || self.shapes[x as usize] == self.shapes[y as usize]
                    || (near(x) && near(y))
            }
            (x, y) => x == y,
        };
        a.label == b.label
            && close(a.t0, b.t0, slack[0])
            && close(a.t1, b.t1, slack[1])
            && same(a.surface_in, b.surface_in, a.t0, slack[0])
            && same(a.surface_out, b.surface_out, a.t1, slack[1])
    }

    /// Whether two tracings of `line` (fast and brute force, say) agree:
    /// segment by segment as [`Model::segment_agrees`] says, once on-surface
    /// slivers up to 1e-6 cm are passed over in both. Segments are decided at
    /// midpoints between crossings, and brute force intersects surfaces that
    /// cannot bound a cell where they cross the line; such a crossing within
    /// the tolerance of a cell's internal surface shows a sliver of the band
    /// around it that the fast path need not show. Gaps and overlaps (wider
    /// than the tolerance) are the same either way.
    pub(crate) fn traces_agree(&self, line: &Line, a: &[Segment], b: &[Segment]) -> bool {
        let (a, b) = (without_slivers(a), without_slivers(b));
        a.len() == b.len()
            && a.iter().zip(&b).all(|((x, sx), (y, sy))| {
                let s = [sx[0].max(sy[0]), sx[1].max(sy[1])];
                self.segment_agrees(line, x, y, s)
            })
    }
}

impl Model {
    /// The maps from the global frame to that of each level of `label`'s
    /// path, as [`Model::path_frames`] takes `p` into them.
    pub(crate) fn path_maps(&self, label: &Label) -> Vec<crate::transform::Affine> {
        use crate::transform::Affine;
        let mut g = Affine::IDENTITY;
        let mut out = vec![g];
        for step in &label.path {
            let t = match (&self.cells[step.cell as usize].fill, step.lattice) {
                (Some(Fill::Universe(t)), _) => t,
                (Some(Fill::Lattice(l)), Some(ix)) => {
                    let lat = &self.lattices[*l as usize];
                    match lat.element(ix) {
                        Element::Fill(t) => {
                            let mut shift = [0.0; 3];
                            for i in 0..3 {
                                for (k, s) in shift.iter_mut().enumerate() {
                                    *s -= lat.basis[i][k] * ix[i] as f64;
                                }
                            }
                            g = Affine::translation(shift).compose(&g);
                            t
                        }
                        _ => break,
                    }
                }
                _ => break,
            };
            if let Some(a) = t.transform() {
                g = a.compose(&g);
            }
            out.push(g);
        }
        out
    }

    /// `p` (global) in the frame of each level of `label`'s path.
    pub(crate) fn path_frames(&self, label: &Label, p: Vec3) -> Vec<Vec3> {
        let mut out = vec![p];
        let mut q = p;
        for step in &label.path {
            let t = match (&self.cells[step.cell as usize].fill, step.lattice) {
                (Some(Fill::Universe(t)), _) => t,
                (Some(Fill::Lattice(l)), Some(ix)) => {
                    let lat = &self.lattices[*l as usize];
                    match lat.element(ix) {
                        Element::Fill(t) => {
                            q = lat.to_element(q, ix);
                            t
                        }
                        _ => break,
                    }
                }
                _ => break,
            };
            q = t.apply(q);
            out.push(q);
        }
        out
    }
}

/// `segs` with on-surface slivers dropped: each with the slack at its ends,
/// the length of a sliver dropped next to it. Neighbours with the same label
/// that a sliver separated are joined.
fn without_slivers(segs: &[Segment]) -> Vec<(Segment, [f64; 2])> {
    let mut out: Vec<(Segment, [f64; 2])> = Vec::new();
    let mut pending = 0.0;
    for s in segs {
        let len = s.t1 - s.t0;
        if s.label.status == Status::OnSurface && len <= SLIVER {
            pending += len;
            if let Some(last) = out.last_mut() {
                last.1[1] = last.1[1].max(pending);
            }
            continue;
        }
        if pending > 0.0 {
            if let Some(last) = out.last_mut() {
                if last.0.label == s.label {
                    last.0.t1 = s.t1;
                    last.0.surface_out = s.surface_out;
                    last.1[1] = 0.0;
                    pending = 0.0;
                    continue;
                }
            }
        }
        out.push((s.clone(), [pending, 0.0]));
        pending = 0.0;
    }
    out
}

impl Model {
    /// [`reference::traces_agree`](crate::reference::traces_agree) for one segment pair (no slivers passed
    /// over).
    pub(crate) fn segments_agree(&self, line: &Line, a: &Segment, b: &Segment) -> bool {
        self.segment_agrees(line, a, b, [0.0; 2])
    }

    /// Trace lines given in the global frame, down to `depth`: with
    /// [`Depth::Level`] labels are the full ones cut to `level + 1` steps.
    /// MCNP's plotter draws level `n` (`LEVEL n`) from these, showing only
    /// the segments whose path is `n + 1` steps long. Results are identical
    /// for any number of threads. Each line needs a finite, ordered range
    /// and a unit direction ([`Error::Line`] otherwise).
    pub fn trace(&self, lines: &[Line], depth: Depth) -> Result<Tracing> {
        check_lines(lines)?;
        Ok(self.trace_unchecked(lines, false, depth.level()))
    }

    /// Trace one line, every level (convenience for tests and small jobs).
    pub fn trace_line(&self, line: &Line) -> Result<Vec<Segment>> {
        let t = self.trace(std::slice::from_ref(line), Depth::All)?;
        Ok(t.segments(0).map(|k| t.segment(k)).collect())
    }

    /// [`Model::trace`] of lines known to be valid ([`check_lines`]).
    pub(crate) fn trace_unchecked(
        &self,
        lines: &[Line],
        brute: bool,
        level: Option<usize>,
    ) -> Tracing {
        debug_assert!(check_lines(lines).is_ok());
        // The compiler refuses this many surfaces.
        debug_assert!(self.shapes.len() < MAX_SURFACES);
        // Labels are interned per thread, not per chunk: lines rarely repeat
        // a label within a chunk (paths five lattices deep), so per-chunk
        // tables held about every segment's key again (core.i, 200 000
        // lines: 30 M keys for 167 k labels). A chunk holds its thread's
        // table while it is traced (nothing in `Tracer::line` runs on the
        // pool, so no other chunk can wait on it on this thread) and keeps
        // the span of its queries there.
        let threads = rayon::current_num_threads();
        let tables: Vec<Mutex<intern::Chunk>> = (0..threads).map(|_| Mutex::default()).collect();
        let parts: Vec<(Span, Part)> = lines
            .par_chunks(chunk_len(lines.len()))
            .map_init(
                || {
                    let mut tr = Tracer::new(self, brute);
                    tr.set_level(level);
                    tr
                },
                |tr, chunk| {
                    let t = rayon::current_thread_index().unwrap_or(0) % threads;
                    let mut table = tables[t].lock().unwrap_or_else(|e| e.into_inner());
                    let mut part = Part {
                        chunk: mem::take(&mut *table),
                        ..Part::default()
                    };
                    let start = part.chunk.ids.len();
                    for l in chunk {
                        tr.line(l, &mut part);
                    }
                    let span = (t, start, part.chunk.ids.len());
                    *table = mem::take(&mut part.chunk);
                    (span, part)
                },
            )
            .collect();
        let tables: Vec<intern::Chunk> = tables
            .into_iter()
            .map(|t| t.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let (spans, parts): (Vec<Span>, Vec<Part>) = parts.into_iter().unzip();
        let mut out = segments(&parts, lines.len());
        drop(parts);
        // Ids by first occurrence over the tables in turn, then numbered
        // again by first occurrence along the lines, as a single table
        // would number them (whatever thread traced what).
        let (canon, keys) = intern::merge(&tables);
        let mut base = Vec::with_capacity(tables.len());
        let mut at = 0;
        for t in &tables {
            base.push(at);
            at += t.ids.len();
        }
        let mut number = vec![u32::MAX; keys.len()];
        let mut order: Vec<u32> = Vec::with_capacity(keys.len());
        out.ids = Vec::with_capacity(canon.len());
        for &(t, s, e) in &spans {
            for &c in &canon[base[t] + s..base[t] + e] {
                let n = &mut number[c as usize];
                if *n == u32::MAX {
                    *n = order.len() as u32;
                    order.push(c);
                }
                out.ids.push(*n);
            }
        }
        out.labels = order
            .par_iter()
            .map(|&c| decode(keys[c as usize]))
            .collect();
        out
    }

    /// [`Depth::Level`](crate::Depth::Level) of a few valid lines on this thread: no
    /// tasks to hand out and one tracer, for single picks.
    pub(crate) fn trace_few(&self, lines: &[Line], level: Option<usize>) -> Tracing {
        debug_assert!(check_lines(lines).is_ok());
        let mut tr = Tracer::new(self, false);
        tr.set_level(level);
        let mut part = Part::default();
        for l in lines {
            tr.line(l, &mut part);
        }
        assemble(vec![part], lines.len(), false)
    }

    /// Every level of valid lines, on this thread, the labels left as local
    /// keys of the part: for callers that reduce segments themselves
    /// (volumes), with no merge of labels and nothing decoded.
    pub(crate) fn trace_part(&self, lines: &[Line]) -> Part {
        debug_assert!(check_lines(lines).is_ok());
        let mut tr = Tracer::new(self, false);
        let mut part = Part::default();
        for l in lines {
            tr.line(l, &mut part);
        }
        part
    }
}

/// The segments of `parts` (chunks of lines, in order) as one tracing;
/// labels decoded on the pool when `parallel`.
fn assemble(parts: Vec<Part>, lines: usize, parallel: bool) -> Tracing {
    let mut out = segments(&parts, lines);
    let chunks: Vec<intern::Chunk> = parts.into_iter().map(|p| p.chunk).collect();
    let (ids, keys) = intern::merge(&chunks);
    out.ids = ids;
    out.labels = if parallel {
        keys.par_iter().map(|k| decode(k)).collect()
    } else {
        keys.iter().map(|k| decode(k)).collect()
    };
    out
}

/// Where a chunk's queries are in the per-thread tables: table, start, end.
type Span = (usize, usize, usize);

/// The segments of `parts` (chunks of lines, in order), without labels.
fn segments(parts: &[Part], lines: usize) -> Tracing {
    let mut offsets = Vec::with_capacity(lines + 1);
    offsets.push(0);
    for p in parts {
        for &n in &p.counts {
            offsets.push(offsets.last().unwrap() + n as usize);
        }
    }
    Tracing {
        offsets,
        t0: concat(parts, |p| &p.t0),
        t1: concat(parts, |p| &p.t1),
        surface_in: concat(parts, |p| &p.surface_in),
        surface_out: concat(parts, |p| &p.surface_out),
        ..Tracing::default()
    }
}

/// Every line has a finite, ordered range and a unit direction.
pub(crate) fn check_lines(lines: &[Line]) -> Result<()> {
    let ok = |l: &Line| {
        l.t_min.is_finite()
            && l.t_max.is_finite()
            && l.t_min <= l.t_max
            && (norm(l.dir) - 1.0).abs() < 1e-6
    };
    match lines.iter().position(|l| !ok(l)) {
        None => Ok(()),
        Some(index) => Err(Error::Line {
            index,
            line: lines[index],
        }),
    }
}

/// One array of all parts, in order, copied in parallel.
fn concat<T: Copy + Default + Send + Sync>(
    parts: &[Part],
    field: impl Fn(&Part) -> &Vec<T> + Sync,
) -> Vec<T> {
    let total = parts.iter().map(|p| field(p).len()).sum();
    let mut out = vec![T::default(); total];
    let mut slices = Vec::with_capacity(parts.len());
    let mut rest: &mut [T] = &mut out;
    for p in parts {
        let (head, tail) = rest.split_at_mut(field(p).len());
        slices.push(head);
        rest = tail;
    }
    slices
        .into_par_iter()
        .zip(parts)
        .for_each(|(dst, p)| dst.copy_from_slice(field(p)));
    out
}

/// The segments of one chunk of lines: per line its number of segments, per
/// segment its ends and its label's local id in `chunk`.
#[derive(Default)]
pub(crate) struct Part {
    pub(crate) chunk: intern::Chunk,
    pub(crate) counts: Vec<u32>,
    pub(crate) t0: Vec<f64>,
    pub(crate) t1: Vec<f64>,
    surface_in: Vec<u32>,
    surface_out: Vec<u32>,
}

/// A final segment of the current line; its key is `keys[key.0..key.1]`.
struct Seg {
    t0: f64,
    t1: f64,
    s0: u32,
    s1: u32,
    key: (usize, usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    True,
    Unknown,
    Unsupported,
}

/// A stretch where one candidate cell has one value.
#[derive(Clone, Copy)]
struct Run {
    t0: f64,
    t1: f64,
    s0: u32,
    s1: u32,
    cell: u32,
    kind: Kind,
}

/// Run ends closer than the tolerance, snapped to `rep`.
#[derive(Clone, Copy)]
struct Cluster {
    first: f64,
    rep: f64,
    surface: u32,
}

/// A stretch of the line with one status at this level; `cells` indexes
/// `Level::piece_cells` (the inside cell, or the cells involved).
#[derive(Clone, Copy)]
struct Piece {
    t0: f64,
    t1: f64,
    s0: u32,
    s1: u32,
    status: Status,
    cells: (usize, usize),
}

/// Buffers for one level of the descent.
#[derive(Default)]
struct Level {
    cands: Vec<(u32, f64, f64)>,
    tmp_cands: Vec<(u32, f64, f64)>,
    /// `(cell, octree item)` of the leaves the line crosses, sorted.
    items: Vec<(u32, u32)>,
    /// Cells found other than through octree leaves along part of the
    /// line (BVH, beyond the region, checked everywhere), sorted: their whole
    /// program counts.
    full: Vec<u32>,
    /// Crossings of each surface met on this line, `(surface, start, len,
    /// last candidate)` into `roots`; found through `Tracer::slot`. The last
    /// candidate that used an entry lets a cell skip its repeated leaves.
    cache: Vec<(u32, u32, u32, u32)>,
    roots: Vec<f64>,
    tmp: Vec<f64>,
    cross: Vec<(f64, u32)>,
    leaves: Vec<u32>,
    runs: Vec<Run>,
    ends: Vec<(f64, u32, bool)>,
    clusters: Vec<Cluster>,
    active: Vec<usize>,
    set: Vec<u32>,
    pieces: Vec<Piece>,
    piece_cells: Vec<u32>,
    reps: Vec<(f64, u32)>,
    breaks: Vec<(f64, u32)>,
    elements: Vec<(f64, f64, u32, u32, [i32; 3])>,
}

pub(crate) struct Tracer<'m> {
    m: &'m Model,
    brute: bool,
    levels: Vec<Level>,
    spans: Vec<(f64, f64)>,
    skip_raw: Vec<(f64, f64)>,
    /// Sparse-set index into a level's `cache` by surface. `slot[s]` is
    /// only trusted when that level's entry there names `s`, so the array
    /// never needs clearing and all levels can share it: an index left by
    /// another level or line at worst misses, and the crossings are
    /// computed again.
    slot: Vec<u32>,
    /// Steps of the current path, 5 words each as in point labels.
    path: Vec<u32>,
    segs: Vec<Seg>,
    keys: Vec<u32>,
    /// Per depth: the line in that universe's frame, and the rotation from
    /// that frame to the global one.
    frames: Vec<(Vec3, Vec3, [Vec3; 3])>,
    /// Depth at which filled cells are cells like any other
    /// ([`Depth::Level`]); `MAX_DEPTH` traces every level.
    max_level: usize,
    /// Set for `first_hit`: only the first visible segment is wanted.
    pub(crate) hit: Option<HitSearch<'m>>,
    /// Counters, when the work is wanted (`Model::work_lines`).
    pub(crate) work: Option<Work>,
}

/// What `first_hit` looks for and what it found.
pub(crate) struct HitSearch<'v> {
    pub vis: &'v Visibility,
    /// Where something visible may be, when `vis` shows no gaps and no
    /// problems (so that skipping the rest is exact).
    pub skip: Option<&'v crate::skip::Skip>,
    /// Per-node flags of unit 0's octree even when gaps/problems are shown:
    /// leaves known to be strictly filled by a single hidden, non-ghost,
    /// unfilled cell with no possible overlap from `always` are false.
    pub exact_nodes: Option<&'v [bool]>,
    pub found: Option<FoundHit>,
    /// The stretches in ghost cells met so far, when there are ghosts.
    pub ghosts: Option<Vec<GhostStretch>>,
    /// The veils gone through so far ([`Visibility::veil`]).
    pub veils: crate::hit::Veils,
}

/// A stretch of a line inside a ghost cell: its ends, bounds and normals
/// there (zero at an end without a surface).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GhostStretch {
    cell: u32,
    t0: f64,
    s0: u32,
    n0: Vec3,
    t1: f64,
    s1: u32,
    n1: Vec3,
}

pub(crate) struct FoundHit {
    pub t: f64,
    pub surface: u32,
    pub normal: Vec3,
    /// Range of `Tracer::keys` holding the label.
    pub key: (usize, usize),
}

#[inline]
fn at(o: Vec3, d: Vec3, t: f64) -> Vec3 {
    add(o, scale(d, t))
}

#[inline]
fn mid(a: f64, b: f64) -> f64 {
    a + 0.5 * (b - a)
}

thread_local! {
    /// The surface slots of this thread's last tracer, for the next one.
    static SLOT: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for Tracer<'_> {
    fn drop(&mut self) {
        if !self.slot.is_empty() {
            // `try_with`: nothing to keep while the thread itself ends.
            let slot = mem::take(&mut self.slot);
            let _ = SLOT.try_with(|s| *s.borrow_mut() = slot);
        }
    }
}

impl<'m> Tracer<'m> {
    pub(crate) fn new(m: &'m Model, brute: bool) -> Self {
        Tracer {
            m,
            brute,
            levels: Vec::new(),
            spans: Vec::new(),
            skip_raw: Vec::new(),
            slot: Vec::new(),
            path: Vec::new(),
            segs: Vec::new(),
            keys: Vec::new(),
            frames: Vec::new(),
            max_level: MAX_DEPTH,
            hit: None,
            work: None,
        }
    }

    /// Trace down to `level` at most ([`Depth::Level`](crate::Depth::Level)).
    pub(crate) fn set_level(&mut self, level: Option<usize>) {
        self.max_level = level.map_or(MAX_DEPTH, |k| k.min(MAX_DEPTH));
    }

    /// Whether a `first_hit` search has its answer.
    #[inline]
    fn done(&self) -> bool {
        self.hit.as_ref().is_some_and(|h| h.found.is_some())
    }

    /// Search one line for its first visible segment (`self.hit` set).
    pub(crate) fn first_hit(&mut self, l: &Line) -> Option<FoundHit> {
        self.keys.clear();
        self.path.clear();
        if let Some(h) = self.hit.as_mut() {
            h.found = None;
            h.veils = Default::default();
            if let Some(g) = h.ghosts.as_mut() {
                g.clear();
            }
        }
        // Only where something visible may be, in order (all of the line
        // without a skip).
        let mut spans = mem::take(&mut self.spans);
        spans.clear();
        if let Some(s) = self.hit.as_ref().and_then(|h| h.skip) {
            let mut raw = mem::take(&mut self.skip_raw);
            s.spans_into(
                self.m,
                0,
                l.origin,
                l.dir,
                (l.t_min, l.t_max),
                SPAN_PAD,
                &mut raw,
                &mut spans,
            );
            self.skip_raw = raw;
        } else if let (Some(flags), Some(g)) = (
            self.hit.as_ref().and_then(|h| h.exact_nodes),
            self.m.units[0].grid.as_deref(),
        ) {
            let mut raw = mem::take(&mut self.skip_raw);
            crate::skip::octree_spans_into(
                &self.m.units[0].roi,
                Some(g),
                Some(flags),
                l.origin,
                l.dir,
                (l.t_min, l.t_max),
                SPAN_PAD,
                &mut raw,
                &mut spans,
            );
            // Outside `u0.roi`, retain any segments before `roi` and after `roi`
            // so gaps/problems outside `u0.roi` are still checked by `u0.outside()`.
            let inv = Aabb::ray_inv(l.dir);
            match self.m.units[0]
                .roi
                .widen()
                .ray_range_inv(l.origin, l.dir, inv, l.t_min, l.t_max)
            {
                None => {
                    spans.clear();
                    spans.push((l.t_min, l.t_max));
                }
                Some((ra, rb)) => {
                    if ra > l.t_min {
                        let tail = (ra + SPAN_PAD).min(l.t_max);
                        if spans.first().is_some_and(|s| s.0 <= tail) {
                            spans[0].0 = l.t_min;
                        } else {
                            spans.insert(0, (l.t_min, tail));
                        }
                    }
                    if rb < l.t_max {
                        let head = (rb - SPAN_PAD).max(l.t_min);
                        if spans.last().is_some_and(|s| s.1 >= head) {
                            spans.last_mut().unwrap().1 = l.t_max;
                        } else {
                            spans.push((head, l.t_max));
                        }
                    }
                }
            }
            self.skip_raw = raw;
        } else {
            spans.push((l.t_min, l.t_max));
        }
        let span_start = spans.first().map_or(l.t_min, |s| s.0);
        for &(t0, t1) in &spans {
            // Growing windows from the start, so that a hit near it does
            // not pay for the rest of the line. Anything visible in an
            // earlier window would have been found there. A later span goes
            // on with windows as large as they would have grown to by then:
            // restarting small ones in the middle of a model costs more.
            let (mut ta, mut w) = (t0, HIT_WINDOW.max(t0 - span_start));
            loop {
                let tb = if t1 - ta <= 2.0 * w { t1 } else { ta + w };
                self.unit(
                    0, l.origin, l.dir, ta, tb, NO_SURFACE, NO_SURFACE, 0, IDENTITY,
                );
                if self.done() {
                    self.spans = spans;
                    return self.hit.as_mut().and_then(|h| h.found.take());
                }
                self.keys.clear();
                self.path.clear();
                if tb >= t1 {
                    break;
                }
                (ta, w) = (tb, 2.0 * w);
            }
        }
        self.spans = spans;
        None
    }

    pub(crate) fn keys(&self) -> &[u32] {
        &self.keys
    }

    /// The ghost boundaries the last [`Tracer::first_hit`] crossed up to
    /// `t_hit` (its hit, or `INFINITY`): the ends of ghost stretches that
    /// are on a surface (not a line's or window's end), those within the
    /// tolerance of each other counted once (at the same place, the cell
    /// entered named first, as the GPU names it). A ghost boundary at the hit
    /// itself counts: the shell is drawn over it.
    pub(crate) fn ghost(&self, t_hit: f64) -> crate::hit::Ghost {
        let mut out = crate::hit::Ghost::default();
        let Some(stretches) = self.hit.as_ref().and_then(|h| h.ghosts.as_ref()) else {
            return out;
        };
        let tol = self.m.tolerance;
        // (t, rank, cell, normal); at the same place the GPU names the
        // ghost cell entered, else the one left: ranks 0, 1.
        let mut ends: Vec<(f64, u8, u32, Vec3)> = Vec::new();
        for g in stretches {
            if g.s0 != NO_SURFACE && g.t0 <= t_hit + tol {
                ends.push((g.t0, 0, g.cell, g.n0));
            }
            if g.s1 != NO_SURFACE && g.t1 <= t_hit + tol {
                ends.push((g.t1, 1, g.cell, g.n1));
            }
        }
        ends.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut last = f64::NEG_INFINITY;
        for (t, _, cell, n) in ends {
            if t - last > tol {
                if out.count == 0 {
                    (out.t, out.cell, out.normal) = (t, cell, n);
                }
                out.count += 1;
            }
            last = t;
        }
        out
    }

    /// Note a stretch of ghost cell `c`, merged with the one before it in
    /// the same cell when they meet (at a window's end).
    fn ghost_stretch(&mut self, c: u32, t0: f64, t1: f64, s0: u32, s1: u32) {
        let n0 = self.normal(t0, s0);
        let n1 = self.normal(t1, s1);
        let Some(g) = self.hit.as_mut().and_then(|h| h.ghosts.as_mut()) else {
            return;
        };
        if let Some(prev) = g.iter_mut().rev().find(|x| x.cell == c && x.t1 == t0) {
            (prev.t1, prev.s1, prev.n1) = (t1, s1, n1);
            return;
        }
        g.push(GhostStretch {
            cell: c,
            t0,
            s0,
            n0,
            t1,
            s1,
            n1,
        });
    }

    /// The veils the last [`Tracer::first_hit`] went through (all before
    /// its hit: none is noted after it).
    pub(crate) fn veils(&self) -> crate::hit::Veils {
        self.hit.as_ref().map(|h| h.veils).unwrap_or_default()
    }

    /// Trace one line, keeping only `self.work`.
    #[cfg(feature = "bench")]
    pub(crate) fn trace_one(&mut self, l: &Line) {
        self.segs.clear();
        self.keys.clear();
        self.path.clear();
        self.unit(
            0, l.origin, l.dir, l.t_min, l.t_max, NO_SURFACE, NO_SURFACE, 0, IDENTITY,
        );
    }

    fn line(&mut self, l: &Line, part: &mut Part) {
        self.segs.clear();
        self.keys.clear();
        self.path.clear();
        self.unit(
            0, l.origin, l.dir, l.t_min, l.t_max, NO_SURFACE, NO_SURFACE, 0, IDENTITY,
        );
        part.counts.push(self.segs.len() as u32);
        for s in &self.segs {
            part.chunk.push(&self.keys[s.key.0..s.key.1]);
            part.t0.push(s.t0);
            part.t1.push(s.t1);
            part.surface_in.push(surface_of(s.s0));
            part.surface_out.push(surface_of(s.s1));
        }
    }

    /// Append a final segment, merged into the previous one when it has the
    /// same label.
    fn emit(&mut self, t0: f64, t1: f64, s0: u32, s1: u32, status: Status, cells: &[u32]) {
        if let Some(h) = &self.hit {
            if h.found.is_some() || !h.vis.shows(status) {
                return;
            }
            // Gaps and problems of cells left out.
            let holder = (self.path.len() >= 5).then(|| self.path[self.path.len() - 5]);
            if !h.vis.owns(status, holder, cells) {
                return;
            }
            // Veils are seen through, noted.
            if h.vis.veils(status) {
                let Some(h) = &mut self.hit else {
                    unreachable!()
                };
                h.veils.gap |= status == Status::Gap;
                h.veils.overlap |= status == Status::Overlap;
                return;
            }
            // Inside: only the level shown, and never in a ghost cell.
            if status == Status::Inside {
                let depth = self.path.len() / 5;
                if h.vis.level.is_some_and(|k| depth != k + 1)
                    || h.vis.is_ghost(self.path[self.path.len() - 5])
                {
                    return;
                }
            }
        }
        let start = self.keys.len();
        self.keys.push(status as u32);
        self.keys.push((self.path.len() / 5) as u32);
        self.keys.extend_from_slice(&self.path);
        self.keys.extend_from_slice(cells);
        if let Some(prev) = self.segs.last_mut() {
            if prev.t1 == t0 && self.keys[prev.key.0..prev.key.1] == self.keys[start..] {
                prev.t1 = t1;
                prev.s1 = s1;
                self.keys.truncate(start);
                return;
            }
        }
        let key = (start, self.keys.len());
        if self.hit.is_some() {
            let normal = self.normal(t0, s0);
            let Some(h) = &mut self.hit else {
                unreachable!()
            };
            h.found = Some(FoundHit {
                t: t0,
                surface: surface_of(s0),
                normal,
                key,
            });
            return;
        }
        self.segs.push(Seg {
            t0,
            t1,
            s0,
            s1,
            key,
        });
    }

    /// Unit normal of bound `b` where the line is at `t`, in the global
    /// frame and facing the line's origin (zero without a surface).
    fn normal(&self, t: f64, b: u32) -> Vec3 {
        if b == NO_SURFACE {
            return [0.0; 3];
        }
        let (o, d, rot) = self.frames[depth_of(b)];
        let g = self.m.shapes[surface_of(b) as usize].gradient(at(o, d, t));
        let n = [0, 1, 2].map(|i| dot(rot[i], g));
        let len = norm(n);
        if len == 0.0 {
            return [0.0; 3];
        }
        // Level 0 is the global frame.
        let s = if dot(n, self.frames[0].1) > 0.0 {
            -1.0
        } else {
            1.0
        } / len;
        scale(n, s)
    }

    /// Trace `[ta, tb]` of the line `o + t·d` (in the unit's frame) through
    /// unit `unit`; `sa` and `sb` are the surfaces at the two ends.
    #[allow(clippy::too_many_arguments)]
    fn unit(
        &mut self,
        unit: u32,
        o: Vec3,
        d: Vec3,
        ta: f64,
        tb: f64,
        sa: u32,
        sb: u32,
        depth: usize,
        rot: [Vec3; 3],
    ) {
        if depth >= MAX_DEPTH {
            self.emit(ta, tb, sa, sb, Status::TooDeep, &[]);
            return;
        }
        self.frames.truncate(depth);
        self.frames.push((o, d, rot));
        if self.levels.len() <= depth {
            self.levels.resize_with(depth + 1, Level::default);
        }
        let m = self.m;
        let mut lv = mem::take(&mut self.levels[depth]);
        if self.slot.is_empty() {
            // Slots are checked against the level caches before use (stale
            // ones only miss), so this thread's last array of the right
            // length will do. A fresh one per tracer costs more than a
            // small batch's tracing: allocators that reuse memory (mimalloc)
            // clear all of it, not only the pages used.
            let old = SLOT.with_borrow_mut(mem::take);
            self.slot = if old.len() == m.shapes.len() {
                old
            } else {
                vec![0; m.shapes.len()]
            };
        }
        let mut slot = mem::take(&mut self.slot);
        let [crossed, midpoints, evals] =
            self.pieces(&mut lv, &mut slot, unit, o, d, ta, tb, sa, sb, depth);
        self.slot = slot;
        if let Some(w) = self.work.as_mut() {
            let l = w.level(depth);
            l.visits += 1;
            l.candidates += lv.cands.len() as u64;
            l.surfaces_crossed += crossed;
            l.midpoints += midpoints;
            l.surface_evals += evals;
        }
        for i in 0..lv.pieces.len() {
            if self.done() {
                break;
            }
            let p = lv.pieces[i];
            let cells = &lv.piece_cells[p.cells.0..p.cells.1];
            if p.status != Status::Inside {
                self.emit(p.t0, p.t1, p.s0, p.s1, p.status, cells);
                continue;
            }
            let c = cells[0];
            // A hidden cell hides everything in it: no descent.
            if let Some(h) = &self.hit {
                if !h.vis.cells[c as usize] {
                    continue;
                }
                if h.ghosts.is_some() && h.vis.is_ghost(c) {
                    self.ghost_stretch(c, p.t0, p.t1, p.s0, p.s1);
                }
            }
            self.path.extend_from_slice(&[c, 0, 0, 0, 0]);
            // At the deepest level traced, a filled cell is a cell (a
            // lattice cell still has its elements).
            let last = depth >= self.max_level;
            match &m.cells[c as usize].fill {
                None => self.emit(p.t0, p.t1, p.s0, p.s1, Status::Inside, &[]),
                Some(Fill::Universe(_)) if last => {
                    self.emit(p.t0, p.t1, p.s0, p.s1, Status::Inside, &[])
                }
                Some(Fill::Universe(t)) => {
                    let (o2, d2, r2) = match &t.transform {
                        Some(tr) => (tr.apply(o), tr.apply_dir(d), child_rotation(&rot, &tr.rot)),
                        None => (o, d, rot),
                    };
                    self.unit(t.unit, o2, d2, p.t0, p.t1, p.s0, p.s1, depth + 1, r2);
                }
                Some(Fill::Lattice(l)) => {
                    let lat = &m.lattices[*l as usize];
                    let mut els = mem::take(&mut lv.elements);
                    lattice_pieces(
                        lat,
                        o,
                        d,
                        p.t0,
                        p.t1,
                        p.s0,
                        p.s1,
                        depth,
                        m.tolerance,
                        &mut lv.breaks,
                        &mut els,
                    );
                    let n = self.path.len();
                    for &(a, b, sa2, sb2, ix) in &els {
                        if self.done() {
                            break;
                        }
                        self.path[n - 4..].copy_from_slice(&[
                            1,
                            ix[0] as u32,
                            ix[1] as u32,
                            ix[2] as u32,
                        ]);
                        match lat.element(ix) {
                            Element::Fill(_) if last => {
                                self.emit(a, b, sa2, sb2, Status::Inside, &[])
                            }
                            Element::Fill(t) => {
                                let q = lat.to_element(o, ix);
                                let (o2, d2, r2) = match &t.transform {
                                    Some(tr) => (
                                        tr.apply(q),
                                        tr.apply_dir(d),
                                        child_rotation(&rot, &tr.rot),
                                    ),
                                    None => (q, d, rot),
                                };
                                self.unit(t.unit, o2, d2, a, b, sa2, sb2, depth + 1, r2);
                            }
                            Element::Own => self.emit(a, b, sa2, sb2, Status::Inside, &[]),
                            Element::Missing => self.emit(a, b, sa2, sb2, Status::Gap, &[]),
                        }
                    }
                    lv.elements = els;
                }
            }
            self.path.truncate(self.path.len() - 5);
        }
        self.levels[depth] = lv;
    }

    /// Fill `lv.pieces` for `[ta, tb]` in unit `unit`, in stages: the
    /// candidate cells ([`Tracer::candidates`]), the runs where each holds
    /// the line ([`Tracer::cell_runs`]), then the runs snapped to the ends
    /// ([`snap`]) and swept into pieces ([`sweep`]). Returns the work done
    /// (surfaces crossed, midpoints, surface evaluations).
    #[allow(clippy::too_many_arguments)]
    fn pieces(
        &self,
        lv: &mut Level,
        slot: &mut [u32],
        unit: u32,
        o: Vec3,
        d: Vec3,
        ta: f64,
        tb: f64,
        sa: u32,
        sb: u32,
        depth: usize,
    ) -> [u64; 3] {
        self.candidates(lv, unit, o, d, ta, tb);
        // Counted locally and added at the end, to keep `self` free.
        let mut work = [0u64; 3];
        lv.cache.clear();
        lv.roots.clear();
        lv.runs.clear();
        for ci in 0..lv.cands.len() {
            self.cell_runs(lv, slot, unit, ci, o, d, ta, tb, depth, &mut work);
        }
        snap(lv, ta, tb, sa, sb, self.m.tolerance);
        sweep(lv);
        work
    }

    /// Push to `lv.runs` the runs of candidate `ci` over its range: the
    /// stretches between the surfaces that can bound it
    /// ([`Tracer::cell_crossings`]) where its program is true or unknown at
    /// the midpoint.
    #[allow(clippy::too_many_arguments)]
    fn cell_runs(
        &self,
        lv: &mut Level,
        slot: &mut [u32],
        unit: u32,
        ci: usize,
        o: Vec3,
        d: Vec3,
        ta: f64,
        tb: f64,
        depth: usize,
        [crossed, midpoints, evals]: &mut [u64; 3],
    ) {
        let m = self.m;
        let u = &m.units[unit as usize];
        let counting = self.work.is_some();
        let (c, lo, hi) = lv.cands[ci];
        let cell = &m.cells[c as usize];
        if cell.unsupported.is_some() || m.is_lattice(c) {
            let kind = if cell.unsupported.is_some() {
                Kind::Unsupported
            } else {
                Kind::True
            };
            lv.runs.push(Run {
                t0: lo,
                t1: hi,
                s0: NO_SURFACE,
                s1: NO_SURFACE,
                cell: c,
                kind,
            });
            return;
        }
        // With octree items, only the surfaces of the programs
        // specialized to the leaves the line crosses can bound it.
        let items = self.cell_items(lv, u.grid.is_some(), c);
        self.cell_crossings(lv, slot, unit, ci, items, o, d, ta, tb, depth, crossed);
        lv.cross.push((hi, NO_SURFACE));
        let (mut prev, mut sprev) = (lo, NO_SURFACE);
        let mut open: Option<Run> = None;
        for k in 0..lv.cross.len() {
            let (t, s) = lv.cross[k];
            if t > prev {
                let q = at(o, d, mid(prev, t));
                let boxes = (!self.brute && u.roi.contains(q)).then(|| m.node_boxes_of(c));
                // The program for `q`: specialized to the octree leaf
                // holding it (`None`: true over the whole leaf; a cell
                // not in the leaf is false over all of it).
                let code = match (items, &u.grid) {
                    (Some(_), Some(g)) => match g.leaf_span(q) {
                        Some((first, n)) => {
                            match g.items[first..first + n].iter().position(|&x| x == c) {
                                Some(i) => Ok(g.prog(first + i, m.code(c))),
                                None => Err(Tri::False),
                            }
                        }
                        None => Ok(Some(m.code(c))),
                    },
                    _ => Ok(Some(m.code(c))),
                };
                *midpoints += counting as u64;
                let v = match code {
                    Err(v) => v,
                    Ok(None) => Tri::True,
                    Ok(Some(code)) if counting => m.eval_code_counted(code, q, boxes, evals),
                    Ok(Some(code)) => m.eval_code(code, q, boxes),
                };
                let kind = match v {
                    Tri::True => Some(Kind::True),
                    Tri::Unknown => Some(Kind::Unknown),
                    Tri::False => None,
                };
                match (&mut open, kind) {
                    (Some(r), Some(k)) if r.kind == k => {
                        r.t1 = t;
                        r.s1 = s;
                    }
                    (_, kind) => {
                        if let Some(r) = open.take() {
                            lv.runs.push(r);
                        }
                        open = kind.map(|kind| Run {
                            t0: prev,
                            t1: t,
                            s0: sprev,
                            s1: s,
                            cell: c,
                            kind,
                        });
                    }
                }
            }
            prev = t;
            sprev = s;
        }
        if let Some(r) = open {
            lv.runs.push(r);
        }
    }

    /// Fill `lv.cross` with the crossings, within candidate `ci`'s range, of
    /// the surfaces that can bound it, sorted along the line. Each surface's
    /// crossings over `[ta, tb]` are found once per line and level, and
    /// kept in `lv.cache` / `lv.roots` for the other candidates.
    #[allow(clippy::too_many_arguments)]
    fn cell_crossings(
        &self,
        lv: &mut Level,
        slot: &mut [u32],
        unit: u32,
        ci: usize,
        items: Option<(usize, usize)>,
        o: Vec3,
        d: Vec3,
        ta: f64,
        tb: f64,
        depth: usize,
        crossed: &mut u64,
    ) {
        let m = self.m;
        let u = &m.units[unit as usize];
        let (c, lo, hi) = lv.cands[ci];
        lv.cross.clear();
        // Node boxes are exact inside the unit's region: where the cell's
        // stretch lies in it, nodes it misses are constant along it and
        // their leaves cannot matter.
        let (pa, pb) = (at(o, d, lo), at(o, d, hi));
        let inv = Aabb::ray_inv(d);
        let prune = !self.brute && u.roi.contains(pa) && u.roi.contains(pb);
        let mut leaves = std::mem::take(&mut lv.leaves);
        leaves.clear();
        let add = |code: &[u32], leaves: &mut Vec<u32>| {
            if prune {
                expr::leaves_where(
                    code,
                    m.node_boxes_of(c),
                    &mut |b| b.ray_range_inv(o, d, inv, lo, hi).is_some(),
                    &mut |s| leaves.push(s),
                );
            } else {
                leaves.extend(expr::leaves(code).map(|(s, _)| s));
            }
        };
        match (items, &u.grid) {
            (Some((a, b)), Some(g)) => {
                for &(_, k) in &lv.items[a..b] {
                    if let Some(code) = g.prog(k as usize, m.code(c)) {
                        add(code, &mut leaves);
                    }
                }
            }
            _ => add(m.code(c), &mut leaves),
        }
        for &s in &leaves {
            let (start, len) = match lv.cache.get_mut(slot[s as usize] as usize) {
                // A surface met again in the same cell adds nothing.
                Some(e) if e.0 == s && e.3 == ci as u32 => continue,
                Some(e) if e.0 == s => {
                    e.3 = ci as u32;
                    (e.1, e.2)
                }
                _ => {
                    *crossed += 1;
                    lv.tmp.clear();
                    m.shapes[s as usize].crossings(o, d, ta, tb, &mut lv.tmp);
                    match lv.tmp.len() {
                        0 | 1 => {}
                        2 => {
                            if lv.tmp[0].total_cmp(&lv.tmp[1]).is_gt() {
                                lv.tmp.swap(0, 1);
                            }
                            if lv.tmp[0] == lv.tmp[1] {
                                lv.tmp.pop();
                            }
                        }
                        _ => {
                            lv.tmp.sort_unstable_by(f64::total_cmp);
                            lv.tmp.dedup();
                        }
                    }
                    let r = (lv.roots.len() as u32, lv.tmp.len() as u32);
                    lv.roots.extend_from_slice(&lv.tmp);
                    slot[s as usize] = lv.cache.len() as u32;
                    lv.cache.push((s, r.0, r.1, ci as u32));
                    r
                }
            };
            let roots = &lv.roots[start as usize..(start + len) as usize];
            let first = roots.partition_point(|&t| t <= lo);
            for &t in &roots[first..] {
                if t >= hi {
                    break;
                }
                lv.cross.push((t, bound(s, depth)));
            }
        }
        lv.leaves = leaves;
        // Coincident crossings: the lowest surface, whatever the leaf order.
        lv.cross.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(surface_of(a.1).cmp(&surface_of(b.1)))
        });
    }

    /// Candidate cells of `unit` along `[ta, tb]` with the range of `t` over
    /// which each may hold the line, one entry per cell.
    fn candidates(&self, lv: &mut Level, unit: u32, o: Vec3, d: Vec3, ta: f64, tb: f64) {
        let m = self.m;
        let u = &m.units[unit as usize];
        let inv = Aabb::ray_inv(d);
        let cands = &mut lv.cands;
        cands.clear();
        lv.items.clear();
        lv.full.clear();
        let all = |a: f64, b: f64, cands: &mut Vec<(u32, f64, f64)>| {
            cands.extend(u.cells.iter().map(|&c| (c, a, b)));
        };
        // Beyond the region the boxes were built for, every cell, or beyond
        // the problem box only its `IMP:N=0` cells (brute force too).
        let outside = |a: f64, b: f64, cands: &mut Vec<(u32, f64, f64)>| {
            cands.extend(u.outside().iter().map(|&c| (c, a, b)));
        };
        let Some((ra, rb)) = u.roi.ray_range_inv(o, d, inv, ta, tb) else {
            outside(ta, tb, cands);
            return;
        };
        if ra > ta {
            outside(ta, ra, cands);
        }
        if rb < tb {
            outside(rb, tb, cands);
        }
        if self.brute {
            all(ra, rb, cands);
            hull_per_cell(cands);
            return;
        }
        let bvh = |a: f64, b: f64, cands: &mut Vec<(u32, f64, f64)>| {
            u.bvh.query_ray(o, d, a, b, |c, x, y| cands.push((c, x, y)));
        };
        match u
            .grid
            .as_ref()
            .and_then(|g| g.root.ray_range_inv(o, d, inv, ra, rb).map(|r| (g, r)))
        {
            Some((g, (ga, gb))) => {
                if ga > ra {
                    bvh(ra, ga, cands);
                }
                if gb < rb {
                    bvh(gb, rb, cands);
                }
                // Each cell's range: the hull of its leaves' ranges, then
                // clipped once to its box. Its items give its programs
                // specialized to those leaves.
                let tmp = &mut lv.tmp_cands;
                tmp.clear();
                let items = &mut lv.items;
                g.walk(o, d, ga, gb, |first, leaf, a, b| {
                    tmp.extend(leaf.iter().map(|&c| (c, a, b)));
                    items.extend(
                        leaf.iter()
                            .enumerate()
                            .map(|(i, &c)| (c, (first + i) as u32)),
                    );
                });
                hull_per_cell(tmp);
                lv.full.extend(cands.iter().map(|e| e.0));
                cands.extend(tmp.iter().filter_map(|&(c, a, b)| {
                    let (x, y) = m.cells[c as usize].bbox.ray_range_inv(o, d, inv, a, b)?;
                    Some((c, x, y))
                }));
                lv.items.sort_unstable();
                lv.items.dedup();
            }
            None => bvh(ra, rb, cands),
        }
        cands.extend(u.always.iter().map(|&c| (c, ra, rb)));
        if lv.items.is_empty() {
            lv.full.clear();
        } else {
            lv.full.extend(u.always.iter().copied());
            lv.full.sort_unstable();
            lv.full.dedup();
        }
        hull_per_cell(cands);
    }

    /// The octree items of candidate `c` along the line (a range of
    /// `lv.items`), or `None` when its whole program counts.
    fn cell_items(&self, lv: &Level, grid: bool, c: u32) -> Option<(usize, usize)> {
        if self.brute || !grid || lv.full.binary_search(&c).is_ok() {
            return None;
        }
        let a = lv.items.partition_point(|e| e.0 < c);
        let b = lv.items.partition_point(|e| e.0 <= c);
        (a < b).then_some((a, b))
    }
}

/// One entry per cell: the hull of its ranges.
fn hull_per_cell(cands: &mut Vec<(u32, f64, f64)>) {
    cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut w = 0;
    for r in 0..cands.len() {
        let e = cands[r];
        if w > 0 && cands[w - 1].0 == e.0 {
            cands[w - 1].2 = cands[w - 1].2.max(e.2);
        } else {
            cands[w] = e;
            w += 1;
        }
    }
    cands.truncate(w);
}

/// Snap run ends closer than `tol` together (onto `ta` or `tb` when near
/// them), so coincident surfaces of neighbouring cells leave no slivers.
fn snap(lv: &mut Level, ta: f64, tb: f64, sa: u32, sb: u32, tol: f64) {
    lv.ends.clear();
    lv.ends.push((ta, sa, true));
    lv.ends.push((tb, sb, true));
    for r in &lv.runs {
        lv.ends.push((r.t0, r.s0, false));
        lv.ends.push((r.t1, r.s1, false));
    }
    lv.ends.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(b.2.cmp(&a.2))
            .then(surface_of(a.1).cmp(&surface_of(b.1)))
    });
    lv.clusters.clear();
    let snapping = tb - ta > tol;
    let mut i = 0;
    while i < lv.ends.len() {
        let first = lv.ends[i].0;
        let mut j = i;
        let (mut rep, mut surface, mut anchored, mut fixed) = (first, NO_SURFACE, false, false);
        while j < lv.ends.len()
            && (lv.ends[j].0 == first || (snapping && lv.ends[j].0 - first <= tol))
        {
            // The anchor's position, and its surface if it has one (a range
            // end inherited from the parent); else the members' lowest, so
            // the choice among coincident surfaces is canonical.
            let (t, s, anchor) = lv.ends[j];
            if anchor && !anchored {
                (rep, anchored) = (t, true);
                if s != NO_SURFACE {
                    (surface, fixed) = (s, true);
                }
            } else if !fixed
                && s != NO_SURFACE
                && (surface == NO_SURFACE || surface_of(s) < surface_of(surface))
            {
                surface = s;
            }
            j += 1;
        }
        lv.clusters.push(Cluster {
            first,
            rep,
            surface,
        });
        i = j;
    }
    let clusters = &lv.clusters;
    let find = |t: f64| clusters[clusters.partition_point(|c| c.first <= t) - 1];
    for r in &mut lv.runs {
        r.t0 = find(r.t0).rep;
        r.t1 = find(r.t1).rep;
    }
    lv.runs.retain(|r| r.t0 < r.t1);
}

/// Turn runs into pieces with the status point classification gives.
fn sweep(lv: &mut Level) {
    lv.pieces.clear();
    lv.piece_cells.clear();
    lv.runs
        .sort_unstable_by(|a, b| a.t0.total_cmp(&b.t0).then(a.cell.cmp(&b.cell)));
    lv.active.clear();
    let mut next = 0;
    lv.reps.clear();
    lv.reps
        .extend(lv.clusters.iter().map(|c| (c.rep, c.surface)));
    lv.reps.dedup_by(|b, a| a.0 == b.0);
    for i in 0..lv.reps.len().saturating_sub(1) {
        let ((x, sx), (y, sy)) = (lv.reps[i], lv.reps[i + 1]);
        if !(y > x) {
            continue;
        }
        lv.active.retain(|&r| lv.runs[r].t1 > x);
        while next < lv.runs.len() && lv.runs[next].t0 <= x {
            if lv.runs[next].t1 > x {
                lv.active.push(next);
            }
            next += 1;
        }
        let pick = |kind: Kind, set: &mut Vec<u32>| {
            set.clear();
            set.extend(
                lv.active
                    .iter()
                    .map(|&r| &lv.runs[r])
                    .filter(|r| r.kind == kind)
                    .map(|r| r.cell),
            );
            set.sort_unstable();
            set.dedup();
        };
        let mut set = mem::take(&mut lv.set);
        pick(Kind::True, &mut set);
        let status = match set.len() {
            1 => Status::Inside,
            n if n > 1 => Status::Overlap,
            _ => {
                pick(Kind::Unsupported, &mut set);
                if !set.is_empty() {
                    Status::Unsupported
                } else {
                    pick(Kind::Unknown, &mut set);
                    if set.is_empty() {
                        Status::Gap
                    } else {
                        Status::OnSurface
                    }
                }
            }
        };
        if let Some(p) = lv.pieces.last_mut() {
            if p.t1 == x && p.status == status && lv.piece_cells[p.cells.0..p.cells.1] == set[..] {
                p.t1 = y;
                p.s1 = sy;
                lv.set = set;
                continue;
            }
        }
        let start = lv.piece_cells.len();
        lv.piece_cells.extend_from_slice(&set);
        lv.pieces.push(Piece {
            t0: x,
            t1: y,
            s0: sx,
            s1: sy,
            status,
            cells: (start, lv.piece_cells.len()),
        });
        lv.set = set;
    }
}

/// Split `[s0, s1]` (in the lattice cell's frame) into lattice elements:
/// `(t0, t1, surface_in, surface_out, element)`. Element boundaries come
/// from walking face to face; those closer than `tol` to the ends or to the
/// previous boundary are dropped, as run ends are snapped. Each piece's
/// element is the one [`Lattice::index`] gives at its midpoint, as for
/// points.
#[allow(clippy::too_many_arguments)]
fn lattice_pieces(
    lat: &Lattice,
    o: Vec3,
    d: Vec3,
    s0: f64,
    s1: f64,
    sa: u32,
    sb: u32,
    depth: usize,
    tol: f64,
    breaks: &mut Vec<(f64, u32)>,
    out: &mut Vec<(f64, f64, u32, u32, [i32; 3])>,
) {
    breaks.clear();
    breaks.push((s0, sa));
    let ray = lat.prepare_ray(d);
    let mut t = s0;
    let mut ix = lat.index(at(o, d, t));
    let mut cached: Option<(f64, f64, Option<usize>)> = None;
    let mut step = LATTICE_STEP;
    while t < s1 {
        let (lo, hi, exit) = cached
            .take()
            .unwrap_or_else(|| lat.element_range_ray(o, &ray, ix));
        if lo <= t + step && hi > t + step {
            if hi >= s1 {
                break;
            }
            let last = breaks.last().unwrap().0;
            if hi - last > tol && s1 - hi > tol {
                breaks.push((
                    hi,
                    exit.map_or(NO_SURFACE, |k| bound(lat.face_surfaces[k], depth)),
                ));
            }
            (t, step) = (hi, LATTICE_STEP);
            if let Some(face) = exit {
                let nb = lat.step_across(ix, face);
                let r = lat.element_range_ray(o, &ray, nb);
                if r.0 < t + step && r.1 > t + step {
                    ix = nb;
                    cached = Some(r);
                }
            }
            continue;
        }
        // The element the line enters next: of the neighbours holding it
        // just past `t`, the one it leaves first.
        let mut best: Option<(f64, [i32; 3])> = None;
        for nb in lat.neighbours(ix) {
            let (l, h, _) = lat.element_range_ray(o, &ray, nb);
            if l <= t + step && h > t + step && best.is_none_or(|b| h < b.0) {
                best = Some((h, nb));
            }
        }
        match best {
            Some((_, nb)) => ix = nb,
            None => {
                // Rounding at a corner: look a little further on.
                step *= 8.0;
                if t + step >= s1 {
                    break;
                }
                ix = lat.index(at(o, d, t + step));
            }
        }
    }
    breaks.push((s1, sb));
    out.clear();
    for w in breaks.windows(2) {
        let ((a, sin), (b, sout)) = (w[0], w[1]);
        let ix = lat.index(at(o, d, mid(a, b)));
        match out.last_mut() {
            Some(p) if p.4 == ix => {
                p.1 = b;
                p.3 = sout;
            }
            _ => out.push((a, b, sin, sout, ix)),
        }
    }
}
