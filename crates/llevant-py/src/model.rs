//! `llevant.Model`: a compiled model, kept in memory and queried.

use llevant::Depth;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use llevant::{
    Aabb, Camera, Clip, Cut, Cutaway, Options, Scope, Slice, Vec3, VolumeGrid, DEFAULT_PRECISION,
};
use llevant_render::colours::Paint;
use llevant_render::shading::{render_image, Probe};
use llevant_scene::bounds;
use llevant_scene::hiding::Hiding;
use llevant_scene::index::ModelIndex;
use llevant_scene::labels::Labels;
use llevant_scene::materials::{Looks, Spec};
use llevant_scene::query::Query;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// The Python signatures write `llevant::DEFAULT_PROBES` out as a literal
// (pyo3 shows other defaults as `...`).
const _: () = assert!(llevant::DEFAULT_PROBES == 12);

use crate::batch::{Classification, Hits, Tracing};
use crate::convert::{self, err, invalid, vec3, Points, Xyz};
use crate::field::Field;
use crate::image::Image;
use crate::results::{
    Cells, Diagnostic, Diff, Fingerprints, Group, Label, Segment, Surface, Volumes,
};

/// A compiled MCNP model. Load it once with `llevant.load(path)` and query
/// it as often as needed: queries run in parallel in Rust, without the GIL.
///
/// Cells, materials and universes are MCNP numbers throughout; lengths are
/// in cm.
#[pyclass(frozen, module = "llevant")]
pub struct Model {
    pub inner: Arc<llevant::Model>,
    path: Option<PathBuf>,
    opts: Options,
    /// Measured extents ([`Model::extent`] with 128 lines per axis), by scope.
    extents: [OnceLock<Aabb>; 2],
    /// What the materials look like ([`Model::looks`](Self::looks)), made on first use
    /// and replaced by `set_materials`.
    looks: RwLock<Option<Arc<Looks>>>,
    gpu: Mutex<Option<llevant_gpu::GpuModel>>,
    /// The universe and cut this model was opened with (`universe`), for
    /// `reload`.
    opened: Option<(i64, Cut)>,
    /// The input text, for a model made from text (`from_text`,
    /// `recompile`): where the cells' labels are read from without a file.
    text: Option<Arc<str>>,
    /// What rules resolve against, made on first use; with the cells'
    /// labels once a rule needs them.
    index: OnceLock<Arc<ModelIndex>>,
}

static GPU: OnceLock<Result<llevant_gpu::Gpu, String>> = OnceLock::new();

fn gpu_device() -> PyResult<&'static llevant_gpu::Gpu> {
    GPU.get_or_init(|| llevant_gpu::Gpu::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| convert::LlevantError::new_err(e.clone()))
}

pub fn options(tolerance: f64, world_half_size: f64, grid: &str) -> PyResult<Options> {
    if !(tolerance > 0.0 && tolerance.is_finite()) {
        return Err(invalid("tolerance must be positive"));
    }
    if !(world_half_size > 0.0 && world_half_size.is_finite()) {
        return Err(invalid("world_half_size must be positive"));
    }
    Ok(Options::default()
        .with_tolerance(tolerance)
        .with_world_half_size(world_half_size)
        .with_grid(convert::grid(grid)?))
}

impl Model {
    pub fn new(inner: llevant::Model, path: Option<PathBuf>, opts: Options) -> Model {
        Model {
            inner: Arc::new(inner),
            path,
            opts,
            extents: [OnceLock::new(), OnceLock::new()],
            looks: RwLock::new(None),
            gpu: Mutex::new(None),
            opened: None,
            text: None,
            index: OnceLock::new(),
        }
    }

    /// The model, made from input `text`.
    pub fn with_text(mut self, text: Arc<str>) -> Model {
        self.text = Some(text);
        self
    }

    /// The index `rules` resolve against: with the cells' labels, read from
    /// the input file or text, if one of them has a `label` term (without
    /// either, those rules are errors that say why).
    fn index_for(&self, py: Python<'_>, rules: &[&Query]) -> PyResult<Arc<ModelIndex>> {
        let m = &self.inner;
        let x = self
            .index
            .get_or_init(|| Arc::new(py.detach(|| ModelIndex::new(m))))
            .clone();
        if x.labels().is_none() && rules.iter().any(|q| q.uses_labels()) {
            let labels = match (&self.text, &self.path) {
                (Some(t), _) => Some(py.detach(|| Labels::of_text(t))),
                (None, Some(p)) => Some(py.detach(|| Labels::read(p)).map_err(err)?),
                (None, None) => None,
            };
            if let Some(l) = labels {
                py.detach(|| x.set_labels(m, Arc::new(l)));
            }
        }
        Ok(x)
    }

    /// The file the model was read from.
    pub fn input_path(&self) -> Option<PathBuf> {
        self.path.clone()
    }

    /// Material looks from `{number: spec}` (`set_materials`'s format).
    fn parse_looks(&self, looks: std::collections::BTreeMap<i64, String>) -> PyResult<Arc<Looks>> {
        let specs = looks
            .into_iter()
            .map(|(id, s)| Ok((id, s.parse::<Spec>().map_err(err)?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Arc::new(Looks::new(&self.inner, &specs).map_err(err)?))
    }

    /// The looks of one image: `materials` if given, else the model's.
    fn looks_or(
        &self,
        materials: Option<std::collections::BTreeMap<i64, String>>,
    ) -> PyResult<Arc<Looks>> {
        match materials {
            Some(m) => self.parse_looks(m),
            None => Ok(self.looks()),
        }
    }

    /// The looks of the materials: classified from the cards, with the
    /// specs of `set_materials`.
    fn looks(&self) -> Arc<Looks> {
        if let Some(l) = self
            .looks
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            return l.clone();
        }
        let l = Arc::new(Looks::new(&self.inner, &[]).expect("no specs"));
        let mut slot = self.looks.write().unwrap_or_else(|e| e.into_inner());
        slot.get_or_insert(l).clone()
    }

    pub fn load(py: Python<'_>, path: PathBuf, opts: Options) -> PyResult<Model> {
        let m = py
            .detach(|| llevant_scene::store::Store::from_env().load(&path, &opts, None))
            .map_err(err)?;
        Ok(Model::new(m, Some(path), opts))
    }

    /// Universe `u` of this model, cut to `cut`, as a model of its own.
    fn open(&self, py: Python<'_>, u: i64, cut: Cut) -> PyResult<Model> {
        let (m, grid) = (&self.inner, self.opts.grid);
        let sub = py.detach(|| m.universe(u, cut, grid)).map_err(err)?;
        let mut out = Model::new(sub, self.path.clone(), self.opts.clone());
        out.opened = Some((u, cut));
        out.text = self.text.clone();
        Ok(out)
    }

    /// The measured extent of `scope`, computed once.
    fn measured(&self, py: Python<'_>, scope: Scope) -> PyResult<Aabb> {
        let k = match scope {
            Scope::Contents => 0,
            _ => 1,
        };
        let m = &self.inner;
        let b = *self.extents[k].get_or_init(|| py.detach(|| m.extent(scope, 128)));
        if !b.is_finite() || b.is_empty() {
            return Err(invalid(
                "could not measure the model's extent; give `region`",
            ));
        }
        Ok(b)
    }

    /// `region` if given, else the measured extent of `scope`.
    fn region_or(
        &self,
        py: Python<'_>,
        region: Option<&Bound<'_, PyAny>>,
        scope: Scope,
    ) -> PyResult<Aabb> {
        match region {
            Some(r) => convert::region(r),
            None => self.measured(py, scope),
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// The box around every instance of the cells of `q`.
    fn selection_box(&self, py: Python<'_>, q: &Query) -> PyResult<Option<Aabb>> {
        let m = &self.inner;
        let x = self.index_for(py, &[q])?;
        py.detach(|| {
            let set = q.resolve(m, &x).map_err(err)?;
            Ok(bounds::selection_box(m, &set))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn visibility(
        &self,
        py: Python<'_>,
        hide: Vec<i64>,
        hide_universes: Vec<i64>,
        hide_materials: Vec<i64>,
        show_void: bool,
        level: Option<usize>,
        ghost: Vec<String>,
        rules: &[String],
        isolate: Option<Query>,
    ) -> PyResult<llevant::Visibility> {
        let hiding = Hiding {
            void: !show_void,
            cells: hide,
            universes: hide_universes,
            materials: hide_materials,
            rules: rules
                .iter()
                .map(|r| query(r, "hide_rules"))
                .collect::<PyResult<_>>()?,
            isolate,
            level,
            ghost: ghost
                .iter()
                .map(|r| query(r, "ghost"))
                .collect::<PyResult<_>>()?,
            ..Hiding::none()
        };
        let rules: Vec<&Query> = hiding
            .isolate
            .iter()
            .chain(&hiding.rules)
            .chain(&hiding.ghost)
            .collect();
        if rules.is_empty() {
            return hiding.visibility(&self.inner).map_err(err);
        }
        let x = self.index_for(py, &rules)?;
        hiding.visibility_with(&self.inner, &x).map_err(err)
    }

    fn grid(
        &self,
        py: Python<'_>,
        h: f64,
        region: Option<&Bound<'_, PyAny>>,
        seed: u64,
        probes: usize,
    ) -> PyResult<VolumeGrid> {
        if !(h > 0.0 && h.is_finite()) {
            return Err(invalid("h must be positive"));
        }
        if probes < 2 {
            return Err(invalid("probes must be at least 2"));
        }
        Ok(VolumeGrid {
            probes,
            ..VolumeGrid::seeded(self.region_or(py, region, Scope::Problem)?, h, seed)
        })
    }

    fn volumes_of(&self, py: Python<'_>, g: VolumeGrid) -> PyResult<llevant::Volumes> {
        let m = &self.inner;
        py.detach(|| m.volumes(&g, None)).map_err(err)
    }
}

#[pymethods]
impl Model {
    /// The file the model was read from (None for `from_text`).
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.input_path()
    }

    /// Distance (cm) within which a point is on a surface.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance()
    }

    /// The cells by MCNP number: `m.cells[304]`, `304 in m.cells`.
    #[getter]
    fn cells(&self) -> Cells {
        Cells(self.inner.clone())
    }

    /// Universe numbers, 0 (the real world) first.
    #[getter]
    fn universes(&self) -> Vec<i64> {
        self.inner.units().iter().map(|u| u.universe).collect()
    }

    /// Number of cell instance ids: every place a cell occupies (each
    /// cell its universe fills, each lattice element) has one id below this
    /// (`Label.instance`). A lattice filled with one universe everywhere
    /// numbers the elements of a box around it, so some ids may name no
    /// place.
    #[getter]
    fn instance_count(&self) -> u64 {
        self.inner.instances().total()
    }

    /// The path of instance `id` in MCNP's repeated-structure notation,
    /// deepest cell first (`"12 < 100[1 0 0] < 5"`); None past the count.
    fn instance_path(&self, id: u64) -> Option<String> {
        let m = &self.inner;
        m.instance_path(id).map(|p| m.format_path(&p))
    }

    /// The instance id of a path written as `instance_path` writes it
    /// (lattice elements with one to three indices). ValueError if the
    /// model has no such path.
    fn instance(&self, path: &str) -> PyResult<u64> {
        let m = &self.inner;
        let steps = m.parse_path(path).map_err(err)?;
        m.instances()
            .of(m, &steps)
            .ok_or_else(|| invalid(format!("path `{path}` has no instance id")))
    }

    /// Global instance ids (`Label.instance`) for every occurrence of cell
    /// `cell_id` in instance-index order. KeyError if there is no such cell.
    fn cell_instances(&self, cell_id: i64) -> PyResult<Vec<u64>> {
        let m = &self.inner;
        let idx = m
            .cell_index(cell_id)
            .ok_or_else(|| PyKeyError::new_err(cell_id))?;
        Ok(m.instances().ids(idx).collect())
    }

    /// Problems found while compiling (unsupported features, undefined
    /// references...).
    #[getter]
    fn diagnostics(&self) -> Vec<Diagnostic> {
        self.inner
            .diagnostics()
            .iter()
            .map(Diagnostic::new)
            .collect()
    }

    /// The `Mn` card: `(zaid, fraction)` pairs as written (negative
    /// fractions are by weight). KeyError if there is none.
    fn material(&self, id: i64) -> PyResult<Vec<(String, f64)>> {
        self.inner
            .material(id)
            .map(|v| v.to_vec())
            .ok_or_else(|| PyKeyError::new_err(id))
    }

    /// What each material looks like in realistic images: per material
    /// (ascending), a dict of `material`, `cells` (that use it), `density`
    /// (smallest and largest, g/cm³, None if unknown), `look` (main
    /// category: steel, concrete, water, ...), `colour` (#rrggbb),
    /// `roughness`, `metalness`, `clarity`, `set` (whether `set_materials` gave it)
    /// and `parts` (category → volume fraction, from the composition).
    fn materials<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let looks = self.looks();
        let num = |x: f64| (!x.is_nan()).then_some(x);
        looks
            .materials
            .iter()
            .map(|x| {
                let a = &x.appearance;
                let [r, g, b] = a.srgb();
                let d = PyDict::new(py);
                d.set_item("material", x.id)?;
                d.set_item("cells", x.cells)?;
                d.set_item("density", (num(x.density.0), num(x.density.1)))?;
                d.set_item("look", a.category.name())?;
                d.set_item("colour", format!("#{r:02x}{g:02x}{b:02x}"))?;
                d.set_item("roughness", a.roughness)?;
                d.set_item("metalness", a.metalness)?;
                d.set_item("clarity", a.clarity)?;
                d.set_item("set", x.spec.is_some())?;
                let parts = PyDict::new(py);
                for (c, v) in &x.mix.parts {
                    parts.set_item(c.name(), v)?;
                }
                d.set_item("parts", parts)?;
                Ok(d)
            })
            .collect()
    }

    /// Change the look of materials in realistic images: `{number: spec}`,
    /// a spec being a category and/or settings, `"steel"`,
    /// `"copper roughness=0.2"`, `"colour=#806040 metalness=0"`. Replaces
    /// earlier settings; None or `{}` goes back to the looks classified
    /// from the material cards.
    ///
    /// This changes how this `Model` object draws later images (and what
    /// `materials()` reports); the compiled geometry, and other `Model`
    /// objects of the same file, are unchanged. For one image only, pass
    /// `materials=` to `slice` or `render` instead.
    #[pyo3(signature = (looks = None))]
    fn set_materials(
        &self,
        looks: Option<std::collections::BTreeMap<i64, String>>,
    ) -> PyResult<()> {
        let looks = self.parse_looks(looks.unwrap_or_default())?;
        *self.looks.write().unwrap_or_else(|e| e.into_inner()) = Some(looks);
        Ok(())
    }

    /// Sizes, compile times (s) and tolerance.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("cells", s.cells)?;
        d.set_item("surfaces", s.surfaces)?;
        d.set_item("universes", s.units)?;
        d.set_item("lattices", s.lattices)?;
        d.set_item("unsupported_cells", s.unsupported_cells)?;
        d.set_item("diagnostics", self.inner.diagnostics().len())?;
        d.set_item("code_words", s.code_words)?;
        d.set_item("octree_universes", s.grid_units)?;
        d.set_item("octree_bytes", s.grid_bytes)?;
        d.set_item("octrees_pending", s.pending_units)?;
        d.set_item("tolerance", self.inner.tolerance())?;
        let t = PyDict::new(py);
        for (n, x) in self.inner.timings() {
            t.set_item(*n, *x)?;
        }
        d.set_item("timings", t)?;
        Ok(d)
    }

    /// Box around the level-0 cells of `scope` from their bounding boxes:
    /// `contents` (cells with a material or fill) or `problem` (all but
    /// `IMP:N=0`). Loose for CAD conversions; see `extent`.
    #[pyo3(signature = (scope = "contents"))]
    fn bounds(&self, scope: &str) -> PyResult<Option<(Xyz, Xyz)>> {
        Ok(convert::bbox(&self.inner.bounds(convert::scope(scope)?)))
    }

    /// Box (global frame) around every instance of the cells of a rule
    /// (`"c 10-20"`, `"u 5 & m 3"`, `'label "BSV"'`: the cells whose `$`
    /// comments contain BSV, read from the input when a rule has
    /// `label`); `None` when they are nowhere (cells of
    /// universes no cell fills). Conservative: padded outward, and only an
    /// outer bound below lattices (see `instance_boxes`).
    fn bbox(&self, py: Python<'_>, cells: &str) -> PyResult<Option<(Xyz, Xyz)>> {
        let q = query(cells, "cells")?;
        Ok(self.selection_box(py, &q)?.and_then(|b| convert::bbox(&b)))
    }

    /// The box (global frame) of every instance of the cells of a rule:
    /// `(cell, (lo, hi), exact)` per instance, cell by cell. `exact` is
    /// false for boxes that are only an outer bound (below a lattice).
    fn instance_boxes(&self, py: Python<'_>, cells: &str) -> PyResult<Vec<InstanceBox>> {
        let q = query(cells, "cells")?;
        let m = &self.inner;
        let x = self.index_for(py, &[&q])?;
        let boxes = py.detach(|| -> PyResult<_> {
            let set = q.resolve(m, &x).map_err(err)?;
            Ok(bounds::instance_boxes(m, &set))
        })?;
        Ok(boxes
            .iter()
            .filter_map(|b| Some((m.cell_id(b.cell), convert::bbox(&b.global)?, b.exact)))
            .collect())
    }

    /// Box around `scope`, measured by tracing `n × n` lines per axis (the
    /// default region of slices, renders and volumes; computed once for
    /// `n = 128`).
    #[pyo3(signature = (scope = "contents", n = 128))]
    fn extent(&self, py: Python<'_>, scope: &str, n: usize) -> PyResult<Option<(Xyz, Xyz)>> {
        let scope = convert::scope(scope)?;
        if n == 128 {
            return Ok(convert::bbox(&self.measured(py, scope)?));
        }
        let m = &self.inner;
        Ok(convert::bbox(&py.detach(|| m.extent(scope, n))))
    }

    /// Where a point is: `locate(x, y, z)` or `locate((x, y, z))`; with
    /// `level`, its path down to that universe level at most (0: the top,
    /// where a filled cell ends it).
    #[pyo3(signature = (x, y = None, z = None, *, level = None))]
    fn locate(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<f64>,
        z: Option<f64>,
        level: Option<usize>,
    ) -> PyResult<Label> {
        let p = match (y, z) {
            (Some(y), Some(z)) => [x.extract::<f64>()?, y, z],
            (None, None) => vec3(x, "point")?,
            _ => return Err(invalid("give x, y and z, or one (x, y, z)")),
        };
        let m = &self.inner;
        let l = py.detach(|| match level {
            None => m.classify_point(p),
            Some(_) => m.classify(&[p], Depth::from(level)).labels.swap_remove(0),
        });
        Ok(Label::new(m, &l))
    }

    /// Labels of many points at once: `points` is an `(N, 3)` array (or
    /// anything `numpy.asarray` takes); `level` as in `locate`.
    #[pyo3(signature = (points, *, level = None))]
    fn classify(
        &self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        level: Option<usize>,
    ) -> PyResult<Classification> {
        let pts = Points::extract(points, "points")?;
        let pts = pts.get();
        let m = &self.inner;
        let c = py.detach(|| m.classify(&pts, Depth::from(level)));
        Classification::new(py, m.clone(), c)
    }

    /// Segments of one line, from `start` to `end`, or from `start` along
    /// `dir` for `t` in `(t_min, t_max)` (cm); `level` as in `locate`.
    #[pyo3(signature = (start, end = None, *, dir = None, t = None, level = None))]
    fn trace(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: Option<&Bound<'_, PyAny>>,
        dir: Option<&Bound<'_, PyAny>>,
        t: Option<(f64, f64)>,
        level: Option<usize>,
    ) -> PyResult<Vec<Segment>> {
        let a = vec3(start, "start")?;
        let line = match (end, dir, t) {
            (Some(e), None, None) => {
                let b = vec3(e, "end")?;
                if a == b {
                    return Err(invalid("start and end are the same point"));
                }
                llevant::Line::between(a, b)
            }
            (None, Some(d), Some((lo, hi))) => llevant::Line::new(a, vec3(d, "dir")?, lo, hi),
            _ => return Err(invalid("give `end`, or `dir` and `t=(t_min, t_max)`")),
        };
        let m = &self.inner;
        let tr = py
            .detach(|| m.trace(std::slice::from_ref(&line), Depth::from(level)))
            .map_err(err)?;
        let segs: Vec<llevant::Segment> = tr.segments(0).map(|k| tr.segment(k)).collect();
        segs.into_iter()
            .map(|s| {
                Ok(Segment {
                    t0: s.t0,
                    t1: s.t1,
                    label: Py::new(py, Label::new(m, &s.label))?,
                    surface_in: Surface::of(m, s.surface_in),
                    surface_out: Surface::of(m, s.surface_out),
                })
            })
            .collect()
    }

    /// Segments of many lines at once: from `start` to `end` (`(N, 3)`
    /// arrays), or from `start` along `dir` for `t` in `(t_min, t_max)`
    /// (numbers or arrays of N); `level` as in `locate`.
    #[pyo3(signature = (start, end = None, *, dir = None, t_min = None, t_max = None, level = None))]
    #[allow(clippy::too_many_arguments)]
    fn trace_lines(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: Option<&Bound<'_, PyAny>>,
        dir: Option<&Bound<'_, PyAny>>,
        t_min: Option<&Bound<'_, PyAny>>,
        t_max: Option<&Bound<'_, PyAny>>,
        level: Option<usize>,
    ) -> PyResult<Tracing> {
        let zero = 0.0f64.into_pyobject(py)?.into_any();
        let lines = convert::lines(start, end, dir, t_min.unwrap_or(&zero), t_max, None)?;
        let m = &self.inner;
        let tr = py
            .detach(|| m.trace(&lines, Depth::from(level)))
            .map_err(err)?;
        Tracing::new(py, m.clone(), tr)
    }

    /// Where rays first enter a visible cell: from `start` along `dir`
    /// (`(N, 3)` arrays) for `t` in `(t_min, t_max)` (default: far beyond
    /// the model). Void cells are hidden unless `show_void`; `hide` and
    /// `hide_universes` hide more (and everything inside them), and
    /// `hide_materials` the cells of those materials (filled cells stay, so
    /// what fills them shows). `level`: only cells at that universe level
    /// are seen, as MCNP's plotter shows `LEVEL n` (a filled cell there is
    /// solid). `ghost`: rules (`"c 300"`, `"u 20040 & m 5"`) of cells seen
    /// as shells, what fills them shown; `Hits.ghost_count` and
    /// `Hits.ghost_cell` say what each ray crossed of them.
    #[pyo3(signature = (start, dir, *, t_min = None, t_max = None, hide = vec![], hide_universes = vec![], hide_materials = vec![], show_void = false, level = None, ghost = vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn first_hit(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        dir: &Bound<'_, PyAny>,
        t_min: Option<&Bound<'_, PyAny>>,
        t_max: Option<&Bound<'_, PyAny>>,
        hide: Vec<i64>,
        hide_universes: Vec<i64>,
        hide_materials: Vec<i64>,
        show_void: bool,
        level: Option<usize>,
        ghost: Vec<String>,
    ) -> PyResult<Hits> {
        let zero = 0.0f64.into_pyobject(py)?.into_any();
        let far = 4.0 * self.opts.world_half_size;
        let rays = convert::lines(
            start,
            None,
            Some(dir),
            t_min.unwrap_or(&zero),
            t_max,
            Some(far),
        )?;
        let vis = self.visibility(
            py,
            hide,
            hide_universes,
            hide_materials,
            show_void,
            level,
            ghost,
            &[],
            None,
        )?;
        let m = &self.inner;
        let h = py.detach(|| m.first_hits(&rays, &vis, None)).map_err(err)?;
        Hits::new(py, m.clone(), h)
    }

    /// Image of the plane `x=`, `y=` or `z=` (give one) across `region`
    /// (default: the measured extent of the contents), `width` pixels wide,
    /// coloured by `cell`, `material`, `universe` or `realistic`, with
    /// `field` (a `Field` of this model) over the cells. `level`: only
    /// cells at that universe level, as MCNP's plotter shows `LEVEL n`
    /// (the rest blank). `outline_by`: the lines between cells (`"cell"`),
    /// or between the materials (`"material"`, void one of them) or
    /// universes (`"universe"`) of the deepest cells.
    ///
    /// `materials`: looks for this image only, `{number: spec}` as for
    /// `set_materials`, instead of the model's.
    #[pyo3(signature = (*, x = None, y = None, z = None, region = None, width = 1000, colour = "cell", outline = true, outline_by = "cell", field = None, level = None, materials = None))]
    #[allow(clippy::too_many_arguments)]
    fn slice(
        &self,
        py: Python<'_>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
        region: Option<&Bound<'_, PyAny>>,
        width: usize,
        colour: &str,
        outline: bool,
        outline_by: &str,
        field: Option<PyRef<'_, Field>>,
        level: Option<usize>,
        materials: Option<std::collections::BTreeMap<i64, String>>,
    ) -> PyResult<Image> {
        let outline_by = llevant_render::colours::Outlines::from_name(outline_by)
            .ok_or_else(|| invalid("outline_by: \"cell\", \"material\" or \"universe\""))?;
        let outline = outline.then_some(outline_by);
        let (axis, at) = match (x, y, z) {
            (Some(v), None, None) => (llevant::Axis::X, v),
            (None, Some(v), None) => (llevant::Axis::Y, v),
            (None, None, Some(v)) => (llevant::Axis::Z, v),
            _ => return Err(invalid("give one of x=, y= or z=")),
        };
        if width == 0 {
            return Err(invalid("width must be positive"));
        }
        let by = convert::colour(colour)?;
        let s = Slice::axis(
            axis,
            at,
            &self.region_or(py, region, Scope::Contents)?,
            width,
        );
        let m = &self.inner;
        let looks = self.looks_or(materials)?;
        let field = field.as_ref().map(|f| f.for_model(m)).transpose()?;
        let img = py.detach(|| {
            m.slice(&s, Depth::from(level))
                .map(|tr| Image::slice(m.clone(), &looks, field, s, tr, by, outline, level))
        });
        img.map_err(err)
    }

    /// 3D view: the first visible cell along each pixel's ray, shaded.
    ///
    /// The camera looks along `direction` and frames `region` (default:
    /// around every instance of the cells of `isolate`, or the measured
    /// extent of the contents), or looks from `eye` at `target` with a
    /// vertical field of view `fov` (degrees). Without `direction`, it
    /// looks along (0.6, 0.7, -0.4), or with `isolate` from where the cells
    /// show most (as `llevant render --isolate`). `clip=(point, normal)`
    /// cuts away the half-space the normal points into. Void cells are
    /// hidden unless `show_void`; `hide`, `hide_universes` and
    /// `hide_materials` hide more, as in `first_hit`, and so do `hide_rules`
    /// (rules of cells, `"u 20040 & m 5"`). `isolate` (a rule) shows only
    /// its cells, what they hold and the cells holding them. `gpu=True` renders on the GPU (faster,
    /// approximate at edges); the model is uploaded once. Not on WARP, Windows'
    /// software adapter: an error there.
    ///
    /// `colour`: `cell`, `material`, `universe` or `realistic` (what the
    /// materials look like, see `materials()`; lit physically, the others
    /// flat). `field`: a `Field` of this model drawn over the cells. `level` and `ghost` as
    /// in `first_hit`: ghosts are drawn as shells over the image.
    ///
    /// `materials`: looks for this image only, `{number: spec}` as for
    /// `set_materials`, instead of the model's.
    #[pyo3(signature = (*, direction = None, eye = None, target = None, fov = 30.0, size = (1000, 1000), region = None, clip = None, hide = vec![], hide_universes = vec![], hide_materials = vec![], show_void = false, colour = "cell", gpu = false, field = None, level = None, ghost = vec![], hide_rules = vec![], isolate = None, materials = None))]
    #[allow(clippy::too_many_arguments)]
    fn render(
        &self,
        py: Python<'_>,
        direction: Option<&Bound<'_, PyAny>>,
        eye: Option<&Bound<'_, PyAny>>,
        target: Option<&Bound<'_, PyAny>>,
        fov: f64,
        size: (usize, usize),
        region: Option<&Bound<'_, PyAny>>,
        clip: Option<(Vec3, Vec3)>,
        hide: Vec<i64>,
        hide_universes: Vec<i64>,
        hide_materials: Vec<i64>,
        show_void: bool,
        colour: &str,
        gpu: bool,
        field: Option<PyRef<'_, Field>>,
        level: Option<usize>,
        ghost: Vec<String>,
        hide_rules: Vec<String>,
        isolate: Option<String>,
        materials: Option<std::collections::BTreeMap<i64, String>>,
    ) -> PyResult<Image> {
        let (w, h) = size;
        if w == 0 || h == 0 {
            return Err(invalid("size must be positive"));
        }
        if !(fov > 0.0 && fov < 180.0) {
            return Err(invalid("fov must be between 0 and 180 degrees"));
        }
        let by = convert::colour(colour)?;
        let isolate = isolate.map(|r| query(&r, "isolate")).transpose()?;
        // With `isolate`, framed on the cells isolated.
        let b = match (&isolate, region, eye) {
            (Some(q), None, None) => self.selection_box(py, q)?.ok_or_else(|| {
                invalid(format!(
                    "isolate `{q}`: the cells are nowhere; give a region"
                ))
            })?,
            _ => self.region_or(py, region, Scope::Contents)?,
        };
        let isolating = isolate.is_some();
        let vis = self.visibility(
            py,
            hide,
            hide_universes,
            hide_materials,
            show_void,
            level,
            ghost,
            &hide_rules,
            isolate,
        )?;
        let mut cam = match (eye, target) {
            (Some(e), Some(t)) => {
                if direction.is_some() {
                    return Err(invalid("give `direction` or `eye` and `target`, not both"));
                }
                let (e, t) = (vec3(e, "eye")?, vec3(t, "target")?);
                let dir = [0, 1, 2].map(|i| t[i] - e[i]);
                let d = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
                if !(d > 0.0) {
                    return Err(invalid("eye and target are the same point"));
                }
                let span = (0..3).map(|i| b.hi[i] - b.lo[i]).fold(0.0, f64::max);
                Camera {
                    eye: e,
                    target: t,
                    up: if dir[2].abs() > 0.99 * d {
                        [0.0, 1.0, 0.0]
                    } else {
                        [0.0, 0.0, 1.0]
                    },
                    fov_deg: fov,
                    width: w,
                    height: h,
                    reach: d + 2.0 * span,
                    cut: Cutaway::none(),
                    ortho: None,
                }
            }
            (None, None) => {
                let d = match direction {
                    Some(d) => vec3(d, "direction")?,
                    // Where the cells isolated show most.
                    None if isolating => {
                        let m = &self.inner;
                        let from = py
                            .detach(|| bounds::view_from(m, &vis, &b, [-0.6, -0.7, 0.4]))
                            .map_err(err)?;
                        from.map(|x| -x)
                    }
                    None => [0.6, 0.7, -0.4],
                };
                if d == [0.0; 3] || d.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("direction must be a non-zero vector"));
                }
                let mut c = Camera::fit(&b, d, w, h);
                if fov != 30.0 {
                    c.fov_deg = fov;
                }
                c
            }
            _ => return Err(invalid("give both `eye` and `target`")),
        };
        cam.cut = clip.map_or_else(Cutaway::none, |(point, normal)| {
            Cutaway::plane(Clip { point, normal })
        });
        let m = &self.inner;
        let looks = self.looks_or(materials)?;
        let looks = &*looks;
        let field = field.as_ref().map(|f| f.for_model(m)).transpose()?;
        let paint = Paint::new(m, looks, by).with_field(field);
        if gpu {
            let device = gpu_device()?;
            let out = py.detach(|| {
                let mut g = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
                if g.is_none() {
                    *g = Some(llevant_gpu::GpuModel::upload(device, m)?);
                }
                let gm = g.as_ref().expect("uploaded");
                gm.set_visibility(device, m, &vis)?;
                let frame = gm.render(device, &cam)?;
                let mut rgb = render_image(&paint, &frame.probes(), &cam)?;
                llevant_render::ghost::overlay(&paint, &frame.ghosts, &cam, &mut rgb);
                Ok::<_, llevant::Error>((frame, rgb))
            });
            let (frame, rgb) = out.map_err(err)?;
            return Ok(Image::gpu(m.clone(), cam, vis, &frame, rgb));
        }
        let out = py.detach(|| {
            let hits = m.render(&cam, &vis)?;
            let probes = Probe::from_hits(m, &hits);
            let mut rgb = render_image(&paint, &probes, &cam)?;
            llevant_render::ghost::overlay(&paint, &hits.ghosts, &cam, &mut rgb);
            Ok::<_, llevant::Error>((hits, rgb))
        });
        let (hits, rgb) = out.map_err(err)?;
        Ok(Image::render(m.clone(), cam, hits, rgb))
    }

    /// Volumes of everything in `region` (default: the measured extent of
    /// the problem), from `probes` lattices of lines in random directions,
    /// as many lines as three axis-parallel families `h` cm apart. Sigma is
    /// the standard error over the probes. The same `seed` gives the same
    /// lines, also on another model. Long runs cannot be interrupted with
    /// Ctrl-C.
    #[pyo3(signature = (h, *, region = None, seed = 1, probes = 12))]
    fn volumes(
        &self,
        py: Python<'_>,
        h: f64,
        region: Option<&Bound<'_, PyAny>>,
        seed: u64,
        probes: usize,
    ) -> PyResult<Volumes> {
        let g = self.grid(py, h, region, seed, probes)?;
        Ok(Volumes {
            model: self.inner.clone(),
            v: self.volumes_of(py, g)?,
        })
    }

    /// Gaps, overlaps, unsupported cells and too-deep nesting found by
    /// `volumes(h, ...)`, largest first (empty if none).
    #[pyo3(signature = (h, *, region = None, seed = 1, probes = 12))]
    fn check(
        &self,
        py: Python<'_>,
        h: f64,
        region: Option<&Bound<'_, PyAny>>,
        seed: u64,
        probes: usize,
    ) -> PyResult<Vec<Group>> {
        let g = self.grid(py, h, region, seed, probes)?;
        let v = self.volumes_of(py, g)?;
        v.problems()
            .into_iter()
            .map(|i| Group::of_label(py, &self.inner, &v, i))
            .collect()
    }

    /// Volumes grouped `by` cell, material, universe or label, those inside
    /// cells refined until sigma / volume is at most `rel_sigma` (halving
    /// `h` over the boxes their cells can be in, down to `h_min`, default
    /// h / 64). Every group the model can hold is listed, met or not. Each
    /// group's `h` says which lines gave it, `converged` whether it got
    /// there and `outcome` how: "converged", "not converged", "not found"
    /// or "empty". A group whose sigma is at most `abs_sigma` cm³ (0: none)
    /// is done too, however large its relative sigma. `max_probes` (default
    /// `probes`: none added) lets groups a little above the target get more
    /// probes at one spacing: more converge, for more time.
    #[pyo3(signature = (h, rel_sigma, *, h_min = None, abs_sigma = 0.0, max_probes = None,
                        by = "cell", region = None, seed = 1, probes = 12))]
    #[allow(clippy::too_many_arguments)]
    fn volumes_to(
        &self,
        py: Python<'_>,
        h: f64,
        rel_sigma: f64,
        h_min: Option<f64>,
        abs_sigma: f64,
        max_probes: Option<usize>,
        by: &str,
        region: Option<&Bound<'_, PyAny>>,
        seed: u64,
        probes: usize,
    ) -> PyResult<Vec<Group>> {
        let g = self.grid(py, h, region, seed, probes)?;
        let target = llevant::VolumeTarget {
            rel_sigma,
            h_min: h_min.unwrap_or(h / 64.0),
            abs_sigma,
            max_probes: max_probes.unwrap_or(probes),
            by: crate::convert::group(by)?,
        };
        let m = &self.inner;
        let c = py
            .detach(|| {
                m.volumes_to(
                    &g,
                    &target,
                    |g| g.status == llevant::Status::Inside,
                    None,
                    None,
                )
            })
            .map_err(err)?;
        converged_groups(py, &self.inner, target.by, c)
    }

    /// Volumes of the cells `cells` (MCNP numbers) alone, in that order:
    /// every instance of each together, lines traced only around them; a
    /// filled cell's volume is that of its region. Each cell starts at the
    /// spacing that puts 8 lines across its box (at most `h`) and, with
    /// `rel_sigma`, is refined until sigma / volume is at most that, down to
    /// `h_min` (default h / 64), or until sigma is at most `abs_sigma` cm³
    /// (0: none); without it, one pass. Up to `max_probes` (default 48) at
    /// one spacing. `converged` and `outcome` as for `volumes_to`.
    #[pyo3(signature = (cells, h, rel_sigma = None, *, h_min = None, abs_sigma = 0.0,
                        max_probes = 48, region = None, seed = 1, probes = 12))]
    #[allow(clippy::too_many_arguments)]
    fn cell_volumes(
        &self,
        py: Python<'_>,
        cells: Vec<i64>,
        h: f64,
        rel_sigma: Option<f64>,
        h_min: Option<f64>,
        abs_sigma: f64,
        max_probes: usize,
        region: Option<&Bound<'_, PyAny>>,
        seed: u64,
        probes: usize,
    ) -> PyResult<Vec<Group>> {
        let g = self.grid(py, h, region, seed, probes)?;
        let m = &self.inner;
        let idx = cells
            .iter()
            .map(|&c| m.cell_index(c).ok_or_else(|| PyKeyError::new_err(c)))
            .collect::<PyResult<Vec<u32>>>()?;
        let target = llevant::BinTarget {
            h_max: h,
            abs_sigma,
            max_probes,
            seed,
            probes,
            ..llevant::BinTarget::new(
                g.region,
                rel_sigma.unwrap_or(f64::INFINITY),
                h_min.unwrap_or(h / 64.0),
            )
        };
        let c = py
            .detach(|| m.cell_volumes(&idx, &target, None))
            .map_err(err)?;
        converged_groups(py, &self.inner, llevant::GroupBy::Cell, c)
    }

    /// Digests of every cell and universe that ignore numbering and card
    /// order, at relative `precision`.
    #[pyo3(signature = (precision = DEFAULT_PRECISION))]
    fn fingerprints(&self, py: Python<'_>, precision: f64) -> PyResult<Fingerprints> {
        let m = &self.inner;
        let f = py.detach(|| m.fingerprints(precision)).map_err(err)?;
        Fingerprints::new(py, m, &f)
    }

    /// How `other` differs from this model, cell by cell and universe by
    /// universe, up to numbering.
    #[pyo3(signature = (other, precision = DEFAULT_PRECISION))]
    fn diff(&self, py: Python<'_>, other: &Model, precision: f64) -> PyResult<Diff> {
        let (a, b) = (&self.inner, &other.inner);
        let d = py
            .detach(|| a.fingerprint_diff(b, precision))
            .map_err(err)?;
        Diff::new(py, &d)
    }

    /// Read and compile the file again, with the same options (and the
    /// same universe, for a model made by `universe`). Only the universes
    /// that changed are built again: the rest is taken from this model.
    fn reload(&self, py: Python<'_>) -> PyResult<Model> {
        let p = self
            .path
            .clone()
            .ok_or_else(|| invalid("this model was not read from a file"))?;
        // A model made by `universe` is cut: no base for the whole input.
        let base = self.opened.is_none().then(|| self.inner.clone());
        let opts = self.opts.clone();
        let m = py
            .detach(|| {
                let store = llevant_scene::store::Store::from_env();
                let staged = store.load_staged(&p, &opts, None, base.as_deref())?;
                let m = store.complete(staged, None)?;
                Ok(Arc::into_inner(m).expect("the only reference to a model just loaded"))
            })
            .map_err(err)?;
        let m = Model::new(m, Some(p), opts);
        match self.opened {
            Some((u, cut)) => m.open(py, u, cut),
            None => Ok(m),
        }
    }

    /// Compile `text`, a new version of this model's input (an edit, or a
    /// new assembly made in memory), with the same options (and the same
    /// universe, for a model made by `universe`). Only the universes that
    /// changed are built again: the rest is taken from this model. The
    /// model it gives is not read from a file (`path` is None).
    fn recompile(&self, py: Python<'_>, text: &str) -> PyResult<Model> {
        let opts = self.opts.clone();
        let inner = self.inner.clone();
        let m = py
            .detach(|| {
                let (m, _) = inner.recompile(text.as_bytes(), &opts, None)?;
                m.into_complete(None)
            })
            .map_err(err)?;
        let m = Model::new(m, None, opts).with_text(text.into());
        match self.opened {
            Some((u, cut)) => m.open(py, u, cut),
            None => Ok(m),
        }
    }

    /// Universe `u` as a model of its own, in its own frame, with
    /// everything it holds; cell and universe numbers stay the same. Cut to
    /// a box beyond which lies one void `IMP:N=0` cell (`cut_cell`):
    /// `"own"` (around what the universe holds, measured, wherever the
    /// model places it), `"none"`, or a box as for `region`.
    #[pyo3(signature = (u, cut = None))]
    fn universe(&self, py: Python<'_>, u: i64, cut: Option<&Bound<'_, PyAny>>) -> PyResult<Model> {
        let cut = match cut {
            None => Cut::Own,
            Some(c) => match c.extract::<String>() {
                Ok(s) if s == "own" => Cut::Own,
                Ok(s) if s == "none" => Cut::None,
                Ok(s) => {
                    return Err(invalid(format!(
                        "cut: expected \"own\", \"none\" or a box, got {s:?}"
                    )))
                }
                Err(_) => Cut::Box(convert::region(c)?),
            },
        };
        self.open(py, u, cut)
    }

    /// The problem box: beyond it only `IMP:N=0` cells are checked (usually
    /// the box of the graveyard sphere; the cut box of a `universe`). None
    /// without one.
    #[getter]
    fn problem_box(&self) -> Option<(Xyz, Xyz)> {
        self.inner.problem_box().and_then(|b| convert::bbox(&b))
    }

    /// The MCNP number of the cell beyond the cut box of a model made by
    /// `universe` (None if it was not cut).
    #[getter]
    fn cut_cell(&self) -> Option<i64> {
        self.inner.cut_cell().map(|c| self.inner.cell_id(c))
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        let name = self
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or("<text>".to_string(), |n| n.to_string_lossy().into_owned());
        format!(
            "<Model {name}: {} cells, {} surfaces, {} universes{}>",
            s.cells,
            s.surfaces,
            s.units,
            match self.inner.diagnostics().len() {
                0 => String::new(),
                n => format!(", {n} diagnostics"),
            }
        )
    }
}

/// Groups of `volumes_to` or `cell_volumes`, with whether each converged
/// and how it ended.
fn converged_groups(
    py: Python<'_>,
    m: &llevant::Model,
    by: llevant::GroupBy,
    c: llevant::Converged,
) -> PyResult<Vec<Group>> {
    let looks = Looks::new(m, &[]).map_err(err)?;
    c.groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let has = |v: &[usize]| v.binary_search(&i).is_ok();
            let inside = g.status == llevant::Status::Inside;
            let mass = llevant_scene::materials::group_mass(m, &looks, by, g, None);
            Ok(Group {
                converged: inside.then(|| !has(&c.unconverged)),
                outcome: inside.then(|| {
                    if has(&c.empty) {
                        "empty"
                    } else if has(&c.unseen) {
                        "not found"
                    } else if has(&c.unconverged) {
                        "not converged"
                    } else {
                        "converged"
                    }
                }),
                ..Group::of(py, m, g, c.h[i], mass)?
            })
        })
        .collect()
}

/// A cell instance's box: the cell number, `(lo, hi)`, and whether it is
/// exact rather than only an outer bound.
type InstanceBox = (i64, (Xyz, Xyz), bool);

/// A rule of cells given as argument `what`.
fn query(s: &str, what: &str) -> PyResult<Query> {
    s.parse().map_err(|e| invalid(format!("{what} `{s}`: {e}")))
}
