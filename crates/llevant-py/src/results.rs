//! Small result types: labels, segments, cells, diagnostics, volumes and
//! fingerprints, in MCNP numbers.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use llevant::{Change, Fill, Status, Subject, SurfaceId as RsSurfaceId};
use llevant_scene::materials::{group_mass, Looks};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{bbox, err, status_code, xyz, Xyz};

/// One level of a path: cell number and lattice element.
type Step = (i64, Option<[i32; 3]>);

/// Where a point is: a status, the path of cells through the universes,
/// and the cells involved in an overlap (or the like).
///
/// `str(label)` gives it in MCNP terms, e.g. `inside 12 > 304[1,-2,0] > 7`.
/// Labels compare and hash by content, so they can be dictionary keys.
#[pyclass(frozen, module = "llevant")]
pub struct Label {
    status: Status,
    path: Vec<Step>,
    involved: Vec<i64>,
    /// Universe and material of the cell that holds the point (inside only).
    universe: Option<i64>,
    material: Option<i64>,
    density: Option<f64>,
    /// The instance holding the point, and its path in MCNP notation.
    instance: Option<(u64, String)>,
    text: String,
}

impl Label {
    pub fn new(m: &llevant::Model, l: &llevant::Label) -> Label {
        let cell = l.cell().map(|c| m.cell(c));
        Label {
            status: l.status,
            path: l
                .path
                .iter()
                .map(|s| (m.cell_id(s.cell), s.lattice))
                .collect(),
            involved: l.involved.iter().map(|&c| m.cell_id(c)).collect(),
            universe: cell.map(|c| m.units()[c.unit as usize].universe),
            material: cell.map(|c| c.material),
            density: cell.map(|c| c.density),
            instance: m.instance(l).map(|i| (i, m.format_path(&l.path))),
            text: m.describe(l),
        }
    }

    fn key(&self) -> (i8, &[Step], &[i64]) {
        (status_code(self.status), &self.path, &self.involved)
    }
}

#[pymethods]
impl Label {
    /// `inside`, `gap`, `overlap`, `on-surface`, `unsupported` or `too-deep`.
    #[getter]
    fn status(&self) -> &'static str {
        self.status.name()
    }

    /// The cell that holds the point (MCNP number), when it is inside one.
    #[getter]
    fn cell(&self) -> Option<i64> {
        (self.status == Status::Inside)
            .then(|| self.path.last().map(|s| s.0))
            .flatten()
    }

    /// Cells from level 0 down, as `(cell, lattice element or None)`. For a
    /// point that is not inside, the path ends at the filled cell whose
    /// universe has the problem.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn path(&self) -> Vec<(i64, Option<(i32, i32, i32)>)> {
        self.path
            .iter()
            .map(|&(c, l)| (c, l.map(|[i, j, k]| (i, j, k))))
            .collect()
    }

    /// The cells involved, for overlaps, on-surface and unsupported points
    /// (at the last level of the path); empty otherwise.
    #[getter]
    fn involved(&self) -> Vec<i64> {
        self.involved.clone()
    }

    /// Universe of the cell that holds the point.
    #[getter]
    fn universe(&self) -> Option<i64> {
        self.universe
    }

    /// Material of the cell that holds the point (0 for void).
    #[getter]
    fn material(&self) -> Option<i64> {
        self.material
    }

    /// Density of the cell that holds the point, with the MCNP sign.
    #[getter]
    fn density(&self) -> Option<f64> {
        self.density
    }

    /// Which instance of its cell holds the point: an id in
    /// `0 .. Model.instance_count`, one per place the cell occupies (each
    /// cell its universe fills, each lattice element). None when not
    /// inside a cell.
    #[getter]
    fn instance(&self) -> Option<u64> {
        self.instance.as_ref().map(|i| i.0)
    }

    /// The path in MCNP's repeated-structure notation, deepest cell first
    /// (`"12 < 100[1 0 0] < 5"`), when inside a cell.
    #[getter]
    fn instance_path(&self) -> Option<&str> {
        self.instance.as_ref().map(|i| i.1.as_str())
    }

    /// Whether the point is inside a cell (`status == "inside"`).
    #[getter]
    fn is_inside(&self) -> bool {
        self.status == Status::Inside
    }

    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!("<Label {}>", self.text)
    }

    /// Equal when status, path and involved cells are equal.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Label>()
            .is_ok_and(|o| o.get().key() == self.key())
    }

    /// Labels hash by content, so they can be dictionary keys.
    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.key().hash(&mut h);
        h.finish()
    }
}

/// A surface: MCNP number and macrobody facet (0 for an ordinary surface).
/// `str()` gives `12` or `12.3`.
#[pyclass(frozen, eq, hash, module = "llevant")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Surface {
    /// MCNP surface number.
    #[pyo3(get)]
    id: i64,
    /// Macrobody facet number (1-based), 0 for an ordinary surface.
    #[pyo3(get)]
    facet: u8,
}

impl Surface {
    pub fn new(s: RsSurfaceId) -> Surface {
        Surface {
            id: s.id,
            facet: s.facet,
        }
    }

    pub fn of(m: &llevant::Model, shape: Option<u32>) -> Option<Surface> {
        shape.and_then(|s| m.surface_id(s)).map(Surface::new)
    }
}

#[pymethods]
impl Surface {
    fn __str__(&self) -> String {
        match self.facet {
            0 => self.id.to_string(),
            f => format!("{}.{f}", self.id),
        }
    }

    fn __repr__(&self) -> String {
        format!("<Surface {}>", self.__str__())
    }
}

/// A stretch of a line in one label, from `t0` to `t1` (cm along the line),
/// entered through `surface_in` and left through `surface_out` (None at the
/// ends of the line).
#[pyclass(frozen, get_all, module = "llevant")]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub label: Py<Label>,
    pub surface_in: Option<Surface>,
    pub surface_out: Option<Surface>,
}

#[pymethods]
impl Segment {
    /// `t1 - t0`, cm.
    #[getter]
    fn length(&self) -> f64 {
        self.t1 - self.t0
    }

    fn __repr__(&self) -> String {
        let s = |x: &Option<Surface>| x.map_or("-".to_string(), |s| s.__str__());
        format!(
            "<Segment {} .. {}: {} (surfaces {} > {})>",
            self.t0,
            self.t1,
            self.label.get().text,
            s(&self.surface_in),
            s(&self.surface_out)
        )
    }
}

/// A problem found while compiling: unsupported features, undefined
/// references... The affected cells are compiled as unsupported.
#[pyclass(frozen, get_all, module = "llevant")]
pub struct Diagnostic {
    /// `cell`, `surface`, `material` or `input`.
    pub subject: &'static str,
    /// The cell, surface or material number.
    pub number: Option<i64>,
    pub message: String,
    /// Line of the card in the input (1-based).
    pub line: Option<usize>,
}

impl Diagnostic {
    pub fn new(d: &llevant::Diagnostic) -> Diagnostic {
        let (subject, number) = match d.subject {
            Subject::Cell(n) => ("cell", Some(n)),
            Subject::Surface(n) => ("surface", Some(n)),
            Subject::Material(n) => ("material", Some(n)),
            _ => ("input", None),
        };
        Diagnostic {
            subject,
            number,
            message: d.message.clone(),
            line: d.line,
        }
    }
}

#[pymethods]
impl Diagnostic {
    fn __str__(&self) -> String {
        match self.line {
            Some(l) => format!("line {l}: {}", self.message),
            None => self.message.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<Diagnostic {}>", self.__str__())
    }
}

/// A cell card, as compiled.
#[pyclass(frozen, get_all, module = "llevant")]
pub struct Cell {
    /// MCNP cell number.
    pub id: i64,
    /// Universe the cell belongs to (0 for the real world).
    pub universe: i64,
    /// Material number, 0 for void.
    pub material: i64,
    /// Density with the MCNP sign (negative: g/cm³), 0 for void.
    pub density: f64,
    /// Universes that fill the cell (all those of a lattice's elements).
    pub fills: Vec<i64>,
    pub lattice: bool,
    /// `IMP:N=0`: outside the problem.
    pub imp_zero: bool,
    /// Why the cell cannot be evaluated, if it uses something unsupported.
    pub unsupported: Option<String>,
    /// Conservative bounding box in the universe's frame (None if empty).
    pub bbox: Option<(Xyz, Xyz)>,
}

impl Cell {
    fn new(m: &llevant::Model, idx: u32) -> Cell {
        let c = m.cell(idx);
        let mut fills: Vec<i64> = m
            .fill_targets(idx)
            .map(|t| m.units()[t.unit() as usize].universe)
            .collect();
        fills.sort_unstable();
        fills.dedup();
        Cell {
            id: c.id,
            universe: m.units()[c.unit as usize].universe,
            material: c.material,
            density: c.density,
            fills,
            lattice: matches!(c.fill, Some(Fill::Lattice(_))),
            imp_zero: c.imp_zero,
            unsupported: c.unsupported.as_deref().map(str::to_string),
            bbox: bbox(&c.bbox),
        }
    }
}

#[pymethods]
impl Cell {
    fn __repr__(&self) -> String {
        let mut s = format!("<Cell {}", self.id);
        if self.universe != 0 {
            s += &format!(" u={}", self.universe);
        }
        if self.material == 0 {
            s += " void";
        } else {
            s += &format!(" m={} rho={}", self.material, self.density);
        }
        if !self.fills.is_empty() {
            let f: Vec<String> = self.fills.iter().map(|u| u.to_string()).collect();
            s += &format!(
                " {}={}",
                if self.lattice { "lattice" } else { "fill" },
                f.join(",")
            );
        }
        if self.imp_zero {
            s += " imp:n=0";
        }
        if self.unsupported.is_some() {
            s += " unsupported";
        }
        s + ">"
    }
}

/// The cells of a model by MCNP number: `cells[304]`, `304 in cells`,
/// `len(cells)`, iteration over the numbers in card order.
#[pyclass(frozen, module = "llevant")]
pub struct Cells(pub Arc<llevant::Model>);

impl Cells {
    fn ids(&self) -> Vec<i64> {
        self.0.cells().iter().map(|c| c.id).collect()
    }
}

#[pymethods]
impl Cells {
    /// Number of cells.
    fn __len__(&self) -> usize {
        self.0.cells().len()
    }

    /// Whether the model has cell `id` (MCNP number).
    fn __contains__(&self, id: i64) -> bool {
        self.0.cell_index(id).is_some()
    }

    /// Cell `id` (MCNP number); KeyError if there is none.
    fn __getitem__(&self, id: i64) -> PyResult<Cell> {
        let idx = self
            .0
            .cell_index(id)
            .ok_or_else(|| PyKeyError::new_err(id))?;
        Ok(Cell::new(&self.0, idx))
    }

    /// The cell numbers, in card order.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.ids())?
            .try_iter()
            .map(|i| i.into_any())
    }

    /// The cell numbers, in card order.
    fn keys(&self) -> Vec<i64> {
        self.ids()
    }

    /// The cells, in card order.
    fn values(&self) -> Vec<Cell> {
        (0..self.0.cells().len() as u32)
            .map(|i| Cell::new(&self.0, i))
            .collect()
    }

    /// `(number, cell)` pairs, in card order.
    fn items(&self) -> Vec<(i64, Cell)> {
        (0..self.0.cells().len() as u32)
            .map(|i| (self.0.cell_id(i), Cell::new(&self.0, i)))
            .collect()
    }

    /// Cell `id` (MCNP number), None if there is none.
    fn get(&self, id: i64) -> Option<Cell> {
        self.0.cell_index(id).map(|i| Cell::new(&self.0, i))
    }

    fn __repr__(&self) -> String {
        format!("<Cells: {}>", self.0.cells().len())
    }
}

/// Volume (and optional mass) of a cell, material, universe or label.
#[pyclass(frozen, get_all, module = "llevant")]
pub struct Group {
    /// The cell, material or universe number; None for a single label
    /// (gaps, overlaps... or `by="label"`).
    pub number: Option<i64>,
    /// The number, or the label in MCNP terms.
    pub name: String,
    pub status: &'static str,
    /// The label, for a group that is one label (no number); else None.
    pub label: Option<Py<Label>>,
    /// Mean of the probes' estimates (cm³).
    pub volume: f64,
    /// Standard error of `volume`, over the independent probes (cm³).
    pub sigma: f64,
    /// Estimated mass (g) when density and atomic mass are known, else None.
    pub mass: Option<f64>,
    /// Standard error of `mass` (g), else None.
    pub mass_sigma: Option<f64>,
    /// Estimate of each probe (0 for those that missed it).
    pub by_probe: Vec<f64>,
    /// Segments found; sigma is unreliable from a few.
    pub segments: u64,
    /// Probes with some line through it.
    pub probes_hit: usize,
    /// Spacing `h` of the grid that gave the volume.
    pub h: f64,
    /// From `Model.volumes_to`: whether it reached the target (else None).
    pub converged: Option<bool>,
    /// From `Model.volumes_to`: "converged", "not converged", "not found"
    /// (no line met it down to `h_min`) or "empty" (its boxes are empty:
    /// volume 0); None for groups not refined (gaps, overlaps).
    pub outcome: Option<&'static str>,
    /// Box around the segments found (None if none).
    pub bbox: Option<(Xyz, Xyz)>,
}

impl Group {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        m: &llevant::Model,
        number: Option<i64>,
        name: String,
        status: Status,
        label: Option<&llevant::Label>,
        v: &llevant::Volume,
        h: f64,
        mass: Option<(f64, f64)>,
    ) -> PyResult<Group> {
        Ok(Group {
            number,
            name,
            status: status.name(),
            label: label.map(|l| Py::new(py, Label::new(m, l))).transpose()?,
            volume: v.estimate,
            sigma: v.sigma,
            mass: mass.map(|x| x.0),
            mass_sigma: mass.map(|x| x.1),
            by_probe: v.by_probe.clone(),
            segments: v.segments,
            probes_hit: v.probes_hit(),
            h,
            converged: None,
            outcome: None,
            bbox: bbox(&v.bbox),
        })
    }

    /// A group of the engine's.
    pub fn of(
        py: Python<'_>,
        m: &llevant::Model,
        g: &llevant::Group,
        h: f64,
        mass: Option<(f64, f64)>,
    ) -> PyResult<Group> {
        Group::new(
            py,
            m,
            g.number,
            g.name.clone(),
            g.status,
            g.label.as_ref(),
            &g.volume,
            h,
            mass,
        )
    }

    /// The group of label `i` of `v` alone.
    pub fn of_label(
        py: Python<'_>,
        m: &llevant::Model,
        v: &llevant::Volumes,
        i: usize,
    ) -> PyResult<Group> {
        let l = &v.labels[i];
        Group::new(
            py,
            m,
            None,
            m.describe(l),
            l.status,
            Some(l),
            &v.volumes[i],
            v.grid.h,
            None,
        )
    }
}

#[pymethods]
impl Group {
    fn __repr__(&self) -> String {
        format!(
            "<Group {}: {:.6e} ± {:.1e} cm3>",
            self.name, self.volume, self.sigma
        )
    }
}

/// Volumes of everything a grid of lines met ([`Model.volumes`]).
#[pyclass(frozen, module = "llevant")]
pub struct Volumes {
    pub model: Arc<llevant::Model>,
    pub v: llevant::Volumes,
}

#[pymethods]
impl Volumes {
    /// Volumes by `cell`, `material`, `universe` or `label` (every path
    /// apart). Cells, materials and universes come first in number order,
    /// then gaps, overlaps and the like, one group per label.
    #[pyo3(signature = (by = "cell"))]
    fn by(&self, py: Python<'_>, by: &str) -> PyResult<Vec<Group>> {
        let g = crate::convert::group(by)?;
        let looks = Looks::new(&self.model, &[]).map_err(err)?;
        let groups = self.v.grouped(&self.model, g);
        let cell_groups = matches!(g, llevant::GroupBy::Material | llevant::GroupBy::Universe)
            .then(|| self.v.grouped(&self.model, llevant::GroupBy::Cell));
        groups
            .iter()
            .map(|grp| {
                let mass = group_mass(&self.model, &looks, g, grp, cell_groups.as_deref());
                Group::of(py, &self.model, grp, self.v.grid.h, mass)
            })
            .collect()
    }

    /// Gaps, overlaps, unsupported cells and too-deep nesting, largest
    /// first.
    fn problems(&self, py: Python<'_>) -> PyResult<Vec<Group>> {
        self.v
            .problems()
            .into_iter()
            .map(|i| Group::of_label(py, &self.model, &self.v, i))
            .collect()
    }

    /// Sum of all volumes: the region, up to the grid's edges.
    #[getter]
    fn total(&self) -> f64 {
        self.v.total()
    }

    /// Line spacing (cm).
    #[getter]
    fn h(&self) -> f64 {
        self.v.grid.h
    }

    /// The box measured, `((x0, y0, z0), (x1, y1, z1))` in cm.
    #[getter]
    fn region(&self) -> (Xyz, Xyz) {
        (xyz(self.v.grid.region.lo), xyz(self.v.grid.region.hi))
    }

    /// `by()` as a Polars data frame (needs `polars`).
    #[pyo3(signature = (by = "cell"))]
    fn to_polars<'py>(slf: &Bound<'py, Self>, by: &str) -> PyResult<Bound<'py, PyAny>> {
        crate::frames(slf.py(), "to_polars", slf.as_any(), Some(by))
    }

    /// `by()` as a pandas data frame (needs `pandas`).
    #[pyo3(signature = (by = "cell"))]
    fn to_pandas<'py>(slf: &Bound<'py, Self>, by: &str) -> PyResult<Bound<'py, PyAny>> {
        crate::frames(slf.py(), "to_pandas", slf.as_any(), Some(by))
    }

    fn __repr__(&self) -> String {
        format!(
            "<Volumes: {} labels, total {:.6e} cm3, h = {} cm>",
            self.v.labels.len(),
            self.v.total(),
            self.v.grid.h
        )
    }
}

/// Digests of every cell and universe that ignore numbering and card
/// order: `cells[n]` and `universes[n]` are `(geometry, content)` hex
/// digests.
#[pyclass(frozen, get_all, module = "llevant")]
pub struct Fingerprints {
    pub version: u32,
    pub precision: f64,
    pub cells: Py<PyDict>,
    pub universes: Py<PyDict>,
}

impl Fingerprints {
    pub fn new(py: Python<'_>, m: &llevant::Model, f: &llevant::Fingerprints) -> PyResult<Self> {
        let pair = |f: &llevant::Fingerprint| (f.geometry.to_string(), f.content.to_string());
        let cells = PyDict::new(py);
        for (c, f) in m.cells().iter().zip(&f.cells) {
            cells.set_item(c.id, pair(f))?;
        }
        let universes = PyDict::new(py);
        for (u, f) in m.units().iter().zip(&f.units) {
            universes.set_item(u.universe, pair(f))?;
        }
        Ok(Fingerprints {
            version: f.version,
            precision: f.precision,
            cells: cells.unbind(),
            universes: universes.unbind(),
        })
    }
}

#[pymethods]
impl Fingerprints {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "<Fingerprints v{}: {} cells, {} universes>",
            self.version,
            self.cells.bind(py).len(),
            self.universes.bind(py).len()
        )
    }
}

/// How the cells (or universes) of two models differ, by MCNP number.
#[pyclass(frozen, get_all, module = "llevant")]
pub struct Changes {
    /// Numbers in both models with equal digests.
    pub unchanged: usize,
    /// Same geometry, other materials or densities.
    pub content: Vec<i64>,
    /// Other geometry.
    pub geometry: Vec<i64>,
    /// `(from, to)`: gone from the first model, same content under a new
    /// number in the second.
    pub renumbered: Vec<(i64, i64)>,
    /// Only in the first model.
    pub removed: Vec<i64>,
    /// Only in the second model.
    pub new: Vec<i64>,
}

impl Changes {
    fn from(c: &llevant::Changes) -> Changes {
        let mut out = Changes {
            unchanged: c.unchanged,
            content: vec![],
            geometry: vec![],
            renumbered: vec![],
            removed: vec![],
            new: vec![],
        };
        for ch in &c.changes {
            match *ch {
                Change::Content(n) => out.content.push(n),
                Change::Geometry(n) => out.geometry.push(n),
                Change::Renumbered { from, to } => out.renumbered.push((from, to)),
                Change::Removed(n) => out.removed.push(n),
                Change::New(n) => out.new.push(n),
                _ => {}
            }
        }
        out
    }

    fn count(&self) -> usize {
        self.content.len()
            + self.geometry.len()
            + self.renumbered.len()
            + self.removed.len()
            + self.new.len()
    }
}

#[pymethods]
impl Changes {
    /// Number of cells (or universes) that changed, were renumbered,
    /// removed or are new.
    fn __len__(&self) -> usize {
        self.count()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Changes: {} unchanged, {} content, {} geometry, {} renumbered, {} removed, {} new>",
            self.unchanged,
            self.content.len(),
            self.geometry.len(),
            self.renumbered.len(),
            self.removed.len(),
            self.new.len()
        )
    }
}

/// What [`Model.diff`] finds: `cells` and `universes` changes. False when
/// the models are the same up to numbering; `str()` gives a report.
#[pyclass(frozen, module = "llevant")]
pub struct Diff {
    #[pyo3(get)]
    pub cells: Py<Changes>,
    #[pyo3(get)]
    pub universes: Py<Changes>,
    text: String,
}

impl Diff {
    pub fn new(py: Python<'_>, d: &llevant::FingerprintDiff) -> PyResult<Diff> {
        Ok(Diff {
            cells: Py::new(py, Changes::from(&d.cells))?,
            universes: Py::new(py, Changes::from(&d.universes))?,
            text: d.to_string(),
        })
    }
}

#[pymethods]
impl Diff {
    /// True when the models differ (beyond numbering).
    fn __bool__(&self) -> bool {
        self.cells.get().count() + self.universes.get().count() > 0
    }

    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "<Diff: {} cell changes, {} universe changes>",
            self.cells.get().count(),
            self.universes.get().count()
        )
    }
}
