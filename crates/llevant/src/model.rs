//! Loading and compiling an MCNP model.

use std::sync::Arc;
use std::time::Instant;

use migjorn::{ReadError, TransformSpec};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::bbox::{Aabb, Bounder};
use crate::bvh::Bvh;
use crate::classify::Label;
use crate::error::{Error, Result};
use crate::expr::{self, Resolve, SurfSlot, Tri};
use crate::grid::{self, Octree, Prog};
use crate::lattice::{self, Lattice, LatticeFill, Target};
use crate::progress::Progress;
use crate::recompile::{Base, Changed};
use crate::surface::{self, Shape, Side};
use crate::transform::Affine;
use crate::vec3::Vec3;

/// Options for compiling a model: start from `Options::default()` and
/// change what you need, with the `with_*` methods or by assigning fields.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// Points closer than this to a surface (in cm) are "on" it: their side
    /// is undecided rather than guessed.
    pub tolerance: f64,
    /// Half-size (cm) of the cube around the origin used as the region of
    /// universe 0 when bounding cells, unless its `IMP:N=0` cells leave a
    /// smaller one ([`Model::problem_box`]). Points outside the region are
    /// still classified: by checking every level-0 cell, or beyond
    /// the problem box its `IMP:N=0` cells.
    pub world_half_size: f64,
    /// When to build octrees of candidate cells (see [`GridMode`]).
    pub grid: GridMode,
    /// Whether octree leaves keep each candidate cell's program specialised
    /// to the leaf (residual programs: faster, more memory). Off is for A/B
    /// runs.
    pub residuals: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            tolerance: 1e-8,
            world_half_size: 1e6,
            grid: GridMode::Auto,
            residuals: true,
        }
    }
}

impl Options {
    /// See [`Options::tolerance`].
    pub fn with_tolerance(mut self, tolerance: f64) -> Options {
        self.tolerance = tolerance;
        self
    }

    /// See [`Options::world_half_size`].
    pub fn with_world_half_size(mut self, half_size: f64) -> Options {
        self.world_half_size = half_size;
        self
    }

    /// See [`Options::grid`].
    pub fn with_grid(mut self, grid: GridMode) -> Options {
        self.grid = grid;
        self
    }

    /// See [`Options::residuals`].
    pub fn with_residuals(mut self, residuals: bool) -> Options {
        self.residuals = residuals;
        self
    }
}

/// Which universes get an octree of candidate cells on top of their BVH.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GridMode {
    /// Universes where many cell boxes contain a typical point, and only if
    /// the octree at least halves the candidates.
    #[default]
    Auto,
    /// Every universe with bounded cells (for tests).
    Always,
    Off,
}

/// Where a compiled shape came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SurfaceId {
    /// MCNP surface number.
    pub id: i64,
    /// Macrobody facet number (1-based), 0 for an ordinary surface.
    pub facet: u8,
}

/// What fills a cell.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Fill {
    /// One universe, through a transform from the cell's frame.
    Universe(Target),
    /// A lattice: index into [`Model::lattices`], which holds what fills
    /// each element.
    Lattice(u32),
}

/// A compiled cell card.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Cell {
    /// MCNP cell number.
    pub id: i64,
    /// Index into [`Model::units`] of the universe the cell belongs to.
    pub unit: u32,
    /// Material number, 0 for void.
    pub material: i64,
    /// Density with the MCNP sign (negative: g/cm³), 0 for void.
    pub density: f64,
    pub(crate) code_start: usize,
    pub(crate) code_len: u32,
    /// Where the cell's node boxes start in [`Model::node_boxes`]: its box
    /// words (`NODE | k`) are local to the cell, so that one cell's boxes
    /// never renumber another's.
    pub(crate) node_start: u32,
    /// Conservative bounding box in the universe's frame.
    pub bbox: Aabb,
    pub fill: Option<Fill>,
    /// `IMP:N=0`: outside the problem.
    pub imp_zero: bool,
    /// Why the cell cannot be evaluated, if it uses something unsupported.
    pub unsupported: Option<Box<str>>,
}

/// The cells of one universe and their search structure.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Unit {
    /// MCNP universe number (0 for the real world).
    pub universe: i64,
    pub cells: Vec<u32>,
    /// Region of the universe that can be reached through the cells it fills.
    pub roi: Aabb,
    pub(crate) bvh: Bvh,
    /// Candidate cells by octree leaf, used instead of the BVH inside its root.
    /// Saved models write it on its own, after the rest (`saved.rs`). Shared,
    /// so that a completed or recompiled model can take it over unchanged.
    #[serde(skip)]
    pub(crate) grid: Option<Arc<Octree>>,
    /// The octree has not been decided yet (a staged model, see
    /// [`Model::compile_staged`]): queries use the BVH until
    /// [`Model::complete`] builds it.
    #[serde(skip)]
    pub(crate) pending: bool,
    /// Cells checked for every point: unbounded ones and lattice cells.
    pub(crate) always: Vec<u32>,
    /// The only cells that can hold a point beyond `roi`, when not every
    /// cell (unit 0 with a problem box: its `IMP:N=0` cells).
    pub(crate) outside: Option<Vec<u32>>,
}

impl Unit {
    /// The cells checked for a point beyond `roi`.
    pub(crate) fn outside(&self) -> &[u32] {
        self.outside.as_deref().unwrap_or(&self.cells)
    }
}

/// Which part of a model [`Model::bounds`] and [`Model::extent`] cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Level-0 cells that hold something: a material or a fill. For views.
    Contents,
    /// Every level-0 cell except `IMP:N=0` ones: where particles go, and so
    /// where gaps and overlaps matter. For volumes and checks.
    Problem,
}

/// What a [`Diagnostic`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum Subject {
    Cell(i64),
    Surface(i64),
    Material(i64),
    /// Another card of the input (one llevant does not read).
    Input,
}

/// A problem found while compiling (an unsupported feature, an undefined
/// reference, a duplicate card). It does not stop the compile: an affected
/// cell is compiled as unsupported.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Diagnostic {
    pub subject: Subject,
    pub message: String,
    /// Line of the card in the input (1-based), when known.
    pub line: Option<usize>,
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(l) => write!(f, "line {l}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// A compiled model: read-only, shared by all query threads.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Model {
    /// Compiled surfaces and macrobody facets, indexed by program leaves.
    pub(crate) shapes: Vec<Shape>,
    pub(crate) surface_ids: Vec<SurfaceId>,
    pub(crate) cells: Vec<Cell>,
    pub(crate) code: Vec<u32>,
    /// `[may-be-true, may-be-false]` boxes of large program nodes.
    pub(crate) node_boxes: Vec<[Aabb; 2]>,
    pub(crate) units: Vec<Unit>,
    pub(crate) lattices: Vec<Lattice>,
    pub(crate) cell_index: FxHashMap<i64, u32>,
    /// `Mn` cards by number: `(zaid, fraction)` as written (a negative
    /// fraction is by weight), and whether the card read cleanly.
    pub(crate) materials: FxHashMap<i64, (Vec<(String, f64)>, bool)>,
    pub(crate) tolerance: f64,
    /// The cube of [`Options::world_half_size`]: cell boxes that reach it
    /// are bounded by it, not by their own surfaces.
    pub(crate) world: Aabb,
    /// The cell beyond the box of a model cut by [`Model::universe`], and
    /// that box (its problem box exactly, not padded as bounds are).
    pub(crate) cut: Option<(u32, Aabb)>,
    pub(crate) diagnostics: Vec<Diagnostic>,
    /// How the octrees are built ([`Options::grid`],
    /// [`Options::residuals`]): pending ones by [`Model::complete`], and
    /// whether a recompile can reuse them.
    pub(crate) grid_options: (GridMode, bool),
    #[serde(skip)]
    pub(crate) timings: Vec<(&'static str, f64)>,
    /// The numbering of cell instances, made on first use from the rest.
    #[serde(skip)]
    pub(crate) instances: std::sync::OnceLock<crate::instance::Instances>,
}

/// Summary numbers for reports and benchmarks.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Stats {
    pub cells: usize,
    pub surfaces: usize,
    pub units: usize,
    pub lattices: usize,
    pub code_words: usize,
    pub unsupported_cells: usize,
    /// Cells left in the per-unit "always" lists (unbounded or lattice).
    pub always_cells: usize,
    pub max_bvh_depth: usize,
    pub largest_unit: usize,
    /// Universes with an octree, and the octrees' total nodes, candidate
    /// entries and largest depth.
    pub grid_units: usize,
    pub grid_nodes: usize,
    pub grid_items: usize,
    pub grid_bytes: usize,
    pub max_grid_depth: usize,
    /// Universes whose octree is still to build: those of a staged model
    /// ([`Model::compile_staged`]), none once it is complete.
    pub pending_units: usize,
}

/// Translate raw input bytes from OpenMC XML, TRIPOLI-4, or DRAGON5 (`GEO:`)
/// into canonical MCNP bytes, or borrow raw MCNP bytes untouched with zero allocation.
pub fn translate_input_bytes(input: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>> {
    if llevant_openmc::is_openmc_bytes(input) {
        return llevant_openmc::maybe_translate_bytes(input)
            .map_err(|e| Error::Invalid(e.to_string()));
    }
    if llevant_tripoli::is_tripoli_bytes(input) {
        return llevant_tripoli::maybe_translate_bytes(input)
            .map_err(|e| Error::Invalid(e.to_string()));
    }
    if llevant_dragon::is_dragon_bytes(input) {
        return llevant_dragon::maybe_translate_bytes(input)
            .map_err(|e| Error::Invalid(e.to_string()));
    }
    Ok(std::borrow::Cow::Borrowed(input))
}

/// Read an input file or OpenMC model directory (OpenMC XML, TRIPOLI-4,
/// DRAGON5 `GEO:`, or MCNP) and return canonical MCNP-compatible bytes.
pub fn read_input_bytes(path: &std::path::Path) -> Result<Vec<u8>> {
    if llevant_openmc::is_openmc_path(path) {
        return llevant_openmc::read_input_bytes(path).map_err(|e| match e {
            llevant_openmc::OpenMcError::Io { path, source } => Error::Io { path, source },
            other => Error::Invalid(other.to_string()),
        });
    }
    if llevant_tripoli::is_tripoli_path(path) {
        return llevant_tripoli::read_input_bytes(path).map_err(|e| match e {
            llevant_tripoli::TripoliError::Io { path, source } => Error::Io { path, source },
            other => Error::Invalid(other.to_string()),
        });
    }
    if llevant_dragon::is_dragon_path(path) {
        return llevant_dragon::read_input_bytes(path).map_err(|e| match e {
            llevant_dragon::DragonError::Io { path, source } => Error::Io { path, source },
            other => Error::Invalid(other.to_string()),
        });
    }
    let bytes = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    match translate_input_bytes(&bytes)? {
        std::borrow::Cow::Owned(v) => Ok(v),
        std::borrow::Cow::Borrowed(_) => Ok(bytes),
    }
}

impl Model {
    /// Read and compile an MCNP, OpenMC XML, TRIPOLI-4, or DRAGON5 input file.
    /// With `progress`, each step is named in it as it starts (parsing, then
    /// the steps of compiling), and loading stops with [`Error::Cancelled`]
    /// between steps once cancelled.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        opts: &Options,
        progress: Option<&Progress>,
    ) -> Result<Model> {
        let t = Instant::now();
        let path = path.as_ref();
        let bytes = read_input_bytes(path)?;
        let read = t.elapsed().as_secs_f64();
        let mut m = Model::compile(&bytes, opts, progress)?;
        m.timings.insert(0, ("read", read));
        Ok(m)
    }

    /// Compile the bytes of an MCNP input (read as UTF-8, invalid sequences
    /// replaced), with `progress` as [`Model::load`].
    pub fn compile(input: &[u8], opts: &Options, progress: Option<&Progress>) -> Result<Model> {
        let none = Progress::new();
        let progress = progress.unwrap_or(&none);
        let (mut m, _) = Model::compile_steps(input, opts, progress, 7, None)?;
        m.build_pending(progress)?;
        Ok(m)
    }

    /// Compile `input`, a new version of this model's input, reusing the
    /// cell boxes and octrees of every universe whose content is unchanged
    /// (see `recompile.rs` for exactly when). The result is staged:
    /// [`Model::complete`] builds the octrees of the universes rebuilt, and
    /// then it is exactly the model [`Model::compile`] gives. With other
    /// `Options`, or from a model made by [`Model::universe`], nothing is
    /// reused. Also returns what changed.
    pub fn recompile(
        &self,
        input: &[u8],
        opts: &Options,
        progress: Option<&Progress>,
    ) -> Result<(Model, Changed)> {
        let none = Progress::new();
        let (m, changes) =
            Model::compile_steps(input, opts, progress.unwrap_or(&none), 6, Some(self))?;
        Ok((m, changes.expect("a recompile reports its changes")))
    }

    /// [`Model::compile`] without the octrees: a staged model, complete but
    /// for them, that answers every query exactly (through its BVHs, more
    /// slowly). [`Model::complete`] builds them. Compiling staged and then
    /// completing gives exactly the model [`Model::compile`] does.
    pub fn compile_staged(
        input: &[u8],
        opts: &Options,
        progress: Option<&Progress>,
    ) -> Result<Model> {
        let none = Progress::new();
        Ok(Model::compile_steps(input, opts, progress.unwrap_or(&none), 6, None)?.0)
    }

    /// Whether every octree is built (see [`Model::compile_staged`]). Only a
    /// complete model can be saved.
    pub fn is_complete(&self) -> bool {
        self.units.iter().all(|u| !u.pending)
    }

    /// This model with its pending octrees built (the others, and everything
    /// else, shared or copied). With `progress`, stops with
    /// [`Error::Cancelled`] once cancelled.
    pub fn complete(&self, progress: Option<&Progress>) -> Result<Model> {
        let none = Progress::new();
        let progress = progress.unwrap_or(&none);
        progress.start(1);
        let mut m = self.clone_shared();
        m.build_pending(progress)?;
        Ok(m)
    }

    /// [`Model::complete`], consuming this model to build in place.
    pub fn into_complete(mut self, progress: Option<&Progress>) -> Result<Model> {
        let none = Progress::new();
        let progress = progress.unwrap_or(&none);
        progress.start(1);
        self.build_pending(progress)?;
        Ok(self)
    }

    /// Build the pending octrees in place.
    fn build_pending(&mut self, progress: &Progress) -> Result<()> {
        if self.is_complete() {
            return Ok(());
        }
        let t = Instant::now();
        progress.set_stage("octrees");
        let (mode, residuals) = self.grid_options;
        build_grids(self, mode, residuals, progress)?;
        self.timings.push(("grid", t.elapsed().as_secs_f64()));
        progress.step();
        Ok(())
    }

    /// A copy sharing the octrees; the instance numbering is made again on
    /// first use.
    pub(crate) fn clone_shared(&self) -> Model {
        Model {
            shapes: self.shapes.clone(),
            surface_ids: self.surface_ids.clone(),
            cells: self.cells.clone(),
            code: self.code.clone(),
            node_boxes: self.node_boxes.clone(),
            units: self.units.clone(),
            lattices: self.lattices.clone(),
            cell_index: self.cell_index.clone(),
            materials: self.materials.clone(),
            tolerance: self.tolerance,
            world: self.world,
            cut: self.cut,
            diagnostics: self.diagnostics.clone(),
            grid_options: self.grid_options,
            timings: self.timings.clone(),
            instances: Default::default(),
        }
    }

    /// Parse and compile up to the BVHs, as the first steps of `steps` in
    /// `progress` (the last one, if any, is the octrees), reusing what can
    /// be of `old`.
    fn compile_steps(
        input: &[u8],
        opts: &Options,
        progress: &Progress,
        steps: usize,
        old: Option<&Model>,
    ) -> Result<(Model, Option<Changed>)> {
        // Parsing, five steps of compiling, and the octrees.
        progress.start(steps);
        let stop = || -> Result<()> {
            if progress.cancelled() {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        };
        stop()?;
        progress.set_stage("parsing");
        let translated = translate_input_bytes(input)?;
        let text = String::from_utf8_lossy(&translated);
        let t = Instant::now();
        let parsed = migjorn::Model::parse(&text);
        let parse = t.elapsed().as_secs_f64();
        progress.step();
        stop()?;
        let m = Compiler::new(&parsed, opts).run_with(progress, old);
        // Freeing the parse tree takes most of a second on large models.
        std::thread::spawn(move || drop(parsed));
        let (mut m, changes) = m?;
        m.timings.insert(0, ("parse", parse));
        Ok((m, changes))
    }

    /// [`Model::recompile`] from an already parsed input (a migjorn tree
    /// kept between edits, so nothing is parsed again). Gives exactly the
    /// model `recompile` gives for `src.to_source()`.
    pub fn recompile_parsed(
        &self,
        src: &migjorn::Model,
        opts: &Options,
        progress: Option<&Progress>,
    ) -> Result<(Model, Changed)> {
        let none = Progress::new();
        let progress = progress.unwrap_or(&none);
        // Five steps of compiling (no parsing).
        progress.start(5);
        if progress.cancelled() {
            return Err(Error::Cancelled);
        }
        let (m, changes) = Compiler::new(src, opts).run_with(progress, Some(self))?;
        Ok((m, changes.expect("a recompile reports its changes")))
    }

    /// Compile MCNP, OpenMC XML, TRIPOLI-4, or DRAGON5 input text. It never
    /// fails: what is rejected or unsupported is listed in [`Model::diagnostics`].
    /// [`Model::compile`] does the same from bytes, and can be cancelled.
    pub fn from_text(text: &str, opts: &Options) -> Model {
        let t = Instant::now();
        let translated = translate_input_bytes(text.as_bytes());
        let parsed = match &translated {
            Ok(std::borrow::Cow::Owned(bytes)) => {
                migjorn::Model::parse(&String::from_utf8_lossy(bytes))
            }
            _ => migjorn::Model::parse(text),
        };
        let parse = t.elapsed().as_secs_f64();
        let mut m = Model::from_migjorn(&parsed, opts);
        m.timings.insert(0, ("parse", parse));
        // Freeing the parse tree takes most of a second on large models.
        std::thread::spawn(move || drop(parsed));
        m
    }

    /// Compile an already parsed model.
    pub fn from_migjorn(src: &migjorn::Model, opts: &Options) -> Model {
        Compiler::new(src, opts).run()
    }

    #[inline]
    pub(crate) fn code(&self, cell: u32) -> &[u32] {
        let c = &self.cells[cell as usize];
        &self.code[c.code_start..c.code_start + c.code_len as usize]
    }

    /// The node boxes a cell's box words (`NODE | k`) index.
    #[inline]
    pub(crate) fn node_boxes_of(&self, cell: u32) -> &[[Aabb; 2]] {
        &self.node_boxes[self.cells[cell as usize].node_start as usize..]
    }

    /// Universes a cell hands points to (none if it is not filled).
    pub fn fill_targets(&self, cell: u32) -> impl Iterator<Item = &Target> + '_ {
        let (one, lattice) = match &self.cells[cell as usize].fill {
            None => (None, None),
            Some(Fill::Universe(t)) => (Some(t), None),
            Some(Fill::Lattice(l)) => (None, Some(&self.lattices[*l as usize])),
        };
        one.into_iter()
            .chain(lattice.into_iter().flat_map(|l| l.targets()))
    }

    pub(crate) fn is_lattice(&self, cell: u32) -> bool {
        matches!(self.cells[cell as usize].fill, Some(Fill::Lattice(_)))
    }

    /// Index of the cell with MCNP number `id`.
    pub fn cell_index(&self, id: i64) -> Option<u32> {
        self.cell_index.get(&id).copied()
    }

    /// All cells; label paths and [`Visibility`](crate::Visibility) index
    /// into this.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// The cell at index `idx` (as in label paths).
    pub fn cell(&self, idx: u32) -> &Cell {
        &self.cells[idx as usize]
    }

    /// MCNP number of the cell at index `idx`.
    pub fn cell_id(&self, idx: u32) -> i64 {
        self.cells[idx as usize].id
    }

    /// Universes, one unit each; [`Cell::unit`] indexes into this.
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// Lattices; [`Fill::Lattice`] indexes into this.
    pub fn lattices(&self) -> &[Lattice] {
        &self.lattices
    }

    /// Compiled surfaces (macrobodies become one per facet, `TRCL` cells
    /// get moved copies); segment ends index into this. For benchmarks
    /// (`bench` feature).
    #[cfg(feature = "bench")]
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }

    /// Number of compiled surfaces.
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// MCNP surface (and macrobody facet) of a compiled surface, as given
    /// at segment ends; `None` for [`NO_SURFACE`](crate::NO_SURFACE).
    pub fn surface_id(&self, shape: u32) -> Option<SurfaceId> {
        self.surface_ids.get(shape as usize).copied()
    }

    /// Material card `Mn` as written: `(zaid, fraction)` pairs, a negative
    /// fraction being by weight. `None` if there is no such card.
    pub fn material(&self, id: i64) -> Option<&[(String, f64)]> {
        self.materials.get(&id).map(|(e, _)| e.as_slice())
    }

    /// Numbers of the material cards, ascending.
    pub fn material_ids(&self) -> Vec<i64> {
        let mut ids: Vec<i64> = self.materials.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Universe 0's problem box, when its `IMP:N=0` cells bound one: the
    /// box of what they leave (usually the box of the graveyard sphere).
    /// Beyond it only those cells are checked, so overlaps with other
    /// cells out there go unreported.
    pub fn problem_box(&self) -> Option<Aabb> {
        self.units[0].outside.is_some().then_some(self.units[0].roi)
    }

    /// Box around part of the model: the union of the boxes of the
    /// level-0 cells in `scope` that are bounded (by their own surfaces, not
    /// by the world box) and not `IMP:N=0`. With [`Scope::Contents`], void
    /// cells (often a far sphere around everything) are left out, unless no
    /// cell holds anything. Cell boxes are conservative and can be far
    /// larger than their cells; [`Model::extent`] measures a tighter box.
    /// Empty if no cell qualifies.
    pub fn bounds(&self, scope: Scope) -> Aabb {
        let world = self.world;
        let bounded = |c: &&Cell| {
            !c.imp_zero
                && !c.bbox.is_empty()
                && (0..3).all(|i| c.bbox.lo[i] > world.lo[i] && c.bbox.hi[i] < world.hi[i])
        };
        let cells = || {
            self.units[0]
                .cells
                .iter()
                .map(|&c| &self.cells[c as usize])
                .filter(bounded)
        };
        let all = || cells().fold(Aabb::EMPTY, |a, c| a.union(&c.bbox));
        match scope {
            Scope::Problem => all(),
            Scope::Contents => {
                let full = cells()
                    .filter(|c| c.material != 0 || c.fill.is_some())
                    .fold(Aabb::EMPTY, |a, c| a.union(&c.bbox));
                if full.is_empty() {
                    all()
                } else {
                    full
                }
            }
        }
    }

    /// Distance (cm) within which a point is on a surface.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Problems found while compiling.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Wall time of each step of making this model (reading, parsing,
    /// compiling; or reading a saved model), in seconds.
    pub fn timings(&self) -> &[(&'static str, f64)] {
        &self.timings
    }

    /// [`Model::timings`], for frontends that load models (from a cache,
    /// say) to record their own steps.
    #[doc(hidden)]
    pub fn timings_mut(&mut self) -> &mut Vec<(&'static str, f64)> {
        &mut self.timings
    }

    /// A label in MCNP terms: its status, then the path as cell numbers
    /// from level 0 down (lattice elements in brackets), then the cells
    /// involved for overlaps and the like, e.g. `inside 12 > 304[1,-2,0] > 7`.
    pub fn describe(&self, label: &Label) -> String {
        let mut out = label.status.name().replace('-', " ");
        for (k, s) in label.path.iter().enumerate() {
            out.push_str(if k == 0 { " " } else { " > " });
            out.push_str(&self.cell_id(s.cell).to_string());
            if let Some([i, j, l]) = s.lattice {
                out.push_str(&format!("[{i},{j},{l}]"));
            }
        }
        if !label.involved.is_empty() {
            let ids: Vec<String> = label
                .involved
                .iter()
                .map(|&c| self.cell_id(c).to_string())
                .collect();
            out.push_str(&format!(" ({})", ids.join(", ")));
        }
        out
    }

    /// Sizes of the compiled model and its search structures.
    pub fn stats(&self) -> Stats {
        Stats {
            cells: self.cells.len(),
            surfaces: self.shapes.len(),
            units: self.units.len(),
            lattices: self.lattices.len(),
            code_words: self.code.len(),
            unsupported_cells: self
                .cells
                .iter()
                .filter(|c| c.unsupported.is_some())
                .count(),
            always_cells: self.units.iter().map(|u| u.always.len()).sum(),
            max_bvh_depth: self.units.iter().map(|u| u.bvh.depth()).max().unwrap_or(0),
            largest_unit: self.units.iter().map(|u| u.cells.len()).max().unwrap_or(0),
            grid_units: self.units.iter().filter(|u| u.grid.is_some()).count(),
            pending_units: self.units.iter().filter(|u| u.pending).count(),
            grid_nodes: self
                .units
                .iter()
                .flat_map(|u| &u.grid)
                .map(|g| g.node_count())
                .sum(),
            grid_items: self
                .units
                .iter()
                .flat_map(|u| &u.grid)
                .map(|g| g.item_count())
                .sum(),
            grid_bytes: self
                .units
                .iter()
                .flat_map(|u| &u.grid)
                .map(|g| g.bytes())
                .sum(),
            max_grid_depth: self
                .units
                .iter()
                .flat_map(|u| &u.grid)
                .map(|g| g.depth())
                .max()
                .unwrap_or(0),
        }
    }
}

/// What one cell card compiles to before units are known.
struct RawCell {
    id: i64,
    universe: i64,
    material: i64,
    density: f64,
    fill: Option<RawFill>,
    /// `LAT` value (0 if none).
    lattice: u8,
    /// `TRCL`: map from the main frame to the frame the cell is written in.
    trcl: Option<Affine>,
    imp_zero: bool,
    code: Vec<u32>,
    /// Leaves (position in `code`) whose surface must be moved by a TRCL.
    pending: Vec<(usize, Affine)>,
    unsupported: Option<String>,
}

enum RawFill {
    Single(i64, Option<Affine>),
    Array {
        lo: [i32; 3],
        size: [i32; 3],
        entries: Vec<(i64, Option<Affine>)>,
    },
}

struct Compiler<'a> {
    src: &'a migjorn::Model,
    opts: &'a Options,
    transforms: FxHashMap<i64, Result<Affine, String>>,
    surf_slots: FxHashMap<i64, Result<SurfSlot, String>>,
    shapes: Vec<Shape>,
    surface_ids: Vec<SurfaceId>,
    diagnostics: Vec<Diagnostic>,
    timings: Vec<(&'static str, f64)>,
}

impl<'a> Compiler<'a> {
    fn new(src: &'a migjorn::Model, opts: &'a Options) -> Self {
        Compiler {
            src,
            opts,
            transforms: FxHashMap::default(),
            surf_slots: FxHashMap::default(),
            shapes: Vec::new(),
            surface_ids: Vec::new(),
            diagnostics: Vec::new(),
            timings: Vec::new(),
        }
    }

    fn lap(&mut self, name: &'static str, t: &mut Instant) {
        self.timings.push((name, t.elapsed().as_secs_f64()));
        *t = Instant::now();
    }

    fn run(self) -> Model {
        let progress = Progress::new();
        let (mut m, _) = self.run_with(&progress, None).expect("not cancelled");
        m.build_pending(&progress).expect("not cancelled");
        m
    }

    /// The staged model (octrees pending), naming each step in `progress` as
    /// it starts, and stopping between steps once cancelled. With `old`, a
    /// recompile: units unchanged from `old` take their boxes and octrees
    /// from it, and what changed is returned too.
    fn run_with(
        mut self,
        progress: &Progress,
        old: Option<&Model>,
    ) -> Result<(Model, Option<Changed>)> {
        let stop = || -> Result<()> {
            if progress.cancelled() {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        };
        let mut t = Instant::now();
        stop()?;
        progress.set_stage("surfaces");
        self.compile_transforms();
        self.compile_surfaces();
        self.lap("surfaces", &mut t);
        progress.step();
        stop()?;
        progress.set_stage("cells");
        let raw = self.compile_cells();
        self.lap("cells", &mut t);
        progress.step();
        stop()?;
        progress.set_stage("assembling");
        let mut model = self.assemble(raw);
        assert!(
            model.shapes.len() < crate::trace::MAX_SURFACES,
            "more than {} compiled surfaces",
            crate::trace::MAX_SURFACES
        );
        self.lap("assemble", &mut t);
        progress.step();
        stop()?;
        progress.set_stage("cell boxes");
        let base = old.and_then(|o| Base::new(o, &model, self.opts));
        let from = compute_boxes_with(&mut model, self.opts.world_half_size, base.as_ref());
        self.lap("boxes", &mut t);
        progress.step();
        stop()?;
        progress.set_stage("search trees");
        build_bvhs(&mut model);
        self.lap("bvh", &mut t);
        progress.step();
        model.grid_options = (self.opts.grid, self.opts.residuals);
        if self.opts.grid != GridMode::Off {
            for u in &mut model.units {
                u.pending = true;
            }
        }
        if let Some(base) = &base {
            for (u, ou) in from.iter().enumerate() {
                if let Some(ou) = ou {
                    base.take_grid(&mut model, u as u32, *ou);
                }
            }
        }
        let changes = old.map(|old| {
            let reused: Vec<bool> = from.iter().map(Option::is_some).collect();
            let c = base.as_ref().map_or_else(
                || Changed::everything(&model, old),
                |b| b.changes(&model, &reused),
            );
            self.lap("changes", &mut t);
            c
        });
        model.diagnostics.extend(self.diagnostics);
        place_diagnostics(self.src, &mut model.diagnostics);
        model.timings = self.timings;
        Ok((model, changes))
    }

    fn compile_transforms(&mut self) {
        for tr in self.src.transforms() {
            let Some(id) = tr.id() else { continue };
            let res = tr
                .affine()
                .map(Affine::from)
                .map_err(|e| format!("TR{id}: {}", e.message));
            self.transforms.insert(id, res);
        }
    }

    fn compile_surfaces(&mut self) {
        let slots: Vec<u32> = self.src.surfaces().map(|s| s.slot()).collect();
        let built: Vec<(i64, bool, Result<Vec<Shape>, String>)> = slots
            .par_iter()
            .filter_map(|&slot| {
                let v = self.src.surface_at(slot)?;
                let id = v.id()?;
                let is_macro = v
                    .mnemonic()
                    .is_some_and(|k| surface::macrobody_facets(&k.to_ascii_uppercase()).is_some());
                let res = (|| {
                    let kind = v.mnemonic().ok_or("surface without a type")?;
                    let coeffs = v.coeffs().map_err(|e| e.message)?;
                    // A negative number makes the surface periodic with surface
                    // |n|: that pairing only matters to transport, and the
                    // surface itself is not transformed.
                    let tr = match v.transform().filter(|&n| n > 0) {
                        None => None,
                        Some(n) => match self.transforms.get(&n) {
                            Some(Ok(t)) => Some(t),
                            Some(Err(e)) => return Err(e.clone()),
                            None => return Err(format!("TR{n} is not defined")),
                        },
                    };
                    build_surface(kind, &coeffs, tr)
                })()
                .map_err(|e| format!("surface {id}: {e}"));
                Some((id, is_macro, res))
            })
            .collect();
        let n: usize = built
            .iter()
            .map(|(_, _, r)| r.as_ref().map_or(0, |v| v.len()))
            .sum();
        self.shapes.reserve(n);
        self.surface_ids.reserve(n);
        for (id, is_macro, res) in built {
            if self.surf_slots.contains_key(&id) {
                self.diagnostics.push(Diagnostic {
                    subject: Subject::Surface(id),
                    message: format!("surface {id} is defined twice; the first is used"),
                    line: None,
                });
                continue;
            }
            let slot = match res {
                Ok(shapes) => {
                    let first = self.shapes.len() as u32;
                    let n = shapes.len();
                    for (k, shape) in shapes.into_iter().enumerate() {
                        self.shapes.push(shape);
                        self.surface_ids.push(SurfaceId {
                            id,
                            facet: if is_macro { k as u8 + 1 } else { 0 },
                        });
                    }
                    if is_macro {
                        Ok(SurfSlot::Macro {
                            first,
                            count: n as u8,
                        })
                    } else {
                        Ok(SurfSlot::Single(first))
                    }
                }
                Err(e) => {
                    self.diagnostics.push(Diagnostic {
                        subject: Subject::Surface(id),
                        message: e.clone(),
                        line: None,
                    });
                    Err(e)
                }
            };
            self.surf_slots.insert(id, slot);
        }
    }

    fn compile_cells(&self) -> Vec<RawCell> {
        let slots: Vec<u32> = self.src.cells().map(|c| c.slot()).collect();
        slots
            .par_iter()
            .filter_map(|&slot| {
                let v = self.src.cell_at(slot)?;
                Some(self.compile_cell(v))
            })
            .collect()
    }

    /// The chain `[cell, base, base of base, …]` of a `LIKE n BUT` cell.
    fn like_chain(&self, v: migjorn::Cell<'a>) -> Result<Vec<migjorn::Cell<'a>>, String> {
        let mut chain = vec![v];
        while let Some(base) = chain.last().unwrap().like() {
            if chain.len() > 64 {
                return Err("LIKE BUT chain too long or cyclic".into());
            }
            let b = self
                .src
                .cell(base)
                .ok_or_else(|| format!("LIKE {base} BUT: cell {base} is not defined"))?;
            chain.push(b);
        }
        Ok(chain)
    }

    fn compile_cell(&self, v: migjorn::Cell<'a>) -> RawCell {
        let id = v.id().unwrap_or(0);
        let mut out = RawCell {
            id,
            universe: 0,
            material: 0,
            density: 0.0,
            fill: None,
            lattice: 0,
            trcl: None,
            imp_zero: false,
            code: Vec::new(),
            pending: Vec::new(),
            unsupported: None,
        };
        if let Err(e) = self.read_cell(v, &mut out) {
            out.unsupported = Some(format!("cell {id}: {e}"));
        }
        out
    }

    fn read_cell(&self, v: migjorn::Cell<'a>, out: &mut RawCell) -> Result<(), String> {
        let chain = self.like_chain(v)?;
        let base = *chain.last().unwrap();
        out.material = base.material().unwrap_or(0);
        out.density = if out.material == 0 {
            0.0
        } else {
            base.density().unwrap_or(0.0)
        };

        // Parameters: the first cell in the chain that sets one wins. A value
        // migjorn cannot read makes the cell unsupported.
        if let Some(m) = first(&chain, |c| c.material_override())? {
            out.material = m;
            if m == 0 {
                out.density = 0.0;
            }
        }
        if let Some(rho) = first(&chain, |c| c.density_override())? {
            out.density = rho;
        }
        if let Some(u) = first(&chain, |c| c.universe())? {
            out.universe = u.abs();
        }
        // Neutron importance: the first `IMP` entry along the chain whose
        // particle list names `n` (one value for every particle it lists).
        if let Some(x) = first(&chain, |c| c.importance("n"))? {
            out.imp_zero = x == 0.0;
        }
        out.trcl = self.trcl(&chain).map_err(|e| format!("TRCL: {e}"))?;
        if let Some(lat) = first(&chain, |c| c.lattice())? {
            out.lattice = lat;
        }
        if let Some(fill) = first(&chain, |c| c.fill())? {
            let tr =
                |t: &Option<TransformSpec>| t.as_ref().map(|t| self.transform_spec(t)).transpose();
            out.fill = Some(match fill {
                migjorn::Fill::Single {
                    universe,
                    transform,
                } => RawFill::Single(universe, tr(&transform)?),
                migjorn::Fill::Array { ranges, entries } => RawFill::Array {
                    lo: ranges.map(|r| r.0),
                    size: ranges.map(|r| r.1 - r.0 + 1),
                    entries: entries
                        .iter()
                        .map(|e| Ok((e.universe.abs(), tr(&e.transform)?)))
                        .collect::<Result<_, String>>()?,
                },
            });
        }

        let ast = self.parse_geometry(base)?;
        expr::lower(&ast, self, out.trcl, &mut out.code, &mut out.pending)?;
        Ok(())
    }

    /// The `TRCL` of a `LIKE n BUT` chain: the first cell that sets one.
    fn trcl(&self, chain: &[migjorn::Cell<'a>]) -> Result<Option<Affine>, String> {
        first(chain, |c| c.trcl())?
            .map(|t| self.transform_spec(&t))
            .transpose()
    }

    /// A transform as written on a cell: a `TRn` number or inline values
    /// (in degrees for `*TRCL` / `*FILL`).
    fn transform_spec(&self, t: &TransformSpec) -> Result<Affine, String> {
        match t {
            TransformSpec::Number(n) => match self.transforms.get(n) {
                Some(Ok(t)) => Ok(*t),
                Some(Err(e)) => Err(e.clone()),
                None => Err(format!("TR{n} is not defined")),
            },
            TransformSpec::Inline { .. } => {
                t.affine(self.src).map(Affine::from).map_err(|e| e.message)
            }
        }
    }

    fn parse_geometry(&self, base: migjorn::Cell<'a>) -> Result<migjorn::Expr, String> {
        base.geometry_expr().map_err(|e| e.message)
    }

    fn materials(&mut self) -> FxHashMap<i64, (Vec<(String, f64)>, bool)> {
        let mut out = FxHashMap::default();
        for m in self.src.materials() {
            let Some(id) = m.id() else { continue };
            if out.contains_key(&id) {
                self.diagnostics.push(Diagnostic {
                    subject: Subject::Material(id),
                    message: format!("material {id} is defined twice; the first is used"),
                    line: None,
                });
                continue;
            }
            let entries = m.entries().into_iter().map(|(z, f)| (z.to_owned(), f));
            out.insert(id, (entries.collect(), m.well_formed()));
        }
        out
    }

    /// Turn raw cells into the model: assign units, resolve fills and lattices.
    fn assemble(&mut self, raw: Vec<RawCell>) -> Model {
        let mut universes: Vec<i64> = raw.iter().map(|c| c.universe).filter(|&u| u != 0).collect();
        universes.sort_unstable();
        universes.dedup();
        let mut units = vec![Unit {
            universe: 0,
            cells: Vec::new(),
            roi: Aabb::INFINITE,
            bvh: Bvh::default(),
            grid: None,
            pending: false,
            always: Vec::new(),
            outside: None,
        }];
        let mut unit_of: FxHashMap<i64, u32> = FxHashMap::default();
        unit_of.insert(0, 0);
        for u in universes {
            unit_of.insert(u, units.len() as u32);
            units.push(Unit {
                universe: u,
                cells: Vec::new(),
                roi: Aabb::INFINITE,
                bvh: Bvh::default(),
                grid: None,
                pending: false,
                always: Vec::new(),
                outside: None,
            });
        }

        let total: usize = raw.iter().map(|c| c.code.len()).sum();
        let mut code = Vec::with_capacity(total);
        let mut cells = Vec::with_capacity(raw.len());
        let mut cell_index = FxHashMap::default();
        let mut lattices = Vec::new();
        // Transformed copies of surfaces for TRCL cells, shared between
        // cells that move the same surface the same way.
        let mut moved: FxHashMap<(u32, [u64; 12]), u32> = FxHashMap::default();
        for (i, mut rc) in raw.into_iter().enumerate() {
            if cell_index.insert(rc.id, i as u32).is_some() {
                self.diagnostics.push(Diagnostic {
                    subject: Subject::Cell(rc.id),
                    message: format!("cell {} is defined twice", rc.id),
                    line: None,
                });
            }
            let unit = unit_of[&rc.universe];
            units[unit as usize].cells.push(i as u32);
            for &(pos, t) in &rc.pending {
                let w = rc.code[pos];
                let base = w & expr::MASK;
                let key = (base, transform_key(&t));
                let idx = *moved.entry(key).or_insert_with(|| {
                    self.shapes.push(self.shapes[base as usize].to_parent(&t));
                    self.surface_ids.push(self.surface_ids[base as usize]);
                    self.shapes.len() as u32 - 1
                });
                rc.code[pos] = (w & !expr::MASK) | idx;
            }
            let mut unsupported = rc.unsupported.take();
            let start = code.len();
            code.extend_from_slice(&rc.code);
            let mut fill = None;
            if unsupported.is_none() {
                match self.resolve_fill(&rc, &unit_of) {
                    Ok(None) => {}
                    Ok(Some(Fill::Lattice(_))) => unreachable!(),
                    Ok(Some(f)) if rc.lattice == 0 => fill = Some(f),
                    Ok(Some(Fill::Universe(_))) => {}
                    Err(e) => unsupported = Some(format!("cell {}: {e}", rc.id)),
                }
                if rc.lattice != 0 && unsupported.is_none() {
                    match self
                        .lattice_fill(&rc, &unit_of)
                        .and_then(|lf| lattice::build(rc.lattice, &rc.code, &self.shapes, lf))
                    {
                        Ok(l) => {
                            lattices.push(l);
                            fill = Some(Fill::Lattice(lattices.len() as u32 - 1));
                        }
                        Err(e) => unsupported = Some(format!("cell {}: {e}", rc.id)),
                    }
                }
            }
            if let Some(e) = &unsupported {
                self.diagnostics.push(Diagnostic {
                    subject: Subject::Cell(rc.id),
                    message: e.clone(),
                    line: None,
                });
            }
            cells.push(Cell {
                id: rc.id,
                unit,
                material: rc.material,
                density: rc.density,
                code_start: start,
                code_len: rc.code.len() as u32,
                node_start: 0,
                bbox: Aabb::INFINITE,
                fill,
                imp_zero: rc.imp_zero,
                unsupported: unsupported.map(String::into_boxed_str),
            });
        }

        let mut model = Model {
            shapes: std::mem::take(&mut self.shapes),
            surface_ids: std::mem::take(&mut self.surface_ids),
            cells,
            code,
            node_boxes: Vec::new(),
            units,
            lattices,
            cell_index,
            materials: self.materials(),
            tolerance: self.opts.tolerance,
            world: Aabb::EMPTY,
            cut: None,
            diagnostics: Vec::new(),
            grid_options: (GridMode::Auto, true),
            timings: Vec::new(),
            instances: Default::default(),
        };
        break_fill_cycles(&mut model, &mut self.diagnostics);
        model
    }
}

impl Compiler<'_> {
    /// A fill target: the universe's unit and the transform from the cell's
    /// parent frame. A TRCL moves the filling universe with the cell only
    /// when the fill has no transform of its own; a fill transform replaces
    /// it rather than composing with it (checked against MCNP 6.2).
    fn target(
        &self,
        universe: i64,
        transform: Option<Affine>,
        trcl: Option<Affine>,
        unit_of: &FxHashMap<i64, u32>,
    ) -> Result<Target, String> {
        let unit = *unit_of
            .get(&universe)
            .ok_or_else(|| format!("fill universe {universe} is not defined"))?;
        Ok(Target {
            unit,
            transform: transform.or(trcl),
        })
    }

    /// The fill of a non-lattice cell.
    fn resolve_fill(
        &self,
        rc: &RawCell,
        unit_of: &FxHashMap<i64, u32>,
    ) -> Result<Option<Fill>, String> {
        match &rc.fill {
            None => Ok(None),
            Some(RawFill::Single(u, t)) => {
                Ok(Some(Fill::Universe(self.target(*u, *t, rc.trcl, unit_of)?)))
            }
            Some(RawFill::Array { .. }) if rc.lattice == 0 => {
                Err("fill array on a cell that is not a lattice".into())
            }
            Some(RawFill::Array { .. }) => Ok(None),
        }
    }

    fn lattice_fill(
        &self,
        rc: &RawCell,
        unit_of: &FxHashMap<i64, u32>,
    ) -> Result<LatticeFill, String> {
        match &rc.fill {
            None => Err("LAT without FILL".into()),
            Some(RawFill::Single(u, t)) => {
                Ok(LatticeFill::Uniform(self.target(*u, *t, rc.trcl, unit_of)?))
            }
            Some(RawFill::Array { lo, size, entries }) => {
                let elements = entries
                    .iter()
                    .map(|(u, t)| {
                        // The lattice's own universe: the element is the
                        // lattice cell itself.
                        if *u == rc.universe {
                            Ok(None)
                        } else {
                            self.target(*u, *t, rc.trcl, unit_of).map(Some)
                        }
                    })
                    .collect::<Result<_, String>>()?;
                Ok(LatticeFill::Array {
                    lo: *lo,
                    size: *size,
                    elements,
                })
            }
        }
    }
}

/// Bit pattern of a transform, for de-duplicating moved surfaces.
fn transform_key(t: &Affine) -> [u64; 12] {
    let mut k = [0u64; 12];
    for i in 0..3 {
        for j in 0..3 {
            k[3 * i + j] = t.rot[i][j].to_bits();
        }
        k[9 + i] = t.disp[i].to_bits();
    }
    k
}

impl Resolve for Compiler<'_> {
    fn surface(&self, id: i64) -> Result<SurfSlot, String> {
        match self.surf_slots.get(&id) {
            Some(r) => r.clone(),
            None => Err(format!("surface {id} is not defined")),
        }
    }

    fn cell_expr(&self, id: i64) -> Result<(migjorn::Expr, Option<Affine>), String> {
        let v = self
            .src
            .cell(id)
            .ok_or_else(|| format!("#{id}: cell {id} is not defined"))?;
        let chain = self.like_chain(v)?;
        let trcl = self.trcl(&chain)?;
        Ok((self.parse_geometry(*chain.last().unwrap())?, trcl))
    }
}

/// MCNP rejects one-sheet cones and tori whose transform is skewed: their
/// axis must stay parallel to x, y or z. Mirror that rather than accept an
/// input MCNP would not run.
/// The compiled shapes of a surface card: its type, coefficients and
/// transform (`TRn`, n > 0), as the compiler builds them.
pub(crate) fn build_surface(
    kind: &str,
    coeffs: &[f64],
    tr: Option<&Affine>,
) -> Result<Vec<Shape>, String> {
    let is_macro = surface::macrobody_facets(&kind.to_ascii_uppercase()).is_some();
    let mut shapes = surface::build(kind, coeffs)?;
    if let Some(tr) = tr {
        if !is_macro {
            check_mcnp_skew(kind, &shapes, tr)?;
        }
        shapes = shapes.iter().map(|s| s.to_parent(tr)).collect();
    }
    Ok(shapes)
}

fn check_mcnp_skew(kind: &str, shapes: &[Shape], tr: &Affine) -> Result<(), String> {
    let axis_dir = match &shapes[0] {
        Shape::Cone(c) => Some(c.axis),
        Shape::Torus(t) => {
            let mut e = [0.0; 3];
            e[t.axis as usize] = 1.0;
            Some(e)
        }
        _ => None,
    };
    let Some(a) = axis_dir else { return Ok(()) };
    let g = tr.apply_dir_inv(a);
    let big = g.iter().filter(|x| x.abs() > 1e-6).count();
    if big > 1 {
        let what = if matches!(shapes[0], Shape::Torus(_)) {
            "torus"
        } else {
            "one-sheet cone"
        };
        return Err(format!(
            "{what} ({kind}) with a skewed transformation (MCNP rejects it)"
        ));
    }
    Ok(())
}

/// The first value a `LIKE n BUT` chain sets, `[cell, base, …]` in order.
fn first<'a, T>(
    chain: &[migjorn::Cell<'a>],
    read: impl Fn(&migjorn::Cell<'a>) -> Result<Option<T>, ReadError>,
) -> Result<Option<T>, String> {
    chain
        .iter()
        .find_map(|c| read(c).transpose())
        .transpose()
        .map_err(|e| e.message)
}

/// Report the cell-parameter data cards (and `READ` cards) llevant does not
/// read, and give every diagnostic the line of its card. Line numbers cost a
/// pass over the input, so they are only computed when there is something
/// to report.
fn place_diagnostics(src: &migjorn::Model, diags: &mut Vec<Diagnostic>) {
    let ignored: Vec<_> = src.cell_parameter_cards().collect();
    if diags.is_empty() && ignored.is_empty() {
        return;
    }
    for d in diags.iter_mut() {
        let slot = match d.subject {
            Subject::Cell(id) => src.cell(id).map(|v| v.slot()),
            Subject::Surface(id) => src.surface(id).map(|v| v.slot()),
            Subject::Material(id) => src.material(id).map(|v| v.slot()),
            Subject::Input => None,
        };
        d.line = slot.and_then(|s| src.line_of(s)).or(d.line);
    }
    for c in ignored {
        diags.push(Diagnostic {
            subject: Subject::Input,
            message: format!(
                "data card `{}` can set cell parameters (IMP, U, FILL, TRCL, LAT), which \
                 llevant reads only from cell cards: the geometry may differ from MCNP's",
                c.name().unwrap_or("#")
            ),
            line: src.line_of(c.slot()),
        });
    }
}

/// Mark cells whose fill closes a cycle of universes as unsupported.
fn break_fill_cycles(model: &mut Model, diags: &mut Vec<Diagnostic>) {
    let n = model.units.len();
    // Edges of each unit: (filling cell, filled unit).
    let edges: Vec<Vec<(u32, usize)>> = (0..n)
        .map(|u| {
            model.units[u]
                .cells
                .iter()
                .flat_map(|&c| model.fill_targets(c).map(move |t| (c, t.unit as usize)))
                .collect()
        })
        .collect();
    // 0 = unvisited, 1 = on stack, 2 = done
    let mut state = vec![0u8; n];
    let mut bad = Vec::new();
    for root in 0..n {
        if state[root] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        state[root] = 1;
        while let Some(&mut (u, ref mut k)) = stack.last_mut() {
            let Some(&(c, t)) = edges[u].get(*k) else {
                state[u] = 2;
                stack.pop();
                continue;
            };
            *k += 1;
            match state[t] {
                0 => {
                    state[t] = 1;
                    stack.push((t, 0));
                }
                1 => bad.push(c),
                _ => {}
            }
        }
    }
    bad.sort_unstable();
    bad.dedup();
    for c in bad {
        let cell = &mut model.cells[c as usize];
        let msg = format!("cell {}: fill creates a cycle of universes", cell.id);
        diags.push(Diagnostic {
            subject: Subject::Cell(cell.id),
            message: msg.clone(),
            line: None,
        });
        cell.fill = None;
        cell.unsupported = Some(msg.into_boxed_str());
    }
}

/// Units in an order where every unit comes after all units that fill it.
pub(crate) fn topo_units(model: &Model) -> Vec<u32> {
    let n = model.units.len();
    let mut indeg = vec![0usize; n];
    let edges: Vec<Vec<usize>> = (0..n)
        .map(|u| {
            let mut e: Vec<usize> = model.units[u]
                .cells
                .iter()
                .flat_map(|&c| model.fill_targets(c).map(|t| t.unit as usize))
                .collect();
            e.sort_unstable();
            e.dedup();
            e
        })
        .collect();
    for e in &edges {
        for &t in e {
            indeg[t] += 1;
        }
    }
    let mut ready: Vec<u32> = (0..n as u32).filter(|&u| indeg[u as usize] == 0).collect();
    ready.sort_by_key(|&u| std::cmp::Reverse(u));
    let mut order = Vec::with_capacity(n);
    while let Some(u) = ready.pop() {
        order.push(u);
        for &t in &edges[u as usize] {
            indeg[t] -= 1;
            if indeg[t] == 0 {
                ready.push(t as u32);
            }
        }
    }
    order
}

/// `(position in the cell program, may-be-true box, may-be-false box)`.
pub(crate) type NodeBox = (usize, Aabb, Aabb);

/// Program nodes at least this long get pruning boxes.
const NODE_BOX_MIN_WORDS: u32 = 16;

pub(crate) fn compute_boxes(model: &mut Model, world: f64) {
    compute_boxes_with(model, world, None);
}

/// [`compute_boxes`], taking the boxes of units unchanged from `base`'s old
/// model from it. Returns, for each unit, the old unit it took them from.
fn compute_boxes_with(model: &mut Model, world: f64, base: Option<&Base>) -> Vec<Option<u32>> {
    let mut from = vec![None; model.units.len()];
    model.world = Aabb {
        lo: [-world; 3],
        hi: [world; 3],
    };
    let shapes = std::mem::take(&mut model.shapes);
    let bounder = Bounder::new(&shapes);
    // Region reachable in each unit: unit 0 is its problem box, or else the
    // whole world; a filled universe only needs to cover what its filling
    // cells can hand it.
    let mut rois: Vec<Option<Aabb>> = vec![None; model.units.len()];
    let (region, outside) = match problem_box(model, &bounder) {
        Some((b, cells)) => (b, Some(cells)),
        None => (model.world, None),
    };
    rois[0] = Some(region);
    model.units[0].outside = outside;
    for u in topo_units(model) {
        let roi = rois[u as usize].unwrap_or(Aabb::INFINITE);
        model.units[u as usize].roi = roi;
        let cells = model.units[u as usize].cells.clone();
        let old = base.and_then(|b| Some((b, b.old_unit(model, u, &roi)?)));
        from[u as usize] = old.map(|o| o.1);
        let boxes: Vec<(Aabb, Vec<NodeBox>)> = if let Some((b, ou)) = old {
            b.boxes(ou)
        } else {
            let m = &*model;
            cells
                .par_iter()
                .map(|&c| {
                    let cell = &m.cells[c as usize];
                    if cell.unsupported.is_some() || m.is_lattice(c) {
                        (roi, Vec::new())
                    } else {
                        let code = m.code(c);
                        let b = bounder.bound(code, roi);
                        let nodes = if b.is_empty() {
                            Vec::new()
                        } else {
                            bounder.node_boxes(code, roi, NODE_BOX_MIN_WORDS)
                        };
                        (b, nodes)
                    }
                })
                .collect()
        };
        for (&c, (b, nodes)) in cells.iter().zip(boxes) {
            let cell = &mut model.cells[c as usize];
            cell.bbox = b;
            let start = cell.code_start;
            cell.node_start = model.node_boxes.len() as u32;
            for (k, (pos, t, f)) in nodes.into_iter().enumerate() {
                model.code[start + pos + 1] = expr::NODE | k as u32;
                model.node_boxes.push([t, f]);
            }
        }
        for &c in &cells {
            let region = match &model.cells[c as usize].fill {
                None => continue,
                Some(Fill::Universe(_)) => model.cells[c as usize].bbox,
                Some(Fill::Lattice(l)) => {
                    lattice_element_box(model, &bounder, c, &model.lattices[*l as usize], roi)
                }
            };
            for t in model.fill_targets(c) {
                let region = match &t.transform {
                    Some(tr) => region.transformed(tr),
                    None => region,
                };
                let slot = &mut rois[t.unit as usize];
                *slot = Some(match slot {
                    Some(r) => r.union(&region),
                    None => region,
                });
            }
        }
    }
    drop(bounder);
    model.shapes = shapes;
    from
}

/// Universe 0's problem box and its `IMP:N=0` cells: a point beyond the box
/// is in one of them, since it is beyond the box of the complement of one.
/// Beyond it only those cells are checked (an overlap with another cell out
/// there goes unreported: `docs/internals/engine.md`, "Deliberate semantics"). `None`
/// without such cells, or when their complements are not bounded.
fn problem_box(model: &Model, bounder: &Bounder) -> Option<(Aabb, Vec<u32>)> {
    let mut region = Aabb::INFINITE;
    let mut cells = Vec::new();
    for &c in &model.units[0].cells {
        let cell = &model.cells[c as usize];
        if !cell.imp_zero || cell.unsupported.is_some() || cell.fill.is_some() {
            continue;
        }
        let inside = expr::negate(model.code(c));
        region = region.intersect(&bounder.bound(&inside, Aabb::INFINITE));
        cells.push(c);
    }
    // The cut of `Model::universe` exactly: padded, its cell would overlap
    // the cells it cuts between its planes and the padding.
    if let Some((_, b)) = model.cut {
        region = region.intersect(&b);
    }
    (!cells.is_empty() && region.is_finite()).then_some((region, cells))
}

/// Box of lattice element `[0,0,0]` (where points land after the lattice
/// shift); unbounded directions take the extent of the lattice's own region
/// when they are axis-aligned.
pub(crate) fn lattice_element_box(
    model: &Model,
    bounder: &Bounder,
    c: u32,
    lat: &Lattice,
    roi: Aabb,
) -> Aabb {
    let mut b = bounder.bound(model.code(c), Aabb::INFINITE);
    if !lat.bounded[2] {
        let aligned = lat.basis[2].iter().filter(|x| x.abs() > 1e-12).count() == 1;
        if !aligned {
            return Aabb::INFINITE;
        }
    }
    for i in 0..3 {
        if !b.lo[i].is_finite() || !b.hi[i].is_finite() {
            b.lo[i] = roi.lo[i];
            b.hi[i] = roi.hi[i];
        }
    }
    b
}

pub(crate) fn build_bvhs(model: &mut Model) {
    let built: Vec<(Bvh, Vec<u32>)> = model
        .units
        .par_iter()
        .map(|unit| {
            let mut prims = Vec::new();
            let mut always = Vec::new();
            for &c in &unit.cells {
                let cell = &model.cells[c as usize];
                let is_lattice = model.is_lattice(c);
                if cell.bbox.is_empty() && !is_lattice && cell.unsupported.is_none() {
                    continue; // can never contain a point
                }
                if is_lattice || !cell.bbox.is_finite() {
                    always.push(c);
                } else {
                    prims.push((c, cell.bbox));
                }
            }
            (Bvh::build(&prims), always)
        })
        .collect();
    for (unit, (bvh, always)) in model.units.iter_mut().zip(built) {
        unit.bvh = bvh;
        unit.always = always;
    }
}

/// Octrees are only considered where at least this many cell boxes contain
/// a typical point, or where their programs come to `GRID_MIN_WORDS` words
/// (a few void cells written as the complement of everything around them,
/// which the octree's residual programs cut down to a few surfaces), and
/// kept if they at least halve the number of candidates, or cut the words
/// evaluated per point (with residual programs) by `GRID_WORD_GAIN`.
/// Keeping those that only halve the words doubles the reference model's
/// octrees (0.96 to 1.8 GB) for a mixed gain (measured 2026-10-18); a 10×
/// cut keeps the octree of a cell written as the complement of a thousand
/// others (`sparse.i`: 742 surface evaluations per point to 4.5).
const GRID_MIN_CANDIDATES: f64 = 8.0;
const GRID_MIN_WORDS: f64 = 256.0;
const GRID_WORD_GAIN: f64 = 10.0;
/// Budget per cell of an octree kept only for its residual programs. On the
/// reference model 64 keeps 7 more octrees for +0.7% octree memory (the full
/// budget kept 27 for +30%, 16 too few); `sparse.i` loses nothing.
const GRID_RESIDUAL_WORDS_PER_CELL: usize = 64;
const GRID_SAMPLES: usize = 2048;
/// Octree memory budget per universe, in u32 words. On the reference model
/// 1024 words per cell is the knee: 256 is 30% slower, 4096 no faster.
const GRID_WORDS_PER_CELL: usize = 1024;
const GRID_MAX_WORDS: usize = 1 << 24;

/// Build the octrees of the pending units (none with `GridMode::Off`),
/// largest first, stopping once `progress` is cancelled.
pub(crate) fn build_grids(
    model: &mut Model,
    mode: GridMode,
    residuals: bool,
    progress: &Progress,
) -> Result<()> {
    let mut todo: Vec<usize> = (0..model.units.len())
        .filter(|&u| model.units[u].pending && mode != GridMode::Off)
        .collect();
    todo.sort_by_key(|&u| std::cmp::Reverse(model.units[u].cells.len()));
    let m = &*model;
    let grids: Vec<Option<Octree>> = todo
        .par_iter()
        .map(|&u| {
            if progress.cancelled() {
                None
            } else {
                build_grid(m, u, mode, residuals)
            }
        })
        .collect();
    if progress.cancelled() {
        return Err(Error::Cancelled);
    }
    for (u, g) in todo.into_iter().zip(grids) {
        model.units[u].grid = g.map(Arc::new);
    }
    for u in &mut model.units {
        u.pending = false;
    }
    Ok(())
}

fn build_grid(m: &Model, u: usize, mode: GridMode, residuals: bool) -> Option<Octree> {
    let unit = &m.units[u];
    let cells: Vec<u32> = unit.bvh.items.clone();
    if cells.is_empty() {
        return None;
    }
    let boxes = &unit.bvh.boxes;
    // Sample points the way queries land in practice: inside cell boxes.
    // Seeded by the universe number, not the unit's index, so that a unit
    // whose content is unchanged gets the same octree when others are
    // added or removed (a recompile reuses it).
    let seed = (unit.universe as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    let mut rng = 0x9e3779b97f4a7c15u64 ^ seed;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let samples: Vec<Vec3> = (0..GRID_SAMPLES)
        .map(|_| {
            let b = boxes[(next() * boxes.len() as f64) as usize % boxes.len()];
            [0, 1, 2].map(|i| b.lo[i] + next() * (b.hi[i] - b.lo[i]))
        })
        .collect();
    let mut bvh_cands = 0usize;
    let mut bvh_words = 0usize;
    for &p in &samples {
        unit.bvh.query(p, |c| {
            bvh_cands += 1;
            bvh_words += m.code(c).len();
        });
    }
    let n = samples.len() as f64;
    if mode == GridMode::Auto
        && (bvh_cands as f64) < GRID_MIN_CANDIDATES * n
        && (bvh_words as f64) < GRID_MIN_WORDS * n
    {
        return None;
    }

    let root = boxes
        .iter()
        .fold(Aabb::EMPTY, |a, b| a.union(b))
        .intersect(&unit.roi);
    let mut edges: Vec<f64> = boxes
        .iter()
        .map(|b| {
            (0..3)
                .map(|a| b.hi[a] - b.lo[a])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    edges.sort_by(f64::total_cmp);
    let params = grid::Params {
        leaf_target: 2,
        min_edge: (edges[edges.len() / 2] / 4.0).max(1e-4),
        max_depth: 24,
        max_words: (GRID_WORDS_PER_CELL * cells.len()).clamp(1 << 20, GRID_MAX_WORDS),
        min_gain: if mode == GridMode::Always { 1.0 } else { 0.75 },
    };
    let tol = m.tolerance;
    let keep = |c: u32, b: &Aabb| {
        let cell = &m.cells[c as usize];
        if !cell.bbox.overlaps(b) {
            return false;
        }
        if cell.unsupported.is_some() {
            return true;
        }
        let v = expr::eval_box(m.code(c), b, m.node_boxes_of(c), &mut |s, want_pos| match (
            m.shapes[s as usize].side_over(b, tol),
            want_pos,
        ) {
            (Side::On, _) => Tri::Unknown,
            (Side::Pos, true) | (Side::Neg, false) => Tri::True,
            _ => Tri::False,
        });
        v != Tri::False
    };
    let mut tree = Octree::build(root, &cells, keep, &params);
    let mut fewer_cands = true;
    if mode == GridMode::Auto {
        let mut grid_cands = 0usize;
        for &p in &samples {
            match tree.leaf(p) {
                Some(l) => {
                    grid_cands += l
                        .iter()
                        .filter(|&&c| m.cells[c as usize].bbox.contains(p))
                        .count()
                }
                None => unit.bvh.query(p, |_| grid_cands += 1),
            }
        }
        fewer_cands = grid_cands as f64 <= 0.5 * bvh_cands as f64;
        // Without residual programs the words per candidate do not change.
        if !fewer_cands && (!residuals || (bvh_words as f64) < GRID_MIN_WORDS * n) {
            return None;
        }
        if !fewer_cands {
            // Kept, if at all, for its residual programs: a smaller budget.
            let small = grid::Params {
                max_words: (GRID_RESIDUAL_WORDS_PER_CELL * cells.len())
                    .clamp(1 << 14, GRID_MAX_WORDS),
                ..params
            };
            tree = Octree::build(root, &cells, keep, &small);
        }
    }
    // Within a leaf only a few surfaces of a long program can change its
    // value: keep each candidate's program specialized to its leaf.
    // (`Options::residuals` off keeps whole programs, for A/B runs.)
    if !residuals {
        return Some(tree);
    }
    specialize_grid(m, &mut tree);
    if !fewer_cands {
        // Kept only for its residual programs: they must pay for it.
        let mut grid_words = 0usize;
        for &p in &samples {
            match tree.leaf_span(p) {
                Some((first, len)) => {
                    for k in first..first + len {
                        let c = tree.items[k];
                        if m.cells[c as usize].bbox.contains(p) {
                            grid_words += tree.prog(k, m.code(c)).map_or(0, <[u32]>::len);
                        }
                    }
                }
                None => unit.bvh.query(p, |c| grid_words += m.code(c).len()),
            }
        }
        if grid_words as f64 * GRID_WORD_GAIN > bvh_words as f64 {
            return None;
        }
    }
    Some(tree)
}

/// Specialize every candidate's program to its octree leaf.
fn specialize_grid(m: &Model, tree: &mut Octree) {
    let tol = m.tolerance;
    tree.specialize(|c, b, out| {
        if m.cells[c as usize].unsupported.is_some() {
            return Prog::Full;
        }
        let start = out.len();
        let v = expr::specialize(
            m.code(c),
            b,
            m.node_boxes_of(c),
            &mut |s, want_pos| match (m.shapes[s as usize].side_over(b, tol), want_pos) {
                (Side::On, _) => Tri::Unknown,
                (Side::Pos, true) | (Side::Neg, false) => Tri::True,
                _ => Tri::False,
            },
            out,
        );
        match v {
            // No shorter than the whole program: keep that, not a copy.
            None if out.len() - start >= m.code(c).len() => {
                out.truncate(start);
                Prog::Full
            }
            None => Prog::Code,
            Some(Tri::True) => Prog::True,
            // The octree kept the cell, so it is not false over the whole
            // leaf; should rounding say otherwise, keep the whole program.
            Some(_) => Prog::Full,
        }
    });
}
