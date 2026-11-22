//! Python bindings of llevant: load a model once, then query it from
//! scripts and notebooks. Batch queries take and return numpy arrays; all
//! the work happens in Rust, in parallel, without the GIL.
//!
//! `python -m llevant ...` runs the command line of `llevant-cli` in the
//! Python process (a `.pyd`/`.so` is loaded by the interpreter, so it works
//! where a stand-alone executable may not be allowed).

/// The system heap on Windows does not scale to large parallel batches
/// (docs/decisions/0009-mimalloc.md).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod batch;
mod convert;
mod field;
mod image;
mod model;
mod results;

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::convert::invalid;
use crate::model::{options, Model};

/// Runs the command line `argv` (the first item is the program name) and
/// returns the exit status (0, 1 when `check` finds problems, 2 for errors).
/// Releases the GIL while it works.
#[pyfunction]
fn main(py: Python<'_>, argv: Vec<String>) -> i32 {
    py.detach(|| llevant_cli::run_cli(argv)).into()
}

/// Read and compile an MCNP input file.
///
/// `tolerance` (cm): points closer to a surface are on it. `world_half_size`
/// (cm): half-size of the cube that bounds level 0. `grid`: octrees of
/// candidate cells, `auto`, `always` or `off`.
#[pyfunction]
#[pyo3(signature = (path, *, tolerance = 1e-8, world_half_size = 1e6, grid = "auto"))]
fn load(
    py: Python<'_>,
    path: PathBuf,
    tolerance: f64,
    world_half_size: f64,
    grid: &str,
) -> PyResult<Model> {
    Model::load(py, path, options(tolerance, world_half_size, grid)?)
}

/// Compile MCNP input text (see `load`).
#[pyfunction]
#[pyo3(signature = (text, *, tolerance = 1e-8, world_half_size = 1e6, grid = "auto"))]
fn from_text(
    py: Python<'_>,
    text: &str,
    tolerance: f64,
    world_half_size: f64,
    grid: &str,
) -> PyResult<Model> {
    let opts = options(tolerance, world_half_size, grid)?;
    let m = py.detach(|| llevant::Model::from_text(text, &opts));
    Ok(Model::new(m, None, opts).with_text(text.into()))
}

/// Read an input file or OpenMC model directory (MCNP, OpenMC, TRIPOLI-4,
/// or DRAGON5) and return canonical MCNP input text.
#[pyfunction]
fn _translate_path(py: Python<'_>, path: PathBuf) -> PyResult<String> {
    let bytes = py
        .detach(|| llevant::read_input_bytes(&path))
        .map_err(convert::err)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Translate MCNP, OpenMC XML, TRIPOLI-4, or DRAGON5 text into canonical MCNP
/// input text.
#[pyfunction]
fn _translate_text(py: Python<'_>, text: &str) -> PyResult<String> {
    let translated = py
        .detach(|| llevant::translate_input_bytes(text.as_bytes()))
        .map_err(convert::err)?;
    Ok(String::from_utf8_lossy(translated.as_ref()).into_owned())
}

/// Read OpenMC `<plot>` slice/voxel metadata from a model path or `plots.xml`.
#[pyfunction]
#[pyo3(signature = (model_path, plots_xml = None))]
fn _read_openmc_plots(
    py: Python<'_>,
    model_path: PathBuf,
    plots_xml: Option<PathBuf>,
) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
    let plots = py
        .detach(|| llevant_openmc::read_openmc_plots(&model_path, plots_xml.as_deref()))
        .map_err(|e| match e {
            llevant_openmc::OpenMcError::Io { path, source } => {
                convert::err(llevant::Error::Io { path, source })
            }
            other => invalid(other.to_string()),
        })?;
    plots
        .into_iter()
        .map(|p| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("id", p.id)?;
            d.set_item("filename", p.filename)?;
            d.set_item("type", p.plot_type)?;
            d.set_item("basis", p.basis)?;
            d.set_item("color_by", p.color_by)?;
            d.set_item("origin", (p.origin[0], p.origin[1], p.origin[2]))?;
            d.set_item("width", (p.width[0], p.width[1]))?;
            d.set_item("pixels", (p.pixels[0], p.pixels[1]))?;
            d.set_item("show_overlaps", p.show_overlaps)?;
            Ok(d.unbind())
        })
        .collect()
}

/// Translate native TRIPOLI-4 input text directly into canonical MCNP text.
#[pyfunction]
fn _tripoli_to_mcnp(py: Python<'_>, text: &str) -> PyResult<String> {
    py.detach(|| llevant_tripoli::tripoli_to_mcnp(text))
        .map_err(|e| invalid(e.to_string()))
}

/// Translate native DRAGON5 CLE-2000 `GEO:` text directly into canonical MCNP text.
#[pyfunction]
fn _dragon_to_mcnp(py: Python<'_>, text: &str) -> PyResult<String> {
    py.detach(|| llevant_dragon::dragon_to_mcnp(text))
        .map_err(|e| invalid(e.to_string()))
}

/// Use `n` worker threads for every query from now on. Only possible
/// before the first query (the thread pool is made once).
#[pyfunction]
fn set_threads(n: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| invalid(format!("set_threads: {e}")))
}

/// Number of worker threads queries use.
#[pyfunction]
fn threads() -> usize {
    rayon::current_num_threads()
}

/// `llevant._frames.<f>(obj[, by])`: data frames are built in Python.
pub(crate) fn frames<'py>(
    py: Python<'py>,
    f: &str,
    obj: &Bound<'py, PyAny>,
    by: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = py.import("llevant._frames")?;
    match by {
        Some(by) => m.call_method1(f, (obj, by)),
        None => m.call_method1(f, (obj,)),
    }
}

#[pymodule]
fn _llevant(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(main, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_text, m)?)?;
    m.add_function(wrap_pyfunction!(_translate_path, m)?)?;
    m.add_function(wrap_pyfunction!(_translate_text, m)?)?;
    m.add_function(wrap_pyfunction!(_read_openmc_plots, m)?)?;
    m.add_function(wrap_pyfunction!(_tripoli_to_mcnp, m)?)?;
    m.add_function(wrap_pyfunction!(_dragon_to_mcnp, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(threads, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<results::Label>()?;
    m.add_class::<results::Surface>()?;
    m.add_class::<results::Segment>()?;
    m.add_class::<results::Diagnostic>()?;
    m.add_class::<results::Cell>()?;
    m.add_class::<results::Cells>()?;
    m.add_class::<results::Group>()?;
    m.add_class::<results::Volumes>()?;
    m.add_class::<results::Fingerprints>()?;
    m.add_class::<results::Changes>()?;
    m.add_class::<results::Diff>()?;
    m.add_class::<batch::Classification>()?;
    m.add_class::<batch::Tracing>()?;
    m.add_class::<batch::Hits>()?;
    m.add_class::<image::Image>()?;
    m.add_class::<field::Field>()?;
    m.add("LlevantError", m.py().get_type::<convert::LlevantError>())?;
    m.add(
        "STATUS_NAMES",
        pyo3::types::PyTuple::new(m.py(), convert::STATUS_NAMES)?,
    )?;
    m.add("NO_HIT", llevant::NO_HIT)?;
    m.add("DEFAULT_PRECISION", llevant::DEFAULT_PRECISION)?;
    Ok(())
}
