//! `llevant`: query MCNP geometry from the command line (the binary and the
//! Python module both call [`run_cli`]). The queries, volume groups and
//! fingerprint comparisons come from the `llevant` library, images from
//! `llevant-render` (and `llevant-gpu` with `--gpu`); this crate parses
//! arguments, formats tables and writes files: a module per command (or
//! pair of commands), with the arguments they share in `args` and `field`
//! and the helpers in `common`; `json` builds the shapes every JSON output
//! shares (a label, a surface, a box).

mod args;
mod common;
mod convert;
mod field;
mod fingerprint;
mod info;
mod json;
mod locate;
mod lost;
mod plots;
mod render;
mod slice;
mod view;
mod volume;

use clap::{Parser, Subcommand};

use args::{Format, Open};

#[derive(Parser)]
#[command(
    name = "llevant",
    version,
    about = "Fast, exact queries on MCNP, OpenMC, TRIPOLI-4, and DRAGON5 geometry"
)]
struct Cli {
    /// Worker threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(flatten)]
    open: Open,
    #[command(subcommand)]
    cmd: Cmd,
}

// Parsed once per run, so the size of the largest variant does not matter.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Cmd {
    /// Model summary: sizes, extent, compile times, diagnostics.
    Info(info::InfoArgs),
    /// Where cells are: the box (global frame) of every instance of each,
    /// and the one around them all, to frame a view, place a mesh or a
    /// source, or give --region. Boxes are conservative: padded outward,
    /// and only an outer bound below lattices.
    Bbox(info::BboxArgs),
    /// Where points are: cell path, material and density.
    Locate(locate::LocateArgs),
    /// Segments of lines: where each is in each cell, and the surfaces crossed.
    Trace(locate::TraceArgs),
    /// Image of a plane through the geometry (exact boundaries along rows).
    Slice(slice::SliceArgs),
    /// Render OpenMC <plot> slice definitions directly to PNG images.
    Plots(plots::PlotsArgs),
    /// 3D view: first visible cell along each pixel's ray, shaded.
    Render(render::RenderArgs),
    /// What each material looks like in realistic images: the categories
    /// its composition sorts into (steel, concrete, water...) and its colour,
    /// roughness, metalness and clarity.
    Materials(info::MaterialsArgs),
    /// Translate OpenMC XML, TRIPOLI-4, or DRAGON5 inputs into MCNP syntax.
    Convert(convert::ConvertArgs),
    /// Diagnose lost-particle coordinates against any supported input model.
    Lost(lost::LostArgs),
    /// Digests of every cell and universe that ignore numbering and card order.
    Fingerprint(fingerprint::FingerprintArgs),
    /// Interactive viewer: 3D view and 2D slices in a window.
    View(view::ViewArgs),
    /// Volumes of cells, materials or universes, from lattices of lines in
    /// random directions, with their standard error. With --rel-sigma,
    /// every group the model can hold is listed, met or not, and the exit
    /// status is 1 if some did not reach it or were not found.
    Volume(volume::VolumeArgs),
    /// Gaps and overlaps, with their volume and where they are. Exits with 1
    /// if any is found.
    Check(volume::CheckArgs),
}

/// Runs the command line `args` (the first is the program name) and
/// returns the exit status: 0; 1 when the answer is "no" (`check` finds gaps
/// or overlaps, `volume --rel-sigma` falls short, `bbox` finds nothing,
/// `fingerprint --diff` finds differences); 2 for errors (bad arguments
/// included), which with JSON output are also `{"error": "..."}` on stdout.
pub fn run_cli<I, T>(args: I) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    // `RUST_LOG=llevant=debug`: the engine's debug log (volume rounds).
    let _ = env_logger::try_init();
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return e.exit_code() as u8;
        }
    };
    let (cmd, open) = (cli.cmd, cli.open);
    let json = cmd.json();
    // A pool for this run only, so every call in one process (the Python
    // module) gets the threads it asks for.
    let result = match cli.threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| format!("--threads: {e}"))
            .and_then(|pool| pool.install(move || run(cmd, &open))),
        None => run(cmd, &open),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("llevant: {e}");
            // A reader of JSON on stdout gets the error there too.
            if json {
                println!("{}", serde_json::json!({ "error": e }));
            }
            2
        }
    }
}

impl Cmd {
    /// Whether the output asked for is JSON (`fingerprint` digests always are).
    fn json(&self) -> bool {
        let f = match self {
            Cmd::Info(a) => a.format,
            Cmd::Bbox(a) => a.format,
            Cmd::Locate(a) => a.format,
            Cmd::Trace(a) => a.format,
            Cmd::Lost(a) => a.format,
            Cmd::Materials(a) => a.format,
            Cmd::Volume(a) => a.format,
            Cmd::Check(a) => a.format,
            Cmd::Fingerprint(a) => return a.format == Format::Json || !a.diffs(),
            Cmd::Convert(_) | Cmd::Plots(_) | Cmd::Slice(_) | Cmd::Render(_) | Cmd::View(_) => {
                return false
            }
        };
        f == Format::Json
    }
}

fn run(cmd: Cmd, open: &Open) -> Result<u8, String> {
    match cmd {
        Cmd::Info(a) => a.run(open),
        Cmd::Bbox(a) => a.run(open),
        Cmd::Locate(a) => a.run(open),
        Cmd::Trace(a) => a.run(open),
        Cmd::Slice(a) => a.run(open),
        Cmd::Plots(a) => a.run(open),
        Cmd::Render(a) => a.run(open),
        Cmd::Materials(a) => a.run(open),
        Cmd::Convert(a) => a.run(open),
        Cmd::Lost(a) => a.run(open),
        Cmd::Fingerprint(a) => a.run(open),
        Cmd::View(a) => a.run(open),
        Cmd::Volume(a) => a.run(open),
        Cmd::Check(a) => a.run(open),
    }
}
