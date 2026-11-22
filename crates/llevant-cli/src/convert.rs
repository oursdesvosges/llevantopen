//! `convert`: translate OpenMC XML, TRIPOLI-4, or DRAGON5 inputs into MCNP syntax.

use std::io::Write;
use std::path::PathBuf;

use crate::args::Open;
use crate::common::err;

#[derive(clap::Args)]
pub(crate) struct ConvertArgs {
    /// Input model (OpenMC XML / directory, TRIPOLI-4, DRAGON5 GEO:, or MCNP).
    model: PathBuf,
    /// Write the translated MCNP input here (default: stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Compile the translated MCNP input with llevant and exit with 1 if any
    /// compile diagnostics are reported.
    #[arg(long)]
    verify: bool,
}

impl ConvertArgs {
    pub(crate) fn run(self, _open: &Open) -> Result<u8, String> {
        let ConvertArgs {
            model,
            output,
            verify,
        } = self;
        let bytes = llevant::read_input_bytes(&model).map_err(err)?;

        if let Some(out) = &output {
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            std::fs::write(out, &bytes).map_err(|e| format!("{}: {e}", out.display()))?;
            eprintln!("wrote {} ({} bytes)", out.display(), bytes.len());
        } else {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&bytes).map_err(|e| e.to_string())?;
            stdout.flush().map_err(|e| e.to_string())?;
        }

        if verify {
            let compiled =
                llevant::Model::compile(&bytes, &llevant::Options::default(), None).map_err(err)?;
            let diags = compiled.diagnostics();
            let stats = compiled.stats();
            if !diags.is_empty() {
                eprintln!("verification found {} compile diagnostics:", diags.len());
                for d in diags {
                    eprintln!("  {d}");
                }
                return Ok(1);
            }
            eprintln!(
                "verified: {} cells, {} surfaces, {} universes (0 diagnostics)",
                stats.cells, stats.surfaces, stats.units
            );
        }
        Ok(0)
    }
}
