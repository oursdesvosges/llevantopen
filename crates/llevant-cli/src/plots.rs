//! `plots`: render OpenMC `<plot>` slice definitions directly to PNGs.

use std::path::PathBuf;
use std::time::Instant;

use llevant::{Aabb, Axis, Depth, Slice};
use llevant_render::colours::{self, ColourBy, Outlines, Paint};
use llevant_render::png::write_png;

use crate::args::{LookArgs, Open};
use crate::common::{err, load};

#[derive(clap::Args)]
pub(crate) struct PlotsArgs {
    /// OpenMC model XML file or directory (`model.xml` / `geometry.xml` +
    /// `plots.xml`), or any model file when `--plots-xml` is given.
    model: PathBuf,
    /// Explicit `plots.xml` file (default: read `<plot>` tags from `model.xml`
    /// or sibling `plots.xml`).
    #[arg(long)]
    plots_xml: Option<PathBuf>,
    /// Directory where rendered PNG files are written.
    #[arg(short = 'O', long, default_value = ".")]
    out_dir: PathBuf,
    /// Override horizontal pixel width (height follows the plot's width aspect
    /// ratio; default uses `<pixels>` from the plot definition).
    #[arg(long)]
    px: Option<usize>,
    /// Only render plots whose OpenMC plot ID is listed (comma-separated).
    #[arg(long, value_delimiter = ',')]
    ids: Vec<i64>,
    /// Draw boundaries between cells, materials or universes.
    #[arg(long)]
    outlines: bool,
    #[command(flatten)]
    looks: LookArgs,
}

impl PlotsArgs {
    pub(crate) fn run(self, open: &Open) -> Result<u8, String> {
        let PlotsArgs {
            model,
            plots_xml,
            out_dir,
            px,
            ids,
            outlines,
            looks,
        } = self;
        let mut plots = llevant_openmc::read_openmc_plots(&model, plots_xml.as_deref())
            .map_err(|e| e.to_string())?;
        if !ids.is_empty() {
            plots.retain(|p| ids.contains(&p.id));
        }
        if plots.is_empty() {
            return Err("no matching OpenMC <plot> entries found".into());
        }

        let m = load(&model, open)?;
        let looks = looks.looks(&m)?;
        std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;

        let mut rendered = 0usize;
        for p in &plots {
            if p.plot_type != "slice" {
                eprintln!(
                    "skipping plot {}: type `{}` is not a 2D slice",
                    p.id, p.plot_type
                );
                continue;
            }
            let (wx, wy) = (p.width[0], p.width[1]);
            if !(wx > 0.0 && wy > 0.0 && wx.is_finite() && wy.is_finite()) {
                return Err(format!("plot {}: invalid width {:?}", p.id, p.width));
            }
            let [ox, oy, oz] = p.origin;
            let (axis, at, region) = match p.basis.as_str() {
                "xy" => (
                    Axis::Z,
                    oz,
                    Aabb {
                        lo: [ox - 0.5 * wx, oy - 0.5 * wy, oz - 1.0],
                        hi: [ox + 0.5 * wx, oy + 0.5 * wy, oz + 1.0],
                    },
                ),
                "xz" => (
                    Axis::Y,
                    oy,
                    Aabb {
                        lo: [ox - 0.5 * wx, oy - 1.0, oz - 0.5 * wy],
                        hi: [ox + 0.5 * wx, oy + 1.0, oz + 0.5 * wy],
                    },
                ),
                "yz" => (
                    Axis::X,
                    ox,
                    Aabb {
                        lo: [ox - 1.0, oy - 0.5 * wx, oz - 0.5 * wy],
                        hi: [ox + 1.0, oy + 0.5 * wx, oz + 0.5 * wy],
                    },
                ),
                other => {
                    return Err(format!("plot {}: unsupported basis `{other}`", p.id));
                }
            };

            let width_px = px.unwrap_or(p.pixels[0]).max(1);
            let s = Slice::axis(axis, at, &region, width_px);
            let t = Instant::now();
            let tr = m.slice(&s, Depth::All).map_err(err)?;
            let pix_ids = s.pixels(&tr);

            let colour_by = match p.color_by.as_str() {
                "cell" => ColourBy::Cell,
                "universe" => ColourBy::Universe,
                _ => ColourBy::Material,
            };
            let paint = Paint::new(&m, &looks, colour_by);
            let outline = outlines.then_some(match colour_by {
                ColourBy::Universe => Outlines::Universe,
                ColourBy::Material => Outlines::Material,
                _ => Outlines::Cell,
            });
            let rgb = colours::slice_rgb(&paint, &s, &tr.labels, &pix_ids, outline);

            let raw_name = std::path::Path::new(p.filename.trim())
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let stem = raw_name
                .strip_suffix(".png")
                .or_else(|| raw_name.strip_suffix(".ppm"))
                .unwrap_or(raw_name);
            let file_name = if stem.is_empty() {
                format!("plot_{}.png", p.id)
            } else {
                format!("{stem}.png")
            };
            let out_path = out_dir.join(&file_name);
            write_png(&out_path, s.nx, s.ny, &rgb)?;
            eprintln!(
                "wrote {} ({}×{}, plot {} `{}` in {:.0} ms)",
                out_path.display(),
                s.nx,
                s.ny,
                p.id,
                p.basis,
                1e3 * t.elapsed().as_secs_f64()
            );
            rendered += 1;
        }
        if rendered == 0 {
            return Err("no slice plots were rendered".into());
        }
        Ok(0)
    }
}
