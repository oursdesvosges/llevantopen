//! `lost`: diagnose lost-particle coordinates from MCNP, OpenMC, or TRIPOLI-4 runs.

use std::path::PathBuf;

use llevant::{Aabb, Axis, Depth, Line, Model, Slice, Status};
use llevant_render::colours::{self, ColourBy, Outlines, Paint};
use llevant_render::png::write_png;
use serde_json::{json, Value};

use crate::args::{Format, Open};
use crate::common::{csv_field, err, load, surface_name, vec3};
use crate::json;

#[derive(clap::Args)]
pub(crate) struct LostArgs {
    /// Input geometry file (MCNP, OpenMC, TRIPOLI-4, or DRAGON5).
    model: PathBuf,
    /// Coordinates as triples: x y z [x y z ...].
    #[arg(allow_negative_numbers = true)]
    coords: Vec<f64>,
    /// Repeatable point `x,y,z` (alternative to positional coordinates).
    #[arg(long = "point", allow_hyphen_values = true)]
    points: Vec<String>,
    /// Extract lost-particle coordinates from a log/output/points file (`-`: stdin).
    #[arg(long)]
    log: Option<PathBuf>,
    /// Optional particle direction `u,v,w` along which to search nearby crossings.
    #[arg(long, allow_hyphen_values = true)]
    dir: Option<String>,
    /// Neighborhood radius (cm) for local surface/defect probes and diagnostic slices.
    #[arg(long, default_value_t = 1.0)]
    radius: f64,
    /// Write zoomed XY/XZ/YZ diagnostic slices around each point into this directory.
    #[arg(long)]
    plots: Option<PathBuf>,
    /// Width in pixels of each diagnostic slice (with `--plots`).
    #[arg(long, default_value_t = 400)]
    px: usize,
    /// Output format: text, csv, or json.
    #[arg(long, value_enum, default_value = "text")]
    pub(crate) format: Format,
}

impl LostArgs {
    pub(crate) fn run(self, open: &Open) -> Result<u8, String> {
        let LostArgs {
            model,
            coords,
            points,
            log,
            dir,
            radius,
            plots,
            px,
            format,
        } = self;

        if !(radius > 0.0 && radius.is_finite()) {
            return Err("--radius must be positive".into());
        }

        let mut pts: Vec<[f64; 3]> = Vec::new();
        if !coords.is_empty() {
            if coords.len() % 3 != 0 {
                return Err("lost: positional coordinates must come in x y z triples".into());
            }
            pts.extend_from_slice(coords.as_chunks::<3>().0);
        }
        for s in &points {
            pts.push(vec3(s, "--point")?);
        }
        if let Some(f) = &log {
            let text = if f.as_os_str() == "-" {
                std::io::read_to_string(std::io::stdin()).map_err(|e| format!("stdin: {e}"))?
            } else {
                std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?
            };
            let extracted = extract_lost_points(&text);
            if extracted.is_empty() {
                return Err(format!("{}: no coordinates found", f.display()));
            }
            pts.extend(extracted);
        }
        if pts.is_empty() {
            return Err("lost: give x y z coordinates, --point x,y,z, or --log FILE".into());
        }

        let probe_dirs: Vec<[f64; 3]> = match dir {
            Some(ref d) => vec![vec3(d, "--dir")?],
            None => vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };

        let m = load(&model, open)?;
        let looks = plots
            .as_ref()
            .map(|_| llevant_scene::materials::Looks::new(&m, &[]).map_err(err))
            .transpose()?;
        if let Some(dir_path) = &plots {
            std::fs::create_dir_all(dir_path)
                .map_err(|e| format!("{}: {e}", dir_path.display()))?;
        }

        let mut entries = Vec::with_capacity(pts.len());
        for (idx, &pt) in pts.iter().enumerate() {
            let entry = inspect_point(&m, pt, &probe_dirs, radius)?;
            let plot_files = if let (Some(dir_path), Some(looks)) = (&plots, &looks) {
                render_point_slices(&m, looks, dir_path, idx + 1, pt, radius, px)?
            } else {
                Vec::new()
            };
            entries.push((entry, plot_files));
        }

        match format {
            Format::Text => {
                for (entry, files) in &entries {
                    println!(
                        "[{}, {}, {}] status={}  cell={}  nearest_surface={} ({:.3e} cm)",
                        entry.point[0],
                        entry.point[1],
                        entry.point[2],
                        entry.status_name,
                        entry.label_text,
                        entry
                            .nearest_surface
                            .as_deref()
                            .filter(|s| !s.is_empty())
                            .unwrap_or("-"),
                        entry.nearest_distance.unwrap_or(f64::NAN),
                    );
                    println!("  diagnosis: {}", entry.diagnosis);
                    if !entry.nearby_defects.is_empty() {
                        println!(
                            "  nearby defects within {radius} cm: {}",
                            entry.nearby_defects.join("; ")
                        );
                    }
                    if !files.is_empty() {
                        println!("  plots: {}", files.join(", "));
                    }
                }
            }
            Format::Csv => {
                println!(
                    "x,y,z,status,label,nearest_surface,nearest_distance,diagnosis,nearby_defects"
                );
                for (entry, _) in &entries {
                    println!(
                        "{},{},{},{},{},{},{},{},{}",
                        entry.point[0],
                        entry.point[1],
                        entry.point[2],
                        entry.status_name,
                        csv_field(&entry.label_text),
                        csv_field(entry.nearest_surface.as_deref().unwrap_or("")),
                        entry
                            .nearest_distance
                            .map_or(String::new(), |d| d.to_string()),
                        csv_field(&entry.diagnosis),
                        csv_field(&entry.nearby_defects.join("; ")),
                    );
                }
            }
            Format::Json => {
                let rows: Vec<Value> = entries
                    .iter()
                    .map(|(entry, files)| {
                        json!({
                            "point": entry.point,
                            "label": entry.label_json,
                            "nearest_surface": entry.nearest_surface_json,
                            "nearest_distance": entry.nearest_distance,
                            "nearest_point": entry.nearest_point,
                            "nearest_normal": entry.nearest_normal,
                            "diagnosis": entry.diagnosis,
                            "nearby_defects": entry.nearby_defects,
                            "plots": files,
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&rows).expect("json"));
            }
        }
        Ok(0)
    }
}

struct PointDiagnosis {
    point: [f64; 3],
    status_name: &'static str,
    label_text: String,
    label_json: Value,
    nearest_surface: Option<String>,
    nearest_surface_json: Value,
    nearest_distance: Option<f64>,
    nearest_point: Option<[f64; 3]>,
    nearest_normal: Option<[f64; 3]>,
    diagnosis: String,
    nearby_defects: Vec<String>,
}

fn inspect_point(
    m: &Model,
    pt: [f64; 3],
    dirs: &[[f64; 3]],
    radius: f64,
) -> Result<PointDiagnosis, String> {
    let lbl = m.classify_point(pt);
    let label_text = m.describe(&lbl);
    let label_json = json::label(m, &lbl);

    let near = m.nearest_surface(pt, dirs, radius, None).map_err(err)?;
    let (nearest_surface, nearest_surface_json, nearest_distance, nearest_point, nearest_normal) =
        match near {
            Some(ns) => {
                let sid = m.surface_id(ns.shape);
                let sname = surface_name(sid);
                let normal = m.surface_normal(ns.shape, ns.point);
                (
                    (!sname.is_empty()).then_some(sname),
                    json::surface(m, Some(ns.shape)),
                    Some(ns.distance),
                    Some(ns.point),
                    normal,
                )
            }
            None => (None, Value::Null, None, None, None),
        };

    // Cast local lines through `pt` along `dirs` to catch adjacent gaps/overlaps.
    let probe_lines: Vec<Line> = dirs
        .iter()
        .filter_map(|&d| {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            (n > 0.0).then(|| {
                let u = [d[0] / n, d[1] / n, d[2] / n];
                Line::new(
                    [
                        pt[0] - radius * u[0],
                        pt[1] - radius * u[1],
                        pt[2] - radius * u[2],
                    ],
                    u,
                    0.0,
                    2.0 * radius,
                )
            })
        })
        .collect();
    let mut nearby_defects = Vec::new();
    if !probe_lines.is_empty() {
        let tr = m.trace(&probe_lines, Depth::All).map_err(err)?;
        for seg_idx in 0..tr.labels.len() {
            let l = &tr.labels[seg_idx];
            if !matches!(l.status, Status::Inside | Status::OnSurface) {
                let d = m.describe(l);
                if !nearby_defects.contains(&d) {
                    nearby_defects.push(d);
                }
            }
        }
    }

    let diagnosis = match lbl.status {
        Status::Gap => match &nearest_surface {
            Some(s) => format!(
                "gap (undefined void); nearest boundary is surface {s} at {:.3e} cm",
                nearest_distance.unwrap_or(0.0)
            ),
            None => "gap (no cell contains this coordinate)".into(),
        },
        Status::Overlap => {
            let involved: Vec<String> = lbl
                .involved
                .iter()
                .map(|&c| m.cell_id(c).to_string())
                .collect();
            format!("cell overlap between cells [{}]", involved.join(", "))
        }
        Status::OnSurface => match &nearest_surface {
            Some(s) => format!(
                "on surface {s} within tolerance ({:.1e} cm) — grazing or coincident boundary",
                m.tolerance()
            ),
            None => format!("on surface within tolerance ({:.1e} cm)", m.tolerance()),
        },
        Status::Unsupported => "unsupported geometry construct at this point".into(),
        Status::TooDeep => "universe nesting depth exceeded".into(),
        Status::Inside => {
            let cell = lbl.cell().map(|c| m.cell(c));
            if cell.is_some_and(|c| c.imp_zero) {
                "inside zero-importance (IMP:N=0) graveyard cell".into()
            } else if !nearby_defects.is_empty() {
                format!(
                    "inside cell, but geometry defects lie within {radius} cm: {}",
                    nearby_defects.join("; ")
                )
            } else if let (Some(s), Some(dist)) = (&nearest_surface, nearest_distance) {
                if dist <= 1e-4 {
                    format!(
                        "inside cell, only {dist:.3e} cm from surface {s} (possible floating-point boundary crossing)"
                    )
                } else {
                    format!("inside valid cell (nearest surface {s} is {dist:.3e} cm away)")
                }
            } else {
                "inside valid cell with no surface crossings inside probe radius".into()
            }
        }
        _ => lbl.status.name().to_string(),
    };

    Ok(PointDiagnosis {
        point: pt,
        status_name: lbl.status.name(),
        label_text,
        label_json,
        nearest_surface,
        nearest_surface_json,
        nearest_distance,
        nearest_point,
        nearest_normal,
        diagnosis,
        nearby_defects,
    })
}

fn render_point_slices(
    m: &Model,
    looks: &llevant_scene::materials::Looks,
    dir: &std::path::Path,
    index_1: usize,
    pt: [f64; 3],
    radius: f64,
    px: usize,
) -> Result<Vec<String>, String> {
    let reg = Aabb {
        lo: [pt[0] - radius, pt[1] - radius, pt[2] - radius],
        hi: [pt[0] + radius, pt[1] + radius, pt[2] + radius],
    };
    let paint = Paint::new(m, looks, ColourBy::Material);
    let mut files = Vec::with_capacity(3);
    for (tag, axis, at) in [
        ("xy", Axis::Z, pt[2]),
        ("xz", Axis::Y, pt[1]),
        ("yz", Axis::X, pt[0]),
    ] {
        let s = Slice::axis(axis, at, &reg, px.max(32));
        let tr = m.slice(&s, Depth::All).map_err(err)?;
        let ids = s.pixels(&tr);
        let rgb = colours::slice_rgb(&paint, &s, &tr.labels, &ids, Some(Outlines::Cell));
        let fname = format!("lost_{index_1:03}_{tag}.png");
        write_png(&dir.join(&fname), s.nx, s.ny, &rgb)?;
        files.push(fname);
    }
    Ok(files)
}

/// Extract `(x, y, z)` coordinates from MCNP `lost` messages, OpenMC lost-particle
/// output, or plain whitespace/comma coordinate files.
fn extract_lost_points(text: &str) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    let parse_float_tok = |tok: &str| -> Option<f64> {
        let cleaned = tok.trim_matches(|c: char| matches!(c, ',' | '(' | ')' | '[' | ']' | ';'));
        if cleaned.is_empty() {
            return None;
        }
        cleaned.parse::<f64>().ok().filter(|v| v.is_finite())
    };
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let tagged = lower.contains("x, y, z")
            || lower.contains("x,y,z")
            || lower.contains("xyz")
            || lower.contains("position")
            || lower.contains("lost at");
        if tagged {
            let tail = if let Some((_, rhs)) = line.split_once('=') {
                rhs
            } else if let Some(pos) = lower.find("lost at") {
                &line[pos + "lost at".len()..]
            } else if let Some(pos) = lower.find("position") {
                &line[pos + "position".len()..]
            } else {
                line
            };
            let nums: Vec<f64> = tail
                .split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | ';'))
                .filter_map(parse_float_tok)
                .collect();
            if nums.len() >= 3 {
                out.push([nums[0], nums[1], nums[2]]);
                continue;
            }
        }
        // Plain 3-number coordinate row.
        let parts: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() == 3 {
            if let (Some(x), Some(y), Some(z)) = (
                parse_float_tok(parts[0]),
                parse_float_tok(parts[1]),
                parse_float_tok(parts[2]),
            ) {
                out.push([x, y, z]);
            }
        }
    }
    out
}
