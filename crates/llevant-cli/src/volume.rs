//! `volume` and `check`: volumes (and masses) of cells, materials or universes
//! with their standard error, and the gaps and overlaps of a model.

use std::path::{Path, PathBuf};
use std::time::Instant;

use llevant::{Aabb, Axis, Depth, Model, Scope, Slice, Volume, VolumeGrid};
use llevant_render::colours::{self, ColourBy, Outlines, Paint};
use llevant_render::png::write_png;
use llevant_scene::materials::{group_masses, Looks};
use serde_json::{json, Value};

use crate::args::{Format, GridArgs, GroupBy, Open};
use crate::common::{csv_field, err, load, region};
use crate::json;

#[derive(clap::Args)]
pub(crate) struct VolumeArgs {
    /// The input model file.
    model: PathBuf,
    /// Line spacing in cm: as many lines as three axis-parallel
    /// families this far apart.
    #[arg(long)]
    h: f64,
    #[command(flatten)]
    grid: GridArgs,
    /// Refine the groups with a number (cells, materials, universes;
    /// every inside label with --by label) until sigma / volume is at
    /// most this, halving h over the boxes their cells can be in. Each
    /// row gets a status: converged, not converged, not found (no line
    /// met it down to --h-min) or empty (its boxes are empty: volume 0).
    #[arg(long)]
    rel_sigma: Option<f64>,
    /// Smallest h to refine to with --rel-sigma, cm (default h / 64).
    #[arg(long)]
    h_min: Option<f64>,
    /// With --rel-sigma: a group whose sigma is at most this many cm3 is
    /// done too, however large its relative sigma (small cells would
    /// otherwise need lines a fraction of their size apart). Default 0:
    /// none.
    #[arg(long, default_value_t = 0.0)]
    abs_sigma: f64,
    /// With --rel-sigma: probes a group may have at one spacing. Groups a
    /// little above the target get more probes there when that is
    /// cheaper than a finer spacing: more converge, for more time.
    /// Default: --probes (none added) for the whole model, 48 with --cells.
    #[arg(long)]
    max_probes: Option<usize>,
    /// Only these cells (MCNP numbers, comma separated), every instance
    /// of each together, lines traced around them alone; a filled
    /// cell's volume is that of its region. --h is then the largest
    /// spacing: each cell starts at the one that puts 8 lines across
    /// its box, then halves it towards --rel-sigma.
    #[arg(long, value_delimiter = ',')]
    cells: Vec<i64>,
    #[arg(long, value_enum, default_value = "cell")]
    by: GroupBy,
    /// Also include mass (g) and uncertainty columns in text/CSV output
    /// (JSON always includes `mass_g` and `mass_sigma_g`).
    #[arg(long)]
    mass: bool,
    /// Output: a text table, CSV, or JSON (its fields: docs/guide/cli.md,
    /// "Scripting and agents";
    /// https://fusion4energy.github.io/llevant/guide/cli/#scripting-and-agents).
    #[arg(long, value_enum, default_value = "text")]
    pub(crate) format: Format,
}

impl VolumeArgs {
    pub(crate) fn run(self, open: &Open) -> Result<u8, String> {
        let VolumeArgs {
            model,
            h,
            grid,
            rel_sigma,
            h_min,
            abs_sigma,
            max_probes,
            cells,
            by,
            mass,
            format,
        } = self;
        let m = load(&model, open)?;
        let looks = Looks::new(&m, &[]).map_err(err)?;
        let h_min = h_min.unwrap_or(h / 64.0);
        if !cells.is_empty() {
            if by != GroupBy::Cell {
                return Err("--cells gives volumes by cell: --by must be cell".into());
            }
            let max_probes = max_probes.unwrap_or(llevant::MAX_PROBES);
            let c = cell_volumes(
                &m, &cells, h, h_min, rel_sigma, abs_sigma, max_probes, &grid,
            )?;
            return Ok(match rel_sigma {
                None => {
                    print_volumes(&m, &looks, by, &c.groups, None, None, mass, format);
                    0
                }
                Some(rel_sigma) => {
                    print_volumes(&m, &looks, by, &c.groups, None, Some(&c), mass, format);
                    report_converged(&c, rel_sigma, h_min)
                }
            });
        }
        let Some(rel_sigma) = rel_sigma else {
            let v = volumes(&m, h, &grid)?;
            let groups = v.grouped(&m, by.into());
            let by_cell = (matches!(by, GroupBy::Material | GroupBy::Universe)
                && (mass || format == Format::Json))
                .then(|| v.grouped(&m, llevant::GroupBy::Cell));
            print_volumes(
                &m,
                &looks,
                by,
                &groups,
                by_cell.as_deref(),
                None,
                mass,
                format,
            );
            return Ok(0);
        };
        let target = llevant::VolumeTarget {
            rel_sigma,
            h_min,
            abs_sigma,
            max_probes: max_probes.unwrap_or(grid.probes),
            by: by.into(),
        };
        let c = converged_volumes(&m, h, &grid, &target)?;
        print_volumes(&m, &looks, by, &c.groups, None, Some(&c), mass, format);
        Ok(report_converged(&c, rel_sigma, h_min))
    }
}

#[derive(clap::Args)]
pub(crate) struct CheckArgs {
    /// The input model file.
    model: PathBuf,
    /// Line spacing in cm.
    #[arg(long)]
    h: f64,
    #[command(flatten)]
    grid: GridArgs,
    /// Write XY/XZ/YZ diagnostic slices for top defects plus `report.json`
    /// and `report.md` into this directory.
    #[arg(long)]
    plots: Option<PathBuf>,
    /// Maximum number of top defects to plot (with `--plots`).
    #[arg(long, default_value_t = 10)]
    max_plots: usize,
    /// Width in pixels of each defect slice plot (with `--plots`).
    #[arg(long, default_value_t = 600)]
    px: usize,
    /// Output: a text table, CSV, or JSON (its fields: docs/guide/cli.md,
    /// "Scripting and agents";
    /// https://fusion4energy.github.io/llevant/guide/cli/#scripting-and-agents).
    #[arg(long, value_enum, default_value = "text")]
    pub(crate) format: Format,
}

impl CheckArgs {
    pub(crate) fn run(self, open: &Open) -> Result<u8, String> {
        let CheckArgs {
            model,
            h,
            grid,
            plots,
            max_plots,
            px,
            format,
        } = self;
        let m = load(&model, open)?;
        let v = volumes(&m, h, &grid)?;
        let found = check(&m, &v, format, plots.as_deref(), max_plots, px)?;
        Ok(if found { 1 } else { 0 })
    }
}

fn volumes(m: &Model, h: f64, g: &GridArgs) -> Result<llevant::Volumes, String> {
    if !(h > 0.0 && h.is_finite()) {
        return Err("--h must be positive".into());
    }
    let grid = grid(m, h, g)?;
    eprintln!("tracing up to {} lines (h = {h} cm)", grid.line_count());
    let t = Instant::now();
    let v = m.volumes(&grid, None).map_err(err)?;
    eprintln!("traced in {:.1} s", t.elapsed().as_secs_f64());
    Ok(v)
}

fn grid(m: &Model, h: f64, g: &GridArgs) -> Result<VolumeGrid, String> {
    if !(h > 0.0 && h.is_finite()) {
        return Err("--h must be positive".into());
    }
    if g.probes < 2 {
        return Err("--probes must be at least 2".into());
    }
    Ok(VolumeGrid {
        probes: g.probes,
        ..VolumeGrid::seeded(region(m, &g.region, Scope::Problem)?, h, g.seed)
    })
}

fn converged_volumes(
    m: &Model,
    h: f64,
    g: &GridArgs,
    target: &llevant::VolumeTarget,
) -> Result<llevant::Converged, String> {
    if !(target.rel_sigma > 0.0 && target.h_min > 0.0 && target.abs_sigma >= 0.0) {
        return Err("--rel-sigma and --h-min must be positive, --abs-sigma not negative".into());
    }
    let grid = grid(m, h, g)?;
    eprintln!(
        "tracing up to {} lines (h = {h} cm), then refining to sigma / volume <= {}",
        grid.line_count(),
        target.rel_sigma
    );
    let t = Instant::now();
    let c = m
        .volumes_to(
            &grid,
            target,
            |g| g.status == llevant::Status::Inside,
            None,
            None,
        )
        .map_err(err)?;
    eprintln!("traced in {:.1} s", t.elapsed().as_secs_f64());
    Ok(c)
}

/// The messages on stderr for groups that did not reach the target, were
/// not found or are empty, and the exit status: 1 if some did not reach it.
fn report_converged(c: &llevant::Converged, rel_sigma: f64, h_min: f64) -> u8 {
    let names = |ix: &mut dyn Iterator<Item = usize>| {
        ix.take(20)
            .map(|i| c.groups[i].name.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let found: Vec<usize> = c
        .unconverged
        .iter()
        .copied()
        .filter(|i| !c.unseen.contains(i))
        .collect();
    if !found.is_empty() {
        eprintln!(
            "{} did not reach sigma / volume <= {rel_sigma} by h = {}: {}",
            found.len(),
            h_min,
            names(&mut found.iter().copied())
        );
    }
    if !c.unseen.is_empty() {
        eprintln!(
            "{} not found by h = {} (their boxes may hold something): {}",
            c.unseen.len(),
            h_min,
            names(&mut c.unseen.iter().copied())
        );
    }
    if !c.empty.is_empty() {
        eprintln!(
            "{} empty (their boxes are empty in the region): {}",
            c.empty.len(),
            names(&mut c.empty.iter().copied())
        );
    }
    if !c.listed_all {
        eprintln!(
            "more than {} instances: only those some line met are listed",
            llevant::MAX_LISTED_INSTANCES
        );
    }
    if c.unconverged.is_empty() {
        0
    } else {
        1
    }
}

/// The volumes of `cells` (MCNP numbers) alone, refined to `rel_sigma` if
/// given (else one pass at the first spacing every cell starts at).
#[allow(clippy::too_many_arguments)]
fn cell_volumes(
    m: &Model,
    cells: &[i64],
    h: f64,
    h_min: f64,
    rel_sigma: Option<f64>,
    abs_sigma: f64,
    max_probes: usize,
    g: &GridArgs,
) -> Result<llevant::Converged, String> {
    let grid = grid(m, h, g)?;
    if !(h_min > 0.0 && h_min <= h && abs_sigma >= 0.0) || rel_sigma.is_some_and(|r| !(r > 0.0)) {
        return Err(
            "--rel-sigma must be positive, --abs-sigma not negative and --h-min in (0, --h]".into(),
        );
    }
    let idx = cells
        .iter()
        .map(|&c| m.cell_index(c).ok_or_else(|| format!("no cell {c}")))
        .collect::<Result<Vec<u32>, String>>()?;
    let target = llevant::BinTarget {
        h_max: h,
        abs_sigma,
        max_probes,
        seed: grid.seed,
        probes: grid.probes,
        ..llevant::BinTarget::new(grid.region, rel_sigma.unwrap_or(f64::INFINITY), h_min)
    };
    eprintln!(
        "tracing around {} cells (h from {h} cm down to {h_min} cm)",
        idx.len()
    );
    let t = Instant::now();
    let c = m.cell_volumes(&idx, &target, None).map_err(err)?;
    eprintln!("traced in {:.1} s", t.elapsed().as_secs_f64());
    Ok(c)
}

/// Groups as a table; with `c`, the `h` each came from and whether it
/// reached the target.
#[allow(clippy::too_many_arguments)]
fn print_volumes(
    m: &Model,
    looks: &Looks,
    by: GroupBy,
    groups: &[llevant::Group],
    cell_groups: Option<&[llevant::Group]>,
    c: Option<&llevant::Converged>,
    mass: bool,
    format: Format,
) {
    let name = by.name();
    let masses = group_masses(m, looks, by.into(), groups, cell_groups);
    let unconverged = |i: usize| c.is_some_and(|c| c.unconverged.binary_search(&i).is_ok());
    let status = |c: &llevant::Converged, i: usize| {
        if c.empty.binary_search(&i).is_ok() {
            "empty"
        } else if c.unseen.binary_search(&i).is_ok() {
            "not found"
        } else if unconverged(i) {
            "not converged"
        } else {
            "converged"
        }
    };
    match format {
        Format::Text => {
            print!(
                "{name:>12} {:>14} {:>12} {:>9}",
                "volume (cm3)", "sigma", "rel"
            );
            if mass {
                print!(" {:>14} {:>12}", "mass (g)", "mass_sigma");
            }
            println!("{}", if c.is_some() { "        h" } else { "" });
            for (i, g) in groups.iter().enumerate() {
                let vol = &g.volume;
                print!(
                    "{:>12} {:>14.6e} {:>12.3e} {:>9.2e}",
                    g.name,
                    vol.estimate,
                    vol.sigma,
                    vol.sigma / vol.estimate
                );
                if mass {
                    match masses[i] {
                        Some((mg, dmg)) => print!(" {mg:>14.6e} {dmg:>12.3e}"),
                        None => print!(" {:>14} {:>12}", "-", "-"),
                    }
                }
                match c {
                    Some(c) => println!(
                        " {:>8}{}",
                        c.h[i],
                        match status(c, i) {
                            "converged" => String::new(),
                            s => format!("  {s}"),
                        }
                    ),
                    None => println!(),
                }
            }
            let total: f64 = groups.iter().map(|g| g.volume.estimate).sum();
            print!("{:>12} {total:>14.6e}", "total");
            if mass {
                let total_mass: f64 = masses.iter().flatten().map(|(mg, _)| *mg).sum();
                print!(" {:>12} {:>9} {total_mass:>14.6e}", "", "");
            }
            println!();
        }
        Format::Csv => {
            print!("{name},volume,sigma,segments,probes_hit");
            if mass {
                print!(",mass_g,mass_sigma_g");
            }
            println!(
                "{}",
                if c.is_some() {
                    ",h,converged,status"
                } else {
                    ""
                }
            );
            for (i, g) in groups.iter().enumerate() {
                let vol = &g.volume;
                print!(
                    "{},{},{},{},{}",
                    csv_field(&g.name),
                    vol.estimate,
                    vol.sigma,
                    vol.segments,
                    vol.probes_hit()
                );
                if mass {
                    match masses[i] {
                        Some((mg, dmg)) => print!(",{mg},{dmg}"),
                        None => print!(",,"),
                    }
                }
                match c {
                    Some(c) => println!(",{},{},{}", c.h[i], !unconverged(i), status(c, i)),
                    None => println!(),
                }
            }
        }
        Format::Json => {
            let rows: Vec<Value> = groups
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    // A group is a sum over labels: its number (null for
                    // gaps, overlaps... and with --by label), and its name;
                    // a group that is one label has the label too.
                    let mut r = volume_json(&g.volume);
                    if name != "label" {
                        r[name] = json!(g.number);
                    }
                    r["name"] = json!(g.name);
                    r["status"] = json!(g.status.name());
                    r["label"] = g.label.as_ref().map_or(Value::Null, |l| json::label(m, l));
                    match masses[i] {
                        Some((mg, dmg)) => {
                            r["mass_g"] = json!(mg);
                            r["mass_sigma_g"] = json!(dmg);
                        }
                        None => {
                            r["mass_g"] = Value::Null;
                            r["mass_sigma_g"] = Value::Null;
                        }
                    }
                    if let Some(c) = c {
                        r["h"] = json!(c.h[i]);
                        r["converged"] = json!(!unconverged(i));
                        r["outcome"] = json!(status(c, i));
                    }
                    r
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&rows).expect("json"));
        }
    }
}

fn volume_json(v: &Volume) -> Value {
    json!({
        "volume": v.estimate,
        "sigma": v.sigma,
        "segments": v.segments,
        "probes_hit": v.probes_hit(),
        "by_probe": v.by_probe,
        "box": json::bbox(&v.bbox),
    })
}

/// Print gaps and overlaps; true if there are any.
fn check(
    m: &Model,
    v: &llevant::Volumes,
    format: Format,
    plots_dir: Option<&Path>,
    max_plots: usize,
    px: usize,
) -> Result<bool, String> {
    let p = v.problems();
    match format {
        Format::Text => {
            if p.is_empty() {
                println!("no gaps or overlaps found (h = {} cm)", v.grid.h);
            }
            for &i in &p {
                let (l, vol) = (&v.labels[i], &v.volumes[i]);
                println!(
                    "{}: {:.4e} ± {:.1e} cm3 in {:.3?} .. {:.3?}",
                    m.describe(l),
                    vol.estimate,
                    vol.sigma,
                    vol.bbox.lo,
                    vol.bbox.hi
                );
            }
            if !m.diagnostics().is_empty() {
                println!(
                    "({} compile diagnostics; see llevant info)",
                    m.diagnostics().len()
                );
            }
        }
        Format::Csv => {
            println!("status,label,volume,sigma,x0,y0,z0,x1,y1,z1");
            for &i in &p {
                let (l, vol) = (&v.labels[i], &v.volumes[i]);
                let (a, b) = (vol.bbox.lo, vol.bbox.hi);
                println!(
                    "{},{},{},{},{},{},{},{},{},{}",
                    l.status.name(),
                    csv_field(&m.describe(l)),
                    vol.estimate,
                    vol.sigma,
                    a[0],
                    a[1],
                    a[2],
                    b[0],
                    b[1],
                    b[2]
                );
            }
        }
        Format::Json => {
            let rows: Vec<Value> = p
                .iter()
                .map(|&i| {
                    let mut r = volume_json(&v.volumes[i]);
                    r["label"] = json::label(m, &v.labels[i]);
                    r
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&rows).expect("json"));
        }
    }
    if let Some(dir) = plots_dir {
        write_check_report(m, v, &p, dir, max_plots, px)?;
    }
    Ok(!p.is_empty())
}

fn write_check_report(
    m: &Model,
    v: &llevant::Volumes,
    problems: &[usize],
    dir: &Path,
    max_plots: usize,
    px: usize,
) -> Result<(), String> {
    use std::fmt::Write as _;

    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let looks = Looks::new(m, &[]).map_err(err)?;
    let paint = Paint::new(m, &looks, ColourBy::Material);
    let px = px.max(32);

    let mut json_rows = Vec::with_capacity(problems.len());
    let mut md = String::new();
    let _ = writeln!(md, "# Geometry QA Report\n");
    let _ = writeln!(
        md,
        "- Sampling spacing `h`: `{}` cm\n- Defects found: `{}`\n- Compile diagnostics: `{}`\n",
        v.grid.h,
        problems.len(),
        m.diagnostics().len()
    );

    if problems.is_empty() {
        md.push_str("No gaps or overlaps found.\n");
    } else {
        md.push_str("| # | Status | Label | Volume (cm3) | Sigma (cm3) | Plots |\n");
        md.push_str("|---|---|---|---|---|---|\n");
    }

    for (rank, &i) in problems.iter().enumerate() {
        let (l, vol) = (&v.labels[i], &v.volumes[i]);
        let desc = m.describe(l);
        let mut png_names = Vec::new();
        if rank < max_plots && vol.bbox.is_finite() && !vol.bbox.is_empty() {
            let lo = vol.bbox.lo;
            let hi = vol.bbox.hi;
            let mid = [
                0.5 * (lo[0] + hi[0]),
                0.5 * (lo[1] + hi[1]),
                0.5 * (lo[2] + hi[2]),
            ];
            let span = (hi[0] - lo[0])
                .max(hi[1] - lo[1])
                .max(hi[2] - lo[2])
                .max(v.grid.h * 2.0)
                .max(0.5);
            let reg = Aabb {
                lo: [0, 1, 2].map(|k| lo[k].min(mid[k] - 0.5 * span) - 0.25 * span),
                hi: [0, 1, 2].map(|k| hi[k].max(mid[k] + 0.5 * span) + 0.25 * span),
            };
            for (tag, axis, at) in [
                ("xy", Axis::Z, mid[2]),
                ("xz", Axis::Y, mid[1]),
                ("yz", Axis::X, mid[0]),
            ] {
                let s = Slice::axis(axis, at, &reg, px);
                let tr = m.slice(&s, Depth::All).map_err(err)?;
                let pix_ids = s.pixels(&tr);
                let rgb =
                    colours::slice_rgb(&paint, &s, &tr.labels, &pix_ids, Some(Outlines::Cell));
                let fname = format!("defect_{:03}_{tag}.png", rank + 1);
                write_png(&dir.join(&fname), s.nx, s.ny, &rgb)?;
                png_names.push(fname);
            }
        }
        let plot_links = if png_names.is_empty() {
            String::from("-")
        } else {
            png_names
                .iter()
                .map(|f| format!("[{f}]({f})"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let _ = writeln!(
            md,
            "| {} | {} | `{}` | {:.4e} | {:.2e} | {} |",
            rank + 1,
            l.status.name(),
            desc.replace('`', "'"),
            vol.estimate,
            vol.sigma,
            plot_links
        );

        let mut row = volume_json(vol);
        row["label"] = json::label(m, l);
        row["plots"] = json!(png_names);
        json_rows.push(row);
    }

    for (rank, row) in json_rows.iter().enumerate() {
        let Some(names) = row["plots"].as_array() else {
            continue;
        };
        if names.is_empty() {
            continue;
        }
        let _ = writeln!(
            md,
            "\n## Defect {}: `{}`\n",
            rank + 1,
            row["label"]["text"].as_str().unwrap_or("")
        );
        for fname in names {
            if let Some(f) = fname.as_str() {
                let _ = writeln!(md, "![{f}]({f})");
            }
        }
    }

    let report_json = dir.join("report.json");
    std::fs::write(
        &report_json,
        serde_json::to_string_pretty(&json_rows).expect("json"),
    )
    .map_err(|e| format!("{}: {e}", report_json.display()))?;
    let report_md = dir.join("report.md");
    std::fs::write(&report_md, md).map_err(|e| format!("{}: {e}", report_md.display()))?;
    eprintln!(
        "wrote QA report ({} defects) to {}",
        problems.len(),
        dir.display()
    );
    Ok(())
}
