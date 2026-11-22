//! The `llevant` binary on the MCNP zoo models.

use std::path::PathBuf;
use std::process::{Command, Output};

fn models() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/mcnp-oracle/models")
}

fn llevant(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_llevant"))
        .args(args)
        .output()
        .expect("run llevant")
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// `locate --points` gives the cell MCNP reported for every golden point.
#[test]
fn locate_agrees_with_mcnp() {
    for name in ["surfaces", "universes", "periodic"] {
        let golden = std::fs::read_to_string(models().join(format!("{name}.golden"))).unwrap();
        let rows: Vec<(String, i64)> = golden
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| {
                let v: Vec<&str> = l.split_whitespace().collect();
                (v[..3].join(" "), v[3].parse().unwrap())
            })
            .collect();
        let pts = tmp(&format!("{name}.points"));
        let text: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
        std::fs::write(&pts, text.join("\n")).unwrap();
        let model = models().join(format!("{name}.i"));
        let o = llevant(&[
            "locate",
            model.to_str().unwrap(),
            "--points",
            pts.to_str().unwrap(),
            "--format",
            "csv",
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let out = stdout(&o);
        let cells: Vec<&str> = out
            .lines()
            .skip(1)
            .map(|l| l.split(',').nth(4).unwrap())
            .collect();
        assert_eq!(cells.len(), rows.len());
        for (row, cell) in rows.iter().zip(&cells) {
            assert_eq!(*cell, row.1.to_string(), "{name}: point {}", row.0);
        }
        // Every point inside a cell is in one instance of it.
        for (line, cell) in out.lines().skip(1).zip(&cells) {
            let instance = line.rsplit(',').next().unwrap();
            assert_eq!(instance.parse::<u64>().is_ok(), !cell.is_empty(), "{line}");
        }
    }
}

#[test]
fn check_finds_the_one_real_overlap_of_the_zoo() {
    for name in ["surfaces", "universes"] {
        let model = models().join(format!("{name}.i"));
        let o = llevant(&["check", model.to_str().unwrap(), "--h", "1"]);
        assert_eq!(o.status.code(), Some(0), "{name}: {}", stdout(&o));
    }
    // Cell 2 of the periodic model is unbounded along z, so it runs into the
    // graveyard beyond z = ±10.
    let model = models().join("periodic.i");
    let o = llevant(&["check", model.to_str().unwrap(), "--h", "0.5"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stdout(&o).starts_with("overlap (2, 9)"), "{}", stdout(&o));
}

/// Every JSON output shares one label, surface and box shape
/// (`crates/llevant-cli/src/json.rs`): numbers are numbers, none is `null`.
#[test]
fn json_outputs_share_one_vocabulary() {
    let model = models().join("periodic.i");
    let model = model.to_str().unwrap();
    let json = |args: &[&str]| -> serde_json::Value {
        let o = llevant(args);
        assert!(
            o.status.code().is_some_and(|c| c <= 1),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        serde_json::from_str(&stdout(&o)).unwrap()
    };

    // check: the overlap of cells 2 and 9, as a label object, with its box.
    let rows = json(&["check", model, "--h", "0.5", "--format", "json"]);
    let r = &rows[0];
    assert_eq!(r["label"]["status"], "overlap", "{r}");
    assert_eq!(r["label"]["involved"], serde_json::json!([2, 9]), "{r}");
    assert_eq!(r["label"]["cell"], serde_json::Value::Null, "{r}");
    assert_eq!(r["label"]["text"], "overlap (2, 9)", "{r}");
    assert_eq!(r["box"]["lo"].as_array().unwrap().len(), 3, "{r}");

    // volume --by label: every group is one label, and has it.
    let rows = json(&[
        "volume", model, "--h", "1", "--by", "label", "--format", "json",
    ]);
    for r in rows.as_array().unwrap() {
        assert_eq!(r["label"]["text"], r["name"], "{r}");
        assert_eq!(r["label"]["status"], r["status"], "{r}");
    }
    // ... and by cell, only the groups without a number (the overlap).
    let rows = json(&["volume", model, "--h", "1", "--format", "json"]);
    for r in rows.as_array().unwrap() {
        assert_eq!(r["cell"].is_null(), r["label"].is_object(), "{r}");
    }

    // locate: the same label shape, plus the point. Cell 2 (material 2)
    // holds (-3, -2, 0).
    let rows = json(&["locate", model, "-3", "-2", "0", "--format", "json"]);
    let r = &rows[0];
    assert_eq!(r["status"], "inside", "{r}");
    assert_eq!(r["cell"], 2, "{r}");
    assert_eq!(r["material"], 2, "{r}");
    assert_eq!(
        r["path"],
        serde_json::json!([{"cell": 2, "lattice": null}]),
        "{r}"
    );
    assert_eq!(r["involved"], serde_json::json!([]), "{r}");
    assert!(r.get("cells").is_none(), "{r}");
    assert_eq!(r["point"], serde_json::json!([-3.0, -2.0, 0.0]), "{r}");

    // trace: one entry per line; cells and surfaces are numbers, and no
    // surface starts the line.
    let rows = json(&[
        "trace", model, "--from", "-7,-1,0", "--to", "7,-1,0", "--format", "json",
    ]);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    let segs = rows[0]["segments"].as_array().unwrap();
    let cells: Vec<i64> = segs
        .iter()
        .map(|s| s["label"]["cell"].as_i64().unwrap())
        .collect();
    assert_eq!(cells, [3, 2, 3, 1, 3]);
    assert_eq!(segs[0]["surface_in"], serde_json::Value::Null);
    assert_eq!(
        segs[0]["surface_out"],
        serde_json::json!({"id": 2, "facet": 0})
    );
    assert_eq!(segs[0]["label"]["material"].as_i64(), Some(0));
}

/// `fingerprint --diff --format json`: the changes by kind, and status 1
/// when the models differ.
#[test]
fn fingerprint_diff_as_json() {
    let (a, b) = (models().join("universes.i"), models().join("periodic.i"));
    let (a, b) = (a.to_str().unwrap(), b.to_str().unwrap());
    let o = llevant(&["fingerprint", a, "--diff", a, "--format", "json"]);
    assert_eq!(o.status.code(), Some(0));
    let d: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(d["same"], true);
    assert_eq!(d["cells"]["unchanged"], 43);
    assert_eq!(d["universes"]["renumbered"], serde_json::json!([]));

    let o = llevant(&["fingerprint", a, "--diff", b, "--format", "json"]);
    assert_eq!(o.status.code(), Some(1));
    let d: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(d["same"], false);
    assert!(
        !d["cells"]["removed"].as_array().unwrap().is_empty()
            || !d["cells"]["renumbered"].as_array().unwrap().is_empty()
    );
}

/// `-` reads points and lines from stdin; with JSON output, an error is
/// also a JSON object on stdout.
#[test]
fn stdin_and_json_errors() {
    use std::io::Write;
    let model = models().join("periodic.i");
    let model = model.to_str().unwrap();
    let run = |args: &[&str], input: &str| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_llevant"))
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        c.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
        c.wait_with_output().unwrap()
    };
    let o = run(
        &["locate", model, "--points", "-", "--format", "json"],
        "-3 -2 0\n# c\n0,0,0\n",
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(rows[0]["cell"], 2);
    assert_eq!(rows[1]["cell"], 1);
    let o = run(
        &["trace", model, "--lines", "-", "--format", "csv"],
        "-7 -1 0 7 -1 0\n",
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(stdout(&o).lines().count(), 6);

    let o = run(
        &["locate", model, "--points", "-", "--format", "json"],
        "1 2\n",
    );
    assert_eq!(o.status.code(), Some(2));
    let e: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(
        e["error"].as_str().unwrap().contains("expected 3 numbers"),
        "{e}"
    );
    // Text output keeps stdout for data only.
    let o = run(&["locate", model, "--points", "-"], "1 2\n");
    assert_eq!(o.status.code(), Some(2));
    assert!(stdout(&o).is_empty());
}

/// `trace --lines` traces a file of lines in one batch, as many single runs
/// would; CSV numbers the lines.
#[test]
fn trace_many_lines() {
    let model = models().join("periodic.i");
    let model = model.to_str().unwrap();
    let lines = [
        [-7.0, -1.0, 0.0, 7.0, -1.0, 0.0],
        [0.0, -7.0, 0.5, 0.0, 7.0, 0.5],
        [-5.0, -5.0, -3.0, 5.0, 5.0, 3.0],
    ];
    let file = tmp("three.lines");
    let text: Vec<String> = lines
        .iter()
        .map(|l| l.map(|x| x.to_string()).join(" "))
        .collect();
    std::fs::write(
        &file,
        format!("# x0 y0 z0 x1 y1 z1\n{}\n\n", text.join("\n")),
    )
    .unwrap();
    let o = llevant(&[
        "trace",
        model,
        "--lines",
        file.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let many: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(many.as_array().unwrap().len(), 3);
    for (i, l) in lines.iter().enumerate() {
        let (a, b) = (
            format!("{},{},{}", l[0], l[1], l[2]),
            format!("{},{},{}", l[3], l[4], l[5]),
        );
        let o = llevant(&["trace", model, "--from", &a, "--to", &b, "--format", "json"]);
        let one: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        assert_eq!(one[0], many[i], "line {i}");
    }

    let o = llevant(&[
        "trace",
        model,
        "--lines",
        file.to_str().unwrap(),
        "--format",
        "csv",
    ]);
    let out = stdout(&o);
    assert!(out.starts_with("line,t0,"));
    let numbered: std::collections::BTreeSet<&str> = out
        .lines()
        .skip(1)
        .map(|r| r.split(',').next().unwrap())
        .collect();
    assert_eq!(numbered.into_iter().collect::<Vec<_>>(), ["0", "1", "2"]);

    // Both forms, or neither, are errors; so is a line of zero length.
    let o = llevant(&[
        "trace",
        model,
        "--lines",
        file.to_str().unwrap(),
        "--from",
        "0,0,0",
        "--to",
        "1,0,0",
    ]);
    assert_eq!(o.status.code(), Some(2));
    let o = llevant(&["trace", model]);
    assert_eq!(o.status.code(), Some(2));
    std::fs::write(&file, "1 2 3 1 2 3\n").unwrap();
    let o = llevant(&["trace", model, "--lines", file.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains(":1:"));
}

#[test]
fn trace_reports_mcnp_surfaces() {
    let model = models().join("periodic.i");
    let o = llevant(&[
        "trace",
        model.to_str().unwrap(),
        "--from",
        "-7,-1,0",
        "--to",
        "7,-1,0",
        "--format",
        "csv",
    ]);
    assert!(o.status.success());
    let out = stdout(&o);
    let rows: Vec<Vec<&str>> = out
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    // Cell 3, the cylinder (cell 2), cell 3, the sphere (cell 1), cell 3.
    assert!(rows.iter().all(|r| r[0] == "0"), "one line");
    let cells: Vec<&str> = rows.iter().map(|r| r[7]).collect();
    assert_eq!(cells, ["3", "2", "3", "1", "3"]);
    assert_eq!(rows[0][5], "2");
    assert_eq!(rows[3][4], "1");
    let total: f64 = rows.iter().map(|r| r[3].parse::<f64>().unwrap()).sum();
    assert!((total - 14.0).abs() < 1e-9);
}

#[test]
fn images_and_fingerprints() {
    let model = models().join("universes.i");
    let m = model.to_str().unwrap();
    let png = tmp("slice.png");
    let o = llevant(&[
        "slice",
        m,
        "--plane",
        "z=0",
        "--px",
        "300",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(std::fs::read(&png).unwrap().starts_with(b"\x89PNG"));
    let png = tmp("render.png");
    let o = llevant(&[
        "render",
        m,
        "--size",
        "120x80",
        "--clip",
        "0,0,0,0,-1,0",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(std::fs::read(&png).unwrap().starts_with(b"\x89PNG"));
    let hide = |what: &str| {
        let o = llevant(&[
            "render",
            m,
            "--size",
            "40x30",
            "--hide-material",
            what,
            "-o",
            png.to_str().unwrap(),
        ]);
        (
            o.status.code(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
        )
    };
    // Cuts of other shapes, an orthographic view, rules.
    let o = llevant(&[
        "render",
        m,
        "--size",
        "60x40",
        "--clip",
        "0,0,0,0,-1,0",
        "--clip",
        "200,0,0,1,0,0",
        "--wedge",
        "100,0,0,0,0,1,0,90",
        "--remove-box",
        "0,-5,-5,20,5,5",
        "--ortho",
        "80",
        "--hide-rule",
        "m 2",
        "--isolate",
        "u 0 | in c 13",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = llevant(&[
        "render",
        m,
        "--hide-rule",
        "x 3",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown word `x`"));
    assert_eq!(hide("1,2").0, Some(0));
    let (code, stderr) = hide("99");
    assert_eq!(code, Some(2));
    assert!(stderr.ends_with("llevant: no material 99\n"), "{stderr}");

    let o = llevant(&["fingerprint", m, "--diff", m]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(
        stdout(&o),
        "cells: 43 unchanged, 0 differences\nuniverses: 12 unchanged, 0 differences\n"
    );
    let o = llevant(&["fingerprint", m]);
    let json = stdout(&o);
    assert!(json.contains("\"cells\"") && json.contains("\"universes\""));
}

/// Realistic images: the material report, looks set from a file and on the
/// command line, PBR shading with every extra.
#[test]
fn realistic_images() {
    let model = models().join("universes.i");
    let m = model.to_str().unwrap();
    let o = llevant(&["materials", m, "--format", "json"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(rows[1]["material"], 2);
    assert_eq!(rows[1]["look"], "steel");

    let looks = tmp("looks.txt");
    std::fs::write(&looks, "# looks\n2 copper roughness=0.2  # the steel\n").unwrap();
    let o = llevant(&[
        "materials",
        m,
        "--materials",
        looks.to_str().unwrap(),
        "--look",
        "3=colour=#806040 metalness=0",
    ]);
    let text = stdout(&o);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(
        text.contains("copper #") && text.contains("#806040"),
        "{text}"
    );

    let png = tmp("realistic.png");
    let o = llevant(&[
        "render",
        m,
        "--size",
        "60x40",
        "--colour",
        "realistic",
        "--clip",
        "0,0,0,0,-1,0",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(std::fs::read(&png).unwrap().starts_with(b"\x89PNG"));
    // Shadows, reflections and the like are no longer drawn.
    let o = llevant(&["render", m, "--extras", "all", "-o", png.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    let o = llevant(&["materials", m, "--look", "2=unobtainium"]);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn volumes_add_up_to_the_region() {
    let model = models().join("surfaces.i");
    let o = llevant(&[
        "volume",
        model.to_str().unwrap(),
        "--h",
        "1",
        "--region",
        "-5,-5,-5,5,5,5",
        "--by",
        "material",
        "--format",
        "csv",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let total: f64 = stdout(&o)
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(1).unwrap().parse::<f64>().unwrap())
        .sum();
    // Random lines: the region up to the sampling error.
    assert!((total - 1000.0).abs() < 5.0, "{total}");

    // Refined to a target: every material gets there, and says so.
    let o = llevant(&[
        "volume",
        model.to_str().unwrap(),
        "--h",
        "1",
        "--region",
        "-5,-5,-5,5,5,5",
        "--by",
        "material",
        "--rel-sigma",
        "1e-3",
        "--format",
        "json",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    for r in rows.as_array().unwrap() {
        assert_eq!(r["converged"], true, "{r}");
        let (v, s) = (r["volume"].as_f64().unwrap(), r["sigma"].as_f64().unwrap());
        assert!(s <= 1e-3 * v, "{r}");
    }
}

/// `--cells`: those cells alone, in the order given, a filled cell as its
/// whole region; an unknown cell is an error.
#[test]
fn volumes_of_chosen_cells() {
    let model = models().join("universes.i");
    let o = llevant(&[
        "volume",
        model.to_str().unwrap(),
        "--h",
        "2",
        "--cells",
        "10,13",
        "--rel-sigma",
        "2e-3",
        "--format",
        "json",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    // Cells 10 and 13 are filled 28 cm cubes (`rpp`).
    for (r, cell) in rows.iter().zip([10, 13]) {
        assert_eq!(r["cell"], cell, "{r}");
        assert_eq!(r["outcome"], "converged", "{r}");
        let (v, s) = (r["volume"].as_f64().unwrap(), r["sigma"].as_f64().unwrap());
        assert!(s <= 2e-3 * v, "{r}");
        assert!((v - 28f64.powi(3)).abs() < 4.0 * s, "{r}");
    }
    let o = llevant(&[
        "volume",
        model.to_str().unwrap(),
        "--h",
        "2",
        "--cells",
        "77777",
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("no cell 77777"));
}

#[test]
fn bad_arguments_are_errors() {
    let model = models().join("surfaces.i");
    let o = llevant(&[
        "slice",
        model.to_str().unwrap(),
        "--plane",
        "w=3",
        "-o",
        "x.png",
    ]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown axis"));
}

/// `run_cli` in-process, as the Python module calls it: exit statuses, and
/// `--threads` honoured on every call, not only the first.
#[test]
fn run_cli_in_process() {
    let periodic = models().join("periodic.i");
    let surfaces = models().join("surfaces.i");
    let run = |args: &[&str]| {
        llevant_cli::run_cli(std::iter::once("llevant").chain(args.iter().copied()))
    };
    let p = periodic.to_str().unwrap();
    let s = surfaces.to_str().unwrap();
    assert_eq!(run(&["--threads", "1", "check", p, "--h", "0.5"]), 1);
    assert_eq!(run(&["--threads", "3", "check", s, "--h", "1"]), 0);
    assert_eq!(run(&["--threads", "0x", "info", s]), 2);
    assert_eq!(run(&["slice", s, "--plane", "w=3", "-o", "x.png"]), 2);
}

/// PNG width and height (from the IHDR chunk).
fn png_size(path: &std::path::Path) -> (u32, u32) {
    let b = std::fs::read(path).unwrap();
    assert!(b.starts_with(b"\x89PNG"));
    let be = |k: usize| u32::from_be_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
    (be(16), be(20))
}

#[test]
fn fields_colour_slices_and_renders() {
    let model = models().join("universes.i");
    let m = model.to_str().unwrap();
    let csv = tmp("field.csv");
    std::fs::write(&csv, "material,heat [W]\n1,1\n2,1000\n").unwrap();
    let csv = csv.to_str().unwrap();
    let png = tmp("field_slice.png");
    let o = llevant(&[
        "slice",
        m,
        "--plane",
        "z=0.3",
        "--px",
        "300",
        "--field",
        csv,
        "-o",
        png.to_str().unwrap(),
    ]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{err}");
    assert!(
        err.contains("field heat [W]: 1 to 1000, logarithmic"),
        "{err}"
    );
    let plain = tmp("plain_slice.png");
    let o = llevant(&[
        "slice",
        m,
        "--plane",
        "z=0.3",
        "--px",
        "300",
        "-o",
        plain.to_str().unwrap(),
    ]);
    assert!(o.status.success());
    // The colour bar goes under the slice (its height as its text needs).
    let (w, h) = png_size(&png);
    let bar = h - png_size(&plain).1;
    assert!(w == 300 && (40..120).contains(&bar), "{w} × {h}");
    let png = tmp("field_render.png");
    let o = llevant(&[
        "render",
        m,
        "--size",
        "120x80",
        "--field",
        csv,
        "--linear",
        "--scale",
        "0,10",
        "--opacity",
        "0.5",
        "--no-bar",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(png_size(&png), (120, 80));
    // A threshold and colours outside the scale; malformed ones refused.
    let styled = |extra: &[&str]| {
        let mut args = vec![
            "slice", m, "--plane", "z=0.3", "--px", "120", "--field", csv,
        ];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["-o", png.to_str().unwrap()]);
        llevant(&args)
    };
    let o = styled(&[
        "--threshold",
        "10,",
        "--above-colour",
        "#ff0000",
        "--below-colour",
        "ffffff",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = styled(&["--exact-ends"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    // The colour bar's style: map, bands, ticks, numbers, title.
    for good in [
        &["--ramp", "turbo", "--bands", "3", "--ticks", "edges"][..],
        &["--ticks", "13", "--numbers", "powers", "--digits", "2"],
        &[
            "--ticks",
            "many",
            "--legend-title",
            "heat",
            "--legend-text",
            "16",
        ],
    ] {
        let o = styled(good);
        assert!(
            o.status.success(),
            "{good:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    for bad in [
        &["--ramp", "jet"][..],
        &["--bands", "1"],
        &["--ticks", "edges"],
        &["--ticks", "1"],
        &["--numbers", "roman"],
        &["--digits", "9"],
        &["--threshold", "10"],
        &["--threshold", "5,1"],
        &["--above-colour", "red"],
        &["--exact-ends", "--scale", "1,2"],
    ] {
        assert_eq!(styled(bad).status.code(), Some(2), "{bad:?}");
    }
    // Bad fields are errors.
    std::fs::write(tmp("bad.csv"), "voxel,x\n1,2\n").unwrap();
    let o = llevant(&[
        "slice",
        m,
        "--field",
        tmp("bad.csv").to_str().unwrap(),
        "-o",
        "x.png",
    ]);
    assert_eq!(o.status.code(), Some(2));
}

/// Outlines between cells, materials or universes: different images, and
/// a choice clap knows.
#[test]
fn slice_outlines_by_material_and_universe() {
    let model = models().join("universes.i");
    let m = model.to_str().unwrap();
    let image = |by: &str| {
        let png = tmp(&format!("outline_{by}.png"));
        let o = llevant(&[
            "slice",
            m,
            "--plane",
            "z=0.3",
            "--px",
            "300",
            "--outline-by",
            by,
            "-o",
            png.to_str().unwrap(),
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        std::fs::read(png).unwrap()
    };
    let cells = image("cell");
    assert_ne!(image("material"), cells);
    assert_ne!(image("universe"), cells);
    assert_eq!(
        llevant(&[
            "slice",
            m,
            "--plane",
            "z=0",
            "--outline-by",
            "surface",
            "-o",
            "x.png"
        ])
        .status
        .code(),
        Some(2)
    );
    assert_eq!(
        llevant(&[
            "slice",
            m,
            "--plane",
            "z=0",
            "--no-outline",
            "--outline-by",
            "material",
            "-o",
            "x.png"
        ])
        .status
        .code(),
        Some(2)
    );
}

/// An MCTAL file as a field (aci_tallies' probe: a ray through a lattice),
/// with bins along other dimensions chosen by `--bin`.
#[test]
fn mctal_fields_colour_slices() {
    let fixtures =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tallies/mctal");
    let model = fixtures.join("probe.mcnp");
    let mctal = fixtures.join("probe.m");
    let png = tmp("mctal_slice.png");
    let slice = |extra: &[&str]| {
        let mut args = vec![
            "slice",
            model.to_str().unwrap(),
            "--plane",
            "z=0.7",
            "--px",
            "200",
            "--field",
            mctal.to_str().unwrap(),
            "-o",
            png.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        llevant(&args)
    };
    let o = slice(&["--tally", "24"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{err}");
    assert!(err.contains("field tally 24 result (n)"), "{err}");
    // Three multiplier bins and no Total: one has to be chosen.
    let o = slice(&["--tally", "54"]);
    assert_eq!(o.status.code(), Some(2));
    let o = slice(&["--tally", "54", "--bin", "m=1"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = slice(&["--tally", "54", "--bin", "q=1"]);
    assert_eq!(o.status.code(), Some(2));
}

/// A universe placed three times, and a universe placed nowhere.
const PLACED: &str = "placed
1 0 -1 fill=1 imp:n=1
2 0 -2 fill=1 (20 0 0) imp:n=1
3 0 -3 fill=1 (0 30 0) imp:n=1
8 0 1 2 3 -9 imp:n=1
9 0 9 imp:n=0
10 3 -7.8 -4 u=1 imp:n=1
11 0 4 u=1 imp:n=1
12 0 -5 u=7 imp:n=1

1 so 5
2 s 20 0 0 5
3 s 0 30 0 5
4 so 2
5 so 1
9 so 100

m3 26000 1
";

/// `bbox` boxes every instance; `render --isolate` frames them.
#[test]
fn bbox_and_isolated_framing() {
    let model = tmp("placed.i");
    std::fs::write(&model, PLACED).unwrap();
    let model = model.to_str().unwrap();
    let o = llevant(&["bbox", model, "--cells", "c 10", "--format", "json"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["instances"].as_array().unwrap().len(), 3);
    assert_eq!(v["exact"], true);
    let lo: Vec<f64> = serde_json::from_value(v["box"]["lo"].clone()).unwrap();
    let hi: Vec<f64> = serde_json::from_value(v["box"]["hi"].clone()).unwrap();
    for (k, (l, h)) in [(-2.0, 22.0), (-2.0, 32.0), (-2.0, 2.0)].iter().enumerate() {
        assert!((lo[k] - l).abs() < 0.1 && (hi[k] - h).abs() < 0.1, "{v}");
    }
    let o = llevant(&["bbox", model, "--cells", "c 10", "--format", "csv"]);
    assert_eq!(stdout(&o).lines().count(), 1 + 3 + 1);
    let o = llevant(&["bbox", model, "--cells", "c 10", "--all"]);
    assert!(stdout(&o).contains("--region=-2.0"), "{}", stdout(&o));
    // Universe 7 is filled nowhere; cell 99 does not exist.
    assert_eq!(
        llevant(&["bbox", model, "--cells", "c 12"]).status.code(),
        Some(1)
    );
    assert!(!llevant(&["bbox", model, "--cells", "c 99"])
        .status
        .success());

    let hit = |args: &[&str]| -> (f64, String) {
        let png = tmp("framed.png");
        let mut a = vec![
            "render",
            model,
            "--size",
            "64x64",
            "-o",
            png.to_str().unwrap(),
        ];
        a.extend_from_slice(args);
        let o = llevant(&a);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let err = String::from_utf8_lossy(&o.stderr).to_string();
        let pc = err
            .split(" of rays hit")
            .next()
            .and_then(|s| s.rsplit('(').next())
            .and_then(|s| s.trim_end_matches('%').parse().ok())
            .unwrap();
        (pc, err)
    };
    // Framed on the three copies of cell 10, not on the 100 cm sphere.
    let (framed, err) = hit(&["--isolate", "c 10"]);
    assert!(err.contains("framing the cells isolated"), "{err}");
    assert!(!err.contains("only "), "{err}");
    // Framed on the 100 cm sphere instead: far fewer, and the message says
    // why.
    let (whole, err) = hit(&["--isolate", "c 10", "--region=-100,-100,-100,100,100,100"]);
    assert!(
        whole < framed / 10.0,
        "{whole}% of rays with the region, {framed}% framed"
    );
    assert!(err.contains("only "), "{err}");
}

/// A plate in the plane x = 0, edge on from y.
const PLATE: &str = "plate
1 3 -7.8 -1 imp:n=1
2 0 1 -9 imp:n=1
9 0 9 imp:n=0

1 rpp -0.5 0.5 -20 20 -20 20
9 so 100

m3 26000 1
";

/// `render --isolate` looks from where the cells show most, unless told.
#[test]
fn isolated_views_face_what_they_show() {
    let model = tmp("plate.i");
    std::fs::write(&model, PLATE).unwrap();
    let png = tmp("plate.png");
    let run = |extra: &[&str]| {
        let mut a = vec!["render", model.to_str().unwrap(), "--isolate", "c 1"];
        a.extend_from_slice(&["--size", "48x48", "-o", png.to_str().unwrap()]);
        a.extend_from_slice(extra);
        let o = llevant(&a);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stderr).to_string()
    };
    let err = run(&[]);
    assert!(
        err.contains("from 1.00,0.00,0.00") || err.contains("from -1.00,0.00,0.00"),
        "{err}"
    );
    let err = run(&["--from", "0,-1,0"]);
    assert!(!err.contains("show more"), "{err}");
}

/// Rules from files (`@path`), and `info --format json`.
#[test]
fn rules_from_files_and_info_json() {
    let model = tmp("placed_rules.i");
    std::fs::write(&model, PLACED).unwrap();
    let model = model.to_str().unwrap();
    let boxed = |cells: &str| -> serde_json::Value {
        let o = llevant(&["bbox", model, "--cells", cells, "--format", "json"]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        serde_json::from_str(&stdout(&o)).unwrap()
    };
    let list = tmp("cells.txt");
    // Numbers on lines, with ranges, commas and comments: those cells.
    std::fs::write(&list, "# the steel\n10\n11, 1-2  # and more\n").unwrap();
    let a = boxed(&format!("@{}", list.display()));
    assert_eq!(a, boxed("c 1 2 10 11"));
    // A rule per line: any of them.
    std::fs::write(&list, "c 10\nm 3 & c 12\n").unwrap();
    assert_eq!(
        boxed(&format!("@{}", list.display())),
        boxed("c 10 | (m 3 & c 12)")
    );
    let o = llevant(&["bbox", model, "--cells", "@/no/such/file"]);
    assert!(!o.status.success());
    // --isolate and --hide-rule read files too.
    std::fs::write(&list, "10\n").unwrap();
    let png = tmp("rules.png");
    let o = llevant(&[
        "render",
        model,
        "--isolate",
        &format!("@{}", list.display()),
        "--hide-rule",
        &format!("@{}", list.display()),
        "--size",
        "16x16",
        "-o",
        png.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stderr).contains("(0% of rays hit)"));

    let o = llevant(&["info", model, "--format", "json"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["cells"], 8);
    assert_eq!(v["universes"], 3);
    assert!(v["extent"]["lo"].is_array() && v["diagnostics"].is_array());
}

/// Cell labels (`$` comments): steel 10 in universe 1, placed three times.
const LABELLED: &str = "labelled
1 0 -1 fill=1 imp:n=1 $ envelope A
2 0 -2 fill=1 (20 0 0) imp:n=1 $ envelope B
8 0 1 2 -9 imp:n=1
9 0 9 imp:n=0
10 3 -7.8 -4 u=1 imp:n=1 $ NB CELL - Steel ball
11 0 4 u=1 imp:n=1 $ NB CELL - void around
12 3 -7.8 -5 u=7 imp:n=1 $ steel nowhere

1 so 5
2 s 20 0 0 5
4 so 2
5 so 1
9 so 100

m3 26000 1
";

/// `label` / `$` rules: read from the input whenever they are used, the
/// compiled model coming from the cache or not.
#[test]
fn label_rules() {
    let model = tmp("labelled.i");
    std::fs::write(&model, LABELLED).unwrap();
    let model = model.to_str().unwrap();
    let cells = |rule: &str, extra: &[&str]| -> (Vec<i64>, String) {
        let mut a = vec!["bbox", model, "--cells", rule, "--format", "json"];
        a.extend_from_slice(extra);
        let o = llevant(&a);
        let err = String::from_utf8_lossy(&o.stderr).to_string();
        if !o.status.success() {
            return (Vec::new(), err);
        }
        let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
        let mut ids: Vec<i64> = v["instances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["cell"].as_i64().unwrap())
            .collect();
        ids.dedup();
        (ids, err)
    };
    // Twice: the second load is from the cache, with no text.
    for _ in 0..2 {
        let (ids, err) = cells(r#"label "steel""#, &[]);
        assert_eq!(ids, [10], "{err}");
        assert!(err.contains("labels of cells"), "{err}");
    }
    assert_eq!(cells("$ nb & m 3", &[]).0, [10]);
    assert_eq!(cells("$ envelope", &[]).0, [1, 2]);
    // Rules without labels do not read them.
    let (_, err) = cells("c 10", &[]);
    assert!(!err.contains("labels"), "{err}");
    // A universe on its own: the same numbers, the same labels.
    assert_eq!(cells("$ nb", &["--universe", "1"]).0, [10, 11]);
    // Isolated by label, framed on it.
    let png = tmp("labelled.png");
    let o = llevant(&[
        "render",
        model,
        "--isolate",
        "$ steel ball",
        "--size",
        "32x32",
        "-o",
        png.to_str().unwrap(),
    ]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        o.status.success() && err.contains("framing the cells isolated"),
        "{err}"
    );
}

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/examples")
}

#[test]
fn volume_mass_reports_text_csv_and_json() {
    let t4 = examples().join("tripoli4/pwr_pin_cell.t4");
    let t4 = t4.to_str().unwrap();

    let o = llevant(&["volume", t4, "--h", "0.2", "--by", "material", "--mass"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let out = stdout(&o);
    assert!(
        out.contains("mass (g)") && out.contains("mass_sigma"),
        "{out}"
    );

    let o = llevant(&[
        "volume", t4, "--h", "0.2", "--by", "material", "--mass", "--format", "csv",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let csv = stdout(&o);
    assert!(
        csv.starts_with("material,volume,sigma,segments,probes_hit,mass_g,mass_sigma_g\n"),
        "{csv}"
    );

    let o = llevant(&[
        "volume", t4, "--h", "0.2", "--by", "material", "--format", "json",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let arr = rows.as_array().unwrap();
    assert!(arr.iter().any(|r| {
        r["material"] == 1
            && r["mass_g"].as_f64().is_some_and(|m| m > 0.0)
            && r["mass_sigma_g"].as_f64().is_some_and(|s| s >= 0.0)
    }));
}

#[test]
fn check_plots_writes_report_and_pngs() {
    let defect_dir = tmp("check_plots_defect");
    let _ = std::fs::remove_dir_all(&defect_dir);
    let periodic = models().join("periodic.i");
    let o = llevant(&[
        "check",
        periodic.to_str().unwrap(),
        "--h",
        "0.5",
        "--plots",
        defect_dir.to_str().unwrap(),
        "--px",
        "64",
    ]);
    assert_eq!(
        o.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let rjson: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(defect_dir.join("report.json")).unwrap())
            .unwrap();
    let arr = rjson.as_array().unwrap();
    assert!(!arr.is_empty());
    let plots = arr[0]["plots"].as_array().unwrap();
    assert_eq!(plots.len(), 3);
    for fname in plots {
        let bytes = std::fs::read(defect_dir.join(fname.as_str().unwrap())).unwrap();
        assert_eq!(&bytes[..4], b"\x89PNG");
    }
    let md = std::fs::read_to_string(defect_dir.join("report.md")).unwrap();
    assert!(
        md.contains("overlap") && md.contains("defect_001_xy.png"),
        "{md}"
    );

    let clean_dir = tmp("check_plots_clean");
    let _ = std::fs::remove_dir_all(&clean_dir);
    let surfaces = models().join("surfaces.i");
    let o = llevant(&[
        "check",
        surfaces.to_str().unwrap(),
        "--h",
        "1",
        "--plots",
        clean_dir.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(0));
    let md = std::fs::read_to_string(clean_dir.join("report.md")).unwrap();
    assert!(md.contains("No gaps or overlaps found."), "{md}");
}

#[test]
fn convert_translates_tripoli_and_dragon_with_verify() {
    let t4 = examples().join("tripoli4/pwr_pin_cell.t4");
    let out_mcnp = tmp("converted_t4.i");
    let o = llevant(&[
        "convert",
        t4.to_str().unwrap(),
        "--verify",
        "-o",
        out_mcnp.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let txt = std::fs::read_to_string(&out_mcnp).unwrap();
    assert!(txt.contains("imp:n=1") && txt.contains("imp:n=0"), "{txt}");

    let d5 = examples().join("dragon5/pwr_3x3_assembly.x2m");
    let o = llevant(&["convert", d5.to_str().unwrap(), "--verify"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let out = stdout(&o);
    assert!(out.contains("lat=1") && out.contains("imp:n=1"), "{out}");
}

#[test]
fn plots_renders_openmc_plots_xml() {
    let openmc_model = examples().join("openmc/fusion_hts_blanket.xml");
    let plots_xml = tmp("cli_plots.xml");
    std::fs::write(
        &plots_xml,
        r#"<?xml version="1.0"?>
<plots>
  <plot id="1" type="slice" basis="xy" color_by="material" filename="hts_xy">
    <origin>0 0 0</origin>
    <width>400 400</width>
    <pixels>64 64</pixels>
  </plot>
  <plot id="2" type="slice" basis="xz" color_by="cell" filename="hts_xz.png">
    <origin>0 0 0</origin>
    <width>400 300</width>
    <pixels>64 48</pixels>
  </plot>
</plots>
"#,
    )
    .unwrap();
    let out_dir = tmp("cli_plots_out");
    let _ = std::fs::remove_dir_all(&out_dir);
    let o = llevant(&[
        "plots",
        openmc_model.to_str().unwrap(),
        "--plots-xml",
        plots_xml.to_str().unwrap(),
        "-O",
        out_dir.to_str().unwrap(),
        "--outlines",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    for name in ["hts_xy.png", "hts_xz.png"] {
        let bytes = std::fs::read(out_dir.join(name)).unwrap();
        assert_eq!(&bytes[..4], b"\x89PNG");
    }
}

#[test]
fn lost_diagnoses_coordinates_and_log_with_plots() {
    let model_path = tmp("lost_test_model.i");
    std::fs::write(
        &model_path,
        "\
lost particle test
1 1 -7.8 -1 imp:n=1
2 2 -1.0 -2 imp:n=1
3 0 3 imp:n=0

1 rpp -2 2 -2 2 -2 2
2 rpp 1 5 -2 2 -2 2
3 rpp -10 10 -10 10 -10 10

mode n
m1 26000 1.0
m2 1001 2.0 8016 1.0
",
    )
    .unwrap();

    let log_path = tmp("lost_mcnp_openmc.log");
    std::fs::write(
        &log_path,
        "\
1   x, y, z =     1.50000E+00   0.00000E+00   0.00000E+00
WARNING: Particle 7 became lost at (-4.0, 0.0, 0.0)
0.0 0.0 0.0
",
    )
    .unwrap();

    let plot_dir = tmp("lost_plots_out");
    let _ = std::fs::remove_dir_all(&plot_dir);
    let o = llevant(&[
        "lost",
        model_path.to_str().unwrap(),
        "--log",
        log_path.to_str().unwrap(),
        "--radius",
        "1.0",
        "--plots",
        plot_dir.to_str().unwrap(),
        "--px",
        "64",
        "--format",
        "json",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    let arr = rows.as_array().unwrap();
    assert_eq!(arr.len(), 3, "{rows}");
    assert_eq!(arr[0]["label"]["status"], "overlap");
    assert_eq!(arr[1]["label"]["status"], "gap");
    assert_eq!(arr[2]["label"]["status"], "inside");
    for fname in arr[0]["plots"].as_array().unwrap() {
        let bytes = std::fs::read(plot_dir.join(fname.as_str().unwrap())).unwrap();
        assert_eq!(&bytes[..4], b"\x89PNG");
    }
}
