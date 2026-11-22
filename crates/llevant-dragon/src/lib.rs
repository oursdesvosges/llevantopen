//! Pure-Rust DRAGON5 (`GEO:`) geometry loader for `llevant`.
//!
//! Translates CLE-2000 `GEO:` module blocks (`CARCEL`, `TUBE`, `HEXCEL`,
//! `CAR2D`, `CAR3D`, and nested `::: sub := GEO: ... ;` sub-geometries) into
//! canonical MCNP syntax understood natively by `migjorn`, `llevant`, and
//! `llevant-scene`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};

/// Errors produced when loading or translating a DRAGON5 `GEO:` input file.
#[derive(Debug)]
pub enum DragonError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse(String),
}

impl fmt::Display for DragonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DragonError::Io { path, source } => {
                write!(f, "reading DRAGON5 input {}: {source}", path.display())
            }
            DragonError::Parse(msg) => write!(f, "DRAGON5 GEO: error: {msg}"),
        }
    }
}

impl std::error::Error for DragonError {}

/// Returns `true` if `path` has a DRAGON5 / CLE-2000 extension (`.x2m`, `.c2m`, `.dragon`, `.d5`).
pub fn is_dragon_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("x2m")
                || ext.eq_ignore_ascii_case("c2m")
                || ext.eq_ignore_ascii_case("dragon")
                || ext.eq_ignore_ascii_case("d5")
        })
}

/// Returns `true` if raw input bytes appear to contain a DRAGON5 `GEO:` module invocation.
pub fn is_dragon_bytes(input: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(input) else {
        return false;
    };
    let toks = tokenize(s);
    toks.windows(2)
        .any(|w| w[0] == ":=" && w[1].eq_ignore_ascii_case("GEO:"))
}

/// Read an input file (DRAGON5 or MCNP) and return MCNP-compatible bytes.
pub fn read_input_bytes(path: &Path) -> Result<Vec<u8>, DragonError> {
    let bytes = fs::read(path).map_err(|source| DragonError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if is_dragon_path(path) || is_dragon_bytes(&bytes) {
        let text = String::from_utf8_lossy(&bytes);
        let mcnp = dragon_to_mcnp(&text)?;
        return Ok(mcnp.into_bytes());
    }
    Ok(bytes)
}

/// If `input` contains a DRAGON5 `GEO:` deck, translate it into canonical MCNP bytes;
/// otherwise borrow `input` untouched with zero allocation.
pub fn maybe_translate_bytes(input: &[u8]) -> Result<Cow<'_, [u8]>, DragonError> {
    if !is_dragon_bytes(input) {
        return Ok(Cow::Borrowed(input));
    }
    let text = String::from_utf8_lossy(input);
    let mcnp = dragon_to_mcnp(&text)?;
    Ok(Cow::Owned(mcnp.into_bytes()))
}

// ---------------------------------------------------------------------------
// Tokenization & Comment Stripping (CLE-2000)
// ---------------------------------------------------------------------------

fn strip_comments(src: &str) -> String {
    // Remove CLE-2000 `(* ... *)` comments and lines whose first non-space char is `*`.
    let mut no_stars = String::with_capacity(src.len());
    for line in src.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('*') {
            no_stars.push('\n');
        } else {
            no_stars.push_str(line);
            no_stars.push('\n');
        }
    }

    let mut out = String::with_capacity(no_stars.len());
    let bytes = no_stars.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'(' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b')') {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            }
            out.push(' ');
        } else {
            let ch = no_stars[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn tokenize(src: &str) -> Vec<String> {
    let clean = strip_comments(src);
    let mut out = Vec::new();
    for raw in clean.split_whitespace() {
        // Keep `;` as its own token while preserving `GEO:`, `::`, `:::`, and `:=`.
        let mut cur = raw;
        while let Some(pos) = cur.find(';') {
            let before = &cur[..pos];
            if !before.is_empty() {
                out.push(before.to_string());
            }
            out.push(";".to_string());
            cur = &cur[pos + 1..];
        }
        if !cur.is_empty() {
            out.push(cur.to_string());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Geometry IR
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum DragonKind {
    Carcel,
    Tube,
    Hexcel,
    Car2d { nx: usize, ny: usize },
    Car3d { nx: usize, ny: usize, nz: usize },
}

#[derive(Debug, Clone)]
struct DragonGeom {
    name: String,
    kind: DragonKind,
    meshx: Vec<f64>,
    meshy: Vec<f64>,
    meshz: Vec<f64>,
    radii: Vec<f64>,
    side: Option<f64>,
    offcenter: [f64; 2],
    mix: Vec<i64>,
    cells: Vec<String>,
    sub_geoms: HashMap<String, DragonGeom>,
}

fn is_keyword(tok: &str) -> bool {
    let u = tok.to_ascii_uppercase();
    matches!(
        u.as_str(),
        ";" | ":::"
            | "::"
            | "CARCEL"
            | "TUBE"
            | "TUBEZ"
            | "HEXCEL"
            | "CAR2D"
            | "CAR3D"
            | "MESHX"
            | "MESHY"
            | "MESHZ"
            | "RADIUS"
            | "SIDE"
            | "OFFCENTER"
            | "MIX"
            | "CELL"
            | "SPLITX"
            | "SPLITY"
            | "SPLITZ"
            | "SPLITR"
            | "SECT"
            | "X-"
            | "X+"
            | "Y-"
            | "Y+"
            | "Z-"
            | "Z+"
            | "HBC"
            | "DIAG"
            | "COMB"
            | "TURN"
            | "BIHET"
    )
}

fn parse_geo_block(
    name: String,
    toks: &[String],
    idx: &mut usize,
) -> Result<DragonGeom, DragonError> {
    if toks.get(*idx).is_some_and(|t| t == "::") {
        *idx += 1;
    }

    let kind_tok = toks
        .get(*idx)
        .ok_or_else(|| DragonError::Parse(format!("GEO: `{name}` missing geometry type")))?
        .to_ascii_uppercase();
    *idx += 1;

    let kind = match kind_tok.as_str() {
        "CARCEL" => {
            while toks
                .get(*idx)
                .and_then(|t| t.parse::<usize>().ok())
                .is_some()
            {
                *idx += 1;
            }
            DragonKind::Carcel
        }
        "TUBE" | "TUBEZ" => {
            while toks
                .get(*idx)
                .and_then(|t| t.parse::<usize>().ok())
                .is_some()
            {
                *idx += 1;
            }
            DragonKind::Tube
        }
        "HEXCEL" => {
            while toks
                .get(*idx)
                .and_then(|t| t.parse::<usize>().ok())
                .is_some()
            {
                *idx += 1;
            }
            DragonKind::Hexcel
        }
        "CAR2D" => {
            let nx: usize = toks
                .get(*idx)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| DragonError::Parse(format!("GEO: `{name}` CAR2D missing nx")))?;
            let ny: usize = toks
                .get(*idx + 1)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| DragonError::Parse(format!("GEO: `{name}` CAR2D missing ny")))?;
            *idx += 2;
            DragonKind::Car2d { nx, ny }
        }
        "CAR3D" => {
            let nx: usize = toks
                .get(*idx)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| DragonError::Parse(format!("GEO: `{name}` CAR3D missing nx")))?;
            let ny: usize = toks
                .get(*idx + 1)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| DragonError::Parse(format!("GEO: `{name}` CAR3D missing ny")))?;
            let nz: usize = toks
                .get(*idx + 2)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| DragonError::Parse(format!("GEO: `{name}` CAR3D missing nz")))?;
            *idx += 3;
            DragonKind::Car3d { nx, ny, nz }
        }
        other => {
            return Err(DragonError::Parse(format!(
                "unsupported DRAGON5 geometry type `{other}` on `{name}`"
            )));
        }
    };

    let mut geom = DragonGeom {
        name,
        kind,
        meshx: Vec::new(),
        meshy: Vec::new(),
        meshz: Vec::new(),
        radii: Vec::new(),
        side: None,
        offcenter: [0.0, 0.0],
        mix: Vec::new(),
        cells: Vec::new(),
        sub_geoms: HashMap::new(),
    };

    while *idx < toks.len() {
        let cur = toks[*idx].as_str();
        if cur == ";" {
            *idx += 1;
            break;
        }
        if cur == ":::" {
            *idx += 1;
            let sub_name = toks
                .get(*idx)
                .cloned()
                .ok_or_else(|| DragonError::Parse("missing sub-geometry name after :::".into()))?;
            *idx += 1;
            if !toks.get(*idx).is_some_and(|t| t == ":=")
                || !toks
                    .get(*idx + 1)
                    .is_some_and(|t| t.eq_ignore_ascii_case("GEO:"))
            {
                return Err(DragonError::Parse(format!(
                    "expected `:= GEO:` after sub-geometry `::: {sub_name}`"
                )));
            }
            *idx += 2;
            let sub = parse_geo_block(sub_name.clone(), toks, idx)?;
            geom.sub_geoms.insert(sub_name.to_ascii_uppercase(), sub);
            continue;
        }

        let kw = cur.to_ascii_uppercase();
        *idx += 1;
        match kw.as_str() {
            "MESHX" => {
                while let Some(v) = toks.get(*idx).and_then(|t| t.parse::<f64>().ok()) {
                    geom.meshx.push(v);
                    *idx += 1;
                }
            }
            "MESHY" => {
                while let Some(v) = toks.get(*idx).and_then(|t| t.parse::<f64>().ok()) {
                    geom.meshy.push(v);
                    *idx += 1;
                }
            }
            "MESHZ" => {
                while let Some(v) = toks.get(*idx).and_then(|t| t.parse::<f64>().ok()) {
                    geom.meshz.push(v);
                    *idx += 1;
                }
            }
            "RADIUS" => {
                while let Some(v) = toks.get(*idx).and_then(|t| t.parse::<f64>().ok()) {
                    if v > 0.0 {
                        geom.radii.push(v);
                    }
                    *idx += 1;
                }
            }
            "SIDE" => {
                let s = toks
                    .get(*idx)
                    .and_then(|t| t.parse::<f64>().ok())
                    .ok_or_else(|| DragonError::Parse("SIDE missing numeric value".into()))?;
                geom.side = Some(s);
                *idx += 1;
            }
            "OFFCENTER" => {
                let dx = toks
                    .get(*idx)
                    .and_then(|t| t.parse::<f64>().ok())
                    .ok_or_else(|| DragonError::Parse("OFFCENTER missing dx".into()))?;
                let dy = toks
                    .get(*idx + 1)
                    .and_then(|t| t.parse::<f64>().ok())
                    .ok_or_else(|| DragonError::Parse("OFFCENTER missing dy".into()))?;
                geom.offcenter = [dx, dy];
                *idx += 2;
            }
            "MIX" => {
                while let Some(v) = toks.get(*idx).and_then(|t| t.parse::<i64>().ok()) {
                    geom.mix.push(v);
                    *idx += 1;
                }
            }
            "CELL" => {
                while let Some(t) = toks.get(*idx) {
                    if is_keyword(t) {
                        break;
                    }
                    geom.cells.push(t.to_string());
                    *idx += 1;
                }
            }
            "X-" | "X+" | "Y-" | "Y+" | "Z-" | "Z+" | "HBC" => {
                // Skip boundary condition token + optional numeric albedo
                if toks.get(*idx).is_some_and(|t| !is_keyword(t)) {
                    *idx += 1;
                }
                if toks
                    .get(*idx)
                    .is_some_and(|t| t.parse::<f64>().is_ok() && !is_keyword(t))
                {
                    *idx += 1;
                }
            }
            "SPLITX" | "SPLITY" | "SPLITZ" | "SPLITR" | "SECT" | "TURN" => {
                while toks.get(*idx).is_some_and(|t| !is_keyword(t)) {
                    *idx += 1;
                }
            }
            _ => {}
        }
    }

    Ok(geom)
}

// ---------------------------------------------------------------------------
// MCNP Emission Helpers
// ---------------------------------------------------------------------------

fn fmt_num(v: f64) -> String {
    if v == 0.0 || v.abs() < 1e-15 {
        return "0".to_string();
    }
    let s = format!("{v:.10}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

struct McnpEmitter {
    cells: Vec<String>,
    surfaces: Vec<String>,
    transforms: Vec<String>,
    next_cell_id: i64,
    next_surf_id: i64,
    next_univ_id: i64,
    next_tr_id: i64,
}

impl McnpEmitter {
    fn new() -> Self {
        Self {
            cells: Vec::new(),
            surfaces: Vec::new(),
            transforms: Vec::new(),
            next_cell_id: 1,
            next_surf_id: 1,
            next_univ_id: 1,
            next_tr_id: 1,
        }
    }

    fn alloc_cell(&mut self) -> i64 {
        let id = self.next_cell_id;
        self.next_cell_id += 1;
        id
    }

    fn alloc_surf(&mut self) -> i64 {
        let id = self.next_surf_id;
        self.next_surf_id += 1;
        id
    }

    fn alloc_univ(&mut self) -> i64 {
        let id = self.next_univ_id;
        self.next_univ_id += 1;
        id
    }

    fn alloc_tr(&mut self, dx: f64, dy: f64, dz: f64) -> i64 {
        let id = self.next_tr_id;
        self.next_tr_id += 1;
        self.transforms.push(format!(
            "tr{id} {} {} {}",
            fmt_num(dx),
            fmt_num(dy),
            fmt_num(dz)
        ));
        id
    }

    fn push_surf(&mut self, mnem: &str, coeffs: &[f64]) -> i64 {
        let id = self.alloc_surf();
        let cstr: Vec<String> = coeffs.iter().map(|&v| fmt_num(v)).collect();
        self.surfaces
            .push(format!("{id} {mnem} {}", cstr.join(" ")));
        id
    }

    fn push_cell(&mut self, mat: i64, region: &str, u: Option<i64>, comment: &str) {
        let cid = self.alloc_cell();
        let rho = if mat > 0 { " -1" } else { "" };
        let u_clause = u.map_or(String::new(), |uv| format!(" u={uv}"));
        self.cells.push(format!(
            "{cid} {mat}{rho} {region}{u_clause} imp:n=1 $ {comment}"
        ));
    }
}

fn bounds_from_mesh(mesh: &[f64], label: &str) -> Result<(f64, f64), DragonError> {
    if mesh.len() < 2 {
        return Err(DragonError::Parse(format!(
            "{label} requires at least 2 monotonically increasing coordinates"
        )));
    }
    for w in mesh.windows(2) {
        if !(w[0] < w[1]) {
            return Err(DragonError::Parse(format!(
                "{label} coordinates must strictly increase: {} >= {}",
                w[0], w[1]
            )));
        }
    }
    Ok((mesh[0], mesh[mesh.len() - 1]))
}

fn validate_radii(radii: &[f64], name: &str) -> Result<(), DragonError> {
    for w in radii.windows(2) {
        if !(w[0] < w[1]) {
            return Err(DragonError::Parse(format!(
                "`{name}` RADIUS entries must strictly increase: {} >= {}",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

type GeomSpan = ([f64; 2], [f64; 2], Vec<f64>);

/// Center `(cx, cy, cz)` and bounding box of a single pin/cell geometry.
fn geom_span(g: &DragonGeom) -> Result<GeomSpan, DragonError> {
    let (x0, x1) = bounds_from_mesh(&g.meshx, &format!("{} MESHX", g.name))?;
    let (y0, y1) = bounds_from_mesh(&g.meshy, &format!("{} MESHY", g.name))?;
    let z = if g.meshz.is_empty() {
        vec![-10.0, 10.0]
    } else {
        bounds_from_mesh(&g.meshz, &format!("{} MESHZ", g.name))?;
        g.meshz.clone()
    };
    Ok(([x0, x1], [y0, y1], z))
}

fn emit_carcel(
    g: &DragonGeom,
    em: &mut McnpEmitter,
    in_univ: Option<i64>,
    center_local: bool,
) -> Result<Option<[i64; 6]>, DragonError> {
    let ([x0, x1], [y0, y1], znodes) = geom_span(g)?;
    validate_radii(&g.radii, &g.name)?;
    let nz = znodes.len() - 1;
    let nr = g.radii.len();
    let nx = g.meshx.len() - 1;
    let ny = g.meshy.len() - 1;

    let shift_x = if center_local { 0.5 * (x0 + x1) } else { 0.0 };
    let shift_y = if center_local { 0.5 * (y0 + y1) } else { 0.0 };
    let shift_z = if center_local {
        0.5 * (znodes[0] + *znodes.last().unwrap())
    } else {
        0.0
    };
    let cx = 0.5 * (x0 + x1) - shift_x + g.offcenter[0];
    let cy = 0.5 * (y0 + y1) - shift_y + g.offcenter[1];

    let simple_expected = (nr + 1) * nz;
    let split_expected = (nr + 1) * nx * ny * nz;
    if g.mix.len() != simple_expected && !((nx * ny > 1) && g.mix.len() == split_expected) {
        return Err(DragonError::Parse(format!(
            "CARCEL `{}` MIX has {} entries, expected {simple_expected} ({} rings + moderator across {nz} z-slices){}",
            g.name,
            g.mix.len(),
            nr,
            if nx * ny > 1 {
                format!(" or {split_expected} for subdivided mesh")
            } else {
                String::new()
            }
        )));
    }

    let csurfs: Vec<i64> = g
        .radii
        .iter()
        .map(|&r| em.push_surf("c/z", &[cx, cy, r]))
        .collect();
    let xsurfs: Vec<i64> = g
        .meshx
        .iter()
        .map(|&x| em.push_surf("px", &[x - shift_x]))
        .collect();
    let ysurfs: Vec<i64> = g
        .meshy
        .iter()
        .map(|&y| em.push_surf("py", &[y - shift_y]))
        .collect();
    let zsurfs: Vec<i64> = znodes
        .iter()
        .map(|&z| em.push_surf("pz", &[z - shift_z]))
        .collect();

    if g.mix.len() == simple_expected {
        let bx_lo = xsurfs[0];
        let bx_hi = *xsurfs.last().unwrap();
        let by_lo = ysurfs[0];
        let by_hi = *ysurfs.last().unwrap();
        let mut mix_idx = 0usize;
        for iz in 0..nz {
            let zbox = format!("{} -{}", zsurfs[iz], zsurfs[iz + 1]);
            for ir in 0..nr {
                let mid = g.mix[mix_idx];
                mix_idx += 1;
                let rad_expr = if ir == 0 {
                    format!("-{} {bx_lo} -{bx_hi} {by_lo} -{by_hi}", csurfs[0])
                } else {
                    format!(
                        "{} -{} {bx_lo} -{bx_hi} {by_lo} -{by_hi}",
                        csurfs[ir - 1],
                        csurfs[ir]
                    )
                };
                em.push_cell(
                    mid,
                    &format!("{rad_expr} {zbox}"),
                    in_univ,
                    &format!("{} ring {ir} z={iz} mix={mid}", g.name),
                );
            }
            let mod_mid = g.mix[mix_idx];
            mix_idx += 1;
            let outer = if let Some(&last_c) = csurfs.last() {
                format!("{last_c} {bx_lo} -{bx_hi} {by_lo} -{by_hi} {zbox}")
            } else {
                format!("{bx_lo} -{bx_hi} {by_lo} -{by_hi} {zbox}")
            };
            em.push_cell(
                mod_mid,
                &outer,
                in_univ,
                &format!("{} moderator z={iz} mix={mod_mid}", g.name),
            );
        }
    } else {
        // Subdivided quadrant CARCEL
        let mut mix_idx = 0usize;
        for iz in 0..nz {
            let zbox = format!("{} -{}", zsurfs[iz], zsurfs[iz + 1]);
            for iy in 0..ny {
                for ix in 0..nx {
                    let qbox = format!(
                        "{} -{} {} -{} {zbox}",
                        xsurfs[ix],
                        xsurfs[ix + 1],
                        ysurfs[iy],
                        ysurfs[iy + 1]
                    );
                    for ir in 0..nr {
                        let mid = g.mix[mix_idx];
                        mix_idx += 1;
                        let rad = if ir == 0 {
                            format!("-{} {qbox}", csurfs[0])
                        } else {
                            format!("{} -{} {qbox}", csurfs[ir - 1], csurfs[ir])
                        };
                        em.push_cell(
                            mid,
                            &rad,
                            in_univ,
                            &format!("{} ({ix},{iy},{iz}) ring {ir} mix={mid}", g.name),
                        );
                    }
                    let mod_mid = g.mix[mix_idx];
                    mix_idx += 1;
                    let outer = if let Some(&last_c) = csurfs.last() {
                        format!("{last_c} {qbox}")
                    } else {
                        qbox
                    };
                    em.push_cell(
                        mod_mid,
                        &outer,
                        in_univ,
                        &format!("{} ({ix},{iy},{iz}) mod mix={mod_mid}", g.name),
                    );
                }
            }
        }
    }

    if in_univ.is_none() {
        Ok(Some([
            xsurfs[0],
            *xsurfs.last().unwrap(),
            ysurfs[0],
            *ysurfs.last().unwrap(),
            zsurfs[0],
            *zsurfs.last().unwrap(),
        ]))
    } else {
        Ok(None)
    }
}

fn emit_tube(g: &DragonGeom, em: &mut McnpEmitter) -> Result<Vec<String>, DragonError> {
    validate_radii(&g.radii, &g.name)?;
    if g.radii.is_empty() {
        return Err(DragonError::Parse(format!(
            "TUBE `{}` requires at least one positive RADIUS",
            g.name
        )));
    }
    let znodes = if g.meshz.is_empty() {
        vec![-10.0, 10.0]
    } else {
        bounds_from_mesh(&g.meshz, &format!("{} MESHZ", g.name))?;
        g.meshz.clone()
    };
    let nr = g.radii.len();
    let nz = znodes.len() - 1;
    let expected = nr * nz;
    if g.mix.len() != expected {
        return Err(DragonError::Parse(format!(
            "TUBE `{}` MIX has {} entries, expected {expected} ({nr} rings x {nz} z-slices)",
            g.name,
            g.mix.len()
        )));
    }

    let csurfs: Vec<i64> = g.radii.iter().map(|&r| em.push_surf("cz", &[r])).collect();
    let zsurfs: Vec<i64> = znodes.iter().map(|&z| em.push_surf("pz", &[z])).collect();
    let mut mix_idx = 0usize;
    for iz in 0..nz {
        let zbox = format!("{} -{}", zsurfs[iz], zsurfs[iz + 1]);
        for ir in 0..nr {
            let mid = g.mix[mix_idx];
            mix_idx += 1;
            let expr = if ir == 0 {
                format!("-{} {zbox}", csurfs[0])
            } else {
                format!("{} -{} {zbox}", csurfs[ir - 1], csurfs[ir])
            };
            em.push_cell(
                mid,
                &expr,
                None,
                &format!("{} TUBE ring {ir} mix={mid}", g.name),
            );
        }
    }

    Ok(vec![
        csurfs.last().unwrap().to_string(),
        format!("-{}", zsurfs[0]),
        zsurfs.last().unwrap().to_string(),
    ])
}

fn emit_hexcel(g: &DragonGeom, em: &mut McnpEmitter) -> Result<String, DragonError> {
    let side = g
        .side
        .ok_or_else(|| DragonError::Parse(format!("HEXCEL `{}` missing SIDE parameter", g.name)))?;
    validate_radii(&g.radii, &g.name)?;
    let znodes = if g.meshz.is_empty() {
        vec![-10.0, 10.0]
    } else {
        bounds_from_mesh(&g.meshz, &format!("{} MESHZ", g.name))?;
        g.meshz.clone()
    };
    let nr = g.radii.len();
    let nz = znodes.len() - 1;
    let expected = (nr + 1) * nz;
    if g.mix.len() != expected {
        return Err(DragonError::Parse(format!(
            "HEXCEL `{}` MIX has {} entries, expected {expected} ({nr} rings + moderator across {nz} z-slices)",
            g.name,
            g.mix.len()
        )));
    }

    let z0 = znodes[0];
    let dz = *znodes.last().unwrap() - z0;
    let r_flat = side * 3.0_f64.sqrt() * 0.5;
    let hex_surf = em.push_surf("rhp", &[0.0, 0.0, z0, 0.0, 0.0, dz, r_flat, 0.0, 0.0]);
    let csurfs: Vec<i64> = g.radii.iter().map(|&r| em.push_surf("cz", &[r])).collect();
    let zsurfs: Vec<i64> = znodes.iter().map(|&z| em.push_surf("pz", &[z])).collect();

    let mut mix_idx = 0usize;
    for iz in 0..nz {
        let zbox = format!("{} -{} -{hex_surf}", zsurfs[iz], zsurfs[iz + 1]);
        for ir in 0..nr {
            let mid = g.mix[mix_idx];
            mix_idx += 1;
            let expr = if ir == 0 {
                format!("-{} {zbox}", csurfs[0])
            } else {
                format!("{} -{} {zbox}", csurfs[ir - 1], csurfs[ir])
            };
            em.push_cell(
                mid,
                &expr,
                None,
                &format!("{} HEXCEL ring {ir} mix={mid}", g.name),
            );
        }
        let mod_mid = g.mix[mix_idx];
        mix_idx += 1;
        let outer = if let Some(&last_c) = csurfs.last() {
            format!("{last_c} {zbox}")
        } else {
            zbox
        };
        em.push_cell(
            mod_mid,
            &outer,
            None,
            &format!("{} HEXCEL mod mix={mod_mid}", g.name),
        );
    }

    Ok(hex_surf.to_string())
}

fn emit_cartesian_grid(
    g: &DragonGeom,
    nx: usize,
    ny: usize,
    nz: usize,
    em: &mut McnpEmitter,
    global_subs: &HashMap<String, DragonGeom>,
) -> Result<[i64; 6], DragonError> {
    if !g.cells.is_empty() {
        // Heterogeneous lattice referenced via `CELL U1 U2 ...`
        let expected = nx * ny * nz;
        if g.cells.len() != expected {
            return Err(DragonError::Parse(format!(
                "`{}` CELL list has {} entries, expected {expected} ({nx}x{ny}x{nz})",
                g.name,
                g.cells.len()
            )));
        }

        let mut merged_subs: HashMap<String, &DragonGeom> = HashMap::new();
        for (k, v) in global_subs {
            merged_subs.insert(k.clone(), v);
        }
        for (k, v) in &g.sub_geoms {
            merged_subs.insert(k.clone(), v);
        }

        let mut univ_by_name: HashMap<String, i64> = HashMap::new();
        let mut pitch: Option<[f64; 3]> = None;

        for cname in &g.cells {
            let key = cname.to_ascii_uppercase();
            if univ_by_name.contains_key(&key) {
                continue;
            }
            let Some(&sub) = merged_subs.get(&key) else {
                return Err(DragonError::Parse(format!(
                    "`{}` references undefined sub-geometry `{cname}`",
                    g.name
                )));
            };
            let ([x0, x1], [y0, y1], znodes) = geom_span(sub)?;
            let dx = x1 - x0;
            let dy = y1 - y0;
            let dz = *znodes.last().unwrap() - znodes[0];
            if pitch.is_none() {
                pitch = Some([dx, dy, dz]);
            }
            let uid = em.alloc_univ();
            emit_carcel(sub, em, Some(uid), true)?;
            univ_by_name.insert(key, uid);
        }

        let [px, py, pz] = pitch.unwrap_or([1.0, 1.0, 20.0]);
        let u_lat = em.alloc_univ();

        let (x_origin, y_origin, z_origin) = (
            g.meshx.first().copied().unwrap_or(0.0),
            g.meshy.first().copied().unwrap_or(0.0),
            g.meshz.first().copied().unwrap_or(-0.5 * pz),
        );

        let ux0 = em.push_surf("px", &[-0.5 * px]);
        let ux1 = em.push_surf("px", &[0.5 * px]);
        let uy0 = em.push_surf("py", &[-0.5 * py]);
        let uy1 = em.push_surf("py", &[0.5 * py]);
        let uz0 = em.push_surf("pz", &[-0.5 * pz]);
        let uz1 = em.push_surf("pz", &[0.5 * pz]);

        let lat_cid = em.alloc_cell();
        let fills: Vec<String> = g
            .cells
            .iter()
            .map(|c| univ_by_name[&c.to_ascii_uppercase()].to_string())
            .collect();
        em.cells.push(format!(
            "{lat_cid} 0 -{ux1} {ux0} -{uy1} {uy0} -{uz1} {uz0} u={u_lat} lat=1 imp:n=1 fill=0:{} 0:{} 0:{} {}",
            nx - 1,
            ny - 1,
            nz - 1,
            fills.join(" ")
        ));

        let gx0 = em.push_surf("px", &[x_origin]);
        let gx1 = em.push_surf("px", &[x_origin + nx as f64 * px]);
        let gy0 = em.push_surf("py", &[y_origin]);
        let gy1 = em.push_surf("py", &[y_origin + ny as f64 * py]);
        let gz0 = em.push_surf("pz", &[z_origin]);
        let gz1 = em.push_surf("pz", &[z_origin + nz as f64 * pz]);

        let tr_id = em.alloc_tr(
            x_origin + 0.5 * px,
            y_origin + 0.5 * py,
            z_origin + 0.5 * pz,
        );
        let env_cid = em.alloc_cell();
        em.cells.push(format!(
            "{env_cid} 0 {gx0} -{gx1} {gy0} -{gy1} {gz0} -{gz1} fill={u_lat} ({tr_id}) imp:n=1 $ {} assembly",
            g.name
        ));

        return Ok([gx0, gx1, gy0, gy1, gz0, gz1]);
    }

    // Homogeneous grid with MIX
    let expected = nx * ny * nz;
    if g.mix.len() != expected {
        return Err(DragonError::Parse(format!(
            "`{}` MIX has {} entries, expected {expected} ({nx}x{ny}x{nz})",
            g.name,
            g.mix.len()
        )));
    }
    if g.meshx.len() != nx + 1 || g.meshy.len() != ny + 1 {
        return Err(DragonError::Parse(format!(
            "`{}` expects {} MESHX nodes and {} MESHY nodes",
            g.name,
            nx + 1,
            ny + 1
        )));
    }
    bounds_from_mesh(&g.meshx, "MESHX")?;
    bounds_from_mesh(&g.meshy, "MESHY")?;
    let znodes = if g.meshz.is_empty() {
        vec![-10.0, 10.0]
    } else {
        if g.meshz.len() != nz + 1 {
            return Err(DragonError::Parse(format!(
                "`{}` expects {} MESHZ nodes",
                g.name,
                nz + 1
            )));
        }
        bounds_from_mesh(&g.meshz, "MESHZ")?;
        g.meshz.clone()
    };

    let xsurfs: Vec<i64> = g.meshx.iter().map(|&x| em.push_surf("px", &[x])).collect();
    let ysurfs: Vec<i64> = g.meshy.iter().map(|&y| em.push_surf("py", &[y])).collect();
    let zsurfs: Vec<i64> = znodes.iter().map(|&z| em.push_surf("pz", &[z])).collect();

    let mut mix_idx = 0usize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mid = g.mix[mix_idx];
                mix_idx += 1;
                let region = format!(
                    "{} -{} {} -{} {} -{}",
                    xsurfs[ix],
                    xsurfs[ix + 1],
                    ysurfs[iy],
                    ysurfs[iy + 1],
                    zsurfs[iz],
                    zsurfs[iz + 1]
                );
                em.push_cell(
                    mid,
                    &region,
                    None,
                    &format!("{} ({ix},{iy},{iz}) mix={mid}", g.name),
                );
            }
        }
    }

    Ok([
        xsurfs[0],
        *xsurfs.last().unwrap(),
        ysurfs[0],
        *ysurfs.last().unwrap(),
        zsurfs[0],
        *zsurfs.last().unwrap(),
    ])
}

/// Translate a DRAGON5 CLE-2000 input file containing `GEO:` definitions
/// into canonical MCNP input text ready for `llevant`.
pub fn dragon_to_mcnp(src: &str) -> Result<String, DragonError> {
    let toks = tokenize(src);
    let mut geoms: Vec<DragonGeom> = Vec::new();
    let mut i = 0usize;

    while i + 1 < toks.len() {
        if toks[i + 1] == ":="
            && toks
                .get(i + 2)
                .is_some_and(|t| t.eq_ignore_ascii_case("GEO:"))
        {
            let name = toks[i].clone();
            i += 3;
            geoms.push(parse_geo_block(name, &toks, &mut i)?);
        } else {
            i += 1;
        }
    }

    if geoms.is_empty() {
        return Err(DragonError::Parse(
            "no `:= GEO:` module definitions found in DRAGON5 input".into(),
        ));
    }

    let root = geoms.pop().unwrap();
    let mut global_subs = HashMap::new();
    for g in geoms {
        global_subs.insert(g.name.to_ascii_uppercase(), g);
    }

    let mut em = McnpEmitter::new();
    let ext_terms: Vec<String> = match root.kind {
        DragonKind::Carcel => {
            let b = emit_carcel(&root, &mut em, None, false)?.unwrap();
            vec![
                format!("-{}", b[0]),
                b[1].to_string(),
                format!("-{}", b[2]),
                b[3].to_string(),
                format!("-{}", b[4]),
                b[5].to_string(),
            ]
        }
        DragonKind::Tube => emit_tube(&root, &mut em)?,
        DragonKind::Hexcel => vec![emit_hexcel(&root, &mut em)?],
        DragonKind::Car2d { nx, ny } => {
            let b = emit_cartesian_grid(&root, nx, ny, 1, &mut em, &global_subs)?;
            vec![
                format!("-{}", b[0]),
                b[1].to_string(),
                format!("-{}", b[2]),
                b[3].to_string(),
                format!("-{}", b[4]),
                b[5].to_string(),
            ]
        }
        DragonKind::Car3d { nx, ny, nz } => {
            let b = emit_cartesian_grid(&root, nx, ny, nz, &mut em, &global_subs)?;
            vec![
                format!("-{}", b[0]),
                b[1].to_string(),
                format!("-{}", b[2]),
                b[3].to_string(),
                format!("-{}", b[4]),
                b[5].to_string(),
            ]
        }
    };

    let grave_id = (em.next_cell_id + 100).max(99999);
    let grave_expr = if ext_terms.len() == 1 {
        ext_terms[0].clone()
    } else {
        format!("({})", ext_terms.join(" : "))
    };
    em.cells.push(format!(
        "{grave_id} 0 {grave_expr} imp:n=0 $ DRAGON5 exterior graveyard"
    ));

    let mut mcnp = String::with_capacity(2048);
    let _ = writeln!(
        mcnp,
        "DRAGON5 GEO: `{}` loaded natively by llevant-dragon",
        root.name
    );
    for c in &em.cells {
        mcnp.push_str(c);
        mcnp.push('\n');
    }
    mcnp.push('\n');
    for s in &em.surfaces {
        mcnp.push_str(s);
        mcnp.push('\n');
    }
    mcnp.push_str("\nmode n\n");
    for tr in &em.transforms {
        mcnp.push_str(tr);
        mcnp.push('\n');
    }
    Ok(mcnp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_carcel_and_validates_mix_count() {
        let good = r#"
PIN := GEO: :: CARCEL 2
  X- REFL X+ REFL Y- REFL Y+ REFL
  MESHX 0.0 1.26 MESHY 0.0 1.26
  RADIUS 0.0 0.41 0.475
  MIX 1 2 3 ;
"#;
        assert!(is_dragon_bytes(good.as_bytes()));
        let mcnp = dragon_to_mcnp(good).expect("translate CARCEL");
        assert!(mcnp.contains("mix=1"));
        assert!(mcnp.contains("mix=2"));
        assert!(mcnp.contains("mix=3"));
        assert!(mcnp.contains("imp:n=0"));

        let bad_mix = r#"
BAD := GEO: :: CARCEL 2
  MESHX 0.0 1.26 MESHY 0.0 1.26
  RADIUS 0.0 0.41 0.475
  MIX 1 2 ;
"#;
        let err = dragon_to_mcnp(bad_mix).unwrap_err().to_string();
        assert!(err.contains("expected 3"), "{err}");
    }

    #[test]
    fn translates_car2d_lattice_and_hexcel_and_tube() {
        let ass = include_str!("../../../docs/examples/dragon5/pwr_3x3_assembly.x2m");
        let mcnp = dragon_to_mcnp(ass).expect("translate pwr_3x3_assembly.x2m");
        assert!(mcnp.contains("lat=1"));
        assert!(mcnp.contains("fill=0:2 0:2 0:0"));

        let hex = r#"
H := GEO: :: HEXCEL 2
  SIDE 1.0713
  RADIUS 0.0 0.40 0.45
  MIX 1 2 3 ;
"#;
        let h_mcnp = dragon_to_mcnp(hex).expect("translate HEXCEL");
        assert!(h_mcnp.contains("rhp 0 0 -10 0 0 20"));

        let tube = r#"
T := GEO: :: TUBE 2
  RADIUS 0.0 0.5 1.0
  MESHZ 0.0 15.0
  MIX 10 20 ;
"#;
        let t_mcnp = dragon_to_mcnp(tube).expect("translate TUBE");
        assert!(t_mcnp.contains("cz 0.5"));
        assert!(t_mcnp.contains("1 10 -1"));
    }
}
