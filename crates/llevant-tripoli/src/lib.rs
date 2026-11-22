//! Pure-Rust TRIPOLI-4 geometry and composition loader for `llevant`.
//!
//! Translates TRIPOLI-4 input decks (`GEOMETRY` ... `ENDG`/`END_GEOMETRY`,
//! `SURF`, `TYPE`, `VOLU`, `BOUNDARY_CONDITION`/`LIMITE`, `COMPOSITION`, and
//! `GEOMCOMP`) into canonical MCNP syntax understood natively by `migjorn`,
//! `llevant`, and `llevant-scene`.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};

/// Errors produced when loading or translating a TRIPOLI-4 input deck.
#[derive(Debug)]
pub enum TripoliError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse(String),
}

impl fmt::Display for TripoliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TripoliError::Io { path, source } => {
                write!(f, "reading TRIPOLI-4 input {}: {source}", path.display())
            }
            TripoliError::Parse(msg) => write!(f, "TRIPOLI-4 input error: {msg}"),
        }
    }
}

impl std::error::Error for TripoliError {}

/// Returns `true` if `path` has a TRIPOLI-4 file extension (`.t4`, `.tripoli`, `.t4g`).
pub fn is_tripoli_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            ext.eq_ignore_ascii_case("t4")
                || ext.eq_ignore_ascii_case("tripoli")
                || ext.eq_ignore_ascii_case("t4g")
        })
}

/// Returns `true` if raw input bytes appear to be a TRIPOLI-4 deck.
pub fn is_tripoli_bytes(input: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(input) else {
        return false;
    };
    let clean = strip_block_comments_fast(s);
    let mut has_geom = false;
    let mut has_t4_kw = false;
    for raw_line in clean.lines() {
        let code = raw_line.split("//").next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        let first = code
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_uppercase();
        if matches!(first.as_str(), "GEOMETRY" | "GEOMETRIE") {
            has_geom = true;
        }
        if matches!(
            first.as_str(),
            "SURF" | "VOLU" | "ENDG" | "END_GEOMETRY" | "GEOMCOMP"
        ) {
            has_t4_kw = true;
        }
        if has_geom && has_t4_kw {
            return true;
        }
    }
    false
}

/// Read an input file (TRIPOLI-4 or MCNP) and return MCNP-compatible bytes.
///
/// For non-TRIPOLI-4 inputs, returns raw file bytes (checking content sniffing
/// as a fallback for `.txt`/extensionless TRIPOLI-4 decks).
pub fn read_input_bytes(path: &Path) -> Result<Vec<u8>, TripoliError> {
    let bytes = fs::read(path).map_err(|source| TripoliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if is_tripoli_path(path) || is_tripoli_bytes(&bytes) {
        let text = String::from_utf8_lossy(&bytes);
        let mcnp = tripoli_to_mcnp(&text)?;
        return Ok(mcnp.into_bytes());
    }
    Ok(bytes)
}

/// If `input` contains a TRIPOLI-4 deck, translate it into canonical MCNP bytes;
/// otherwise borrow `input` untouched with zero allocation.
pub fn maybe_translate_bytes(input: &[u8]) -> Result<Cow<'_, [u8]>, TripoliError> {
    if !is_tripoli_bytes(input) {
        return Ok(Cow::Borrowed(input));
    }
    let text = String::from_utf8_lossy(input);
    let mcnp = tripoli_to_mcnp(&text)?;
    Ok(Cow::Owned(mcnp.into_bytes()))
}

// ---------------------------------------------------------------------------
// Nuclide Conversion
// ---------------------------------------------------------------------------

/// Map a chemical element symbol (e.g. `"Fe"`, `"ZR"`, `"C"`) to atomic number `Z`.
fn element_z(sym: &str) -> Option<u32> {
    const ELEMENTS: [&str; 118] = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
        "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ];
    ELEMENTS
        .iter()
        .position(|e| e.eq_ignore_ascii_case(sym))
        .map(|i| (i + 1) as u32)
}

/// Convert a TRIPOLI-4 nuclide or natural-element identifier (e.g. `"U235"`, `"FE-NAT"`,
/// `"FE_NAT"`, `"H1_H2O"`, `"C0"`, `"C_NAT"`, `"92235"`, `"92235.09c"`) to an MCNP ZAID.
pub fn tripoli_nuclide_to_zaid(name: &str) -> Option<String> {
    let core = name.trim().split('.').next()?.trim();
    if let Ok(num) = core.parse::<u32>() {
        if num >= 1000 {
            return Some(num.to_string());
        }
    }
    let clean = core.split(['_', '-']).next().unwrap_or(core).trim();
    // Handle deuterium / tritium aliases
    if clean.eq_ignore_ascii_case("D") {
        return Some("1002".to_string());
    }
    if clean.eq_ignore_ascii_case("T") {
        return Some("1003".to_string());
    }
    let split_idx = clean
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(clean.len());
    let sym = &clean[..split_idx];
    let z = element_z(sym)?;
    let a: u32 = if split_idx < clean.len() {
        let tail = &clean[split_idx..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse::<u32>().unwrap_or(0)
    } else {
        0
    };
    Some(format!("{}", z * 1000 + a))
}

// ---------------------------------------------------------------------------
// Tokenization & Comment Stripping
// ---------------------------------------------------------------------------

fn strip_block_comments_fast(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
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
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

#[derive(Debug, Clone)]
struct CommentedTokens {
    tokens: Vec<String>,
    /// Line-ending `//` comments mapped by the index of the last token on that line
    /// (for attaching labels to `VOLU`, `SURF`, and `TYPE` items).
    line_comments: HashMap<usize, String>,
}

fn tokenize_with_comments(src: &str) -> CommentedTokens {
    let clean = strip_block_comments_fast(src);
    let mut tokens = Vec::new();
    let mut line_comments = HashMap::new();

    for raw_line in clean.lines() {
        let (code, comment) = match raw_line.split_once("//") {
            Some((c, cm)) => (c, Some(cm.trim().to_string())),
            None => (raw_line, None),
        };
        let start = tokens.len();
        for tok in code.split_whitespace() {
            // Strip trailing statement terminators like `;` or `.` only if standalone? No: keep floats `10.` valid!
            let cleaned = tok.trim_end_matches(';');
            if !cleaned.is_empty() {
                tokens.push(cleaned.to_string());
            }
        }
        if let Some(cm) = comment {
            if !cm.is_empty() && tokens.len() > start {
                line_comments.insert(tokens.len() - 1, cm);
            }
        }
    }
    CommentedTokens {
        tokens,
        line_comments,
    }
}

// ---------------------------------------------------------------------------
// Geometry & Composition Domain Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct T4Transform {
    disp: [f64; 3],
    rot: [[f64; 3]; 3],
}

#[derive(Debug, Clone)]
struct T4Surface {
    id: i64,
    stype: String,
    params: Vec<f64>,
    transform_id: Option<i64>,
    comment: Option<String>,
}

#[derive(Debug, Clone)]
enum T4ShapeDef {
    Box([f64; 6]),                // xmin, xmax, ymin, ymax, zmin, zmax (centered offsets)
    Sphere([f64; 3], f64),        // offset xyz, radius
    Cylinder([f64; 3], f64, f64), // offset xyz, radius, height
    Hexag(f64, f64),              // flat-to-flat half-pitch or apothem radius r, height
    CombiSurfs(Vec<i64>),         // shorthand e.g. TYPE 1 COMBI 1 -1
}

#[derive(Debug, Clone)]
enum T4VolBody {
    Equa {
        pluses: Vec<i64>,
        minuses: Vec<i64>,
        ops: Option<(String, Vec<i64>)>, // INTE / UNION
        vpluses: Vec<i64>,
        vminuses: Vec<i64>,
    },
    Combi {
        type_id: i64,
        origin: [f64; 3],
    },
}

#[derive(Debug, Clone)]
struct T4Volume {
    body: T4VolBody,
    ecrase: Vec<i64>,
    fictive: bool,
    comment: Option<String>,
}

#[derive(Debug, Clone)]
struct T4Material {
    id: i64,
    name: String,
    mcnp_density: f64,
    nuclides: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// Parsing Helpers
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

fn parse_f64_tok(tok: &str) -> Option<f64> {
    tok.parse::<f64>().ok()
}

fn parse_i64_tok(tok: &str) -> Option<i64> {
    tok.parse::<i64>().ok()
}

fn read_floats(
    toks: &[String],
    idx: &mut usize,
    count: usize,
    ctx: &str,
) -> Result<Vec<f64>, TripoliError> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let Some(t) = toks.get(*idx) else {
            return Err(TripoliError::Parse(format!(
                "{ctx}: unexpected end of input"
            )));
        };
        let v = parse_f64_tok(t)
            .ok_or_else(|| TripoliError::Parse(format!("{ctx}: expected number, got `{t}`")))?;
        out.push(v);
        *idx += 1;
    }
    Ok(out)
}

fn read_counted_ints(
    toks: &[String],
    idx: &mut usize,
    ctx: &str,
) -> Result<Vec<i64>, TripoliError> {
    let Some(cnt_tok) = toks.get(*idx) else {
        return Err(TripoliError::Parse(format!("{ctx}: missing count")));
    };
    let n: usize = cnt_tok.parse().map_err(|_| {
        TripoliError::Parse(format!("{ctx}: expected integer count, got `{cnt_tok}`"))
    })?;
    *idx += 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let Some(t) = toks.get(*idx) else {
            return Err(TripoliError::Parse(format!(
                "{ctx}: unexpected end of list"
            )));
        };
        let id = parse_i64_tok(t)
            .ok_or_else(|| TripoliError::Parse(format!("{ctx}: expected integer id, got `{t}`")))?;
        out.push(id);
        *idx += 1;
    }
    Ok(out)
}

fn comment_between(
    comments: &HashMap<usize, String>,
    start: usize,
    end_excl: usize,
) -> Option<String> {
    for i in (start..end_excl).rev() {
        if let Some(c) = comments.get(&i) {
            return Some(c.clone());
        }
    }
    None
}

fn write_wrapped_mcnp_card(buf: &mut String, tokens: &[String], comment: Option<&str>) {
    let mut line_len = 0usize;
    for (idx, tok) in tokens.iter().enumerate() {
        let sep = if idx == 0 { 0 } else { 1 };
        if line_len > 0 && line_len + sep + tok.len() > 75 {
            buf.push('\n');
            buf.push_str("     ");
            line_len = 5;
        } else if idx > 0 {
            buf.push(' ');
            line_len += 1;
        }
        buf.push_str(tok);
        line_len += tok.len();
    }
    if let Some(c) = comment {
        let clean_c = c.trim().replace(['\n', '\r'], " ");
        if !clean_c.is_empty() {
            if line_len + 3 + clean_c.len() > 78 {
                buf.push('\n');
                buf.push_str("     $ ");
            } else {
                buf.push_str(" $ ");
            }
            buf.push_str(&clean_c);
        }
    }
    buf.push('\n');
}

// ---------------------------------------------------------------------------
// Surface Algebra & Affine Transformations
// ---------------------------------------------------------------------------

/// Build MCNP `gq` parameters `[A, B, C, D, E, F, G, H, I, J]` from
/// centered quadratic `(p - p0)^T M (p - p0) + c0 = 0`, where `M` is symmetric.
fn quadric_from_matrix(p0: [f64; 3], m: [[f64; 3]; 3], c0: f64) -> Vec<f64> {
    let a = m[0][0];
    let b = m[1][1];
    let c = m[2][2];
    let d = 2.0 * m[0][1];
    let e = 2.0 * m[1][2];
    let f = 2.0 * m[2][0];
    let mp0 = [
        m[0][0] * p0[0] + m[0][1] * p0[1] + m[0][2] * p0[2],
        m[1][0] * p0[0] + m[1][1] * p0[1] + m[1][2] * p0[2],
        m[2][0] * p0[0] + m[2][1] * p0[1] + m[2][2] * p0[2],
    ];
    let g = -2.0 * mp0[0];
    let h = -2.0 * mp0[1];
    let i = -2.0 * mp0[2];
    let j = p0[0] * mp0[0] + p0[1] * mp0[1] + p0[2] * mp0[2] + c0;
    vec![a, b, c, d, e, f, g, h, i, j]
}

/// Apply `p_local = R^T * (p_world - disp)` (or standard T4 transform matrix:
/// `p_world = disp + R * p_local`) to an MCNP `(mnemonic, params)` surface.
fn transform_mcnp_surface(mnemonic: &str, params: &[f64], tr: &T4Transform) -> (String, Vec<f64>) {
    // Convert surface first to either a plane (n·p - d = 0) or general quadric p^T Q p + b·p + c = 0.
    // p_local = M * p_world + offset, with M = R^T and offset = -R^T * disp (assuming orthonormal R).
    let rt = [
        [tr.rot[0][0], tr.rot[1][0], tr.rot[2][0]],
        [tr.rot[0][1], tr.rot[1][1], tr.rot[2][1]],
        [tr.rot[0][2], tr.rot[1][2], tr.rot[2][2]],
    ];
    let mul = |mat: &[[f64; 3]; 3], v: [f64; 3]| -> [f64; 3] {
        [
            mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2],
            mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2],
            mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2],
        ]
    };

    match mnemonic {
        "px" | "py" | "pz" | "p" => {
            let (n_loc, d_loc) = match mnemonic {
                "px" => ([1.0, 0.0, 0.0], params[0]),
                "py" => ([0.0, 1.0, 0.0], params[0]),
                "pz" => ([0.0, 0.0, 1.0], params[0]),
                _ => ([params[0], params[1], params[2]], params[3]),
            };
            // n_loc · (R^T (p - disp)) - d_loc = (R * n_loc) · p - (d_loc + (R * n_loc) · disp)
            let nw = mul(&tr.rot, n_loc);
            let dw = d_loc + nw[0] * tr.disp[0] + nw[1] * tr.disp[1] + nw[2] * tr.disp[2];
            ("p".to_string(), vec![nw[0], nw[1], nw[2], dw])
        }
        "s" | "so" => {
            let (c_loc, r) = if mnemonic == "so" {
                ([0.0, 0.0, 0.0], params[0])
            } else {
                ([params[0], params[1], params[2]], params[3])
            };
            let rc = mul(&tr.rot, c_loc);
            let cw = [rc[0] + tr.disp[0], rc[1] + tr.disp[1], rc[2] + tr.disp[2]];
            ("s".to_string(), vec![cw[0], cw[1], cw[2], r])
        }
        "gq" if params.len() >= 10 => {
            // f(q) = q^T Q q + b·q + c where q = R^T (p - disp) = M (p - disp)
            let qm = [
                [params[0], 0.5 * params[3], 0.5 * params[5]],
                [0.5 * params[3], params[1], 0.5 * params[4]],
                [0.5 * params[5], 0.5 * params[4], params[2]],
            ];
            let bw_loc = [params[6], params[7], params[8]];
            let c_loc = params[9];
            // New Q_w = R * Q * R^T
            let mut qr = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    qr[i][j] = (0..3).map(|k| qm[i][k] * rt[k][j]).sum();
                }
            }
            let mut qw = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    qw[i][j] = (0..3).map(|k| tr.rot[i][k] * qr[k][j]).sum();
                }
            }
            // Linear term b_loc · R^T (p - disp) = (R * b_loc) · (p - disp)
            let rb = mul(&tr.rot, bw_loc);
            // f(p) = (p - disp)^T qw (p - disp) + rb · (p - disp) + c_loc
            let mut gq = quadric_from_matrix(tr.disp, qw, c_loc);
            gq[6] += rb[0];
            gq[7] += rb[1];
            gq[8] += rb[2];
            gq[9] -= rb[0] * tr.disp[0] + rb[1] * tr.disp[1] + rb[2] * tr.disp[2];
            ("gq".to_string(), gq)
        }
        _ => (mnemonic.to_string(), params.to_vec()),
    }
}

fn convert_t4_surface_params(surf: &T4Surface) -> Result<(String, Vec<f64>), TripoliError> {
    let p = &surf.params;
    let need = |n: usize| -> Result<(), TripoliError> {
        if p.len() < n {
            Err(TripoliError::Parse(format!(
                "surface {} ({}) needs at least {n} coefficients, got {}",
                surf.id,
                surf.stype,
                p.len()
            )))
        } else {
            Ok(())
        }
    };
    let res = match surf.stype.as_str() {
        "PLANEX" | "PLANX" => {
            need(1)?;
            ("px".to_string(), vec![p[0]])
        }
        "PLANEY" | "PLANY" => {
            need(1)?;
            ("py".to_string(), vec![p[0]])
        }
        "PLANEZ" | "PLANZ" => {
            need(1)?;
            ("pz".to_string(), vec![p[0]])
        }
        "PLANE" => {
            need(4)?;
            // TRIPOLI-4 PLANE A B C D satisfies A*x + B*y + C*z + D = 0
            // MCNP P a b c d satisfies a*x + b*y + c*z - d = 0
            ("p".to_string(), vec![p[0], p[1], p[2], -p[3]])
        }
        "SPHERE" => {
            if p.len() == 1 {
                ("so".to_string(), vec![p[0]])
            } else {
                need(4)?;
                ("s".to_string(), vec![p[0], p[1], p[2], p[3]])
            }
        }
        "CYLX" => {
            if p.len() == 1 {
                ("cx".to_string(), vec![p[0]])
            } else {
                need(3)?;
                ("c/x".to_string(), vec![p[0], p[1], p[2]])
            }
        }
        "CYLY" => {
            if p.len() == 1 {
                ("cy".to_string(), vec![p[0]])
            } else {
                need(3)?;
                ("c/y".to_string(), vec![p[0], p[1], p[2]])
            }
        }
        "CYLZ" => {
            if p.len() == 1 {
                ("cz".to_string(), vec![p[0]])
            } else {
                need(3)?;
                ("c/z".to_string(), vec![p[0], p[1], p[2]])
            }
        }
        "CYL" => {
            need(7)?;
            // x0 y0 z0 r ux uy uz -> |d|^2 - (d·u)^2 - r^2 = 0
            let ulen = (p[4] * p[4] + p[5] * p[5] + p[6] * p[6]).sqrt();
            if ulen == 0.0 {
                return Err(TripoliError::Parse(format!(
                    "CYL surface {} has zero direction vector",
                    surf.id
                )));
            }
            let u = [p[4] / ulen, p[5] / ulen, p[6] / ulen];
            let m = [
                [1.0 - u[0] * u[0], -u[0] * u[1], -u[0] * u[2]],
                [-u[1] * u[0], 1.0 - u[1] * u[1], -u[1] * u[2]],
                [-u[2] * u[0], -u[2] * u[1], 1.0 - u[2] * u[2]],
            ];
            let gq = quadric_from_matrix([p[0], p[1], p[2]], m, -(p[3] * p[3]));
            ("gq".to_string(), gq)
        }
        "CONEX" => {
            need(4)?;
            let t = p[3].to_radians().tan();
            ("k/x".to_string(), vec![p[0], p[1], p[2], t * t])
        }
        "CONEY" => {
            need(4)?;
            let t = p[3].to_radians().tan();
            ("k/y".to_string(), vec![p[0], p[1], p[2], t * t])
        }
        "CONEZ" => {
            need(4)?;
            let t = p[3].to_radians().tan();
            ("k/z".to_string(), vec![p[0], p[1], p[2], t * t])
        }
        "CONE" => {
            need(7)?;
            let ulen = (p[4] * p[4] + p[5] * p[5] + p[6] * p[6]).sqrt();
            if ulen == 0.0 {
                return Err(TripoliError::Parse(format!(
                    "CONE surface {} has zero direction vector",
                    surf.id
                )));
            }
            let u = [p[4] / ulen, p[5] / ulen, p[6] / ulen];
            let t = p[3].to_radians().tan();
            let k = 1.0 + t * t;
            let m = [
                [1.0 - k * u[0] * u[0], -k * u[0] * u[1], -k * u[0] * u[2]],
                [-k * u[1] * u[0], 1.0 - k * u[1] * u[1], -k * u[1] * u[2]],
                [-k * u[2] * u[0], -k * u[2] * u[1], 1.0 - k * u[2] * u[2]],
            ];
            let gq = quadric_from_matrix([p[0], p[1], p[2]], m, 0.0);
            ("gq".to_string(), gq)
        }
        "QUAD" => {
            need(10)?;
            ("gq".to_string(), p[..10].to_vec())
        }
        "TORUSX" | "TORUSY" | "TORUSZ" => {
            let mnem = match surf.stype.as_str() {
                "TORUSX" => "tx",
                "TORUSY" => "ty",
                _ => "tz",
            }
            .to_string();
            if p.len() == 5 {
                (mnem, vec![p[0], p[1], p[2], p[3], p[4], p[4]])
            } else {
                need(6)?;
                (mnem, p[..6].to_vec())
            }
        }
        other => {
            return Err(TripoliError::Parse(format!(
                "unsupported TRIPOLI-4 surface type `{other}` on surface {}",
                surf.id
            )));
        }
    };
    Ok(res)
}

fn t4_surf_arity(stype: &str, toks: &[String], idx: usize) -> usize {
    match stype {
        "PLANEX" | "PLANX" | "PLANEY" | "PLANY" | "PLANEZ" | "PLANZ" => 1,
        "PLANE" => 4,
        "QUAD" => 10,
        "CYL" | "CONE" => 7,
        "CONEX" | "CONEY" | "CONEZ" => 4,
        "TORUSX" | "TORUSY" | "TORUSZ" => {
            if toks.get(idx + 5).and_then(|t| parse_f64_tok(t)).is_some() {
                6
            } else {
                5
            }
        }
        "SPHERE" => {
            if toks.get(idx + 1).and_then(|t| parse_f64_tok(t)).is_some() {
                4
            } else {
                1
            }
        }
        "CYLX" | "CYLY" | "CYLZ" => {
            if toks.get(idx + 1).and_then(|t| parse_f64_tok(t)).is_some() {
                3
            } else {
                1
            }
        }
        _ => {
            let mut k = 0;
            while toks.get(idx + k).and_then(|t| parse_f64_tok(t)).is_some() {
                k += 1;
            }
            k
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive Volume Boolean Expression Expansion
// ---------------------------------------------------------------------------

fn build_raw_volume_expr(
    vol_id: i64,
    volumes: &BTreeMap<i64, T4Volume>,
    combi_exprs: &HashMap<i64, String>,
    cache: &mut HashMap<i64, String>,
    visiting: &mut HashSet<i64>,
) -> Result<String, TripoliError> {
    if let Some(cached) = cache.get(&vol_id) {
        return Ok(cached.clone());
    }
    if !visiting.insert(vol_id) {
        return Err(TripoliError::Parse(format!(
            "cyclic volume reference involving VOLU {vol_id}"
        )));
    }
    let Some(vol) = volumes.get(&vol_id) else {
        return Err(TripoliError::Parse(format!(
            "referenced undefined TRIPOLI-4 volume {vol_id}"
        )));
    };

    let expr = match &vol.body {
        T4VolBody::Combi { .. } => combi_exprs.get(&vol_id).cloned().unwrap_or_default(),
        T4VolBody::Equa {
            pluses,
            minuses,
            ops,
            vpluses,
            vminuses,
        } => {
            let mut parts: Vec<String> = Vec::new();
            for &s in pluses {
                parts.push(s.to_string());
            }
            for &s in minuses {
                parts.push(format!("-{s}"));
            }
            if let Some((op_kind, sub_vols)) = ops {
                let mut sub_parts = Vec::with_capacity(sub_vols.len());
                for &sv in sub_vols {
                    sub_parts.push(build_raw_volume_expr(
                        sv,
                        volumes,
                        combi_exprs,
                        cache,
                        visiting,
                    )?);
                }
                if !sub_parts.is_empty() {
                    if op_kind == "UNION" {
                        let wrapped: Vec<String> =
                            sub_parts.into_iter().map(|p| format!("({p})")).collect();
                        parts.push(format!("({})", wrapped.join(" : ")));
                    } else {
                        for p in sub_parts {
                            parts.push(format!("({p})"));
                        }
                    }
                }
            }
            for &sv in vpluses {
                let e = build_raw_volume_expr(sv, volumes, combi_exprs, cache, visiting)?;
                parts.push(format!("({e})"));
            }
            for &sv in vminuses {
                let e = build_raw_volume_expr(sv, volumes, combi_exprs, cache, visiting)?;
                parts.push(format!("#({e})"));
            }
            parts.join(" ")
        }
    };

    visiting.remove(&vol_id);
    cache.insert(vol_id, expr.clone());
    Ok(expr)
}

fn complement_region_term(raw_expr: &str) -> Option<String> {
    let trimmed = raw_expr.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Single signed surface literal `-s` -> `s` and `s` -> `-s`.
    if !trimmed.contains(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ':' | '#')) {
        if let Ok(sid) = trimmed.parse::<i64>() {
            return Some((-sid).to_string());
        }
    }
    Some(format!("#({trimmed})"))
}

// ---------------------------------------------------------------------------
// Main Translator
// ---------------------------------------------------------------------------

/// Translate a TRIPOLI-4 input deck into canonical MCNP input text ready for `llevant`.
pub fn tripoli_to_mcnp(src: &str) -> Result<String, TripoliError> {
    let ct = tokenize_with_comments(src);
    let toks = &ct.tokens;
    let comments = &ct.line_comments;
    if toks.is_empty() {
        return Err(TripoliError::Parse("empty TRIPOLI-4 input".into()));
    }

    let mut transforms: BTreeMap<i64, T4Transform> = BTreeMap::new();
    let mut surfaces: BTreeMap<i64, T4Surface> = BTreeMap::new();
    let mut shapes: BTreeMap<i64, T4ShapeDef> = BTreeMap::new();
    let mut volumes: BTreeMap<i64, T4Volume> = BTreeMap::new();
    let mut boundary_surfs: Vec<i64> = Vec::new();
    let mut materials: Vec<T4Material> = Vec::new();
    let mut mat_name_to_id: HashMap<String, i64> = HashMap::new();
    let mut geomcomp: HashMap<i64, (i64, f64, Option<String>)> = HashMap::new();
    let mut in_geometry = false;
    let mut title = String::from("TRIPOLI-4 model loaded natively by llevant-tripoli");

    let mut i = 0usize;
    while i < toks.len() {
        let kw = toks[i].to_ascii_uppercase();
        match kw.as_str() {
            "LANG" => {
                i += 2;
            }
            "TITLE" | "TITRE" => {
                i += 1;
                let mut twords = Vec::new();
                while i < toks.len() {
                    let u = toks[i].to_ascii_uppercase();
                    if matches!(
                        u.as_str(),
                        "HASH_TABLE"
                            | "SURF"
                            | "TYPE"
                            | "VOLU"
                            | "TRANSFORM"
                            | "ENDG"
                            | "END_GEOMETRY"
                            | "GEOMETRY"
                            | "GEOMETRIE"
                    ) {
                        break;
                    }
                    twords.push(toks[i].clone());
                    i += 1;
                }
                if !twords.is_empty() {
                    title = twords.join(" ");
                }
            }
            "GEOMETRY" | "GEOMETRIE" => {
                in_geometry = true;
                i += 1;
            }
            "HASH_TABLE" => {
                i += 1;
            }
            "ENDG" | "END_GEOMETRY" | "FIN_GEOM" => {
                in_geometry = false;
                i += 1;
            }
            "TRANSFORM" | "TRMATRIX" => {
                i += 1;
                let tid = toks
                    .get(i)
                    .and_then(|t| parse_i64_tok(t))
                    .ok_or_else(|| TripoliError::Parse("TRANSFORM missing integer id".into()))?;
                i += 1;
                if toks
                    .get(i)
                    .is_some_and(|t| t.eq_ignore_ascii_case("MATRIX"))
                {
                    i += 1;
                }
                let vals = read_floats(toks, &mut i, 12, "TRANSFORM MATRIX")?;
                transforms.insert(
                    tid,
                    T4Transform {
                        disp: [vals[0], vals[1], vals[2]],
                        rot: [
                            [vals[3], vals[4], vals[5]],
                            [vals[6], vals[7], vals[8]],
                            [vals[9], vals[10], vals[11]],
                        ],
                    },
                );
            }
            "SURF" => {
                let start = i;
                i += 1;
                let sid = toks
                    .get(i)
                    .and_then(|t| parse_i64_tok(t))
                    .ok_or_else(|| TripoliError::Parse("SURF missing surface id".into()))?;
                i += 1;
                let mut tid = None;
                if toks
                    .get(i)
                    .is_some_and(|t| t.eq_ignore_ascii_case("TRANSFORM"))
                {
                    i += 1;
                    tid = toks.get(i).and_then(|t| parse_i64_tok(t));
                    i += 1;
                }
                let stype = toks
                    .get(i)
                    .ok_or_else(|| TripoliError::Parse(format!("SURF {sid} missing type")))?
                    .to_ascii_uppercase();
                i += 1;
                let arity = t4_surf_arity(&stype, toks, i);
                let params = read_floats(toks, &mut i, arity, &format!("SURF {sid}"))?;
                let comment = comment_between(comments, start, i);
                surfaces.insert(
                    sid,
                    T4Surface {
                        id: sid,
                        stype,
                        params,
                        transform_id: tid,
                        comment,
                    },
                );
            }
            "TYPE" => {
                i += 1;
                let id = toks
                    .get(i)
                    .and_then(|t| parse_i64_tok(t))
                    .ok_or_else(|| TripoliError::Parse("TYPE missing integer id".into()))?;
                i += 1;
                let shape_kw = toks
                    .get(i)
                    .ok_or_else(|| TripoliError::Parse(format!("TYPE {id} missing shape keyword")))?
                    .to_ascii_uppercase();
                i += 1;
                let def = match shape_kw.as_str() {
                    "BOX" | "BOITE" => {
                        let mut nums = Vec::new();
                        while nums.len() < 6 && toks.get(i).and_then(|t| parse_f64_tok(t)).is_some()
                        {
                            nums.push(parse_f64_tok(&toks[i]).unwrap());
                            i += 1;
                        }
                        if nums.len() >= 6 {
                            T4ShapeDef::Box([nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]])
                        } else if nums.len() >= 3 {
                            let (hx, hy, hz) = (0.5 * nums[0], 0.5 * nums[1], 0.5 * nums[2]);
                            T4ShapeDef::Box([-hx, hx, -hy, hy, -hz, hz])
                        } else {
                            return Err(TripoliError::Parse(format!(
                                "TYPE {id} BOX expects 3 dimensions or 6 bounds"
                            )));
                        }
                    }
                    "SPHERE" => {
                        let mut nums = Vec::new();
                        while nums.len() < 4 && toks.get(i).and_then(|t| parse_f64_tok(t)).is_some()
                        {
                            nums.push(parse_f64_tok(&toks[i]).unwrap());
                            i += 1;
                        }
                        if nums.len() >= 4 {
                            T4ShapeDef::Sphere([nums[0], nums[1], nums[2]], nums[3])
                        } else if !nums.is_empty() {
                            T4ShapeDef::Sphere([0.0, 0.0, 0.0], nums[0])
                        } else {
                            return Err(TripoliError::Parse(format!(
                                "TYPE {id} SPHERE expects radius"
                            )));
                        }
                    }
                    "CYLINDRE" | "CYLINDER" => {
                        let mut nums = Vec::new();
                        while nums.len() < 5 && toks.get(i).and_then(|t| parse_f64_tok(t)).is_some()
                        {
                            nums.push(parse_f64_tok(&toks[i]).unwrap());
                            i += 1;
                        }
                        if nums.len() >= 5 {
                            T4ShapeDef::Cylinder([nums[0], nums[1], nums[2]], nums[3], nums[4])
                        } else if nums.len() >= 2 {
                            T4ShapeDef::Cylinder([0.0, 0.0, 0.0], nums[0], nums[1])
                        } else {
                            return Err(TripoliError::Parse(format!(
                                "TYPE {id} CYLINDRE expects radius and height"
                            )));
                        }
                    }
                    "HEXAG" | "HEX" => {
                        let vals = read_floats(toks, &mut i, 2, &format!("TYPE {id} HEXAG"))?;
                        T4ShapeDef::Hexag(vals[0], vals[1])
                    }
                    "COMBI" => {
                        // Shorthand like `TYPE 1 COMBI 1 -1` where first int may be surface count.
                        let mut ints = Vec::new();
                        while let Some(t) = toks.get(i) {
                            let u = t.to_ascii_uppercase();
                            if matches!(
                                u.as_str(),
                                "ENDG"
                                    | "END_GEOMETRY"
                                    | "SURF"
                                    | "TYPE"
                                    | "VOLU"
                                    | "END"
                                    | "FINV"
                                    | "ENDV"
                            ) {
                                break;
                            }
                            if let Some(v) = parse_i64_tok(t) {
                                ints.push(v);
                                i += 1;
                            } else {
                                break;
                            }
                        }
                        let surfs = if ints.len() >= 2 && (ints[0] as usize) + 1 == ints.len() {
                            ints[1..].to_vec()
                        } else {
                            ints
                        };
                        T4ShapeDef::CombiSurfs(surfs)
                    }
                    other => {
                        return Err(TripoliError::Parse(format!(
                            "unsupported TYPE shape `{other}` on TYPE {id}"
                        )));
                    }
                };
                shapes.insert(id, def);
            }
            "VOLU" => {
                let start = i;
                i += 1;
                let vid = toks
                    .get(i)
                    .and_then(|t| parse_i64_tok(t))
                    .ok_or_else(|| TripoliError::Parse("VOLU missing integer id".into()))?;
                i += 1;
                let mode = toks
                    .get(i)
                    .ok_or_else(|| TripoliError::Parse(format!("VOLU {vid} missing body")))?
                    .to_ascii_uppercase();

                let mut ecrase = Vec::new();
                let mut fictive = false;
                let mut inline_name = None;

                let body = if mode == "EQUA" {
                    i += 1;
                    let mut pluses = Vec::new();
                    let mut minuses = Vec::new();
                    let mut ops = None;
                    let mut vpluses = Vec::new();
                    let mut vminuses = Vec::new();

                    while i < toks.len() {
                        let sub = toks[i].to_ascii_uppercase();
                        if matches!(sub.as_str(), "ENDV" | "FINV" | "FIN_VOLU") {
                            i += 1;
                            break;
                        }
                        if matches!(
                            sub.as_str(),
                            "ENDG" | "END_GEOMETRY" | "VOLU" | "SURF" | "TYPE"
                        ) {
                            break;
                        }
                        match sub.as_str() {
                            "PLUS" => {
                                i += 1;
                                pluses.extend(read_counted_ints(toks, &mut i, "VOLU PLUS")?);
                            }
                            "MINUS" | "MOINS" => {
                                i += 1;
                                minuses.extend(read_counted_ints(toks, &mut i, "VOLU MINUS")?);
                            }
                            "INTE" | "INTERSECTION" => {
                                i += 1;
                                let list = read_counted_ints(toks, &mut i, "VOLU INTE")?;
                                ops = Some(("INTE".to_string(), list));
                            }
                            "UNION" => {
                                i += 1;
                                let list = read_counted_ints(toks, &mut i, "VOLU UNION")?;
                                ops = Some(("UNION".to_string(), list));
                            }
                            "VPLUS" => {
                                i += 1;
                                vpluses.extend(read_counted_ints(toks, &mut i, "VOLU VPLUS")?);
                            }
                            "VMOINS" | "VMINUS" => {
                                i += 1;
                                vminuses.extend(read_counted_ints(toks, &mut i, "VOLU VMOINS")?);
                            }
                            "ECRASE" => {
                                i += 1;
                                ecrase.extend(read_counted_ints(toks, &mut i, "VOLU ECRASE")?);
                            }
                            "FICTIVE" | "FICTIF" => {
                                i += 1;
                                fictive = true;
                            }
                            "NAME" => {
                                i += 1;
                                if let Some(n) = toks.get(i) {
                                    inline_name = Some(n.clone());
                                    i += 1;
                                }
                            }
                            other => {
                                return Err(TripoliError::Parse(format!(
                                    "unexpected keyword `{other}` in VOLU {vid} EQUA"
                                )));
                            }
                        }
                    }

                    T4VolBody::Equa {
                        pluses,
                        minuses,
                        ops,
                        vpluses,
                        vminuses,
                    }
                } else if mode == "COMBI" {
                    i += 1;
                    let tid = toks.get(i).and_then(|t| parse_i64_tok(t)).ok_or_else(|| {
                        TripoliError::Parse(format!("VOLU {vid} COMBI missing type id"))
                    })?;
                    i += 1;
                    let mut coords = Vec::new();
                    while coords.len() < 3 && toks.get(i).and_then(|t| parse_f64_tok(t)).is_some() {
                        coords.push(parse_f64_tok(&toks[i]).unwrap());
                        i += 1;
                    }
                    let origin = if coords.len() == 3 {
                        [coords[0], coords[1], coords[2]]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    while i < toks.len() {
                        let sub = toks[i].to_ascii_uppercase();
                        if matches!(sub.as_str(), "ENDV" | "FINV" | "FIN_VOLU") {
                            i += 1;
                            break;
                        }
                        if matches!(
                            sub.as_str(),
                            "ENDG" | "END_GEOMETRY" | "VOLU" | "SURF" | "TYPE"
                        ) {
                            break;
                        }
                        match sub.as_str() {
                            "ECRASE" => {
                                i += 1;
                                ecrase.extend(read_counted_ints(toks, &mut i, "VOLU ECRASE")?);
                            }
                            "FICTIVE" | "FICTIF" => {
                                i += 1;
                                fictive = true;
                            }
                            "NAME" => {
                                i += 1;
                                if let Some(n) = toks.get(i) {
                                    inline_name = Some(n.clone());
                                    i += 1;
                                }
                            }
                            _ => {
                                i += 1;
                            }
                        }
                    }
                    T4VolBody::Combi {
                        type_id: tid,
                        origin,
                    }
                } else {
                    return Err(TripoliError::Parse(format!(
                        "unsupported VOLU {vid} mode `{mode}` (expected EQUA or COMBI)"
                    )));
                };

                let comment = inline_name.or_else(|| comment_between(comments, start, i));
                volumes.insert(
                    vid,
                    T4Volume {
                        body,
                        ecrase,
                        fictive,
                        comment,
                    },
                );
            }
            "BOUNDARY_CONDITION" | "LIMITE" => {
                i += 1;
                if toks.get(i).and_then(|t| parse_i64_tok(t)).is_some()
                    && toks.get(i + 1).is_some_and(|n| {
                        matches!(
                            n.to_ascii_uppercase().as_str(),
                            "ALL_COMPLETE"
                                | "REFLECTION"
                                | "REFLEXION"
                                | "COSINUS"
                                | "FUITE"
                                | "VACUUM"
                                | "VOID"
                                | "ALBEDO"
                        )
                    })
                {
                    i += 1;
                }
                while i < toks.len() {
                    let sub = toks[i].to_ascii_uppercase();
                    if matches!(
                        sub.as_str(),
                        "END_BOUNDARY_CONDITION" | "END_LIMITE" | "END"
                    ) {
                        i += 1;
                        break;
                    }
                    if matches!(sub.as_str(), "COMPOSITION" | "GEOMCOMP" | "GEOMETRY") {
                        break;
                    }
                    if let Some(sid) = parse_i64_tok(&toks[i]) {
                        boundary_surfs.push(sid);
                    }
                    i += 1;
                }
            }
            "COMPOSITION" => {
                i += 1;
                let mut next_mid = 1i64;
                // Skip optional material count line
                if toks.get(i).and_then(|t| parse_i64_tok(t)).is_some() {
                    i += 1;
                }
                while i < toks.len() {
                    let sub = toks[i].to_ascii_uppercase();
                    if matches!(sub.as_str(), "END_COMPOSITION" | "END_COMPO") {
                        i += 1;
                        break;
                    }
                    if matches!(sub.as_str(), "GEOMCOMP" | "GEOMETRY" | "END") {
                        break;
                    }
                    if matches!(
                        sub.as_str(),
                        "DENSITY" | "DENSITE" | "POINT_WISE" | "PONCTUELLE"
                    ) {
                        let kind = sub;
                        i += 1;
                        // Expect temperature (numeric) + material name
                        if toks.get(i).and_then(|t| parse_f64_tok(t)).is_some() {
                            i += 1;
                        }
                        let Some(mname) = toks.get(i).cloned() else {
                            break;
                        };
                        i += 1;
                        let mut mcnp_density = -1.0_f64;
                        let mut is_mass_frac = false;
                        if kind == "DENSITY" || kind == "DENSITE" {
                            if let Some(rho) = toks.get(i).and_then(|t| parse_f64_tok(t)) {
                                mcnp_density = -rho.abs();
                                i += 1;
                            }
                            if toks.get(i).is_some_and(|t| {
                                matches!(
                                    t.to_ascii_uppercase().as_str(),
                                    "NB_ATOM"
                                        | "NB_ATOMES"
                                        | "POURCENTAGE_MASSIQUE"
                                        | "MASS_FRACTION"
                                )
                            }) {
                                let opt = toks[i].to_ascii_uppercase();
                                is_mass_frac = matches!(
                                    opt.as_str(),
                                    "POURCENTAGE_MASSIQUE" | "MASS_FRACTION"
                                );
                                i += 1;
                            }
                        }
                        let n_iso = toks
                            .get(i)
                            .and_then(|t| t.parse::<usize>().ok())
                            .unwrap_or(0);
                        if toks.get(i).and_then(|t| t.parse::<usize>().ok()).is_some() {
                            i += 1;
                        }
                        let mut nuclides = Vec::new();
                        let mut pw_sum = 0.0_f64;
                        for _ in 0..n_iso {
                            if i + 1 >= toks.len() {
                                break;
                            }
                            let iname = &toks[i];
                            let frac = parse_f64_tok(&toks[i + 1]).unwrap_or(0.0);
                            i += 2;
                            if let Some(zaid) = tripoli_nuclide_to_zaid(iname) {
                                let signed = if is_mass_frac {
                                    -frac.abs()
                                } else {
                                    frac.abs()
                                };
                                pw_sum += frac.abs();
                                nuclides.push((zaid, signed));
                            }
                        }
                        if kind == "POINT_WISE" || kind == "PONCTUELLE" {
                            mcnp_density = if pw_sum > 0.0 { pw_sum } else { 1e-24 };
                        }
                        let mid = if mname.eq_ignore_ascii_case("m0")
                            || mname.eq_ignore_ascii_case("VIDE")
                            || mname.eq_ignore_ascii_case("VOID")
                        {
                            0
                        } else {
                            let id = next_mid;
                            next_mid += 1;
                            id
                        };
                        mat_name_to_id.insert(mname.to_ascii_lowercase(), mid);
                        materials.push(T4Material {
                            id: mid,
                            name: mname,
                            mcnp_density,
                            nuclides,
                        });
                    } else {
                        i += 1;
                    }
                }
            }
            "GEOMCOMP" => {
                i += 1;
                let mut fallback_mid = (materials.len() as i64) + 1;
                while i < toks.len() {
                    let sub = toks[i].to_ascii_uppercase();
                    if matches!(sub.as_str(), "END_GEOMCOMP" | "END") {
                        i += 1;
                        break;
                    }
                    let mname = &toks[i];
                    i += 1;
                    let cnt = toks
                        .get(i)
                        .and_then(|t| t.parse::<usize>().ok())
                        .ok_or_else(|| {
                            TripoliError::Parse(format!("GEOMCOMP `{mname}` missing volume count"))
                        })?;
                    i += 1;
                    let key = mname.to_ascii_lowercase();
                    let mid = if matches!(sub.as_str(), "VIDE" | "VOID" | "M0") {
                        0
                    } else if let Some(&id) = mat_name_to_id.get(&key) {
                        id
                    } else {
                        let id = fallback_mid;
                        fallback_mid += 1;
                        mat_name_to_id.insert(key.clone(), id);
                        id
                    };
                    let rho = materials
                        .iter()
                        .find(|m| m.id == mid)
                        .map_or(-1.0, |m| m.mcnp_density);
                    for _ in 0..cnt {
                        let vid = toks.get(i).and_then(|t| parse_i64_tok(t)).ok_or_else(|| {
                            TripoliError::Parse(format!("GEOMCOMP `{mname}` missing volume id"))
                        })?;
                        i += 1;
                        geomcomp.insert(vid, (mid, rho, Some(mname.clone())));
                    }
                }
            }
            _ => {
                if !in_geometry && kw == "END" {
                    i += 1;
                    continue;
                }
                i += 1;
            }
        }
    }

    // Handle minimal shorthand files (e.g. `TYPE 1 COMBI 1 -1` without explicit `VOLU`).
    if volumes.is_empty() && !shapes.is_empty() {
        for (&sid, sdef) in &shapes {
            if let T4ShapeDef::CombiSurfs(surfs) = sdef {
                let mut pluses = Vec::new();
                let mut minuses = Vec::new();
                for &s in surfs {
                    if s < 0 {
                        minuses.push(-s);
                    } else if s > 0 {
                        pluses.push(s);
                    }
                }
                volumes.insert(
                    sid,
                    T4Volume {
                        body: T4VolBody::Equa {
                            pluses,
                            minuses,
                            ops: None,
                            vpluses: Vec::new(),
                            vminuses: Vec::new(),
                        },
                        ecrase: Vec::new(),
                        fictive: false,
                        comment: Some(format!("TRIPOLI-4 TYPE {sid} COMBI")),
                    },
                );
            }
        }
    }

    if volumes.is_empty() {
        return Err(TripoliError::Parse(
            "no VOLU or TYPE COMBI geometry cells found in TRIPOLI-4 input".into(),
        ));
    }

    let mut next_surf_id = surfaces.keys().copied().max().unwrap_or(0).max(0) + 100;
    let mut extra_surfaces: Vec<(i64, String, Vec<f64>)> = Vec::new();
    let mut combi_exprs: HashMap<i64, String> = HashMap::new();

    for (&vid, vol) in &volumes {
        if let T4VolBody::Combi { type_id, origin } = &vol.body {
            let Some(shape) = shapes.get(type_id) else {
                return Err(TripoliError::Parse(format!(
                    "VOLU {vid} references unknown TYPE {type_id}"
                )));
            };
            let expr = match shape {
                T4ShapeDef::Box(b) => {
                    let s = next_surf_id;
                    next_surf_id += 1;
                    extra_surfaces.push((
                        s,
                        "rpp".to_string(),
                        vec![
                            origin[0] + b[0],
                            origin[0] + b[1],
                            origin[1] + b[2],
                            origin[1] + b[3],
                            origin[2] + b[4],
                            origin[2] + b[5],
                        ],
                    ));
                    format!("-{s}")
                }
                T4ShapeDef::Sphere(off, r) => {
                    let s = next_surf_id;
                    next_surf_id += 1;
                    extra_surfaces.push((
                        s,
                        "s".to_string(),
                        vec![
                            origin[0] + off[0],
                            origin[1] + off[1],
                            origin[2] + off[2],
                            *r,
                        ],
                    ));
                    format!("-{s}")
                }
                T4ShapeDef::Cylinder(off, r, h) => {
                    let s = next_surf_id;
                    next_surf_id += 1;
                    extra_surfaces.push((
                        s,
                        "rcc".to_string(),
                        vec![
                            origin[0] + off[0],
                            origin[1] + off[1],
                            origin[2] + off[2] - 0.5 * h,
                            0.0,
                            0.0,
                            *h,
                            *r,
                        ],
                    ));
                    format!("-{s}")
                }
                T4ShapeDef::Hexag(r, h) => {
                    let s = next_surf_id;
                    next_surf_id += 1;
                    extra_surfaces.push((
                        s,
                        "rhp".to_string(),
                        vec![
                            origin[0],
                            origin[1],
                            origin[2] - 0.5 * h,
                            0.0,
                            0.0,
                            *h,
                            *r,
                            0.0,
                            0.0,
                        ],
                    ));
                    format!("-{s}")
                }
                T4ShapeDef::CombiSurfs(surfs) => surfs
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            combi_exprs.insert(vid, expr);
        }
    }

    // Build raw region expressions for every volume (expanding INTE/UNION/VPLUS/VMOINS)
    let mut raw_exprs = HashMap::new();
    let mut visiting = HashSet::new();
    for &vid in volumes.keys() {
        build_raw_volume_expr(vid, &volumes, &combi_exprs, &mut raw_exprs, &mut visiting)?;
    }

    // Collect ECRASE priority subtractions: if `crusher` has `ECRASE target`,
    // then `target`'s region excludes `raw_expr(crusher)`.
    let mut crushed_by: HashMap<i64, Vec<String>> = HashMap::new();
    for (&vid, vol) in &volumes {
        if let Some(crusher_expr) = raw_exprs.get(&vid) {
            if !crusher_expr.trim().is_empty() {
                for &target in &vol.ecrase {
                    crushed_by
                        .entry(target)
                        .or_default()
                        .push(crusher_expr.clone());
                }
            }
        }
    }

    let _ = &boundary_surfs;

    // -----------------------------------------------------------------------
    // Emit MCNP Input
    // -----------------------------------------------------------------------
    let mut mcnp = String::with_capacity(4096);
    let _ = writeln!(mcnp, "{title}");

    // Block 1: Cells
    let mut max_cell_id = 0i64;
    for (&vid, vol) in &volumes {
        if vol.fictive {
            continue;
        }
        max_cell_id = max_cell_id.max(vid);
        let mut expr = raw_exprs.get(&vid).cloned().unwrap_or_default();
        if expr.trim().is_empty() {
            continue;
        }
        if let Some(crushers) = crushed_by.get(&vid) {
            for ce in crushers {
                expr.push_str(" #(");
                expr.push_str(ce);
                expr.push(')');
            }
        }

        let mut toks = vec![vid.to_string()];
        match geomcomp.get(&vid) {
            Some((mid, _, _)) if *mid == 0 => {
                toks.push("0".to_string());
            }
            Some((mid, rho, _)) => {
                toks.push(mid.to_string());
                toks.push(fmt_num(*rho));
            }
            None => {
                toks.push("0".to_string());
            }
        }
        for tok in expr.split_whitespace() {
            toks.push(tok.to_string());
        }
        toks.push("imp:n=1".to_string());

        let label_comment = match (&vol.comment, geomcomp.get(&vid).and_then(|g| g.2.as_ref())) {
            (Some(vc), _) => Some(vc.clone()),
            (None, Some(mc)) => Some(mc.clone()),
            (None, None) => None,
        };
        write_wrapped_mcnp_card(&mut mcnp, &toks, label_comment.as_deref());
    }

    // Exterior graveyard IMP:N=0: exact complement of all emitted non-fictive volumes
    // so internal ECRASE surfaces can never create false exterior/interior overlaps.
    let mut graveyard_terms = Vec::new();
    let mut seen_terms = HashSet::new();
    for (&vid, vol) in &volumes {
        if vol.fictive {
            continue;
        }
        if let Some(expr) = raw_exprs.get(&vid) {
            if let Some(term) = complement_region_term(expr) {
                if seen_terms.insert(term.clone()) {
                    graveyard_terms.push(term);
                }
            }
        }
    }
    if !graveyard_terms.is_empty() {
        let grave_id = (max_cell_id + 100).max(99999);
        let mut gtoks = vec![grave_id.to_string(), "0".to_string()];
        for term in graveyard_terms {
            for tok in term.split_whitespace() {
                gtoks.push(tok.to_string());
            }
        }
        gtoks.push("imp:n=0".to_string());
        write_wrapped_mcnp_card(&mut mcnp, &gtoks, Some("TRIPOLI-4 exterior graveyard"));
    }

    mcnp.push('\n');

    // Block 2: Surfaces
    for surf in surfaces.values() {
        let (mut mnem, mut params) = convert_t4_surface_params(surf)?;
        if let Some(tid) = surf.transform_id {
            if let Some(tr) = transforms.get(&tid) {
                (mnem, params) = transform_mcnp_surface(&mnem, &params, tr);
            }
        }
        let mut stoks = vec![surf.id.to_string(), mnem];
        for c in params {
            stoks.push(fmt_num(c));
        }
        write_wrapped_mcnp_card(&mut mcnp, &stoks, surf.comment.as_deref());
    }
    for (sid, mnem, params) in &extra_surfaces {
        let mut stoks = vec![sid.to_string(), mnem.clone()];
        for c in params {
            stoks.push(fmt_num(*c));
        }
        write_wrapped_mcnp_card(&mut mcnp, &stoks, None);
    }

    mcnp.push('\n');

    // Block 3: Data (Materials)
    mcnp.push_str("mode n\n");
    for mat in &materials {
        if mat.id <= 0 || mat.nuclides.is_empty() {
            continue;
        }
        let mut mtoks = vec![format!("m{}", mat.id)];
        for (zaid, frac) in &mat.nuclides {
            mtoks.push(zaid.clone());
            mtoks.push(fmt_num(*frac));
        }
        write_wrapped_mcnp_card(&mut mcnp, &mtoks, Some(&mat.name));
    }

    Ok(mcnp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_tripoli_nuclides() {
        assert_eq!(tripoli_nuclide_to_zaid("U235").as_deref(), Some("92235"));
        assert_eq!(tripoli_nuclide_to_zaid("H1_H2O").as_deref(), Some("1001"));
        assert_eq!(tripoli_nuclide_to_zaid("FE-NAT").as_deref(), Some("26000"));
        assert_eq!(tripoli_nuclide_to_zaid("C_NAT").as_deref(), Some("6000"));
        assert_eq!(tripoli_nuclide_to_zaid("D").as_deref(), Some("1002"));
        assert_eq!(
            tripoli_nuclide_to_zaid("92238.09c").as_deref(),
            Some("92238")
        );
    }

    #[test]
    fn translates_simple_combi_shorthand() {
        let t4 = "GEOMETRY\n  SURF 1 SPHERE 0 0 0 10\n  TYPE 1 COMBI 1 -1\nENDG\nEND\n";
        assert!(is_tripoli_bytes(t4.as_bytes()));
        let mcnp = tripoli_to_mcnp(t4).expect("translate simple.tripoli");
        assert!(mcnp.contains("1 0 -1 imp:n=1"));
        assert!(mcnp.contains("1 s 0 0 0 10"));
        assert!(mcnp.contains("imp:n=0"));
    }

    #[test]
    fn translates_equa_ecrase_compositions_and_geomcomp() {
        let t4 = include_str!("../../../docs/examples/tripoli4/pwr_pin_cell.t4");
        let mcnp = tripoli_to_mcnp(t4).expect("translate pwr_pin_cell.t4");
        assert!(mcnp.contains("1 3 -0.74"));
        assert!(mcnp.contains("2 2 -6.55"));
        assert!(mcnp.contains("3 1 -10.4"));
        assert!(mcnp.contains("m1 92235 0.04 92238 0.96"));
        assert!(mcnp.contains("Water Moderator"));
        assert!(mcnp.contains("UO2 Fuel Pellet"));
    }

    #[test]
    fn translates_surfaces_combi_and_transformed_planes() {
        let t4 = r#"
GEOMETRY
TRANSFORM 1 MATRIX 10.0 0.0 0.0  1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0
SURF 1 TRANSFORM 1 PLANEX 0.0
SURF 2 CONEZ 0.0 0.0 0.0 45.0
SURF 3 PLANE 0.0 0.0 1.0 -5.0
SURF 4 QUAD 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 -100.0
SURF 5 CYL 0.0 0.0 0.0 2.0 0.0 1.0 1.0
SURF 6 SPHERE 0.0 0.0 0.0 3.0
VOLU 1 EQUA PLUS 1 1 MINUS 1 4 FICTIVE FINV
VOLU 2 EQUA MINUS 1 6 ECRASE 1 3 FINV
VOLU 3 EQUA INTE 1 1 ENDV
ENDG
"#;
        let mcnp = tripoli_to_mcnp(t4).expect("translate complex T4");
        // Transformed PLANEX 0 by disp (10,0,0) -> x = 10 -> p 1 0 0 10
        assert!(mcnp.contains("1 p 1 0 0 10"), "{mcnp}");
        // CONEZ 45 deg -> tan(45)^2 = 1 -> k/z 0 0 0 1
        assert!(mcnp.contains("2 k/z 0 0 0 1"), "{mcnp}");
        // PLANE 0 0 1 -5 -> z - 5 = 0 -> p 0 0 1 5
        assert!(mcnp.contains("3 p 0 0 1 5"), "{mcnp}");
        assert!(mcnp.contains("4 gq 1 1 1 0 0 0 0 0 0 -100"), "{mcnp}");
        assert!(mcnp.contains("5 gq 1 0.5 0.5 0 -1 0 0 0 0 -4"), "{mcnp}");
        // VOLU 1 is FICTIVE so excluded as physical cell, while VOLU 3 expanded it via INTE & subtracted VOLU 2 via ECRASE
        assert!(!mcnp.contains("\n1 0 "), "{mcnp}");
        assert!(mcnp.contains("3 0 (1 -4) #(-6) imp:n=1"), "{mcnp}");
    }
}
