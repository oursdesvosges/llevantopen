//! Pure-Rust OpenMC XML (`geometry.xml`, `materials.xml`, `model.xml`) loader for `llevant`.
//!
//! Translates OpenMC CSG geometry, nested universes, rectilinear lattices, affine
//! transforms, and isotopic/elemental material definitions into canonical MCNP
//! syntax understood natively by `migjorn`, `llevant`, and `llevant-scene`.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};

/// Errors produced when loading or translating an OpenMC XML model.
#[derive(Debug)]
pub enum OpenMcError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse(String),
}

impl fmt::Display for OpenMcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenMcError::Io { path, source } => {
                write!(f, "reading OpenMC input {}: {source}", path.display())
            }
            OpenMcError::Parse(msg) => write!(f, "OpenMC XML error: {msg}"),
        }
    }
}

impl std::error::Error for OpenMcError {}

/// Returns true if `path` looks like an OpenMC XML file or an OpenMC model directory
/// containing `model.xml` or `geometry.xml`.
pub fn is_openmc_path(path: &Path) -> bool {
    if path.is_dir() {
        return path.join("model.xml").is_file() || path.join("geometry.xml").is_file();
    }
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("xml"))
}

/// Returns true if raw bytes look like an OpenMC XML document (`<model>` or `<geometry>`).
pub fn is_openmc_bytes(input: &[u8]) -> bool {
    let s = std::str::from_utf8(input).unwrap_or("");
    let trimmed = s.trim_start_matches('\u{feff}').trim_start();
    if !trimmed.starts_with('<') {
        return false;
    }
    let prefix = &trimmed[..trimmed.len().min(4096)];
    prefix.contains("<geometry") || prefix.contains("<model")
}

/// Read an input file (either MCNP or OpenMC XML / directory) and return bytes ready
/// for `llevant` (`migjorn`).
///
/// For non-OpenMC files, this performs a direct `fs::read`.
/// For OpenMC `.xml` files (or directories containing `model.xml` / `geometry.xml`),
/// it also discovers a sibling `materials.xml` when present and translates the model
/// into canonical MCNP bytes.
pub fn read_input_bytes(path: &Path) -> Result<Vec<u8>, OpenMcError> {
    if !is_openmc_path(path) {
        let bytes = fs::read(path).map_err(|source| OpenMcError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if is_openmc_bytes(&bytes) {
            let text = String::from_utf8_lossy(&bytes);
            let mcnp = openmc_xml_to_mcnp(&text, None)?;
            return Ok(mcnp.into_bytes());
        }
        return Ok(bytes);
    }

    let (geom_path, mats_path) = if path.is_dir() {
        let model_xml = path.join("model.xml");
        if model_xml.is_file() {
            (model_xml, None)
        } else {
            let g = path.join("geometry.xml");
            let m = path.join("materials.xml");
            (g, m.is_file().then_some(m))
        }
    } else {
        let sibling_mats = path.parent().and_then(|dir| {
            let m = dir.join("materials.xml");
            (m.is_file() && m != path).then_some(m)
        });
        (path.to_path_buf(), sibling_mats)
    };

    let geom_text = fs::read_to_string(&geom_path).map_err(|source| OpenMcError::Io {
        path: geom_path.clone(),
        source,
    })?;
    let mats_text = match mats_path {
        Some(mp) => {
            Some(fs::read_to_string(&mp).map_err(|source| OpenMcError::Io { path: mp, source })?)
        }
        None => None,
    };

    let mcnp = openmc_xml_to_mcnp(&geom_text, mats_text.as_deref())?;
    Ok(mcnp.into_bytes())
}

/// If `input` contains OpenMC XML bytes, translate them to MCNP bytes; otherwise
/// borrow `input` untouched with zero allocation.
pub fn maybe_translate_bytes(input: &[u8]) -> Result<Cow<'_, [u8]>, OpenMcError> {
    if !is_openmc_bytes(input) {
        return Ok(Cow::Borrowed(input));
    }
    let text = String::from_utf8_lossy(input);
    let mcnp = openmc_xml_to_mcnp(&text, None)?;
    Ok(Cow::Owned(mcnp.into_bytes()))
}

// ---------------------------------------------------------------------------
// Minimal Zero-Dependency XML DOM Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct XmlNode {
    tag: String,
    attrs: HashMap<String, String>,
    text: String,
    children: Vec<XmlNode>,
}

impl XmlNode {
    /// Look up an XML attribute value by name on this element.
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.get(key).map(|s| s.as_str())
    }

    /// Return the trimmed text body of the first direct child element with tag `tag`.
    fn child_text(&self, tag: &str) -> Option<&str> {
        self.children
            .iter()
            .find(|c| c.tag == tag)
            .map(|c| c.text.trim())
    }
}

/// Decode standard XML character entities (`&lt;`, `&gt;`, `&quot;`, `&apos;`, `&amp;`).
fn unescape_xml(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Parse an OpenMC XML document into a tree of root [`XmlNode`] elements.
fn parse_xml_nodes(src: &str) -> Result<Vec<XmlNode>, OpenMcError> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut stack: Vec<XmlNode> = Vec::new();
    let mut roots: Vec<XmlNode> = Vec::new();

    while pos < len {
        if let Some(rel_lt) = src[pos..].find('<') {
            let text_chunk = &src[pos..pos + rel_lt];
            if !text_chunk.is_empty() {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&unescape_xml(text_chunk));
                }
            }
            pos += rel_lt;
        } else {
            let rest = &src[pos..];
            if let Some(top) = stack.last_mut() {
                top.text.push_str(&unescape_xml(rest));
            }
            break;
        }

        // Now src[pos..] starts with '<'
        if src[pos..].starts_with("<!--") {
            if let Some(end) = src[pos + 4..].find("-->") {
                pos += 4 + end + 3;
                continue;
            } else {
                return Err(OpenMcError::Parse("unterminated XML comment".into()));
            }
        }
        if src[pos..].starts_with("<?") {
            if let Some(end) = src[pos + 2..].find("?>") {
                pos += 2 + end + 2;
                continue;
            } else {
                return Err(OpenMcError::Parse(
                    "unterminated XML processing instruction".into(),
                ));
            }
        }
        if src[pos..].starts_with("<![CDATA[") {
            if let Some(end) = src[pos + 9..].find("]]>") {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&src[pos + 9..pos + 9 + end]);
                }
                pos += 9 + end + 3;
                continue;
            } else {
                return Err(OpenMcError::Parse("unterminated CDATA section".into()));
            }
        }
        if src[pos..].starts_with("<!") {
            if let Some(end) = src[pos + 2..].find('>') {
                pos += 2 + end + 1;
                continue;
            } else {
                break;
            }
        }

        // Closing tag: </tag>
        if src[pos..].starts_with("</") {
            pos += 2;
            let end = src[pos..]
                .find('>')
                .ok_or_else(|| OpenMcError::Parse("unterminated closing XML tag".into()))?;
            let close_name = src[pos..pos + end].trim();
            pos += end + 1;

            if let Some(finished) = stack.pop() {
                if finished.tag != close_name {
                    return Err(OpenMcError::Parse(format!(
                        "mismatched closing tag </{close_name}> for <{}>",
                        finished.tag
                    )));
                }
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(finished);
                } else {
                    roots.push(finished);
                }
            }
            continue;
        }

        // Opening or self-closing tag: <tag attr="val" ... />
        pos += 1; // skip '<'
        let tag_start = pos;
        while pos < len && !matches!(bytes[pos], b' ' | b'\t' | b'\r' | b'\n' | b'/' | b'>') {
            pos += 1;
        }
        let tag_name = src[tag_start..pos].to_string();
        let mut attrs = HashMap::new();
        let mut self_closing = false;

        loop {
            while pos < len && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos >= len {
                return Err(OpenMcError::Parse(format!("unterminated tag <{tag_name}>")));
            }
            if bytes[pos] == b'>' {
                pos += 1;
                break;
            }
            if bytes[pos] == b'/' && pos + 1 < len && bytes[pos + 1] == b'>' {
                self_closing = true;
                pos += 2;
                break;
            }

            // Parse attribute name
            let k_start = pos;
            while pos < len
                && !matches!(
                    bytes[pos],
                    b'=' | b' ' | b'\t' | b'\r' | b'\n' | b'>' | b'/'
                )
            {
                pos += 1;
            }
            let key = src[k_start..pos].to_string();
            while pos < len && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < len && bytes[pos] == b'=' {
                pos += 1;
                while pos < len && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < len && (bytes[pos] == b'"' || bytes[pos] == b'\'') {
                    let quote = bytes[pos];
                    pos += 1;
                    let v_start = pos;
                    while pos < len && bytes[pos] != quote {
                        pos += 1;
                    }
                    let val = unescape_xml(&src[v_start..pos]);
                    if pos < len {
                        pos += 1; // skip closing quote
                    }
                    attrs.insert(key, val);
                } else {
                    let v_start = pos;
                    while pos < len && !bytes[pos].is_ascii_whitespace() && bytes[pos] != b'>' {
                        pos += 1;
                    }
                    attrs.insert(key, unescape_xml(&src[v_start..pos]));
                }
            }
        }

        let node = XmlNode {
            tag: tag_name,
            attrs,
            text: String::new(),
            children: Vec::new(),
        };

        if self_closing {
            if let Some(parent) = stack.last_mut() {
                parent.children.push(node);
            } else {
                roots.push(node);
            }
        } else {
            stack.push(node);
        }
    }

    Ok(roots)
}

/// Recursively collect all XML elements matching `tag` in document order.
fn collect_elements<'a>(nodes: &'a [XmlNode], tag: &str, out: &mut Vec<&'a XmlNode>) {
    for n in nodes {
        if n.tag == tag {
            out.push(n);
        }
        collect_elements(&n.children, tag, out);
    }
}

// ---------------------------------------------------------------------------
// OpenMC Domain Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct OpenMcMaterial {
    id: i64,
    name: String,
    /// MCNP-signed density: negative = g/cm^3, positive = atom/(b*cm).
    mcnp_density: f64,
    /// Pairs of (ZAID string like "26056", signed fraction: +ao or -wo).
    nuclides: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
struct OpenMcSurface {
    id: i64,
    stype: String,
    coeffs: Vec<f64>,
    boundary: String,
    name: String,
}

#[derive(Debug, Clone)]
struct OpenMcCell {
    id: i64,
    name: String,
    material: Option<i64>,
    universe: i64,
    region: String,
    fill: Option<i64>,
    translation: Option<[f64; 3]>,
    rotation: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct OpenMcLattice {
    id: i64,
    name: String,
    dimension: Vec<usize>,
    lower_left: Vec<f64>,
    pitch: Vec<f64>,
    universes: Vec<i64>,
    outer: Option<i64>,
}

#[derive(Debug, Clone)]
struct OpenMcHexLattice {
    id: i64,
    name: String,
    n_rings: usize,
    n_axial: Option<usize>,
    orientation: char,
    center: Vec<f64>,
    pitch: Vec<f64>,
    universes: Vec<i64>,
    outer: Option<i64>,
}

/// Parse a whitespace-separated list of 64-bit floating-point numbers.
fn parse_f64_list(s: &str) -> Result<Vec<f64>, OpenMcError> {
    s.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>()
                .map_err(|_| OpenMcError::Parse(format!("invalid float `{tok}` in `{s}`")))
        })
        .collect()
}

/// Parse a whitespace-separated list of 64-bit signed integers.
fn parse_i64_list(s: &str) -> Result<Vec<i64>, OpenMcError> {
    s.split_whitespace()
        .map(|tok| {
            tok.parse::<i64>()
                .map_err(|_| OpenMcError::Parse(format!("invalid integer `{tok}` in `{s}`")))
        })
        .collect()
}

/// Map a chemical element symbol (e.g. `"Fe"`, `"W"`, `"Li"`) to its atomic number `Z`.
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

/// Convert an OpenMC nuclide name (e.g. `Li6`, `Fe56`, `C0`, `U235`, `Am242_m1`) or
/// element symbol (`W`, `Fe`) into an MCNP integer ZAID string (e.g. `3006`, `26056`, `6000`).
pub fn openmc_nuclide_to_zaid(name: &str) -> Option<String> {
    let base = name.split('_').next().unwrap_or(name).trim();
    let split_idx = base
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(base.len());
    let sym = &base[..split_idx];
    let z = element_z(sym)?;
    let a: u32 = if split_idx < base.len() {
        base[split_idx..].parse::<u32>().unwrap_or(0)
    } else {
        0
    };
    Some(format!("{}", z * 1000 + a))
}

/// Parse `<material>` nodes into `OpenMcMaterial` structs.
fn parse_materials(nodes: &[XmlNode]) -> Result<BTreeMap<i64, OpenMcMaterial>, OpenMcError> {
    let mut raw_mats = Vec::new();
    collect_elements(nodes, "material", &mut raw_mats);

    let mut out = BTreeMap::new();
    for mnode in raw_mats {
        let id: i64 = match mnode.attr("id").and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let name = mnode.attr("name").unwrap_or("").replace(['\n', '\r'], " ");

        let mut nuclides: Vec<(String, f64)> = Vec::new();
        let mut ao_sum = 0.0_f64;

        for ch in &mnode.children {
            if ch.tag == "nuclide" || ch.tag == "element" {
                let Some(nname) = ch.attr("name") else {
                    continue;
                };
                let Some(zaid) = openmc_nuclide_to_zaid(nname) else {
                    continue;
                };
                if let Some(ao_s) = ch.attr("ao") {
                    if let Ok(ao) = ao_s.parse::<f64>() {
                        if ao > 0.0 {
                            ao_sum += ao;
                            nuclides.push((zaid, ao));
                        }
                    }
                } else if let Some(wo_s) = ch.attr("wo") {
                    if let Ok(wo) = wo_s.parse::<f64>() {
                        if wo > 0.0 {
                            nuclides.push((zaid, -wo));
                        }
                    }
                }
            }
        }

        // Parse density
        let mut mcnp_density = -1.0_f64;
        if let Some(dnode) = mnode.children.iter().find(|c| c.tag == "density") {
            let units = dnode.attr("units").unwrap_or("g/cm3").to_ascii_lowercase();
            let val = dnode
                .attr("value")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            mcnp_density = match units.as_str() {
                "g/cm3" | "g/cc" => -val.abs(),
                "kg/m3" => -(val.abs() * 1e-3),
                "atom/b-cm" | "a/b-cm" => val.abs(),
                "atom/cm3" | "a/cm3" => val.abs() * 1e-24,
                "sum" => {
                    if ao_sum > 0.0 {
                        ao_sum
                    } else {
                        -1.0
                    }
                }
                _ => -val.abs(),
            };
        }

        // MCNP Mn cards require all fractions to have the same sign (all +ao or all -wo).
        let has_pos = nuclides.iter().any(|(_, v)| *v > 0.0);
        let has_neg = nuclides.iter().any(|(_, v)| *v < 0.0);
        if has_pos && has_neg {
            // Convert all to positive atomic fractions if mixed
            for (_, v) in &mut nuclides {
                *v = v.abs();
            }
        }

        out.insert(
            id,
            OpenMcMaterial {
                id,
                name,
                mcnp_density,
                nuclides,
            },
        );
    }
    Ok(out)
}

/// Convert OpenMC boolean region expression to MCNP syntax.
fn openmc_region_to_mcnp(region: &str) -> String {
    let mut out = String::with_capacity(region.len() + 8);
    for c in region.chars() {
        match c {
            '|' => {
                out.push(' ');
                out.push(':');
                out.push(' ');
            }
            '~' => {
                out.push(' ');
                out.push('#');
            }
            '(' => {
                out.push('(');
            }
            ')' => {
                out.push(')');
            }
            '+' => {
                // MCNP accepts bare positive surface IDs; omit leading '+' inside expressions
                if !out.ends_with(' ') && !out.ends_with('(') && !out.ends_with('#') {
                    out.push(' ');
                }
            }
            _ if c.is_whitespace() => {
                if !out.ends_with(' ') && !out.ends_with('(') && !out.ends_with('#') {
                    out.push(' ');
                }
            }
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

/// Extract `(surface_id, is_positive)` occurrences from an OpenMC region expression.
fn collect_region_surface_senses(region: &str, out: &mut HashMap<i64, (bool, bool)>) {
    let bytes = region.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let mut sign = 1i64;
        if bytes[i] == b'-' {
            sign = -1;
            i += 1;
        } else if bytes[i] == b'+' {
            sign = 1;
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > start {
            if let Ok(sid) = region[start..i].parse::<i64>() {
                let entry = out.entry(sid).or_insert((false, false));
                if sign < 0 {
                    entry.0 = true; // negative sense seen
                } else {
                    entry.1 = true; // positive sense seen
                }
            }
        } else {
            i += 1;
        }
    }
}

/// Convert OpenMC Euler rotation angles (phi, theta, psi in degrees about x, y, z)
/// or a 9-element row-major rotation matrix into a 3×3 matrix.
fn openmc_rotation_matrix(rot: &[f64]) -> [[f64; 3]; 3] {
    if rot.len() >= 9 {
        return [
            [rot[0], rot[1], rot[2]],
            [rot[3], rot[4], rot[5]],
            [rot[6], rot[7], rot[8]],
        ];
    }
    if rot.len() >= 3 {
        // OpenMC rotation=[phi, theta, psi] in degrees: R = Rz(psi) * Ry(theta) * Rx(phi)
        let (sx, cx) = rot[0].to_radians().sin_cos();
        let (sy, cy) = rot[1].to_radians().sin_cos();
        let (sz, cz) = rot[2].to_radians().sin_cos();
        return [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ];
    }
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Format an MCNP card wrapped safely under 78 columns with 5-space continuation lines.
fn write_wrapped_mcnp_card(buf: &mut String, tokens: &[String], comment: Option<&str>) {
    let mut line_len = 0usize;
    for (idx, tok) in tokens.iter().enumerate() {
        let sep = if idx == 0 { 0 } else { 1 };
        if idx > 0 && line_len + sep + tok.len() > 76 {
            buf.push('\n');
            buf.push_str("     ");
            buf.push_str(tok);
            line_len = 5 + tok.len();
        } else {
            if idx > 0 {
                buf.push(' ');
                line_len += 1;
            }
            buf.push_str(tok);
            line_len += tok.len();
        }
    }
    if let Some(c) = comment {
        let clean = c.trim();
        if !clean.is_empty() {
            if line_len + 3 + clean.len() <= 78 {
                let _ = write!(buf, " $ {clean}");
            } else {
                let _ = write!(buf, "\n     $ {clean}");
            }
        }
    }
    buf.push('\n');
}

/// Map an OpenMC `<hex_lattice>` XML `<universes>` token stream into
/// `(i1, i2, iz) -> universe_id` lattice element coordinates matching MCNP `LAT=2`.
fn parse_hex_lattice_grid(
    hlat: &OpenMcHexLattice,
) -> Result<HashMap<(i64, i64, i64), i64>, OpenMcError> {
    let n_rings = hlat.n_rings as i64;
    if n_rings < 1 {
        return Err(OpenMcError::Parse(format!(
            "hex_lattice {} requires n_rings >= 1",
            hlat.id
        )));
    }
    let nz = hlat.n_axial.unwrap_or(1);
    if hlat.n_axial.is_some() && nz < 1 {
        return Err(OpenMcError::Parse(format!(
            "hex_lattice {} requires n_axial >= 1",
            hlat.id
        )));
    }
    let expected_per_slice = if hlat.n_rings == 1 {
        1
    } else {
        1 + 3 * hlat.n_rings * (hlat.n_rings - 1)
    };
    let expected_total = expected_per_slice * nz;
    if hlat.universes.len() < expected_total {
        return Err(OpenMcError::Parse(format!(
            "hex_lattice {} expected {expected_total} universe entries, got {}",
            hlat.id,
            hlat.universes.len()
        )));
    }

    let is_valid = |i1: i64, i2: i64| i1.abs().max(i2.abs()).max((-i1 - i2).abs()) < n_rings;
    let mut grid = HashMap::with_capacity(expected_total);
    let mut j = 0usize;

    for iz in 0..nz as i64 {
        if hlat.orientation == 'y' {
            let mut x = 0i64;
            let mut alpha = n_rings - 1;
            loop {
                let u = *hlat.universes.get(j).ok_or_else(|| {
                    OpenMcError::Parse(format!("hex_lattice {} universe stream truncated", hlat.id))
                })?;
                grid.insert((x, alpha, iz), u);
                j += 1;

                x += 2;
                alpha -= 1;
                if !is_valid(x, alpha) {
                    alpha += x - 1;
                    x = 1 - x;
                    if !is_valid(x, alpha) {
                        x += 2;
                        alpha -= 1;
                        if !is_valid(x, alpha) {
                            break;
                        }
                    }
                }
            }
        } else {
            let mut alpha = 1 - n_rings;
            let mut y = n_rings - 1;
            loop {
                let u = *hlat.universes.get(j).ok_or_else(|| {
                    OpenMcError::Parse(format!("hex_lattice {} universe stream truncated", hlat.id))
                })?;
                grid.insert((alpha, y, iz), u);
                j += 1;

                alpha += 1;
                if !is_valid(alpha, y) {
                    alpha = 1 - n_rings;
                    y -= 1;
                    if y == -n_rings {
                        break;
                    }
                    while !is_valid(alpha, y) {
                        alpha += 1;
                    }
                }
            }
        }
    }

    Ok(grid)
}

/// Format a floating-point number without trailing zeroes or `-0`.
fn fmt_num(v: f64) -> String {
    if v == 0.0 {
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

/// Translate OpenMC XML text (`model.xml` or `geometry.xml` + optional `materials.xml`)
/// into a canonical MCNP input deck ready for `llevant`.
pub fn openmc_xml_to_mcnp(
    geometry_or_model_xml: &str,
    materials_xml: Option<&str>,
) -> Result<String, OpenMcError> {
    let geom_roots = parse_xml_nodes(geometry_or_model_xml)?;
    let mut materials = parse_materials(&geom_roots)?;
    if let Some(mxml) = materials_xml {
        let mat_roots = parse_xml_nodes(mxml)?;
        for (id, mat) in parse_materials(&mat_roots)? {
            materials.insert(id, mat);
        }
    }

    // Parse surfaces
    let mut raw_surfs = Vec::new();
    collect_elements(&geom_roots, "surface", &mut raw_surfs);
    let mut surfaces: BTreeMap<i64, OpenMcSurface> = BTreeMap::new();
    let mut max_surf_id: i64 = 0;

    for snode in raw_surfs {
        let id: i64 = snode
            .attr("id")
            .ok_or_else(|| OpenMcError::Parse("<surface> missing `id` attribute".into()))?
            .parse()
            .map_err(|_| OpenMcError::Parse("invalid surface id".into()))?;
        let stype = snode
            .attr("type")
            .ok_or_else(|| OpenMcError::Parse(format!("surface {id} missing `type`")))?
            .to_ascii_lowercase();
        let coeffs_str = snode.attr("coeffs").unwrap_or("");
        let coeffs = parse_f64_list(coeffs_str)?;
        let boundary = snode
            .attr("boundary")
            .unwrap_or("transmission")
            .to_ascii_lowercase();
        let name = snode.attr("name").unwrap_or("").to_string();
        max_surf_id = max_surf_id.max(id);
        surfaces.insert(
            id,
            OpenMcSurface {
                id,
                stype,
                coeffs,
                boundary,
                name,
            },
        );
    }

    // Parse lattices
    let mut raw_lats = Vec::new();
    collect_elements(&geom_roots, "lattice", &mut raw_lats);
    let mut lattices: BTreeMap<i64, OpenMcLattice> = BTreeMap::new();

    for lnode in raw_lats {
        let id: i64 = lnode
            .attr("id")
            .ok_or_else(|| OpenMcError::Parse("<lattice> missing `id`".into()))?
            .parse()
            .map_err(|_| OpenMcError::Parse("invalid lattice id".into()))?;
        let name = lnode.attr("name").unwrap_or("").to_string();
        let dim_str = lnode
            .child_text("dimension")
            .ok_or_else(|| OpenMcError::Parse(format!("lattice {id} missing <dimension>")))?;
        let dimension: Vec<usize> = dim_str
            .split_whitespace()
            .map(|t| {
                t.parse::<usize>()
                    .map_err(|_| OpenMcError::Parse(format!("invalid lattice dimension `{t}`")))
            })
            .collect::<Result<_, _>>()?;
        let lower_left =
            parse_f64_list(lnode.child_text("lower_left").ok_or_else(|| {
                OpenMcError::Parse(format!("lattice {id} missing <lower_left>"))
            })?)?;
        let pitch = parse_f64_list(
            lnode
                .child_text("pitch")
                .ok_or_else(|| OpenMcError::Parse(format!("lattice {id} missing <pitch>")))?,
        )?;
        let universes = parse_i64_list(
            lnode
                .child_text("universes")
                .ok_or_else(|| OpenMcError::Parse(format!("lattice {id} missing <universes>")))?,
        )?;
        let outer = lnode
            .child_text("outer")
            .and_then(|s| s.parse::<i64>().ok());
        lattices.insert(
            id,
            OpenMcLattice {
                id,
                name,
                dimension,
                lower_left,
                pitch,
                universes,
                outer,
            },
        );
    }

    // Parse hexagonal lattices
    let mut raw_hex_lats = Vec::new();
    collect_elements(&geom_roots, "hex_lattice", &mut raw_hex_lats);
    let mut hex_lattices: BTreeMap<i64, OpenMcHexLattice> = BTreeMap::new();

    for lnode in raw_hex_lats {
        let id: i64 = lnode
            .attr("id")
            .ok_or_else(|| OpenMcError::Parse("<hex_lattice> missing `id`".into()))?
            .parse()
            .map_err(|_| OpenMcError::Parse("invalid hex_lattice id".into()))?;
        let name = lnode.attr("name").unwrap_or("").to_string();
        let n_rings: usize = lnode
            .attr("n_rings")
            .ok_or_else(|| OpenMcError::Parse(format!("hex_lattice {id} missing `n_rings`")))?
            .trim()
            .parse()
            .map_err(|_| OpenMcError::Parse(format!("invalid hex_lattice {id} `n_rings`")))?;
        if n_rings < 1 {
            return Err(OpenMcError::Parse(format!(
                "hex_lattice {id} requires n_rings >= 1"
            )));
        }
        let n_axial = match lnode.attr("n_axial") {
            Some(s) => {
                let v = s.trim().parse::<usize>().map_err(|_| {
                    OpenMcError::Parse(format!("invalid hex_lattice {id} `n_axial`"))
                })?;
                if v < 1 {
                    return Err(OpenMcError::Parse(format!(
                        "hex_lattice {id} requires n_axial >= 1"
                    )));
                }
                Some(v)
            }
            None => None,
        };
        let orient_str = lnode
            .attr("orientation")
            .unwrap_or("y")
            .trim()
            .to_ascii_lowercase();
        let orientation = match orient_str.as_str() {
            "y" => 'y',
            "x" => 'x',
            other => {
                return Err(OpenMcError::Parse(format!(
                    "hex_lattice {id} unsupported orientation `{other}`"
                )));
            }
        };
        let center =
            parse_f64_list(lnode.child_text("center").ok_or_else(|| {
                OpenMcError::Parse(format!("hex_lattice {id} missing <center>"))
            })?)?;
        let pitch = parse_f64_list(
            lnode
                .child_text("pitch")
                .ok_or_else(|| OpenMcError::Parse(format!("hex_lattice {id} missing <pitch>")))?,
        )?;
        if pitch.is_empty() || !(pitch[0] > 0.0) {
            return Err(OpenMcError::Parse(format!(
                "hex_lattice {id} requires positive <pitch>"
            )));
        }
        if center.len() < 2 {
            return Err(OpenMcError::Parse(format!(
                "hex_lattice {id} <center> requires at least 2 coordinates"
            )));
        }
        if n_axial.is_some() && (center.len() < 3 || pitch.len() < 2) {
            return Err(OpenMcError::Parse(format!(
                "3D hex_lattice {id} requires 3D <center> and 2D <pitch>"
            )));
        }
        let universes =
            parse_i64_list(lnode.child_text("universes").ok_or_else(|| {
                OpenMcError::Parse(format!("hex_lattice {id} missing <universes>"))
            })?)?;
        let outer = lnode
            .child_text("outer")
            .and_then(|s| s.parse::<i64>().ok());

        hex_lattices.insert(
            id,
            OpenMcHexLattice {
                id,
                name,
                n_rings,
                n_axial,
                orientation,
                center,
                pitch,
                universes,
                outer,
            },
        );
    }

    // Parse cells
    let mut raw_cells = Vec::new();
    collect_elements(&geom_roots, "cell", &mut raw_cells);
    let mut cells: Vec<OpenMcCell> = Vec::new();
    let mut max_cell_id: i64 = 0;
    let mut all_cell_universes: BTreeSet<i64> = BTreeSet::new();
    let mut filled_targets: HashSet<i64> = HashSet::new();

    for cnode in raw_cells {
        let id: i64 = cnode
            .attr("id")
            .ok_or_else(|| OpenMcError::Parse("<cell> missing `id`".into()))?
            .parse()
            .map_err(|_| OpenMcError::Parse("invalid cell id".into()))?;
        let name = cnode.attr("name").unwrap_or("").to_string();
        let material = match cnode.attr("material") {
            Some(m) if !m.eq_ignore_ascii_case("void") && !m.trim().is_empty() => {
                m.trim().parse::<i64>().ok()
            }
            _ => None,
        };
        let universe = cnode
            .attr("universe")
            .and_then(|u| u.trim().parse::<i64>().ok())
            .unwrap_or(0);
        let region = cnode.attr("region").unwrap_or("").to_string();
        let fill = cnode
            .attr("fill")
            .and_then(|f| f.trim().parse::<i64>().ok());
        let translation = match cnode.attr("translation") {
            Some(ts) => {
                let v = parse_f64_list(ts)?;
                if v.len() >= 3 {
                    Some([v[0], v[1], v[2]])
                } else {
                    None
                }
            }
            None => None,
        };
        let rotation = match cnode.attr("rotation") {
            Some(rs) => Some(parse_f64_list(rs)?),
            None => None,
        };

        max_cell_id = max_cell_id.max(id);
        all_cell_universes.insert(universe);
        if let Some(f) = fill {
            filled_targets.insert(f);
        }
        cells.push(OpenMcCell {
            id,
            name,
            material,
            universe,
            region,
            fill,
            translation,
            rotation,
        });
    }

    if cells.is_empty() {
        return Err(OpenMcError::Parse(
            "no <cell> elements found in OpenMC XML".into(),
        ));
    }

    // Also mark universes referenced inside lattices as non-root
    for lat in lattices.values() {
        for &u in &lat.universes {
            filled_targets.insert(u);
        }
        if let Some(ou) = lat.outer {
            filled_targets.insert(ou);
        }
    }
    for hlat in hex_lattices.values() {
        for &u in &hlat.universes {
            filled_targets.insert(u);
        }
        if let Some(ou) = hlat.outer {
            filled_targets.insert(ou);
        }
    }

    // Detect OpenMC root universe: universe 0 if present, else lowest cell universe not referenced by any fill/lattice
    let root_universe = if all_cell_universes.contains(&0) {
        0
    } else {
        all_cell_universes
            .iter()
            .copied()
            .find(|u| !filled_targets.contains(u))
            .unwrap_or_else(|| *all_cell_universes.iter().next().unwrap())
    };

    let mut max_univ_id = all_cell_universes.iter().copied().max().unwrap_or(0);
    for lat in lattices.values() {
        max_univ_id = max_univ_id.max(lat.id);
    }
    for hlat in hex_lattices.values() {
        max_univ_id = max_univ_id.max(hlat.id);
    }

    // Synthesize MCNP transforms (TRn), lattice helper universes, and extra bounding planes
    type McnpTransform = (i64, [f64; 3], Option<[[f64; 3]; 3]>);
    let mut transforms: Vec<McnpTransform> = Vec::new();
    let mut next_tr_id: i64 = 1;
    let mut next_surf_id: i64 = (max_surf_id + 100).max(10000);
    let mut next_cell_id: i64 = (max_cell_id + 100).max(10000);
    let base_univ_id: i64 = (max_univ_id + 100).max(1000);
    let mut next_lat_univs = base_univ_id..;

    let mut extra_surfaces: Vec<(i64, String, String)> = Vec::new();
    let mut extra_cells: Vec<(Vec<String>, Option<String>)> = Vec::new();

    // Map each OpenMC lattice ID -> (synthesized MCNP universe ID, element [0,0,0] center offset)
    let mut lat_to_mcnp: HashMap<i64, (i64, [f64; 3])> = HashMap::new();
    for (u_lat, lat) in next_lat_univs.by_ref().zip(lattices.values()) {
        let nx = *lat.dimension.first().unwrap_or(&1);
        let ny = *lat.dimension.get(1).unwrap_or(&1);
        let is_3d = lat.dimension.len() >= 3 && lat.lower_left.len() >= 3 && lat.pitch.len() >= 3;
        let nz = if is_3d { lat.dimension[2] } else { 1 };
        let px = *lat.pitch.first().unwrap_or(&1.0);
        let py = *lat.pitch.get(1).unwrap_or(&1.0);
        let pz = if is_3d { lat.pitch[2] } else { 0.0 };

        let x0 = *lat.lower_left.first().unwrap_or(&0.0);
        let y0 = *lat.lower_left.get(1).unwrap_or(&0.0);
        let z0 = if is_3d { lat.lower_left[2] } else { 0.0 };

        // Center of element [0, 0, 0] in the lattice's parent frame
        let c0 = [
            x0 + 0.5 * px,
            y0 + 0.5 * py,
            if is_3d { z0 + 0.5 * pz } else { 0.0 },
        ];

        let lat_cell_id = next_cell_id;
        next_cell_id += 1;

        // Unit box centered at (0, 0, 0) in lattice-local frame: [-px/2, +px/2] x [-py/2, +py/2]
        let sx_lo = next_surf_id;
        let sx_hi = next_surf_id + 1;
        let sy_lo = next_surf_id + 2;
        let sy_hi = next_surf_id + 3;
        next_surf_id += 4;
        extra_surfaces.push((sx_lo, "px".into(), fmt_num(-0.5 * px)));
        extra_surfaces.push((sx_hi, "px".into(), fmt_num(0.5 * px)));
        extra_surfaces.push((sy_lo, "py".into(), fmt_num(-0.5 * py)));
        extra_surfaces.push((sy_hi, "py".into(), fmt_num(0.5 * py)));

        let mut region_toks = vec![
            format!("-{sx_hi}"),
            format!("{sx_lo}"),
            format!("-{sy_hi}"),
            format!("{sy_lo}"),
        ];
        if is_3d {
            let sz_lo = next_surf_id;
            let sz_hi = next_surf_id + 1;
            next_surf_id += 2;
            extra_surfaces.push((sz_lo, "pz".into(), fmt_num(-0.5 * pz)));
            extra_surfaces.push((sz_hi, "pz".into(), fmt_num(0.5 * pz)));
            region_toks.push(format!("-{sz_hi}"));
            region_toks.push(format!("{sz_lo}"));
        }

        // Convert OpenMC lattice ordering (z ascending, y DESCENDING, x ascending)
        // to MCNP Fortran ordering (i=x fastest, j=y middle, k=z slowest)
        let mut fill_vals: Vec<String> = Vec::with_capacity(nx * ny * nz);
        for iz in 0..nz {
            for iy in 0..ny {
                let openmc_row = ny - 1 - iy;
                for ix in 0..nx {
                    let openmc_idx = (iz * ny + openmc_row) * nx + ix;
                    let u_elem = lat
                        .universes
                        .get(openmc_idx)
                        .copied()
                        .or(lat.outer)
                        .unwrap_or(u_lat);
                    fill_vals.push(u_elem.to_string());
                }
            }
        }

        let mut ctoks = vec![lat_cell_id.to_string(), "0".to_string()];
        ctoks.extend(region_toks);
        ctoks.push(format!("u={u_lat}"));
        ctoks.push("lat=1".to_string());
        ctoks.push("imp:n=1".to_string());
        ctoks.push(format!(
            "fill=0:{} 0:{} 0:{}",
            nx.saturating_sub(1),
            ny.saturating_sub(1),
            nz.saturating_sub(1)
        ));
        ctoks.extend(fill_vals);

        let comment = if lat.name.is_empty() {
            format!("OpenMC lattice {}", lat.id)
        } else {
            format!("OpenMC lattice {} ({})", lat.id, lat.name)
        };
        extra_cells.push((ctoks, Some(comment)));
        lat_to_mcnp.insert(lat.id, (u_lat, c0));
    }

    // Synthesize MCNP LAT=2 cells for OpenMC <hex_lattice> elements
    let sqrt3_half = 3.0_f64.sqrt() * 0.5;
    for (u_lat, hlat) in next_lat_univs.zip(hex_lattices.values()) {
        let grid = parse_hex_lattice_grid(hlat)?;
        let is_3d = hlat.n_axial.is_some();
        let nz = hlat.n_axial.unwrap_or(1);
        let p_xy = hlat.pitch[0];
        let hp = 0.5 * p_xy;
        let pz = if is_3d { hlat.pitch[1] } else { 0.0 };

        let c0 = [
            hlat.center[0],
            hlat.center[1],
            if is_3d {
                hlat.center[2] + (0.5 - 0.5 * nz as f64) * pz
            } else {
                0.0
            },
        ];

        let lat_cell_id = next_cell_id;
        next_cell_id += 1;

        let s0 = next_surf_id;
        let s1 = next_surf_id + 1;
        let s2 = next_surf_id + 2;
        let s3 = next_surf_id + 3;
        let s4 = next_surf_id + 4;
        let s5 = next_surf_id + 5;
        next_surf_id += 6;

        // MCNP LAT=2 surface order: +a, -a, +b, -b, +(b-a), -(b-a), then optional +z, -z
        if hlat.orientation == 'x' {
            extra_surfaces.push((s0, "px".into(), fmt_num(hp)));
            extra_surfaces.push((s1, "px".into(), fmt_num(-hp)));
            extra_surfaces.push((
                s2,
                "p".into(),
                format!("0.5 {} 0 {}", fmt_num(sqrt3_half), fmt_num(hp)),
            ));
            extra_surfaces.push((
                s3,
                "p".into(),
                format!("0.5 {} 0 {}", fmt_num(sqrt3_half), fmt_num(-hp)),
            ));
            extra_surfaces.push((
                s4,
                "p".into(),
                format!("-0.5 {} 0 {}", fmt_num(sqrt3_half), fmt_num(hp)),
            ));
            extra_surfaces.push((
                s5,
                "p".into(),
                format!("-0.5 {} 0 {}", fmt_num(sqrt3_half), fmt_num(-hp)),
            ));
        } else {
            extra_surfaces.push((
                s0,
                "p".into(),
                format!("{} 0.5 0 {}", fmt_num(sqrt3_half), fmt_num(hp)),
            ));
            extra_surfaces.push((
                s1,
                "p".into(),
                format!("{} 0.5 0 {}", fmt_num(sqrt3_half), fmt_num(-hp)),
            ));
            extra_surfaces.push((s2, "py".into(), fmt_num(hp)));
            extra_surfaces.push((s3, "py".into(), fmt_num(-hp)));
            extra_surfaces.push((
                s4,
                "p".into(),
                format!("-{} 0.5 0 {}", fmt_num(sqrt3_half), fmt_num(hp)),
            ));
            extra_surfaces.push((
                s5,
                "p".into(),
                format!("-{} 0.5 0 {}", fmt_num(sqrt3_half), fmt_num(-hp)),
            ));
        }

        let mut region_toks = vec![
            format!("-{s0}"),
            format!("{s1}"),
            format!("-{s2}"),
            format!("{s3}"),
            format!("-{s4}"),
            format!("{s5}"),
        ];
        if is_3d {
            let hpz = 0.5 * pz;
            let s6 = next_surf_id;
            let s7 = next_surf_id + 1;
            next_surf_id += 2;
            extra_surfaces.push((s6, "pz".into(), fmt_num(hpz)));
            extra_surfaces.push((s7, "pz".into(), fmt_num(-hpz)));
            region_toks.push(format!("-{s6}"));
            region_toks.push(format!("{s7}"));
        }

        let r_max = (hlat.n_rings as i64) - 1;
        let pad = if hlat.outer.is_some() {
            if r_max <= 10 {
                2
            } else {
                1
            }
        } else {
            0
        };
        let i_lo = -(r_max + pad);
        let i_hi = r_max + pad;
        let j_lo = -(r_max + pad);
        let j_hi = r_max + pad;
        let k_pad = i64::from(is_3d && hlat.outer.is_some());
        let k_lo = -k_pad;
        let k_hi = if is_3d { (nz as i64 - 1) + k_pad } else { 0 };

        let ni = (i_hi - i_lo + 1) as usize;
        let nj = (j_hi - j_lo + 1) as usize;
        let nk = (k_hi - k_lo + 1) as usize;
        let mut fill_vals = Vec::with_capacity(ni * nj * nk);
        for k in k_lo..=k_hi {
            for j in j_lo..=j_hi {
                for i in i_lo..=i_hi {
                    let u_elem = grid
                        .get(&(i, j, k))
                        .copied()
                        .or(hlat.outer)
                        .unwrap_or(u_lat);
                    fill_vals.push(u_elem.to_string());
                }
            }
        }

        let mut ctoks = vec![lat_cell_id.to_string(), "0".to_string()];
        ctoks.extend(region_toks);
        ctoks.push(format!("u={u_lat}"));
        ctoks.push("lat=2".to_string());
        ctoks.push("imp:n=1".to_string());
        ctoks.push(format!("fill={i_lo}:{i_hi} {j_lo}:{j_hi} {k_lo}:{k_hi}"));
        ctoks.extend(fill_vals);

        let comment = if hlat.name.is_empty() {
            format!("OpenMC hex_lattice {}", hlat.id)
        } else {
            format!("OpenMC hex_lattice {} ({})", hlat.id, hlat.name)
        };
        extra_cells.push((ctoks, Some(comment)));
        lat_to_mcnp.insert(hlat.id, (u_lat, c0));
    }

    // Track root-cell surface senses to synthesize an exact IMP:N=0 exterior graveyard
    let mut root_senses: HashMap<i64, (bool, bool)> = HashMap::new();
    for cell in &cells {
        if cell.universe == root_universe {
            collect_region_surface_senses(&cell.region, &mut root_senses);
        }
    }

    let mut mcnp = String::with_capacity(4096);
    mcnp.push_str("OpenMC XML model loaded natively by llevant-openmc\n");

    // -----------------------------------------------------------------------
    // Block 1: Cell Cards
    // -----------------------------------------------------------------------
    for cell in &cells {
        let mut toks: Vec<String> = Vec::new();
        toks.push(cell.id.to_string());
        match cell.material {
            Some(mid) => {
                toks.push(mid.to_string());
                let rho = materials.get(&mid).map(|m| m.mcnp_density).unwrap_or(-1.0);
                toks.push(fmt_num(rho));
            }
            None => {
                toks.push("0".to_string());
            }
        }

        let reg_mcnp = openmc_region_to_mcnp(&cell.region);
        if reg_mcnp.is_empty() {
            // Unbounded cell in a child universe (common in OpenMC pin/lattice universes)
            // MCNP requires a non-empty geometry expression: use an enclosing huge sphere or dummy plane union
            let big_s = next_surf_id;
            next_surf_id += 1;
            extra_surfaces.push((big_s, "so".into(), "1e6".into()));
            toks.push(format!("-{big_s}"));
        } else {
            for t in reg_mcnp.split_whitespace() {
                toks.push(t.to_string());
            }
        }

        if cell.universe != root_universe {
            toks.push(format!("u={}", cell.universe));
        }

        if let Some(fill_id) = cell.fill {
            let (target_u, lat_offset) = match lat_to_mcnp.get(&fill_id) {
                Some(&(u_lat, c0)) => (u_lat, Some(c0)),
                None => (fill_id, None),
            };

            let mut disp = cell.translation.unwrap_or([0.0, 0.0, 0.0]);
            let rot_mat = cell.rotation.as_deref().map(openmc_rotation_matrix);
            if let Some(c0) = lat_offset {
                if let Some(r) = rot_mat {
                    for k in 0..3 {
                        disp[k] += r[0][k] * c0[0] + r[1][k] * c0[1] + r[2][k] * c0[2];
                    }
                } else {
                    disp[0] += c0[0];
                    disp[1] += c0[1];
                    disp[2] += c0[2];
                }
            }
            let has_tr = disp.iter().any(|&v| v.abs() > 1e-12) || rot_mat.is_some();

            if has_tr {
                let tr_id = next_tr_id;
                next_tr_id += 1;
                transforms.push((tr_id, disp, rot_mat));
                toks.push(format!("fill={target_u} ({tr_id})"));
            } else {
                toks.push(format!("fill={target_u}"));
            }
        }

        toks.push("imp:n=1".to_string());
        let comment = (!cell.name.is_empty()).then_some(cell.name.as_str());
        write_wrapped_mcnp_card(&mut mcnp, &toks, comment);
    }

    for (ctoks, comment) in &extra_cells {
        write_wrapped_mcnp_card(&mut mcnp, ctoks, comment.as_deref());
    }

    // Synthesize exterior IMP:N=0 graveyard cell from vacuum boundaries (or single-sense outer surfaces)
    let mut exterior_terms: Vec<String> = Vec::new();
    let vac_surfs: Vec<i64> = surfaces
        .values()
        .filter(|s| s.boundary == "vacuum")
        .map(|s| s.id)
        .collect();

    if !vac_surfs.is_empty() {
        for sid in vac_surfs {
            if let Some(&(seen_neg, seen_pos)) = root_senses.get(&sid) {
                if seen_neg && !seen_pos {
                    exterior_terms.push(format!("{sid}"));
                } else if seen_pos && !seen_neg {
                    exterior_terms.push(format!("-{sid}"));
                } else {
                    exterior_terms.push(format!("{sid}"));
                }
            }
        }
    } else {
        // Fallback for minimal OpenMC decks without explicit boundary="vacuum":
        // surfaces used exclusively with one sign in root cells form the outer envelope.
        for (&sid, &(seen_neg, seen_pos)) in &root_senses {
            if seen_neg && !seen_pos {
                exterior_terms.push(format!("{sid}"));
            } else if seen_pos && !seen_neg {
                exterior_terms.push(format!("-{sid}"));
            }
        }
    }

    if !exterior_terms.is_empty() {
        let grave_id = next_cell_id.max(99999);
        let mut gtoks = vec![grave_id.to_string(), "0".to_string()];
        if exterior_terms.len() == 1 {
            gtoks.push(exterior_terms[0].clone());
        } else {
            let joined = exterior_terms.join(" : ");
            gtoks.push(format!("({joined})"));
        }
        gtoks.push("imp:n=0".to_string());
        write_wrapped_mcnp_card(&mut mcnp, &gtoks, Some("OpenMC exterior vacuum graveyard"));
    }

    // Blank line separating Cell and Surface blocks
    mcnp.push('\n');

    // -----------------------------------------------------------------------
    // Block 2: Surface Cards
    // -----------------------------------------------------------------------
    for surf in surfaces.values() {
        let (mnemonic, params): (&str, Vec<f64>) = match surf.stype.as_str() {
            "x-plane" => ("px", surf.coeffs.clone()),
            "y-plane" => ("py", surf.coeffs.clone()),
            "z-plane" => ("pz", surf.coeffs.clone()),
            "plane" => ("p", surf.coeffs.clone()),
            "sphere" => ("s", surf.coeffs.clone()),
            "x-cylinder" => ("c/x", surf.coeffs.clone()),
            "y-cylinder" => ("c/y", surf.coeffs.clone()),
            "z-cylinder" => ("c/z", surf.coeffs.clone()),
            "x-cone" => ("k/x", surf.coeffs.clone()),
            "y-cone" => ("k/y", surf.coeffs.clone()),
            "z-cone" => ("k/z", surf.coeffs.clone()),
            "x-torus" => ("tx", surf.coeffs.clone()),
            "y-torus" => ("ty", surf.coeffs.clone()),
            "z-torus" => ("tz", surf.coeffs.clone()),
            "quadric" => ("gq", surf.coeffs.clone()),
            other => {
                return Err(OpenMcError::Parse(format!(
                    "unsupported OpenMC surface type `{other}` on surface {}",
                    surf.id
                )));
            }
        };

        let mut stoks = Vec::with_capacity(2 + params.len());
        stoks.push(surf.id.to_string());
        stoks.push(mnemonic.to_string());
        for c in params {
            stoks.push(fmt_num(c));
        }
        let comment = (!surf.name.is_empty()).then_some(surf.name.as_str());
        write_wrapped_mcnp_card(&mut mcnp, &stoks, comment);
    }

    for (sid, mnem, pstr) in &extra_surfaces {
        let _ = writeln!(mcnp, "{sid} {mnem} {pstr}");
    }

    // Blank line separating Surface and Data blocks
    mcnp.push('\n');

    // -----------------------------------------------------------------------
    // Block 3: Data Cards (Materials & Transforms)
    // -----------------------------------------------------------------------
    mcnp.push_str("mode n\n");

    for (tr_id, disp, rot_opt) in &transforms {
        let mut ttoks = vec![
            format!("tr{tr_id}"),
            fmt_num(disp[0]),
            fmt_num(disp[1]),
            fmt_num(disp[2]),
        ];
        if let Some(r) = rot_opt {
            for row in r {
                for &v in row {
                    ttoks.push(fmt_num(v));
                }
            }
        }
        write_wrapped_mcnp_card(&mut mcnp, &ttoks, None);
    }

    for mat in materials.values() {
        if mat.nuclides.is_empty() {
            continue;
        }
        let mut mtoks = Vec::with_capacity(1 + mat.nuclides.len() * 2);
        mtoks.push(format!("m{}", mat.id));
        for (zaid, frac) in &mat.nuclides {
            mtoks.push(zaid.clone());
            mtoks.push(fmt_num(*frac));
        }
        let comment = (!mat.name.is_empty()).then_some(mat.name.as_str());
        write_wrapped_mcnp_card(&mut mcnp, &mtoks, comment);
    }

    Ok(mcnp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify OpenMC nuclide/element names convert to standard MCNP ZAID strings.
    #[test]
    fn converts_nuclides_and_elements() {
        assert_eq!(openmc_nuclide_to_zaid("Li6").as_deref(), Some("3006"));
        assert_eq!(openmc_nuclide_to_zaid("Fe56").as_deref(), Some("26056"));
        assert_eq!(openmc_nuclide_to_zaid("C0").as_deref(), Some("6000"));
        assert_eq!(openmc_nuclide_to_zaid("W184").as_deref(), Some("74184"));
        assert_eq!(openmc_nuclide_to_zaid("U235").as_deref(), Some("92235"));
        assert_eq!(openmc_nuclide_to_zaid("Am242_m1").as_deref(), Some("95242"));
        assert_eq!(openmc_nuclide_to_zaid("Cu").as_deref(), Some("29000"));
    }

    /// Verify combined OpenMC `<model>` XML translates surfaces, materials, and exterior graveyard.
    #[test]
    fn translates_model_xml_with_surfaces_materials_and_graveyard() {
        let xml = r#"<?xml version='1.0' encoding='utf-8'?>
<model>
  <materials>
    <material id="1" name="Tungsten Armor">
      <density units="g/cm3" value="19.25"/>
      <nuclide ao="0.3064" name="W184"/>
      <nuclide ao="0.2843" name="W186"/>
    </material>
    <material id="2" name="PbLi Breeder">
      <density units="g/cm3" value="9.36"/>
      <nuclide ao="0.83" name="Pb208"/>
      <nuclide ao="0.153" name="Li6"/>
      <nuclide ao="0.017" name="Li7"/>
    </material>
  </materials>
  <geometry>
    <surface coeffs="0.0 0.0 0.0 10.0" id="1" type="sphere"/>
    <surface boundary="vacuum" coeffs="0.0 0.0 0.0 25.0" id="2" type="sphere"/>
    <cell id="1" material="1" name="Armor" region="-1" universe="1"/>
    <cell id="2" material="2" name="Breeder" region="1 -2" universe="1"/>
  </geometry>
</model>"#;

        let mcnp = openmc_xml_to_mcnp(xml, None).unwrap();
        assert!(mcnp.contains("1 1 -19.25 -1 imp:n=1"));
        assert!(mcnp.contains("2 2 -9.36 1 -2 imp:n=1"));
        assert!(mcnp.contains("imp:n=0"));
        assert!(mcnp.contains("1 s 0 0 0 10"));
        assert!(mcnp.contains("2 s 0 0 0 25"));
        assert!(mcnp.contains("m1 74184 0.3064 74186 0.2843"));
        assert!(mcnp.contains("m2 82208 0.83 3006 0.153 3007 0.017"));
    }

    /// Verify OpenMC rectilinear `<lattice>` and child `<universe>` translation into MCNP `LAT=1`.
    #[test]
    fn translates_rectilinear_lattice_and_child_universes() {
        let xml = r#"<model>
  <materials>
    <material id="1" name="UO2"><density units="g/cm3" value="10.4"/><nuclide ao="0.33" name="U235"/><nuclide ao="0.67" name="O16"/></material>
    <material id="2" name="Water"><density units="g/cm3" value="0.72"/><nuclide ao="2.0" name="H1"/><nuclide ao="1.0" name="O16"/></material>
  </materials>
  <geometry>
    <surface id="1" type="z-cylinder" coeffs="0 0 0.4"/>
    <surface id="10" type="x-plane" coeffs="-1.26" boundary="vacuum"/>
    <surface id="11" type="x-plane" coeffs="1.26" boundary="vacuum"/>
    <surface id="12" type="y-plane" coeffs="-1.26" boundary="vacuum"/>
    <surface id="13" type="y-plane" coeffs="1.26" boundary="vacuum"/>
    <surface id="14" type="z-plane" coeffs="-5.0" boundary="vacuum"/>
    <surface id="15" type="z-plane" coeffs="5.0" boundary="vacuum"/>
    <cell id="1" universe="2" material="1" region="-1"/>
    <cell id="2" universe="2" material="2" region="1"/>
    <lattice id="100" name="2x2 Pin Array">
      <dimension>2 2</dimension>
      <lower_left>-1.26 -1.26</lower_left>
      <pitch>1.26 1.26</pitch>
      <universes>
        2 2
        2 2
      </universes>
    </lattice>
    <cell id="10" universe="1" fill="100" region="10 -11 12 -13 14 -15"/>
  </geometry>
</model>"#;

        let mcnp = openmc_xml_to_mcnp(xml, None).unwrap();
        let norm = mcnp.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(norm.contains("lat=1"), "{mcnp}");
        assert!(norm.contains("fill=0:1 0:1 0:0 2 2 2 2"), "{mcnp}");
        assert!(norm.contains("tr1 -0.63 -0.63 0"), "{mcnp}");
    }

    /// Verify 2D OpenMC `<hex_lattice>` translation for both `y` and `x` orientations.
    #[test]
    fn translates_2d_hex_lattice_y_and_x_orientations() {
        for orient in ["y", "x"] {
            let xml = format!(
                r#"<model>
  <materials>
    <material id="1" name="Fuel"><density units="g/cm3" value="10.4"/><nuclide ao="1.0" name="U235"/></material>
    <material id="2" name="Water"><density units="g/cm3" value="0.72"/><nuclide ao="2.0" name="H1"/><nuclide ao="1.0" name="O16"/></material>
    <material id="3" name="Poison"><density units="g/cm3" value="2.5"/><nuclide ao="1.0" name="B10"/></material>
  </materials>
  <geometry>
    <surface id="1" type="z-cylinder" coeffs="0 0 0.4"/>
    <surface id="2" type="z-cylinder" coeffs="0 0 3.0" boundary="vacuum"/>
    <cell id="1" universe="1" material="1" region="-1"/>
    <cell id="2" universe="1" material="2" region="1"/>
    <cell id="3" universe="3" material="3" region="-1"/>
    <cell id="4" universe="3" material="2" region="1"/>
    <cell id="5" universe="2" material="2"/>
    <hex_lattice id="100" n_rings="2" orientation="{orient}">
      <pitch>2.0</pitch>
      <outer>2</outer>
      <center>1.5 -0.5</center>
      <universes>
        1 1
        1 3 1
        1 1
      </universes>
    </hex_lattice>
    <cell id="10" universe="0" fill="100" region="-2"/>
  </geometry>
</model>"#
            );
            let mcnp = openmc_xml_to_mcnp(&xml, None).unwrap();
            let norm = mcnp.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(norm.contains("lat=2"), "{mcnp}");
            assert!(norm.contains("fill=-3:3 -3:3 0:0"), "{mcnp}");
            assert!(norm.contains("tr1 1.5 -0.5 0"), "{mcnp}");
        }
    }

    /// Verify 3D OpenMC `<hex_lattice>` with multiple axial tiers translates `n_axial` pitch & fill bounds.
    #[test]
    fn translates_3d_hex_lattice_with_axial_layers() {
        let xml = r#"<model>
  <materials>
    <material id="1" name="Fuel"><density units="g/cm3" value="10.4"/><nuclide ao="1.0" name="U235"/></material>
    <material id="2" name="Poison"><density units="g/cm3" value="2.5"/><nuclide ao="1.0" name="B10"/></material>
    <material id="3" name="Coolant"><density units="g/cm3" value="0.72"/><nuclide ao="1.0" name="H1"/></material>
  </materials>
  <geometry>
    <surface id="1" type="z-cylinder" coeffs="0 0 0.4"/>
    <surface id="2" type="z-cylinder" coeffs="0 0 3.0" boundary="vacuum"/>
    <surface id="3" type="z-plane" coeffs="-6.0" boundary="vacuum"/>
    <surface id="4" type="z-plane" coeffs="6.0" boundary="vacuum"/>
    <cell id="1" universe="1" material="1" region="-1"/>
    <cell id="2" universe="1" material="3" region="1"/>
    <cell id="3" universe="2" material="2" region="-1"/>
    <cell id="4" universe="2" material="3" region="1"/>
    <cell id="5" universe="3" material="3"/>
    <hex_lattice id="100" n_rings="2" n_axial="2" orientation="x">
      <pitch>2.0 4.0</pitch>
      <outer>3</outer>
      <center>0.0 0.0 1.0</center>
      <universes>
        1 1 1 1 1 1 1
        2 2 2 2 2 2 2
      </universes>
    </hex_lattice>
    <cell id="10" universe="0" fill="100" region="-2 3 -4"/>
  </geometry>
</model>"#;
        let mcnp = openmc_xml_to_mcnp(xml, None).unwrap();
        let norm = mcnp.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(norm.contains("lat=2"), "{mcnp}");
        assert!(norm.contains("fill=-3:3 -3:3 -1:2"), "{mcnp}");
        // Element [0,0,0] axial center: 1.0 + (0.5 - 1.0)*4.0 = -1.0
        assert!(norm.contains("tr1 0 0 -1"), "{mcnp}");
    }
}
