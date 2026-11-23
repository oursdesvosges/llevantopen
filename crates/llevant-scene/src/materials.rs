//! What materials look like, for realistic images: each `Mn` card is
//! sorted by its composition into [`Category`]s (steel, concrete, water,
//! ...) and given an [`Appearance`] for physically based shading (linear
//! albedo, roughness, metalness).
//!
//! The classification is a heuristic on element mass fractions:
//!
//! 1. Hydrogen and oxygen in the ratio of water are water.
//! 2. Oxygen left over oxidises the other elements, the most
//!    electropositive first (Ca, Mg, Na, K, Li, Ba, Sr, Al, Si, B, Ti, ...,
//!    then Fe and the other metals): those oxides are mineral, concrete when
//!    the material has calcium, ceramic otherwise.
//! 3. Hydrogen and carbon left are polymer (carbon with little hydrogen is
//!    carbon: graphite, carbides; boron, silicon, phosphorus and sulphur
//!    without oxygen go there too).
//! 4. Metals left are steel (Fe, Cr, Ni, Mn, Co, V, Nb, rare ones), copper
//!    (Cu, Zn, Sn, Ag, Au), tungsten (W, Ta, Mo, Hf, Re), lead (Pb, Bi, Tl,
//!    Hg) or light metal (Al, Mg, Ti, Zr, Be, Li).
//! 5. A cell lighter than 0.02 g/cm³ is gas, whatever its material.
//!
//! Homogenised materials (steel and water smeared together, as neutronics
//! models often have) are not forced into one category: the parts are
//! weighed by volume (mass over a nominal density) and their appearances
//! blended, water and gas counting less (they are clear).
//!
//! The heuristic will be wrong somewhere: a [`Spec`] per material
//! overrides it (`12 steel`, `13 copper roughness=0.2`,
//! `14 colour=#806040 metalness=0`), see [`parse_specs`] and [`Looks::new`].
//! [`Looks`] belong to whoever draws the model (the model never changes):
//! new specs make new looks.

use std::fmt;
use std::str::FromStr;

use llevant::{Error, Model, Result};
use rustc_hash::FxHashMap;

/// A kind of material, as it looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// No material (also cells that only hold a fill).
    Void,
    /// Cells lighter than 0.02 g/cm³.
    Gas,
    Water,
    /// Hydrocarbons: polyethylene, borated polyethylene, resins.
    Polymer,
    /// Oxides with calcium.
    Concrete,
    /// Other oxides: alumina, silica, breeder ceramics, glass.
    Ceramic,
    /// Graphite, carbides, boron.
    Carbon,
    /// Iron, chromium, nickel alloys and the metals that look like them.
    Steel,
    /// Copper and its alloys.
    Copper,
    /// Tungsten and other refractory metals.
    Tungsten,
    /// Lead, lead-lithium, bismuth.
    Lead,
    /// Aluminium, magnesium, titanium, zirconium, beryllium, lithium.
    LightMetal,
    /// A material number without an `Mn` card, or an empty card.
    Unknown,
}

impl Category {
    pub const ALL: [Category; 13] = [
        Category::Void,
        Category::Gas,
        Category::Water,
        Category::Polymer,
        Category::Concrete,
        Category::Ceramic,
        Category::Carbon,
        Category::Steel,
        Category::Copper,
        Category::Tungsten,
        Category::Lead,
        Category::LightMetal,
        Category::Unknown,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::Void => "void",
            Category::Gas => "gas",
            Category::Water => "water",
            Category::Polymer => "polymer",
            Category::Concrete => "concrete",
            Category::Ceramic => "ceramic",
            Category::Carbon => "carbon",
            Category::Steel => "steel",
            Category::Copper => "copper",
            Category::Tungsten => "tungsten",
            Category::Lead => "lead",
            Category::LightMetal => "light-metal",
            Category::Unknown => "unknown",
        }
    }

    /// Density (g/cm³) used to turn the mass of a part into its volume.
    fn nominal_density(self) -> f64 {
        match self {
            Category::Void => 0.001,
            // Coolant helium, air: little mass, and they count little.
            Category::Gas => 0.1,
            Category::Water => 1.0,
            Category::Polymer => 1.1,
            Category::Concrete => 2.3,
            Category::Ceramic => 2.5,
            Category::Carbon => 2.0,
            Category::Steel => 7.9,
            Category::Copper => 8.9,
            Category::Tungsten => 19.3,
            Category::Lead => 11.3,
            Category::LightMetal => 2.7,
            Category::Unknown => 1.0,
        }
    }

    /// [`Category::nominal_density`], minerals heavier for their share of
    /// iron oxide (5.2 g/cm³).
    fn density_with(self, iron_oxide: f64) -> f64 {
        let d = self.nominal_density();
        match self {
            Category::Concrete | Category::Ceramic => d + (5.2 - d) * iron_oxide,
            _ => d,
        }
    }

    /// How much a part shows in a blend, per unit volume: water and gas
    /// are clear.
    fn visual_weight(self) -> f64 {
        match self {
            Category::Water => 0.3,
            Category::Gas | Category::Void => 0.05,
            _ => 1.0,
        }
    }

    /// The look of the pure category.
    pub fn appearance(self) -> Appearance {
        // Albedos are linear; metals give their reflectance at normal
        // incidence (F0).
        let (albedo, roughness, metalness) = match self {
            Category::Void => (srgb_to_linear3([245, 245, 245]), 1.0, 0.0),
            Category::Gas => ([0.80, 0.86, 0.92], 1.0, 0.0),
            // Water: the colour of the light it scatters back.
            Category::Water => ([0.012, 0.11, 0.17], 0.05, 0.0),
            Category::Polymer => ([0.74, 0.74, 0.66], 0.45, 0.0),
            Category::Concrete => ([0.36, 0.35, 0.32], 0.9, 0.0),
            Category::Ceramic => ([0.72, 0.70, 0.65], 0.55, 0.0),
            Category::Carbon => ([0.045, 0.045, 0.05], 0.7, 0.0),
            Category::Steel => ([0.56, 0.57, 0.58], 0.35, 1.0),
            Category::Copper => ([0.955, 0.64, 0.54], 0.3, 1.0),
            Category::Tungsten => ([0.50, 0.50, 0.48], 0.4, 1.0),
            Category::Lead => ([0.45, 0.46, 0.50], 0.6, 1.0),
            Category::LightMetal => ([0.91, 0.92, 0.92], 0.4, 1.0),
            Category::Unknown => ([0.4, 0.4, 0.4], 0.6, 0.0),
        };
        Appearance {
            albedo,
            roughness,
            metalness,
            // Water half clear: half the glossy depth, half an opaque
            // surface.
            clarity: if self == Category::Water { 0.5 } else { 0.0 },
            category: self,
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Category {
    type Err = Error;
    fn from_str(s: &str) -> Result<Category> {
        let s = s.trim().to_ascii_lowercase().replace('_', "-");
        let alias = match s.as_str() {
            "light-metal" | "aluminium" | "aluminum" => Some(Category::LightMetal),
            "graphite" => Some(Category::Carbon),
            "plastic" => Some(Category::Polymer),
            "air" => Some(Category::Gas),
            _ => None,
        };
        alias
            .or_else(|| Category::ALL.into_iter().find(|c| c.name() == s))
            .ok_or_else(|| {
                let names: Vec<&str> = Category::ALL.iter().map(|c| c.name()).collect();
                Error::Invalid(format!(
                    "unknown material category `{s}` (one of {})",
                    names.join(", ")
                ))
            })
    }
}

/// How a surface looks under physically based shading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Appearance {
    /// Linear RGB: diffuse colour, or reflectance at normal incidence for
    /// metals.
    pub albedo: [f32; 3],
    /// 0 (mirror) to 1 (matte).
    pub roughness: f32,
    /// 0 (dielectric) to 1 (metal).
    pub metalness: f32,
    /// 0 (opaque) to 1 (clear: a glossy surface over a coloured depth).
    pub clarity: f32,
    /// The main category.
    pub category: Category,
}

impl Appearance {
    /// The albedo as 8-bit sRGB (how slices and the palette show it).
    pub fn srgb(&self) -> [u8; 3] {
        self.albedo.map(linear_to_srgb8)
    }
}

/// sRGB-encoded fraction to linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear fraction to sRGB-encoded fraction.
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn linear_to_srgb8(c: f32) -> u8 {
    (linear_to_srgb(c) * 255.0 + 0.5) as u8
}

fn srgb_to_linear3(c: [u8; 3]) -> [f32; 3] {
    c.map(|x| srgb_to_linear(x as f32 / 255.0))
}

/// What a material is made of, by category: volume fractions (summing to 1,
/// largest first).
#[derive(Clone, Debug, PartialEq)]
pub struct Mix {
    pub parts: Vec<(Category, f64)>,
    /// Share of the mineral mass that is iron oxide (darkens concrete).
    pub iron_oxide: f64,
    /// Mean atomic mass (u per atom), to read atom densities.
    pub atomic_mass: f64,
    /// Density (g/cm³) the parts would have at their nominal densities.
    pub nominal_density: f64,
}

impl Mix {
    fn single(c: Category) -> Mix {
        Mix {
            parts: vec![(c, 1.0)],
            iron_oxide: 0.0,
            atomic_mass: 0.0,
            nominal_density: c.nominal_density(),
        }
    }

    /// The category with the largest volume.
    pub fn main(&self) -> Category {
        self.parts.first().map_or(Category::Unknown, |p| p.0)
    }

    /// The parts' appearances blended by visible volume.
    pub fn appearance(&self) -> Appearance {
        let mut sum = 0.0f64;
        let mut albedo = [0.0f64; 3];
        let mut rough = 0.0f64;
        let mut metal = 0.0f64;
        let mut clear = 0.0f64;
        for &(c, v) in &self.parts {
            let mut a = c.appearance();
            if c == Category::Concrete || c == Category::Ceramic {
                // Iron oxides (heavy concrete, magnetite) darken and warm it.
                let k = (0.8 * self.iron_oxide) as f32;
                let rust = [0.09, 0.07, 0.06];
                for i in 0..3 {
                    a.albedo[i] += k * (rust[i] - a.albedo[i]);
                }
            }
            let w = v * c.visual_weight();
            sum += w;
            for i in 0..3 {
                albedo[i] += w * a.albedo[i] as f64;
            }
            rough += w * a.roughness as f64;
            metal += w * a.metalness as f64;
            clear += w * a.clarity as f64;
        }
        if !(sum > 0.0) {
            return Category::Unknown.appearance();
        }
        Appearance {
            albedo: albedo.map(|x| (x / sum) as f32),
            roughness: (rough / sum) as f32,
            metalness: (metal / sum) as f32,
            clarity: (clear / sum) as f32,
            category: self.main(),
        }
    }
}

impl fmt::Display for Mix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for &(c, v) in &self.parts {
            if v < 0.005 {
                continue;
            }
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "{c} {:.0}%", 100.0 * v)?;
        }
        Ok(())
    }
}

/// Standard atomic weights, Z = 1 to 100.
const ATOMIC_WEIGHT: [f64; 100] = [
    1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305,
    26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996,
    54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798,
    85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98.0, 101.07, 102.91, 106.42, 107.87, 112.41,
    114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,
    145.0, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49,
    180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209.0,
    210.0, 222.0, 223.0, 226.0, 227.0, 232.04, 231.04, 238.03, 237.0, 244.0, 243.0, 247.0, 247.0,
    251.0, 252.0, 257.0,
];

/// Mass of oxygen bound per unit mass of each element in its usual oxide,
/// in the order oxygen is handed out.
const OXIDES: [(usize, f64); 30] = [
    (20, 0.399), // CaO
    (12, 0.658), // MgO
    (11, 0.348), // Na2O
    (19, 0.205), // K2O
    (3, 1.153),  // Li2O
    (56, 0.117), // BaO
    (38, 0.183), // SrO
    (13, 0.889), // Al2O3
    (14, 1.139), // SiO2
    (5, 2.220),  // B2O3
    (22, 0.668), // TiO2
    (15, 1.291), // P2O5
    (16, 1.497), // SO3
    (40, 0.351), // ZrO2
    (4, 1.775),  // BeO
    (26, 0.382), // Fe3O4
    (25, 0.291), // MnO
    (24, 0.462), // Cr2O3
    (28, 0.273), // NiO
    (27, 0.271), // CoO
    (23, 0.785), // V2O5
    (41, 0.431), // Nb2O5
    (42, 0.500), // MoO3
    (74, 0.261), // WO3
    (73, 0.221), // Ta2O5
    (29, 0.252), // CuO
    (30, 0.245), // ZnO
    (50, 0.270), // SnO2
    (82, 0.077), // PbO
    (83, 0.115), // Bi2O3
];

/// Category of an element as a metal (or what is left of it unoxidised).
fn metal_category(z: usize) -> Option<Category> {
    Some(match z {
        3 | 4 | 12 | 13 | 22 | 40 => Category::LightMetal,
        29 | 30 | 47 | 50 | 79 => Category::Copper,
        42 | 72 | 73 | 74 | 75 => Category::Tungsten,
        80..=83 => Category::Lead,
        5 | 14 | 15 | 16 | 6 => Category::Carbon,
        2 | 10 | 18 | 36 | 54 | 86 => Category::Gas,
        // Halogens left over: not seen.
        9 | 17 | 35 | 53 => return None,
        _ => Category::Steel,
    })
}

/// Element (Z) of a ZAID: `26056.31c`, `26000`, `1001`.
fn zaid_element(zaid: &str) -> Option<usize> {
    let head = zaid.split('.').next()?.trim();
    let n: u64 = head.parse().ok()?;
    let z = (n / 1000) as usize;
    (1..=100).contains(&z).then_some(z)
}

/// Element mass fractions (by Z) of a material card and its mean atomic
/// mass (u per atom); `None` if the card has no known element.
fn mass_fractions(entries: &[(String, f64)]) -> Option<([f64; 101], f64)> {
    let mut w = [0.0f64; 101];
    let by_weight = entries.first().is_some_and(|e| e.1 < 0.0);
    let mut atoms = 0.0;
    let mut mass = 0.0;
    for (zaid, f) in entries {
        let Some(z) = zaid_element(zaid) else {
            continue;
        };
        let f = f.abs();
        if !f.is_finite() {
            continue;
        }
        let a = ATOMIC_WEIGHT[z - 1];
        if by_weight {
            w[z] += f;
            mass += f;
            atoms += f / a;
        } else {
            w[z] += f * a;
            mass += f * a;
            atoms += f;
        }
    }
    if !(mass > 0.0) {
        return None;
    }
    for x in w.iter_mut() {
        *x /= mass;
    }
    Some((w, mass / atoms))
}

/// The mass (u) of one atom of a ZAID: the standard atomic weight of a
/// natural element (`26000`), else the mass formula's for the nuclide
/// (`26056.80c`; within 0.12 % of the measured masses, deuterium the
/// worst, and 0.01 % past A = 15). `None` if the ZAID names no element.
pub fn zaid_mass(zaid: &str) -> Option<f64> {
    let head = zaid.split('.').next()?.trim();
    let n: u64 = head.parse().ok()?;
    let z = (n / 1000) as usize;
    let a = n % 1000;
    if !(1..=100).contains(&z) {
        return None;
    }
    // Metastable states (A + 300 + 100 m) keep their ground-state mass.
    let a = if a > 300 { (a - 300) % 100 } else { a };
    if a == 0 || (a as usize) < z {
        return Some(ATOMIC_WEIGHT[z - 1]);
    }
    Some(nuclide_mass(z as f64, a as f64))
}

/// Atomic mass (u) of nuclide (Z, A) by the semi-empirical mass formula.
fn nuclide_mass(z: f64, a: f64) -> f64 {
    const H: f64 = 1.007_825_032; // the hydrogen atom
    const N: f64 = 1.008_664_916;
    const MEV: f64 = 931.494_102;
    if a <= 1.0 {
        return if z >= 1.0 { H } else { N };
    }
    let n = a - z;
    let pairing = match (z as u64 % 2, n as u64 % 2) {
        (0, 0) => 1.0,
        (1, 1) => -1.0,
        _ => 0.0,
    } * 11.18
        / a.sqrt();
    let b = 15.75 * a
        - 17.8 * a.powf(2.0 / 3.0)
        - 0.711 * z * (z - 1.0) / a.cbrt()
        - 23.7 * (n - z).powi(2) / a
        + pairing;
    z * H + n * N - b.max(0.0) / MEV
}

/// The element mass fractions of a material card, by Z, summing to 1
/// (empty if it names no element).
pub fn element_mass_fractions(entries: &[(String, f64)]) -> Vec<(usize, f64)> {
    match mass_fractions(entries) {
        Some((w, _)) => (1..=100)
            .filter(|&z| w[z] > 0.0)
            .map(|z| (z, w[z]))
            .collect(),
        None => Vec::new(),
    }
}

/// Mean atomic mass (u per atom) of a material card, to read atom
/// densities ([`grams_per_cm3`]); 0 if the card has no known element.
pub fn atomic_mass(entries: &[(String, f64)]) -> f64 {
    mass_fractions(entries).map_or(0.0, |(_, a)| a)
}

/// A material's mass split into categories.
struct Split {
    mass: FxHashMap<Category, f64>,
    /// Share of the mineral mass that is iron oxide.
    iron_oxide: f64,
}

impl Split {
    /// Density (g/cm³) of the parts at their nominal densities.
    fn density(&self) -> f64 {
        let (mut m, mut v) = (0.0, 0.0);
        for (&c, &x) in &self.mass {
            m += x;
            v += x / c.density_with(self.iron_oxide);
        }
        if v > 0.0 {
            m / v
        } else {
            f64::NAN
        }
    }
}

/// Classify a material card (`(zaid, fraction)` pairs, negative fractions
/// by weight) whose density is unknown.
pub fn analyse(entries: &[(String, f64)]) -> Mix {
    analyse_at(entries, None)
}

/// Classify a material card used at `density` (g/cm³).
///
/// Hydrogen and oxygen in the ratio of water are free water, unless they
/// can be bound: in a polymer at least as heavy as the water (its oxygen
/// and hydrogen: epoxy, cellulose) or in a mineral at least as heavy
/// (water of hydration; carbon in a mineral four times heavier is
/// carbonate). In a mineral four times heavier than water and polymer
/// together (concrete) they are bound. Otherwise the reading whose
/// density, from the parts' nominal densities, is nearer `density` is
/// kept; without a density, bound in a polymer, free otherwise.
pub fn analyse_at(entries: &[(String, f64)], density: Option<f64>) -> Mix {
    let Some((w, atomic_mass)) = mass_fractions(entries) else {
        return Mix::single(Category::Unknown);
    };
    let free = split(&w);
    let s = match bound(&free) {
        None => free,
        Some((b, true)) => b,
        Some((b, false)) => match density.filter(|r| *r > 0.0 && r.is_finite()) {
            Some(rho) => {
                let off = |s: &Split| (s.density() / rho).ln().abs();
                if off(&free) <= off(&b) {
                    free
                } else {
                    b
                }
            }
            None if b.mass.contains_key(&Category::Polymer) => b,
            None => free,
        },
    };
    finish(s, atomic_mass)
}

/// The reading of `free` with its water bound, if it can be, and whether
/// it surely is.
fn bound(free: &Split) -> Option<(Split, bool)> {
    let get = |c: Category| free.mass.get(&c).copied().unwrap_or(0.0);
    let water = get(Category::Water);
    if water <= 0.0 {
        return None;
    }
    let polymer = get(Category::Polymer);
    let mineral_cat = [Category::Concrete, Category::Ceramic]
        .into_iter()
        .find(|c| get(*c) > 0.0);
    let mineral = mineral_cat.map_or(0.0, get);
    let mut b = Split {
        mass: free.mass.clone(),
        iron_oxide: free.iron_oxide,
    };
    let sure = mineral >= 4.0 * (water + polymer);
    let mut to = |from: Category, into: Category, m: f64| {
        b.mass.remove(&from);
        *b.mass.entry(into).or_insert(0.0) += m;
    };
    match mineral_cat {
        Some(c) if polymer > 0.0 && mineral >= 4.0 * polymer && mineral >= water => {
            // Carbonate and water of hydration.
            to(Category::Polymer, c, polymer);
            to(Category::Water, c, water);
        }
        _ if polymer >= water => to(Category::Water, Category::Polymer, water),
        Some(c) if mineral >= water => to(Category::Water, c, water),
        _ => return None,
    }
    // Iron oxide's share of a mineral that took the water.
    if let Some(c) = mineral_cat {
        let after = b.mass.get(&c).copied().unwrap_or(0.0);
        if after > 0.0 {
            b.iron_oxide = free.iron_oxide * mineral / after;
        }
    }
    Some((b, sure))
}

/// Split element mass fractions `w` into categories, hydrogen and oxygen
/// in the ratio of water as free water.
fn split(w: &[f64; 101]) -> Split {
    let mut w = *w;
    let calcium = w[20] > 0.0;
    // Hydrocarbons are told from graphite and carbides by the hydrogen
    // before water takes its share.
    let organic = w[1] > 0.0 && w[1] >= 0.03 * w[6];
    let mut parts: FxHashMap<Category, f64> = FxHashMap::default();
    let mut add = |c: Category, m: f64| {
        if m > 0.0 {
            *parts.entry(c).or_insert(0.0) += m;
        }
    };

    // 1. Water.
    let water = f64::min(w[1] / 0.1119, w[8] / 0.8881);
    w[1] -= 0.1119 * water;
    w[8] -= 0.8881 * water;
    add(Category::Water, water);

    // 2. Oxides.
    let mut mineral = 0.0;
    let mut iron = 0.0;
    for (z, per) in OXIDES {
        if w[8] <= 0.0 {
            break;
        }
        if w[z] <= 0.0 {
            continue;
        }
        let bound = f64::min(w[z], w[8] / per);
        w[8] -= bound * per;
        w[z] -= bound;
        mineral += bound * (1.0 + per);
        if z == 26 {
            iron += bound * (1.0 + 0.382);
        }
    }
    // Oxygen left over (peroxides, a card short of cations) counts as
    // mineral.
    mineral += w[8].max(0.0);
    w[8] = 0.0;
    if mineral > 0.0 {
        let c = if calcium {
            Category::Concrete
        } else {
            Category::Ceramic
        };
        add(c, mineral);
    }

    // 3. Hydrocarbons.
    if organic {
        add(Category::Polymer, w[1] + w[6] + w[7] + w[9] + w[17]);
        for z in [1, 6, 7, 9, 17] {
            w[z] = 0.0;
        }
    }
    // Nitrogen and hydrogen left (ammonia, traces) are not seen.
    w[1] = 0.0;
    w[7] = 0.0;

    // 4. Metals and the rest. Carbon, boron, silicon, phosphorus and
    // sulphur under a tenth of the metals' mass are alloying elements: they
    // join the main metal.
    let mut rest: FxHashMap<Category, f64> = FxHashMap::default();
    for z in 1..=100 {
        if w[z] > 0.0 {
            if let Some(c) = metal_category(z) {
                *rest.entry(c).or_insert(0.0) += w[z];
            }
        }
    }
    let nonmetal = rest.get(&Category::Carbon).copied().unwrap_or(0.0);
    let main_metal = rest
        .iter()
        .filter(|(c, _)| !matches!(c, Category::Carbon | Category::Gas))
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(c, m)| (*c, *m));
    let metals: f64 = rest
        .iter()
        .filter(|(c, _)| !matches!(c, Category::Carbon | Category::Gas))
        .map(|(_, m)| m)
        .sum();
    if let Some((c, _)) = main_metal {
        if nonmetal > 0.0 && nonmetal < 0.1 * metals {
            rest.remove(&Category::Carbon);
            *rest.entry(c).or_insert(0.0) += nonmetal;
        }
    }
    for (c, m) in rest {
        add(c, m);
    }
    Split {
        mass: parts,
        iron_oxide: if mineral > 0.0 { iron / mineral } else { 0.0 },
    }
}

/// The volume fractions of a split.
fn finish(s: Split, atomic_mass: f64) -> Mix {
    let nominal_density = s.density();
    let mut v: Vec<(Category, f64)> = s
        .mass
        .iter()
        .map(|(&c, &m)| (c, m / c.density_with(s.iron_oxide)))
        .collect();
    let total: f64 = v.iter().map(|p| p.1).sum();
    if !(total > 0.0) {
        return Mix::single(Category::Unknown);
    }
    for p in v.iter_mut() {
        p.1 /= total;
    }
    // Specks below 1% of the volume do not change the look.
    v.retain(|p| p.1 >= 0.01);
    let kept: f64 = v.iter().map(|p| p.1).sum();
    for p in v.iter_mut() {
        p.1 /= kept;
    }
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Mix {
        parts: v,
        iron_oxide: s.iron_oxide,
        atomic_mass,
        nominal_density,
    }
}

/// Density in g/cm³ from a cell density with the MCNP sign (negative:
/// g/cm³, positive: atoms/b·cm).
pub fn grams_per_cm3(density: f64, atomic_mass: f64) -> Option<f64> {
    if density < 0.0 {
        Some(-density)
    } else if density > 0.0 && atomic_mass > 0.0 {
        Some(density * atomic_mass / 0.602_214_076)
    } else {
        None
    }
}

#[inline]
fn material_look(looks: &Looks, id: i64) -> Option<&MaterialLook> {
    looks
        .materials
        .binary_search_by_key(&id, |m| m.id)
        .ok()
        .map(|i| &looks.materials[i])
}

/// Mass and sampling standard error `(mass_g, mass_sigma_g)` of one cell
/// given its volume in cm³ and standard error, or `None` for void cells or
/// positive atom densities whose material atomic mass is unknown.
pub fn cell_mass(
    model: &Model,
    looks: &Looks,
    cell_id: i64,
    volume_cm3: f64,
    sigma_cm3: f64,
) -> Option<(f64, f64)> {
    let idx = model.cell_index(cell_id)?;
    let cell = model.cell(idx);
    if cell.material == 0 {
        return None;
    }
    let atomic_mass = material_look(looks, cell.material).map_or(0.0, |m| m.mix.atomic_mass);
    let rho = grams_per_cm3(cell.density, atomic_mass)?;
    Some((volume_cm3 * rho, sigma_cm3 * rho))
}

/// Mass and standard error `(mass_g, mass_sigma_g)` of a volume group.
///
/// Only `Status::Inside` groups have a defined mass. For `GroupBy::Material`
/// and `GroupBy::Universe`, providing per-cell groups (`cell_groups`) sums
/// cell masses and combines their uncertainties in quadrature, which handles
/// materials used at multiple densities or heterogeneous universes accurately.
pub fn group_mass(
    model: &Model,
    looks: &Looks,
    by: llevant::GroupBy,
    group: &llevant::Group,
    cell_groups: Option<&[llevant::Group]>,
) -> Option<(f64, f64)> {
    if group.status != llevant::Status::Inside {
        return None;
    }
    match by {
        llevant::GroupBy::Cell => {
            let cid = group.number?;
            cell_mass(model, looks, cid, group.volume.estimate, group.volume.sigma)
        }
        llevant::GroupBy::Label => {
            let cidx = group.label.as_ref()?.cell()?;
            let cid = model.cell(cidx).id;
            cell_mass(model, looks, cid, group.volume.estimate, group.volume.sigma)
        }
        llevant::GroupBy::Material => {
            let mid = group.number?;
            if mid == 0 {
                return None;
            }
            if let Some(cgs) = cell_groups {
                let mut total_mass = 0.0;
                let mut total_var = 0.0;
                let mut any = false;
                for cg in cgs {
                    if cg.status != llevant::Status::Inside {
                        continue;
                    }
                    let Some(cid) = cg.number else {
                        continue;
                    };
                    let Some(cidx) = model.cell_index(cid) else {
                        continue;
                    };
                    if model.cell(cidx).material != mid {
                        continue;
                    }
                    if let Some((m, dm)) =
                        cell_mass(model, looks, cid, cg.volume.estimate, cg.volume.sigma)
                    {
                        any = true;
                        total_mass += m;
                        total_var += dm * dm;
                    }
                }
                return any.then(|| (total_mass, total_var.sqrt()));
            }
            let ml = material_look(looks, mid)?;
            if ml.density.0.is_finite()
                && ml.density.1.is_finite()
                && (ml.density.1 - ml.density.0).abs() <= 1e-12 * ml.density.1.abs().max(1.0)
            {
                Some((
                    group.volume.estimate * ml.density.0,
                    group.volume.sigma * ml.density.0,
                ))
            } else {
                None
            }
        }
        llevant::GroupBy::Universe => {
            let uid = group.number?;
            if let Some(cgs) = cell_groups {
                let mut total_mass = 0.0;
                let mut total_var = 0.0;
                let mut any = false;
                for cg in cgs {
                    if cg.status != llevant::Status::Inside {
                        continue;
                    }
                    let Some(cid) = cg.number else {
                        continue;
                    };
                    let Some(cidx) = model.cell_index(cid) else {
                        continue;
                    };
                    let cell = model.cell(cidx);
                    if model.units()[cell.unit as usize].universe != uid || cell.material == 0 {
                        continue;
                    }
                    if let Some((m, dm)) =
                        cell_mass(model, looks, cid, cg.volume.estimate, cg.volume.sigma)
                    {
                        any = true;
                        total_mass += m;
                        total_var += dm * dm;
                    }
                }
                return any.then(|| (total_mass, total_var.sqrt()));
            }
            let mut uniform_rho: Option<f64> = None;
            for cell in model.cells() {
                if cell.fill.is_some() || model.units()[cell.unit as usize].universe != uid {
                    continue;
                }
                let rho = if cell.material == 0 {
                    0.0
                } else {
                    let am = material_look(looks, cell.material).map_or(0.0, |m| m.mix.atomic_mass);
                    grams_per_cm3(cell.density, am)?
                };
                match uniform_rho {
                    None => uniform_rho = Some(rho),
                    Some(prev) if (rho - prev).abs() <= 1e-12 * prev.abs().max(1.0) => {}
                    Some(_) => return None,
                }
            }
            let rho = uniform_rho.filter(|&r| r > 0.0)?;
            Some((group.volume.estimate * rho, group.volume.sigma * rho))
        }
    }
}

/// Batch version of [`group_mass`]: when `cell_groups` is given for
/// `Material` or `Universe` grouping, aggregates all cell masses in a single
/// `O(C + M)` pass instead of scanning all cells per group.
pub fn group_masses(
    model: &Model,
    looks: &Looks,
    by: llevant::GroupBy,
    groups: &[llevant::Group],
    cell_groups: Option<&[llevant::Group]>,
) -> Vec<Option<(f64, f64)>> {
    if matches!(by, llevant::GroupBy::Material | llevant::GroupBy::Universe) {
        if let Some(cgs) = cell_groups {
            let mut accum: FxHashMap<i64, (f64, f64)> = FxHashMap::default();
            for cg in cgs {
                if cg.status != llevant::Status::Inside {
                    continue;
                }
                let Some(cid) = cg.number else {
                    continue;
                };
                let Some(cidx) = model.cell_index(cid) else {
                    continue;
                };
                let cell = model.cell(cidx);
                if cell.material == 0 {
                    continue;
                }
                let Some((m, dm)) =
                    cell_mass(model, looks, cid, cg.volume.estimate, cg.volume.sigma)
                else {
                    continue;
                };
                let key = match by {
                    llevant::GroupBy::Material => cell.material,
                    llevant::GroupBy::Universe => model.units()[cell.unit as usize].universe,
                    _ => unreachable!(),
                };
                let entry = accum.entry(key).or_insert((0.0, 0.0));
                entry.0 += m;
                entry.1 += dm * dm;
            }
            return groups
                .iter()
                .map(|g| {
                    if g.status != llevant::Status::Inside {
                        return None;
                    }
                    let num = g.number?;
                    if by == llevant::GroupBy::Material && num == 0 {
                        return None;
                    }
                    accum.get(&num).map(|&(m, var)| (m, var.sqrt()))
                })
                .collect();
        }
    }
    groups
        .iter()
        .map(|g| group_mass(model, looks, by, g, cell_groups))
        .collect()
}

/// Changes to the look of a material: a category to start from and any of
/// colour, roughness, metalness and clarity. Written `steel`,
/// `copper roughness=0.2`, `colour=#806040 metalness=0`, `albedo=0.2,0.2,0.25`
/// (linear) or `roughness=0.8`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Spec {
    pub category: Option<Category>,
    /// Linear RGB.
    pub albedo: Option<[f32; 3]>,
    pub roughness: Option<f32>,
    pub metalness: Option<f32>,
    pub clarity: Option<f32>,
}

impl Spec {
    /// `base` changed as the spec says.
    pub fn apply(&self, base: Appearance) -> Appearance {
        let mut a = match self.category {
            Some(c) => c.appearance(),
            None => base,
        };
        if let Some(x) = self.albedo {
            a.albedo = x;
        }
        if let Some(x) = self.roughness {
            a.roughness = x;
        }
        if let Some(x) = self.metalness {
            a.metalness = x;
        }
        if let Some(x) = self.clarity {
            a.clarity = x;
        }
        a
    }
}

impl FromStr for Spec {
    type Err = Error;
    fn from_str(s: &str) -> Result<Spec> {
        let mut out = Spec::default();
        let bad = |m: String| Error::Invalid(format!("material look `{s}`: {m}"));
        let unit = |k: &str, v: &str| -> Result<f32> {
            match v.parse::<f32>() {
                Ok(x) if (0.0..=1.0).contains(&x) => Ok(x),
                _ => Err(bad(format!("{k} must be a number from 0 to 1"))),
            }
        };
        for tok in s.split_whitespace() {
            match tok.split_once('=') {
                None => {
                    if out.category.is_some() {
                        return Err(bad(format!("two categories (`{tok}`)")));
                    }
                    out.category = Some(tok.parse().map_err(|e: Error| bad(e.to_string()))?);
                }
                Some((k, v)) => match k.to_ascii_lowercase().as_str() {
                    "roughness" => out.roughness = Some(unit(k, v)?),
                    "metalness" => out.metalness = Some(unit(k, v)?),
                    "clarity" => out.clarity = Some(unit(k, v)?),
                    "colour" | "color" => {
                        let hex = v.trim_start_matches('#');
                        let n = u32::from_str_radix(hex, 16)
                            .ok()
                            .filter(|_| hex.len() == 6)
                            .ok_or_else(|| bad(format!("{k} must be #rrggbb")))?;
                        let rgb = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
                        out.albedo = Some(srgb_to_linear3(rgb));
                    }
                    "albedo" => {
                        let v: Vec<f32> =
                            v.split(',').map(|x| unit(k, x)).collect::<Result<_>>()?;
                        if v.len() != 3 {
                            return Err(bad("albedo needs r,g,b".into()));
                        }
                        out.albedo = Some([v[0], v[1], v[2]]);
                    }
                    _ => return Err(bad(format!("unknown key `{k}`"))),
                },
            }
        }
        Ok(out)
    }
}

/// Read a file of looks: one material per line, `<number> <spec>`
/// (`12 steel roughness=0.2`); `#` starts a comment.
pub fn parse_specs(text: &str) -> Result<Vec<(i64, Spec)>> {
    let mut out = Vec::new();
    for (k, line) in text.lines().enumerate() {
        let line = strip_comment(line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (id, spec) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let id: i64 = id
            .trim_start_matches(['m', 'M'])
            .parse()
            .map_err(|_| Error::Invalid(format!("line {}: bad material number `{id}`", k + 1)))?;
        let spec: Spec = spec
            .parse()
            .map_err(|e| Error::Invalid(format!("line {}: {e}", k + 1)))?;
        out.push((id, spec));
    }
    Ok(out)
}

/// A line without its comment: `#` starts one, except right after `=`
/// (`colour=#806040`).
fn strip_comment(line: &str) -> &str {
    let mut prev = ' ';
    for (i, ch) in line.char_indices() {
        if ch == '#' && prev != '=' {
            return &line[..i];
        }
        prev = ch;
    }
    line
}

/// The look of one material of a model.
#[derive(Clone, Debug)]
pub struct MaterialLook {
    pub id: i64,
    pub mix: Mix,
    /// After the spec, if any.
    pub appearance: Appearance,
    pub spec: Option<Spec>,
    /// Cells that use the material.
    pub cells: usize,
    /// Smallest and largest cell density, g/cm³ (NaN if unknown).
    pub density: (f64, f64),
}

/// The looks of a model's materials and cells.
#[derive(Clone, Debug)]
pub struct Looks {
    /// By cell index.
    pub cells: Vec<Appearance>,
    /// By material number, ascending (material 0 left out).
    pub materials: Vec<MaterialLook>,
}

impl Looks {
    /// The errors of [`Looks::new`] without its work (a spec of a material
    /// the model does not use or define); the materials of `m`, sorted.
    pub fn check(m: &Model, specs: &[(i64, Spec)]) -> Result<Vec<i64>> {
        let mut ids = m.material_ids();
        ids.extend(m.cells().iter().map(|c| c.material).filter(|&x| x != 0));
        ids.sort_unstable();
        ids.dedup();
        for &(id, _) in specs {
            if id != 0 && ids.binary_search(&id).is_err() {
                return Err(Error::Invalid(format!(
                    "material {id} has a look but no card and no cell"
                )));
            }
        }
        Ok(ids)
    }

    /// Classify every material of `m`, then apply `specs`.
    /// [`Error::Invalid`] for a spec of a material the model does not use or
    /// define.
    pub fn new(m: &Model, specs: &[(i64, Spec)]) -> Result<Looks> {
        let ids = Looks::check(m, specs)?;
        let spec_of: FxHashMap<i64, &Spec> = specs.iter().map(|(i, s)| (*i, s)).collect();
        // Densities of each material's cells (g/cm³), to tell water from
        // bound hydrogen and oxygen: the median is used.
        let mut rhos: Vec<Vec<f64>> = vec![Vec::new(); ids.len()];
        let masses: Vec<f64> = ids
            .iter()
            .map(|&id| m.material(id).map_or(0.0, atomic_mass))
            .collect();
        for c in m.cells().iter().filter(|c| c.material != 0) {
            let k = ids.binary_search(&c.material).expect("listed");
            if let Some(rho) = grams_per_cm3(c.density, masses[k]) {
                rhos[k].push(rho);
            }
        }
        let mut materials: Vec<MaterialLook> = ids
            .iter()
            .zip(rhos.iter_mut())
            .map(|(&id, rho)| {
                rho.sort_by(f64::total_cmp);
                let median = rho.get(rho.len() / 2).copied();
                let mix = match m.material(id) {
                    Some(e) => analyse_at(e, median),
                    None => Mix::single(Category::Unknown),
                };
                let base = mix.appearance();
                let spec = spec_of.get(&id).map(|s| (*s).clone());
                MaterialLook {
                    id,
                    appearance: spec.as_ref().map_or(base, |s| s.apply(base)),
                    mix,
                    spec,
                    cells: 0,
                    density: (f64::NAN, f64::NAN),
                }
            })
            .collect();
        let void = match spec_of.get(&0) {
            Some(s) => s.apply(Category::Void.appearance()),
            None => Category::Void.appearance(),
        };
        let gas = Category::Gas.appearance();
        let cells = m
            .cells()
            .iter()
            .map(|c| {
                if c.material == 0 {
                    return void;
                }
                let k = ids.binary_search(&c.material).expect("listed");
                let ml = &mut materials[k];
                ml.cells += 1;
                let rho = grams_per_cm3(c.density, ml.mix.atomic_mass).unwrap_or(f64::NAN);
                if rho.is_finite() {
                    ml.density.0 = if ml.density.0.is_nan() {
                        rho
                    } else {
                        ml.density.0.min(rho)
                    };
                    ml.density.1 = if ml.density.1.is_nan() {
                        rho
                    } else {
                        ml.density.1.max(rho)
                    };
                }
                if ml.spec.is_none() && rho < 0.02 {
                    gas
                } else {
                    ml.appearance
                }
            })
            .collect();
        for ml in materials.iter_mut() {
            if ml.spec.is_none() && ml.density.1 < 0.02 {
                ml.appearance = gas;
            }
        }
        Ok(Looks { cells, materials })
    }

    /// Look of the cell at index `cell`.
    pub fn cell(&self, cell: u32) -> &Appearance {
        &self.cells[cell as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nuclide_masses_near_the_measured_ones() {
        for (zaid, m) in [
            ("1001.80c", 1.007825),
            ("1002", 2.014102),
            ("8016.31c", 15.994915),
            ("26056", 55.934936),
            ("92235.80c", 235.043928),
            ("82208", 207.976652),
        ] {
            let got = zaid_mass(zaid).unwrap();
            assert!((got / m - 1.0).abs() < 2e-3, "{zaid}: {got} vs {m}");
        }
        assert!((zaid_mass("26000").unwrap() - 55.845).abs() < 1e-3);
        assert!(zaid_mass("0").is_none());
        let w = element_mass_fractions(&[("1001".into(), 2.0), ("8016".into(), 1.0)]);
        assert_eq!(w.len(), 2);
        assert!((w[0].1 - 0.1119).abs() < 1e-3, "{w:?}");
    }

    fn e(v: &[(&str, f64)]) -> Vec<(String, f64)> {
        v.iter().map(|(z, f)| (z.to_string(), *f)).collect()
    }

    #[test]
    fn pure_materials() {
        let cases: [(&[(&str, f64)], Category); 12] = [
            (&[("1001.31c", 2.0), ("8016.31c", 1.0)], Category::Water),
            (&[("1001.31c", 2.0), ("6000.31c", 1.0)], Category::Polymer),
            (
                &[
                    ("26000", -0.68),
                    ("24000", -0.17),
                    ("28000", -0.12),
                    ("42000", -0.025),
                    ("25055", -0.005),
                ],
                Category::Steel,
            ),
            // Stainless steel with its carbon, silicon, phosphorus, sulphur.
            (
                &[
                    ("26000", -0.66),
                    ("24000", -0.17),
                    ("28000", -0.12),
                    ("14000", -0.01),
                    ("6000", -0.0008),
                    ("15031", -0.0004),
                ],
                Category::Steel,
            ),
            (&[("29063", 0.69), ("29065", 0.31)], Category::Copper),
            (&[("74184.70c", 1.0)], Category::Tungsten),
            (&[("82208", 0.83), ("3006", 0.17)], Category::Lead),
            (&[("5010", 0.8), ("6012", 0.2)], Category::Carbon),
            (&[("13027", 2.0), ("8016", 3.0)], Category::Ceramic),
            (&[("4009", 1.0)], Category::LightMetal),
            (&[("2004.32c", 1.0)], Category::Gas),
            // An insulator: its oxygen and hydrogen are not water.
            (
                &[
                    ("1001", -0.027),
                    ("6012", -0.69),
                    ("7014", -0.074),
                    ("8016", -0.21),
                ],
                Category::Polymer,
            ),
        ];
        for (entries, want) in cases {
            let mix = analyse(&e(entries));
            assert_eq!(mix.main(), want, "{entries:?}: {mix}");
            assert!(mix.parts[0].1 > 0.6, "{entries:?}: {mix}");
        }
        // The insulator's hydrogen and oxygen are bound in it.
        let insulator = analyse_at(&e(cases[11].0), Some(1.85));
        assert_eq!(insulator.parts.len(), 1, "{insulator}");
        assert_eq!(Category::Water.appearance().clarity, 0.5);
        assert_eq!(Category::Steel.appearance().clarity, 0.0);
    }

    #[test]
    fn water_bound_or_free_by_density() {
        // Silica and water, 2:1 by mass: wet sand (1.7 g/cm³) or a hydrated
        // mineral (2.5 g/cm³).
        let (silica, water) = (2.0 / 3.0, 1.0 / 3.0);
        let card = e(&[
            ("14028", -silica * 0.4674),
            ("8016", -(silica * 0.5326 + water * 0.8881)),
            ("1001", -water * 0.1119),
        ]);
        let has_water = |m: &Mix| m.parts.iter().any(|p| p.0 == Category::Water);
        let wet = analyse_at(&card, Some(1.7));
        assert!(has_water(&wet), "{wet}");
        assert!(
            (wet.nominal_density - 1.67).abs() < 0.05,
            "{}",
            wet.nominal_density
        );
        let hydrated = analyse_at(&card, Some(2.5));
        assert_eq!(hydrated.parts, vec![(Category::Ceramic, 1.0)], "{hydrated}");
        assert!(has_water(&analyse(&card)), "free without a density");
    }

    #[test]
    fn concrete_and_heavy_concrete() {
        // Normal concrete (weight %, from the reference model's M10).
        let normal = analyse(&e(&[
            ("1001", -0.0039),
            ("6012", -0.059),
            ("8016", -0.489),
            ("11023", -0.0025),
            ("12024", -0.0057),
            ("13027", -0.0149),
            ("14028", -0.154),
            ("19039", -0.0041),
            ("20040", -0.252),
            ("26056", -0.0091),
        ]));
        assert_eq!(normal.main(), Category::Concrete, "{normal}");
        // Water of hydration and carbonate are concrete.
        assert_eq!(normal.parts.len(), 1, "{normal}");
        // Borated heavy concrete: 56% iron, all of it oxide.
        let heavy = analyse(&e(&[
            ("1001", -0.003),
            ("5010", -0.0042),
            ("8016", -0.319),
            ("14028", -0.022),
            ("20040", -0.066),
            ("26056", -0.558),
        ]));
        assert_eq!(heavy.main(), Category::Concrete, "{heavy}");
        assert!(heavy.iron_oxide > 0.5);
        let (n, h) = (normal.appearance(), heavy.appearance());
        assert!(h.albedo[0] < n.albedo[0], "heavy concrete is darker");
        assert_eq!(h.metalness, 0.0);
    }

    #[test]
    fn homogenised_steel_and_water() {
        // 60% steel, 40% water by volume.
        let (steel, water) = (0.6 * 7.9, 0.4 * 1.0);
        let total = steel + water;
        let mix = analyse(&e(&[
            ("26000", -steel / total),
            ("1001", -0.1119 * water / total),
            ("8016", -0.8881 * water / total),
        ]));
        assert_eq!(mix.main(), Category::Steel);
        assert!((mix.parts[0].1 - 0.6).abs() < 0.02, "{mix}");
        assert_eq!(mix.parts.len(), 2, "{mix}");
        assert_eq!(mix.parts[1].0, Category::Water);
        let a = mix.appearance();
        assert!(a.metalness > 0.7 && a.metalness < 1.0, "{a:?}");
    }

    #[test]
    fn specs() {
        let s: Spec = "copper roughness=0.2".parse().unwrap();
        assert_eq!(s.category, Some(Category::Copper));
        assert_eq!(s.roughness, Some(0.2));
        let s: Spec = "colour=#ff0000 metalness=0".parse().unwrap();
        assert_eq!(s.albedo, Some([1.0, 0.0, 0.0]));
        assert!("steel copper".parse::<Spec>().is_err());
        assert!("roughness=2".parse::<Spec>().is_err());
        assert!("glass".parse::<Spec>().is_err());
        let v = parse_specs("# looks\nm12 steel  # the vessel\n13 colour=#806040\n\n").unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].0, 12);
        assert_eq!(v[1].1.albedo.unwrap(), srgb_to_linear3([0x80, 0x60, 0x40]));
        assert!(parse_specs("x steel").is_err());
    }

    #[test]
    fn densities() {
        assert_eq!(grams_per_cm3(-7.9, 56.0), Some(7.9));
        // Water at 0.1003 atoms/b·cm is 1 g/cm³.
        let mix = analyse(&e(&[("1001", 2.0), ("8016", 1.0)]));
        let rho = grams_per_cm3(0.1003, mix.atomic_mass).unwrap();
        assert!((rho - 1.0).abs() < 0.01, "{rho}");
    }

    #[test]
    fn cell_and_group_masses() {
        let mcnp = "\
mass balance test
1 1 -2.0 -1 imp:n=1
2 1 -4.0  1 -2 imp:n=1
3 2  0.1003  2 -3 imp:n=1
4 0  3 imp:n=0

1 rpp 0 1 0 1 0 1
2 rpp 0 2 0 1 0 1
3 rpp 0 3 0 1 0 1

mode n
m1 26056 1.0
m2 1001 2.0 8016 1.0
";
        let m = Model::from_text(mcnp, &Default::default());
        assert!(m.diagnostics().is_empty(), "{:?}", m.diagnostics());
        let looks = Looks::new(&m, &[]).unwrap();
        let (m1, dm1) = cell_mass(&m, &looks, 1, 1.0, 0.1).unwrap();
        assert!((m1 - 2.0).abs() < 1e-12 && (dm1 - 0.2).abs() < 1e-12);
        assert!(cell_mass(&m, &looks, 4, 1.0, 0.1).is_none());

        let grid = llevant::VolumeGrid::new(m.extent(llevant::Scope::Problem, 32), 0.1);
        let vols = m.volumes(&grid, None).unwrap();
        let by_cell = vols.grouped(&m, llevant::GroupBy::Cell);
        let by_mat = vols.grouped(&m, llevant::GroupBy::Material);
        let mat1 = by_mat.iter().find(|g| g.number == Some(1)).unwrap();
        // Multi-density material needs cell breakdown for exact mass & sigma.
        assert!(group_mass(&m, &looks, llevant::GroupBy::Material, mat1, None).is_none());
        let (mm1, dmm1) =
            group_mass(&m, &looks, llevant::GroupBy::Material, mat1, Some(&by_cell)).unwrap();
        let (c1m, c1s) = cell_mass(
            &m,
            &looks,
            1,
            by_cell
                .iter()
                .find(|g| g.number == Some(1))
                .unwrap()
                .volume
                .estimate,
            by_cell
                .iter()
                .find(|g| g.number == Some(1))
                .unwrap()
                .volume
                .sigma,
        )
        .unwrap();
        let (c2m, c2s) = cell_mass(
            &m,
            &looks,
            2,
            by_cell
                .iter()
                .find(|g| g.number == Some(2))
                .unwrap()
                .volume
                .estimate,
            by_cell
                .iter()
                .find(|g| g.number == Some(2))
                .unwrap()
                .volume
                .sigma,
        )
        .unwrap();
        assert!((mm1 - (c1m + c2m)).abs() < 1e-9);
        assert!((dmm1 - (c1s * c1s + c2s * c2s).sqrt()).abs() < 1e-9);

        let mat2 = by_mat.iter().find(|g| g.number == Some(2)).unwrap();
        let (mm2, _) =
            group_mass(&m, &looks, llevant::GroupBy::Material, mat2, Some(&by_cell)).unwrap();
        assert!((mm2 - 1.0).abs() < 0.05, "{mm2}");

        for by in [
            llevant::GroupBy::Cell,
            llevant::GroupBy::Material,
            llevant::GroupBy::Universe,
            llevant::GroupBy::Label,
        ] {
            let gs = vols.grouped(&m, by);
            let want: Vec<_> = gs
                .iter()
                .map(|g| group_mass(&m, &looks, by, g, Some(&by_cell)))
                .collect();
            let got = group_masses(&m, &looks, by, &gs, Some(&by_cell));
            assert_eq!(got, want, "batch group_masses mismatch for {by:?}");
        }
    }
}
