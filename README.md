<p align="center">
  <img src="docs/img/logo.png" alt="llevant logo" width="110">
</p>

<h1 align="center">llevantopen</h1>

<p align="center">
  <strong>Fast, exact geometry queries, 3D viewer, QA, and tally visualization for OpenMC, MCNP, TRIPOLI-4, and DRAGON5</strong>
</p>

<p align="center">
  <a href="https://github.com/oursdesvosges/llevantopen/actions/workflows/ci.yml"><img src="https://github.com/oursdesvosges/llevantopen/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/oursdesvosges/llevantopen/actions/workflows/openmc-integration.yml"><img src="https://github.com/oursdesvosges/llevantopen/actions/workflows/openmc-integration.yml/badge.svg?branch=main" alt="OpenMC Integration"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="License: EUPL-1.2"></a>
  <a href="https://github.com/Fusion4Energy/llevant"><img src="https://img.shields.io/badge/upstream-Fusion4Energy%2Fllevant-lightgrey" alt="Upstream: Fusion4Energy/llevant"></a>
</p>

**llevantopen** extends [llevant](https://github.com/Fusion4Energy/llevant) with native pure-Rust loaders for **OpenMC XML** (`llevant-openmc`), **TRIPOLI-4** (`llevant-tripoli`), and **DRAGON5 `GEO:`** (`llevant-dragon`), plus **OpenMC HDF5 statepoint tally visualization**, lost-particle diagnostics, mass balances, and multi-code conversion — alongside full MCNP support (through [migjorn](https://crates.io/crates/migjorn)). From the interactive 3D viewer, the command line, Python, or Rust, it answers:

- **Multi-code geometry**: load OpenMC (`model.xml`, `geometry.xml` + `materials.xml`, directory, or Python objects via `from_openmc`), TRIPOLI-4 (`.t4`, `.tripoli`, `.t4g`), DRAGON5 (`GEO:` in `.x2m`/`.c2m`/`.dragon`), MCNP inputs, or convert between them (`llevant convert`, `llevant.to_mcnp`);
- **exact point & ray queries**: where points lie (`locate` / `classify`) and every segment a line crosses (`trace`) through nested universes, transforms, and rectilinear/hexagonal lattices;
- **lost-particle root-cause diagnosis**: inspect failing `(x, y, z)` coordinates or parse MCNP/OpenMC output logs (`llevant lost`, `diagnose_lost_particles`) with zoomed XY/XZ/YZ slices;
- **2D slices, `plots.xml` & GPU 3D views**: interactive `llevant-viewer`, `llevant plots` for OpenMC `<plots>`, and batch CPU/GPU renders with hiding, isolation, cutaways, and ghost shells;
- **QA reports, volumes & masses**: gaps and overlaps (`check --plots <DIR>` writing `report.json`, `report.md`, and PNGs) plus cell, material, and universe volumes & **masses (g)** (`volume --mass`, `Group.mass`) with honest uncertainties;
- **fingerprints & diffs**: tell two revisions of a model apart cell by cell (`fingerprint --diff`);
- **cell label queries**: select, isolate, hide, or bound OpenMC `name="..."`, TRIPOLI-4 volume names/comments, or MCNP `$` labels (`label "Tungsten"`);
- **tally & field overlays**: OpenMC HDF5 statepoints (`openmc_tallies`, `read_openmc_tally` with `energy`/`particle`/`nuclide`/`CellInstanceFilter`/`DistribcellFilter`, `export_openmc_tally_csv`), MCNP `meshtal`, `MCTAL`, D1SUNED `cuvmsh`, and CSV drawn over the geometry.

| OpenMC fusion HTS blanket (3D cutaway) | OpenMC VVER hexagonal lattice (slice at `z=0`) |
|---|---|
| ![OpenMC fusion HTS blanket 3D cutaway](docs/img/openmc-fusion-blanket.png) | ![OpenMC VVER hexagonal assembly slice](docs/img/openmc-vver-hex.png) |

Ready-to-run OpenMC, TRIPOLI-4, DRAGON5, and MCNP examples live in [`docs/examples`](docs/examples).

![The viewer: a reactor core cut open in 3D](docs/img/viewer.png)

| Results over the geometry (mock pin powers) | A check finds the lid overlapping the cask |
|---|---|
| ![Pin powers drawn over the core in 3D and in a slice](docs/img/viewer-results.png) | ![The Check workspace listing two overlaps, hatched in the slice](docs/img/viewer-check.png) |

## Multi-code support (OpenMC, TRIPOLI-4, DRAGON5)

No external neutronics runtime is needed to inspect, check, slice, render, convert, or view OpenMC, TRIPOLI-4, DRAGON5, or MCNP geometry:

| Ecosystem | Supported features |
|---|---|
| **OpenMC XML & HDF5** (`llevant-openmc`) | `model.xml`, `geometry.xml` (+ sibling `materials.xml`), directories, `from_openmc(obj)`, `plots.xml` rendering (`llevant plots`); all CSG surfaces, boundary conditions (`vacuum`, `reflective`, `white`, `periodic`), nested fills, Euler/`rotation`, 2D/3D `<lattice>` & `<hex_lattice>`, distributed materials, cell `name="..."` labels; `openmc_tallies`, `read_openmc_tally` (`cell`, `material`, `cellinstance`, `distribcell`, `mesh` + `energy`/`particle`/`nuclide`), `export_openmc_tally_csv` |
| **TRIPOLI-4** (`llevant-tripoli`) | `GEOMETRY`/`SURF`/`TYPE`/`VOLU` (`EQUA`, `COMBI`, `PLUS`, `MINUS`/`MOINS`, `INTE`, `UNION`, `VPLUS`, `VMOINS`, **`ECRASE`**, `FICTIVE`), `TRANSFORM MATRIX`, `COMPOSITION` (`DENSITY`, `POINT_WISE`, `NB_ATOM`, mass fractions), `GEOMCOMP`, `BOUNDARY_CONDITION` |
| **DRAGON5 CLE-2000** (`llevant-dragon`) | `GEO:` `CARCEL`, `TUBE`, `HEXCEL`, `CAR2D`, `CAR3D`, inline `::: SUB := GEO: ... ;`, `SPLITR`, `MIX` count validation |

Full compatibility details: [`docs/reference/supported-input.md`](docs/reference/supported-input.md).

## Install

Install the OpenMC-enabled fork from GitHub:

**Rust CLI & interactive viewer:**

```sh
cargo install --git https://github.com/oursdesvosges/llevantopen llevant-cli --features gpu,viewer
```

**Python module (with CLI and viewer included):**

```sh
git clone https://github.com/oursdesvosges/llevantopen.git
cd llevantopen/crates/llevant-py
pip install .            # builds the Rust extension (or: maturin develop --release)
pip install h5py         # optional: needed only for llevant.read_openmc_tally
```

**Prebuilt binaries & Python wheels:** download or trigger builds for Linux and Windows (`x86_64`) from the fork's [Release workflow](https://github.com/oursdesvosges/llevantopen/actions/workflows/release.yml).

> **Note:** Upstream PyPI (`pip install llevant`) and `Fusion4Energy/llevant` releases track upstream MCNP-only `llevant` and do not include `llevant-openmc` or `read_openmc_tally`.

## Try it (OpenMC & MCNP)

### OpenMC XML (CLI & Python)

Try the included OpenMC examples (`fusion_hts_blanket.xml`, `vver_hex_assembly.xml`, `pwr_pin_lattice.xml`):

```sh
llevant info   docs/examples/openmc/fusion_hts_blanket.xml
llevant locate docs/examples/openmc/fusion_hts_blanket.xml 0 62 0  0 95 0
llevant render docs/examples/openmc/fusion_hts_blanket.xml --wedge 0,0,0,0,0,1,-110,10 --colour realistic -o blanket.png
llevant slice  docs/examples/openmc/vver_hex_assembly.xml --plane z=0 --colour realistic -o vver.png
llevant check  docs/examples/openmc/pwr_pin_lattice.xml --h 0.75
llevant view   docs/examples/openmc/fusion_hts_blanket.xml
```

From Python (use `python -m llevant` for the CLI bundled in the wheel):

```python
import llevant

m = llevant.load("docs/examples/openmc/fusion_hts_blanket.xml")
m.locate(0, 62, 0)                      # <Label inside 2>
m.bbox('label "Tungsten"')              # searches OpenMC cell name="..."
m.check(12.0)                           # gaps and overlaps

# Overlay OpenMC HDF5 statepoint tallies on any slice or 3D render (requires h5py):
f = llevant.read_openmc_tally(m, "statepoint.100.h5", score="flux")
m.slice(z=0, field=f)
```

See [`docs/guide/python.md`](docs/guide/python.md) and [`docs/guide/cli.md`](docs/guide/cli.md) for the full API and CLI options.

### MCNP

[`docs/examples/cask.i`](docs/examples/cask.i) is a small MCNP cask model (fuel pins on a concrete floor):

```sh
llevant info   docs/examples/cask.i            # sizes, extent, diagnostics
llevant locate docs/examples/cask.i 0 0 100  80 0 100
llevant slice  docs/examples/cask.i --plane z=100 --colour material -o slice.png
llevant render docs/examples/cask.i --hide 5 --clip 0,0,0,0,-1,0 --colour realistic -o view.png
llevant check  docs/examples/cask.i --h 2      # exit status 1 if gaps or overlaps
llevant view   docs/examples/cask.i            # the viewer (or: llevant-viewer)
```

```python
import llevant

m = llevant.load("docs/examples/cask.i")  # compiled once, cached on disk
m.classify(points)                        # (N, 3) numpy array -> numpy columns
m.volumes(h=2).by("material")             # volumes with their uncertainty
```

## How fast

On a 425 k-cell fusion reactor model, on a 16-thread laptop ([`docs/benchmarks.md`](docs/benchmarks.md)):

| | |
|---|---|
| Load | 1.3 s from the cache (36 s to compile the first time) |
| Points | 5.4 million classified per second |
| Lines | 3.3·10⁵ lines of 100 cm traced per second |
| Slices | 110 ms for 1000 rows |
| Gaps and overlaps | the 36 × 37 × 35 m building checked on lines 5 cm apart in under 3 minutes |
| Viewer | 60 fps while orbiting (GPU) |

## What it does

| Query | Command | Library |
|---|---|---|
| Where points are: cell path, material, density; gaps, overlaps, on-surface | `locate` | `Model::classify` |
| What lines cross: exact segments, the surface at each end | `trace` (`--lines` for a file of them) | `Model::trace` |
| 2D slices with exact boundaries, cell and surface labels | `slice` | `Model::slice`, `llevant_render::labels` |
| 3D views: hidden cells, cutaways, universe levels, ghost shells | `render` (`--gpu`) | `Model::render`, `Model::first_hits`, `llevant-gpu` |
| Realistic images: materials sorted by composition, physically based shading | `render --colour realistic`, `materials` | `llevant_scene::materials`, `llevant_render::shading` |
| Volumes and masses by cell, material, universe or tally bin, with uncertainty | `volume --mass` | `Model::volumes`, `group_mass`, `mass_balance` |
| Gaps and overlaps, with volume, location, and defect slice bundles | `check --plots` | `Volumes::problems`, `check_report` |
| Lost-particle log/coordinate root-cause diagnosis & zoomed slices | `lost` | `Model::nearest_surface`, `diagnose_lost_particles` |
| Multi-code conversion (OpenMC / TRIPOLI-4 / DRAGON5 -> MCNP) | `convert` | `read_input_bytes`, `to_mcnp`, `from_tripoli`, `from_dragon` |
| OpenMC `<plots>` XML rendering directly to PNG | `plots` | `openmc_plots`, `render_openmc_plots` |
| Fingerprints of cells and universes; model diffs | `fingerprint` | `Model::fingerprints` |
| A universe (a filler) as a model of its own | `--universe` | `Model::universe` |
| Cell instances: each place a cell occupies, numbered | `locate` | `Model::instances`, `Model.cell_instances` |
| Results over the geometry: OpenMC HDF5 statepoint, meshtal, MCTAL, D1SUNED cuvmsh, CSV | `slice`/`render --field`, `read_openmc_tally` | `llevant_scene::field`, `openmc_tallies`, `export_openmc_tally_csv` |
| An interactive viewer: 3D view and slice, picking, rules, cuts, fields, FMESH layout, card editing | `llevant-viewer`, `llevant view` | `crates/llevant-viewer` |
| Python: OpenMC/MCNP/TRIPOLI-4/DRAGON5 in memory, numpy batch queries, QA reports, the viewer | `python -m llevant` | `crates/llevant-py` |

What OpenMC XML and MCNP input is read, and what is not: [`docs/reference/supported-input.md`](docs/reference/supported-input.md).

## Library

```rust,no_run
use llevant::{Depth, Line, Model, Options, Status, DEFAULT_PRECISION};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Loads OpenMC XML (`model.xml`, `geometry.xml` + `materials.xml`) or MCNP inputs identically.
    let model = Model::load("docs/examples/openmc/fusion_hts_blanket.xml", &Options::default(), None)?;
    let res = model.classify(&[[0.0, 0.0, 0.0], [0.0, 62.0, 0.0]], Depth::All);
    for &id in &res.ids {
        let label = &res.labels[id as usize];
        println!("{}", model.describe(label)); // e.g. "inside 12 > 304[1,-2,0] > 7"
        if label.status == Status::Inside {
            let cell = model.cell(label.cell().unwrap());
            println!("cell {} material {} density {}", cell.id, cell.material, cell.density);
        }
    }

    let line = Line::between([-250.0, 0.0, 0.0], [250.0, 0.0, 0.0]);
    let tr = model.trace(&[line], Depth::All)?;
    for k in tr.segments(0) {
        let out = model.surface_id(tr.surface_out[k]);
        let label = &tr.labels[tr.ids[k] as usize];
        println!("{} .. {}: {} -> {out:?}", tr.t0[k], tr.t1[k], model.describe(label));
    }

    let fp = model.fingerprints(DEFAULT_PRECISION)?;
    println!("{} cells fingerprinted", fp.cells.len());
    Ok(())
}
```

- **Batches.** Queries take batches and return flat arrays, ready for numpy: one label id per point or segment, with the labels interned.
- **Indices and numbers.** Labels refer to cells by index into `model.cells()`; `describe`, `cell_id`, and `surface_id` give cell/surface numbers.
- **Determinism.** Results are identical for any number of threads.
- **Errors.** Bad arguments are an `Error`, never a panic.

[`docs/contracts.md`](docs/contracts.md) lists what each query guarantees, and upstream Rust API docs are on [docs.rs](https://docs.rs/llevant).

## Layout

| Path | Purpose |
|---|---|
| `crates/llevant` | The engine (pure Rust, geometry only) |
| `crates/llevant-openmc` | Pure-Rust OpenMC XML translator (`model.xml`, `geometry.xml` + `materials.xml`, `<plots>`) |
| `crates/llevant-tripoli` | Pure-Rust TRIPOLI-4 deck translator (`SURF`, `TYPE`, `VOLU`, `ECRASE`, `COMPOSITION`, `GEOMCOMP`) |
| `crates/llevant-dragon` | Pure-Rust DRAGON5 CLE-2000 `GEO:` translator (`CARCEL`, `TUBE`, `HEXCEL`, `CAR2D`, `CAR3D`, `MIX` checks) |
| `crates/llevant-scene` | What frontends build from a model: cell rules, hidden cells, material looks and masses, field data and tally files, input cards, disk cache |
| `crates/llevant-render` | Images: colours, colour scales, field looks, physically based shading, labels, PNG |
| `crates/llevant-gpu` | The GPU renderer (`wgpu`) |
| `crates/llevant-viewer` | The `llevant-viewer` desktop app and its `egui` widgets |
| `crates/llevant-cli` | The `llevant` command (`convert`, `plots`, `lost`, `volume --mass`, `check --plots`, ...) |
| `crates/llevant-py` | The Python module `llevant` (PyO3, abi3 wheels, OpenMC HDF5 tally & multi-code helpers) |
| `tools/bench` | `llevant-bench`: benchmarks, brute-force checks, and GPU vs CPU comparisons |
| `tools/mcnp-oracle` | Checks `llevant` against MCNP itself |
| `tools/synthetic` | Generators of benchmark models |
| `docs/` | Documentation site sources: user guides, [architecture](docs/architecture.md), [contracts](docs/contracts.md), [decisions](docs/decisions/README.md), [benchmarks](docs/benchmarks.md), and [internals](docs/internals/engine.md) |

## Develop

[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) has the full setup, test rules, and oracle instructions.

```sh
cargo test --release                                 # unit, MCNP golden, property, OpenMC/TRIPOLI-4/DRAGON5, and CLI tests
cargo test --release -p llevant-openmc -p llevant-tripoli -p llevant-dragon
cargo test --release -p llevant --test openmc --test tripoli --test dragon
cargo test --release -p llevant-gpu -p llevant-viewer # GPU renderer and viewer (skip without an adapter)
pytest crates/llevant-py/tests                       # Python API, multi-code, & HDF5 statepoint tally tests
cargo run --release -p llevant-bench                 # benchmarks and checks
```

Licence: [EUPL-1.2](LICENSE).
