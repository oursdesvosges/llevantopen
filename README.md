<p align="center">
  <img src="docs/img/logo.png" alt="" width="110">
</p>

<h1 align="center">llevant</h1>

<p align="center">
  <a href="https://github.com/Fusion4Energy/llevant/actions/workflows/ci.yml"><img src="https://github.com/Fusion4Energy/llevant/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/llevant/"><img src="https://img.shields.io/pypi/v/llevant" alt="PyPI"></a>
  <a href="https://crates.io/crates/llevant"><img src="https://img.shields.io/crates/v/llevant" alt="crates.io"></a>
  <a href="https://fusion4energy.github.io/llevant/"><img src="https://img.shields.io/badge/docs-fusion4energy.github.io%2Fllevant-blue" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="License: EUPL-1.2"></a>
</p>

Fast, exact geometry queries on MCNP and OpenMC XML models: an interactive viewer, a command line
and a Python module, on a Rust engine. llevant answers:

- where points are, and what lines cross;
- slices and 3D views;
- volumes, and the gaps and overlaps of a model;
- fingerprints that tell two versions of a model apart;
- tally results (meshtal, MCTAL, OpenMC HDF5 statepoint, per cell) drawn over the geometry.

It reads MCNP input (through [migjorn](https://crates.io/crates/migjorn)) and OpenMC XML
(`model.xml`, or a directory containing `geometry.xml` and `materials.xml`), with
surfaces and macrobodies, universes at any depth, rectilinear and hexagonal lattices,
and transforms.

**Documentation: <https://fusion4energy.github.io/llevant/>**: install, a ten-minute
tour, how-to guides and references.

![The viewer: a reactor core cut open in 3D](docs/img/viewer.png)

| Results over the geometry (mock pin powers) | A check finds the lid overlapping the cask |
|---|---|
| ![Pin powers drawn over the core in 3D and in a slice](docs/img/viewer-results.png) | ![The Check workspace listing two overlaps, hatched in the slice](docs/img/viewer-check.png) |

The models are in [`docs/examples`](docs/examples).

## Install

```sh
pip install llevant          # Python 3.9+: the module, the `llevant` command and the viewer
```

Windows and Linux (x86-64). The [releases](https://github.com/Fusion4Energy/llevant/releases)
also have the `llevant` and `llevant-viewer` programs to download, and
`cargo install llevant-cli --features gpu,viewer` builds them from source. Details,
and what your computer needs: [Install](https://fusion4energy.github.io/llevant/install/).

## Try it

[`docs/examples/cask.i`](docs/examples/cask.i) is a small model to try (a cask of fuel
pins on a concrete floor); [Getting started](https://fusion4energy.github.io/llevant/getting-started/)
walks through it.

```sh
llevant info   cask.i                          # sizes, extent, diagnostics
llevant locate cask.i 0 0 100  80 0 100        # cell path, material, density of each point
llevant slice  cask.i --plane z=100 --colour material -o slice.png
llevant render cask.i --hide 5 --clip 0,0,0,0,-1,0 --colour realistic -o view.png
llevant check  cask.i --h 2                    # exit status 1 if gaps or overlaps
llevant view   cask.i                          # the viewer (or: llevant-viewer cask.i)
```

With pip, `llevant` is `python -m llevant`. From Python:

```python
import llevant

m = llevant.load("cask.i")             # compiled once, cached on disk
m.locate(0, 0, 100)                    # <Label inside 4 > 30[0,0,0] > 41>
m.classify(points)                     # (N, 3) numpy array -> numpy columns
m.slice(z=100, colour="material")      # an image, shown inline in Jupyter
m.volumes(h=2).by("material")          # volumes with their uncertainty
m.check(h=2)                           # gaps and overlaps
```

## How fast

On a 425 k-cell fusion reactor model, on a 16-thread laptop
([`docs/benchmarks.md`](docs/benchmarks.md)):

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
| Volumes by cell, material, universe or tally bin, with an honest uncertainty | `volume` | `Model::volumes`, `volumes_to`, `instance_volumes` |
| Gaps and overlaps, with volume and location | `check` | `Volumes::problems` |
| Fingerprints of cells and universes; model diffs | `fingerprint` | `Model::fingerprints` |
| A universe (a filler) as a model of its own | `--universe` | `Model::universe` |
| Cell instances: each place a cell occupies, numbered | `locate` | `Model::instances` |
| Results over the geometry: values per cell, instance or mesh voxel (CSV, meshtal, MCTAL, D1SUNED cuvmsh, OpenMC HDF5 statepoint) | `slice`/`render --field` (`llevant.read_openmc_tally` in Python) | `llevant_scene::field` |
| An interactive viewer: 3D view and slice, picking, rules, cuts, fields, FMESH layout, card editing | `llevant-viewer`, `llevant view` | `crates/llevant-viewer` |
| Python: models kept in memory, numpy batch queries, inline images, the viewer | `python -m llevant` | `crates/llevant-py` |

What MCNP and OpenMC input is read, and what is not:
[Supported input](https://fusion4energy.github.io/llevant/reference/supported-input/).

## OpenMC XML

OpenMC models load natively in the CLI, Python, and viewer (no OpenMC runtime needed
for geometry), including [`docs/examples/openmc/fusion_hts_blanket.xml`](docs/examples/openmc/fusion_hts_blanket.xml)
and [`docs/examples/openmc/vver_hex_assembly.xml`](docs/examples/openmc/vver_hex_assembly.xml):

```sh
llevant info   docs/examples/openmc/fusion_hts_blanket.xml
llevant slice  docs/examples/openmc/vver_hex_assembly.xml --plane z=0 --colour realistic -o vver.png
llevant check  docs/examples/openmc/pwr_pin_lattice.xml --h 0.75
```

```python
import llevant

m = llevant.load("docs/examples/openmc/fusion_hts_blanket.xml")
m.bbox('label "Tungsten"')                      # searches OpenMC cell name="..."
f = llevant.read_openmc_tally(m, "statepoint.100.h5", score="flux")
m.slice(z=0, field=f)
```

## Library

```rust,no_run
use llevant::{Depth, Line, Model, Options, Status, DEFAULT_PRECISION};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load("model.mcnp", &Options::default(), None)?;
    let res = model.classify(&[[0.0, 0.0, 0.0], [120.0, -3.5, 40.0]], Depth::All);
    for &id in &res.ids {
        let label = &res.labels[id as usize];
        println!("{}", model.describe(label)); // e.g. "inside 12 > 304[1,-2,0] > 7"
        if label.status == Status::Inside {
            let cell = model.cell(label.cell().unwrap());
            println!("cell {} material {} density {}", cell.id, cell.material, cell.density);
        }
    }

    let line = Line::between([-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]);
    let tr = model.trace(&[line], Depth::All)?;
    for k in tr.segments(0) {
        let out = model.surface_id(tr.surface_out[k]); // MCNP surface (and facet), if any
        let label = &tr.labels[tr.ids[k] as usize];
        println!("{} .. {}: {} -> {out:?}", tr.t0[k], tr.t1[k], model.describe(label));
    }

    let fp = model.fingerprints(DEFAULT_PRECISION)?; // fp.cells[i].geometry, fp.units[u].content
    println!("{} cells fingerprinted", fp.cells.len());
    Ok(())
}
```

- **Batches.** Queries take batches and return flat arrays, ready for numpy: one
  label id per point or segment, with the labels interned.
- **Indices and numbers.** Labels refer to cells by index into `model.cells()`;
  `describe`, `cell_id` and `surface_id` give MCNP numbers.
- **Determinism.** Results are identical for any number of threads.
- **Errors.** Bad arguments are an `Error`, never a panic.

The API is on [docs.rs](https://docs.rs/llevant) (views, volumes and images included),
and [`docs/contracts.md`](docs/contracts.md) lists what each query guarantees.

## Layout

| Path | Purpose |
|---|---|
| `crates/llevant` | The engine (pure Rust, geometry only) |
| `crates/llevant-openmc` | Pure-Rust OpenMC XML translator (`model.xml`, `geometry.xml` + `materials.xml`) |
| `crates/llevant-scene` | What frontends build from a model: cell rules, hidden cells, material looks, field data and tally files, the input's cards, the cache of compiled models |
| `crates/llevant-render` | Images: colours, colour scales, field looks, physically based shading, labels, PNG |
| `crates/llevant-gpu` | The GPU renderer (wgpu) |
| `crates/llevant-viewer` | The `llevant-viewer` desktop app and its egui widgets |
| `crates/llevant-cli` | The `llevant` command |
| `crates/llevant-py` | The Python module `llevant` (PyO3, abi3 wheels) |
| `tools/bench` | `llevant-bench`: benchmarks, checks against brute force and of the GPU against the CPU |
| `tools/mcnp-oracle` | Checks llevant against MCNP itself |
| `tools/synthetic` | Generators of benchmark models |
| `docs/` | The documentation site's pages: user guides, and for developers the [architecture](docs/architecture.md), [contracts](docs/contracts.md), [decisions](docs/decisions/README.md), [benchmarks](docs/benchmarks.md) and [internals](docs/internals/engine.md) |

## Develop

[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) has the commands, the rules and how to
check against MCNP.

```sh
cargo test --release                       # unit, MCNP golden, property and CLI tests
cargo test --release -p llevant-gpu -p llevant-viewer   # GPU renderer and viewer (skip without an adapter)
cargo run --release -p llevant-bench       # benchmarks and checks: every command
```

Licence: [EUPL-1.2](LICENSE).
