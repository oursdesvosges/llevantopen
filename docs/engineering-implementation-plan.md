# `llevantopen` Engineering Implementation Plan: Evolution to a Full-Spectrum, Maximum-Performance Neutronics Platform

**Document Status**: Architectural & Engineering Specification (`M1`)  
**Repository Baseline**: `oursdesvosges/llevantopen` (`main...fork/main` @ `9e506f7`, workspace `v0.1.3`, Rust MSRV `1.95`, `EUPL-1.2`)  
**Companion Artifacts**: `web_viewer.py`, `docs/architecture.md`, `docs/internals/engine.md`, `docs/internals/rendering.md`  

---

## Table of Contents

1. [Executive Summary & Scope Boundary](#1-executive-summary--scope-boundary)
2. [Verified Codebase & Interface Baseline (R2)](#2-verified-codebase--interface-baseline-r2)
3. [Cross-Ecosystem Feature Catalog & Baseline Cross-Reference (R1 + R2)](#3-cross-ecosystem-feature-catalog--baseline-cross-reference-r1--r2)
4. [Target System Architecture (R3)](#4-target-system-architecture-r3)
5. [Concrete Data Structures (R3)](#5-concrete-data-structures-r3)
6. [Maximum-Performance Engineering Design (R3)](#6-maximum-performance-engineering-design-r3)
7. [Target Interface Specifications (R2 + R3)](#7-target-interface-specifications-r2--r3)
8. [Prioritized Phased Implementation Roadmap (R3)](#8-prioritized-phased-implementation-roadmap-r3)

---

## 1. Executive Summary & Scope Boundary

### 1.1 Strategic Evolution of `llevantopen`

`llevantopen` (forked from `Fusion4Energy/llevant`) has established a production-grade, pure-Rust Constructive Solid Geometry (CSG) kernel capable of compiling 400k+ cell fusion/fission models in seconds, running deterministic `rayon`-parallel exact line tracing and point classification verified against MCNP6 (`tools/mcnp-oracle`), computing stochastic volumes and material mass balances with rigorous order-independent uncertainties, generating automated QA defect bundles (`llevant check --plots`, `llevant lost`), and rendering interactive 3D G-buffer scenes via `wgpu` compute shaders (`llevant-gpu`, `llevant-viewer`).

However, as documented in `docs/architecture.md` (*"Transport or tallies: frontends draw results, but llevant computes none"*), the baseline platform stops at geometry compilation, QA, and post-processing visualization. Meanwhile, production neutronics users across **MCNP**, **OpenMC**, **TRIPOLI-4**, **DRAGON5**, **Serpent 2**, **PHITS**, **Geant4**, **FLUKA**, **SCALE/Shift**, and **DAGMC/CAD** workflows suffer from acute daily friction:

1. **Fragmented Toolchains**: Analysts translate geometries repeatedly between CAD (STEP/BREP), DAGMC (`.h5m`), MCNP decks, OpenMC XML, TRIPOLI-4 `.t4`, and DRAGON5 `.x2m` scripts—losing metadata, introducing translation bugs, and paying text-roundtrip parsing overhead.
2. **Legacy Single-Node / Subprocess Bottlenecks**: Tools like `web_viewer.py` spawn a fresh CLI process per HTTP request; `migjorn` consumes ~2 GB peak RAM on 378 MB decks; `llevant-gpu` stalls 8–30 s compiling WGSL variants on cold start; and CPU `volumes_to` on 425k-cell models takes hours to converge tiny cells.
3. **Absence of Native Physics, Nuclear Data, and Accelerated Transport**: Without native continuous-energy/multigroup cross-section engines, event-based CPU-SIMD and `wgpu` GPU transport kernels, automated variance reduction (`CADIS`/`FW-CADIS`/`RMX`), and `CRAM` depletion/activation solvers, `llevantopen` remains an auxiliary pre-/post-processor rather than a unified neutronics platform.

This plan specifies the complete engineering blueprint to evolve `llevantopen` from a **CSG geometry/QA/visualization kernel** into a **full-spectrum, maximum-performance neutronics platform**—combining:
- A **native multi-code Geometry Intermediate Representation (`llevant-ir`)** that eliminates lossy MCNP-text round-tripping while preserving lossless card editing;
- **Hybrid CSG + Watertight DAGMC/Embree-style Surface Mesh + Tetrahedral Unstructured Mesh (UM)** tracking;
- **Maximum-performance CPU SIMD (`std::simd`/`std::arch` ray packets, SoA shapes, BVH4/8) and GPU (`wgpu` event-based transport, radix sort, shared-memory tally privatization, `PipelineCache`)** kernels;
- A **memory-safe Nuclear Data Engine (`llevant-nucdata`)** supporting ENDF-6, ACE, OpenMC HDF5, GNDS, Windowed Multipole (WMP), URR probability tables, and `NCrystal` FFI;
- **Monte Carlo + Hybrid Deterministic (`RandomRay` / `CMFD` / `CADIS` / `RMX`) + `CRAM` Depletion/R2S** solvers; and
- **Unified Modern Interfaces** across Rust, zero-copy Arrow/NumPy Python (`llevant-py`), CLI (`llevant-cli`), Desktop GPU Viewer (`llevant-viewer`), persistent-daemon-backed `web_viewer.py`, `llevant serve` daemon, `MCP` server (`rmcp`), and `LSP` IDE tooling.

### 1.2 Scope Boundary & Non-Regression Invariants

Evolution **must preserve zero regression** on every existing guarantee in `docs/contracts.md` and `docs/architecture.md`:
1. **CPU Exact Reference Invariant**: Exact `f64` CSG classification, line tracing, volume estimation, and brute-force oracle verification (`llevant::reference`, `tools/mcnp-oracle`) remain the immutable ground truth. Approximate GPU (`f32`) or delta-tracking fast paths must never corrupt deterministic QA checks.
2. **Thread-Count Determinism**: All existing queries and all new Monte Carlo / Random Ray / depletion solvers must produce bit-identical results across `1` and `N` CPU threads (via index-ordered aggregation, fixed-point `u128` summation, and counter-based Philox4x32-10 RNG).
3. **Backward Compatibility of Existing Crates**: The 12 existing workspace crates retain their public APIs (or clean deprecation shims) while new capabilities layer cleanly via new workspace crates and trait boundaries.
4. **Hard Plan-Phase Boundary**: This document is strictly an engineering specification. No tracked files in `repo` or `web_viewer.py` are modified during this deliverable.

### 1.3 Reconciliation with Prior Roadmaps

This plan explicitly consolidates and extends two existing planning documents without blind duplication:

| Source Document | Prioritized Items | How Reconciled in This Plan |
|---|---|---|
| `neutronics_user_demands_and_roadmap.md` | **P0** (`llevant-tripoli`, `llevant-dragon`, OpenMC XML hardening, `convert --verify`, `lost`), **P1** (`check --plots`, `volume --mass`, statepoint filter slicing, `plots`), **P2** (`llevant serve` / MCP), **Next-Horizon** (T4 lattices/`REPERE`, DR5 `SALT:`/`SECTOR`, DAGMC `.h5m`) | Audited against actual code in §2.5 (identifying verified caveats: `llevant-dragon` skipping `SPLITR`/`SPLITX`/`SPLITY`/`SPLITZ`/`SECT`/`TURN`, `llevant-tripoli` ignoring `boundary_surfs`, `llevant-openmc` assigning only the first distributed material). P0/P1 gaps are closed in **Phase 0**, and P2/Next-Horizon items are fully architected in **Phases 1–4**. |
| `docs/architecture.md` § Roadmap | Mesh voxel volumes, faster level-1 fillers, cheaper `volumes_to` convergence, faster sparse-world CPU `first_hits`, tangent-line tests, Viewer (`TR` numeric fields, instance/value hiding, faded pins, voxel outlines, `--universe` fields), browser WebGPU/WASM plotter, diff tracing on changed fingerprints | Directly integrated into **Phase 0** (quick wins & engine/viewer parity), **Phase 1** (SIMD kernels, `VolumeGrid` QMC/stratification, `first_hits` sparse stepping, shader caching, WebGPU/WASM + `web_viewer.py` daemon), and **Phase 2** (native Arrow/HDF5 viewer & instance/value hiding). |

---

## 2. Verified Codebase & Interface Baseline (R2)

All baseline facts below were verified directly against `repo` (HEAD `9e506f7` + 16 uncommitted working-tree files) and `web_viewer.py`.

### 2.1 Workspace Crates & Architectural Rules (`repo`)

Workspace `Cargo.toml` targets `version = "0.1.3"`, `edition = "2021"`, `rust-version = "1.95"`, `license = "EUPL-1.2"`, `profile.release` (`opt-level = 3`), `profile.dist` (`lto = "fat"`, `codegen-units = 1`).

| Crate Path | Crate Name | Default Member | Key Public APIs, Internals & Dependencies |
|---|---|---|---|
| `crates/llevant` | `llevant` | Yes | Pure-Rust CSG geometry engine (`migjorn = "=0.4.0"`, `rayon`, `blake3`, `bitcode`, `rustc-hash`, plus `llevant-openmc`, `llevant-tripoli`, `llevant-dragon`). Core APIs: `Model::{load, compile, compile_staged, complete, into_complete, from_text, from_migjorn, recompile, recompile_parsed, save, read, from_bytes}`, `read_input_bytes`, `translate_input_bytes`, `Options`, `Progress`, `GridMode`, `Model::{classify, classify_point, trace, first_hits, slice, render, pixel_hits, universe, fingerprints, fingerprint_diff, volumes, volumes_within, volumes_to, instance_volumes, cell_volumes, instance_boxes, placed_boxes, cells_kept, instances, skip, export, export_lean}`, `Draft::{begin, set}`, `reference::{classify, trace, traces_agree}`. |
| `crates/llevant-openmc` | `llevant-openmc` | Yes | Zero-dep OpenMC XML translator (`openmc_xml_to_mcnp`, `is_openmc_path/bytes`, `openmc_nuclide_to_zaid`, `parse_openmc_plots`, `read_openmc_plots`, `OpenMcPlot`). Handles `model.xml`, `geometry.xml`+`materials.xml`, CSG regions (`&`, `\|`, `~`), `LAT=1`/`LAT=2` lattices, rotations/translations, `<plots>`. |
| `crates/llevant-tripoli` | `llevant-tripoli` | Yes | Zero-dep TRIPOLI-4 translator (`tripoli_to_mcnp`, `is_tripoli_path/bytes`, `tripoli_nuclide_to_zaid`). Handles `SURF`, `TYPE`, `VOLU EQUA`/`COMBI`, `ECRASE` priority subtraction, `TRANSFORM MATRIX`, `COMPOSITION`, `GEOMCOMP`. |
| `crates/llevant-dragon` | `llevant-dragon` | Yes | Zero-dep DRAGON5 CLE-2000 `GEO:` translator (`dragon_to_mcnp`, `is_dragon_path/bytes`). Handles `CARCEL`, `TUBE`/`TUBEZ`, `HEXCEL`, `CAR2D`, `CAR3D`, inline `::: sub := GEO:`, strict `MIX` count checks. |
| `crates/llevant-scene` | `llevant-scene` | Yes | Frontend shared layer: `Store` (`memmap2` + `blake3` LRU cache, default 8 GB), `Source` (lossless `migjorn::Model` card editing), `ModelIndex` (`CellSet`), `Query` DSL, `Labels` (`memchr` `$` comments), `Hiding` (`Visibility`, `ProblemScope`), `materials` (`Category`, `Looks`, `Appearance`, `zaid_mass`, `cell_mass`, `group_mass`, `group_masses`), `Field` & `TallyFile` (`aci-tallies`, `aci-tallies-geom` for `meshtal`, `MCTAL`, D1SUNED `cuvmsh`, CSV). |
| `crates/llevant-render` | `llevant-render` | Yes | CPU 2D/3D rasterization: `Paint`, `ColourBy`, `Outlines`, `Scale` (`Ramp::{Viridis, Turbo, Inferno, Grey}`), `FieldLook`, `Overlay`, `legend`, Cook-Torrance GGX + Lambert PBR + analytic clear dielectric `shading::render_image`, `labels` (chamfer distance transform + `ab_glyph`), `ghost`, `png`. |
| `crates/llevant-cli` | `llevant-cli` | Yes | `llevant` binary & `run_cli` (`clap 4.5`, `mimalloc`, features `tallies`, `gpu`, `viewer`). Exposes 14 subcommands (`Cmd` enum in `crates/llevant-cli/src/lib.rs`). |
| `crates/llevant-gpu` | `llevant-gpu` | No | `wgpu 30.0.1` compute tracer (`bytemuck`, `pollster`, `static-dxc` on Windows MSVC). `GpuModel` (`MAX_LEVELS = 8`, `MAX_WORK = 1<<16`, `SLICE_WORK = 256`, `TAIL_WORK = 64*256`, `CAPACITY = 1<<17`, `STATE_WORDS = 20`, `HIT_WORDS = 9`, `GHOST_WORDS = 4`), `Job`, `Frame`, `HitBuffer`, `GpuVisibility`, `GpuModel::patch(&DraftPatch)`. |
| `crates/llevant-viewer` | `llevant-viewer` | No | `eframe 0.36` / `egui 0.36` / `egui-wgpu` desktop app + embeddable `View3d` & `SliceView` widgets, `rfd 0.17`, `mimalloc`. G-buffer WGSL `PassChain` (`reproject.wgsl`, `resolve.wgsl`, `colour.wgsl`, `field.wgsl`, `outline.wgsl`, `ssao.wgsl`, `ghost.wgsl`, `highlight.wgsl`, `marks.wgsl`), interactive `SelectTool`, `RulerTool`, `CutHandles`, `MeshHandles`, and 19 panel modules (`crates/llevant-viewer/src/panels/mod.rs`; 31 module files including `inspector/` submodules). |
| `crates/llevant-py` | `llevant-py` | No | PyO3 `0.30` (`abi3-py39`) + `numpy 0.30` cdylib `_llevant` + `mimalloc` (`local_dynamic_tls`), bundling CLI with `gpu`+`viewer`. Python helpers: `_openmc.py`, `_wrappers.py`, `_frames.py`. |
| `tools/bench` | `llevant-bench` | No | Benchmarks, brute-force verification, GPU-vs-CPU checks, work counters (`work-baseline.json`), `pprof` profiling. |
| `tools/mcnp-oracle` | `mcnp-oracle` | No | Validation oracle against `mcnp6` via `SDEF`/`PTRAC` + submodel cutter (`models/*.i`, `*.golden`). |

### 2.2 Baseline Interface Inventory

1. **CLI (`crates/llevant-cli/src/lib.rs`) — 14 Subcommands**:
   `info`, `bbox`, `locate`, `trace`, `slice`, `plots` (OpenMC `<plot>` PNG renderer), `render`, `materials`, `convert` (`--verify`), `lost` (`--point`, `--log`, `--dir`, `--radius`, `--plots`), `fingerprint` (`--diff`), `view` (`--state`, `--backend`), `volume` (`--h`, `--rel-sigma`, `--h-min`, `--abs-sigma`, `--max-probes`, `--by`, `--mass`), `check` (`--h`, `--plots`, `--max-plots`).
2. **Python (`crates/llevant-py/`)**:
   - Compiled PyO3 `_llevant`: `load`, `from_text`, `main`, `set_threads`, `threads`; `Model`, `Label`, `Surface`, `Segment`, `Diagnostic`, `Cell`, `Cells`, `Group` (`.mass`, `.mass_sigma`), `Volumes` (`to_polars`, `to_pandas`), `Fingerprints`, `Changes`, `Diff`, `Classification`, `Tracing`, `Hits`, `Image` (`_repr_png_`), `Field` (`cells`, `materials`, `instances`, `mesh`, `read`, `with_scale`, `with_opacity`, `with_cell_colours_under`). Releases GIL (`py.detach(...)`), returns read-only NumPy arrays.
   - Python wrappers: `_openmc.py` (`to_mcnp`, `from_openmc`, `openmc_tallies`, `read_openmc_tally` supporting `cell`, `material`, `cellinstance`, `distribcell`, `mesh` + `score`/`nuclide`/`energy`/`particle`/`error`, `export_openmc_tally_csv`, `diagnose_lost_particles`), `_wrappers.py` (`tripoli_to_mcnp`, `from_tripoli`, `dragon_to_mcnp`, `from_dragon`, `openmc_plots`, `render_openmc_plots`, `check_report`, `mass_balance`), `_frames.py`.
3. **Desktop GPU Viewer (`crates/llevant-viewer/`)**:
   - Strict UI-thread rule (`ui_pool()` & `background_pool()` in `tasks.rs`); staged loading; file-watch reload; `History` undo/redo of `Edit`s; `Source` card editing + live `Draft` coefficient dragging patched via `GpuModel::patch`; interactive `MeshHandles` generating FMESH cards via `aci-tallies-geom`.
   - **Gaps**: Does not expose `lost`, `check --plots`, `volume --mass`, native OpenMC `.h5` statepoints, instance/field-value hiding, faded obscured pins, numeric `TR` card fields, voxel boundary slice outlines, or `--universe` fields.
4. **Web Viewer (`web_viewer.py`)**:
   - Python stdlib `ThreadingHTTPServer` on port `8055` serving an inline HTML/JS SPA and **12 HTTP GET routes** (`/`, `/api/models`, `/api/info`, `/api/render`, `/api/slice`, `/api/locate`, `/api/trace`, `/api/volume`, `/api/check`, `/api/fingerprint`, `/api/bbox`, `/api/demo_field`).
   - **Critical Bottlenecks & Parity Gaps**:
     - Spawns a fresh `subprocess.run` of `target/release/llevant` on **every HTTP request** (even during interactive orbit dragging, forcing full model reload/mmap + cold GPU/CPU work each frame);
     - Disk cache in `tempfile.gettempdir() / "llevant_web_viewer"` keys only on MD5 of query params (stale if model file changes on disk);
     - `convert_openmc_statepoint_to_csv` (lines 159–278) only reads `"cell"` or `"material"` filters and `arr[:, 0, 0]`, completely missing `cellinstance`, `distribcell`, `mesh` filters, and `score`/`nuclide`/`energy`/`particle`/`error` UI selection available in `_openmc.py`;
     - Missing endpoints/UI for `lost`, `convert`, `plots`, `volume --mass`, `check --plots`, and WebGPU/WASM client rendering.

### 2.3 Verified Engine Internals, 16-File Dirty Baseline & Known Caveats

#### Existing Engine Memory Layout & Performance Baseline
- **Arena Model (`crates/llevant/src/model.rs`, `surface.rs`, `bvh.rs`, `grid.rs`)**:
  - `Shape` (`crates/llevant/src/surface.rs:165-174`) is an `enum` holding `Plane { n: Vec3, d: f64 }` (32 B payload), `Quadric(Quadric)` (104 B payload: `origin: [f64;3]` + `a: [f64;6]` + `b: [f64;3]` + `c: f64`), `Cone(Box<Cone>)` (8 B pointer + heap alloc), `Torus(Box<Torus>)` (8 B pointer + heap alloc). Total `size_of::<Shape>() == 112` bytes—wasting 72% of cache lines for planes!
  - `Bvh::Node` (`crates/llevant/src/bvh.rs:7-13`): 56-byte AoS (`Aabb` 48 B + `start: u32` + `count: u32`).
  - `Cell` (`crates/llevant/src/model.rs:116-138`): AoS with `code_start`, `code_len`, `node_start`, `bbox: Aabb`, `fill: Option<Fill>`, `unsupported: Option<Box<str>>`.
  - `Octree` (`crates/llevant/src/grid.rs:47-56`): Adaptive best-first octree with residual programs (`arena: Vec<u32>`, `progs: Vec<u32>`). On the 425k-cell reference model, octrees take 0.23 GB and residual programs add 0.72 GB + 4.5 s build time (`docs/internals/engine.md:133`).
  - `llevant-gpu` (`crates/llevant-gpu/src/pack.rs`, `hits.rs`, `lib.rs`): Packs pool words (`CELL_WORDS=13`, `UNIT_WORDS=20`, `BVH_NODE_WORDS=8`, `BVH_ITEM_WORDS=8`, `LATTICE_WORDS=42`, `PLANE_WORDS=5`, `SLACK_WORDS=1<<16`) into `wgpu::Buffer`s dynamically chunked by `device.limits()` (e.g., 128 MB on lavapipe/WebGPU per `pack.rs:341-346`); executes 64-ray workgroup slices (`STATE_WORDS=20`, `HIT_WORDS=9`, `GHOST_WORDS=4`) using `f32` recentred quadrics and `exit_bound` void-skipping.
- **Pre-Existing 16-File Dirty Optimization Baseline (`9e506f7` + `+649/-215`)**:
  Already in the working tree (untouched by this plan): `Aabb::ray_inv` & `ray_range_inv` (`bbox.rs`); stack-allocated traversal & child culling (`bvh.rs`, `grid.rs`, `skip.rs`); thread-local `Mutex<intern::Chunk>` reuse (`classify.rs`); `HashMap::entry` interning (`intern.rs`); `LatticeRay`/`prepare_ray`/`step_across` (`lattice.rs`); bounded `utf8_prefix` sniffing in `llevant-{openmc,dragon,tripoli}`; `O(C+M)` `group_masses` (`materials.rs`, `cli/volume.rs`, `py/results.rs`).
- **Observable Scaling Bottlenecks Remaining**:
  1. `migjorn::parse` peaks at ~2 GB RAM on the 378 MB reference deck; `Source` parses the deck a second time; multi-code loaders emit intermediate MCNP text strings and re-parse through `migjorn`.
  2. Scalar `f64` surface evaluation and 112 B `Shape` AoS without explicit SIMD (`std::arch`/`portable_simd` absent across workspace).
  3. Level-1 filler evaluations dominate CPU queries on multi-level models (`docs/architecture.md:259`).
  4. `Model::volumes_to` takes ~2 hours for whole-model 1% convergence (`h/64`) on 425k-cell models (`docs/architecture.md:260-262`).
  5. CPU `first_hits` is slower than full `trace` in sparse worlds (`sparse.i`) due to 16 cm $\times 2^k$ doubling windows (`docs/architecture.md:263-264`).
  6. `llevant-gpu` incurs 8–30 s WGSL shader compilation stalls per `(MAXL, CODE_STACK, BVH_STACK)` variant on cold runs (`docs/internals/rendering.md:337`) and ~1 ms single-ray locate-step tail latency on 425k-cell models (`rendering.md:511`).

#### Verified Code Caveats in `neutronics_user_demands_and_roadmap.md` Reconciliation
1. **`llevant-dragon` (`crates/llevant-dragon/src/lib.rs:422-426`)**: Tokens `SPLITX`, `SPLITY`, `SPLITZ`, `SPLITR`, `SECT`, `TURN` are **parsed and skipped** (as are boundary conditions `X-`..`HBC` at lines 410–421). Only `CARCEL` `MESHX`/`MESHY` quadrant subdivision is implemented—radial `SPLITR` meshing claimed in `neutronics_user_demands_and_roadmap.md` §1 is **not yet implemented**.
2. **`llevant-tripoli` (`crates/llevant-tripoli/src/lib.rs:1574`)**: `BOUNDARY_CONDITION`/`LIMITE` surfaces are parsed into `boundary_surfs`, then discarded (`let _ = &boundary_surfs;`) in favor of a synthesized complement graveyard over all non-fictive volumes; repeated lattices and `REPERE` transforms are absent.
3. **`llevant-openmc` (`crates/llevant-openmc/src/lib.rs:1106-1117`)**: Distributed materials (multiple space-separated material IDs on one `<cell>`) record the full list only in a `$` comment and assign **only the first material ID** to the compiled cell (`first = parsed.iter().flatten().copied().next()`), because MCNP cell cards cannot express per-instance material arrays directly.

---

## 3. Cross-Ecosystem Feature Catalog & Baseline Cross-Reference (R1 + R2)

Every capability surveyed across all **12 neutronics & tooling ecosystems** (92 items total) is cataloged below with:
- **Role Tag(s)**: `[User]` (physics/engineering analyst), `[Developer]` (solvers/HPC/kernels), `[Interface]` (APIs/CLI/GUIs/protocols)
- **Concrete Source Citation**: Verifiable repository URL, module path, or official manual/card
- **Baseline Status**: `[Implemented]` (exact crate/symbol), `[Partial / Extension]` (exact path + verified gap), or `[Net-New]`
- **Priority Tier**: `P0` (immediate parity/debt closure), `P1` (near-term core speed & UX), `P2` (production CAD/tallies/transport), `P3` (full physics, VR, depletion, cluster, MCP/LSP)

### 3.1 Role Summary View (Quick Navigation)

- **`[User]` Workflows**: Pre-flight geometry/overlap/lost-particle debugging (`MCNP-07`, `FLK-03`, `PH-07`, `GH-06`), stochastic volumes & mass balance (`OMC-09`), source verification (`GH-02`), comprehensive tallies (`MCNP-01/02/05`, `PH-05`, `FLK-04`, `G4-06`), automated VR (`MCNP-04`, `T4-01`, `SRP-07`, `SCL-01`), CAD/DAGMC/tet/voxel geometry (`MCNP-06`, `SRP-04`, `PH-03`, `FLK-06`, `DAG-01..06`), depletion/R2S/UQ (`OMC-05`, `SRP-03`, `PH-02`, `SCL-05`, `ND-06/07`), sensitivity/perturbation (`MCNP-03`, `T4-03/04`).
- **`[Developer]` Kernels**: Hybrid Woodcock/surface tracking (`SRP-01`), `f32`-BVH + `f64`-watertight mesh (`DAG-02/03`), O(1) tet-face tracking (`PH-03`), SoA/SIMD ray & BVH kernels (`G4-04`, `GH-05`), event-based GPU transport + radix sort + tally privatization (`OMC-02`, `G4-05`, `SCL-02`), logarithmic/hashed XS + WMP + `NCrystal` (`OMC-04`, `SRP-02`, `ND-04/05`), counter-based RNG, `CRAM` (`SRP-03`), `CMFD`/`RandomRay` (`OMC-03/08`), `CDD` (`SRP-08`), `GANLIB` IR (`DR5-01`).
- **`[Interface]` Surfaces**: Rust core + PyO3/`openmc.lib` live control (`OMC-01`, `GH-08`), HDF5/Arrow/Parquet (`MCNP-05/08`, `OMC-10`, `GH-08`), CLI (`PH-01`, `SCL-05`, `ND-03`), Desktop GPU viewer (`FLK-02`, `PH-07`, `SCL-04`), Web Studio (`web_viewer.py`, `GH-08`), Persistent Daemon / MCP / LSP (`T4-07`, `GH-07`, `GH-09`).

### 3.2 Complete Ecosystem Feature Matrix (All 12 Ecosystems, 92 Items)

#### Ecosystem 1: MCNP (MCNP6.2 / MCNP6.3)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `MCNP-01` | **FMESH & Turbotally Mesh Tallies** | `[User]`, `[Developer]`, `[Interface]` | LANL MCNP6.3 Theory & User Manual (`turbotally`, `FMESH`, HDF5+XDMF) | Geometry-independent 3D flux/dose/heating mesh scoring & interactive mesh card editing | `[Partial / Extension]` — Reads/overlays `meshtal` & creates FMESH cards (`llevant_scene::field`, `aci-tallies-geom`, `crates/llevant-viewer/src/tools/mesh.rs`); **computes 0 tallies** & lacks HDF5/XDMF output | `P2` |
| `MCNP-02` | **F4 / F5 / F6 / F8 Estimator Suite** | `[User]` | LANL MCNP6.3 User Manual (`F4`, `F5`, `F6`, `F8` cards) | Volume flux (F4), point/ring detector (F5), kerma heating (F6), pulse-height spectroscopy (F8) | `[Partial / Extension]` — Reads `MCTAL` F4/F6/F7/F8 (`llevant_scene::field::TallyFile`); **no runtime estimators**; surface/F5 tallies unsupported | `P3` |
| `MCNP-03` | **PERT, KPERT & KSEN Sensitivity/Perturbation** | `[User]`, `[Developer]` | LANL MCNP6.3 User Manual (`PERT`, `KPERT`, `KSEN` cards; IFP adjoint) | Perturbation $\Delta R, \Delta k_{\text{eff}}$ and continuous-energy $k_{\text{eff}}$ sensitivity coefficients without re-running MC | `[Net-New]` — Absent | `P3` |
| `MCNP-04` | **Weight Windows (`WWG`/`WWE`/`WWP`)** | `[User]`, `[Developer]` | LANL MCNP6.3 User Manual (`WWG`, `WWE`, `WWP`, `WWINP`) | Deep-penetration variance reduction via space-energy splitting/roulette | `[Net-New]` — Reads `IMP:N=0` solely for graveyard/problem box (`llevant::Model::problem_box`) | `P3` |
| `MCNP-05` | **Parallel HDF5 PTRAC** | `[User]`, `[Interface]` | LANL MCNP6.3 User Manual (`PTRAC` card; OSTI HDF5 PTRAC docs) | Event-by-event phase-space track recording & replay for safeguards/debugging | `[Net-New]` — Absent (only `tools/mcnp-oracle` consumes external `mcnp6` PTRAC) | `P3` |
| `MCNP-06` | **Unstructured Mesh (UM) Transport & HDF5 I/O** | `[User]`, `[Developer]`, `[Interface]` | LANL MCNP6.3 User Manual (UM / Abaqus / Cubit / HDF5 UM) | Direct tracking & scoring on FEA tet/hex/wedge meshes inside CSG universes | `[Net-New]` — Absent | `P2` |
| `MCNP-07` | **Interactive Plotter (`mcnp6 ip` / `mcnp6.qt ip`)** | `[User]`, `[Interface]` | LANL MCNP6.3 User Manual (`ip` / `mcplot>`) | Pre-run 2D/3D CSG slice inspection, surface labels, `LEVEL n`, tally overlays | `[Implemented]` — Exceeded by `llevant-viewer` (`View3d`, `SliceView`), `llevant slice`/`render`, `llevant_render::labels` | `P0` |
| `MCNP-08` | **Hybrid MPI+OpenMP & HDF5 `RUNTPE` Restart** | `[Developer]`, `[Interface]` | LANL MCNP6.3 User Manual (`tasks`, MPI, HDF5 `RUNTPE`) | Multi-node cluster scaling & versioned portable checkpoint/restart | `[Partial / Extension]` — Single-node `rayon` parallelism + `bitcode` model cache (`llevant::saved`, `llevant_scene::store`); no MPI or transport checkpoint | `P3` |
| `MCNP-09` | **`TALLYX` Custom Scoring Hook** | `[Developer]`, `[Interface]` | LANL MCNP6.3 User Manual (`TALLYX` subroutine) | Custom user scoring logic during transport without core fork | `[Net-New]` — Plan replaces Fortran recompilation with safe Rust trait / WASM / Python expression filter DSL | `P3` |
| `MCNP-10` | **`MCNPTools` & Legacy Input/CAD Converters** | `[Interface]`, `[User]` | LANL `MCNPTools` (GitHub); `mcnp2cad`; `McCad`; `openmc_mcnp_adapter` | Unified output parsing (`mctal`/`ptrac`/`meshtal`), STEP->CSG, MCNP->OpenMC migration | `[Partial / Extension]` — `aci-tallies` parses `meshtal`/`mctal`; `llevant convert` is one-way (OpenMC/T4/D5 -> MCNP); no MCNP -> OpenMC/IR export | `P1` |

#### Ecosystem 2: OpenMC (`openmc-dev/openmc`)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `OMC-01` | **Dual Core + Python API + `openmc.lib` C-API** | `[Developer]`, `[Interface]` | `https://github.com/openmc-dev/openmc` (`src/`, `openmc/`, `openmc/lib/__init__.py`) | Ergonomic Python modeling + live in-memory $T/\rho$/tally control without disk XML round-trips | `[Partial / Extension]` — `llevant-py` (`_llevant`) provides NumPy batch geometry queries & `from_openmc`, but lacks live mutable material/$T$/tally simulation control | `P1` |
| `OMC-02` | **Event-Based Transport & GPU Target Offload** | `[Developer]` | `openmc-dev/openmc` (`src/simulation.cpp`, `src/particle.cpp`) | Eliminates GPU/SIMD control-flow divergence via flat event queues & active-particle sorting | `[Net-New]` — `llevant-gpu` implements sliced WGSL camera first-hits only; no transport event queues | `P3` |
| `OMC-03` | **Coarse Mesh Finite Difference (CMFD)** | `[Developer]`, `[User]` | `openmc-dev/openmc` (`src/cmfd_solver.cpp`, `openmc.cmfd`) | Accelerates fission source convergence in high-dominance-ratio cores | `[Net-New]` — Absent | `P3` |
| `OMC-04` | **Windowed Multipole (WMP) Doppler** | `[Developer]`, `[User]` | `openmc-dev/openmc` (`src/wmp.cpp`, `include/openmc/wmp.h`) | On-the-fly continuous-energy resonance XS at arbitrary $T(\mathbf{r})$ without multi-T RAM bloat | `[Net-New]` — Absent | `P3` |
| `OMC-05` | **Integrated Depletion (`openmc.deplete`)** | `[User]`, `[Developer]` | `openmc-dev/openmc` (`openmc/deplete/`, CRAM, predictor-corrector) | Coupled transport + Bateman transmutation/decay + decay photon sources | `[Net-New]` — Absent | `P3` |
| `OMC-06` | **Direct DAGMC CAD Transport (`OPENMC_USE_DAGMC`)** | `[User]`, `[Developer]` | `openmc-dev/openmc` (`src/dagmc.cpp`) | Direct particle tracking on faceted `.h5m` CAD models alongside CSG | `[Net-New]` — Absent (Next-Horizon #4 in `neutronics_user_demands_and_roadmap.md`) | `P2` |
| `OMC-07` | **Multi-Group XS Generation (`openmc.mgxs`)** | `[User]`, `[Interface]` | `openmc-dev/openmc` (`openmc/mgxs/`) | Automated few-group XS, $P_n$ scattering, $\chi$ collapse into DataFrames/HDF5 | `[Net-New]` — Absent | `P3` |
| `OMC-08` | **Random Ray Solver (`src/random_ray`)** | `[Developer]`, `[User]` | `openmc-dev/openmc` (`src/random_ray/`, `convert_to_random_ray()`) | Fast multi-group SoA ray-segment solver giving low-variance full-domain flux maps without VR | `[Net-New]` — Absent | `P3` |
| `OMC-09` | **Stochastic Volume Calculation** | `[User]`, `[Interface]` | `openmc-dev/openmc` (`openmc/volume.py`, `openmc_calculate_volumes`) | Stochastic volumes & uncertainties per cell/material/universe/instance | `[Implemented]` — Exceeded by `llevant::Model::{volumes, volumes_to, cell_volumes, instance_volumes}` + `llevant_scene::materials::group_masses`; needs faster `volumes_to` convergence (`P1`) | `P1` |
| `OMC-10` | **HDF5 `StatePoint` & Geometry/Mesh Plotting** | `[User]`, `[Interface]` | `openmc-dev/openmc` (`openmc/statepoint.py`, `openmc-dev/openmc-plotter`) | Self-describing HDF5 tally/source storage + interactive Qt geometry/tally plotter | `[Partial / Extension]` — Python `_openmc.py` reads `statepoint*.h5` (`h5py`) & `llevant plots` renders `plots.xml`; **Rust CLI/Viewer lack native HDF5 statepoints**, and `web_viewer.py` lacks `cellinstance`/`distribcell`/`mesh` & `score`/`nuclide`/`energy`/`particle` UI | `P0` |

#### Ecosystem 3: TRIPOLI-4 (CEA)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `T4-01` | **`INIPOND` Automated Importance & Exponential Transform** | `[User]`, `[Developer]` | CEA TRIPOLI-4 User Manual (`INIPOND`) | Automated space-energy-angle-time importance maps + path stretching toward attractors | `[Net-New]` — Absent | `P3` |
| `T4-02` | **`GREEN_FUNCTION` Response Kernel** | `[User]`, `[Developer]` | CEA TRIPOLI-4 User Manual (`GREEN_FUNCTION`) | Pre-computed source-to-detector response kernels for rapid source-loading sweeps | `[Net-New]` — Absent | `P3` |
| `T4-03` | **Iterated Fission Probability (`IFP`) Adjoint & Kinetics** | `[User]`, `[Developer]` | CEA TRIPOLI-4 User Manual (`IFP`); EPJ-N papers | Continuous-energy $\beta_{\text{eff}}$, $\Lambda$, and $k_{\text{eff}}$ sensitivities | `[Net-New]` — Absent | `P3` |
| `T4-04` | **`PERTURB` Sensitivity & Perturbation Suite** | `[User]`, `[Developer]` | CEA TRIPOLI-4 User Manual (`PERTURB`: Correlated, Taylor, GPT) | Single-run material/density/XS perturbations & reaction-rate ratio sensitivities | `[Net-New]` — Absent | `P3` |
| `T4-05` | **Hybrid Deterministic-MC Coupling** | `[Developer]`, `[User]` | CEA TRIPOLI-4 / APOLLO3 / IDT Coupling Docs | Imports deterministic adjoint fluxes to drive importance meshes | `[Net-New]` — Absent | `P3` |
| `T4-06` | **Native CERN `ROOT` Geometry Integration** | `[Interface]`, `[Developer]` | CEA TRIPOLI-4 User Manual (`ROOT` interface) | Direct tracking on `.root` TGeo detector geometries without translation | `[Net-New]` — Future pluggable `GeometryKernel` trait target (via GDML/ROOT bridge) | `P3` |
| `T4-07` | **Socket-Based `PARALLEL` Execution** | `[Developer]`, `[Interface]` | CEA TRIPOLI-4 User Manual (`PARALLEL` BSD socket worker pool) | Heterogeneous LAN/cluster dynamic worker coordination without strict MPI | `[Net-New]` — Architectural inspiration for `llevant serve` distributed/worker RPC | `P1` |
| `T4-08` | **TRIPOLI-4 CSG (`ECRASE`), Lattices, `REPERE` & Multiphysics** | `[User]`, `[Developer]`, `[Interface]` | CEA TRIPOLI-4 User Manual (`VOLU`, `ECRASE`, `REPERE`, DARWIN/FIFRELIN) | Full TRIPOLI-4 geometry QA (`ECRASE`, boundary conditions, lattices, `REPERE`) & tally/burnup coupling | `[Partial / Extension]` — `crates/llevant-tripoli/src/lib.rs` parses `SURF`/`TYPE`/`VOLU`/`ECRASE`/`TRANSFORM`/`COMPOSITION`/`GEOMCOMP`, **but ignores `boundary_surfs` (line 1574) and lacks repeated lattices, `REPERE`, and native IR** | `P0` |

#### Ecosystem 4: DRAGON5 (`GANLIB` / `DRAGON5` — Polytechnique Montréal)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `DR5-01` | **`GANLIB` Kernel & `LCM` / `XSM` Hierarchical Objects** | `[Developer]`, `[Interface]` | Polytechnique Montréal `Ganlib5 Developer's Guide` (`merlin.polymtl.ca/version5.htm`; `Ahmed-Naceur/DRAGON5`) | Decoupled module pipeline exchanging typed hierarchical trees serialized cleanly to disk | `[Partial / Extension]` — Inspired `llevant-ir` hierarchical arena + `rkyv`/`bitcode` `Store` decoupling loaders from solvers | `P1` |
| `DR5-02` | **`CLE-2000` Procedural Scripting** | `[User]`, `[Interface]` | DRAGON5 User's Guide (`.x2m` scripts) | Parameterized lattice/burnup scripting in input decks | `[Partial / Extension]` — `llevant-dragon` parses inline `::: sub := GEO:` and `GEO:` tokens, but skips CLE-2000 control-flow evaluation | `P2` |
| `DR5-03` | **`LIB:` Microscopic XS & Interpolation** | `[User]`, `[Developer]` | DRAGON5 User Manual (`LIB:`, DRAGLIB/WIMSD/APOLIB) | Multigroup microscopic XS import & $T$/dilution interpolation | `[Net-New]` — Absent | `P3` |
| `DR5-04` | **`GEO:`, `EXCELT:` & `NXT:` Generalized Tracking** | `[User]`, `[Developer]` | DRAGON5 `NXT Module Theory Guide` (`GEO:`, `CARCEL`, `SPLITR`, `SECTOR`, `SALT:`) | Pin/hex/cluster lattice geometry + cached cyclic integration tracks | `[Partial / Extension]` — `crates/llevant-dragon/src/lib.rs` supports `CARCEL`/`TUBE`/`HEXCEL`/`CAR2D`/`CAR3D` & `MIX` checks, **but skips `SPLITR`/`SPLITX`/`SPLITY`/`SPLITZ`/`SECT`/`TURN` (lines 422–426), `X-/+..HBC` (410–421), `SALT:`, and cluster cells** | `P0` |
| `DR5-05` | **`USS:` & `SHI:` Resonance Self-Shielding** | `[Developer]`, `[User]` | DRAGON5 User Manual (`USS:`, `SHI:`) | Subgroup / probability-table resonance self-shielding in heterogeneous lattices | `[Net-New]` — Absent | `P3` |
| `DR5-06` | **`ASM:` & `FLU:` Deterministic Solvers (CP/MOC/$S_N$)** | `[Developer]` | DRAGON5 User Manual (`ASM:`, `FLU:`) | Collision probability / MOC / $S_N$ transport with $B_1$/$P_1$ leakage & DSA | `[Net-New]` — Plan targets `RandomRay`/MOC (`OMC-08`/`GH-05`) | `P3` |
| `DR5-07` | **`EDI:`, `EVO:`, `CPO:` / `MPO:` Homogenization & `MULTICOMPO`** | `[User]`, `[Interface]` | DRAGON5 User Manual (`EDI:`, `EVO:`, `MPO:`); `terrapower/dragon-armi-plugin` | Few-group homogenization, burnup, and multi-parameter reactor state tables | `[Net-New]` — Absent | `P3` |

#### Ecosystem 5: Serpent 2 (VTT Technical Research Centre of Finland)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `SRP-01` | **Woodcock Delta-Tracking + Surface Hybrid** | `[Developer]` | VTT Serpent 2 User Manual (Transport physics; OECD-NEA docs) | Bypasses dense pin/CAD surface ray-casts via majorant $\Sigma_{\text{maj}}$ with automatic surface fallback | `[Net-New]` — Essential for heterogeneous/DAGMC transport speed | `P3` |
| `SRP-02` | **Unionized Energy Grid & `opti 1..4` Modes** | `[Developer]` | VTT Serpent 2 User Manual (`set opti`) | Explicit RAM-vs-speed trade-off from compact fractional grids to pre-computed macroscopic $\Sigma_t$ | `[Net-New]` — Adopted as `XsOptiTier` in `llevant-nucdata` | `P3` |
| `SRP-03` | **CRAM Depletion & Bateman Solver** | `[Developer]`, `[User]` | VTT Serpent 2 User Manual; Pusa & Leppänen VTT CRAM papers | Stiff 1000+ nuclide transmutation/decay in single sparse linear solve (CRAM-14/16) | `[Net-New]` — Core of `llevant-deplete` | `P3` |
| `SRP-04` | **Native CAD/STL & Tet-Mesh Universes** | `[User]`, `[Developer]` | VTT Serpent 2 User Manual (STL & tet-mesh universes) | Embeds engineering STL/tet meshes directly inside CSG universes with per-element materials | `[Net-New]` — Core of `llevant-cad` | `P2` |
| `SRP-05` | **Multiphysics Coupling Interface (`ifc`)** | `[Interface]`, `[Developer]` | VTT Serpent 2 User Manual (`ifc` card; VTT Kraken papers) | Exchanges 3D $T(\mathbf{r}), \rho(\mathbf{r})$ fields (point-cloud/mesh/tet) + TMS rejection sampling | `[Net-New]` — Planned in `llevant-py` / Arrow live coupling | `P3` |
| `SRP-06` | **Automated Group Constant Generation** | `[User]`, `[Interface]` | VTT Serpent 2 User Manual (`set gcu`, `b1`, ADF/CDF/JDF) | Generates homogenized few-group constants, $B_1$ leakage, kinetics & discontinuity factors | `[Net-New]` — Absent | `P3` |
| `SRP-07` | **Automated Weight Windows (`wwd` + `rmx`)** | `[User]`, `[Developer]` | VTT Serpent 2 User Manual (`wwd`, `set rmx` response matrix) | Built-in response-matrix forward/adjoint importance solver for automated 3D weight windows | `[Net-New]` — Core of `llevant-vr` | `P3` |
| `SRP-08` | **Collision-Based Domain Decomposition (`CDD`)** | `[Developer]` | VTT Serpent 2 HPC / BEAVRS full-core burnup papers | Partitions 100k+ depletion-zone XS/inventories across MPI ranks via async particle migration | `[Net-New]` — Phase 4 HPC scaling architecture | `P3` |
| `SRP-09` | **Built-in `-plot` & Decay Photon Activation Sources** | `[User]`, `[Interface]` | VTT Serpent 2 User Manual (`-plot`, decay gamma sources) | Fast CLI slice plotting (`-plot`) and delayed $\gamma$ transport from activation inventories | `[Partial / Extension]` — Geometry `-plot` is `[Implemented]` (`llevant slice`, `llevant plots`); decay $\gamma$ transport is `[Net-New]` | `P3` |

#### Ecosystem 6: PHITS (JAEA)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `PH-01` | **`icntl` Multi-Mode CLI & `dumpall` Replay (`icntl=12`)** | `[User]`, `[Interface]` | JAEA PHITS User Manual (`icntl`, `dumpall = 1`) | Single binary for transport/check/render + instantaneous re-tallying from phase-space dumps | `[Partial / Extension]` — Geometry check/render modes `[Implemented]` in `llevant-cli`; binary phase-space `dump`/`replay` is `[Net-New]` | `P3` |
| `PH-02` | **`DCHAIN-PHITS` Activation & Decay Coupling** | `[User]`, `[Interface]` | JAEA PHITS User Manual (`[t-dchain]`, JENDL-5) | Coupled 1968-group flux + yield -> activation inventory, decay heat, $\gamma$ sources | `[Net-New]` — Covered by `llevant-deplete` + R2S pipeline | `P3` |
| `PH-03` | **Tetrahedral Mesh Geometry & Tallies (`mesh=tet`)** | `[User]`, `[Developer]` | JAEA PHITS User Manual (`mesh=tet`) | O(1) face-neighbor stepping & scoring on tetrahedral meshes | `[Net-New]` — Covered by `TetMesh` in `llevant-cad` | `P2` |
| `PH-04` | **Event Generator Mode (`E-mode`) + `JAMQMD`/`INCL4.6`** | `[User]`, `[Developer]` | JAEA PHITS User Manual (`e-mode=2`, `JAMQMD`, `INCL4.6`) | Event-by-event recoil kinematics (<20 MeV) & heavy-ion/spallation cascades | `[Net-New]` — Extended physics module (`P3`/post-neutron-photon) | `P3` |
| `PH-05` | **Modular Declarative Tallies (`[t-track]`, `[t-deposit]`, `[t-cross]`)** | `[User]` | JAEA PHITS User Manual (Tally sections) | Declarative tallies with built-in dose/LET multipliers across `reg`/`xyz`/`r-z`/`tet` | `[Net-New]` — Adopted into `llevant-tally` filter/multiplier AST | `P3` |
| `PH-06` | **`ANGEL` Integrated Plotter & `gshow`/`rshow`** | `[User]`, `[Interface]` | JAEA PHITS User Manual (`ANGEL`, `[t-gshow]`, `[t-rshow]`) | Declarative publication-ready 1D/2D/slice plots directly from tally outputs | `[Partial / Extension]` — `llevant-render` renders 2D slice+tally PNGs with `legend`; 1D spectra/batch plot templates `[Net-New]` | `P2` |
| `PH-07` | **`PHIG-3D` Interactive 3D & `[T-4Dtrack]` Viewer** | `[User]`, `[Interface]` | JAEA PHITS Docs (`PHIG-3D` / `Gxsview`; `[T-4Dtrack]`) | Interactive 3D geometry/overlap inspection + 4D particle trajectory animation | `[Partial / Extension]` — Interactive 3D/overlap viewer `[Implemented]` in `llevant-viewer`; 4D trajectory animation `[Net-New]` | `P2` |
| `PH-08` | **`[T-Userdefined]` & Hybrid MPI+OpenMP** | `[Developer]` | JAEA PHITS User Manual (`usrtally.f`, `MPIFLGS`) | User-defined custom scoring hooks & hybrid cluster scaling | `[Net-New]` — Replaced with safe Rust trait / Python-Arrow / WASM hook | `P3` |

#### Ecosystem 7: Geant4 (`Geant4/geant4`)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `G4-01` | **Modular Physics Lists (`G4VPhysicsConstructor`, `ParticleHP`)** | `[Developer]`, `[User]` | CERN GitLab `geant4/geant4` (`source/physics_lists/`, `G4ParticleHP`) | Composable EM/hadronic/HP neutron physics builders swappable per regime | `[Net-New]` — Adopted as `PhysicsList` trait composition in `llevant-transport` | `P3` |
| `G4-02` | **`GDML` XML Interchange & `G4TessellatedSolid`** | `[Interface]`, `[User]` | Geant4 User Guide (`GDML`, `G4TessellatedSolid`) | Standardized XML detector geometry interchange & voxelized STL solids | `[Net-New]` — `llevant-ir` adds GDML import/export adapter (`P2`) | `P2` |
| `G4-03` | **`G4TaskRunManager` Work-Stealing Parallelism** | `[Developer]` | Geant4 User Guide (`G4TaskRunManager`, PTL/TBB) | Eliminates thread tail imbalance on uneven history/shower complexity | `[Partial / Extension]` — `llevant` uses `rayon` `par_chunks` (`classify.rs`, `trace.rs`); extended to transport particle batches with deterministic chunk reduction | `P1` |
| `G4-04` | **`VecGeom` Vectorized BVH Navigation** | `[Developer]` | CERN `VecGeom` (`gitlab.cern.ch/VecGeom/VecGeom`) | SIMD-vectorized solid kernels & BVH4/8 pruning for CPU & GPU | `[Net-New]` — Directly shapes `Bvh4`/`Bvh8` & SIMD ray-packet kernels in `llevant` | `P1` |
| `G4-05` | **`AdePT` & `Celeritas` GPU EM Shower Offloading** | `[Developer]` | CERN `AdePT`; `https://github.com/celeritas-project/celeritas` | SoA track state, flat device views, and radix/partition sort by action/material to maximize GPU occupancy | `[Net-New]` — Core blueprint for `llevant-transport` GPU event-based solver | `P3` |
| `G4-06` | **`G4VSensitiveDetector` & `/score/` Parallel-World Meshes** | `[User]`, `[Developer]`, `[Interface]` | Geant4 Example `RE03` (`G4ScoringManager`, `/score/`) | Zero-code parallel-world 3D scoring meshes decoupled from physical geometry | `[Partial / Extension]` — Parallel-world tally overlay `[Implemented]` (`llevant_scene::field`); runtime `/score/` scorers `[Net-New]` | `P3` |
| `G4-07` | **`G4GeneralParticleSource` (GPS) & `G4VisExecutive`** | `[User]`, `[Interface]` | Geant4 App Dev Guide (`G4GeneralParticleSource`, `G4VisExecutive`) | Declarative complex spatial/angular/energy source distributions + interactive visualization | `[Partial / Extension]` — Interactive visualization `[Implemented]` (`llevant-viewer`); declarative GPS source engine `[Net-New]` | `P3` |

#### Ecosystem 8: FLUKA + Flair GUI

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `FLK-01` | **Flair Integrated Project IDE & Card Validator** | `[User]`, `[Interface]` | CERN FLUKA/Flair (`https://fluka.cern/flair`) | Per-card structured editing dialogs, live validation, job control, post-processing | `[Partial / Extension]` — `crates/llevant-viewer/src/panels/inspector/` has text card editor + live `Draft`; lacks structured numeric `TR`/surface widgets (`architecture.md:267`) & simulation runner | `P1` |
| `FLK-02` | **Flair `geoviewer` Real-Time 3D Ray-Tracing** | `[User]`, `[Interface]`, `[Developer]` | CERN Flair `geoviewer` (C++ exact ray-tracing + layer overlay) | Exact un-tessellated CSG 2D/3D ray-tracing with cutaways, outlines & `USRBIN` heatmaps | `[Implemented]` — Exceeded on GPU by `llevant-viewer` + `llevant-gpu` (G-buffer, SSAO, PBR, ghost, fields) | `P0` |
| `FLK-03` | **Integrated Visual Geometry Debugger** | `[User]`, `[Interface]` | CERN Flair Geometry Editor (`Debugger` tool) | Real-time visual overlap & blackhole detection overlaid on 2D/3D viewports | `[Partial / Extension]` — `llevant check --plots` & `llevant-viewer` gaps/problems rendering `[Implemented]`; interactive `lost` & `check --plots` trigger inside `llevant-viewer`/`web_viewer.py` missing | `P0` |
| `FLK-04` | **`USRBIN`/`USRBDX`/`RESNUCLEI` + `usbsuw` Mergers** | `[User]`, `[Interface]` | CERN FLUKA Manual (`USRBIN`, `USRBDX`, `RESNUCLEI`, `usbsuw`) | 3D mesh, boundary current, residual nuclide tallies + statistical multi-seed binary merging | `[Net-New]` — Adopted into `llevant-tally` Arrow/HDF5 statistical batch merger | `P2` |
| `FLK-05` | **`BIASING` / `WW-FACTOR` Visual Variance Reduction** | `[User]`, `[Interface]` | CERN FLUKA Manual (`BIASING`, `WW-FACTOR`); Flair layers | Visualizes region importance & weight windows directly over 2D/3D geometry slices | `[Net-New]` — `WeightWindowMesh` rendered as a native `Field` overlay in `llevant-viewer`/`web_viewer.py` | `P3` |
| `FLK-06` | **DICOM-to-`VOXELS`, `ROT-DEFI`/`LATTICE` + `FluDAG`** | `[User]`, `[Interface]`, `[Developer]` | CERN FLUKA/Flair Manual (`VOXELS`, `ROT-DEFI`); `svalinn/DAGMC` `FluDAG` | Medical CT/MRI voxel phantoms (O(1) DDA traversal) & CAD bridging | `[Net-New]` — `VoxelGrid` DDA kernel in `llevant-cad` | `P2` |

#### Ecosystem 9: SCALE / Shift / MAVRIC / Denovo (ORNL)

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `SCL-01` | **`MAVRIC` Automated `CADIS` & `FW-CADIS`** | `[User]`, `[Developer]` | ORNL SCALE Manual (`MAVRIC`, `Denovo` $S_n$ + `Shift` MC) | Automated coarse deterministic forward/adjoint -> consistent biased source & weight windows ($10^2$–$10^4\times$ FoM) | `[Net-New]` — Core of `llevant-vr` (`RMX`/`S_N` + `CADIS`/`FW-CADIS`) | `P3` |
| `SCL-02` | **`Shift` Kokkos CPU/GPU MC & Non-Uniform DD** | `[Developer]` | ORNL `Shift` / `Exnihilo` (`epj-n.org`, `ornl.gov`, `github.com/ORNL`) | Portable CPU/GPU execution + non-uniform spatial domain decomposition for huge models | `[Net-New]` — `wgpu` provides portable GPU; `llevant-transport` adds MPI/DD | `P3` |
| `SCL-03` | **`Omnibus` Python Driver & Pre-Flight Validation** | `[Interface]`, `[User]` | ORNL `Omnibus` User Manual | Declarative Python orchestration + pre-allocation validation of geometry/materials/tallies | `[Partial / Extension]` — Python `llevant` validates geometry; full simulation pre-flight validator `[Net-New]` | `P2` |
| `SCL-04` | **`Fulcrum` GUI & `OPUS` Inventory Plotting** | `[User]`, `[Interface]` | ORNL SCALE Manual (`Fulcrum`, `OPUS`) | Integrated desktop geometry view + interactive depletion/isotopic/spectrum plotting | `[Partial / Extension]` — Geometry/3D GUI `[Implemented]` (`llevant-viewer`); `OPUS`-style isotopic/spectrum chart panel `[Net-New]` | `P3` |
| `SCL-05` | **`ORIGEN` Depletion & `OBIWAN` Binary Query CLI/JSON** | `[User]`, `[Interface]`, `[Developer]` | ORNL SCALE Manual (`ORIGEN`, `OBIWAN`, `.f71`, `ii.json`) | Decay heat/activity/source spectra + fast CLI inspection/JSON export of binary depletion/tally archives | `[Net-New]` — Pattern adopted in `llevant data` / `llevant inspect` CLI & `llevant-deplete` | `P3` |

#### Ecosystem 10: DAGMC / CAD-to-MC Ecosystem

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `DAG-01` | **`svalinn/DAGMC` & `MOAB` `.h5m` Facet Transport** | `[Developer]`, `[Interface]` | `https://github.com/svalinn/DAGMC` (`DagMC` API, MOAB `.h5m`, `UWUW`) | Direct ray-tracing on faceted B-Rep CAD volumes with embedded `mat:`/`graveyard`/`tally:` metadata | `[Net-New]` — Core of `llevant-cad` (`hdf5-metno` MOAB `.h5m` reader without C++ MOAB runtime required) | `P2` |
| `DAG-02` | **`pshriwise/double-down` + Embree `f32`/`f64` Hybrid BVH** | `[Developer]` | `https://github.com/pshriwise/double-down` | 5–20× faster DAGMC ray-triangle intersections via SIMD BVH4/8 (`f32` AABBs) + `f64` watertight hit testing | `[Net-New]` — Pure-Rust `Bvh4`/`Bvh8` + `f64` Plücker watertight triangle kernel (plus optional `embree4-rs` backend) | `P2` |
| `DAG-03` | **`make_watertight` & Imprint/Merge Healing** | `[User]`, `[Developer]` | `svalinn/DAGMC` (`make_watertight`) | Seals micro-gaps & enforces conformal shared boundary facets so particles never leak on curved interfaces | `[Net-New]` — Pure-Rust vertex-welding / edge-zipping healer + Woop/Benthin watertight intersection | `P2` |
| `DAG-04` | **`cad_to_dagmc` & `CAD_to_OpenMC` Python Bridges** | `[User]`, `[Interface]` | `https://github.com/fusion-energy/cad_to_dagmc`; `https://github.com/svalinn/CAD_to_OpenMC` | Converts STEP/CadQuery/`build123d` assemblies into tagged `.h5m` without proprietary Cubit | `[Net-New]` — Direct Python interop in `llevant-py` (`from_dagmc`, `from_cad`) | `P2` |
| `DAG-05` | **`fusion-energy/stellarmesh`** | `[User]`, `[Developer]` | `https://github.com/fusion-energy/stellarmesh` | Conformal non-overlapping meshes for nested toroidal/stellarator NURBS | `[Net-New]` — Supported via conformal `.h5m` & tetrahedral mesh ingestion in `llevant-cad` | `P2` |
| `DAG-06` | **`dagmc_geometry_slice_plotter`** | `[User]`, `[Interface]` | `https://github.com/fusion-energy/dagmc_geometry_slice_plotter` | Fast 2D cross-section plotting of `.h5m` colored by volume/material tag | `[Net-New]` — Automatically unlocked across `llevant slice`, `llevant-viewer`, and `web_viewer.py` once `llevant-cad` implements the `GeometryKernel` trait | `P2` |

#### Ecosystem 11: Nuclear Data Tools & Formats

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `ND-01` | **`NJOY21`, `ENDFtk`, `ACEtk`, `GNDStk`** | `[Developer]`, `[Interface]` | `https://github.com/njoy/NJOY21`, `njoy/ENDFtk`, `njoy/ACEtk`, `njoy/GNDStk` | Zero-copy/span-based parsing of ENDF-6, ACE, and GNDS nuclear data formats | `[Net-New]` — Pure-Rust zero-copy parsers in `llevant-nucdata` (`ace`, `endf`, `gnds`) | `P3` |
| `ND-02` | **`LLNL/fudge` & `GNDS` Standard** | `[Developer]`, `[Interface]` | `https://github.com/LLNL/fudge` | Hierarchical XML/HDF5 nuclear data standard replacing 80-col ENDF-6 with covariance/yield schemas | `[Net-New]` — Supported via `llevant-nucdata` GNDS/HDF5 reader | `P3` |
| `ND-03` | **`openmc-dev/data` & `openmc_data_downloader`** | `[User]`, `[Interface]` | `https://github.com/openmc-dev/data`; `https://github.com/openmc-data-storage/openmc_data_downloader` | One-command SHA-pinned nuclear data library download + `cross_sections.xml` management | `[Net-New]` — Exposed as `llevant data pull` / `llevant data index` | `P3` |
| `ND-04` | **`mctools/ncrystal` Thermal Scattering** | `[Developer]`, `[User]` | `https://github.com/mctools/ncrystal` | On-the-fly coherent Bragg + inelastic thermal scattering in crystalline/textured materials | `[Net-New]` — Optional C-API FFI feature `ncrystal` in `llevant-nucdata` | `P3` |
| `ND-05` | **URR Probability Tables (`PURR`) & $S(\alpha,\beta)$ (`THERMR`)** | `[Developer]`, `[User]` | `openmc.data` / `NJOY21` (`PURR`, `THERMR`) | Unresolved resonance self-shielding (with correlated RNG) & bound-atom thermal scattering | `[Net-New]` — Core continuous-energy physics in `llevant-nucdata` + `llevant-transport` | `P3` |
| `ND-06` | **`SANDY` Stochastic ENDF Covariance Perturbation** | `[User]`, `[Developer]` | `https://github.com/luca-fiorito-11/sandy` | Perturbed ENDF/ACE/HDF5 sampling from MF31–35/40 covariances for Total Monte Carlo UQ | `[Net-New]` — Supported via `llevant-nucdata` multi-library batch perturbation runner | `P3` |
| `ND-07` | **`IAEA-NDS/endf-parserpy` & `arfc/pyne` (`r2s`)** | `[User]`, `[Interface]` | `https://github.com/IAEA-NDS/endf-parserpy`; `https://github.com/arfc/pyne` (`pyne.r2s`, `pyne.nucname`) | Fast ENDF parsing, ZAID/nuclide normalization, and Rigorous Two-Step (`r2s`) shutdown dose workflows | `[Partial / Extension]` — Basic ZAID parsing & atomic mass (`llevant_scene::materials`, `openmc_nuclide_to_zaid`, `tripoli_nuclide_to_zaid`); full `r2s` pipeline `[Net-New]` | `P3` |

#### Ecosystem 12: GitHub Neutronics, Rust/GPU Crates & Modern Interface Tooling

| ID | Feature Name | Role(s) | Concrete Source Citation | Workflow Need Addressed | Baseline Status in `llevantopen` | Priority |
|---|---|---|---|---|---|---|
| `GH-01` | **`paramak` & `neutronics-workshop`** | `[User]`, `[Interface]` | `https://github.com/fusion-energy/paramak`; `https://github.com/fusion-energy/neutronics-workshop` | Parametric 3D fusion CAD modeling & reproducible Jupyter workflows | `[Net-New]` — First-class integration in `llevant-py` (STEP/DAGMC + Jupyter `Image`) | `P2` |
| `GH-02` | **`openmc_plasma_source` & `openmc_source_plotter`** | `[User]`, `[Interface]` | `https://github.com/fusion-energy/openmc_plasma_source`; `https://github.com/fusion-energy/openmc_source_plotter` | Tokamak/stellarator D-T/D-D plasma sources + pre-flight 2D/3D source visualization | `[Net-New]` — Source distribution overlay in `llevant-viewer` / `web_viewer.py` + `llevant-transport::source` | `P3` |
| `GH-03` | **`neutronics_material_maker` & `openmc_regular_mesh_plotter`** | `[User]`, `[Interface]` | `https://github.com/fusion-energy/neutronics_material_maker`; `https://github.com/fusion-energy/openmc_regular_mesh_plotter` | Standardized multi-code material library + quick 2D regular mesh tally slice overlay | `[Partial / Extension]` — Mesh slice overlay `[Implemented]` (`llevant slice --field`, `llevant-py`); `neutronics_material_maker` IR material bridge `[Net-New]` | `P2` |
| `GH-04` | **`ANL-CEF/Cardinal` (MOOSE + OpenMC + NekRS)** | `[Developer]`, `[Interface]` | `https://github.com/ANL-CEF/Cardinal` | Zero-file-I/O in-memory multiphysics coupling of volumetric heating & $T/\rho$ fields | `[Net-New]` — Zero-copy Arrow/NumPy/C-ABI `LiveSession` interface | `P3` |
| `GH-05` | **`mit-crpg/OpenMOC` & `RandomRay`** | `[Developer]` | `https://github.com/mit-crpg/OpenMOC`; `https://github.com/mit-crpg/RandomRay` | SoA flat-source-region ray sweeping with SIMD/GPU bandwidth saturation | `[Net-New]` — Reference layout for `llevant-transport::random_ray` | `P3` |
| `GH-06` | **`Fusion4Energy/migjorn` & `idaholab/MontePy`** | `[Interface]`, `[Developer]` | `https://github.com/Fusion4Energy/migjorn`; `https://github.com/idaholab/MontePy` | Fast Rust MCNP parsing/renumbering (`migjorn`) + semantic Python deck manipulation (`MontePy`) | `[Implemented]` / `[Partial / Extension]` — `migjorn = "=0.4.0"` powers `llevant` & `llevant_scene::source`; semantic `llevant-ir` Python editing & streaming zero-copy parsing (`2 GB -> <400 MB`) `[Extension]` | `P1` |
| `GH-07` | **`nukehub-dev/nucleide` & `rpease/mcnp-language-server`** | `[Developer]`, `[Interface]` | `https://github.com/nukehub-dev/nucleide`; `https://github.com/rpease/mcnp-language-server` | Rust material/XS primitives + VS Code/Neovim Language Server Protocol (hover, go-to, overlap diagnostics) | `[Net-New]` — `llevant-serve` LSP server (`tower-lsp`) across MCNP/OpenMC/T4/DRAGON5 | `P2` |
| `GH-08` | **Rust High-Perf GPU/Data Stack (`wgpu`, `rayon`, `pyo3`, `arrow`, `hdf5`)** | `[Developer]`, `[Interface]` | Crates: `wgpu`, `encase`, `glam`, `rayon`, `pyo3`, `arrow`/`parquet`, `hdf5-metno`, `rkyv` | Portable GPU compute, fearless CPU parallelism, zero-copy Arrow/Parquet/HDF5 streaming | `[Partial / Extension]` — Uses `wgpu 30.0.1`, `rayon`, `pyo3 0.30`, `memmap2`, `bitcode`; adds `arrow`/`parquet`, `hdf5-metno`, `rkyv`, `PipelineCache` | `P1` |
| `GH-09` | **Persistent Simulation Daemon & MCP Server (`rmcp`)** | `[Interface]`, `[Developer]` | Rust `rmcp` / JSON-RPC daemon architecture | Resident `Arc<Model>` + VRAM + nuclear data serving CLI, `web_viewer.py`, IDE/LSP, and AI agents | `[Net-New]` — Planned P2 (`llevant serve`), critical to eliminate `web_viewer.py` subprocess bottleneck | `P1` |

---

## 4. Target System Architecture (R3)

### 4.1 Design Principles

1. **Eliminate MCNP-Text Intermediaries**: Currently, `llevant-openmc`, `llevant-tripoli`, and `llevant-dragon` serialize parsed ASTs into giant MCNP text strings and re-parse them via `migjorn`. Introducing **`llevant-ir`** allows every frontend (MCNP `migjorn`, OpenMC XML, TRIPOLI-4, DRAGON5, GDML, Python builders) to compile **directly into a canonical, lossless, strongly-typed Geometry & Physics IR**—preserving native distributed materials, boundary conditions (`reflective`/`white`/`periodic`), `ECRASE`, `CARCEL` `SPLITR`/`SECT`, and source card spans without MCNP syntax limits—while still supporting bidirectional `llevant-ir -> MCNP/OpenMC` export.
2. **Unified `GeometryKernel` Abstraction**: Transport, plotting, volume estimation, and QA interact with geometry through a polymorphic, monomorphized kernel trait (`CsgKernel`, `DagmcMeshKernel`, `TetMeshKernel`, `VoxelKernel`, or `HybridKernel` where CSG cells can be filled by DAGMC/Tet/Voxel universes).
3. **Strict Decoupling of Pure Data & Execution**: All heavy state (`ModelIr`, `CsgSoa`, `DagmcMesh`, `NucDataLib`, `GpuModel`) lives in immutable `Arc`-wrapped, `mmap`/`rkyv`-backed structures, enabling zero-copy sharing across `rayon` threads, `wgpu` uploads, `llevant serve` daemon sessions, `PyO3`, `MCP`, and `LSP`.
4. **Backward Compatibility**: All 12 existing workspace crates remain functional; `llevant` re-exports compatible types and delegates internally to the optimized SIMD/SoA and IR layers.

### 4.2 Proposed Modular Workspace Crate Topology

```text
  ┌────────────────────────── INPUT FRONTENDS & CONVERTERS ──────────────────────────┐
  │  migjorn (MCNP)   llevant-openmc   llevant-tripoli   llevant-dragon   GDML/CAD   │
  └────────┬────────────────┬────────────────┬─────────────────┬──────────────┬──────┘
           └────────────────┴────────────────┼─────────────────┴──────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ llevant-ir : Native Multi-Code Geometry/Material/Source/Tally IR (bidirectional) │
  └────────┬─────────────────────────────────┬───────────────────────────────┬───────┘
           ▼                                 ▼                               ▼
  ┌──────────────────────┐     ┌───────────────────────────┐   ┌─────────────────────┐
  │ llevant (CSG Kernel) │     │ llevant-cad (Mesh Kernel) │   │ llevant-nucdata     │
  │ - SoA Shapes/Planes  │◄───►│ - DAGMC .h5m (hdf5-metno) │   │ - ENDF/ACE/HDF5/GNDS│
  │ - SIMD Bvh4/Bvh8     │     │ - f32 BVH + f64 Plücker   │   │ - Hashed XS / Opti  │
  │ - Octrees + Residuals│     │ - TetMesh O(1) adjacency  │   │ - WMP + URR + NCrys │
  │ - QMC/Strat Volumes  │     │ - VoxelGrid DDA + Healing │   └──────────┬──────────┘
  └────────┬─────────────┘     └─────────────┬─────────────┘              │
           └─────────────────┬───────────────┘                            │
                             ▼                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ llevant-transport : CPU SIMD & wgpu Event-Based MC / Woodcock / RandomRay / CMFD │
  └────────┬─────────────────────────────────┬───────────────────────────────┬───────┘
           ▼                                 ▼                               ▼
  ┌──────────────────────┐     ┌───────────────────────────┐   ┌─────────────────────┐
  │ llevant-vr           │     │ llevant-tally             │   │ llevant-deplete     │
  │ - RMX / Coarse Sn    │     │ - Privatized Accumulators │   │ - CRAM-14/16 Sparse │
  │ - CADIS / FW-CADIS   │     │ - Arrow IPC / Parquet     │   │ - Decay Chains / R2S│
  │ - WWG Mesh Overlays  │     │ - HDF5 StatePoint / MCTAL │   │ - OBIWAN-style JSON │
  └──────────────────────┘     └─────────────┬─────────────┘   └─────────────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ llevant-scene  +  llevant-render  +  llevant-gpu (wgpu Compute + PipelineCache)  │
  └────────┬─────────────────┬─────────────────┬─────────────────┬───────────────────┘
           ▼                 ▼                 ▼                 ▼
     llevant-cli       llevant-viewer     llevant-py       llevant-serve
     (`llevant`)       (Desktop GPU)    (PyO3/Arrow/lib)   (Daemon + MCP + LSP)
                                                                 ▲
                                                                 │ HTTP/WS/Arrow IPC
                                                           web_viewer.py
```

### 4.3 End-to-End Data Flow (Ingestion $\to$ Kernel $\to$ Transport/QA $\to$ Tally $\to$ Interfaces)

```text
 [Input Files / Python Objects]
        │  (zero-copy mmap / streaming parser)
        ▼
 [llevant-ir::ModelIr] ──(export)──► MCNP / OpenMC / GDML text
        │
        ├──► [llevant::CompiledCsg (SoA + Bvh4/8 + Octree)] ──► [wgpu::GpuModel + PipelineCache]
        ├──► [llevant-cad::DagmcMesh / TetMesh]             ──► [wgpu::GpuMeshBuffers]
        └──► [llevant-nucdata::NucDataLib (mmap/rkyv)]
                    │
        ┌───────────┴───────────────────────────────────┐
        ▼                                               ▼
 [QA / Geometry Engine]                        [Transport & Physics Solvers]
  - SIMD ray packets (4/8-wide)                 - Philox4x32-10 Counter RNG
  - Adaptive QMC VolumeGrid                     - CPU Event/History + Woodcock
  - Overlap/Gap & Lost Particle                 - GPU Event Queues + Radix Sort
  - Structural Fingerprints                     - RandomRay / CMFD / CADIS / CRAM
        │                                               │
        └───────────────────┬───────────────────────────┘
                            ▼
 [llevant-tally::TallyResult -> Arrow RecordBatch / HDF5 StatePoint / Parquet]
                            │
        ┌───────────┬───────┴────────┬──────────────┬─────────────┐
        ▼           ▼                ▼              ▼             ▼
   llevant-cli  llevant-viewer   llevant-py   llevant-serve   web_viewer.py
   (Text/JSON)  (G-buffer+Passes)(Arrow/NumPy)(MCP + LSP)    (Daemon Client)
```

---

## 5. Concrete Data Structures (R3)

All performance-critical structures follow **Structure-of-Arrays (SoA)** packing, explicit `#[repr(C)]` alignment for zero-copy `bytemuck` / `rkyv` / GPU storage buffer mapping, and `u32` arena indexing (no pointer chasing).

### 5.1 Geometry & Spatial Acceleration Structures

#### A. Splitting the 112-Byte `Shape` Enum into Specialized SoA Arenas (`llevant`)
Currently `crates/llevant/src/surface.rs:165-174` stores every plane in a `112 B` `Shape` enum (`Quadric` dominates size; `Cone`/`Torus` box-allocate). Splitting into typed SoA tables cuts plane memory **3.5×** (`112 B -> 32 B`) and unlocks AVX2/AVX-512 contiguous loads:

```rust
/// Tagged handle packed in 32 bits: 3-bit kind + 29-bit index into typed SoA table.
/// Supports up to 536M surfaces per kind with zero pointer chasing.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShapeHandle(pub u32);

impl ShapeHandle {
    pub const KIND_SHIFT: u32 = 29;
    pub const IDX_MASK: u32 = (1 << 29) - 1;
    #[inline(always)]
    pub fn kind(self) -> ShapeKind {
        match (self.0 >> Self::KIND_SHIFT) as u8 {
            0 => ShapeKind::Plane,
            1 => ShapeKind::Quadric,
            2 => ShapeKind::Cone,
            3 => ShapeKind::Torus,
            _ => ShapeKind::BoundaryMask,
        }
    }
    #[inline(always)]
    pub fn idx(self) -> usize { (self.0 & Self::IDX_MASK) as usize }
}

#[repr(u8)]
pub enum ShapeKind { Plane = 0, Quadric = 1, Cone = 2, Torus = 3, BoundaryMask = 4 }

/// 32-byte aligned SoA planes: 4 planes = 2 cache lines (128 B) for AVX2 4-wide eval.
/// Note: `std::vec::Vec<f64>` only guarantees 8-byte alignment; back columns with
/// `aligned_vec::AVec<f64, ConstAlign<32>>` (or pack AoSoA-4 `#[repr(C, align(32))] struct Plane4`)
/// when issuing aligned `_mm256_load_pd`, or use unaligned `_mm256_loadu_pd` loads on `Vec<f64>`.
pub type AlignedF64Vec = aligned_vec::AVec<f64, aligned_vec::ConstAlign<32>>;

#[derive(Clone, Debug, Default)]
pub struct PlaneSoa {
    pub nx: AlignedF64Vec, // unit normal x (32-byte aligned)
    pub ny: AlignedF64Vec, // unit normal y
    pub nz: AlignedF64Vec, // unit normal z
    pub d:  AlignedF64Vec, // plane offset: n·p - d = 0
}

/// 104-byte Quadric split into hot 32-byte aligned SIMD SoA columns (no padding waste).
#[derive(Clone, Debug, Default)]
pub struct QuadricSoa {
    pub ox: AlignedF64Vec, pub oy: AlignedF64Vec, pub oz: AlignedF64Vec, // local origin
    pub axx: AlignedF64Vec, pub ayy: AlignedF64Vec, pub azz: AlignedF64Vec,
    pub axy: AlignedF64Vec, pub ayz: AlignedF64Vec, pub azx: AlignedF64Vec,
    pub bx: AlignedF64Vec, pub by: AlignedF64Vec, pub bz: AlignedF64Vec,
    pub c:  AlignedF64Vec,
}

/// Flat Cone & Torus arenas (eliminating Box<Cone> / Box<Torus> heap allocations).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ConeData { pub quadric_idx: u32, pub _pad: u32, pub apex: [f64; 3], pub axis: [f64; 3] } // 56 B

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TorusData { pub rot: [f64; 9], pub disp: [f64; 3], pub a: f64, pub b: f64, pub c: f64, pub axis: u8 } // 128 B padded
```

#### B. SIMD `Bvh4` / `Bvh8` Node Layout (`llevant` & `llevant-cad`)
Replaces binary 56 B `Bvh::Node` (`crates/llevant/src/bvh.rs:7-13`) with a 4-wide (`Bvh4`, 128 B = 2 cache lines) or 8-wide (`Bvh8`, 256 B) node storing quantized or `f32` child AABBs SoA so one SIMD comparison tests all 4 or 8 children simultaneously:

```rust
/// 4-wide SIMD BVH node (128 bytes, cache-line aligned).
/// Children are conservative f32 outward-rounded bounds (see pack::round_out).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Bvh4Node {
    pub min_x: [f32; 4], // 16 B
    pub max_x: [f32; 4], // 16 B
    pub min_y: [f32; 4], // 16 B
    pub max_y: [f32; 4], // 16 B
    pub min_z: [f32; 4], // 16 B
    pub max_z: [f32; 4], // 16 B
    /// Inner child (bit 31 = 0): bits 0..30 = child node index.
    /// Leaf (bit 31 = 1):
    /// - Fast path (<= 2^24 primitives): bits 0..23 = direct item start, bits 24..30 = count.
    /// - Large CAD/DAGMC volumes (> 2^24 primitives, >16,777,216 tris): prevent silent 24-bit
    ///   overflow by indexing an external `LeafSlice { start: u32, count: u32 }` table via bits
    ///   0..30 (or storing full 32-bit `leaf_start` in `_pad[i]` with count in `child[i]` bits 0..30).
    pub child: [u32; 4], // 16 B
    pub _pad:  [u32; 4], // 16 B -> exact 128 B (optional 32-bit leaf_start[4] or parent/mask metadata)
}

/// External leaf descriptor when a single BVH exceeds 2^24 primitives (`llevant-cad`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafSlice {
    pub start: u32, // Full 32-bit offset into tri_verts / item arena
    pub count: u32,
} // 8 B
```

#### C. Watertight `DagmcMesh` & Face-Adjacent `TetMesh` (`llevant-cad`)
Inspired by `pshriwise/double-down` (`DAG-02`) and PHITS `mesh=tet` (`PH-03`):

```rust
/// Watertight CAD facet mesh loaded from MOAB .h5m (DAGMC) or STL.
/// Uses f32 Bvh4/8 for pruning + f64 vertex coordinates for watertight Plücker intersection.
pub struct DagmcMesh {
    pub vertices_f64: Vec<[f64; 3]>,    // Exact f64 coordinates (shared across adjacent volumes -> watertight!)
    pub tri_verts:    Vec<[u32; 3]>,    // 12 B/tri vertex indices
    pub tri_surf:     Vec<u32>,         // Owning CAD surface ID
    /// Forward/reverse volume ID per CAD surface: [front_vol, back_vol] (u32::MAX = graveyard/void)
    pub surf_vols:    Vec<[u32; 2]>,
    pub bvh4:         Vec<Bvh4Node>,    // Top-level + per-volume BVH4s
    pub vol_mats:     Vec<u32>,         // Resolved material ID per volume from UWUW/`mat:` tags
    pub vol_bcs:      Vec<BoundaryCond>,// Vacuum, Reflective, White, Periodic
}

/// Unstructured tetrahedral mesh with O(1) neighbor-face stepping (PH-03 / MCNP-06).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TetElem {
    pub verts:     [u32; 4], // 4 vertex indices into vertices_f64
    pub neighbors: [u32; 4], // Opposite face neighbor tet ID, or u32::MAX (boundary -> CSG/BC)
    pub mat_id:    u32,
    pub region_id: u32,      // Depletion/tally zone ID
} // 40 B/tet; 10M tets = 400 MB RAM
```

#### D. Native Multi-Code Geometry & Physics IR (`llevant-ir`)
Eliminates `openmc -> MCNP text -> migjorn` translation overhead and preserves non-MCNP semantics:

```rust
pub struct ModelIr {
    pub title: String,
    pub metadata: IrSourceFormat, // Mcnp | OpenMc | Tripoli4 | Dragon5 | Gdml | Dagmc
    pub surfaces: Vec<IrSurface>, // Carries BoundaryCond { Vacuum, Reflective, White, Periodic(u32) }
    pub transforms: Vec<IrTransform>,
    pub materials: Vec<IrMaterial>, // Preserves nuclides, thermal S(a,b), temperature K, density
    pub cells: Vec<IrCell>,         // Supports MaterialFill::Single(u32) | Distributed(Vec<u32>) | Universe | Lattice
    pub lattices: Vec<IrLattice>,
    pub dagmc_refs: Vec<IrDagmcUniverse>, // Embedded .h5m / TetMesh / Voxel universes
    pub sources: Vec<IrSource>,
    pub tallies: Vec<IrTally>,
    pub settings: IrRunSettings,
    /// Optional lossless span map back to original file lines/tokens for LSP & Viewer card editing.
    pub source_spans: Option<SourceSpanIndex>,
}
```

### 5.2 Particle Banks, Counter-Based RNG & GPU Event Queues

#### A. CPU/GPU SoA `ParticleBank` & Counter-Based `Philox4x32-10` PRNG
Eliminates history divergence, guarantees bit-exact `1` vs `N` thread reproducibility on CPU (`f64`), and provides statistically validated ($1\sigma$ against the CPU `f64` reference) GPU (`f32`) transport (`OMC-02`, `G4-05`, Pitfalls #3 & #5):

```rust
/// Counter-based Philox4x32-10 PRNG state (24 bytes; zero sequential state dependencies).
/// Output stream is a pure function of (seed, history_id, stream_id, counter).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhiloxRng {
    pub key: [u32; 2],     // (seed_lo ^ stream_id, seed_hi)
    pub counter: [u32; 4], // [step_ctr, bounce_ctr, history_lo, history_hi]
}

/// CPU SoA Particle Bank (64-byte aligned slabs for AVX2/AVX-512 gather-free vectorization).
pub struct ParticleBankSoa {
    pub len: usize,
    pub pos_x: Vec<f64>, pub pos_y: Vec<f64>, pub pos_z: Vec<f64>,
    pub dir_x: Vec<f64>, pub dir_y: Vec<f64>, pub dir_z: Vec<f64>,
    pub energy_ev: Vec<f64>,
    pub weight:    Vec<f64>,
    pub time_s:    Vec<f64>,
    pub particle:  Vec<u8>,   // 0=Neutron, 1=Photon, 2=Electron, 3=Positron
    pub event:     Vec<u8>,   // EventKind: Advance=0, CrossSurface=1, Collide=2, Tally=3, Dead=4
    pub cell_id:   Vec<u32>,
    pub mat_id:    Vec<u32>,
    pub rng:       Vec<PhiloxRng>,
}
```

#### B. GPU WGSL `ParticleSoA` & Event-Sorted Queues (`llevant-transport` + `llevant-gpu`)
Designed for `wgpu` compute shaders (`encase`/`bytemuck` compatible, `f32` positions with shifted-origin cell refinement):

```wgsl
// WGSL storage buffers (all SoA arrays indexed by particle_id: u32)
struct ParticleMeta {
    cell_id: u32,
    mat_id:  u32,
    // bits 0..7 = EventKind, 8..15 = particle, 16..27 = sub-step bounce_ctr (Philox counter[1]),
    // 28..31 = flags (upgrade to an explicit `bounce_ctr: u32` word if >4096 RNG draws/step are needed).
    event_and_particle: u32,
    history_lo: u32,
    history_hi: u32,
    step_ctr:   u32,
    rng_key0:   u32,
    rng_key1:   u32,
}; // 32 bytes

// Active queues sorted by radix key: (event: 3b | mat_id: 21b | energy_bucket: 8b)
@group(1) @binding(0) var<storage, read_write> pos_xyz_e: array<vec4<f32>>; // (x, y, z, log_E)
@group(1) @binding(1) var<storage, read_write> dir_xyz_w: array<vec4<f32>>; // (u, v, w, weight)
@group(1) @binding(2) var<storage, read_write> p_meta:    array<ParticleMeta>;
@group(1) @binding(3) var<storage, read_write> sort_keys: array<u32>;
@group(1) @binding(4) var<storage, read_write> sort_vals: array<u32>;      // particle indices
@group(1) @binding(5) var<storage, read_write> queue_offsets: array<u32>;  // prefix-sum offsets per EventKind
@group(1) @binding(6) var<storage, read_write> time_s: array<f32>;         // Optional: allocated when time-dependent scoring/sources are active
```

### 5.3 Nuclear Data & Cross-Section Structures (`llevant-nucdata`)

Combines OpenMC's logarithmic/hashed energy bucket lookup (avoiding unionized memory blowup on 300+ depletion nuclides), Serpent's `opti 1..4` trade-off modes (`SRP-02`), and OpenMC's Windowed Multipole (`OMC-04`):

```rust
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XsOptiTier {
    /// Opti1 (Min RAM): per-nuclide binary/hashed lookup, no macroscopic precompute.
    Opti1Compact = 1,
    /// Opti2 (Balanced, default): logarithmic hash buckets (N_bins = 8192) -> O(1) bracket per nuclide.
    Opti2Hashed = 2,
    /// Opti3 (Fast): unionized energy grid over transport-dominant materials + pointer table.
    Opti3Unionized = 3,
    /// Opti4 (Max Speed): full unionized grid + precomputed macroscopic Sigma_t/a/f per material.
    Opti4MacroPrecomputed = 4,
}

/// O(1) Logarithmic Energy Hash Accelerator (per nuclide or unionized).
/// Maps log(E) into `n_bins` buckets storing [lo_idx, hi_idx] in the nuclide's energy grid.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct EnergyHashIndex {
    pub log_e_min: f64,
    pub inv_d_log_e: f64,
    pub n_bins: u32,
    pub bin_starts: Vec<u32>, // length n_bins + 1; average binary search span <= 3-4 points
}

/// Zero-copy mmap/rkyv-compatible Continuous-Energy Nuclide Table.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NuclideTable {
    pub zaid: u32,
    pub awr: f64,                   // Atomic weight ratio
    pub temp_k: f64,
    pub energy_ev: Vec<f64>,        // Sorted grid
    pub hash_idx: EnergyHashIndex,
    /// Packed SoA reaction cross sections [total, elastic, capture, fission, inelastic...]
    pub xs_total: Vec<f64>,
    pub xs_abs: Vec<f64>,
    pub xs_fission: Vec<f64>,
    pub nu_total: Vec<f64>,
    pub reactions: Vec<ReactionBlock>, // MT, Q-value, angular/energy law tables
    pub urr: Option<UrrProbTable>,     // Unresolved Resonance Probability Table (ND-05)
    pub wmp: Option<WmpTable>,         // Optional Windowed Multipole poles/residues (OMC-04)
}

/// Windowed Multipole (WMP) representation for on-the-fly Doppler broadening.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WmpTable {
    pub e_min: f64,
    pub sqrt_e_step: f64,
    pub windows: Vec<[u32; 2]>,     // [pole_start..pole_end] per sqrt(E) window
    /// Complex pole sqrt(E_p) = (re, im) and residues [r_s, r_a, r_f] (complex pairs)
    pub poles_re: Vec<f64>, pub poles_im: Vec<f64>,
    pub res_s_re: Vec<f64>, pub res_s_im: Vec<f64>,
    pub res_a_re: Vec<f64>, pub res_a_im: Vec<f64>,
    pub res_f_re: Vec<f64>, pub res_f_im: Vec<f64>,
    pub curvefit_coeffs: Vec<[f64; 4]>, // Smooth polynomial background per window
}
```

### 5.4 Tallies, Variance Reduction & Depletion Structures

```rust
/// Thread-local / Workgroup-privatized Tally Accumulator (reduces contention -> deterministic merge).
pub struct TallyAccumulator {
    pub tally_id: u32,
    pub n_bins: usize,
    /// Current batch sum and sum-of-squares across completed batches (exact f64 Kahan/Neumaier or u128 fixed-point).
    pub batch_sum: Vec<f64>,
    pub batch_sum_sq: Vec<f64>,
    pub n_batches: u64,
    /// Filter axes mapped directly to Arrow RecordBatch / HDF5 StatePoint dimensions.
    pub filter_axes: Vec<TallyAxis>, // Cell, CellInstance, Material, Mesh(x,y,z|r,z,theta|Tet), Energy, Time, Angle, Nuclide
}

/// Canonical Apache Arrow / Parquet / HDF5 StatePoint column schema for zero-copy tally streaming.
/// Schema columns: (bin_idx: UInt64, cell_id: Int64?, instance_id: UInt64?, mat_id: Int64?,
///                  voxel_id: UInt32?, energy_lo_ev: Float64?, energy_hi_ev: Float64?,
///                  score_mt: Utf8, nuclide: Utf8?, particle: UInt8, mean: Float64, rel_err: Float64).
pub struct ArrowTallyBatch {
    pub tally_id: u32,
    pub name: String,
    pub n_batches: u64,
    pub schema_version: u16, // v1 Arrow C Data Interface compatible
    pub batch: arrow_array::RecordBatch,
}

/// Space-Energy Weight Window Mesh (llevant-vr: CADIS / FW-CADIS / RMX).
/// Host representation; uploads `w_lower` as a flat `array<f32>` storage buffer to `llevant-gpu` / `llevant-transport`.
pub struct WeightWindowMesh {
    pub geom: aci_tallies_geom::Geometry, // Reuses existing mesh geometry crate
    pub energy_bounds_ev: Vec<f64>,       // Descending or ascending energy group edges
    /// Lower weight bound w_low per [voxel * n_groups + g] (f32 compact for GPU & Field overlay!)
    pub w_lower: Vec<f32>,
    pub survival_ratio: f32,              // Default 3.0 -> w_survive = 3 * w_low
    pub upper_ratio: f32,                 // Default 5.0 -> w_upper = 5 * w_low (splitting threshold)
    pub biased_source_pdf: Option<Vec<f64>>, // Consistent CADIS source biasing weights
}

/// CSR Sparse Transmutation Matrix for Order-14/16 CRAM Depletion (llevant-deplete).
pub struct CramSolver {
    pub n_nuclides: usize,
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    /// Decay constants lambda_i (s^-1) + one-group (sigma * phi) reactions
    pub values: Vec<f64>,
    pub cram_order: u8, // 14 or 16 (Pusa & Leppänen partial-fraction poles/residues)
}
```

### 5.5 Persistent Daemon, MCP & LSP Protocol Schemas (`llevant-serve`)

```rust
/// Shared resident session inside `llevant serve` (eliminates cold load & subprocess spawns).
pub struct DaemonSession {
    pub model_id: blake3::Hash,
    pub ir: Arc<ModelIr>,
    pub csg: Arc<llevant::Model>,
    pub cad: Option<Arc<DagmcMesh>>,
    pub gpu: Option<Arc<llevant_gpu::GpuModel>>, // Cached via wgpu::PipelineCache
    pub nucdata: Option<Arc<NucDataLib>>,
    pub tallies: DashMap<String, Arc<ArrowTallyBatch>>, // Live or loaded StatePoints
    pub lsp_index: Arc<LspSymbolIndex>,
}

/// Real-time LSP symbol & diagnostic index backed by `SourceSpanIndex` (GH-07).
pub struct LspSymbolIndex {
    pub uri: String,
    pub format: IrSourceFormat,
    /// Sorted byte-span intervals -> (SymbolKind { Cell, Surface, Material, Transform, Universe }, id)
    pub spans: Vec<(u32, u32, SymbolRef)>,
    /// Reverse cross-references for go-to-definition / find-references / live hover
    pub refs_by_symbol: FxHashMap<SymbolRef, Vec<(u32, u32)>>,
    pub diagnostics: Vec<llevant::Diagnostic>,
}

/// JSON-RPC / Unix-Socket / WebSocket / HTTP Binary Framing for CLI, web_viewer.py, MCP, and LSP.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum DaemonRequest {
    LoadModel { path: String, options: LoadOptions, gpu: bool },
    Locate { model_id: String, points: Vec<[f64; 3]>, depth: Option<usize> },
    Trace { model_id: String, lines: Vec<([f64; 3], [f64; 3])> },
    RenderFrame { model_id: String, camera: CameraSpec, visibility: VisibilitySpec, field: Option<FieldSpec> },
    SliceFrame  { model_id: String, slice: SliceSpec, field: Option<FieldSpec> },
    CheckQa     { model_id: String, h: f64, write_plots_dir: Option<String> },
    DiagnoseLost { model_id: String, coords: Vec<[f64; 3]>, log_path: Option<String> },
    QueryTallyArrow { tally_id: String, filters: Vec<FilterSlice> }, // Returns Arrow IPC stream!
    RunSimulation { model_id: String, mode: SimMode, batches: u32, particles_per_batch: u64 },
}
```

---

## 6. Maximum-Performance Engineering Design (R3)

This section establishes the concrete engineering optimizations required to achieve hardware-saturating throughput across CPUs, GPUs, clusters, and memory/I/O subsystems.

### 6.1 CPU & SIMD Vectorization (`llevant`, `llevant-cad`, `llevant-transport`)

1. **4-Wide / 8-Wide Ray-Packet Tracing & `Bvh4`/`Bvh8` Traversal**:
   - **Target**: `portable_simd` (`std::simd::f64x4`, `f32x4`, `f32x8` gated behind `nightly` or `wide`/`pulp` runtime-dispatched `std::arch` `avx2`/`avx512f`/`neon` backends on stable Rust `1.95`!). Using `pulp` or explicit `#[target_feature(enable = "avx2,fma")]` dispatch guarantees stable Rust `1.95` compatibility without nightly lock-in.
   - **How**: Group coherent rays (e.g., `Slice::rows`, `Camera` pixel tiles $2\times2$ or $4\times2$, `VolumeGrid` parallel probe lines, and `RandomRay` angular bundles) into ray packets. Test one ray packet against `Bvh4Node` (4 child AABBs in one `f32x4` slab test) or 4 rays against 1 child AABB simultaneously, keeping an active-lane bitmask.
   - **Expected Gain**: $2.5$–$4.5\times$ speedup on `Model::slice`, CPU `Model::render`, `VolumeGrid` tracing, and `DagmcMesh` BVH traversal.
2. **Plane & Axial-Quadric SIMD Fast Paths**:
   - Exploiting `PlaneSoa` (§5.1.A): evaluating 4 planes simultaneously (`n_x*p_x + n_y*p_y + n_z*p_z - d`) fits in 4 `f64x4` FMA instructions with zero branches, replacing scalar `Shape::side` virtual `match` dispatch on planes (which comprise >70% of surfaces in reactor/tokamak models). Because `Vec<f64>` only guarantees 8-byte alignment, SIMD plane evaluation must back columns with `aligned_vec::AVec<f64, ConstAlign<32>>` / `AoSoA-4` (`#[repr(C, align(32))]`), or issue unaligned `_mm256_loadu_pd` loads.
   - Specialize axis-aligned cylinders (`CX`/`CY`/`CZ`, `C/X`/`C/Y`/`C/Z`) and spheres (`S`/`SO`) into a `SimpleQuadricSoa` fast path avoiding full $3\times3$ symmetric matrix-vector products.
3. **Fixing Baseline Engine Bottlenecks (`docs/architecture.md` Roadmap)**:
   - **Sparse-World `first_hits` (`sparse.i`)**: Replace the blind $16\text{ cm} \times 2^k$ doubling-window loop in `crates/llevant/src/hit.rs` with **BVH-driven span jumping**: intersect the ray directly with `Skip::unit_box(0)` and top-level `Bvh4` leaf intervals to jump straight to candidate-bearing `t`-ranges, reducing empty-space ray steps from $O(\log(5\text{ km}))$ window restarts to 1 BVH walk.
   - **Level-1 Filler Evaluation Bottleneck**: In multi-level models where level 1 dominates (`architecture.md:259`), cache the **parent-frame-to-level-1 transform inverse** per `Unit` edge and build **per-filler-instance tight clip octrees** when a universe fills >8 envelopes with wildly differing ROIs, plus hoist `Skip` visibility masks into a 64-bit bloom/bitset tested before descending into level-1 programs.
   - **Cheaper `Model::volumes_to` Convergence (`2 hours -> minutes`)**:
     - Replace pure uniform random-shift `VolumeGrid` probes with **Stratified Quasi-Monte Carlo (QMC) Halton/Sobol 2D jittered line grids** per probe (retaining independent probe seeds for unbiased $\sigma$ estimation!)—cutting variance from $O(N^{-1})$ to $O(N^{-1.5})$–$O(N^{-2})$ on smooth boundaries.
     - Add **Per-Group Early Termination & Dead-Box Culling**: in `refine.rs`, once a cell/group meets `rel_sigma` or `abs_sigma`, immediately mask it out of subsequent finer `Covered` box unions so `h/64` lines only traverse unconverged bounding sub-boxes rather than re-tracing converged neighboring geometry.
     - Add **GPU-Assisted Pre-Screening** (optional `gpu` path for `volumes_to` initial coarse `h` discovery) while retaining exact `f64` CPU `u128` line accumulation as the authoritative estimator.

### 6.2 GPU (`wgpu` Compute) Architecture (`llevant-gpu`, `llevant-transport`)

1. **Eliminating the 8–30 s WGSL Shader Compilation Stall**:
   - **Root Cause**: `GpuModel::upload` (`crates/llevant-gpu/src/pack.rs`) compiles custom WGSL strings per `(MAXL, CODE_STACK, BVH_STACK, n_pool_bindings)` from scratch when drivers miss their opaque internal cache.
   - **Solution**:
     1. Enable `wgpu::Features::PIPELINE_CACHE` (`wgpu 30.0.1` native `Device::create_pipeline_cache` + `PipelineCache::get_data()`), persisting serialized Vulkan/DX12/Metal binary pipeline caches in `LLEVANT_CACHE_DIR/pipelines/<adapter_id>_<shader_id>.bin`!
     2. **Quantized Stack Buckets with `naga` Specialization**: Bucket `CODE_STACK` and `BVH_STACK` cleanly (e.g., `8, 16, 24, 32, 48, 64`) where `decision 0015` showed negligible cost, OR ship pre-compiled SPIR-V / `PipelineCache` artifacts for common tiers—cutting cold startup from 8–30 s to **<150 ms**.
2. **Event-Based Transport Pipeline + GPU Radix Sort (`llevant-transport`)**:
   - Following `Celeritas` (`G4-05`) and `OpenMC` (`OMC-02`), transport NEVER loops recursively per thread. Each iteration runs 5 specialized compute dispatches chained via indirect dispatch arguments (`dispatch_workgroups_indirect`):
     1. `classify_and_sort`: Compute `sort_key = (event << 29) | (mat_id << 8) | energy_bucket`. Run a **GPU 8-bit/pass Blelloch/OneSweep Radix Sort** over active particle indices (pure WGSL, 4 passes for 32-bit key) + prefix-sum counts per `EventKind`.
     2. `advance_step`: Coherent warps sample free-flight $\Delta s = -\ln\xi / \Sigma_{\text{maj}}$ (Woodcock) or surface distance `d_surf` (`f32` BVH4 traversal + shifted-origin `f64`-emulated / double-double residual quadric step).
     3. `cross_surface`: Update `cell_id`, handle reflective/periodic/vacuum boundary conditions, and trigger surface-crossing tallies.
     4. `collide_reaction`: Particles sorted by `(mat_id, energy_bucket)` hit the **same nuclide & energy bin** across a 32/64-thread warp, maximizing L1/texture cache hit rate on `EnergyHashIndex` & reaction blocks!
     5. `tally_reduce`: **Workgroup-Shared-Memory Privatization**—each workgroup maintains a local `var<workgroup> local_bins: array<atomic<u32>, 256>` using portable `atomic<u32>` CAS (`f32` bitcast compare-exchange loop) or split `u32` hi/lo fixed-point accumulators as the default across Metal, WebGPU, and Vulkan (gating native `atomic<u64>` behind `wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS`), flushing once per workgroup to global tally buffers to cut global atomic contention by $32\times$–$64\times$.
3. **Mitigating `f32` Precision Loss on GPUs Without Native `f64` Throughput**:
   - Retain `llevant-gpu`'s `pack::recentred_shapes` and outward-rounded `Bvh4` AABBs (`pack::round_out`).
   - For high-aspect-ratio grazing hits or thin CAD triangles where `f32` leaks, apply **one-step Newton-Raphson residual refinement using Dekker/TwoSum double-float (`vec2<f32>` hi+lo)** strictly on the final candidate intersection—achieving ~48 bits of mantissa precision at <10% ALU overhead versus `1/32` native `f64` GPU penalty.

### 6.3 Parallelism, Determinism & Cluster Scaling

1. **NUMA-Aware Work-Stealing (`rayon` + `mimalloc`)**:
   - Continue using `mimalloc::MiMalloc` globally; pin `rayon` worker pools on multi-socket nodes (`hwloc`/affinity option) and allocate thread-local scratch arenas (`Tracer`, `ParticleBankSoa`, `TallyAccumulator`) per NUMA domain.
2. **Thread- & Backend-Invariant Reproducibility**:
   - Replace sequential PRNGs with `Philox4x32-10` keyed by `(run_seed, history_id, step_ctr, bounce_ctr)`. Any particle history $h$ generates the exact same random stream whether executed on 1 CPU thread, 128 CPU threads via `rayon`, 10,000 GPU lanes, or across 64 MPI ranks; CPU (`f64`) execution produces bit-identical tallies across thread/rank counts, while GPU (`f32`) transport is statistically validated within $1\sigma$ against the CPU `f64` reference.
   - Reduce tallies via deterministic batch-index ordering (`u128` fixed-point or compensated tree reduction).
3. **MPI + Collision-Based Domain Decomposition (`CDD` — `SRP-08` / `SCL-02`)**:
   - Phase 4 multi-node engine introduces `llevant-mpi` (`mpi` feature):
     - **Replicated-Geometry Mode**: Standard batch-parallel MPI for models fitting in node RAM (<16 GB).
     - **Collision-Based Domain Decomposition (`CDD`)**: When >100k burnable depletion materials exceed single-node RAM, partition material/depletion zones across ranks (`mat_id % n_ranks` or graph-partitioned locality), keeping geometry lightweight on all ranks while asynchronously migrating colliding particle states (`MPI_Isend`/`MPI_Irecv` batched buffers) to the rank owning `mat_id`.

### 6.4 Memory Footprint & I/O Pipeline

1. **Slashing `migjorn` 2 GB Peak RAM on 378 MB Decks**:
   - Currently `migjorn::parse` allocates a full token/AST tree (~2 GB) and `llevant_scene::source::Source` parses the file a *second* time (`baseline_audit.md §5`).
   - Fix: Implement **Streaming Chunked Parsing + `SourceSpanIndex`**: stream line-buffered blocks (Cells -> Surfaces -> Data) directly into `ModelIr` arenas and record `(card_kind, id, byte_offset, byte_len)` slices against a single `memmap2::Mmap` of the input file! `Source` then slices card text directly from the mmap in **O(1) bytes** instead of re-parsing 378 MB. Target: **<450 MB peak RAM** (4.5× reduction).
2. **Zero-Copy Nuclear Data & Model Persistence (`rkyv` + `memmap2`)**:
   - Replace `bitcode` deserialization allocation overhead on multi-GB nuclear data libraries with **`rkyv` 0.8 aligned zero-copy archives** backed by `memmap2::Mmap`: loading a 4 GB ENDF/ACE library takes **<5 ms** (`mmap` page fault on demand) and shares physical RAM pages across multiple processes on the same node!
3. **Apache Arrow IPC & Parquet for Tallies (`llevant-tally`)**:
   - Store and stream tally results as Apache Arrow `RecordBatch`es (`arrow-array`, `arrow-ipc`, `parquet`). Enables zero-copy transfer across Rust $\to$ Python (`pyarrow` / `polars` / `pandas`), `llevant serve` $\to$ `web_viewer.py` (Arrow Flight / HTTP Arrow IPC), and out-of-core multi-million-voxel mesh queries via DuckDB/Parquet.

---

## 7. Target Interface Specifications (R2 + R3)

Every target interface explicitly closes the gaps identified in `baseline_audit.md` and `ecosystem_survey.md`.

### 7.1 Rust Core API (`llevant`, `llevant-ir`, `llevant-nucdata`, `llevant-transport`, `llevant-tally`)

- **Ergonomic Builder & Multi-Format Entrypoints**:
  - `llevant_ir::ModelIr::from_path(path)` / `from_bytes(bytes, FormatHint)` directly parses MCNP, OpenMC XML, TRIPOLI-4, DRAGON5, GDML, or DAGMC `.h5m` without MCNP-text round-tripping.
  - `ModelIr::to_mcnp(&self)` / `to_openmc_xml(&self)` / `to_gdml(&self)` for bidirectional export.
- **Unified `GeometryKernel` Trait**:
  ```rust
  pub trait GeometryKernel: Send + Sync {
      fn locate(&self, pts: &[[f64; 3]], out_cells: &mut [u32], out_mats: &mut [u32]);
      fn distance_to_boundary(&self, origins: &[[f64; 3]], dirs: &[[f64; 3]], cells: &[u32], out_d: &mut [f64], out_surf: &mut [u32]);
      fn trace(&self, lines: &[Line], depth: Depth) -> Result<Tracing>;
  }
  ```
- **Simulation & Coupling API**:
  - `TransportRunner::new(kernel, nucdata, settings).with_tallies(...).with_vr(ww_mesh).run(progress)` returning `SimulationResult` (`k_eff`, `Tallies` as Arrow `RecordBatch` / HDF5 `StatePoint`, `PtracStream`).
  - Zero-panic `Result<T, LlevantError>` throughout all crates.

### 7.2 Python API (`llevant-py` — `import llevant`)

Extends existing PyO3 `Model`/`Field`/`Volumes`/`_openmc.py` without breaking any existing signature:
1. **Zero-Copy Arrow & Polars/NumPy Interop**:
   - `Tally.to_arrow() -> pyarrow.Table`, `Tally.to_polars() -> pl.DataFrame` (via Arrow C Data Interface `arro3`/`pyarrow` capsule—zero copy!), `Model.statepoint(path)` native Rust HDF5 reader (removing `h5py` hard dependency).
2. **`openmc.lib`-Style Live In-Memory Control (`llevant.LiveSession`)**:
   - `session = model.live_session(nucdata=...)`
   - `session.set_temperature(cell_or_mat_id, temp_k)` (uses WMP or Doppler rejection), `session.set_density(cell_id, g_cm3)`, `session.run_batches(n)`, `session.tally_values(tally_id)` (zero-copy NumPy view)—enabling `Cardinal`-style MOOSE/CFD multiphysics Picard loops in pure Python with zero disk I/O.
3. **CAD, Plasma & Material Ecosystem Bridges**:
   - `llevant.from_dagmc("blanket.h5m")`, `llevant.from_cad(cadquery_or_step_path, heal=True)` (bridging `cad_to_dagmc`/`paramak`), `llevant.Material.from_nmm(...)` (`neutronics_material_maker` adapter), `llevant.Source.from_plasma(...)` (`openmc_plasma_source` adapter + `plot_source()`).
4. **Depletion & UQ Helpers**:
   - `llevant.deplete(session, schedule, chain_xml, cram_order=16)`, `llevant.r2s_shutdown_dose(...)`.

### 7.3 CLI (`llevant-cli` — `llevant` binary)

Retains all 14 existing subcommands (`info`, `bbox`, `locate`, `trace`, `slice`, `plots`, `render`, `materials`, `convert`, `lost`, `fingerprint`, `view`, `volume`, `check`) and adds:
- **`llevant convert`**: Bidirectional `--to mcnp|openmc|gdml` via `llevant-ir` (replacing one-way MCNP-only translation).
- **`llevant serve`**: Starts persistent daemon (`--unix <sock>`, `--tcp 127.0.0.1:8056`, `--http`, `--mcp`, `--lsp`, `--preload <model>`).
- **`llevant run <model>`**: Executes transport (`--mode mc|random-ray|cmfd`, `--batches`, `--particles`, `--gpu`, `--xs <path>`, `--opti 1..4`, `--wwg`, `--dump <ptrac.h5>`).
- **`llevant replay <dump.h5>`**: Instantaneous PHITS `icntl=12`-style post-transport re-tallying with new energy/mesh/angle filters without re-running physics.
- **`llevant deplete <model>`**: Runs CRAM depletion / activation (`--schedule`, `--chain`, `--r2s`).
- **`llevant data <pull|index|inspect>`**: Reproducible nuclear data library downloader/indexer (`openmc_data_downloader` + `OBIWAN`-style `.h5`/`.f71`/depletion archive inspection to JSON/CSV).
- **`llevant volume` / `check` enhancements**: `--mesh-voxels <spec>` (voxel-wise volumes inside a mesh), `--qmc` stratified sampling, `--cache-shader` warm-up.

### 7.4 Desktop GPU Viewer (`llevant-viewer`)

Closes every viewer gap from `docs/architecture.md` and `baseline_audit.md`:
1. **Integrated QA & Diagnostics Panels**:
   - Expose **`Lost Particle` Inspector** (`llevant lost` GUI: paste MCNP/OpenMC log or click point $\to$ auto-frame 3D + 3 orthogonal slice views + cell/surface highlight).
   - Expose **`Check & Mass Balance` Panel** (`llevant check --plots` interactive defect browser jumping camera to top overlap/gap boxes; `volume --mass` sortable table with `mass_g ± mass_sigma_g` + CSV/JSON export).
2. **Native Tallies, Filters & Fields Parity**:
   - Native Rust HDF5 `StatePoint` loader (`hdf5-metno` inside `llevant-scene`) with interactive **Energy / Particle / Nuclide / Score / Error / `CellInstance` / `Distribcell`** dropdowns matching `_openmc.py`.
   - **Fields inside `--universe` submodels**, **voxel-boundary outlines** on 2D slices, **hiding/selecting by cell instance and by field value threshold slider** (e.g., hide cells below flux $10^{12}$), and **Weight-Window (`WWG`) mesh overlays** (FLUKA Flair style).
3. **Editing & Visualization Polish**:
   - Structured **numeric spinbox/vector fields for surface & `TRn` coefficients** in the `inspector` card panel alongside text editing.
   - **Faded occluded pins** (depth-tested alpha pass for pins behind solid geometry).
   - **DAGMC `.h5m` facet/edge wireframe debug mode** + **4D particle track / `PTRAC` animated playback** (`PHIG-3D` style) + **source distribution point-cloud overlay** (`openmc_source_plotter` style).

### 7.5 `web_viewer.py` Web Studio (Daemon Client + Parity + WebGPU/WASM)

Eliminates every subprocess and feature gap in `web_viewer.py`:
1. **Architecture Migration (`Subprocess -> Persistent Daemon Client`)**:
   - Replace per-request `subprocess.run([LLEVANT_BIN, ...])` with a persistent connection to **`llevant serve`** (auto-spawned if not running, communicating over Unix socket / localhost HTTP+WebSocket + Arrow IPC).
   - **Result**: Orbit dragging drops from ~500–2000 ms/frame (cold process + reload) to **<16–30 ms/frame** over persistent VRAM `GpuModel` streaming WebP/PNG or raw quantized depth/hit tiles.
   - Invalidate caches automatically via daemon model `blake3` hash + file `mtime` rather than static parameter MD5.
2. **Full Feature & Statepoint Parity**:
   - Replace `convert_openmc_statepoint_to_csv`'s partial filter logic (`cell`/`material` only) with `llevant serve`'s native Rust/Arrow statepoint reader supporting **`cellinstance`, `distribcell`, `mesh` (Regular/Rectilinear/Cylindrical), `score`, `nuclide`, `energy`, `particle`, and `error`**.
   - Add Web Studio tabs & endpoints for: `POST /api/lost` (`lost` particle debugger), `POST /api/convert` (download converted MCNP/OpenMC/GDML), `GET /api/plots` (`plots.xml` gallery), `GET /api/volume` with **`--mass` mass balance**, and `POST /api/check_report` (`check --plots` downloadable QA bundle).
3. **Browser WebGPU / WASM Client Option (`docs/architecture.md` Later Horizon)**:
   - Compile `llevant` (geometry slice/pick) to `wasm32-unknown-unknown` and `llevant-gpu` WGSL shaders directly to **browser WebGPU** (`wgpu` supports WebGPU target natively!) so remote cluster models can either stream rendered frames from `llevant serve` OR download lean `Model::export_lean` buffers for local 60 fps browser WebGPU orbiting.

### 7.6 Candidate New Interfaces (`llevant-serve`: Daemon, MCP & LSP)

1. **Persistent Query & Simulation Daemon (`llevant serve`)**:
   - Single multi-tenant Rust process (`tokio` + `rayon` + `wgpu`) holding `Arc<ModelIr>`, `Arc<Model>`, `Arc<GpuModel>`, `Arc<NucDataLib>`, and active `SimulationRunner`s.
   - Protocols: Binary/JSON-RPC over Unix Domain Socket (for CLI/Python/`web_viewer.py`), HTTP/WebSocket + Arrow IPC (for Web Studio / remote HPC tunneling), and `rmcp` stdio/SSE.
2. **Model Context Protocol (`MCP`) Server (`rmcp` crate)**:
   - Exposes structured tools for AI neutronics assistants & automated QA pipelines:
     - `inspect_model(path)` $\to$ diagnostics, material mass summary, extent, universe tree.
     - `diagnose_lost_particle(model_id, x, y, z, dir?)` $\to$ exact cell/gap/overlap path + nearest bounding surfaces + base64/path 3-view slice PNGs.
     - `check_overlaps(model_id, h, region?)` $\to$ ranked defects + bounding boxes.
     - `query_tally(tally_id, filter_expr, top_k)` $\to$ statistical summary, peak voxel/cell, relative errors > 10% warnings.
     - `diff_models(model_a, model_b)` $\to$ fingerprint diff listing exact changed cells/universes.
3. **Language Server Protocol (`LSP`) & IDE Extension (`tower-lsp`)**:
   - Unified LSP server (`llevant serve --lsp`) supporting **MCNP**, **OpenMC XML**, **TRIPOLI-4**, and **DRAGON5 `.x2m`**:
     - **Real-time Diagnostics**: Undefined surfaces/materials/universes, DRAGON5 `CARCEL`/`CAR2D` `MIX` count mismatches, unclosed parentheses, invalid macrobody facets, and background stochastic overlap hints.
     - **Go-to-Definition / Find References**: Jump between cell boolean expressions (`-12 34`), surface cards, `TRn`, `FILL=u`, and `Mn` material cards.
     - **Hover Documentation**: Hovering a surface ID shows equation & bounding box; hovering `Mn` or ZAID shows isotopic composition, mass density, PBR category, and available nuclear data temperatures; hovering a cell shows compiled volume/mass if cached.

---

## 8. Prioritized Phased Implementation Roadmap (R3)

### 8.1 Dependency Graph

```text
Phase 0 (Quick Wins & Parity Debt)
   ├──► Phase 1 (llevant-ir + SIMD CPU + PipelineCache + llevant serve + web_viewer.py Parity)
   │       ├──► Phase 2 (llevant-cad DAGMC/Tet + Native HDF5/Arrow Tallies + Viewer Parity + LSP/MCP)
   │       │       └──► Phase 3 (llevant-nucdata + CPU/GPU Event Transport + RandomRay)
   │       │               └──► Phase 4 (llevant-vr CADIS/RMX + llevant-deplete CRAM/R2S + MPI CDD)
```

### 8.2 Phased Deliverables, Verification Gates & Benchmarks

#### Phase 0: Baseline Debt Closure & Quick-Win Parity (Weeks 1–3)
- **Scope**:
  - Fix the 3 verified loader caveats: implement DRAGON5 `SPLITR`/`SPLITX`/`SPLITY`/`SPLITZ`/`SECT`/`TURN` meshing (`crates/llevant-dragon/src/lib.rs:422-426`); respect TRIPOLI-4 `BOUNDARY_CONDITION`/`LIMITE` (`crates/llevant-tripoli/src/lib.rs:1574`); add distributed-material warning/metadata preservation in `llevant-openmc`.
  - Fix CPU `first_hits` sparse-world stepping (`sparse.i`) using BVH/unit-box span jumping instead of blind 16 cm $\times 2^k$ doubling (`crates/llevant/src/hit.rs`).
  - Expose `llevant lost`, `llevant check --plots`, `llevant plots`, and `volume --mass` in `llevant-viewer` UI panels.
  - Add tangent-line / near-coincident surface regression tests (`docs/architecture.md:265`).
- **Acceptance Gates**:
  - `cargo test --workspace` + `tools/mcnp-oracle` pass 100%; `llevant-bench` shows $\ge 5\times$ speedup on `sparse.i` CPU `first_hits`; `SPLITR` DRAGON5 tests match analytical concentric ring areas.

#### Phase 1: Native IR, CPU SIMD, Shader Caching & Persistent Daemon (`llevant serve` + `web_viewer.py`) (Weeks 4–10)
- **Scope**:
  - Introduce **`llevant-ir`**: direct compilation from OpenMC XML, TRIPOLI-4, DRAGON5, and streaming zero-copy `migjorn` MCNP (`<450 MB` peak RAM on 378 MB deck, removing `Source` double-parse).
  - Refactor `Shape` AoS (112 B) into `PlaneSoa` (32 B/plane) + `QuadricSoa` + flat `ConeData`/`TorusData`; build **`Bvh4`/`Bvh8`** with `std::arch` (`avx2`/`neon`) 4-wide/8-wide ray-packet tracing; optimize level-1 filler evaluation.
  - Upgrade `VolumeGrid` (`volumes_to`) with stratified QMC Halton/Sobol probes + converged-box masking.
  - Enable `wgpu::PipelineCache` disk persistence in `llevant-gpu` (`8–30 s -> <150 ms` cold startup).
  - Build **`llevant-serve`** daemon (`Unix/HTTP/WS` + `rmcp` MCP MVP) and migrate `web_viewer.py` from subprocess-per-request to a persistent daemon client with full `_openmc.py` statepoint filter parity (`cellinstance`, `distribcell`, `mesh`, `score`/`nuclide`/`energy`/`particle`/`error`) + `lost`/`convert`/`plots`/`--mass`/`check --plots`.
- **Acceptance Gates**:
  - Peak RSS on 378 MB reference deck drops from ~2 GB to `<500 MB`; `volumes_to` on 425k-cell reference model converges in `<10 min` (down from ~2 hours); `Model::slice` and `VolumeGrid` achieve $\ge 2.5\times$ CPU speedup (`tools/bench/work-baseline.json` updated); `web_viewer.py` orbit latency `<30 ms/frame`; zero GPU/CPU pixel regression (`llevant-bench gpu-check`).

#### Phase 2: Watertight CAD/DAGMC, Unstructured Meshes, Native Arrow/HDF5 Tallies & Viewer/LSP Parity (Weeks 11–18)
- **Scope**:
  - Introduce **`llevant-cad`**: pure-Rust MOAB `.h5m` (`hdf5-metno`) DAGMC reader, `f32`-BVH4 + `f64`-watertight Plücker ray-triangle kernel (`double-down` architecture), conformal vertex-welding healer, `TetMesh` $O(1)$ face-adjacency stepper, `VoxelGrid` DDA, and `GDML` adapter.
  - Introduce **`llevant-tally`**: native Rust HDF5 `StatePoint` reader/writer, `Arrow` `RecordBatch` + `Parquet` streaming, `MCTAL`/`meshtal`/`USRBIN` statistical merging, zero-copy `arro3`/`pyarrow` Python bridge.
  - Complete `llevant-viewer` parity: native HDF5 statepoints, instance/field-value hiding, voxel slice outlines, `--universe` fields, numeric `TRn` card editors, faded occluded pins, DAGMC facet/edge debug, and optional WebGPU/WASM browser build.
  - Ship **LSP server** (`llevant serve --lsp`) with diagnostics, hover, and go-to-definition for MCNP/OpenMC/TRIPOLI-4/DRAGON5, plus `from_dagmc`/`from_cad`/`paramak` Python helpers.
- **Acceptance Gates**:
  - Watertight DAGMC test suite (nested toroidal/tokamak `.h5m` with 0 lost particles across $10^7$ rays); ray-triangle throughput within $1.3\times$ of Intel Embree `double-down`; Python `Tally.to_polars()` zero-copy transfer benchmarked at $>5\text{ GB/s}$.

#### Phase 3: Nuclear Data Engine & Maximum-Performance CPU/GPU Transport (Weeks 19–28)
- **Scope**:
  - Introduce **`llevant-nucdata`**: zero-copy `rkyv` + `mmap` ENDF/ACE/HDF5/GNDS parser, `EnergyHashIndex` + `XsOptiTier` (`Opti1..4`), Windowed Multipole (`WmpTable` + Faddeeva $W(z)$), URR probability tables (`PURR`), $S(\alpha,\beta)$ thermal scattering, `NCrystal` FFI feature, and `llevant data pull` downloader.
  - Introduce **`llevant-transport`**: `Philox4x32-10` counter-based PRNG; CPU SIMD + `rayon` event/history MC (`n`, `p` coupled physics, Woodcock/surface hybrid, `F4/F5/F6/F8/FMESH/USRBDX`); **`wgpu` GPU event-based transport** (SoA particle banks, GPU radix sort by `(event, mat_id, energy_bucket)`, shared-memory tally privatization, Dekker double-float surface refinement); `RandomRay` SoA solver & `CMFD` acceleration; `OpenMC`-compatible `LiveSession` (`openmc.lib`-style) in `llevant-py`; `llevant run` & `llevant replay` (`dumpall`).
- **Acceptance Gates**:
  - **ICSBEP / MCNP6 / OpenMC V&V Suite**: $k_{\text{eff}}$ within $\pm 50\text{ pcm}$ ($1\sigma$) and tally bin agreement $<1\%$ on pincell, Godiva, Jezebel, BEAVRS, and ITER-like fusion benchmarks;
  - **Determinism**: Bit-identical tallies across `1` vs `64` CPU threads;
  - **GPU Speedup**: $\ge 5\times$ histories/sec on discrete GPU vs 16-core CPU on continuous-energy event-based benchmark.

#### Phase 4: Automated Variance Reduction, CRAM Depletion, UQ & Exascale Cluster Scaling (Weeks 29–38)
- **Scope**:
  - Introduce **`llevant-vr`**: internal Response Matrix (`RMX`) & coarse $S_N$ forward/adjoint solver generating consistent `CADIS` / `FW-CADIS` biased sources and `WeightWindowMesh` (with live `llevant-viewer` / `web_viewer.py` overlay).
  - Introduce **`llevant-deplete`**: Order-14/16 sparse `CRAM` Bateman solver, `ORIGEN`/`DCHAIN`-style decay chains, Rigorous Two-Step (`R2S`) shutdown dose rate workflow, `IFP` adjoint kinetics ($\beta_{\text{eff}}, \Lambda$), `KSEN`/`PERT` sensitivity, `SANDY`-compatible covariance UQ runner, and `OBIWAN`-style JSON archive export.
  - Enable **`llevant-mpi` + Collision-Based Domain Decomposition (`CDD`)** for $>100\text{k}$ depletion-zone full-core burnup and `Cardinal`-style in-memory Arrow/C-ABI multiphysics coupling.
- **Acceptance Gates**:
  - `CADIS`/`FW-CADIS` achieves $\ge 100\times$ Figure-of-Merit speedup on deep-penetration concrete/bioshield benchmark;
  - `CRAM-16` matches `openmc.deplete` / `Serpent 2` isotopic inventories to $<10^{-5}$ relative error across 1000+ nuclides;
  - `CDD` weak-scales across $\ge 8$ MPI nodes on a 100k-zone depleted core without exceeding per-node RAM budget.

### 8.3 Technical Risks & Mitigations

| Risk | Severity | Mitigation Strategy |
|---|---|---|
| **`wgpu` driver portability & `f32` precision leaks on GPU transport** | High | Keep GPU transport opt-in behind CPU exact validation (`llevant-bench` V&V); apply shifted-origin `recentred_shapes` + two-float Dekker residual refinement on candidate intersections; validate every benchmark against `f64` CPU `Philox` reference. |
| **Shader compilation or memory exhaustion on 1M-cell + 300-nuclide models** | High | `wgpu::PipelineCache` disk caching + quantized stack buckets; `XsOptiTier::Opti2Hashed` default + `WMP` avoiding multi-temperature RAM explosion; `CDD` MPI partitioning for >100k burnable zones. |
| **Multi-code translation loss (`IR` vs native deck round-trip)** | Medium | Retain `SourceSpanIndex` pointing to original `mmap` text for lossless card edits; only translate explicitly when `llevant convert` is invoked. |
| **Scope creep breaking geometry/QA stability** | Medium | Enforce strict crate boundaries (`llevant` geometry kernel stays dependency-free of physics/UI); gate every PR on `tools/mcnp-oracle`, `tests/vs_cpu.rs`, and `work-baseline.json` CI checks. |
