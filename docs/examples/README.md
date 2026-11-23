# Llevant Use-Case Examples

Minimal input fixtures in `docs/examples/` combined with runnable end-to-end recipes in `crates/llevant-py/examples/use_cases.py`.

## Showcase Video

- [Use-Case Video Walkthrough (`llevant_use_cases.mp4`)](../img/llevant_use_cases.mp4) — 20-second 720p showcase rendered directly from live Llevant queries across OpenMC, TRIPOLI-4, DRAGON5, and MCNP. Regenerate anytime with:

```sh
python crates/llevant-py/examples/make_showcase_video.py
```

## Input Fixtures

| Code | File | Highlights |
|---|---|---|
| OpenMC XML | `openmc/fusion_hts_blanket.xml` | Tokamak sector, diagnostic port, labelled components |
| OpenMC XML | `openmc/pwr_pin_lattice.xml` | 3x3 rectilinear lattice + guide tube |
| OpenMC XML | `openmc/vver_hex_assembly.xml` | 3-ring hexagonal fuel assembly (`LAT=2`) |
| OpenMC XML | `openmc/distributed_pin_plots.xml` | Distributed-material cell warning + `<plots>` slice definitions |
| TRIPOLI-4 | `tripoli4/pwr_pin_cell.t4` | `ECRASE` priority subtraction, `COMPOSITION`, `BOUNDARY_CONDITION` |
| TRIPOLI-4 | `tripoli4/fusion_shield_box.t4` | `QUAD`, cones, rotated planes, nested shielding |
| DRAGON5 `GEO:` | `dragon5/pwr_3x3_assembly.x2m` | `CAR2D` lattice with inline `::: PIN`/`GT` `CARCEL` sub-geometries |
| DRAGON5 `GEO:` | `dragon5/depletion_pin_splitr.x2m` | Equal-area radial `SPLITR` + axial `SPLITZ` sub-meshing |
| MCNP | `cask.i`, `cask-v2.i`, `core.i` | Spent-fuel casks (for diffing) and multi-level reactor core |

## Use-Case Matrix

Run any Python recipe via:

```sh
python crates/llevant-py/examples/use_cases.py --case <name> --out out_dir
```

| Use Case | CLI One-Liner | Python `--case` | Desktop / Web UI |
|---|---|---|---|
| **Multi-code convert & verify** | `llevant convert docs/examples/dragon5/depletion_pin_splitr.x2m --verify` | `multicode_load_and_convert` | `llevant view <input>` / `/api/convert` |
| **Geometry QA & defect plots** | `llevant check docs/examples/openmc/fusion_hts_blanket.xml --h 5 --plots qa_out/` | `qa_check_and_plots` | Check workspace (Ctrl+2) / **Run QA Check** |
| **Lost-particle triage** | `llevant lost docs/examples/tripoli4/pwr_pin_cell.t4 0 0 0 --plots lost_out/` | `lost_particle_triage` | Problems -> **Lost particle** / `/api/lost` |
| **Volumes & mass inventory** | `llevant volume docs/examples/openmc/vver_hex_assembly.xml --h 0.8 --by material --mass` | `volumes_and_mass` | Volumes panel / `/api/volume?mass=1` |
| **OpenMC statepoint overlays** | `llevant slice docs/examples/openmc/fusion_hts_blanket.xml --plane z=0 --field sp.csv` | `openmc_statepoint_tallies` | Field tab / `/api/tallies` + `statepoint.h5` |
| **Distributed materials & `<plots>`** | `llevant plots docs/examples/openmc/distributed_pin_plots.xml -O plots_out/ --outlines` | `distributed_materials_and_plots` | Compiler diagnostics in Problems panel |
| **Ray tracing & chord lengths** | `llevant trace docs/examples/openmc/fusion_hts_blanket.xml --from -250,0,0 --to 250,0,0` | `ray_trace_and_chord_lengths` | Ruler tool (`M`) / Trace tab |
| **Model diff & fingerprints** | `llevant fingerprint docs/examples/cask.i --diff docs/examples/cask-v2.i` | `model_diff_and_fingerprints` | Fingerprint tab (`/api/fingerprint`) |
| **3D cutaway & label ghosts** | `llevant render model.xml --clip 0,0,0,0,-1,0 --ghost 'label "Tungsten"' -o view.png` | `render_cutaway_ghost_labels` | 3D viewport cutaway & Ghost inputs |
| **REST API / Web Studio** | `python web_viewer.py --port 8055` | `web_studio_api` | Browser at `http://127.0.0.1:8055` |
