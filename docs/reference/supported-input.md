# Supported MCNP and OpenMC input

llevant reads MCNP inputs through [migjorn](https://crates.io/crates/migjorn), and
follows MCNP 6.2 wherever the two could differ: it has been checked point by point
against MCNP itself ([how](../mcnp-behaviour.md)). It also reads OpenMC XML models
directly in pure Rust (`llevant-openmc`).

## Geometry

- **Surfaces**:
    - P (4 coefficients or 3 points), PX/PY/PZ, SO/S/SX/SY/SZ;
    - C/X/C/Y/C/Z and CX/CY/CZ;
    - K/X/K/Y/K/Z and KX/KY/KZ (one or two sheets);
    - SQ, GQ;
    - TX/TY/TZ, including degenerate A = 0;
    - X/Y/Z by points;
    - periodic surfaces (`n -m type ...`).
- **Macrobodies** RPP, BOX, SPH, RCC, TRC, RHP/HEX, REC, ELL, WED and ARB, as whole
  bodies or facets `n.k`.
- **Transforms and cell forms**:
    - `TRn`/`*TRn` (3, 9, 12 or 13 entries, `m = ±1`);
    - inline `FILL` transforms and `*FILL`;
    - `TRCL`/`*TRCL`, inline or by number;
    - `U` and nested fills, at any depth;
    - complements `#n` and `#( )`;
    - `LIKE n BUT`.
- **Lattices** `LAT=1` (any parallelepiped) and `LAT=2` (hexagonal prisms), 2-D or
  3-D:
    - filled uniformly or by a fill array, with `nR` repeats and per-element
      transforms;
    - elements may be filled by the lattice cell itself.
- **MCNP's own restrictions** (skewed one-sheet cones and tori, macrobodies with
  vectors that are not at right angles…) make the cell unsupported, as MCNP would
  refuse it. `llevant info` lists them; points that only such a cell could hold are
  labelled `unsupported`.

## Not read

- **Cell parameters on data cards.** `IMP`, `U`, `FILL`, `TRCL` and `LAT` are read
  from the cell cards only. A data card that gives them for every cell
  (`IMP:N 1 1 1 0`) is reported as a diagnostic, and the geometry may then differ from
  MCNP's: put these parameters on the cell cards.
- **`READ` cards.** The files they name are not read; this too is reported as a
  diagnostic. Assemble the input into one file first.

`llevant info model.i` lists every diagnostic with its line.

## Results files (`--field`)

| File | What is drawn |
|---|---|
| CSV | values per cell, material, cell instance or repeated-structure path |
| MCNP meshtal (FMESH) | the mesh, exactly as the FMESH card in the input places it, `TR=` included |
| D1SUNED cuvmsh (`out=cuv`) | each voxel's flux |
| MCTAL | F4, F6, F7 and F8 cell tallies, per cell or per cell instance |

Not drawn yet: surface tallies, point detectors, and each cell's own flux inside a
cuvmsh voxel. Details: [Fields](../guide/cli.md#fields).

## OpenMC XML input

llevant translates OpenMC XML models into MCNP geometry natively (`llevant-openmc`)
with zero OpenMC runtime dependency:

- **Files and directories**: `model.xml`, `geometry.xml` (automatically discovering a
  sibling `materials.xml`), or a directory containing either, as well as in-memory
  XML strings via `Model::from_text` and Python `llevant.from_text`.
- **Surfaces**: `x-plane`, `y-plane`, `z-plane`, `plane`, `sphere`, `x-cylinder`,
  `y-cylinder`, `z-cylinder`, `x-cone`, `y-cone`, `z-cone`, `x-torus`, `y-torus`,
  `z-torus`, and `quadric`, with `vacuum`, `reflective`, `white`, and `periodic`
  outer boundary conditions.
- **Cells, universes, and transforms**: boolean regions (intersection, `|` union,
  `~` complement, parentheses), nested `universe`/`fill`, `translation`, and 3-angle
  Euler or 9-entry `rotation`.
- **Lattices**: 2-D and 3-D rectilinear `<lattice>` and hexagonal `<hex_lattice>`
  (`orientation="y"` and `"x"`, `<outer>` padding).
- **Materials and labels**: nuclide `ao`/`wo` and elemental compositions (`g/cm3`,
  `kg/m3`, `atom/b-cm`, `atom/cm3`, `sum`), plus cell `name="..."` attributes exposed
  to `label "..."` queries.
- **Results (`read_openmc_tally`)**: in Python ([the Python guide](../guide/python.md)),
  `llevant.read_openmc_tally(model, "statepoint.h5")` loads `cell`, `material`, and
  `mesh` (`RegularMesh`, `RectilinearMesh`, `CylindricalMesh`) tallies directly from
  OpenMC HDF5 statepoints (when `h5py` is installed).

Examples: `docs/examples/openmc/fusion_hts_blanket.xml`, `pwr_pin_lattice.xml`, and
`vver_hex_assembly.xml`.

## Labels

Every point is given a status:

| Status | Means |
|---|---|
| `inside` | in exactly one cell |
| `gap` | in no cell: MCNP would lose a particle here |
| `overlap` | in more than one cell |
| `on-surface` | on a surface, within the tolerance (1e-8 cm by default) |
| `unsupported` | in no cell but possibly an unsupported one (see the diagnostics) |
| `too-deep` | universes nested deeper than llevant follows (64 levels) |
