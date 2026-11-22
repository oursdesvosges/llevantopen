# The `llevant` command

The `llevant` command comes with the Python package (as `python -m llevant`) and with
the downloaded programs ([Install](../install.md)). Every subcommand has `--help`.
New to it? Start with [Getting started](../getting-started.md).

```sh
llevant info   model.mcnp                     # sizes, extent, diagnostics
llevant locate model.mcnp 0 0 0  120 -3.5 40  # or --points file, --format csv|json
llevant trace  model.mcnp --from -100,0,0 --to 100,0,0   # or --lines file (x0 y0 z0 x1 y1 z1 per line)
llevant slice  model.mcnp --plane z=37 --px 1200 --colour material -o slice.png
llevant render model.mcnp --clip 0,0,0,0,-1,0 --hide 12,13 -o view.png  # cut away y > 0
llevant render model.mcnp --colour realistic -o real.png   # steel, concrete, water...
llevant materials model.mcnp                  # what each material looks like, and why
llevant fingerprint model.mcnp -o fp.json
llevant fingerprint old.mcnp --diff new.mcnp  # cells/universes changed, new, removed, renumbered (exit 1 if any)
llevant volume model.mcnp --h 2 --by material # volume ± sigma per material
llevant check  model.mcnp --h 2               # exit status 1 if gaps or overlaps
llevant render model.mcnp --universe 20130 -o filler.png  # a filler on its own (any command)
llevant slice  model.mcnp --plane z=37 --level 0 -o envelopes.png   # MCNP's LEVEL 0
llevant render model.mcnp --ghost "c 300" -o shells.png   # cell 300 as a shell over its filler
llevant slice  model.mcnp --plane z=37 --field heat.csv -o heat.png     # values per cell
llevant slice  model.mcnp --plane z=37 --field heat.csv --outline-by material -o heat.png
llevant render model.mcnp --field run.msht --tally 14 --log -o flux.png # an FMESH tally
llevant view   model.mcnp                     # the viewer (feature `viewer`)
llevant view                                  # its start screen: open a model from it
llevant check  model.xml --h 1.0              # OpenMC XML models work with every command
```

## Regions, labels and levels

- **Framing.** Slices and renders frame the model's contents unless
  `--region x0,y0,z0,x1,y1,z1` is given. `volume` and `check` cover everything but
  the `IMP:N=0` cells.
- **Labels** read like `inside 12 > 304[1,-2,0] > 7`: cell 12 at level 0, lattice
  cell 304 at element (1, -2, 0), and cell 7 inside it.
- **Slices on a surface.** A slice that lies exactly on a surface shows it as
  on-surface (grey); move the plane off round numbers.
- **`--level N`** shows one universe level as MCNP's plotter does: only the cells at
  that level, a filled cell drawn solid.
- **`--outline-by cell|material|universe`** draws a slice's lines between cells (the
  default), or between the materials (void is one) or universes of the deepest
  cells: with a field, the material boundaries are those that explain it. With
  `--level`, lines are between cells.
- **`--ghost RULE`** draws the cells of a rule as shells, with what they hold showing
  through.
- **`--universe U`** opens universe U as the model, in its own frame, with everything
  it holds. It is cut to a box around its contents (or `--cut none`, or a box
  `x0,y0,z0,x1,y1,z1`), and beyond the box lies one void `IMP:N=0` cell. This is
  much faster than hiding everything else, and the parts of a filler that its
  envelopes clip are seen.

## Fields

`--field FILE` colours the cells by numbers. It accepts four kinds of file:

- **A CSV file.** The first column is `cell`, `material`, `instance` (an id, as
  `locate` gives) or `path` (MCNP's repeated-structure notation,
  `12 < 100[1 0 0] < 5`). The second column holds the values, with the name and unit
  as its header (`dose [Sv/h]`).
- **An MCNP meshtal file**, with `--tally`, `--energy`, `--time` and `--error`. The
  mesh is the exact one of the FMESH card in the model's input, `TR=` included.
- **A D1SUNED cuvmsh** (an FMESH with `out=cuv`), with the same options as a meshtal:
  each voxel's flux, made from its cells' fluxes weighted by their share of its volume.
  This is what the mesh would score without `out=cuv`; the errors are only estimates
  (they treat the cells as independent). Its `meshinfo` holds no values and is refused.
  Drawing each cell's own flux inside a voxel is not done yet.
- **An MCTAL file**: F4, F6, F7 and F8 cell tallies, per cell, or per cell instance
  where bins are repeated-structure chains (`(8 < 7[1 0 0] < 3)`) or filled cells.
  - `--bin m=1` chooses a bin along the other dimensions (d, u, s, m, c), and
    `--bin m=sum` adds them up.
  - `--run-input` gives the MCNP input of the run when it is not the model's; its F
    cards say which cells a bin of groups or levels covers.

Surface tallies and detectors are not drawn yet. `--scale lo,hi`, `--log`/`--linear`
and `--opacity` set how the field is drawn, and a colour bar goes under the image.
Cells are a neutral grey under the field, and where it has no value (a grey with a
tint of blue on the grey colour map, whose middle is grey); `--cell-colours-under`
keeps the cells' own colours there instead. A tally that scored nothing in a cell
or voxel gives 0, a value: on a log scale it takes the scale's low end's colour.
A fitted scale's ends are rounded out to decades (log) or a 1-2-5 step
(`--exact-ends` keeps the values' own). `--threshold lo,hi` draws only values within
it (either side may be empty: `--threshold 1e3,`), the others as no value;
`--below-colour '#ffffff'` and `--above-colour '#e61e1e'` colour values outside the
scale instead of its ends.

The colour bar under the image is the viewer's legend: `--ramp viridis|turbo|inferno|grey`
and `--bands N` (discrete, of equal decades on a log scale) set the colours;
`--ticks few|many|N|edges` the ticks (few and many leave out labels that would
overlap; a number gives exactly so many, evenly spaced, and `edges` one at each edge
between the bands, every one labelled, on two rows if they need it);
`--numbers auto|powers|decimal` and `--digits N` the numbers, all of a bar in one
notation; `--legend-title` and `--legend-text PX` its title and text size (scaled with
the width by default). The 3D view's "copy `llevant render` command" passes these on.

```sh
llevant slice model.mcnp --plane z=0 --field run.msht --tally 14 \
    --bands 12 --ticks edges --numbers powers --outline-by material -o flux.png
```

## Realistic images

`--colour realistic` (in `slice`, `render`, the viewer and Python) colours each cell by
what its material is made of. Every `Mn` card is sorted by composition:

- hydrogen and oxygen in the ratio of water are water;
- oxygen left over makes oxides: concrete with calcium, ceramic without;
- hydrocarbons are polymer;
- the metals left are steel, copper, tungsten, lead or light metal;
- cells lighter than 0.02 g/cm³ are gas.

Hydrogen and oxygen can also be bound: in a polymer (epoxy insulators), or in a
mineral (water of hydration in concrete). Where both readings make sense, the density
of the material's cells decides. Homogenised materials blend their looks by volume.
`llevant materials` lists the result: categories, volume fractions, colour,
roughness, metalness and clarity.

Realistic colours are lit physically; the other colourings stay flat. Lighting is
GGX specular and Lambert diffuse, with a key and a fill light that follow the camera
and a sky-to-ground environment. Water is half clear. Shading uses only each pixel's
hit, so it costs nothing next to tracing. There are no secondary rays; in the viewer,
screen-space ambient occlusion and outlines give the depth cues.

The heuristic will be wrong somewhere. Give looks by material number, in a file
(`--materials looks.txt`, one per line) or one at a time (`--look`):

```text
# looks.txt
12 steel
13 copper roughness=0.2
14 colour=#806040 metalness=0      # any category, then colour, roughness, metalness, clarity
```

The categories are void, gas, water, polymer, concrete, ceramic, carbon, steel,
copper, tungsten, lead and light-metal. The viewer reads the same specs from
`"materials": {"12": "steel"}` in its view file, and Python from
`m.set_materials({12: "steel"})`.

## Volumes and checks

**Lines.** `volume` and `check` trace `--probes` (default 12) independent lattices of
parallel lines, each in a random direction. There are as many lines as three
axis-parallel families `--h` apart would give. Lengths along each line are exact, so
thin gaps and plates are found whenever a line crosses them. Sigma is the standard
error over the probes, and it is honest: 2 sigma covers the error about 95 % of the
time.

**Refining.** `--rel-sigma 1e-3` refines every cell (or material, or universe; with
`--by label`, every instance) until its sigma is that fraction of its volume. It
halves `--h` over the boxes the group can be in, down to `--h-min`. Every cell the
model can hold is listed, met by the first lines or not, with one of four statuses:

- *converged*;
- *not converged*;
- *not found*: no line met it, so its volume, if any, is about the cube of the last
  spacing or less;
- *empty*: its boxes are empty, so its volume is 0.

The command exits with 1, naming the groups not converged or not found.

**Seeds.** The same `--seed` gives the same lines, so two models can be compared on
identical lines.

## Fingerprints

`llevant fingerprint` gives every cell and universe two digests: `geometry` and
`content` (geometry plus material compositions and densities). They are independent
of numbering and card order, and change for any change beyond the precision.
`--diff new.mcnp` lists the cells and universes that changed, are new, were removed or
were only renumbered.

## Scripting and agents

`info`, `bbox`, `locate`, `trace`, `materials`, `volume` and `check` take
`--format text|csv|json`; `fingerprint` writes JSON, and `fingerprint --diff` text or,
with `--format json`, JSON. Data
goes to stdout; progress, timings and warnings go to stderr, so `2>/dev/null` leaves
only the data.

**Exit status.**

| Status | Meaning |
|---|---|
| 0 | Done |
| 1 | Done, and the answer is "no": `check` found gaps or overlaps, `volume --rel-sigma` left groups not converged or not found, `bbox` found the cells nowhere, `fingerprint --diff` found the models differ |
| 2 | An error: bad arguments, an unreadable file, an unknown cell. The message is on stderr; with JSON output it is also on stdout, as `{"error": "..."}` (except for errors in the arguments themselves, found before the output is known) |

**JSON.** Every JSON output describes a label, a surface and a box in the same way
(the same fields as Python's `Label` and `Surface`). Cells, materials, universes and
surfaces are MCNP numbers; lengths are in cm; a missing value is `null`.

- A **label** (where a point or segment is):

  ```json
  {"status": "inside", "cell": 7, "path": [{"cell": 12, "lattice": null},
   {"cell": 304, "lattice": [1, -2, 0]}, {"cell": 7, "lattice": null}],
   "involved": [], "universe": 5, "material": 3, "density": -7.93,
   "instance": 1041, "instance_path": "7 < 304[1 -2 0] < 12",
   "text": "inside 12 > 304[1,-2,0] > 7"}
  ```

  `status` is `inside`, `gap`, `overlap`, `on-surface`, `unsupported` or `too-deep`.
  `cell`, `universe`, `material`, `density` and `instance` are `null` unless inside.
  `involved` lists the cells involved in an overlap, an on-surface point or an
  unsupported cell, at the last level of `path` (empty otherwise). `text` is the label as the text output prints it.
- A **surface**: `{"id": 12, "facet": 0}` (`facet` is a macrobody facet, 0 for an
  ordinary surface), or `null` at the ends of a line.
- A **box**: `{"lo": [x, y, z], "hi": [x, y, z]}`, or `null` when empty.

What each command gives:

| Command | JSON |
|---|---|
| `locate` | One label per point, with its `point` |
| `trace` | One entry per line: `from`, `to` and `segments`, each with `t0`, `t1`, `length` (cm along the line), `surface_in`, `surface_out` and `label` |
| `check` | One row per gap or overlap, largest first: its `label`, `volume`, `sigma` (cm³) and `box` |
| `volume` | One row per group: its number under `cell`, `material` or `universe` (`null` for gaps and overlaps), `name`, `status`, `label` (for a group that is one label: gaps, overlaps, every group of `--by label`; else `null`), `volume`, `sigma` and `box`; with `--rel-sigma`, `h`, `converged` and `outcome` |
| `bbox` | `box` around all the cells, and the `box` of each instance |
| `fingerprint --diff` | `same`, and for `cells` and `universes`: `unchanged` (a count), and the numbers that changed `content` or `geometry`, were `renumbered` (`{"from", "to"}`), `removed` or are `new` |
| `info`, `materials`, `fingerprint` | The model's sizes, extent and diagnostics; each material's look; each cell's and universe's digests |

**Many queries.** `locate --points` and `trace --lines` read a file, or stdin with `-`
(one point or line per line, numbers separated by spaces or commas, `#` comments), and
answer in one batch, on every core. Each run loads the model; the cache of compiled models
keeps that to about a second on the largest ones. For many small queries in a row,
Python keeps the model in memory ([`python.md`](python.md)).
