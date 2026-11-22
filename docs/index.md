---
title: llevant
hide:
  - navigation
---

# llevant

**See and check MCNP, OpenMC, TRIPOLI-4, and DRAGON5 geometry, fast.** llevant
reads MCNP input files, OpenMC XML models (`model.xml`, `geometry.xml` +
`materials.xml`), TRIPOLI-4 decks (`.t4`/`.tripoli`), and DRAGON5 `GEO:`
procedures (`.x2m`/`.c2m`) and tells you what is where in them. It works on any
model, from a test problem to a 425 000-cell fusion reactor:

- **look at it**: 3D views, OpenMC `<plots>`, and exact slices, in an interactive viewer or as images;
- **find mistakes**: gaps and overlaps (where particles are lost), with their size, location, and automated QA slice reports;
- **diagnose lost particles**: inspect `(x, y, z)` coordinates or MCNP/OpenMC logs to pinpoint gaps, overlaps, and grazing surfaces;
- **measure**: the volume and mass of every cell, material or universe, with an
  honest uncertainty;
- **ask questions**: which cell holds this point, which cells this line crosses;
- **compare**: which cells changed between two versions of a model;
- **show results**: OpenMC statepoint tallies (with energy/particle/nuclide/instance filtering), meshtal, MCTAL, D1SUNED cuvmsh, and per-cell results drawn over the geometry;
- **convert**: translate OpenMC, TRIPOLI-4, and DRAGON5 geometry to canonical MCNP input.

It reads every MCNP surface type and macrobody, OpenMC CSG XML models (surfaces,
universes, rectilinear and hexagonal lattices, rotations and materials),
TRIPOLI-4 volumes (`EQUA`, `COMBI`, `ECRASE`), DRAGON5 `GEO:` cells (`CARCEL`,
`TUBE`, `HEXCEL`, `CAR2D`, `CAR3D`), and transforms ([the full list](reference/supported-input.md)).

![The llevant viewer: a 3D view and a slice side by side](img/viewer.png)

## Three ways to use it

<div class="grid cards" markdown>

-   **The viewer**

    ---

    A window with a 3D view and a slice side by side. Click on things, hide them,
    cut the model open, look for problems. No programming needed.

    [:octicons-arrow-right-24: The viewer](guide/viewer.md)

-   **The command line**

    ---

    One command per question (`llevant check model.i`), for images, checks and
    volumes in scripts and batch jobs.

    [:octicons-arrow-right-24: The command line](guide/cli.md)

-   **Python**

    ---

    Load a model once, then ask it millions of questions with numpy arrays; images
    show inline in Jupyter.

    [:octicons-arrow-right-24: Python](guide/python.md)

</div>

## Start here

1. [Install llevant](install.md): one `pip install`, or a download.
2. [Getting started](getting-started.md): a ten-minute tour on a small example model.
3. [How do I…?](how-to.md): the common jobs, step by step.

Stuck? See [Help and FAQ](help.md), or
[open an issue](https://github.com/Fusion4Energy/llevant/issues).

## How fast?

On a 425 000-cell fusion reactor model, on a 16-thread laptop:

| | |
|---|---|
| Open the model | 1.3 s (36 s the first time, then it is cached) |
| Locate points | 5.4 million per second |
| Trace lines | 330 000 lines of 1 m per second |
| A slice | 0.1 s for a 1000 × 1000 image |
| Check for gaps and overlaps | the whole 36 × 37 × 35 m building in under 3 minutes |
| Viewer | 60 frames per second while turning the model (GPU) |

The details are in [Benchmarks](benchmarks.md).

## Licence

llevant is free software under the
[EUPL-1.2](https://github.com/Fusion4Energy/llevant/blob/main/LICENSE) licence.
