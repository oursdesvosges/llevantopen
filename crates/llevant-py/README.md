# llevant

Fast, exact geometry queries on MCNP, OpenMC XML, TRIPOLI-4, and DRAGON5 models, from Python or the command line.
llevant answers:

- where points are, and what lines cross;
- slices, OpenMC `<plots>`, and 3D views;
- volumes, masses (`Group.mass`, `mass_balance`), gaps and overlaps (`check`, `check_report`), and lost-particle diagnosis (`diagnose_lost_particles`);
- fingerprints that tell two versions of a model apart;
- multi-code conversion to MCNP (`to_mcnp`, `tripoli_to_mcnp`, `dragon_to_mcnp`) and OpenMC HDF5 statepoint tally loading (`openmc_tallies`, `read_openmc_tally`, `export_openmc_tally_csv`).

It reads MCNP inputs, OpenMC XML (`from_openmc`), TRIPOLI-4 (`from_tripoli`), and DRAGON5 `GEO:` (`from_dragon`), handling every universe level, lattice and transform.
The Python module is built on the Rust engine of the same name: models are compiled once (and cached on disk), then queried in numpy batches.

![The viewer: a reactor core cut open in 3D, beside an exact slice through its fuel pins](https://raw.githubusercontent.com/Fusion4Energy/llevant/main/docs/img/viewer.png)

## Install

```sh
pip install llevant              # numpy comes with it; add polars or pandas for data frames
```

One abi3 wheel serves Python 3.9 and later, on Windows and Linux (x86-64; no macOS
yet). It includes the GPU renderer and the interactive viewer.

## Use

```python
import llevant

m = llevant.load("model.mcnp")          # compiled once (cached on disk), then queried as often as needed
m.locate(0, 0, 0)                       # <Label inside 12 > 304[1,-2,0] > 7>: .cell, .material, .path...
r = m.classify(points)                  # (N, 3) array -> numpy columns: r.ids, r.cell, r.material, r.status
tr = m.trace_lines(starts, ends)        # ragged: tr.offsets, tr.t0, tr.t1, tr.ids, tr.surface_in...
m.slice(z=37, colour="material")        # an Image: shown inline in Jupyter, .save("s.png")
m.render(direction=(1, 1, -0.5), hide=[12], gpu=True)
m.volumes(h=2).by("material")           # volumes with an honest uncertainty
m.check(h=2)                            # gaps and overlaps
f = llevant.Field.read(m, "run.msht", tally=14)   # meshtal, MCTAL, D1SUNED cuvmsh, CSV per cell
m.slice(z=37, field=f)                  # results over the geometry
```

The package also has the `llevant` command and the viewer:

```sh
python -m llevant slice model.mcnp --plane z=37 --colour material -o slice.png
python -m llevant render model.mcnp --gpu -o view.png
python -m llevant view model.mcnp
```

## Documentation

**<https://fusion4energy.github.io/llevant/>**:

- [Install](https://fusion4energy.github.io/llevant/install/) and
  [getting started](https://fusion4energy.github.io/llevant/getting-started/) on an example model
- [How do I…?](https://fusion4energy.github.io/llevant/how-to/): gaps and overlaps, volumes,
  tallies, comparing models
- [Python guide](https://fusion4energy.github.io/llevant/guide/python/) and
  [API reference](https://fusion4energy.github.io/llevant/reference/python-api/)
- [Source, issues and releases](https://github.com/Fusion4Energy/llevant)

Licence: EUPL-1.2.
