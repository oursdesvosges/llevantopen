# Python

## Install

`pip install llevant` installs an abi3 wheel for Windows or Linux (the release page has
the same wheels); one wheel serves Python 3.9 and later, and numpy comes with it. For data
frames, add `pip install polars` or `pandas`. The wheel includes the GPU renderer and
the viewer:

```sh
python -m llevant render model.mcnp --gpu -o out.png   # the `llevant` command
python -m llevant view model.mcnp                      # the viewer, from the wheel alone
```

`python -m llevant view` is the way to run the viewer where only Python packages can
be installed.

Other ways to install, and what your computer needs: [Install](../install.md).
Building the module from source: [contributing](../CONTRIBUTING.md).

## Use

[`tour.py`](https://github.com/Fusion4Energy/llevant/blob/main/crates/llevant-py/examples/tour.py)
walks through the whole API. Run it as a script (`python tour.py model.i out_dir`), or open it
as a notebook: VS Code and PyCharm run its `# %%` cells, and
`jupytext --to notebook tour.py` makes an `.ipynb`.

```python
import llevant, numpy as np

m = llevant.load("model.mcnp")          # or OpenMC XML (model.xml / dir); compiled once and cached
m.locate(0, 0, 0)                       # <Label inside 12 > 304[1,-2,0] > 7>: .cell, .material, .path...
r = m.classify(points)                  # (N, 3) array -> numpy columns: r.ids, r.cell, r.material, r.status
tr = m.trace_lines(starts, ends)        # ragged: tr.offsets, tr.t0, tr.t1, tr.ids, tr.surface_in...
tr.lengths("material")                  # {material: total length}
m.slice(z=37, colour="material")        # an Image: shown inline in Jupyter, .save("s.png"), .at(i, j)
m.render(direction=(1, 1, -0.5), hide=[12], gpu=True)
m.render(colour="realistic")            # physically based; m.materials(), m.set_materials({12: "steel"})
m.volumes(h=2).by("material")           # [Group(number, volume, sigma, bbox...)]
m.volumes_to(2, 1e-3, by="cell")        # every cell, refined to sigma <= 0.1 % (Group.converged, .outcome)
m.check(h=2)                            # gaps and overlaps
m.diff(m.reload())                      # what changed in the file, cell by cell
m.recompile(text)                       # a new version of the input, from text: only what changed is built
f = m.universe(20130)                   # a filler as a model of its own (cut="none" or a box)
r.to_polars(); tr.to_pandas()           # data frames, with polars or pandas installed
m.locate(0, 0, 0).instance_path         # "7 < 304[1 -2 0] < 12": which copy of the cell
f = llevant.Field.read(m, "run.msht", tally=14)   # or Field.cells(m, {12: 3.5}), .instances, .mesh
f_sp = llevant.read_openmc_tally(m, "statepoint.10.h5", score="flux")  # OpenMC HDF5 cell, material, or mesh tally
m.slice(z=37, field=f.with_scale(1e-6, 1e-2))      # values over the cells
m.slice(z=37, field=f, outline_by="material")      # lines between materials only
llevant.view("model.mcnp")              # the viewer; returns when its window closes
llevant.view()                          # the viewer's start screen, to open a model from
```

- **Numbers.** Cells, materials, universes and surfaces are MCNP numbers.
- **Inputs.** Batch queries take anything `numpy.asarray` accepts. C-ordered float64
  is read in place; other layouts are copied by rows.
- **Outputs.** Results are read-only numpy columns that Rust built: no Python object
  per point or segment. `Label` objects are made once per distinct label, on demand.
- **Threads.** Every query runs in Rust, in parallel, without holding the GIL.
- **The GPU model** is uploaded on the first `render(gpu=True)` and kept.
- **Material looks.** `set_materials` changes how that `Model` object draws later
  images; `slice(..., materials={12: "steel"})` and `render(...)` take looks for one
  image only. The compiled geometry is never changed.
- **Errors.** `OSError` for files, `ValueError` for bad arguments, and
  `LlevantError` for other failures (the GPU).

Every function, method and property is described in the
[Python API reference](../reference/python-api.md): units, MCNP numbers or indices,
and what `None` or `-1` means. The same text ships with the module as type stubs
(`_llevant.pyi`), so editors show it as you type.
