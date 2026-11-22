"""llevant: fast, exact queries on MCNP and OpenMC XML geometry.

Load a model once, then query it::

    import llevant
    m = llevant.load("model.i")        # or model.xml / geometry directory
    m.locate(0, 0, 0)                  # where a point is
    m.classify(points)                 # many points: numpy columns
    m.trace((0, 0, 0), (100, 0, 0))    # segments of a line
    m.slice(z=0)                       # an image (shown inline in Jupyter)
    m.volumes(h=1.0).by("material")

Cells, materials and universes keep their input numbers; lengths are in cm. The
command line is ``python -m llevant --help``.
"""

from ._llevant import (
    DEFAULT_PRECISION,
    NO_HIT,
    STATUS_NAMES,
    Cell,
    Cells,
    Changes,
    Classification,
    Diagnostic,
    Diff,
    Field,
    Fingerprints,
    Group,
    Hits,
    Image,
    Label,
    LlevantError,
    Model,
    Segment,
    Surface,
    Tracing,
    Volumes,
    from_text,
    load,
    main,
    set_threads,
    threads,
)
from ._openmc import read_openmc_tally

def view(model=None, *, state=None, backend=None):
    """Open the interactive viewer on a model; returns when the window closes.

    Without `model`, the viewer's start screen, to open one from. Runs in this
    process (no extra executable). Call it from a script, not a notebook cell:
    it blocks. `state`: view file to open and save to (default
    ``<model>.view.json``; needs `model`); `backend`: ``"auto"``, ``"gpu"`` or
    ``"cpu"``.
    """
    argv = ["llevant", "view"]
    if model is not None:
        argv.append(str(model))
    if state is not None:
        argv += ["--state", str(state)]
    if backend is not None:
        argv += ["--backend", backend]
    if main(argv) != 0:
        raise LlevantError("view failed (see the message above)")


__all__ = [
    "DEFAULT_PRECISION",
    "NO_HIT",
    "STATUS_NAMES",
    "Cell",
    "Cells",
    "Changes",
    "Classification",
    "Diagnostic",
    "Diff",
    "Field",
    "Fingerprints",
    "Group",
    "Hits",
    "Image",
    "Label",
    "LlevantError",
    "Model",
    "Segment",
    "Surface",
    "Tracing",
    "Volumes",
    "from_text",
    "load",
    "main",
    "read_openmc_tally",
    "set_threads",
    "threads",
    "view",
]
