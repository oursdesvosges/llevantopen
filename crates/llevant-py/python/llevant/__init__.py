"""llevant: fast, exact queries on MCNP, OpenMC, TRIPOLI-4, and DRAGON5 geometry.

Load a model once, then query it::

    import llevant
    m = llevant.load("model.i")        # MCNP, OpenMC XML/dir, TRIPOLI-4, or DRAGON5 GEO:
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
from ._openmc import (
    diagnose_lost_particles,
    export_openmc_tally_csv,
    from_openmc,
    openmc_tallies,
    read_openmc_tally,
    to_mcnp,
)
from ._wrappers import (
    check_report,
    dragon_to_mcnp,
    from_dragon,
    from_tripoli,
    mass_balance,
    openmc_plots,
    render_openmc_plots,
    tripoli_to_mcnp,
)

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
    "check_report",
    "diagnose_lost_particles",
    "dragon_to_mcnp",
    "export_openmc_tally_csv",
    "from_dragon",
    "from_openmc",
    "from_text",
    "from_tripoli",
    "load",
    "main",
    "mass_balance",
    "openmc_plots",
    "openmc_tallies",
    "read_openmc_tally",
    "render_openmc_plots",
    "set_threads",
    "threads",
    "to_mcnp",
    "tripoli_to_mcnp",
    "view",
]
