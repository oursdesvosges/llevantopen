"""Type stubs of `llevant._llevant`, the module compiled from Rust (re-exported by `llevant`)."""

from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from typing import Any, Dict, Literal, Optional, Union, final

import numpy as np
import numpy.typing as npt

_Xyz = tuple[float, float, float]
_Box = tuple[_Xyz, _Xyz]
_Region = Union[Sequence[float], tuple[Sequence[float], Sequence[float]]]
_Points = npt.ArrayLike
_Numbers = Union[float, npt.ArrayLike]
_Scope = Literal["contents", "problem"]
_ColourBy = Literal["cell", "material", "universe", "realistic"]
_GroupBy = Literal["cell", "material", "universe", "label"]
_StatusName = Literal["inside", "gap", "overlap", "on-surface", "unsupported", "too-deep"]

__all__ = [
    "main",
    "load",
    "from_text",
    "_translate_path",
    "_translate_text",
    "_read_openmc_plots",
    "_tripoli_to_mcnp",
    "_dragon_to_mcnp",
    "set_threads",
    "threads",
    "Model",
    "Label",
    "Surface",
    "Segment",
    "Diagnostic",
    "Cell",
    "Cells",
    "Group",
    "Volumes",
    "Fingerprints",
    "Changes",
    "Diff",
    "Classification",
    "Tracing",
    "Hits",
    "Image",
    "Field",
    "LlevantError",
    "STATUS_NAMES",
    "NO_HIT",
    "DEFAULT_PRECISION",
]

DEFAULT_PRECISION: float
NO_HIT: int
STATUS_NAMES: tuple[str, ...]

class LlevantError(Exception): ...

def load(
    path: Union[str, PathLike[str]],
    *,
    tolerance: float = 1e-8,
    world_half_size: float = 1e6,
    grid: Literal["auto", "always", "off"] = "auto",
) -> Model:
    """Read and compile an MCNP input file."""

def from_text(
    text: str,
    *,
    tolerance: float = 1e-8,
    world_half_size: float = 1e6,
    grid: Literal["auto", "always", "off"] = "auto",
) -> Model:
    """Compile MCNP input text."""

def _translate_path(path: Union[str, PathLike[str]]) -> str:
    """Read and translate an input file or OpenMC directory to MCNP text."""

def _translate_text(text: str) -> str:
    """Translate input text (OpenMC, TRIPOLI-4, DRAGON5, or MCNP) to MCNP text."""

def _read_openmc_plots(
    model_path: Union[str, PathLike[str]],
    plots_xml: Optional[Union[str, PathLike[str]]] = None,
) -> list[dict[str, Any]]:
    """Read OpenMC `<plot>` slice/voxel metadata from a model path or `plots.xml`."""

def _tripoli_to_mcnp(text: str) -> str:
    """Translate native TRIPOLI-4 input text directly into canonical MCNP text."""

def _dragon_to_mcnp(text: str) -> str:
    """Translate native DRAGON5 CLE-2000 `GEO:` text directly into canonical MCNP text."""

def set_threads(n: int) -> None:
    """Use `n` worker threads; only before the first query."""

def threads() -> int:
    """Number of worker threads queries use."""
def main(argv: list[str]) -> int:
    """Run the command line (`argv[0]` is the program name)."""

@final
class Label:
    """Where a point is. `str()` gives it in MCNP terms."""

    @property
    def status(self) -> _StatusName:
        """`inside`, `gap`, `overlap`, `on-surface`, `unsupported` or `too-deep`."""
    @property
    def cell(self) -> Optional[int]:
        """The cell that holds the point (MCNP number), when it is inside one."""
    @property
    def path(self) -> list[tuple[int, Optional[tuple[int, int, int]]]]:
        """Cells from level 0 down, as `(cell, lattice element or None)`. For a
        point that is not inside, the path ends at the filled cell whose
        universe has the problem.
        """
    @property
    def involved(self) -> list[int]:
        """The cells involved, for overlaps, on-surface and unsupported points
        (at the last level of the path); empty otherwise.
        """
    @property
    def universe(self) -> Optional[int]:
        """Universe of the cell that holds the point."""
    @property
    def material(self) -> Optional[int]:
        """Material of the cell that holds the point (0 for void)."""
    @property
    def density(self) -> Optional[float]:
        """Density of the cell that holds the point, with the MCNP sign."""
    @property
    def instance(self) -> Optional[int]:
        """Which instance of its cell holds the point: an id in
        `0 .. Model.instance_count`, one per place the cell occupies (each
        cell its universe fills, each lattice element). None when not
        inside a cell.
        """
    @property
    def instance_path(self) -> Optional[str]:
        """The path in MCNP's repeated-structure notation, deepest cell first
        (`"12 < 100[1 0 0] < 5"`), when inside a cell.
        """
    @property
    def is_inside(self) -> bool:
        """Whether the point is inside a cell (`status == "inside"`)."""
    def __hash__(self) -> int:
        """Labels hash by content, so they can be dictionary keys."""
    def __eq__(self, other: object, /) -> bool:
        """Equal when status, path and involved cells are equal."""

@final
class Surface:
    """MCNP surface number and macrobody facet (0 for ordinary surfaces)."""

    @property
    def id(self) -> int:
        """MCNP surface number."""
    @property
    def facet(self) -> int:
        """Macrobody facet number (1-based), 0 for an ordinary surface."""

@final
class Segment:
    """A stretch of a line in one label, from `t0` to `t1` (cm along the line),
    entered through `surface_in` and left through `surface_out` (None at the
    ends of the line).
    """
    t0: float
    t1: float
    label: Label
    surface_in: Optional[Surface]
    surface_out: Optional[Surface]
    @property
    def length(self) -> float:
        """`t1 - t0`, cm."""

@final
class Diagnostic:
    """A problem found while compiling: unsupported features, undefined
    references... The affected cells are compiled as unsupported.
    """
    subject: Literal["cell", "surface", "material", "input"]
    number: Optional[int]
    message: str
    line: Optional[int]

@final
class Cell:
    """A cell card, as compiled."""
    id: int
    universe: int
    material: int
    density: float
    fills: list[int]
    lattice: bool
    imp_zero: bool
    unsupported: Optional[str]
    bbox: Optional[_Box]

@final
class Cells:
    """Cells by MCNP number, in card order."""

    def __len__(self) -> int:
        """Number of cells."""
    def __contains__(self, id: int, /) -> bool:
        """Whether the model has cell `id` (MCNP number)."""
    def __getitem__(self, id: int, /) -> Cell:
        """Cell `id` (MCNP number); KeyError if there is none."""
    def __iter__(self) -> Iterator[int]:
        """The cell numbers, in card order."""
    def keys(self) -> list[int]:
        """The cell numbers, in card order."""
    def values(self) -> list[Cell]:
        """The cells, in card order."""
    def items(self) -> list[tuple[int, Cell]]:
        """`(number, cell)` pairs, in card order."""
    def get(self, id: int) -> Optional[Cell]:
        """Cell `id` (MCNP number), None if there is none."""

class _LabelColumns:
    @property
    def labels(self) -> list[Label]:
        """The distinct labels; `ids` index into them."""
    @property
    def ids(self) -> npt.NDArray[np.uint32]:
        """Label id of each item (`uint32`), into `labels`."""
    @property
    def cell(self) -> npt.NDArray[np.int64]:
        """MCNP cell holding each item (`int64`), -1 when not inside one."""
    @property
    def material(self) -> npt.NDArray[np.int64]:
        """Material of that cell (`int64`, 0 for void), -1 when not inside one."""
    @property
    def universe(self) -> npt.NDArray[np.int64]:
        """Universe of that cell (`int64`), -1 when not inside one."""
    @property
    def instance(self) -> npt.NDArray[np.int64]:
        """Instance of that cell (`int64`, `Label.instance`), -1 when
        not inside one.
        """
    @property
    def status(self) -> npt.NDArray[np.int8]:
        """Status of each item (`int8`): an index into
        `llevant.STATUS_NAMES`, -1 for none.
        """
    def to_polars(self) -> Any:
        """The columns as a Polars data frame (needs `polars`); labels
        and statuses are categorical.
        """
    def to_pandas(self) -> Any:
        """The columns as a pandas data frame (needs `pandas`); labels
        and statuses are categorical.
        """

@final
class Classification(_LabelColumns):
    """Labels of a batch of points."""

    def __len__(self) -> int:
        """Number of points."""
    def __getitem__(self, i: int, /) -> Label:
        """The label of point `i`."""

@final
class Tracing(_LabelColumns):
    """Segments of a batch of lines, ragged: line i is offsets[i]:offsets[i+1]."""

    @property
    def offsets(self) -> npt.NDArray[np.uint64]:
        """Start of each line's segments (`uint64`, one more than the lines)."""
    @property
    def t0(self) -> npt.NDArray[np.float64]:
        """`t` (cm along the line) where each segment starts."""
    @property
    def t1(self) -> npt.NDArray[np.float64]:
        """`t` where each segment ends."""
    @property
    def line(self) -> npt.NDArray[np.int64]:
        """Line of each segment (`int64`)."""
    @property
    def surface_in(self) -> npt.NDArray[np.int64]:
        """Surface each segment is entered through (MCNP number, `int64`), -1 at
        the start of a line.
        """
    @property
    def surface_out(self) -> npt.NDArray[np.int64]:
        """Surface each segment is left through, -1 at the end of a line."""
    @property
    def facet_in(self) -> npt.NDArray[np.uint8]:
        """Macrobody facet of `surface_in` (`uint8`, 0 for ordinary surfaces)."""
    @property
    def facet_out(self) -> npt.NDArray[np.uint8]:
        """Macrobody facet of `surface_out`."""
    def __len__(self) -> int:
        """Number of lines."""
    def __getitem__(self, i: int, /) -> list[Segment]:
        """The segments of line `i`."""
    def lengths(self, by: _GroupBy = "cell") -> dict[Any, float]:
        """Total length (cm) by `cell`, `material`, `universe` or `label`.
        Segments not inside a cell are keyed by their status name (`gap`,
        `overlap`...) except with `by="label"`, where every key is a label.
        """

@final
class Hits(_LabelColumns):
    """First visible hits of a batch of rays."""

    @property
    def t(self) -> npt.NDArray[np.float64]:
        """`t` (cm along the ray) of each hit (`float64`), inf where the ray
        hit nothing.
        """
    @property
    def hit(self) -> npt.NDArray[np.bool_]:
        """Whether each ray hit something visible."""
    @property
    def surface(self) -> npt.NDArray[np.int64]:
        """Surface entered (MCNP number, `int64`), -1 when the ray starts inside
        or misses.
        """
    @property
    def facet(self) -> npt.NDArray[np.uint8]:
        """Macrobody facet of `surface` (`uint8`, 0 for ordinary surfaces)."""
    @property
    def normal(self) -> npt.NDArray[np.float64]:
        """Unit normal of the surface entered, `(N, 3)`, zero without one."""
    @property
    def ghost_count(self) -> npt.NDArray[np.uint32]:
        """Ghost boundaries each ray crossed before its hit (0 without ghosts)."""
    @property
    def ghost_cell(self) -> npt.NDArray[np.int64]:
        """MCNP number of the ghost cell at each ray's first ghost boundary,
        -1 for none.
        """
    def __len__(self) -> int:
        """Number of rays."""
    def __getitem__(self, i: int, /) -> Optional[Label]:
        """Label hit by ray `i`, None if it hit nothing."""

@final
class Image:
    """A slice or render; shown inline in Jupyter."""

    @property
    def width(self) -> int:
        """Width in pixels."""
    @property
    def height(self) -> int:
        """Height in pixels."""
    @property
    def kind(self) -> Literal["slice", "render"]:
        """`slice` or `render`."""
    @property
    def rgb(self) -> npt.NDArray[np.uint8]:
        """The pixels, `(height, width, 3)` `uint8`, rows from the top (for
        `matplotlib.pyplot.imshow`).
        """
    @property
    def cell(self) -> npt.NDArray[np.int64]:
        """MCNP cell under each pixel, `(height, width)` `int64`, -1 for none."""
    @property
    def ids(self) -> Optional[npt.NDArray[np.uint32]]:
        """Label id under each pixel, `(height, width)` `uint32` into `labels`
        (`NO_HIT` where a ray hits nothing); None for GPU renders.
        """
    @property
    def labels(self) -> list[Label]:
        """The distinct labels in the image (empty for GPU renders)."""
    def at(self, i: int, j: int) -> Optional[Label]:
        """What is under pixel `(i, j)` (column `i`, row `j` from the top): the
        label at the pixel centre (slices) or the first one the ray hits
        (renders; None if it hits nothing).
        """
    def point(self, i: int, j: int) -> Optional[_Xyz]:
        """The point under pixel `(i, j)`: its centre on the plane (slices) or
        where its ray hits (renders; None if it hits nothing).
        """
    def problems(self) -> list[tuple[Label, int]]:
        """Statuses found in the image other than `inside` (gaps, overlaps...),
        by number of pixels.
        """
    def png(self) -> bytes:
        """The image as PNG bytes."""
    def save(self, path: Union[str, PathLike[str]]) -> None:
        """Write the image as a PNG file."""

@final
class Group:
    """Volume of a cell, material, universe or label (cm³)."""
    number: Optional[int]
    name: str
    status: _StatusName
    label: Optional[Label]
    volume: float
    sigma: float
    mass: Optional[float]
    mass_sigma: Optional[float]
    by_probe: list[float]
    segments: int
    probes_hit: int
    h: float
    converged: Optional[bool]
    outcome: Optional[Literal["converged", "not converged", "not found", "empty"]]
    bbox: Optional[_Box]

@final
class Volumes:
    """Volumes of everything a grid of lines met (`Model.volumes`)."""
    def by(self, by: _GroupBy = "cell") -> list[Group]:
        """Volumes by `cell`, `material`, `universe` or `label` (every path
        apart). Cells, materials and universes come first in number order,
        then gaps, overlaps and the like, one group per label.
        """
    def problems(self) -> list[Group]:
        """Gaps, overlaps, unsupported cells and too-deep nesting, largest
        first.
        """
    @property
    def total(self) -> float:
        """Sum of all volumes: the region, up to the grid's edges."""
    @property
    def h(self) -> float:
        """Line spacing (cm)."""
    @property
    def region(self) -> _Box:
        """The box measured, `((x0, y0, z0), (x1, y1, z1))` in cm."""
    def to_polars(self, by: _GroupBy = "cell") -> Any:
        """`by()` as a Polars data frame (needs `polars`)."""
    def to_pandas(self, by: _GroupBy = "cell") -> Any:
        """`by()` as a pandas data frame (needs `pandas`)."""

@final
class Fingerprints:
    """Digests of every cell and universe that ignore numbering and card
    order: `cells[n]` and `universes[n]` are `(geometry, content)` hex
    digests.
    """
    version: int
    precision: float
    cells: dict[int, tuple[str, str]]
    universes: dict[int, tuple[str, str]]

@final
class Changes:
    """How the cells (or universes) of two models differ, by MCNP number."""
    unchanged: int
    content: list[int]
    geometry: list[int]
    renumbered: list[tuple[int, int]]
    removed: list[int]
    new: list[int]
    def __len__(self) -> int:
        """Number of cells (or universes) that changed, were renumbered,
        removed or are new.
        """

@final
class Diff:
    """What `Model.diff` finds: `cells` and `universes` changes. False when
    the models are the same up to numbering; `str()` gives a report.
    """
    cells: Changes
    universes: Changes
    def __bool__(self) -> bool:
        """True when the models differ (beyond numbering)."""

@final
class Field:
    """Numbers over a model's geometry, drawn over its images with `field=`."""

    @staticmethod
    def cells(model: Model, values: Mapping[int, float], *, name: str = "", unit: str = "") -> Field:
        """Values per cell, `{cell number: value}`; every instance of a cell
        has its value.
        """
    @staticmethod
    def materials(model: Model, values: Mapping[int, float], *, name: str = "", unit: str = "") -> Field:
        """Values per material, `{material number: value}`, on every unfilled
        cell of each.
        """
    @staticmethod
    def instances(
        model: Model,
        values: Union[Mapping[int, float], Mapping[str, float], npt.ArrayLike],
        *,
        name: str = "",
        unit: str = "",
    ) -> Field:
        """Values per cell instance (`Label.instance`): an array of
        `model.instance_count` values (NaN for none), or a dict of instance
        ids or paths (`"12 < 100[1 0 0] < 5"`) to values.
        """
    @staticmethod
    def mesh(
        model: Model,
        values: npt.ArrayLike,
        *,
        x: Optional[Sequence[float]] = None,
        y: Optional[Sequence[float]] = None,
        z: Optional[Sequence[float]] = None,
        r: Optional[Sequence[float]] = None,
        theta: Optional[Sequence[float]] = None,
        origin: Optional[Sequence[float]] = None,
        axis: Optional[Sequence[float]] = None,
        vec: Optional[Sequence[float]] = None,
        transform: Optional[tuple[Sequence[Sequence[float]], Sequence[float]]] = None,
        name: str = "",
        unit: str = "",
    ) -> Field:
        """Values on a structured mesh, one per voxel in C order of its three
        axes (an array of that shape, or flat; NaN for none). A rectangular
        mesh gives the bounds along `x`, `y` and `z` (cm), placed by
        `transform` (`(rotation, translation)`: world = rotation · local +
        translation, as MCNP's TR=); a cylindrical one `r`, `z` and `theta`
        (revolutions, 0 to 1) about `axis` through `origin`, theta 0 along
        `vec`.
        """
    @staticmethod
    def read(
        model: Model,
        path: Union[str, PathLike[str]],
        *,
        tally: Optional[int] = None,
        energy: str = "total",
        time: str = "total",
        bins: Optional[Dict[str, Union[int, str]]] = None,
        error: bool = False,
    ) -> Field:
        """A field from a file: a CSV file (first column `cell`, `material`,
        `instance` or `path`; second the values, its header the name with
        the unit in brackets) or an MCNP meshtal file (exact meshes from
        the model's input, when it has their FMESH cards), D1SUNED cuvmsh
        (`out=cuv`, as each voxel's flux) or MCTAL file (cell
        tallies, per cell or per instance, bins as the model's input writes
        them): `tally` (default the first), `energy` and `time` (`"total"`
        or an index), `bins` for MCTAL's other dimensions (`{"m": 1}`),
        `error` for relative errors.
        """
    @property
    def name(self) -> str:
        """What the values are (shown on the legend)."""
    @property
    def unit(self) -> str:
        """The values' unit (shown on the legend)."""
    @property
    def kind(self) -> Literal["cells", "instances", "mesh"]:
        """`"cells"`, `"instances"` or `"mesh"`."""
    @property
    def scale(self) -> tuple[float, float, bool]:
        """The colour scale: `(low, high, logarithmic)`."""
    @property
    def opacity(self) -> float:
        """How far the field covers the cells' colours, 0 to 1."""
    @property
    def cell_colours_under(self) -> bool:
        """Whether the cells' own colours are under the field (and where it
        has no value) rather than a neutral grey.
        """
    def with_scale(
        self, low: Optional[float] = None, high: Optional[float] = None, *, log: Optional[bool] = None
    ) -> Field:
        """The same field on another scale: ends left out are kept, and a
        change of `log` fits the scale to the values again.
        """
    def with_opacity(self, opacity: float) -> Field:
        """The same field covering the cells' colours `opacity` of the way."""
    def with_cell_colours_under(self, on: bool) -> Field:
        """The same field over the cells' own colours (`True`) or over a
        neutral grey, which cells without a value show (`False`, the
        default).
        """
    def value(
        self, x: Union[float, Sequence[float]], y: Optional[float] = None, z: Optional[float] = None
    ) -> Optional[float]:
        """The value at a point, None where the field has none."""
    def values(self, points: _Points) -> npt.NDArray[np.float64]:
        """The values at many points (an `(N, 3)` array), NaN where none."""

@final
class Model:
    """A compiled MCNP model, loaded once and queried."""

    @property
    def path(self) -> Optional[str]:
        """The file the model was read from (None for `from_text`)."""
    @property
    def tolerance(self) -> float:
        """Distance (cm) within which a point is on a surface."""
    @property
    def cells(self) -> Cells:
        """The cells by MCNP number: `m.cells[304]`, `304 in m.cells`."""
    @property
    def universes(self) -> list[int]:
        """Universe numbers, 0 (the real world) first."""
    @property
    def instance_count(self) -> int:
        """Number of cell instance ids: every place a cell occupies (each
        cell its universe fills, each lattice element) has one id below this
        (`Label.instance`). A lattice filled with one universe everywhere
        numbers the elements of a box around it, so some ids may name no
        place.
        """
    def instance_path(self, id: int) -> Optional[str]:
        """The path of instance `id` in MCNP's repeated-structure notation,
        deepest cell first (`"12 < 100[1 0 0] < 5"`); None past the count.
        """
    def instance(self, path: str) -> int:
        """The instance id of a path written as `instance_path` writes it
        (lattice elements with one to three indices). ValueError if the
        model has no such path.
        """
    def cell_instances(self, cell_id: int) -> list[int]:
        """Global instance ids for every occurrence of cell `cell_id` in
        instance-index order. KeyError if there is no such cell.
        """
    @property
    def diagnostics(self) -> list[Diagnostic]:
        """Problems found while compiling (unsupported features, undefined
        references...).
        """
    def material(self, id: int) -> list[tuple[str, float]]:
        """The `Mn` card: `(zaid, fraction)` pairs as written (negative
        fractions are by weight). KeyError if there is none.
        """
    def materials(self) -> list[dict[str, Any]]:
        """What each material looks like in realistic images: per material
        (ascending), a dict of `material`, `cells` (that use it), `density`
        (smallest and largest, g/cm³, None if unknown), `look` (main
        category: steel, concrete, water, ...), `colour` (#rrggbb),
        `roughness`, `metalness`, `clarity`, `set` (whether `set_materials` gave it)
        and `parts` (category → volume fraction, from the composition).
        """
    def set_materials(self, looks: Optional[Mapping[int, str]] = None) -> None:
        """Change the look of materials in realistic images: `{number: spec}`,
        a spec being a category and/or settings, `"steel"`,
        `"copper roughness=0.2"`, `"colour=#806040 metalness=0"`. Replaces
        earlier settings; None or `{}` goes back to the looks classified
        from the material cards.

        This changes how this `Model` object draws later images (and what
        `materials()` reports); the compiled geometry, and other `Model`
        objects of the same file, are unchanged. For one image only, pass
        `materials=` to `slice` or `render` instead.
        """
    def info(self) -> dict[str, Any]:
        """Sizes, compile times (s) and tolerance."""
    def bounds(self, scope: _Scope = "contents") -> Optional[_Box]:
        """Box around the level-0 cells of `scope` from their bounding boxes:
        `contents` (cells with a material or fill) or `problem` (all but
        `IMP:N=0`). Loose for CAD conversions; see `extent`.
        """
    def extent(self, scope: _Scope = "contents", n: int = 128) -> Optional[_Box]:
        """Box around `scope`, measured by tracing `n × n` lines per axis (the
        default region of slices, renders and volumes; computed once for
        `n = 128`).
        """
    def bbox(self, cells: str) -> Optional[_Box]:
        """Box (global frame) around every instance of the cells of a rule
        (`"c 10-20"`, `"u 5 & m 3"`, `'label "BSV"'`: the cells whose `$`
        comments contain BSV, read from the input when a rule has
        `label`); `None` when they are nowhere (cells of
        universes no cell fills). Conservative: padded outward, and only an
        outer bound below lattices (see `instance_boxes`).
        """
    def instance_boxes(self, cells: str) -> list[tuple[int, _Box, bool]]:
        """The box (global frame) of every instance of the cells of a rule:
        `(cell, (lo, hi), exact)` per instance, cell by cell. `exact` is
        false for boxes that are only an outer bound (below a lattice).
        """
    def locate(
        self,
        x: Union[float, Sequence[float]],
        y: Optional[float] = None,
        z: Optional[float] = None,
        *,
        level: Optional[int] = None,
    ) -> Label:
        """Where a point is: `locate(x, y, z)` or `locate((x, y, z))`; with
        `level`, its path down to that universe level at most (0: the top,
        where a filled cell ends it).
        """
    def classify(self, points: _Points, *, level: Optional[int] = None) -> Classification:
        """Labels of many points at once: `points` is an `(N, 3)` array (or
        anything `numpy.asarray` takes); `level` as in `locate`.
        """
    def trace(
        self,
        start: Sequence[float],
        end: Optional[Sequence[float]] = None,
        *,
        dir: Optional[Sequence[float]] = None,
        t: Optional[tuple[float, float]] = None,
        level: Optional[int] = None,
    ) -> list[Segment]:
        """Segments of one line, from `start` to `end`, or from `start` along
        `dir` for `t` in `(t_min, t_max)` (cm); `level` as in `locate`.
        """
    def trace_lines(
        self,
        start: _Points,
        end: Optional[_Points] = None,
        *,
        dir: Optional[_Points] = None,
        t_min: Optional[_Numbers] = None,
        t_max: Optional[_Numbers] = None,
        level: Optional[int] = None,
    ) -> Tracing:
        """Segments of many lines at once: from `start` to `end` (`(N, 3)`
        arrays), or from `start` along `dir` for `t` in `(t_min, t_max)`
        (numbers or arrays of N); `level` as in `locate`.
        """
    def first_hit(
        self,
        start: _Points,
        dir: _Points,
        *,
        t_min: Optional[_Numbers] = None,
        t_max: Optional[_Numbers] = None,
        hide: Sequence[int] = ...,
        hide_universes: Sequence[int] = ...,
        hide_materials: Sequence[int] = ...,
        show_void: bool = False,
        level: Optional[int] = None,
        ghost: Sequence[str] = ...,
    ) -> Hits:
        """Where rays first enter a visible cell: from `start` along `dir`
        (`(N, 3)` arrays) for `t` in `(t_min, t_max)` (default: far beyond
        the model). Void cells are hidden unless `show_void`; `hide` and
        `hide_universes` hide more (and everything inside them), and
        `hide_materials` the cells of those materials (filled cells stay, so
        what fills them shows). `level`: only cells at that universe level
        are seen, as MCNP's plotter shows `LEVEL n` (a filled cell there is
        solid). `ghost`: rules (`"c 300"`, `"u 20040 & m 5"`) of cells seen
        as shells, what fills them shown; `Hits.ghost_count` and
        `Hits.ghost_cell` say what each ray crossed of them.
        """
    def slice(
        self,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        region: Optional[_Region] = None,
        width: int = 1000,
        colour: _ColourBy = "cell",
        outline: bool = True,
        outline_by: Literal["cell", "material", "universe"] = "cell",
        field: Optional[Field] = None,
        level: Optional[int] = None,
        materials: Optional[Mapping[int, str]] = None,
    ) -> Image:
        """Image of the plane `x=`, `y=` or `z=` (give one) across `region`
        (default: the measured extent of the contents), `width` pixels wide,
        coloured by `cell`, `material`, `universe` or `realistic`, with
        `field` (a `Field` of this model) over the cells. `level`: only
        cells at that universe level, as MCNP's plotter shows `LEVEL n`
        (the rest blank). `outline_by`: the lines between cells (`"cell"`),
        or between the materials (`"material"`, void one of them) or
        universes (`"universe"`) of the deepest cells.

        `materials`: looks for this image only, `{number: spec}` as for
        `set_materials`, instead of the model's.
        """
    def render(
        self,
        *,
        direction: Optional[Sequence[float]] = None,
        eye: Optional[Sequence[float]] = None,
        target: Optional[Sequence[float]] = None,
        fov: float = 30.0,
        size: tuple[int, int] = ...,  # (1000, 1000)
        region: Optional[_Region] = None,
        clip: Optional[tuple[Sequence[float], Sequence[float]]] = None,
        hide: Sequence[int] = ...,
        hide_universes: Sequence[int] = ...,
        hide_materials: Sequence[int] = ...,
        show_void: bool = False,
        colour: _ColourBy = "cell",
        gpu: bool = False,
        field: Optional[Field] = None,
        level: Optional[int] = None,
        ghost: Sequence[str] = ...,
        hide_rules: Sequence[str] = ...,
        isolate: Optional[str] = None,
        materials: Optional[Mapping[int, str]] = None,
    ) -> Image:
        """3D view: the first visible cell along each pixel's ray, shaded.

        The camera looks along `direction` and frames `region` (default:
        around every instance of the cells of `isolate`, or the measured
        extent of the contents), or looks from `eye` at `target` with a
        vertical field of view `fov` (degrees). Without `direction`, it
        looks along (0.6, 0.7, -0.4), or with `isolate` from where the cells
        show most (as `llevant render --isolate`). `clip=(point, normal)`
        cuts away the half-space the normal points into. Void cells are
        hidden unless `show_void`; `hide`, `hide_universes` and
        `hide_materials` hide more, as in `first_hit`, and so do `hide_rules`
        (rules of cells, `"u 20040 & m 5"`). `isolate` (a rule) shows only
        its cells, what they hold and the cells holding them. `gpu=True` renders on the GPU (faster,
        approximate at edges); the model is uploaded once. Not on WARP, Windows'
        software adapter: an error there.

        `colour`: `cell`, `material`, `universe` or `realistic` (what the
        materials look like, see `materials()`; lit physically, the others
        flat). `field`: a `Field` of this model drawn over the cells. `level` and `ghost` as
        in `first_hit`: ghosts are drawn as shells over the image.

        `materials`: looks for this image only, `{number: spec}` as for
        `set_materials`, instead of the model's.
        """
    def volumes(
        self, h: float, *, region: Optional[_Region] = None, seed: int = 1, probes: int = 12
    ) -> Volumes:
        """Volumes of everything in `region` (default: the measured extent of
        the problem), from `probes` lattices of lines in random directions,
        as many lines as three axis-parallel families `h` cm apart. Sigma is
        the standard error over the probes. The same `seed` gives the same
        lines, also on another model. Long runs cannot be interrupted with
        Ctrl-C.
        """
    def check(
        self, h: float, *, region: Optional[_Region] = None, seed: int = 1, probes: int = 12
    ) -> list[Group]:
        """Gaps, overlaps, unsupported cells and too-deep nesting found by
        `volumes(h, ...)`, largest first (empty if none).
        """
    def volumes_to(
        self,
        h: float,
        rel_sigma: float,
        *,
        h_min: Optional[float] = None,
        abs_sigma: float = 0.0,
        max_probes: Optional[int] = None,
        by: _GroupBy = "cell",
        region: Optional[_Region] = None,
        seed: int = 1,
        probes: int = 12,
    ) -> list[Group]:
        """Volumes grouped `by` cell, material, universe or label, those inside
        cells refined until sigma / volume is at most `rel_sigma` (halving
        `h` over the boxes their cells can be in, down to `h_min`, default
        h / 64). Every group the model can hold is listed, met or not. Each
        group's `h` says which lines gave it, `converged` whether it got
        there and `outcome` how: "converged", "not converged", "not found"
        or "empty". A group whose sigma is at most `abs_sigma` cm³ (0: none)
        is done too, however large its relative sigma. `max_probes` (default
        `probes`: none added) lets groups a little above the target get more
        probes at one spacing: more converge, for more time.
        """
    def cell_volumes(
        self,
        cells: Sequence[int],
        h: float,
        rel_sigma: Optional[float] = None,
        *,
        h_min: Optional[float] = None,
        abs_sigma: float = 0.0,
        max_probes: int = 48,
        region: Optional[_Region] = None,
        seed: int = 1,
        probes: int = 12,
    ) -> list[Group]:
        """Volumes of the cells `cells` (MCNP numbers) alone, in that order:
        every instance of each together, lines traced only around them; a
        filled cell's volume is that of its region. Each cell starts at the
        spacing that puts 8 lines across its box (at most `h`) and, with
        `rel_sigma`, is refined until sigma / volume is at most that, down to
        `h_min` (default h / 64), or until sigma is at most `abs_sigma` cm³
        (0: none); without it, one pass. Up to `max_probes` (default 48) at
        one spacing. `converged` and `outcome` as for `volumes_to`.
        """
    def fingerprints(self, precision: float = ...) -> Fingerprints:
        """Digests of every cell and universe that ignore numbering and card
        order, at relative `precision`.
        """
    def diff(self, other: Model, precision: float = ...) -> Diff:
        """How `other` differs from this model, cell by cell and universe by
        universe, up to numbering.
        """
    def reload(self) -> Model:
        """Read and compile the file again, with the same options (and the
        same universe, for a model made by `universe`). Only the universes
        that changed are built again: the rest is taken from this model.
        """
    def recompile(self, text: str) -> Model:
        """Compile `text`, a new version of this model's input (an edit, or a
        new assembly made in memory), with the same options (and the same
        universe, for a model made by `universe`). Only the universes that
        changed are built again: the rest is taken from this model. The
        model it gives is not read from a file (`path` is None).
        """
    def universe(self, u: int, cut: Union[Literal["own", "none"], _Region, None] = None) -> Model:
        """Universe `u` as a model of its own, in its own frame, with
        everything it holds; cell and universe numbers stay the same. Cut to
        a box beyond which lies one void `IMP:N=0` cell (`cut_cell`):
        `"own"` (around what the universe holds, measured, wherever the
        model places it), `"none"`, or a box as for `region`.
        """
    @property
    def problem_box(self) -> Optional[_Box]:
        """The problem box: beyond it only `IMP:N=0` cells are checked (usually
        the box of the graveyard sphere; the cut box of a `universe`). None
        without one.
        """
    @property
    def cut_cell(self) -> Optional[int]:
        """The MCNP number of the cell beyond the cut box of a model made by
        `universe` (None if it was not cut).
        """

