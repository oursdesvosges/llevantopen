"""High-level Python wrappers for TRIPOLI-4, DRAGON5, OpenMC plots, QA reports, and mass balances."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

from ._llevant import (
    Model,
    _dragon_to_mcnp,
    _read_openmc_plots,
    _tripoli_to_mcnp,
    from_text,
    load,
)


def _read_source_text(
    source: Union[str, os.PathLike[str]],
    suffixes: tuple[str, ...],
) -> str:
    """Read raw input text when given a path, or return inline text directly."""
    if isinstance(source, os.PathLike):
        return Path(source).read_text(encoding="utf-8", errors="replace")
    if isinstance(source, str):
        looks_like_path = (
            "\n" not in source
            and len(source) <= 4096
            and (
                os.path.exists(source)
                or any(source.lower().endswith(ext) for ext in suffixes)
            )
        )
        if looks_like_path:
            return Path(source).read_text(encoding="utf-8", errors="replace")
        return source
    raise TypeError("source must be a file path or input text string")


def _write_optional(mcnp: str, output: Optional[Union[str, os.PathLike[str]]]) -> None:
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(mcnp, encoding="utf-8")


def tripoli_to_mcnp(
    source: Union[str, os.PathLike[str]],
    *,
    output: Optional[Union[str, os.PathLike[str]]] = None,
) -> str:
    """Translate a TRIPOLI-4 deck (file path or text) strictly into canonical MCNP text."""
    raw = _read_source_text(source, (".t4", ".tripoli", ".t4g"))
    mcnp = _tripoli_to_mcnp(raw)
    _write_optional(mcnp, output)
    return mcnp


def from_tripoli(
    source: Union[str, os.PathLike[str]],
    *,
    tolerance: float = 1e-8,
    world_half_size: float = 1e6,
    grid: Literal["auto", "always", "off"] = "auto",
) -> Model:
    """Translate a TRIPOLI-4 deck (path or text) and compile it as a :class:`Model`."""
    mcnp = tripoli_to_mcnp(source)
    return from_text(
        mcnp, tolerance=tolerance, world_half_size=world_half_size, grid=grid
    )


def dragon_to_mcnp(
    source: Union[str, os.PathLike[str]],
    *,
    output: Optional[Union[str, os.PathLike[str]]] = None,
) -> str:
    """Translate a DRAGON5 ``GEO:`` procedure (path or text) strictly into MCNP text.

    Validates lattice and ring ``MIX`` material counts against the declared cell
    structure.
    """
    raw = _read_source_text(source, (".x2m", ".c2m", ".dragon", ".d5"))
    mcnp = _dragon_to_mcnp(raw)
    _write_optional(mcnp, output)
    return mcnp


def from_dragon(
    source: Union[str, os.PathLike[str]],
    *,
    tolerance: float = 1e-8,
    world_half_size: float = 1e6,
    grid: Literal["auto", "always", "off"] = "auto",
) -> Model:
    """Translate a DRAGON5 ``GEO:`` procedure (path or text) and compile it as a :class:`Model`."""
    mcnp = dragon_to_mcnp(source)
    return from_text(
        mcnp, tolerance=tolerance, world_half_size=world_half_size, grid=grid
    )


def openmc_plots(
    model_path: Union[str, os.PathLike[str]],
    *,
    plots_xml: Optional[Union[str, os.PathLike[str]]] = None,
) -> list[dict[str, Any]]:
    """Parse OpenMC ``<plot>`` definitions from ``model.xml`` or ``plots.xml``."""
    return _read_openmc_plots(model_path, plots_xml)


def render_openmc_plots(
    model: Union[Model, str, os.PathLike[str]],
    out_dir: Union[str, os.PathLike[str]] = ".",
    *,
    plots_xml: Optional[Union[str, os.PathLike[str]]] = None,
    px: Optional[int] = None,
    ids: Optional[Sequence[int]] = None,
    outlines: bool = False,
) -> list[Path]:
    """Render OpenMC ``<plot>`` 2D slice definitions to PNG files in ``out_dir``."""
    if isinstance(model, Model):
        m = model
        src_path: Optional[Union[str, os.PathLike[str]]] = plots_xml or model.path
        if src_path is None:
            raise ValueError(
                "pass plots_xml= when rendering plots for an in-memory Model without a file path"
            )
    else:
        src_path = Path(model)
        m = load(src_path)

    plots = openmc_plots(src_path, plots_xml=plots_xml)
    if ids is not None:
        want = set(ids)
        plots = [p for p in plots if p["id"] in want]
    if not plots:
        raise ValueError("no matching OpenMC <plot> entries found")

    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for p in plots:
        if p["type"] != "slice":
            continue
        wx, wy = float(p["width"][0]), float(p["width"][1])
        if not (wx > 0.0 and wy > 0.0 and math.isfinite(wx) and math.isfinite(wy)):
            raise ValueError(f"plot {p['id']}: invalid width {p['width']!r}")
        ox, oy, oz = (float(c) for c in p["origin"])
        basis = str(p["basis"]).lower()
        sx: Optional[float] = None
        sy: Optional[float] = None
        sz: Optional[float] = None
        if basis == "xy":
            sz = oz
            reg = ((ox - 0.5 * wx, oy - 0.5 * wy, oz - 1.0), (ox + 0.5 * wx, oy + 0.5 * wy, oz + 1.0))
        elif basis == "xz":
            sy = oy
            reg = ((ox - 0.5 * wx, oy - 1.0, oz - 0.5 * wy), (ox + 0.5 * wx, oy + 1.0, oz + 0.5 * wy))
        elif basis == "yz":
            sx = ox
            reg = ((ox - 1.0, oy - 0.5 * wx, oz - 0.5 * wy), (ox + 1.0, oy + 0.5 * wx, oz + 0.5 * wy))
        else:
            raise ValueError(f"plot {p['id']}: unsupported basis {p['basis']!r}")

        width_px = max(1, int(px if px is not None else p["pixels"][0]))
        color_by = str(p["color_by"]).lower()
        colour: Literal["cell", "material", "universe"] = (
            "cell"
            if color_by == "cell"
            else ("universe" if color_by == "universe" else "material")
        )
        outline_by: Literal["cell", "material", "universe"] = colour

        img = m.slice(
            x=sx,
            y=sy,
            z=sz,
            region=reg,
            width=width_px,
            colour=colour,
            outline=outlines,
            outline_by=outline_by,
        )

        raw_name = Path(str(p["filename"]).strip()).name
        for ext in (".png", ".ppm"):
            if raw_name.endswith(ext):
                stem = raw_name[: -len(ext)]
                break
        else:
            stem = raw_name
        fname = f"{stem}.png" if stem else f"plot_{p['id']}.png"
        dest = out_path_dir / fname
        img.save(dest)
        written.append(dest)

    if not written:
        raise ValueError("no slice plots were rendered")
    return written


def check_report(
    model: Model,
    h: float,
    out_dir: Union[str, os.PathLike[str]],
    *,
    region: Optional[Any] = None,
    seed: int = 1,
    probes: int = 12,
    max_plots: int = 10,
    px: int = 600,
) -> dict[str, Any]:
    """Run ``model.check(h)`` and write ``report.json``, ``report.md``, and defect PNGs."""
    problems = model.check(h, region=region, seed=seed, probes=probes)
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    px_clamped = max(32, int(px))

    defects: list[dict[str, Any]] = []
    md_lines = [
        "# Geometry QA Report",
        "",
        f"- Sampling spacing `h`: `{h}` cm",
        f"- Defects found: `{len(problems)}`",
        f"- Compile diagnostics: `{len(model.diagnostics)}`",
        "",
    ]
    if not problems:
        md_lines.append("No gaps or overlaps found.")
    else:
        md_lines.append("| # | Status | Label | Volume (cm3) | Sigma (cm3) | Plots |")
        md_lines.append("|---|---|---|---|---|---|")

    for rank, g in enumerate(problems):
        png_names: list[str] = []
        if rank < max_plots and g.bbox is not None:
            lo, hi = g.bbox
            if all(math.isfinite(c) for pt in (lo, hi) for c in pt):
                mid = [0.5 * (lo[k] + hi[k]) for k in range(3)]
                span = max(
                    hi[0] - lo[0],
                    hi[1] - lo[1],
                    hi[2] - lo[2],
                    h * 2.0,
                    0.5,
                )
                reg = (
                    tuple(min(lo[k], mid[k] - 0.5 * span) - 0.25 * span for k in range(3)),
                    tuple(max(hi[k], mid[k] + 0.5 * span) + 0.25 * span for k in range(3)),
                )
                for tag, sx, sy, sz in (
                    ("xy", None, None, mid[2]),
                    ("xz", None, mid[1], None),
                    ("yz", mid[0], None, None),
                ):
                    img = model.slice(
                        x=sx,
                        y=sy,
                        z=sz,
                        region=reg,
                        width=px_clamped,
                        colour="material",
                        outline=True,
                        outline_by="cell",
                    )
                    fname = f"defect_{rank + 1:03d}_{tag}.png"
                    img.save(out_path_dir / fname)
                    png_names.append(fname)

        links = ", ".join(f"[{f}]({f})" for f in png_names) if png_names else "-"
        safe_name = g.name.replace("`", "'")
        md_lines.append(
            f"| {rank + 1} | {g.status} | `{safe_name}` | {g.volume:.4e} | {g.sigma:.2e} | {links} |"
        )
        defects.append({
            "status": g.status,
            "name": g.name,
            "volume": float(g.volume),
            "sigma": float(g.sigma),
            "bbox": g.bbox,
            "plots": png_names,
        })

    for rank, row in enumerate(defects):
        if row["plots"]:
            md_lines.append("")
            md_lines.append(f"## Defect {rank + 1}: `{row['name']}`")
            md_lines.append("")
            for fname in row["plots"]:
                md_lines.append(f"![{fname}]({fname})")

    report_json_path = out_path_dir / "report.json"
    report_md_path = out_path_dir / "report.md"
    report_json_path.write_text(json.dumps(defects, indent=2), encoding="utf-8")
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "h": float(h),
        "defects": defects,
        "diagnostics": [str(d) for d in model.diagnostics],
        "out_dir": str(out_path_dir),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
    }


def mass_balance(
    model: Model,
    h: float,
    *,
    by: str = "material",
    region: Optional[Any] = None,
    seed: int = 1,
    probes: int = 12,
) -> dict[str, Any]:
    """Compute stochastic volumes and masses (g / kg) grouped by cell, material, universe, or label."""
    vols = model.volumes(h, region=region, seed=seed, probes=probes)
    groups = vols.by(by)  # type: ignore[arg-type]

    total_volume = sum(float(g.volume) for g in groups)
    known_masses = [g for g in groups if g.mass is not None]
    total_mass_g = sum(float(g.mass) for g in known_masses if g.mass is not None)
    total_mass_sigma_g = math.sqrt(
        sum(float(g.mass_sigma or 0.0) ** 2 for g in known_masses)
    )

    return {
        "by": by,
        "h": float(vols.h),
        "total_volume_cm3": total_volume,
        "total_mass_g": total_mass_g,
        "total_mass_kg": total_mass_g / 1000.0,
        "total_mass_sigma_g": total_mass_sigma_g,
        "groups": [
            {
                "number": g.number,
                "name": g.name,
                "status": g.status,
                "volume_cm3": float(g.volume),
                "volume_sigma_cm3": float(g.sigma),
                "mass_g": g.mass,
                "mass_sigma_g": g.mass_sigma,
            }
            for g in groups
        ],
    }
