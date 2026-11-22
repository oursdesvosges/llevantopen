"""OpenMC, multi-code translation, statepoint tally, and lost-particle helpers."""

from __future__ import annotations

import csv
import importlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from ._llevant import Field, Model, _translate_path, _translate_text, from_text, load

_INPUT_SUFFIXES = (
    ".xml",
    ".t4",
    ".tripoli",
    ".t4g",
    ".x2m",
    ".c2m",
    ".dragon",
    ".d5",
    ".i",
    ".inp",
    ".mcnp",
)


def _need_h5py() -> Any:
    """Import ``h5py`` on demand so pure-geometry installs do not require it."""
    try:
        return importlib.import_module("h5py")
    except ImportError:
        raise ImportError(
            "reading OpenMC HDF5 statepoints requires h5py: pip install h5py"
        ) from None


def _decode(raw: Any) -> str:
    """Decode an HDF5 string or byte scalar into a trimmed Python ``str``."""
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", errors="ignore").strip()
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        return _decode(raw.item())
    return str(raw).strip()


def to_mcnp(
    source: Union[str, os.PathLike[str]],
    *,
    output: Optional[Union[str, os.PathLike[str]]] = None,
) -> str:
    """Translate an OpenMC XML, TRIPOLI-4, DRAGON5 ``GEO:``, or MCNP input to MCNP text.

    :param source: Path to an input file / OpenMC directory, or raw model text.
    :param output: Optional path where the translated MCNP text is also written.
    """
    if isinstance(source, os.PathLike):
        mcnp = _translate_path(source)
    elif isinstance(source, str):
        stripped = source.lstrip()
        looks_like_path = (
            "\n" not in source
            and len(source) <= 4096
            and (
                os.path.exists(source)
                or (
                    not stripped.startswith("<")
                    and any(source.lower().endswith(ext) for ext in _INPUT_SUFFIXES)
                )
            )
        )
        mcnp = _translate_path(source) if looks_like_path else _translate_text(source)
    else:
        raise TypeError("source must be a path or string of input text")

    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(mcnp, encoding="utf-8")
    return mcnp


def from_openmc(
    obj: Any,
    *,
    tolerance: float = 1e-8,
    world_half_size: float = 1e6,
    grid: str = "auto",
) -> Model:
    """Compile an OpenMC ``Model``, ``Geometry``, ``(Geometry, Materials)``, XML text, or path.

    In-memory OpenMC Python objects are exported into a temporary directory and
    compiled directly via ``from_text`` so ``label "..."`` queries continue to
    work after returning.
    """
    if isinstance(obj, os.PathLike):
        return load(obj, tolerance=tolerance, world_half_size=world_half_size, grid=grid)
    if isinstance(obj, str):
        if obj.lstrip().startswith("<"):
            return from_text(obj, tolerance=tolerance, world_half_size=world_half_size, grid=grid)
        return load(obj, tolerance=tolerance, world_half_size=world_half_size, grid=grid)

    with tempfile.TemporaryDirectory(prefix="llevant_openmc_") as tmpdir:
        tmp = Path(tmpdir)
        if hasattr(obj, "export_to_model_xml"):
            model_xml = tmp / "model.xml"
            try:
                obj.export_to_model_xml(str(model_xml))
            except TypeError:
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    obj.export_to_model_xml()
                finally:
                    os.chdir(old_cwd)
            if model_xml.is_file():
                xml_text = model_xml.read_text(encoding="utf-8")
                return from_text(
                    xml_text, tolerance=tolerance, world_half_size=world_half_size, grid=grid
                )
            mcnp_text = _translate_path(tmp)
            return from_text(
                mcnp_text, tolerance=tolerance, world_half_size=world_half_size, grid=grid
            )

        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            geom, mats = obj
            if not hasattr(geom, "export_to_xml"):
                raise TypeError("expected (geometry, materials) with export_to_xml methods")
            geom.export_to_xml(str(tmp / "geometry.xml"))
            if mats is not None and hasattr(mats, "export_to_xml"):
                mats.export_to_xml(str(tmp / "materials.xml"))
            mcnp_text = _translate_path(tmp)
            return from_text(
                mcnp_text, tolerance=tolerance, world_half_size=world_half_size, grid=grid
            )

        if hasattr(obj, "export_to_xml"):
            obj.export_to_xml(str(tmp / "geometry.xml"))
            mcnp_text = _translate_path(tmp)
            return from_text(
                mcnp_text, tolerance=tolerance, world_half_size=world_half_size, grid=grid
            )

    raise TypeError(
        "expected an openmc.Model, openmc.Geometry, (Geometry, Materials), XML string, or path"
    )


def _mesh_grid_dims(mgrp: Any) -> tuple[str, tuple[int, ...]]:
    """Return ``(mesh_type, dims)`` for an OpenMC ``tallies/meshes/mesh <id>`` group."""
    mtype = _decode(mgrp["type"][()]).lower()
    if mtype == "regular":
        dims = [int(d) for d in np.asarray(mgrp["dimension"][()]).ravel().tolist()]
        while len(dims) < 3:
            dims.append(1)
        return mtype, (dims[0], dims[1], dims[2])
    if mtype == "rectilinear":
        nx = len(mgrp["x_grid"][()]) - 1
        ny = len(mgrp["y_grid"][()]) - 1
        nz = len(mgrp["z_grid"][()]) - 1
        return mtype, (nx, ny, nz)
    if mtype == "cylindrical":
        nr = len(mgrp["r_grid"][()]) - 1
        nphi = len(mgrp["phi_grid"][()]) - 1
        nz = len(mgrp["z_grid"][()]) - 1
        return mtype, (nr, nphi, nz)
    raise ValueError(f"unsupported OpenMC mesh type: {mtype!r}")


def _mesh_geometry_axes(mgrp: Any) -> dict[str, Any]:
    """Return mesh type, dimensions, grid edges, and optional origin."""
    mtype, dims = _mesh_grid_dims(mgrp)
    if mtype == "regular":
        nx, ny, nz = dims
        ll = [float(v) for v in np.asarray(mgrp["lower_left"][()], dtype=np.float64).ravel().tolist()]
        if "upper_right" in mgrp:
            ur = [float(v) for v in np.asarray(mgrp["upper_right"][()], dtype=np.float64).ravel().tolist()]
        elif "width" in mgrp:
            w = [float(v) for v in np.asarray(mgrp["width"][()], dtype=np.float64).ravel().tolist()]
            ur = [ll[i] + w[i] * dims[i] for i in range(len(ll))]
        else:
            raise ValueError("regular OpenMC mesh has neither `upper_right` nor `width`")
        while len(ll) < 3:
            ll.append(-1e30)
            ur.append(1e30)
        return {
            "type": mtype,
            "dims": dims,
            "x": np.linspace(ll[0], ur[0], nx + 1).tolist(),
            "y": np.linspace(ll[1], ur[1], ny + 1).tolist(),
            "z": np.linspace(ll[2], ur[2], nz + 1).tolist(),
        }
    if mtype == "rectilinear":
        return {
            "type": mtype,
            "dims": dims,
            "x": np.asarray(mgrp["x_grid"][()], dtype=np.float64).tolist(),
            "y": np.asarray(mgrp["y_grid"][()], dtype=np.float64).tolist(),
            "z": np.asarray(mgrp["z_grid"][()], dtype=np.float64).tolist(),
        }
    if mtype == "cylindrical":
        return {
            "type": mtype,
            "dims": dims,
            "r": np.asarray(mgrp["r_grid"][()], dtype=np.float64).tolist(),
            "phi": np.asarray(mgrp["phi_grid"][()], dtype=np.float64),
            "z": np.asarray(mgrp["z_grid"][()], dtype=np.float64).tolist(),
            "origin": (
                np.asarray(mgrp["origin"][()], dtype=np.float64).ravel().tolist()
                if "origin" in mgrp
                else None
            ),
        }
    raise ValueError(f"unsupported OpenMC mesh type: {mtype!r}")


def _mesh_field(
    model: Model,
    mgrp: Any,
    vals: np.ndarray,
    field_name: str,
    unit: str,
) -> Field:
    """Build a ``Field.mesh`` from an OpenMC HDF5 mesh group and 1D Fortran-ordered values."""
    info = _mesh_geometry_axes(mgrp)
    mtype = info["type"]
    dims = info["dims"]
    if mtype in ("regular", "rectilinear"):
        arr = np.ascontiguousarray(vals.reshape(dims, order="F"))
        return Field.mesh(
            model,
            arr,
            x=info["x"],
            y=info["y"],
            z=info["z"],
            name=field_name,
            unit=unit,
        )
    nr, nphi, nz = dims
    # OpenMC flattens cylindrical meshes in (r, phi, z) Fortran order;
    # llevant's cylindrical Field.mesh uses C order (r, z, theta) with theta in revolutions.
    arr = np.ascontiguousarray(vals.reshape((nr, nphi, nz), order="F").transpose(0, 2, 1))
    theta = (info["phi"] / (2.0 * np.pi)).tolist()
    return Field.mesh(
        model,
        arr,
        r=info["r"],
        z=info["z"],
        theta=theta,
        origin=info["origin"],
        name=field_name,
        unit=unit,
    )


def _energy_bounds(raw_bins: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
    """Normalize 1D `(N+1,)` energy edges or 2D `(N, 2)` bin pairs into `(lo, hi)` pairs."""
    arr = np.asarray(raw_bins, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(float(row[0]), float(row[1])) for row in arr]
    flat = arr.ravel()
    if flat.size == n_bins + 1:
        return [(float(flat[i]), float(flat[i + 1])) for i in range(n_bins)]
    if flat.size == 2 * n_bins and n_bins > 0:
        pairs = flat.reshape((n_bins, 2))
        return [(float(r[0]), float(r[1])) for r in pairs]
    raise ValueError(f"unexpected EnergyFilter bins shape {arr.shape} for n_bins={n_bins}")


def _resolve_energy_idx(
    raw_bins: np.ndarray,
    n_bins: int,
    energy: Union[int, float, str],
) -> Optional[int]:
    if isinstance(energy, str):
        if energy.lower() == "total":
            return None
        raise ValueError(f"unsupported string energy selector {energy!r} (expected 'total' or number)")
    pairs = _energy_bounds(raw_bins, n_bins)
    if isinstance(energy, int) and not isinstance(energy, bool):
        idx = energy if energy >= 0 else n_bins + energy
        if idx < 0 or idx >= n_bins:
            raise ValueError(f"energy bin index {energy} out of range for {n_bins} bins")
        return idx
    e_val = float(energy)
    for idx, (lo, hi) in enumerate(pairs):
        is_last = idx == len(pairs) - 1
        if (lo <= e_val < hi) or (is_last and lo <= e_val <= hi):
            return idx
    raise ValueError(
        f"energy {e_val} eV outside EnergyFilter range [{pairs[0][0]}, {pairs[-1][1]}]"
    )


def openmc_tallies(path: Union[str, os.PathLike[str]]) -> list[dict[str, Any]]:
    """Inspect all tallies, filters, scores, and nuclides in an OpenMC HDF5 statepoint."""
    h5py = _need_h5py()
    path_str = os.fspath(path)
    out: list[dict[str, Any]] = []
    with h5py.File(path_str, "r") as f:
        if "tallies" not in f:
            raise ValueError(f"{path_str} does not contain a /tallies HDF5 group")
        tallies_grp = f["tallies"]
        if "ids" in tallies_grp.attrs:
            ids = [int(v) for v in np.atleast_1d(tallies_grp.attrs["ids"]).tolist()]
        else:
            ids = sorted(
                int(k.split(" ", 1)[1])
                for k in tallies_grp
                if k.startswith("tally ") and k.split(" ", 1)[1].isdigit()
            )
        filters_grp = tallies_grp.get("filters")
        meshes_grp = tallies_grp.get("meshes")
        for tid in ids:
            tname = f"tally {tid}"
            if tname not in tallies_grp:
                continue
            tgrp = tallies_grp[tname]
            if tgrp.attrs.get("internal"):
                continue
            filter_ids = (
                [int(x) for x in np.atleast_1d(tgrp["filters"][()]).tolist()]
                if "filters" in tgrp
                else []
            )
            fmeta: list[dict[str, Any]] = []
            for fid in filter_ids:
                if filters_grp is None or f"filter {fid}" not in filters_grp:
                    fmeta.append({"id": fid, "type": "unknown", "n_bins": 0})
                    continue
                fgrp = filters_grp[f"filter {fid}"]
                raw_type = fgrp["type"][()] if "type" in fgrp else fgrp.attrs.get("type", b"")
                ftype = _decode(raw_type).lower()
                bins_ds = fgrp["bins"][()] if "bins" in fgrp else None
                if ftype == "mesh" and bins_ds is not None and meshes_grp is not None:
                    mid = int(np.asarray(bins_ds).ravel()[0])
                    mname = f"mesh {mid}"
                    if mname in meshes_grp:
                        mtype, mdims = _mesh_grid_dims(meshes_grp[mname])
                        fmeta.append({
                            "id": fid,
                            "type": "mesh",
                            "n_bins": int(np.prod(mdims)),
                            "mesh_id": mid,
                            "mesh_type": mtype,
                            "dimension": list(mdims),
                        })
                        continue
                bins_arr = np.atleast_1d(bins_ds) if bins_ds is not None else np.empty(0)
                if "n_bins" in fgrp:
                    nb = int(np.asarray(fgrp["n_bins"][()]).item())
                elif bins_ds is None:
                    nb = 0
                elif ftype == "cellinstance":
                    nb = int(bins_arr.shape[0]) if bins_arr.ndim == 2 else int(bins_arr.size // 2)
                elif ftype in ("energy", "energyout") and bins_arr.ndim == 1:
                    nb = max(1, int(bins_arr.size) - 1)
                else:
                    nb = int(bins_arr.shape[0])
                entry: dict[str, Any] = {"id": fid, "type": ftype, "n_bins": nb}
                if bins_ds is not None:
                    if ftype in ("energy", "energyout"):
                        entry["bins"] = _energy_bounds(np.asarray(bins_ds), nb)
                    elif ftype == "particle":
                        entry["bins"] = [_decode(x) for x in np.atleast_1d(bins_ds).tolist()]
                    else:
                        entry["bins"] = np.asarray(bins_ds).tolist()
                fmeta.append(entry)

            scores = (
                [_decode(s) for s in np.atleast_1d(tgrp["score_bins"][()]).tolist()]
                if "score_bins" in tgrp
                else []
            )
            nuclides = (
                [_decode(n) for n in np.atleast_1d(tgrp["nuclides"][()]).tolist()]
                if "nuclides" in tgrp
                else ["total"]
            )
            n_real = (
                int(np.asarray(tgrp["n_realizations"][()]).item())
                if "n_realizations" in tgrp
                else 1
            )
            out.append({
                "id": tid,
                "name": _decode(tgrp["name"][()]) if "name" in tgrp else "",
                "n_realizations": n_real,
                "scores": scores,
                "nuclides": nuclides,
                "filters": fmeta,
            })
    return out


def _extract_tally_data(
    model: Model,
    path: Union[str, os.PathLike[str]],
    *,
    tally: Optional[int] = None,
    score: Optional[Union[str, int]] = None,
    nuclide: Optional[Union[str, int]] = None,
    energy: Optional[Union[int, float, str]] = None,
    particle: Optional[Union[str, int]] = None,
    name: str = "",
    unit: str = "",
) -> dict[str, Any]:
    """Read raw mean and relative-error vectors plus metadata for one OpenMC tally."""
    h5py = _need_h5py()
    path_str = os.fspath(path)

    with h5py.File(path_str, "r") as f:
        if "tallies" not in f:
            raise ValueError(f"{path_str} does not contain a /tallies HDF5 group")
        tallies_grp = f["tallies"]
        if "ids" in tallies_grp.attrs:
            ids = [int(v) for v in np.atleast_1d(tallies_grp.attrs["ids"]).tolist()]
        else:
            ids = sorted(
                int(k.split(" ", 1)[1])
                for k in tallies_grp
                if k.startswith("tally ") and k.split(" ", 1)[1].isdigit()
            )
        if not ids:
            raise ValueError(f"no tallies found in {path_str}")

        filters_grp = tallies_grp.get("filters")
        meshes_grp = tallies_grp.get("meshes")
        search_ids = [tally] if tally is not None else ids

        chosen_tid: Optional[int] = None
        chosen_tgrp: Any = None
        chosen_spatial_idx: Optional[int] = None
        chosen_spatial_type: Optional[str] = None
        chosen_bins: Optional[np.ndarray] = None
        chosen_filter_shapes: list[int] = []
        chosen_energy_info: Optional[tuple[int, np.ndarray, int]] = None
        chosen_particle_info: Optional[tuple[int, list[str]]] = None

        for tid in search_ids:
            tname = f"tally {tid}"
            if tname not in tallies_grp:
                continue
            tgrp = tallies_grp[tname]
            if tgrp.attrs.get("internal"):
                continue

            filter_ids = (
                [int(x) for x in np.atleast_1d(tgrp["filters"][()]).tolist()]
                if "filters" in tgrp
                else []
            )
            if not filter_ids or filters_grp is None:
                continue

            fshapes: list[int] = []
            spatial_idx: Optional[int] = None
            spatial_type: Optional[str] = None
            spatial_bins: Optional[np.ndarray] = None
            energy_info: Optional[tuple[int, np.ndarray, int]] = None
            particle_info: Optional[tuple[int, list[str]]] = None
            valid = True

            for idx, fid in enumerate(filter_ids):
                fname = f"filter {fid}"
                if fname not in filters_grp:
                    valid = False
                    break
                fgrp = filters_grp[fname]
                raw_type = fgrp["type"][()] if "type" in fgrp else fgrp.attrs.get("type", b"")
                ftype = _decode(raw_type).lower()
                if ftype == "mesh" and "bins" in fgrp:
                    if meshes_grp is None:
                        valid = False
                        break
                    mesh_id = int(np.asarray(fgrp["bins"][()]).ravel()[0])
                    mname = f"mesh {mesh_id}"
                    if mname not in meshes_grp:
                        valid = False
                        break
                    _, mdims = _mesh_grid_dims(meshes_grp[mname])
                    nb = int(np.prod(mdims))
                    if spatial_idx is None:
                        spatial_idx = idx
                        spatial_type = "mesh"
                        spatial_bins = np.array([mesh_id], dtype=np.int64)
                else:
                    bins_raw = fgrp["bins"][()] if "bins" in fgrp else np.empty(0)
                    bins_arr = np.atleast_1d(bins_raw)
                    if "n_bins" in fgrp:
                        nb = int(np.asarray(fgrp["n_bins"][()]).item())
                    elif ftype == "cellinstance":
                        arr2 = np.asarray(bins_arr)
                        nb = int(arr2.shape[0]) if arr2.ndim == 2 else int(arr2.size // 2)
                    elif ftype in ("energy", "energyout") and bins_arr.ndim == 1:
                        nb = max(1, int(bins_arr.size) - 1)
                    elif ftype in ("energy", "energyout") and bins_arr.ndim == 2:
                        nb = int(bins_arr.shape[0])
                    else:
                        nb = int(bins_arr.size)

                    if spatial_idx is None:
                        if ftype in ("cell", "material", "distribcell") and bins_arr.size > 0:
                            spatial_idx = idx
                            spatial_type = ftype
                            spatial_bins = np.asarray(bins_arr, dtype=np.int64)
                        elif ftype == "cellinstance" and bins_arr.size > 0:
                            spatial_idx = idx
                            spatial_type = "cellinstance"
                            spatial_bins = np.asarray(bins_arr, dtype=np.int64).reshape(-1, 2)

                    if ftype in ("energy", "energyout") and energy_info is None and bins_arr.size > 0:
                        energy_info = (idx, np.asarray(bins_arr, dtype=np.float64), nb)
                    elif ftype == "particle" and particle_info is None and bins_arr.size > 0:
                        particle_info = (
                            idx,
                            [_decode(b).lower() for b in bins_arr.ravel().tolist()],
                        )
                fshapes.append(max(nb, 1))

            if not valid or spatial_idx is None:
                continue
            if tally is None:
                if energy is not None and energy != "total" and energy_info is None:
                    continue
                if particle is not None and particle != "total" and particle_info is None:
                    continue

            chosen_tid = tid
            chosen_tgrp = tgrp
            chosen_spatial_idx = spatial_idx
            chosen_spatial_type = spatial_type
            chosen_bins = spatial_bins
            chosen_filter_shapes = fshapes
            chosen_energy_info = energy_info
            chosen_particle_info = particle_info
            break

        if (
            chosen_tid is None
            or chosen_tgrp is None
            or chosen_spatial_idx is None
            or chosen_spatial_type is None
            or chosen_bins is None
        ):
            raise ValueError(
                "no supported cell, material, cellinstance, distribcell, or mesh tally filter "
                f"found in {path_str}"
            )

        n_real = (
            float(np.asarray(chosen_tgrp["n_realizations"][()]).item())
            if "n_realizations" in chosen_tgrp
            else 1.0
        )
        if n_real <= 0.0:
            n_real = 1.0

        scores_list = (
            [_decode(s) for s in np.atleast_1d(chosen_tgrp["score_bins"][()]).tolist()]
            if "score_bins" in chosen_tgrp
            else ["score 0"]
        )
        n_scores = max(1, len(scores_list))
        nuclides_list = (
            [_decode(n) for n in np.atleast_1d(chosen_tgrp["nuclides"][()]).tolist()]
            if "nuclides" in chosen_tgrp
            else ["total"]
        )
        n_nuclides = max(1, len(nuclides_list))

        if score is None:
            score_idx = 0
        elif isinstance(score, int) and not isinstance(score, bool):
            score_idx = score
            if score_idx < 0 or score_idx >= n_scores:
                raise ValueError(f"score index {score} out of range for {n_scores} scores")
        elif isinstance(score, str):
            matches = [i for i, s in enumerate(scores_list) if s == score]
            if not matches:
                raise ValueError(
                    f"score {score!r} not found in tally {chosen_tid}; available: {scores_list}"
                )
            score_idx = matches[0]
        else:
            raise TypeError("score must be a score name, integer index, or None")

        nuclide_slice: slice = slice(None)
        if nuclide is not None:
            if isinstance(nuclide, int) and not isinstance(nuclide, bool):
                if nuclide < 0 or nuclide >= n_nuclides:
                    raise ValueError(
                        f"nuclide index {nuclide} out of range for {n_nuclides} nuclides"
                    )
                nuclide_slice = slice(nuclide, nuclide + 1)
            elif isinstance(nuclide, str):
                matches = [i for i, n in enumerate(nuclides_list) if n == nuclide]
                if not matches:
                    raise ValueError(
                        f"nuclide {nuclide!r} not found in tally {chosen_tid}; "
                        f"available: {nuclides_list}"
                    )
                nuclide_slice = slice(matches[0], matches[0] + 1)
            else:
                raise TypeError("nuclide must be a nuclide name, integer index, or None")

        axis_slices: dict[int, slice] = {}
        if energy is not None and energy != "total":
            if chosen_energy_info is None:
                raise ValueError(f"tally {chosen_tid} does not have an energy filter")
            eax, ebins, enb = chosen_energy_info
            eidx = _resolve_energy_idx(ebins, enb, energy)
            if eidx is not None:
                axis_slices[eax] = slice(eidx, eidx + 1)

        if particle is not None and particle != "total":
            if chosen_particle_info is None:
                raise ValueError(f"tally {chosen_tid} does not have a particle filter")
            pax, pnames = chosen_particle_info
            if isinstance(particle, int) and not isinstance(particle, bool):
                if particle < 0 or particle >= len(pnames):
                    raise ValueError(
                        f"particle index {particle} out of range for {len(pnames)} particles"
                    )
                pidx = particle
            elif isinstance(particle, str):
                plower = particle.strip().lower()
                matches = [i for i, p in enumerate(pnames) if p == plower]
                if not matches:
                    raise ValueError(
                        f"particle {particle!r} not found in tally {chosen_tid}; available: {pnames}"
                    )
                pidx = matches[0]
            else:
                raise TypeError("particle must be a particle name, integer index, or None")
            axis_slices[pax] = slice(pidx, pidx + 1)

        score_name = scores_list[score_idx] if score_idx < len(scores_list) else f"score {score_idx}"
        raw_tname = _decode(chosen_tgrp["name"][()]) if "name" in chosen_tgrp else ""
        base_name = raw_tname if raw_tname else f"tally {chosen_tid}"
        default_name = f"{base_name} ({score_name})"
        field_name = name if name else default_name

        if "results" not in chosen_tgrp:
            raise ValueError(f"tally {chosen_tid} in {path_str} has no `results` dataset")
        results = np.asarray(chosen_tgrp["results"][()], dtype=np.float64)
        if results.ndim < 3 or results.shape[-1] < 2:
            raise ValueError(
                f"unexpected OpenMC tally results shape {results.shape} in {path_str}"
            )

        target_shape = tuple(chosen_filter_shapes) + (n_nuclides, n_scores)
        sum_arr = np.reshape(results[..., 0], target_shape)
        sum_sq_arr = np.reshape(results[..., 1], target_shape)

        n_f = len(chosen_filter_shapes)
        indexer: list[slice] = [slice(None)] * sum_arr.ndim
        for ax, sl in axis_slices.items():
            indexer[ax] = sl
        indexer[n_f] = nuclide_slice
        indexer[n_f + 1] = slice(score_idx, score_idx + 1)

        s_sub = sum_arr[tuple(indexer)]
        sq_sub = sum_sq_arr[tuple(indexer)]
        reduce_axes = tuple(i for i in range(s_sub.ndim) if i != chosen_spatial_idx)
        sum_1d = s_sub.sum(axis=reduce_axes) if reduce_axes else s_sub
        sum_sq_1d = sq_sub.sum(axis=reduce_axes) if reduce_axes else sq_sub

        mean = sum_1d / n_real
        if n_real > 1.0:
            var = np.maximum((sum_sq_1d / n_real - mean * mean) / (n_real - 1.0), 0.0)
            std = np.sqrt(var)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.where(mean != 0.0, std / np.abs(mean), np.nan)
        else:
            rel_err = np.full_like(mean, np.nan)

        mesh_info = None
        if chosen_spatial_type == "mesh":
            mesh_id = int(chosen_bins[0])
            assert meshes_grp is not None
            mesh_info = _mesh_geometry_axes(meshes_grp[f"mesh {mesh_id}"])

        return {
            "tally_id": chosen_tid,
            "spatial_type": chosen_spatial_type,
            "bins": chosen_bins,
            "mean": mean,
            "rel_err": rel_err,
            "field_name": field_name,
            "unit": unit,
            "mesh_info": mesh_info,
        }


def _instance_mapping(
    model: Model,
    spatial_type: str,
    bins: np.ndarray,
    vals: np.ndarray,
) -> dict[int, float]:
    """Map ``cellinstance`` or ``distribcell`` tally bins to global ``llevant`` instance ids."""
    if spatial_type == "cellinstance":
        pairs = np.asarray(bins, dtype=np.int64).reshape(-1, 2)
        if pairs.shape[0] != vals.size:
            raise ValueError(
                f"cellinstance filter has {pairs.shape[0]} pairs but tally has {vals.size} bins"
            )
        cache: dict[int, list[int]] = {}
        mapping: dict[int, float] = {}
        for (cid_raw, inst_idx_raw), val in zip(pairs.tolist(), vals.tolist()):
            if not np.isfinite(val):
                continue
            cid = int(cid_raw)
            inst_idx = int(inst_idx_raw)
            if cid not in cache:
                cache[cid] = model.cell_instances(cid)
            c_insts = cache[cid]
            if inst_idx < 0 or inst_idx >= len(c_insts):
                raise ValueError(
                    f"cell {cid} instance index {inst_idx} out of range (model has {len(c_insts)})"
                )
            mapping[int(c_insts[inst_idx])] = float(val)
        return mapping

    # distribcell
    cid = int(np.asarray(bins, dtype=np.int64).ravel()[0])
    c_insts = model.cell_instances(cid)
    if len(c_insts) != vals.size:
        raise ValueError(
            f"distribcell for cell {cid} has {vals.size} tally bins, model has {len(c_insts)} instances"
        )
    return {
        int(gid): float(v)
        for gid, v in zip(c_insts, vals.tolist())
        if np.isfinite(v)
    }


def read_openmc_tally(
    model: Model,
    path: Union[str, os.PathLike[str]],
    *,
    tally: Optional[int] = None,
    score: Optional[Union[str, int]] = None,
    nuclide: Optional[Union[str, int]] = None,
    energy: Optional[Union[int, float, str]] = None,
    particle: Optional[Union[str, int]] = None,
    error: bool = False,
    name: str = "",
    unit: str = "",
) -> Field:
    """Read an OpenMC HDF5 statepoint tally into a ``Field``.

    Supports ``cell``, ``material``, ``cellinstance``, ``distribcell``, and
    ``mesh`` (regular, rectilinear, cylindrical) spatial filters alongside optional
    ``score``, ``nuclide``, ``energy`` (bin index, eV value, or ``"total"``),
    and ``particle`` filter selection. Requires ``h5py``.
    """
    data = _extract_tally_data(
        model,
        path,
        tally=tally,
        score=score,
        nuclide=nuclide,
        energy=energy,
        particle=particle,
        name=name,
        unit=unit,
    )
    vals = data["rel_err"] if error else data["mean"]
    field_name = data["field_name"]
    if error and not name:
        field_name += " rel. error"
    stype = data["spatial_type"]
    bins = data["bins"]

    if stype == "cell":
        mapping = {
            int(cid): float(v)
            for cid, v in zip(bins.tolist(), vals.tolist())
            if np.isfinite(v)
        }
        return Field.cells(model, mapping, name=field_name, unit=unit)

    if stype == "material":
        mapping = {
            int(mid): float(v)
            for mid, v in zip(bins.tolist(), vals.tolist())
            if np.isfinite(v)
        }
        return Field.materials(model, mapping, name=field_name, unit=unit)

    if stype in ("cellinstance", "distribcell"):
        mapping = _instance_mapping(model, stype, bins, vals)
        return Field.instances(model, mapping, name=field_name, unit=unit)

    info = data["mesh_info"]
    assert info is not None
    mtype = info["type"]
    dims = info["dims"]
    if mtype in ("regular", "rectilinear"):
        arr = np.ascontiguousarray(vals.reshape(dims, order="F"))
        return Field.mesh(
            model, arr, x=info["x"], y=info["y"], z=info["z"], name=field_name, unit=unit
        )
    nr, nphi, nz = dims
    arr = np.ascontiguousarray(vals.reshape((nr, nphi, nz), order="F").transpose(0, 2, 1))
    theta = (info["phi"] / (2.0 * np.pi)).tolist()
    return Field.mesh(
        model,
        arr,
        r=info["r"],
        z=info["z"],
        theta=theta,
        origin=info["origin"],
        name=field_name,
        unit=unit,
    )


def export_openmc_tally_csv(
    model: Model,
    path: Union[str, os.PathLike[str]],
    output: Union[str, os.PathLike[str]],
    *,
    tally: Optional[int] = None,
    score: Optional[Union[str, int]] = None,
    nuclide: Optional[Union[str, int]] = None,
    energy: Optional[Union[int, float, str]] = None,
    particle: Optional[Union[str, int]] = None,
) -> Path:
    """Export an OpenMC HDF5 statepoint tally table to CSV."""
    data = _extract_tally_data(
        model,
        path,
        tally=tally,
        score=score,
        nuclide=nuclide,
        energy=energy,
        particle=particle,
    )
    stype = data["spatial_type"]
    bins = data["bins"]
    mean = data["mean"]
    rel_err = data["rel_err"]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if stype in ("cell", "material"):
            w.writerow([stype, "value", "rel_error"])
            for bid, mval, rval in zip(bins.tolist(), mean.tolist(), rel_err.tolist()):
                w.writerow([int(bid), mval, rval])
        elif stype in ("cellinstance", "distribcell"):
            mean_map = _instance_mapping(model, stype, bins, mean)
            err_map = _instance_mapping(model, stype, bins, rel_err)
            w.writerow(["instance", "instance_path", "value", "rel_error"])
            for gid in sorted(mean_map):
                w.writerow([
                    gid,
                    model.instance_path(gid) or "",
                    mean_map[gid],
                    err_map.get(gid, float("nan")),
                ])
        else:
            info = data["mesh_info"]
            assert info is not None
            mtype = info["type"]
            dims = info["dims"]
            m_3d = mean.reshape(dims, order="F")
            e_3d = rel_err.reshape(dims, order="F")
            if mtype in ("regular", "rectilinear"):
                xc = 0.5 * (np.asarray(info["x"][:-1]) + np.asarray(info["x"][1:]))
                yc = 0.5 * (np.asarray(info["y"][:-1]) + np.asarray(info["y"][1:]))
                zc = 0.5 * (np.asarray(info["z"][:-1]) + np.asarray(info["z"][1:]))
                w.writerow(["i", "j", "k", "x", "y", "z", "value", "rel_error"])
                for ix in range(dims[0]):
                    for iy in range(dims[1]):
                        for iz in range(dims[2]):
                            w.writerow([
                                ix,
                                iy,
                                iz,
                                float(xc[ix]),
                                float(yc[iy]),
                                float(zc[iz]),
                                float(m_3d[ix, iy, iz]),
                                float(e_3d[ix, iy, iz]),
                            ])
            else:
                rc = 0.5 * (np.asarray(info["r"][:-1]) + np.asarray(info["r"][1:]))
                pc = 0.5 * (np.asarray(info["phi"][:-1]) + np.asarray(info["phi"][1:]))
                zc = 0.5 * (np.asarray(info["z"][:-1]) + np.asarray(info["z"][1:]))
                w.writerow(["ir", "iphi", "iz", "r", "phi", "z", "value", "rel_error"])
                for ir in range(dims[0]):
                    for iphi in range(dims[1]):
                        for iz in range(dims[2]):
                            w.writerow([
                                ir,
                                iphi,
                                iz,
                                float(rc[ir]),
                                float(pc[iphi]),
                                float(zc[iz]),
                                float(m_3d[ir, iphi, iz]),
                                float(e_3d[ir, iphi, iz]),
                            ])
    return out_path


_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")


def _extract_lost_points(source: Union[str, os.PathLike[str], Sequence[Any], np.ndarray]) -> np.ndarray:
    """Normalize a point, `(N, 3)` array, log text, or log path to an `(N, 3)` float64 array."""
    if isinstance(source, os.PathLike) or (
        isinstance(source, str) and "\n" not in source and len(source) <= 4096 and os.path.isfile(source)
    ):
        text = Path(source).read_text(encoding="utf-8", errors="ignore")
    elif isinstance(source, str):
        text = source
    else:
        arr = np.asarray(source, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError(f"expected (3,) point or (N, 3) array, got shape {arr.shape}")
            return arr.reshape(1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"expected (N, 3) point array, got shape {arr.shape}")
        return arr

    pts: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        tagged = (
            "x, y, z" in lower
            or "x,y,z" in lower
            or "xyz" in lower
            or "position" in lower
            or "lost at" in lower
        )
        if tagged:
            if "=" in line:
                tail = line.split("=", 1)[1]
            elif "lost at" in lower:
                tail = line[lower.index("lost at") + len("lost at") :]
            elif "position" in lower:
                tail = line[lower.index("position") + len("position") :]
            else:
                tail = line
            nums = [float(t.replace("d", "e").replace("D", "E")) for t in _FLOAT_RE.findall(tail)]
            if len(nums) >= 3:
                pts.append(nums[:3])
                continue
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        if len(parts) == 3:
            try:
                pts.append([float(p.replace("d", "e").replace("D", "E")) for p in parts])
            except ValueError:
                pass
    if not pts:
        raise ValueError("no coordinates found in lost-particle input")
    return np.asarray(pts, dtype=np.float64)


def diagnose_lost_particles(
    model: Model,
    points: Union[str, os.PathLike[str], Sequence[Any], np.ndarray],
    *,
    radius: float = 1.0,
    dir: Optional[Sequence[float]] = None,
    plots: Optional[Union[str, os.PathLike[str]]] = None,
    px: int = 400,
) -> list[dict[str, Any]]:
    """Diagnose lost-particle coordinates or log files against ``model``.

    Classifies each point and traces local probe lines of half-length
    ``radius`` (along ``dir`` if given, else +/-X, +/-Y, +/-Z) to detect adjacent
    gaps, overlaps, or surface grazing near the reported coordinate. If
    ``plots`` is given, writes zoomed ``lost_###_{xy,xz,yz}.png`` slices into
    that directory.
    """
    if not (radius > 0.0 and np.isfinite(radius)):
        raise ValueError("radius must be positive")
    pts = _extract_lost_points(points)
    if dir is not None:
        dvec = np.asarray(dir, dtype=np.float64).ravel()
        if dvec.size != 3 or np.linalg.norm(dvec) == 0.0:
            raise ValueError("dir must be a non-zero 3-vector")
        dirs = np.asarray([dvec / np.linalg.norm(dvec)], dtype=np.float64)
    else:
        dirs = np.eye(3, dtype=np.float64)

    plot_dir: Optional[Path] = None
    if plots is not None:
        plot_dir = Path(plots)
        plot_dir.mkdir(parents=True, exist_ok=True)
    px_clamped = max(32, int(px))

    cls = model.classify(pts)
    starts = np.repeat(pts, len(dirs), axis=0) - np.tile(dirs * radius, (len(pts), 1))
    ends = np.repeat(pts, len(dirs), axis=0) + np.tile(dirs * radius, (len(pts), 1))
    tr = model.trace_lines(starts, ends)
    line_ids = np.asarray(tr.line)
    tr_labels = tr.labels
    tr_label_ids = np.asarray(tr.ids)

    # Map segments per line index cleanly.
    n_dirs = len(dirs)
    out: list[dict[str, Any]] = []
    for k in range(len(pts)):
        lbl = cls[k]
        l0 = k * n_dirs
        l1 = l0 + n_dirs
        mask = (line_ids >= l0) & (line_ids < l1)
        nearby: list[str] = []
        for lid in np.unique(tr_label_ids[mask]).tolist():
            seg_lbl = tr_labels[int(lid)]
            if seg_lbl.status not in ("inside", "on-surface"):
                s = str(seg_lbl)
                if s not in nearby:
                    nearby.append(s)

        if lbl.status == "gap":
            diag = "gap (undefined void); no cell covers this coordinate"
        elif lbl.status == "overlap":
            diag = f"overlap between cells {lbl.involved}"
        elif lbl.status == "on-surface":
            diag = f"on surface within model tolerance ({model.tolerance:.1e} cm)"
        elif lbl.status == "inside":
            if nearby:
                diag = f"inside cell {lbl.cell}, adjacent to defects within {radius} cm: {'; '.join(nearby)}"
            else:
                diag = f"inside cell {lbl.cell} (material {lbl.material})"
        else:
            diag = str(lbl)

        plot_files: list[str] = []
        if plot_dir is not None:
            px0, py0, pz0 = float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2])
            reg = (
                (px0 - radius, py0 - radius, pz0 - radius),
                (px0 + radius, py0 + radius, pz0 + radius),
            )
            for tag, axis_kw in (
                ("xy", {"z": pz0}),
                ("xz", {"y": py0}),
                ("yz", {"x": px0}),
            ):
                img = model.slice(
                    **axis_kw,
                    region=reg,
                    width=px_clamped,
                    colour="material",
                    outline=True,
                    outline_by="cell",
                )
                fname = f"lost_{k + 1:03d}_{tag}.png"
                img.save(plot_dir / fname)
                plot_files.append(fname)

        out.append({
            "point": (float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2])),
            "status": lbl.status,
            "cell": lbl.cell,
            "universe": lbl.universe,
            "material": lbl.material,
            "instance_path": lbl.instance_path,
            "involved": list(lbl.involved),
            "nearby_defects": nearby,
            "diagnosis": diag,
            "plots": plot_files,
        })
    return out
