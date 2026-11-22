"""OpenMC HDF5 statepoint tally loader for llevant fields."""

from __future__ import annotations

import importlib
import os
from typing import Any, Optional, Union

import numpy as np

from ._llevant import Field, Model


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


def _mesh_field(
    model: Model,
    mgrp: Any,
    vals: np.ndarray,
    field_name: str,
    unit: str,
) -> Field:
    """Build a ``Field.mesh`` from an OpenMC HDF5 mesh group and 1D Fortran-ordered values."""
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
        x = np.linspace(ll[0], ur[0], nx + 1).tolist()
        y = np.linspace(ll[1], ur[1], ny + 1).tolist()
        z = np.linspace(ll[2], ur[2], nz + 1).tolist()
        arr = np.ascontiguousarray(vals.reshape((nx, ny, nz), order="F"))
        return Field.mesh(model, arr, x=x, y=y, z=z, name=field_name, unit=unit)

    if mtype == "rectilinear":
        x = np.asarray(mgrp["x_grid"][()], dtype=np.float64).tolist()
        y = np.asarray(mgrp["y_grid"][()], dtype=np.float64).tolist()
        z = np.asarray(mgrp["z_grid"][()], dtype=np.float64).tolist()
        arr = np.ascontiguousarray(vals.reshape(dims, order="F"))
        return Field.mesh(model, arr, x=x, y=y, z=z, name=field_name, unit=unit)

    if mtype == "cylindrical":
        nr, nphi, nz = dims
        r = np.asarray(mgrp["r_grid"][()], dtype=np.float64).tolist()
        phi = np.asarray(mgrp["phi_grid"][()], dtype=np.float64)
        z = np.asarray(mgrp["z_grid"][()], dtype=np.float64).tolist()
        origin = (
            np.asarray(mgrp["origin"][()], dtype=np.float64).ravel().tolist()
            if "origin" in mgrp
            else None
        )
        # OpenMC flattens cylindrical meshes in (r, phi, z) Fortran order;
        # llevant's cylindrical Field.mesh uses C order (r, z, theta) with theta in revolutions.
        arr = np.ascontiguousarray(vals.reshape((nr, nphi, nz), order="F").transpose(0, 2, 1))
        theta = (phi / (2.0 * np.pi)).tolist()
        return Field.mesh(
            model,
            arr,
            r=r,
            z=z,
            theta=theta,
            origin=origin,
            name=field_name,
            unit=unit,
        )

    raise ValueError(f"unsupported OpenMC mesh type: {mtype!r}")


def read_openmc_tally(
    model: Model,
    path: Union[str, os.PathLike[str]],
    *,
    tally: Optional[int] = None,
    score: Optional[Union[str, int]] = None,
    error: bool = False,
    name: str = "",
    unit: str = "",
) -> Field:
    """Read an OpenMC HDF5 statepoint tally (cell, material, or mesh filter) into a ``Field``.

    Supports ``cell``, ``material``, and ``mesh`` (regular, rectilinear, and
    cylindrical) tally filters written by OpenMC in ``statepoint.*.h5``.
    Requires ``h5py`` to be installed.

    :param model: Compiled model the tally belongs to.
    :param path: Path to the OpenMC statepoint ``.h5`` file.
    :param tally: OpenMC tally id; defaults to the first tally with a supported
        cell, material, or mesh filter.
    :param score: Score name (for example ``"flux"``) or 0-based score index;
        defaults to the first score.
    :param error: Return relative standard errors instead of mean tally values.
    :param name: Override the field name shown on legends.
    :param unit: Optional physical unit shown on legends.
    """
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
                    bins_arr = np.atleast_1d(fgrp["bins"][()]) if "bins" in fgrp else np.empty(0)
                    nb = (
                        int(np.asarray(fgrp["n_bins"][()]).item())
                        if "n_bins" in fgrp
                        else int(bins_arr.size)
                    )
                    if spatial_idx is None and ftype in ("cell", "material") and bins_arr.size > 0:
                        spatial_idx = idx
                        spatial_type = ftype
                        spatial_bins = np.asarray(bins_arr, dtype=np.int64)
                fshapes.append(max(nb, 1))

            if valid and spatial_idx is not None:
                chosen_tid = tid
                chosen_tgrp = tgrp
                chosen_spatial_idx = spatial_idx
                chosen_spatial_type = spatial_type
                chosen_bins = spatial_bins
                chosen_filter_shapes = fshapes
                break

        if (
            chosen_tid is None
            or chosen_tgrp is None
            or chosen_spatial_idx is None
            or chosen_spatial_type is None
            or chosen_bins is None
        ):
            raise ValueError(
                f"no supported cell, material, or mesh tally filter found in {path_str}"
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
        n_nuclides = (
            max(1, int(np.atleast_1d(chosen_tgrp["nuclides"][()]).size))
            if "nuclides" in chosen_tgrp
            else 1
        )

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

        score_name = scores_list[score_idx] if score_idx < len(scores_list) else f"score {score_idx}"
        raw_tname = _decode(chosen_tgrp["name"][()]) if "name" in chosen_tgrp else ""
        base_name = raw_tname if raw_tname else f"tally {chosen_tid}"
        default_name = f"{base_name} ({score_name})"
        if error:
            default_name += " rel. error"
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

        # Slice out the chosen score, then collapse non-spatial filter axes and nuclide axis.
        score_selector = (slice(None),) * (sum_arr.ndim - 1) + (score_idx,)
        sum_slice = sum_arr[score_selector]
        sum_sq_slice = sum_sq_arr[score_selector]
        reduce_axes = tuple(i for i in range(sum_slice.ndim) if i != chosen_spatial_idx)
        sum_1d = sum_slice.sum(axis=reduce_axes) if reduce_axes else sum_slice
        sum_sq_1d = sum_sq_slice.sum(axis=reduce_axes) if reduce_axes else sum_sq_slice

        mean = sum_1d / n_real
        if error:
            if n_real > 1.0:
                var = np.maximum((sum_sq_1d / n_real - mean * mean) / (n_real - 1.0), 0.0)
                std = np.sqrt(var)
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = np.where(mean != 0.0, std / np.abs(mean), np.nan)
            else:
                vals = np.full_like(mean, np.nan)
        else:
            vals = mean

        if chosen_spatial_type == "cell":
            mapping = {
                int(cid): float(v)
                for cid, v in zip(chosen_bins.tolist(), vals.tolist())
                if np.isfinite(v)
            }
            return Field.cells(model, mapping, name=field_name, unit=unit)

        if chosen_spatial_type == "material":
            mapping = {
                int(mid): float(v)
                for mid, v in zip(chosen_bins.tolist(), vals.tolist())
                if np.isfinite(v)
            }
            return Field.materials(model, mapping, name=field_name, unit=unit)

        mesh_id = int(chosen_bins[0])
        assert meshes_grp is not None
        mgrp = meshes_grp[f"mesh {mesh_id}"]
        return _mesh_field(model, mgrp, vals, field_name, unit)
