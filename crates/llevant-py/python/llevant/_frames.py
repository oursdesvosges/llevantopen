"""Data frames of llevant results (Polars or pandas, imported on use).

The columns come from Rust as numpy arrays; labels and statuses become
categorical columns, with null where there is none (a ray that hits nothing).
"""

import importlib

from ._llevant import STATUS_NAMES, Volumes


def _need(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(
            f"this needs {name}: pip install {name} (or llevant[{name}])"
        ) from None


def _groups(v, by):
    rows = v.by(by)
    nan = float("nan")
    box = [g.bbox or ((nan,) * 3, (nan,) * 3) for g in rows]
    return {
        by: [g.number for g in rows],
        "name": [g.name for g in rows],
        "status": [g.status for g in rows],
        "volume": [g.volume for g in rows],
        "sigma": [g.sigma for g in rows],
        "mass_g": [g.mass for g in rows],
        "mass_sigma_g": [g.mass_sigma for g in rows],
        "segments": [g.segments for g in rows],
        "probes_hit": [g.probes_hit for g in rows],
        "x0": [b[0][0] for b in box],
        "y0": [b[0][1] for b in box],
        "z0": [b[0][2] for b in box],
        "x1": [b[1][0] for b in box],
        "y1": [b[1][1] for b in box],
        "z1": [b[1][2] for b in box],
    }


def to_polars(obj, by=None):
    pl = _need("polars")
    if isinstance(obj, Volumes):
        return pl.DataFrame(_groups(obj, by), strict=False)
    cols = dict(obj._columns())
    names = [str(l) for l in obj.labels]
    label = pl.Series("label", cols.pop("label"))
    status = pl.Series("status", cols.pop("status"))
    df = pl.DataFrame(cols)

    def categorical(name, codes, cats):
        idx = pl.select(pl.when(codes >= 0).then(codes)).to_series()
        return pl.Series(name, cats, dtype=pl.Enum(cats)).gather(idx)

    return df.with_columns(
        categorical("label", label, names),
        categorical("status", status, list(STATUS_NAMES)),
    ).select(["label", "status", *cols])


def to_pandas(obj, by=None):
    pd = _need("pandas")
    if isinstance(obj, Volumes):
        return pd.DataFrame(_groups(obj, by))
    cols = dict(obj._columns())
    cols["label"] = pd.Categorical.from_codes(
        cols["label"], [str(l) for l in obj.labels], validate=False
    )
    cols["status"] = pd.Categorical.from_codes(
        cols["status"], list(STATUS_NAMES), validate=False
    )
    first = ["label", "status"]
    return pd.DataFrame(cols)[first + [c for c in cols if c not in first]]
