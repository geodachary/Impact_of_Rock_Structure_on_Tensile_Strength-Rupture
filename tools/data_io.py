"""Single reader for the experimental specimen table.

``tensile_samples_data.csv`` is rebuilt from the per-replicate measurements in
``selected_all_samples.csv``, and that rebuild has changed the schema more than
once. The change that mattered was the loss of the serial-number column: the
notebooks loaded the table with ``index_col=0``, so dropping ``SN`` silently
promoted ``Rock_type`` to the index and every ``df.loc[sample_id, ...]`` lookup
broke at once.

Reading through one function makes the row index explicit rather than
positional. Specimens are numbered 1--14 in the order the experimental
programme defines them — gneiss 0--90 degrees, then schist 0--90 degrees — which
is what ``SN`` used to carry and what the notebooks index by.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import lithology as lith

TABLE = "tensile_samples_data.csv"

#: Renames applied so a rebuilt table still satisfies older column references.
COLUMN_ALIASES = {
    "UCS (Mpa)": "UCS_(Mpa)",
    "Volume": "Volume_mm^3",
}

#: Columns every consumer of the table relies on.
REQUIRED = ("Rock_type", "Angle", "Diameter_mm", "Thickness_mm", "Load_(KN)",
            "Tensile_strength_Mpa", "Modulus_of_Elasticity", "Poisson_Ratio",
            "Radians", "Cohesion", "Friction_Angle")


REPLICATES = "selected_all_samples.csv"


def load_replicate_table(root=None) -> pd.DataFrame:
    """Per-replicate measurements, with ``Rock_type`` canonicalised.

    The replicate file and the per-angle summary carry the same lithology names
    but not necessarily the same capitalisation, and code that filters either
    one with an equality test is silently emptied by a mismatch. Both tables are
    therefore read through this module so a single spelling reaches the
    analysis.
    """
    root = Path(root or lith.REPO_ROOT)
    df = pd.read_csv(root / REPLICATES, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Rock_type" not in df.columns:
        raise ValueError(f"{REPLICATES} has no Rock_type column")
    # Spreadsheet exports carry trailing rows that are blank in every column.
    # They are not specimens, and mapping them raises on the Rock_type value,
    # so drop them before canonicalising rather than after.
    blank = df.isna().all(axis=1)
    if blank.any():
        df = df.loc[~blank].copy()
    df["Rock_type"] = df["Rock_type"].map(canonical_rock_type)
    return df


def canonical_rock_type(name) -> str:
    """Canonical display spelling for a ``Rock_type`` value.

    The strength table has carried both ``Psammatic`` and ``Psammitic``. Code
    that filters rows with an equality test against one spelling returns an
    empty frame under the other, and because such filters are usually guarded
    by ``if len(...) > 0`` the result is a silently missing series rather than
    an error. Canonicalising once here removes that failure mode.
    """
    key = " ".join(str(name).strip().lower().split())
    token = key.replace(" ", "_")
    token = lith.LEGACY_TOKENS.get(token, token)
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        if token == lit.key:
            return lit.display_name
    raise ValueError(f"unrecognised Rock_type {name!r}")


def load_specimen_table(root=None, indexed=True) -> pd.DataFrame:
    """The specimen table with ``Rock_type`` as a column and specimens 1--14.

    ``indexed=False`` returns a plain range index, for callers that only want
    the columns.
    """
    root = Path(root or lith.REPO_ROOT)
    df = pd.read_csv(root / TABLE, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # accept either spelling of the renamed columns
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
        elif new in df.columns and old not in df.columns:
            df[old] = df[new]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{TABLE} is missing required columns {missing}; "
            "check the rebuild in data_mean.ipynb")

    df["Rock_type"] = df["Rock_type"].map(canonical_rock_type)

    if not indexed:
        return df.reset_index(drop=True)

    if "SN" in df.columns:
        # the table carries its own specimen numbering; use it rather than
        # inferring one, so the index means what the data says it means
        sn = df["SN"].astype(int)
        df = df.drop(columns=["SN"])
        df.index = pd.Index(sn.to_numpy(), name="SN")
        df = df.sort_index()
    else:
        # SN was absent from an earlier rebuild; reconstruct it from the order
        # the experimental programme defines
        df = _order_by_programme(df)
        df.index = pd.RangeIndex(1, len(df) + 1, name="SN")

    if len(df) != 14:
        raise ValueError(f"expected 14 specimens in {TABLE}, found {len(df)}")
    if list(df.index) != list(range(1, 15)):
        raise ValueError(f"specimen numbering is not 1..14: {list(df.index)}")
    return df


def _order_by_programme(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into gneiss 0--90 then schist 0--90, the order the numbering assumes."""
    order = {lith.AUGEN_GNEISS.display_name.lower(): 0,
             lith.PSAMMITIC_SCHIST.display_name.lower(): 1}
    legacy = {"psammatic schist": 1}
    key = df.Rock_type.str.strip().str.lower()
    rank = key.map(lambda k: order.get(k, legacy.get(k)))
    if rank.isna().any():
        unknown = sorted(set(key[rank.isna()]))
        raise ValueError(f"unrecognised Rock_type values: {unknown}")
    out = df.assign(_rank=rank, _angle=df.Angle.round().astype(int))
    out = out.sort_values(["_rank", "_angle"], kind="stable")
    return out.drop(columns=["_rank", "_angle"])
