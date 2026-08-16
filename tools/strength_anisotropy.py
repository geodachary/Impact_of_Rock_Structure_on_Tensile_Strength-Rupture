"""Tensile and compressive strength anisotropy, computed from the replicate data.

The anisotropy ratios shown in the strength-anisotropy figure were previously
hard-coded constants in the plotting code, so they tracked neither dataset and
could not respond to a change in the measurements. This module computes them from
``selected_all_samples.csv``, the per-replicate table, so the figure and the text
follow the data.

The ratio is the plain strength anisotropy over the seven fabric orientations,

.. math::

    A = \\frac{\\max_\\alpha \\bar{\\sigma}(\\alpha)}{\\min_\\alpha \\bar{\\sigma}(\\alpha)},

where :math:`\\bar{\\sigma}(\\alpha)` is the mean over replicates at that
orientation. It is dimensionless and at least one. Tensile strength comes from
the Brazilian tests, compressive strength from the uniaxial compressive tests, so
the two ratios are formed the same way from different measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import lithology as lith

REPLICATES = "selected_all_samples.csv"
TENSILE_COL = "Tensile_strength_Mpa"
COMPRESSIVE_COL = "UCS_(Mpa)"


def _load(root=None) -> pd.DataFrame:
    # read through tools.data_io so Rock_type arrives in one canonical spelling
    from .data_io import load_replicate_table
    df = load_replicate_table(root)
    missing = {TENSILE_COL, COMPRESSIVE_COL, "Rock_type", "Angle"} - set(df.columns)
    if missing:
        raise KeyError(f"{REPLICATES} is missing {sorted(missing)}")
    return df


def per_angle_means(root=None) -> pd.DataFrame:
    """Mean tensile and compressive strength at each lithology and angle."""
    df = _load(root)
    g = df.groupby([df.Rock_type, df.Angle.round().astype(int)])
    out = g[[TENSILE_COL, COMPRESSIVE_COL]].mean().reset_index()
    out.columns = ["lithology", "angle_deg", "tensile_MPa", "compressive_MPa"]
    out["n_replicates"] = g.size().values
    return out


def anisotropy_ratios(root=None) -> pd.DataFrame:
    """Max/min strength ratio per lithology, for both loading modes."""
    m = per_angle_means(root)
    rows = []
    for rock, grp in m.groupby("lithology"):
        t, c = grp.tensile_MPa, grp.compressive_MPa
        rows.append(dict(
            lithology=rock,
            tensile_anisotropy=float(t.max() / t.min()),
            compressive_anisotropy=float(c.max() / c.min()),
            tensile_min_angle=int(grp.loc[t.idxmin(), "angle_deg"]),
            tensile_max_angle=int(grp.loc[t.idxmax(), "angle_deg"]),
            compressive_min_angle=int(grp.loc[c.idxmin(), "angle_deg"]),
            compressive_max_angle=int(grp.loc[c.idxmax(), "angle_deg"]),
            n_angles=int(len(grp)),
        ))
    out = pd.DataFrame(rows)
    out["tensile_exceeds_compressive"] = (
        out.tensile_anisotropy > out.compressive_anisotropy)
    out["relative_separation_pct"] = 100.0 * (
        (out.tensile_anisotropy - out.compressive_anisotropy)
        / out[["tensile_anisotropy", "compressive_anisotropy"]].min(axis=1))
    return out


def verdict(root=None) -> dict:
    """Is the blanket claim 'tensile anisotropy exceeds compressive' supportable?"""
    r = anisotropy_ratios(root)
    return dict(
        n_lithologies=int(len(r)),
        n_tensile_exceeds=int(r.tensile_exceeds_compressive.sum()),
        blanket_claim_supported=bool(r.tensile_exceeds_compressive.all()),
        ordering_is_lithology_dependent=bool(
            0 < r.tensile_exceeds_compressive.sum() < len(r)),
        max_relative_separation_pct=float(r.relative_separation_pct.abs().max()),
    )
