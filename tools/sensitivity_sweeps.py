"""Sensitivity of the failed area to weak-plane spacing and to strength disorder.

Both sweeps ask the same question -- by how much does the failed fraction of the
disc interior change when one structural input is varied -- but they act through
different routes, which is the point of running them separately.

Spacing enters the weak-plane proximity weight and therefore the resistance,
never the elastic field: the orthotropic solution is homogeneous and its
equilibrium and compatibility conditions contain no term in ``s``. That is why
the sweep can vary spacing without re-solving the stress problem, and why the
stress field is computed once per specimen and reused across every level.

Disorder acts on the strength itself. Following the description in the text --
disorder shifts point values of the tensile strength and cohesion *downward* --
each point is multiplied by ``1 - h*u`` with ``u`` uniform on ``[0, 1]``, so the
mean strength falls with the amplitude and a fraction of the interior crosses
from sub-critical to super-critical without any change in the stress or in the
fabric geometry. A mean-preserving alternative is available for comparison and
gives no net change, because most of the interior sits far from the threshold.

``delta_failure`` is reported in percentage points relative to the matrix-only
baseline for spacing, and relative to the undisturbed strengths for disorder, so
both curves start at zero by construction.

The values these functions produce were previously written into a notebook as
literal arrays; nothing recomputed them, and they carried the same provenance
problem as the mode counts in the same cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import fabric_tractions as ft
from .data_io import load_specimen_table
from .ddm._toolkit import failure_mode_map

#: Weak-plane spacings swept, in metres. Zero is the matrix-only baseline.
SPACINGS = np.array([0.000, 0.001, 0.002, 0.003, 0.004])

#: Relative amplitude of the multiplicative strength disorder.
HETEROGENEITY = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

#: Fixed so the disorder realisations are reproducible.
SEED = 20260807

WEAK_T_RATIO, WEAK_C_RATIO = 0.35, 0.60


def _inputs(f, row):
    """Stress field and strengths for one specimen, as the failure map wants them.

    The classifier works on the flat list of interior points, which the export
    stores as ``xg``/``yg``; the gridded copies are for plotting only.
    """
    M = f["M"].astype(bool)
    xg, yg = np.asarray(f["xg"], float), np.asarray(f["yg"], float)
    ones = np.ones(xg.shape, float)
    core = np.hypot(xg, yg) < ft.CORE_FRAC * float(f["R_m"])
    return dict(
        sxx=f["sxx"][M], syy=f["syy"][M], txy=f["txy"][M], s1=f["s1"][M],
        mask=core,
        Tm=ones * float(row["Tensile_strength_Mpa"]),
        Coh=ones * float(row["Cohesion"]),
        Phi=ones * float(np.deg2rad(float(row["Friction_Angle"]))),
        alpha=float(f["alpha_const_rad"]),
        alpha_wp=float(((f["alpha_const_rad"] + np.pi / 2) % np.pi) - np.pi / 2),
        xg=xg, yg=yg)


def _failed_fraction(d, spacing, Tm=None, Coh=None):
    """Fraction of the disc interior classified as failed."""
    kw = dict(alpha_const=d["alpha"], basis="first", util_min=0.98, mixed_band=0.45,
              margin=0.0, n_theta_mc=361, mc_compression_only=True,
              mc_sigma_comp_min=0.0, conf_k=0.65, wing_k=1.00, wing_p=2.0)
    if spacing > 0:
        kw.update(strength_model="weak_plane", weak_T_ratio=WEAK_T_RATIO,
                  weak_C_ratio=WEAK_C_RATIO, weak_phi=None,
                  alpha_wp_line=d["alpha_wp"],
                  x=d["xg"], y=d["yg"],
                  weak_spacing=float(spacing), weak_bandwidth_frac=0.25,
                  weak_floor=0.10, stress_sign_mode="auto")
    else:
        kw.update(strength_model="matrix", stress_sign_mode="auto")
    modes, *_ = failure_mode_map(
        d["sxx"], d["syy"], d["txy"], d["s1"],
        Tm=d["Tm"] if Tm is None else Tm,
        Coh=d["Coh"] if Coh is None else Coh,
        Phi=d["Phi"], **kw)
    sel = modes[d["mask"]]
    return float(np.mean(sel != "no_failure"))


def spacing_sweep(root=None, spacings=None) -> pd.DataFrame:
    """Failed fraction against weak-plane spacing, for every specimen."""
    spacings = SPACINGS if spacings is None else np.asarray(spacings, float)
    df = load_specimen_table(root)
    rows = []
    for p in ft.field_files(root):
        f = ft.load_field(p)
        d = _inputs(f, df.loc[f["sample_id"]])
        base = _failed_fraction(d, 0.0)
        for s in spacings:
            frac = base if s == 0 else _failed_fraction(d, float(s))
            rows.append(dict(sample_id=f["sample_id"], rock=f["rock"],
                             angle_deg=f["angle_deg"], spacing_m=float(s),
                             failed_fraction=frac,
                             delta_failure_pct=100.0 * (frac - base)))
    return pd.DataFrame(rows)


def heterogeneity_sweep(root=None, levels=None, seed=SEED,
                        mean_preserving=False) -> pd.DataFrame:
    """Failed fraction against relative strength disorder, at the native spacing."""
    levels = HETEROGENEITY if levels is None else np.asarray(levels, float)
    df = load_specimen_table(root)
    rows = []
    for p in ft.field_files(root):
        f = ft.load_field(p)
        row = df.loc[f["sample_id"]]
        d = _inputs(f, row)
        s_native = 0.002 if f["sample_id"] >= 8 else 0.010
        rng = np.random.default_rng(seed + int(f["sample_id"]))
        base = _failed_fraction(d, s_native)
        for h in levels:
            if h == 0:
                frac = base
            else:
                if mean_preserving:
                    g = 1.0 + float(h) * rng.uniform(-1.0, 1.0, d["Tm"].shape)
                else:
                    g = 1.0 - float(h) * rng.uniform(0.0, 1.0, d["Tm"].shape)
                g = np.clip(g, 0.05, None)
                frac = _failed_fraction(d, s_native, Tm=d["Tm"] * g, Coh=d["Coh"] * g)
            rows.append(dict(sample_id=f["sample_id"], rock=f["rock"],
                             angle_deg=f["angle_deg"], heterogeneity=float(h),
                             failed_fraction=frac,
                             delta_failure_pct=100.0 * (frac - base)))
    return pd.DataFrame(rows)
