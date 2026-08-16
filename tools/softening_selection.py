"""Selection of the damage-softening factors from the mixed-mode response.

The classifier used for this sweep softens local strength in proportion to
stored elastic energy, with the softening steered by the same fabric weights the
failure map uses. Two caps control it: ``kC_max`` on cohesion and ``kT_max`` on
tensile resistance. Only ``kC_max`` is swept; ``kT_max`` is tied to it through
:data:`KT_RATIO`, so there is one free quantity rather than two.

The value is fixed by a stated rule rather than by reading a plateau off a
curve. Sweeping ``kC_max`` over :data:`DEFAULT_GRID`, the adopted value is the
smallest one whose mean mixed-mode fraction reaches within :data:`REL_TOL` of
the largest value attained anywhere in the sweep. Taking the smallest qualifying
value rather than the argmax matters because the curve saturates: the argmax
would sit wherever the sweep happened to stop, whereas the near-peak rule
returns the least aggressive softening consistent with the observed response.

This lived only inside a notebook cell, which meant the reported values could
not be checked without running the whole notebook, and the manuscript quoted
numbers that no longer matched. Everything needed to reproduce them is here, and
the accompanying regression test pins the result.

The metric is the mixed fraction among failed points, ``mixed / (tensile +
shear + mixed)``, averaged over the seven orientations of a lithology.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import lithology as lith
from . import output_dirs
from .data_io import load_specimen_table
from .ddm._toolkit import (
    choose_smallest_near_peak,
    load_sample_fields_physical,
    npz_path_for_sample,
    summarize_modes,
)

FIELD_DIR = output_dirs.FIELDS_NPZ

#: Tensile cap as a fraction of the cohesion cap; not independently swept.
KT_RATIO = 0.35

#: Exponents shaping how energy, weak-plane proximity and shear dominance
#: drive local damage.
P_U, P_WP, P_S = 1.25, 1.20, 1.10

#: Percentiles for the robust normalisation of stored energy to [0, 1].
U_QLOW, U_QHIGH = 20.0, 95.0

#: Fractional tolerance defining "near peak" in the selection rule.
REL_TOL = 0.02

#: Swept values of the cohesion cap.
DEFAULT_GRID = np.linspace(0.05, 0.90, 18)

#: The model is not defined outside this range.
K_LIMITS = (0.0, 0.95)


def ensure_radians(phi):
    """Friction angles, in radians whichever unit they arrive in."""
    phi = np.asarray(phi, float)
    vals = phi[np.isfinite(phi)]
    if vals.size == 0:
        return phi
    return np.deg2rad(phi) if np.nanmax(np.abs(vals)) > (np.pi + 1e-6) else phi


def robust_unit_interval(x, qlo=U_QLOW, qhi=U_QHIGH, eps=1e-12):
    """Percentile normalisation to [0, 1], insensitive to the platen spikes."""
    x = np.asarray(x, float)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, float)
    lo = float(np.percentile(vals, qlo))
    hi = float(np.percentile(vals, qhi))
    if hi <= lo + eps:
        return np.zeros_like(x, float)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def classify_local_damage(s1, s3, U, T0, C0, phi0, tensile_w, wp_weight, conf_w,
                          kC_max, kT_max=None, *, phi_drop_deg=0.0,
                          mc_compression_only=True):
    """Failure mode at every point under energy-driven local softening.

    Returns an object array of ``no_failure``/``tensile``/``shear``/``mixed``.
    Points failing both criteria are ``mixed``; that is the definition used for
    the metric, and it is deliberately not the near-tie test used by the
    four-mechanism classifier elsewhere.
    """
    s1 = np.asarray(s1, float)
    s3 = np.asarray(s3, float)
    smax, smin = np.maximum(s1, s3), np.minimum(s1, s3)
    s1, s3 = smax, smin

    lo, hi = K_LIMITS
    if not (lo <= float(kC_max) <= hi):
        raise ValueError(f"kC_max={kC_max} outside the valid range {K_LIMITS}")
    kC_max = float(kC_max)
    kT_max = KT_RATIO * kC_max if kT_max is None else float(kT_max)
    if not (lo <= kT_max <= hi):
        raise ValueError(f"kT_max={kT_max} outside the valid range {K_LIMITS}")

    phi0 = ensure_radians(phi0)
    wt = np.clip(np.asarray(tensile_w, float), 0.0, 1.0)
    wwp = np.clip(np.asarray(wp_weight, float), 0.0, 1.0)
    wc = np.clip(np.asarray(conf_w, float), 0.0, 1.0)
    ws = 1.0 - wt

    Uhat = robust_unit_interval(np.asarray(U, float), qlo=U_QLOW, qhi=U_QHIGH)

    Dt = kT_max * (Uhat ** P_U) * (0.25 + 0.75 * wt)
    Ds = (kC_max * (Uhat ** P_U)
          * (0.20 + 0.80 * (wwp ** P_WP))
          * (0.20 + 0.80 * (ws ** P_S))
          * (0.50 + 0.50 * wc))
    Dt = np.clip(Dt, 0.0, 0.95)
    Ds = np.clip(Ds, 0.0, 0.95)

    T0 = np.asarray(T0, float)
    C0 = np.asarray(C0, float)
    T_eff = np.maximum(0.05 * T0, T0 * (1.0 - Dt))
    C_eff = np.maximum(0.05 * C0, C0 * (1.0 - Ds))

    if abs(float(phi_drop_deg)) > 0.0:
        phi_eff = np.clip(phi0 - np.deg2rad(float(phi_drop_deg)) * Ds,
                          np.deg2rad(5.0), np.deg2rad(85.0))
    else:
        phi_eff = np.asarray(phi0, float).copy()

    tensile_fail = s1 >= T_eff
    f_mc = (s1 - s3) + (s1 + s3) * np.sin(phi_eff) - 2.0 * C_eff * np.cos(phi_eff)
    shear_fail = f_mc >= 0.0
    if mc_compression_only:
        sigma_n_crit = 0.5 * (s1 + s3) - 0.5 * (s1 - s3) * np.sin(phi_eff)
        shear_fail = shear_fail & (sigma_n_crit <= 0.0)

    modes = np.full(s1.shape, "no_failure", dtype=object)
    modes[tensile_fail & ~shear_fail] = "tensile"
    modes[shear_fail & ~tensile_fail] = "shear"
    modes[tensile_fail & shear_fail] = "mixed"
    return modes


def metric_curve(sample_ids, grid=None, root=None) -> np.ndarray:
    """Mean mixed-among-failed fraction at each swept ``kC_max``."""
    grid = DEFAULT_GRID if grid is None else np.asarray(grid, float)
    lo, hi = K_LIMITS
    if np.any(grid < lo) or np.any(grid > hi):
        raise ValueError(f"swept values must lie within {K_LIMITS}")

    root = Path(root or lith.REPO_ROOT)
    df = load_specimen_table(root)
    sub = df.loc[list(sample_ids)].sort_values("Angle")

    per_sample = []
    for sid, row in sub.iterrows():
        path = npz_path_for_sample(str(root / FIELD_DIR), sid)
        if not Path(path).exists():
            raise FileNotFoundError(f"missing exported field for specimen {sid}: {path}")
        s1, s3, U, wt, wwp, wc, T0, C0, phi0 = load_sample_fields_physical(path, row)
        curve = []
        for k in grid:
            modes = classify_local_damage(s1, s3, U, T0, C0, phi0, wt, wwp, wc,
                                          kC_max=float(k))
            c = summarize_modes(modes)
            curve.append(c["mixed"] / float(c["failed_total"])
                         if c["failed_total"] > 0 else np.nan)
        per_sample.append(curve)
    return np.nanmean(np.asarray(per_sample, float), axis=0)


def select_for_lithology(sample_ids, grid=None, rel_tol=REL_TOL, root=None) -> dict:
    """Adopted softening caps for one lithology, with the curve behind them."""
    grid = DEFAULT_GRID if grid is None else np.asarray(grid, float)
    curve = metric_curve(sample_ids, grid, root=root)
    idx, kC, peak, at_upper = choose_smallest_near_peak(grid, curve, rel_tol=rel_tol)
    return dict(kC_max=float(kC), kT_max=float(KT_RATIO * kC),
                index=int(idx), peak_metric=float(peak),
                metric_at_selection=float(curve[idx]),
                selection_hit_upper_bound=bool(at_upper),
                grid=grid, curve=curve, rel_tol=float(rel_tol))


def select_all(root=None) -> pd.DataFrame:
    """Adopted caps for both lithologies."""
    rows = []
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        r = select_for_lithology(lit.sample_ids, root=root)
        rows.append(dict(lithology=lit.display_name, kC_max=r["kC_max"],
                         kT_max=r["kT_max"], peak_metric=r["peak_metric"],
                         metric_at_selection=r["metric_at_selection"],
                         hit_upper_bound=r["selection_hit_upper_bound"]))
    return pd.DataFrame(rows)
