"""Model anisotropy ratio from the solved principal stresses.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 49 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import numpy as np
# <<PRELUDE_IMPORTS>>


# --- implementation ---------------------------------------------------


def principal_stresses_2d(sxx, syy, txy):
    s_avg = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5*(sxx - syy))**2 + txy**2)
    s1 = s_avg + rad
    s3 = s_avg - rad
    return s1, s3


def compute_model_anisotropy_ratio(
    diameter, thickness, base_tensile_strength_mpa,
    angles_deg, spacing_m, P_required_N,
    *,
    # we must provide the stress function used elsewhere:
    stress_solver_fn,
    # controls
    grid_points=40000,
    seed=0,
    exclude_boundary_frac=0.04,
    reference_angle_deg=0.0,
    aggregation="p99",  # "max" or "p99" for robustness
):
    """
    Model-based anisotropy ratio computed from stress-field amplification.

    Parameters
    ----------
    diameter, thickness : float
        Disk geometry (m).
    base_tensile_strength_mpa : float
        Reference tensile strength (MPa).
    angles_deg : array-like
        Angles (deg) to evaluate.
    spacing_m : float
        Weak-plane spacing (m) passed into stress model.
    P_required_N : float
        Load used in stress model (N).
    stress_solver_fn : callable
        Function that computes stresses for a given angle:
            sxx, syy, txy = stress_solver_fn(diameter, thickness, P_required_N, angle_rad, spacing_m, grid_points, seed)
        It must return arrays (same length) in MPa (recommended) or Pa (consistent).

    grid_points, seed : sampling controls (if the stress function uses sampling)
    exclude_boundary_frac : exclude outer ring to avoid boundary/contact artifacts
    reference_angle_deg : angle used as reference for scaling (default 0°)
    aggregation : tensile amplification measure:
        - "max" uses max(sigma1_plus)
        - "p99" uses 99th percentile (more stable)

    Returns
    -------
    anisotropy_ratio : float
        max(T_eff) / min(T_eff)
    angles_deg : ndarray
    T_eff_mpa : ndarray
        Effective tensile strength predicted for each angle (MPa)
    amp : ndarray
        Tensile amplification metric for each angle (same units as stresses)
    """

    angles_deg = np.asarray(angles_deg, float)
    angles_rad = np.deg2rad(angles_deg)

    # compute amplification metric for each angle
    amp = np.zeros_like(angles_rad, dtype=float)

    for i, ang in enumerate(angles_rad):
        sxx, syy, txy, x, y = stress_solver_fn(
            diameter, thickness, P_required_N, ang, spacing_m,
            grid_points=grid_points, seed=seed + i
        )

        sxx = np.asarray(sxx, float)
        syy = np.asarray(syy, float)
        txy = np.asarray(txy, float)
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        r = np.hypot(x, y)
        R = diameter / 2.0
        keep = r <= (1.0 - float(exclude_boundary_frac)) * R

        s1, _s3 = principal_stresses_2d(sxx[keep], syy[keep], txy[keep])
        s1_plus = np.maximum(s1, 0.0)

        if s1_plus.size == 0 or not np.any(np.isfinite(s1_plus)):
            amp[i] = np.nan
            continue

        if aggregation == "max":
            amp[i] = float(np.nanmax(s1_plus))
        else:
            amp[i] = float(np.nanpercentile(s1_plus, 99.0))

    # reference amplification at reference angle
    ref_ang = float(reference_angle_deg)
    # find closest angle in list; if not present, compute separately using the stress solver
    jref = int(np.argmin(np.abs(angles_deg - ref_ang)))
    amp_ref = amp[jref]

    if not np.isfinite(amp_ref) or amp_ref <= 0:
        raise RuntimeError("Reference amplification is invalid. Check stress_solver_fn outputs/units.")

    # Effective tensile strength scaling (inverse of amplification)
    # Higher amplification => lower strength required to fail
    T_eff_mpa = float(base_tensile_strength_mpa) * (amp_ref / (amp + 1e-30))

    # anisotropy ratio
    finite = np.isfinite(T_eff_mpa) & (T_eff_mpa > 0)
    if not np.any(finite):
        raise RuntimeError("No finite T_eff values. Check stress solver outputs.")
    anisotropy_ratio = float(np.nanmax(T_eff_mpa[finite]) / np.nanmin(T_eff_mpa[finite]))

    return anisotropy_ratio, angles_deg, T_eff_mpa, amp
