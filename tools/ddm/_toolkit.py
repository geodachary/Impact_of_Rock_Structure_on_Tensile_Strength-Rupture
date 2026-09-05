"""Shared implementation of the displacement-discontinuity toolkit.

Every function here was defined identically in both lithology notebooks.
It lives in one place so the two cannot drift apart, and so the notebooks
themselves carry analysis and interpretation rather than machinery.

Import from the themed views (:mod:`tools.ddm.angles`, :mod:`tools.ddm.grids`,
...) or from the :mod:`tools.ddm` package root, not from this module.

**Maintained directly.** Originally generated during the notebook-to-package
migration; that migration is complete and this module is now the source.
Edit the functions
here — never paste a copy back into a notebook.
"""
from __future__ import annotations

from matplotlib import tri
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import norm, t
from tools import output_dirs
from tools.bound_all_helpers import (
    points_in_disk,
    map_angle_to_alpha,
    rot_to_material,
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
    fit_orthotropic_airy_disk,
    fit_orthotropic_airy_disk_auto,
)
from tools.bound_all_helpers import (
    points_in_disk, map_angle_to_alpha,
    rot_to_material, stress_material_to_global,
    principal_from_components, eval_stress_field_material,
    fit_orthotropic_airy_disk, fit_orthotropic_airy_disk_auto,
    failure_mode_map, compute_psi_pref_field, weak_plane_weight_field,
    solve_cracked_disk_correction_ddm,
    _extract_tip_sifs, _seg_intersect,
    Gc_theta_weak_plane, G_from_K_orthotropic,
)
import importlib.util
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import re
import sys

# --------------------------------------------------------------------------
# Module-level constants (identical in both notebooks)
# --------------------------------------------------------------------------

PLOT_STYLE = {
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
}

TICK_PARAMS = {
    'major': {'which': 'major', 'direction': 'out', 'length': 5, 'width': 1.5},
    'minor': {'which': 'minor', 'direction': 'out', 'length': 3, 'width': 1},
}

platen_smooth = "cosine"       # cosine taper is smoother than a hard cutoff

boundary_exclude_frac = 0.05

cap_exclude_frac = 0.10

density = 18

length_exponent = 0.85

max_seg_frac = 0.20

min_show_frac = 0.02

tick_params = {
    "major": {"which": "major", "direction": "out", "length": 5, "width": 1.5},
    "minor": {"which": "minor", "direction": "out", "length": 3, "width": 1.0},
}

_ANGLE_MARKERS = {0: "o", 15: "s", 30: "^", 45: "D", 60: "v", 75: "P", 90: "X"}

_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442"]

bd = None

# Keyed on the normalised Rock_type string of the experimental table. Both the
# canonical spelling and the historical "psammatic" one are accepted, because
# the table is regenerated from the per-replicate measurements and the spelling
# has changed between regenerations.
#
# Adopted, not measured. E2 = E1 / ratio; the replicate table carries E1, nu12
# and G12 only, and no combination of the measured quantities reproduces these
# values (test_e2_provenance pins that). Table 5 of the manuscript reports them
# as adopted; add a citation there and here if a source is found. The ratio sets
# the compliance contrast that Section 5.3 credits with reproducing the gneiss's
# weaker localization, so its status matters.
#
# Eight analysis modules assign a shadow copy of this dict inside main(). None
# is read: every caller goes through get_anisotropy_ratio below.
ROCK_ANISO_RATIO = {
    "augen gneiss": 2.037,
    "psammitic schist": 3.763,
    "psammatic schist": 3.763,
}

TOL_VIBRANT = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
]

# --------------------------------------------------------------------------
# angles: Angle wrapping, axial statistics and orientation blending.
# --------------------------------------------------------------------------

def angle_axis(theta):
    return ((theta + np.pi/2) % np.pi) - np.pi/2


def axial_stats_from_delta(delta_rad, weights=None, R_min=0.05):
    """
    Axial circular stats for axis angles using doubling trick.
    If resultant length R is too small, mean/std are not meaningful.
    """
    d = np.asarray(delta_rad, float)
    m = np.isfinite(d)
    d = d[m]
    if d.size == 0:
        return dict(R=np.nan, mean=np.nan, std=np.nan, n=0)

    ang2 = 2.0 * d

    if weights is None:
        w = np.ones_like(ang2)
    else:
        w = np.asarray(weights, float)[m]
        w = np.clip(w, 0.0, np.inf)
        if np.sum(w) <= 0:
            w = np.ones_like(ang2)

    C = np.sum(w * np.cos(ang2))
    S = np.sum(w * np.sin(ang2))
    W = np.sum(w)

    R = np.hypot(C, S) / (W + 1e-30)

    if R < R_min:
        # essentially uniform/random -> mean direction meaningless
        return dict(R=float(R), mean=np.nan, std=np.nan, n=int(d.size))

    mean = 0.5 * np.arctan2(S, C)
    # “circular std” (can exceed 90° if distribution is close to uniform)
    std = 0.5 * np.sqrt(max(0.0, -2.0 * np.log(np.clip(R, 1e-12, 1.0))))

    return dict(R=float(R), mean=float(mean), std=float(std), n=int(d.size))


def axis_angle(theta):
    return ((theta + np.pi/2) % np.pi) - np.pi/2


def deviation_arrays(d, rmax_frac=0.985, keep_top_percent=20.0):
    inside_disk = d["r_pts"] <= float(rmax_frac) * d["R"]
    th_axis = wrap_axis_angle(d["th"])
    fol_axis = float(wrap_axis_angle(d["alpha_wp_line"]))
    delta_deg = np.degrees(wrap_axis_angle(th_axis - fol_axis))
    drive = sigma1_plus(d["sxx"], d["syy"], d["txy"])

    valid = inside_disk & np.isfinite(th_axis) & np.isfinite(delta_deg) & np.isfinite(drive)
    if np.any(valid):
        thr = np.nanpercentile(drive[valid], 100.0 - float(keep_top_percent))
        keep = valid & (drive >= thr)
    else:
        keep = valid

    return delta_deg[keep], drive[keep]


def trough_angle_from_xi(xi):
    xi = np.clip(xi, -0.999, 0.999)
    beta_star = np.degrees(np.arctan(np.sqrt((1.0 + xi) / (1.0 - xi))))
    return float(beta_star)


def wrap_axis_angle(theta_rad):
    theta_rad = np.asarray(theta_rad, dtype=float)
    return ((theta_rad + 0.5 * np.pi) % np.pi) - 0.5 * np.pi

# --------------------------------------------------------------------------
# elasticity: Orthotropic constitutive relations, stresses and strains.
# --------------------------------------------------------------------------

def band_open_factor_from_sigma_n(sigma_n_mpa, sigma0_mpa):
    s0 = float(max(sigma0_mpa, 1e-9))
    return 0.5 * (1.0 + np.tanh(np.asarray(sigma_n_mpa, float) / s0))


def compute_sample_stresses(row_dict, args):
    ang_deg = float(row_dict["Angle"])
    alpha_const = float(map_angle_to_alpha(ang_deg, angle_map=str(args.angle_map)))
    alpha_wp_line = float(((alpha_const + 0.5 * np.pi) % np.pi) - 0.5 * np.pi)

    D = float(row_dict["Diameter_mm"]) * 1e-3
    t = float(row_dict["Thickness_mm"]) * 1e-3
    P = float(row_dict["Load_(KN)"]) * 1e3 * float(args.load_factor)
    R = D / 2.0

    props = get_row_material_props(
        row_dict,
        default_E1_GPa=float(args.E1_GPa),
        default_G12_GPa=float(args.G12_GPa),
        default_nu12=float(args.nu12),
    )
    E1 = props["E1"]
    E2 = props["E2"]
    G12 = props["G12"]
    nu12 = props["nu12"]

    xg, yg, X, Y, M = points_in_disk(D, n=int(args.points_per_row))
    r_pts = np.hypot(xg, yg)

    do_auto = bool(args.airy_auto_tune) and (fit_orthotropic_airy_disk_auto is not None)
    if do_auto:
        fit = fit_orthotropic_airy_disk_auto(
            E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
            beta_deg=float(args.platen_half_angle_deg),
            smooth_deg=float(args.platen_smooth_deg),
            mu=float(args.platen_mu),
            Nbd=int(args.airy_Nbd_base),
            Nbd_arc_each=int(args.airy_Nbd_arc_each),
        )
    else:
        fit = fit_orthotropic_airy_disk(
            E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
            Nbd=int(args.airy_Nbd_base),
            beta_deg=float(args.platen_half_angle_deg),
            smooth_deg=float(args.platen_smooth_deg),
            mu=float(args.platen_mu),
            lam=1e-8, w_arc=12.0,
            Nbd_arc_each=int(args.airy_Nbd_arc_each),
        )

    xm, ym = rot_to_material(xg, yg, alpha_const)
    s11_m, s22_m, s12_m = eval_stress_field_material(
        xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"]
    )
    sxx, syy, txy = stress_material_to_global(s11_m, s22_m, s12_m, alpha_const)
    s1, s3, th = principal_from_components(sxx, syy, txy)

    return dict(
        ang_deg=ang_deg,
        alpha_const=alpha_const,
        alpha_wp_line=alpha_wp_line,
        R=R,
        r_pts=r_pts,
        sxx=sxx,
        syy=syy,
        txy=txy,
        th=th,
        E1_GPa=props["E1_GPa"],
        E2_GPa=props["E2_GPa"],
        G12_GPa=props["G12_GPa"],
        nu12=props["nu12"],
        aniso_ratio=props["ratio"],
    )


def modulus_to_Pa_from_csv(val):
    v = float(val)
    if v < 1e3:
        return v * 1e9
    return v * 1e6


def nominal_splitting_stress(P, diameter, thickness):
    return 2.0 * P / (np.pi * diameter * thickness)


def orthotropic_roots_p(E1, E2, nu12, G12, eps=1e-14):
    """
    Compute Lekhnitskii p-roots for orthotropic plane stress:
        s11 p^4 + (2 s12 + s66) p^2 + s22 = 0
    Returns p1, p2 with Im(p)>0 (complex).
    """
    E1 = float(E1); E2 = float(E2); nu12 = float(nu12); G12 = float(G12)
    s11 = 1.0 / E1
    s22 = 1.0 / E2
    s12 = -nu12 / E1
    s66 = 1.0 / G12

    a = s11
    b = (2.0 * s12 + s66)
    c = s22

    u_roots = np.roots([a, b, c])  # two roots (complex generally)
    p_roots = []
    for u in u_roots:
        p = np.sqrt(u + 0j)  # ensure complex
        p_roots.extend([p, -p])

    # choose roots with positive imaginary part
    p_pos = [p for p in p_roots if np.imag(p) > eps]

    # enforce uniqueness
    uniq = []
    for p in p_pos:
        if not any(abs(p - q) < 1e-10 for q in uniq):
            uniq.append(p)

    if len(uniq) >= 2:
        return uniq[0], uniq[1]

    # fallback: pick two with largest imaginary part, distinct
    p_sorted = sorted(p_roots, key=lambda z: np.imag(z), reverse=True)
    uniq = []
    for p in p_sorted:
        if not any(abs(p - q) < 1e-10 for q in uniq):
            uniq.append(p)
        if len(uniq) == 2:
            break
    if len(uniq) < 2:
        raise RuntimeError("Failed to obtain two distinct p-roots.")
    return uniq[0], uniq[1]


def principal_from_components(sxx, syy, txy):
    rad = np.sqrt(((sxx - syy) * 0.5)**2 + txy**2)
    sigma_1 = 0.5 * (sxx + syy) + rad
    sigma_3 = 0.5 * (sxx + syy) - rad
    theta_p = 0.5 * np.arctan2(2.0 * txy, (sxx - syy))
    return sigma_1, sigma_3, theta_p


def principal_strain_and_angle(ex, ey, gxy):
    exy = 0.5 * gxy
    avg = 0.5 * (ex + ey)
    rad = np.sqrt((0.5*(ex - ey))**2 + exy**2)
    eps1 = avg + rad
    theta_e = 0.5 * np.arctan2(gxy, (ex - ey))
    return eps1, theta_e


def principal_stresses_and_angle(sxx, syy, txy):
    rad = np.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)
    sigma1 = (sxx + syy) / 2.0 + rad
    sigma2 = (sxx + syy) / 2.0 - rad
    theta_p = 0.5 * np.arctan2(2 * txy, (sxx - syy))
    return sigma1, sigma2, theta_p


def q_matrix_plane_stress_orthotropic(E1, E2, nu12, G12):
    nu21 = nu12 * (E2 / (E1 + 1e-30))
    denom = 1.0 - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12
    return np.array([[Q11, Q12, 0.0],
                     [Q12, Q22, 0.0],
                     [0.0, 0.0, Q66]])


def required_load_for_target_sigma_t(sigma_t_mpa, diameter, thickness):
    return (sigma_t_mpa * 1e6) * (np.pi * diameter * thickness) / 2.0


def sigma1_plus(sxx, syy, txy):
    sxx = np.asarray(sxx, float)
    syy = np.asarray(syy, float)
    txy = np.asarray(txy, float)
    mean = 0.5 * (sxx + syy)
    rad = np.sqrt(0.25 * (sxx - syy) ** 2 + txy ** 2)
    return np.maximum(mean + rad, 0.0)


def splitting_stress_pa(P, D, t):
    return 2.0 * P / (np.pi * D * t)


def strain_energy_density_ortho_plane_stress_material(s11, s22, s12, E1, E2, nu12, G12):
    s11 = np.asarray(s11, float)
    s22 = np.asarray(s22, float)
    s12 = np.asarray(s12, float)
    E1, E2, G12, nu12 = float(E1), float(E2), float(G12), float(nu12)
    nu21 = nu12 * (E2 / (E1 + 1e-30))
    eps1 = (s11 / (E1 + 1e-30)) - (nu21 * s22 / (E2 + 1e-30))
    eps2 = (-nu12 * s11 / (E1 + 1e-30)) + (s22 / (E2 + 1e-30))
    gam12 = s12 / (G12 + 1e-30)
    return 0.5 * (s11 * eps1 + s22 * eps2 + s12 * gam12)


def strains_from_stress_TIsotropic(sigma_x, sigma_y, tau_xy, E1, E2, nu12, G12, theta_rad):
    Q = q_matrix_plane_stress_orthotropic(E1, E2, nu12, G12)
    Qbar = transform_Q_to_angle(Q, theta_rad)
    try:
        Sbar = np.linalg.inv(Qbar)
    except np.linalg.LinAlgError:
        Sbar = np.linalg.pinv(Qbar)

    sig = np.vstack([
        np.asarray(sigma_x).ravel(),
        np.asarray(sigma_y).ravel(),
        np.asarray(tau_xy).ravel()
    ])
    eps = Sbar @ sig
    ex  = eps[0, :].reshape(np.shape(sigma_x))
    ey  = eps[1, :].reshape(np.shape(sigma_y))
    gxy = eps[2, :].reshape(np.shape(tau_xy))
    return ex, ey, gxy


def surrogate_tensile_weight(s1, s3, eps=1e-12):
    """
    Local tensile tendency in [0,1].
    High when tensile principal stress dominates and compression is weak.
    """
    s1 = np.asarray(s1, float)
    s3 = np.asarray(s3, float)

    sig_t = np.maximum(s1, 0.0)
    sig_c = np.maximum(-s3, 0.0)
    wt = sig_t / (sig_t + sig_c + eps)
    return np.clip(wt, 0.0, 1.0)


def tensile_stress_from_sigma1_theta(sigma1_plus_mpa, theta_p_rad):
    s1_pa = np.asarray(sigma1_plus_mpa, float) * 1e6
    th = np.asarray(theta_p_rad, float)
    c = np.cos(th); s = np.sin(th)
    sxx = s1_pa * c*c
    syy = s1_pa * s*s
    txy = s1_pa * s*c
    return sxx, syy, txy


def transform_Q_to_angle(Q, theta_rad):
    m = np.cos(theta_rad)
    n = np.sin(theta_rad)
    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]

    m2 = m*m; n2 = n*n; m4 = m2*m2; n4 = n2*n2
    m3n = m*m*m*n; mn3 = m*n*n*n

    Qb11 = Q11*m4 + 2*(Q12 + 2*Q66)*m2*n2 + Q22*n4
    Qb22 = Q11*n4 + 2*(Q12 + 2*Q66)*m2*n2 + Q22*m4
    Qb12 = (Q11 + Q22 - 4*Q66)*m2*n2 + Q12*(m4 + n4)
    Qb16 = (Q11 - Q12 - 2*Q66)*m3n - (Q22 - Q12 - 2*Q66)*mn3
    Qb26 = (Q11 - Q12 - 2*Q66)*mn3 - (Q22 - Q12 - 2*Q66)*m3n
    Qb66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m2*n2 + Q66*(m4 + n4)

    return np.array([[Qb11, Qb12, Qb16],
                     [Qb12, Qb22, Qb26],
                     [Qb16, Qb26, Qb66]])

# --------------------------------------------------------------------------
# failure: Failure criteria, weak-band weighting and contact/platen models.
# --------------------------------------------------------------------------

def _safe_compute_psi_pref(X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
                           drive_basis="first", mixed_band=0.45, eps=1e-12):
    fn = bd.compute_psi_pref_field
    sig = inspect.signature(fn)
    names = list(sig.parameters.keys())
    if len(names) >= 3 and names[0].lower() == "x":
        return fn(
            X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
            drive_basis=drive_basis, mixed_band=mixed_band, eps=eps
        )
    return fn(
        Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
        drive_basis=drive_basis, mixed_band=mixed_band, eps=eps
    )


def _safe_wp_weight(X, Y, alpha_wp_line, spacing, bandwidth_frac):
    fn = bd.weak_plane_weight_field
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    if "phase" in params and "bandwidth_frac" in params:
        w = fn(
            X, Y,
            alpha_wp_line=float(alpha_wp_line),
            spacing=float(spacing),
            phase=0.0,
            bandwidth_frac=float(bandwidth_frac),
        )
    elif "bandwidth_frac" in params:
        w = fn(
            X, Y,
            alpha_wp_line=float(alpha_wp_line),
            spacing=float(spacing),
            bandwidth_frac=float(bandwidth_frac),
        )
    else:
        w = fn(X, Y, float(alpha_wp_line), float(spacing))
    return np.clip(np.asarray(w, float), 0.0, 1.0)


def build_psi_pref_field(X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
                         drive_basis="first", mixed_band=0.45, eps=1e-12):
    sig = inspect.signature(compute_psi_pref_field)
    names = list(sig.parameters.keys())
    if len(names) >= 3 and names[0].lower() == "x" and names[1].lower() == "y" and names[2].lower() == "m":
        return compute_psi_pref_field(
            X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
            drive_basis=drive_basis, mixed_band=mixed_band, eps=eps
        )
    return compute_psi_pref_field(
        Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
        drive_basis=drive_basis, mixed_band=mixed_band, eps=eps
    )


def compute_wp_weight_img(X, Y, alpha_wp_line, spacing, bandwidth_frac):
    sig = inspect.signature(weak_plane_weight_field)
    params = list(sig.parameters.keys())
    if ("phase" in params) and ("bandwidth_frac" in params):
        w = weak_plane_weight_field(
            X, Y,
            alpha_wp_line=float(alpha_wp_line),
            spacing=float(spacing),
            phase=0.0,
            bandwidth_frac=float(bandwidth_frac),
        )
    elif "bandwidth_frac" in params:
        w = weak_plane_weight_field(
            X, Y,
            alpha_wp_line=float(alpha_wp_line),
            spacing=float(spacing),
            bandwidth_frac=float(bandwidth_frac),
        )
    else:
        w = weak_plane_weight_field(X, Y, float(alpha_wp_line), float(spacing))
    w = np.asarray(w, float)
    return np.clip(w, 0.0, 1.0)


def disk_failure_stats_all_points(modes, r_pts, R, rmax_frac=0.985):
    m = np.asarray(modes, dtype=str)
    r_pts = np.asarray(r_pts, float)
    mask_valid = np.isfinite(r_pts) & (r_pts <= float(rmax_frac) * float(R))
    total = int(np.sum(mask_valid))
    keys = ["no_failure", "tensile", "shear", "mixed"]
    counts = {k: int(np.sum(mask_valid & (m == k))) for k in keys}
    denom = total if total > 0 else 1
    pct_all = {k: 100.0 * counts[k] / denom for k in keys}
    fail_total = counts["tensile"] + counts["shear"] + counts["mixed"]
    if fail_total > 0:
        pct_fail = {
            "tensile": 100.0 * counts["tensile"] / fail_total,
            "shear": 100.0 * counts["shear"] / fail_total,
            "mixed": 100.0 * counts["mixed"] / fail_total,
        }
    else:
        pct_fail = {"tensile": np.nan, "shear": np.nan, "mixed": np.nan}
    return total, counts, pct_all, fail_total, pct_fail, mask_valid


def evaluate_failure_mode(
    sigma_1, sigma_3,
    cohesion, friction_angle, tensile_limit,
    *,
    compression_positive=False,
    prefer="tensile",
    return_booleans=False
):
    """
    Failure mode classifier using:
      - Tensile cutoff
      - Mohr-Coulomb shear criterion in principal stress form

    Parameters
    ----------
    sigma_1, sigma_3 : array-like
        Major and minor principal stresses (same units).
        Be consistent with 'compression_positive'.
    cohesion : float or array-like
        Cohesion (same units).
    friction_angle : float
        Friction angle (radians).
    tensile_limit : float
        Tensile strength limit (same units, positive magnitude).
    compression_positive : bool
        If True: compression is +, tension is -.
        If False: tension is +, compression is -.
    prefer : {"tensile","shear","mixed"}
        Which label to assign if both criteria trigger.
    return_booleans : bool
        If True return (mode, tensile_failure, shear_failure)

    Returns
    -------
    mode : ndarray(dtype=object)
        'no_failure', 'tensile', 'shear', or 'mixed'
    """

    s1 = np.asarray(sigma_1, dtype=float)
    s3 = np.asarray(sigma_3, dtype=float)
    c = np.asarray(cohesion, dtype=float)

    # Ensure ordering: sigma_1 >= sigma_3 in the chosen sign convention sense
    # For principal values, sigma_1 should be the "maximum" numerically.
    # If user passes swapped, fix it.
    smax = np.maximum(s1, s3)
    smin = np.minimum(s1, s3)
    s1 = smax
    s3 = smin

    phi = float(friction_angle)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    # ---- Tensile failure ----
    # Convert to a tension-positive measure for checking tensile cutoff
    # If compression_positive, tensile stresses are negative -> tension_measure = -sigma
    # We check the most tensile principal stress (largest in tension-positive sense).
    if compression_positive:
        # tension-positive equivalent
        t1 = -s3  # s3 is least compressive / most tensile in compression-positive convention
        tensile_failure = (t1 >= float(tensile_limit))
    else:
        # tension is positive already; most tensile principal is sigma_1
        tensile_failure = (s1 >= float(tensile_limit))

    # ---- Mohr-Coulomb shear failure ----
    # Standard MC in principal stresses (compression-positive form):
    #   σ1 >= σ3 * (1+sinφ)/(1-sinφ) + 2c cosφ/(1-sinφ)
    #
    # If the input is tension-positive (compression negative), convert by σc = -σ
    if compression_positive:
        sig1_c = s1
        sig3_c = s3
    else:
        sig1_c = -s3  # careful: convert to compression-positive and keep ordering
        sig3_c = -s1

        # reorder after conversion so sig1_c >= sig3_c
        sig1_c, sig3_c = np.maximum(sig1_c, sig3_c), np.minimum(sig1_c, sig3_c)

    denom = (1.0 - sinp) + 1e-30
    mc_rhs = sig3_c * (1.0 + sinp) / denom + (2.0 * c * cosp) / denom
    shear_failure = (sig1_c >= mc_rhs)

    # ---- Decide mode ----
    mode = np.full(s1.shape, "no_failure", dtype=object)
    mode[shear_failure] = "shear"
    mode[tensile_failure] = "tensile"

    both = tensile_failure & shear_failure
    if np.any(both):
        if prefer == "mixed":
            mode[both] = "mixed"
        elif prefer == "shear":
            mode[both] = "shear"
        else:
            mode[both] = "tensile"

    if return_booleans:
        return mode, tensile_failure, shear_failure
    return mode


def evaluate_failure_mode_mohr_coulomb_constant(s1, s3, tensile_strength,
                                                cohesion, friction_angle, k_max):
    s1 = np.asarray(s1)
    s3 = np.asarray(s3)
    tensile_strength = np.asarray(tensile_strength)
    cohesion = np.asarray(cohesion)
    friction_angle = np.asarray(friction_angle)

    sigma_n = 0.5 * (s1 + s3)
    tau = 0.5 * (s1 - s3)

    cohesion_soft = cohesion * np.clip(1.0 - k_max, 0.0, 1.0)
    sigma_n_comp = np.maximum(0.0, -sigma_n)
    tau_strength = cohesion_soft + sigma_n_comp * np.tan(friction_angle)

    tensile_fail = s1 >= tensile_strength
    shear_fail = tau >= tau_strength

    modes = np.full(s1.shape, 'no_failure', dtype=object)
    modes[tensile_fail & ~shear_fail] = 'tensile'
    modes[shear_fail & ~tensile_fail] = 'shear'
    modes[tensile_fail & shear_fail] = 'mixed'
    return modes


def get_anisotropy_ratio(rock_name):
    key = _normalize_rock_name(rock_name)
    if key not in ROCK_ANISO_RATIO:
        valid = ", ".join(sorted(ROCK_ANISO_RATIO.keys()))
        raise RuntimeError(
            f"Rock_type '{rock_name}' not found in ROCK_ANISO_RATIO. "
            f"Available keys: {valid}"
        )
    return float(ROCK_ANISO_RATIO[key])


def hertz_contact_halfwidth(P_total_N, thickness_m, R_m, E_eff_MPa, nu):
    Pprime = float(P_total_N) / (float(thickness_m) + 1e-30)
    EeffPa = float(E_eff_MPa) * 1e6
    Eprime = EeffPa / (1.0 - float(nu)**2 + 1e-30)
    b = np.sqrt(4.0 * Pprime * float(R_m) / (np.pi * Eprime + 1e-30))
    return float(b)


def platen_weight(dtheta, beta):
    ad = np.abs(dtheta)
    if platen_smooth == "cosine":
        w = np.zeros_like(ad)
        m = ad < beta
        w[m] = 0.5*(1.0 + np.cos(np.pi * ad[m]/beta))
        return w
    else:
        return (ad < beta).astype(float)


def surrogate_conf_weight(s1, s3, eps=1e-12):
    """
    Local confinement/compressive tendency in [0,1].
    """
    s1 = np.asarray(s1, float)
    s3 = np.asarray(s3, float)

    sig_t = np.maximum(s1, 0.0)
    sig_c = np.maximum(-s3, 0.0)
    wc = sig_c / (sig_t + sig_c + eps)
    return np.clip(wc, 0.0, 1.0)


def surrogate_energy_proxy(sxx, syy, txy):
    """
    Positive stress-energy proxy.
    Absolute units are not critical because only robust normalization is used.
    """
    sxx = np.asarray(sxx, float)
    syy = np.asarray(syy, float)
    txy = np.asarray(txy, float)
    return 0.5 * (sxx**2 + syy**2 + 2.0 * txy**2)


def surrogate_wp_weight(x, y, anis_angle, spacing):
    """
    Surrogate weak-plane proximity weight in [0,1].
    If spacing <= 0, no periodic weak planes are activated.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if spacing is None or float(spacing) <= 0.0:
        return np.zeros_like(x, dtype=float)

    ca = np.cos(float(anis_angle))
    sa = np.sin(float(anis_angle))
    xp = ca * x + sa * y   # coordinate along anisotropy normal

    # peaks at the band centers, period ~ spacing
    w = np.cos(np.pi * xp / float(spacing))**2
    return np.clip(w, 0.0, 1.0)

# --------------------------------------------------------------------------
# figures: Publication figure helpers.
# --------------------------------------------------------------------------

def _get_marker(res):
    try:
        ang_i = int(float(res.get("angle_deg", 0)))
    except Exception:
        ang_i = 0
    return _ANGLE_MARKERS.get(ang_i, "o")


def _label_full(res):
    ang = res.get("angle_deg", "")
    rock = res.get("rock", "Sample")
    try:
        ang_i = int(float(ang))
    except Exception:
        ang_i = ang
    return f"{rock}, {ang_i}°"


def _label_short(res):
    ang = res.get("angle_deg", "")
    rock = res.get("rock", "")
    rock_short = str(rock).split()[0] if rock else ""
    try:
        ang_i = int(float(ang))
    except Exception:
        ang_i = ang
    return f"{rock_short} {ang_i}°" if rock_short else f"{ang_i}°"


def _make_all_combined_plots(results_list, out_dir=None):
    """Write the three combined crack-path figures.

    ``out_dir`` is accepted so a caller can redirect the figures, but it
    defaults to the figure directory rather than to the directory the step
    diagnostics were written to. The stepper emits both kinds under one
    ``out_dir``, and routing the figures separately is what keeps every figure
    in one place instead of leaving three of them filed under the name of the
    section that produced them.
    """
    out_dir = out_dir or output_dirs.figures()
    os.makedirs(out_dir, exist_ok=True)
    print("\n[plots] Saving publication figures...")
    rocks = list(dict.fromkeys(
        str(r.get("rock", "sample")).replace(" ", "_").lower()
        for r in results_list
    ))
    stem = "_".join(rocks) if rocks else "samples"
    save_all_paths_plot(results_list, os.path.join(out_dir, f"{stem}_all_paths.pdf"))
    save_all_energy_plot(results_list, os.path.join(out_dir, f"{stem}_all_energy.pdf"))
    save_all_ratio_plot(results_list, os.path.join(out_dir, f"{stem}_all_ratio.pdf"))


def _mode_to_code_img(M, modes):
    lut = {"no_failure": 0, "tensile": 1, "shear": 2, "mixed": 3}
    out = np.full(M.shape, -1, dtype=np.int8)
    out[M] = np.array([lut.get(str(m), 0) for m in modes], dtype=np.int8)
    return out


def apply_plot_style():
    plt.rcParams.update(PLOT_STYLE)
    return TICK_PARAMS


def make_figure1(sample_records, out_path,
                 fig_width_in=7.08,
                 fig_height_in=4.05):
    recs = sorted(sample_records, key=lambda r: (r["ang_deg"], r["sid"]))
    angles = np.array([rec["ang_deg"] for rec in recs], dtype=float)

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    max_half_width = 2.75

    for a in angles:
        ax.hlines(a, -90, 90, color="0.965", lw=0.55, zorder=0)

    for i, rec in enumerate(recs):
        x = np.asarray(rec["delta_deg"], float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue

        col = TOL_VIBRANT[i % len(TOL_VIBRANT)]
        y0 = rec["ang_deg"]
        stats = summarize_distribution(x)

        lo = max(-90.0, stats["q05"])
        hi = min(90.0, stats["q95"])
        if hi <= lo:
            lo = np.min(x)
            hi = np.max(x)
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0

        grid = np.linspace(lo, hi, 220)
        bw = robust_bandwidth(x)
        dens = gaussian_kde_manual(x, grid, bw=bw)

        if np.max(dens) > 0:
            dens = dens / np.max(dens) * max_half_width
        else:
            dens = np.zeros_like(grid)

        ax.fill_between(
            grid, y0 - dens, y0 + dens,
            facecolor=col, edgecolor=col,
            linewidth=0.85, alpha=0.24, zorder=2
        )

        ax.plot(
            [stats["q25"], stats["q75"]], [y0, y0],
            color="0.10", lw=3.5, solid_capstyle="round", zorder=4
        )

        ax.scatter(
            stats["med"], y0,
            s=42, facecolor="white", edgecolor="0.10",
            linewidth=0.9, zorder=5
        )

    ax.axvline(0, color="0.35", lw=1.0, ls="--", zorder=1)

    ax.set_xlim(-90, 90)
    ax.set_ylim(-5, 97)

    ax.set_xlabel(r"Signed deviation  $\Delta = \theta_p - \alpha_{\mathrm{wp}}$  (°)")
    ax.set_ylabel(r"Foliation angle  $\alpha$  (°)")

    ax.set_yticks(angles)
    ax.set_yticklabels([f"{a:.0f}" for a in angles])

    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(15))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

    ax.grid(axis="x", which="major", color="0.90", lw=0.55)
    ax.grid(axis="x", which="minor", color="0.95", lw=0.35)

    legend_handles = [
        Patch(facecolor="#0077BB", edgecolor="#0077BB", alpha=0.24, label="Distribution"),
        Line2D([0], [0], color="0.10", lw=3.5, label="IQR"),
        Line2D([0], [0], marker="o", markersize=6, linestyle="None",
               markerfacecolor="white", markeredgecolor="0.10", label="Median"),
        Line2D([0], [0], color="0.35", lw=1.0, ls="--", label=r"$\Delta = 0$"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=2,
        borderpad=0.45,
        handlelength=1.7,
        columnspacing=1.0
    )

    fig.savefig(out_path, format="pdf")
    print(f"Figure 1 saved → {out_path}")
    return fig


def make_figure2(sample_records, out_path,
                 fig_width_in=3.54,
                 fig_height_in=3.30):
    recs = sorted(sample_records, key=lambda r: (r["ang_deg"], r["sid"]))

    ang_vals = []
    med_vals = []
    q25_vals = []
    q75_vals = []

    for rec in recs:
        d = np.asarray(rec["delta_deg"], float)
        if d.size < 5:
            continue
        stats = summarize_distribution(d)
        ang_vals.append(rec["ang_deg"])
        med_vals.append(stats["med"])
        q25_vals.append(stats["q25"])
        q75_vals.append(stats["q75"])

    ang_vals = np.asarray(ang_vals, float)
    med_vals = np.asarray(med_vals, float)
    q25_vals = np.asarray(q25_vals, float)
    q75_vals = np.asarray(q75_vals, float)

    pool_sorted = pooled_equal_weight_abs_delta(recs, seed=42)
    if pool_sorted.size > 0:
        cdf = np.arange(1, len(pool_sorted) + 1) / len(pool_sorted)
        med_abs = float(np.percentile(pool_sorted, 50))
        q25_abs = float(np.percentile(pool_sorted, 25))
        q75_abs = float(np.percentile(pool_sorted, 75))
    else:
        cdf = np.array([])
        med_abs = np.nan
        q25_abs = np.nan
        q75_abs = np.nan

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    ax = fig.add_axes([0.14, 0.14, 0.81, 0.77])

    ax.axhline(0, color="0.35", lw=0.95, ls="--", zorder=0)

    ax.plot(
        ang_vals, med_vals,
        color="#0077BB", lw=1.9, zorder=2
    )

    for i, (a, m, q25, q75) in enumerate(zip(ang_vals, med_vals, q25_vals, q75_vals)):
        col = TOL_VIBRANT[i % len(TOL_VIBRANT)]
        ax.vlines(a, q25, q75, color=col, lw=1.35, alpha=0.95, zorder=3)
        ax.scatter(a, m, s=40, facecolor=col, edgecolor="white", linewidth=0.65, zorder=4)

    ax.set_xlim(-2, 92)
    ax.set_ylim(-90, 90)

    ax.set_xlabel(r"Foliation angle  $\alpha$  (°)")
    ax.set_ylabel(r"Signed deviation  $\Delta = \theta_p - \alpha_{\mathrm{wp}}$  (°)")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))

    ax.grid(axis="y", which="major", color="0.92", lw=0.6)
    ax.grid(axis="y", which="minor", color="0.96", lw=0.35)

    ax.text(0.97, 0.98, "(a)", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")

    legend_handles = [
        Line2D([0], [0], color="#0077BB", lw=1.9, label=r"Median $\Delta$"),
        Line2D([0], [0], color="0.25", lw=1.35, marker="|", markersize=10,
               linestyle="None", label="IQR"),
        Line2D([0], [0], color="0.35", lw=0.95, ls="--", label=r"$\Delta = 0$"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        borderpad=0.42,
        handlelength=1.6
    )

    ax_ins = fig.add_axes([0.23, 0.595, 0.37, 0.28])

    if pool_sorted.size > 0:
        ax_ins.plot(pool_sorted, cdf, color="#0077BB", lw=1.15, zorder=2)
        ax_ins.axvline(med_abs, color="0.25", lw=0.85, ls="--", zorder=1)
        ax_ins.axvspan(q25_abs, q75_abs, color="0.6", alpha=0.12, zorder=0)

        stats_text = (
            f"med = {med_abs:.1f}°\n"
            f"IQR = [{q25_abs:.1f}, {q75_abs:.1f}]°"
        )
        ax_ins.text(
            0.97, 0.08, stats_text,
            transform=ax_ins.transAxes,
            ha="right", va="bottom",
            fontsize=5.7, color="0.20",
            bbox=dict(
                boxstyle="square,pad=0.18",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.96
            )
        )

    ax_ins.set_xlim(0, 90)
    ax_ins.set_ylim(0, 1.0)
    ax_ins.set_xlabel(r"$|\Delta|$ (°)", fontsize=6.0, labelpad=1)
    ax_ins.set_ylabel("ECDF", fontsize=6.0, labelpad=1)

    ax_ins.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax_ins.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_ins.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax_ins.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax_ins.tick_params(which="both", labelsize=5.7, direction="in")

    ax_ins.set_title(r"Pooled $|\Delta|$ distribution", fontsize=6.0, pad=2)
    ax_ins.text(0.04, 0.95, "(b)", transform=ax_ins.transAxes,
                fontsize=6.3, fontweight="bold", va="top")

    fig.savefig(out_path, format="pdf")
    print(f"Figure 2 saved → {out_path}")
    return fig


def plot_direction_circle(ax, Xs, Ys, Us, Vs, colors, R, title, norm, cmap="magma"):
    q = ax.quiver(
        Xs, Ys, Us, Vs, colors,
        cmap=cmap, norm=norm,
        angles="xy", scale_units="xy", scale=1,
        width=0.0040, headwidth=4, headlength=6,
        pivot="mid", alpha=1.0
    )
    ax.add_patch(plt.Circle((0, 0), R, color="black", fill=False, linewidth=2.0))
    ax.set_title(title, pad=10)
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(**tick_params["major"])
    ax.tick_params(**tick_params["minor"])
    ax.minorticks_on()
    ax.grid(False)
    return q


def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.2,

        "lines.linewidth": 1.3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.72",
        "legend.fancybox": False,
    })

# --------------------------------------------------------------------------
# fitting: Model fitting, metrics and measured-data comparison.
# --------------------------------------------------------------------------

def choose_smallest_near_peak(kvals, metric, rel_tol=0.02):
    """
    Choose the smallest k whose metric is within (1-rel_tol) of the peak.
    This avoids boundary-hugging optima when the curve saturates.
    """
    kvals = np.asarray(kvals, dtype=float)
    metric = np.asarray(metric, dtype=float)

    ok = np.isfinite(metric)
    if not np.any(ok):
        raise ValueError("No finite metric values available.")

    mmax = float(np.nanmax(metric[ok]))
    threshold = (1.0 - float(rel_tol)) * mmax

    idx_candidates = np.where(ok & (metric >= threshold))[0]
    if len(idx_candidates) == 0:
        raise ValueError("Could not identify a near-peak k value.")

    idx = int(idx_candidates[0])
    hit_upper_bound = (idx == len(kvals) - 1)

    return idx, float(kvals[idx]), mmax, hit_upper_bound


def compare_with_measured_data(x, y, stress_magnitude_numerical, measured_data_file=None):
    """
    Illustrative stub for pixel/point-based stress comparison.
    'measured_data_file' should contain columns [x, y, measured_stress].
    
    Returns a DataFrame or prints a summary of differences.
    """
    if measured_data_file is None or not os.path.isfile(measured_data_file):
        print("[WARNING] No measured data file provided. Skipping quantitative verification.")
        return None
    
    measured_df = pd.read_csv(measured_data_file)
    # For demonstration, let's assume measured_df has columns: x, y, stress_measured(MPa)
    
    # We must match each (x, y) point from our mesh to the nearest measured data point:
    # A simple way is to do a nearest-neighbor match. 
    # Here, we do a naive approach: each point in measured_df matches the "same index" in numerical.
    
    # Check length consistency or do a more sophisticated approach
    n_num = len(stress_magnitude_numerical)
    n_meas = len(measured_df)
    n_min = min(n_num, n_meas)
    
    # Just compare up to the smaller length for demonstration
    comparison_df = pd.DataFrame({
        'x': x[:n_min],
        'y': y[:n_min],
        'stress_numerical(MPa)': stress_magnitude_numerical[:n_min],
        'stress_measured(MPa)': measured_df['stress_measured(MPa)'][:n_min].values
    })
    
    # Calculate error metrics
    comparison_df['difference(MPa)'] = comparison_df['stress_numerical(MPa)'] - comparison_df['stress_measured(MPa)']
    mse = np.mean(comparison_df['difference(MPa)']**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(comparison_df['difference(MPa)']))
    
    print("=== Quantitative Verification ===")
    print(f"RMSE: {rmse:.4f} MPa")
    print(f"MAE:  {mae:.4f} MPa")
      
    return comparison_df


def fit_coeffs_collocation(theta, R, p1, p2, S, T, M=12, lam=1e-6, w=None):
    """
    Collocation fit for complex polynomial coefficients a1,a2 (m>=2 terms only),
    matching target tractions:
        tr(theta) = S(theta)
        tt(theta) = T(theta)

    Unknown vector contains:
        Re(a1m), Im(a1m), Re(a2m), Im(a2m) for m=2..M
    """
    theta = np.asarray(theta, float)
    S = np.asarray(S, float)
    T = np.asarray(T, float)

    nb = len(theta)
    if w is None:
        w = np.ones(nb, dtype=float)
    else:
        w = np.asarray(w, float)

    # Unknowns count
    ncoef = (M - 1)  # m=2..M
    nunk = 4 * ncoef

    A = np.zeros((2 * nb, nunk), dtype=float)
    b = np.zeros(2 * nb, dtype=float)
    b[0:nb] = S
    b[nb:] = T

    x = R * np.cos(theta)
    y = R * np.sin(theta)
    nx = np.cos(theta)
    ny = np.sin(theta)

    z1 = x + p1 * y
    z2 = x + p2 * y

    def tr_tt_from_fpp(p, fpp):
        sxx = np.real(p**2 * fpp)
        syy = np.real(fpp)
        txy = -np.real(p * fpp)

        tx = sxx * nx + txy * ny
        ty = txy * nx + syy * ny

        tr = tx * nx + ty * ny
        tt = -tx * ny + ty * nx
        return tr, tt

    col = 0
    for m in range(2, M + 1):
        mm = m * (m - 1)

        # Basis f'' for coefficient = 1
        fpp1 = mm * (z1 ** (m - 2))
        fpp2 = mm * (z2 ** (m - 2))

        # Real-part columns: coefficient 1.0
        tr1, tt1 = tr_tt_from_fpp(p1, fpp1)
        tr2, tt2 = tr_tt_from_fpp(p2, fpp2)

        # Imag-part columns: coefficient 1j
        tr1i, tt1i = tr_tt_from_fpp(p1, 1j * fpp1)
        tr2i, tt2i = tr_tt_from_fpp(p2, 1j * fpp2)

        # Fill A: first nb rows are tr, next nb are tt
        A[0:nb, col + 0] = tr1
        A[nb:,  col + 0] = tt1
        A[0:nb, col + 1] = tr1i
        A[nb:,  col + 1] = tt1i

        A[0:nb, col + 2] = tr2
        A[nb:,  col + 2] = tt2
        A[0:nb, col + 3] = tr2i
        A[nb:,  col + 3] = tt2i

        col += 4

    # Apply weights
    Wsqrt = np.sqrt(np.concatenate([w, w]))
    Aw = A * Wsqrt[:, None]
    bw = b * Wsqrt

    # Tikhonov regularization
    ATA = Aw.T @ Aw
    ATb = Aw.T @ bw
    ATA += float(lam) * np.eye(nunk)

    c = np.linalg.solve(ATA, ATb)

    # Unpack to complex a1,a2
    a1 = np.zeros(M + 1, dtype=complex)
    a2 = np.zeros(M + 1, dtype=complex)

    col = 0
    for m in range(2, M + 1):
        a1[m] = c[col + 0] + 1j * c[col + 1]
        a2[m] = c[col + 2] + 1j * c[col + 3]
        col += 4

    return a1, a2


def generate_random_props(n, base_T, base_phi_deg, base_coh, hetero_scale,
                          rng=None, base_draws=None):
    """
    Generates random material fields:
      Tens (MPa), phi (rad), Coh (MPa)

    hetero_scale = 0 gives homogeneous properties
    """
    if base_draws is None:
        if rng is None:
            rng = np.random.default_rng()
        u_T = rng.uniform(-1, 1, size=n)
        u_P = rng.uniform(-1, 1, size=n)
        u_C = rng.uniform(-1, 1, size=n)
    else:
        u_T, u_P, u_C = base_draws

    Tens = base_T * (1.0 + hetero_scale * u_T)
    Tens = np.clip(Tens, 0.01, None)

    phi0 = np.radians(base_phi_deg)
    phi = phi0 * (1.0 + hetero_scale * u_P)
    phi = np.clip(phi, np.radians(5.0), np.radians(85.0))

    Coh = base_coh * (1.0 + hetero_scale * u_C)
    Coh = np.clip(Coh, 0.0, None)

    return Tens, phi, Coh, (u_T, u_P, u_C)


def get_row_material_props(row_dict, default_E1_GPa=50.0, default_G12_GPa=12.0, default_nu12=0.25):
    rock = str(row_dict["Rock_type"]).strip()
    rock_key = rock.lower()

    E1_GPa = (
        float(row_dict["Modulus_of_Elasticity"])
        if pd.notna(row_dict["Modulus_of_Elasticity"])
        else float(default_E1_GPa)
    )

    G12_GPa = (
        float(row_dict["Shear_Modulus"])
        if pd.notna(row_dict["Shear_Modulus"])
        else float(default_G12_GPa)
    )

    nu12 = (
        float(row_dict["Poisson_Ratio"])
        if pd.notna(row_dict["Poisson_Ratio"])
        else float(default_nu12)
    )

    ratio = ROCK_ANISO_RATIO.get(rock_key, 1.0)
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0

    E2_GPa = E1_GPa / ratio

    return {
        "rock": rock,
        "ratio": ratio,
        "E1_GPa": E1_GPa,
        "E2_GPa": E2_GPa,
        "G12_GPa": G12_GPa,
        "nu12": nu12,
        "E1": E1_GPa * 1e3,   # MPa
        "E2": E2_GPa * 1e3,   # MPa
        "G12": G12_GPa * 1e3, # MPa
    }


def pooled_equal_weight_abs_delta(sample_records, seed=42):
    arrays = [np.abs(rec["delta_deg"]) for rec in sample_records if len(rec["delta_deg"]) > 0]
    if len(arrays) == 0:
        return np.array([])

    min_n = min(arr.size for arr in arrays)
    rng = np.random.default_rng(seed)
    pool = np.concatenate([rng.choice(arr, size=min_n, replace=False) for arr in arrays])
    return np.sort(pool)


def print_material_parameters(df, sample_idx):
    """
    Prints E and ν for a given sample index to confirm correctness
    """
    E_gpa = df.loc[sample_idx, 'Modulus_of_Elasticity']
    nu = df.loc[sample_idx, 'Poisson_Ratio']
    rock_type = df.loc[sample_idx, 'Rock_type']
    
    print(f"Sample {sample_idx} - Rock Type: {rock_type}")
    print(f"  Young's Modulus (E): {E_gpa:.2f} GPa")
    print(f"  Poisson's Ratio (ν): {nu:.3f}")
    print("  (Values read directly from CSV / Table 1.)\n")


def summarize_distribution(x):
    x = np.asarray(x, float)
    return {
        "q05": float(np.percentile(x, 5)),
        "q25": float(np.percentile(x, 25)),
        "med": float(np.percentile(x, 50)),
        "q75": float(np.percentile(x, 75)),
        "q95": float(np.percentile(x, 95)),
    }


def summarize_modes(modes):
    counts = {
        "no_failure": int(np.count_nonzero(modes == "no_failure")),
        "tensile": int(np.count_nonzero(modes == "tensile")),
        "shear": int(np.count_nonzero(modes == "shear")),
        "mixed": int(np.count_nonzero(modes == "mixed")),
    }
    counts["failed_total"] = counts["tensile"] + counts["shear"] + counts["mixed"]
    counts["all_total"] = int(modes.size)
    return counts

# --------------------------------------------------------------------------
# grids: Disc grids, masks, polylines and segment geometry.
# --------------------------------------------------------------------------

def _grid_axes(X, Y):
    return np.asarray(X[0, :], float), np.asarray(Y[:, 0], float)


def _grid_axes_from_mesh(X, Y):
    x1d = np.asarray(X[0, :], float)
    y1d = np.asarray(Y[:, 0], float)
    return x1d, y1d


def _grid_cell_area(X, Y):
    dx = float(np.nanmean(np.diff(X[0, :]))) if X.shape[1] > 1 else 0.0
    dy = float(np.nanmean(np.diff(Y[:, 0]))) if Y.shape[0] > 1 else 0.0
    return abs(dx * dy)


def _img_from_mask(M, vals, fill=np.nan):
    img = np.full(M.shape, fill, float)
    img[M] = np.asarray(vals, float)
    return img


def _polyline_length(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def _thin_points_for_glyphs(x, y, R, cells=18, per_cell=1, max_keep=650):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cx = np.floor((np.clip(x, -R, R) + R) / (2 * R) * cells).astype(int)
    cy = np.floor((np.clip(y, -R, R) + R) / (2 * R) * cells).astype(int)
    cx = np.clip(cx, 0, cells - 1)
    cy = np.clip(cy, 0, cells - 1)
    used = {}
    keep = np.zeros_like(x, dtype=bool)
    for i in range(len(x)):
        key = (int(cx[i]), int(cy[i]))
        c = used.get(key, 0)
        if c < per_cell:
            keep[i] = True
            used[key] = c + 1
    idx = np.where(keep)[0]
    if max_keep is not None and len(idx) > int(max_keep):
        step = int(np.ceil(len(idx) / int(max_keep)))
        idx = idx[::step]
    return idx


def build_segments_from_axis(X, Y, mask, theta_axis, mag, R, ref_mag, color_vmax):
    r = np.hypot(X, Y)
    m = mask.copy()
    m &= (r <= (1.0 - boundary_exclude_frac) * R)
    m &= (np.abs(Y) <= (1.0 - cap_exclude_frac) * R)

    step = max(1, int(X.shape[0] // density))
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ms = m[::step, ::step]
    th = theta_axis[::step, ::step]
    mg = mag[::step, ::step]

    ok = Ms & np.isfinite(th) & np.isfinite(mg) & (mg > 0)
    Xs = Xs[ok]; Ys = Ys[ok]; th = th[ok]; mg = mg[ok]

    if mg.size == 0:
        return np.zeros((0, 2, 2)), np.zeros((0,)), np.zeros((0,))

    thr = float(min_show_frac) * float(ref_mag)
    keep = mg >= thr
    Xs = Xs[keep]; Ys = Ys[keep]; th = th[keep]; mg = mg[keep]

    if mg.size == 0:
        return np.zeros((0, 2, 2)), np.zeros((0,)), np.zeros((0,))

    ratio = np.clip(mg / (float(ref_mag) + 1e-30), 0.0, 1.0)
    segL = (ratio ** float(length_exponent)) * (float(max_seg_frac) * R)

    dx = 0.5 * segL * np.cos(th)
    dy = 0.5 * segL * np.sin(th)

    x1 = Xs - dx; y1 = Ys - dy
    x2 = Xs + dx; y2 = Ys + dy

    epsb = 9.9e-4 * R
    def clip_to_circle(x, y):
        rr = np.sqrt(x*x + y*y)
        over = rr > (R - epsb)
        if np.any(over):
            fac = (R - epsb) / (rr[over] + 1e-30)
            x[over] *= fac
            y[over] *= fac
        return x, y

    x1, y1 = clip_to_circle(x1, y1)
    x2, y2 = clip_to_circle(x2, y2)

    segments = np.stack([np.stack([x1, y1], axis=1),
                         np.stack([x2, y2], axis=1)], axis=1)

    colors = np.clip(mg, 0.0, float(color_vmax))
    return segments, colors, mg


def clip_to_disk(Xs, Ys, Us, Vs, R, epsb):
    Xe = Xs + Us
    Ye = Ys + Vs
    over = (Xe*Xe + Ye*Ye) > (R - epsb)**2
    if np.any(over):
        fac = (R - epsb) / (np.sqrt(Xe[over]**2 + Ye[over]**2) + 1e-30)
        Us[over] *= fac
        Vs[over] *= fac
    return Us, Vs


def create_grid_mesh(diameter, points_per_row=50):
    R = diameter / 2.0
    x_vals = np.linspace(-R, R, points_per_row)
    y_vals = np.linspace(-R, R, points_per_row)
    X, Y = np.meshgrid(x_vals, y_vals)
    mask = (X * X + Y * Y) <= R * R
    x = X[mask]
    y = Y[mask]
    r = np.sqrt(x * x + y * y)
    return x, y, r, R


def create_structured_grid(diameter, grid_size):
    radius = diameter / 2
    x = np.linspace(-radius, radius, grid_size)
    y = np.linspace(-radius, radius, grid_size)
    X, Y = np.meshgrid(x, y)
    mask = (X ** 2 + Y ** 2) <= radius ** 2
    return X, Y, mask


def create_triangular_mesh_in_disk(diameter, num_points=60000, seed=0):
    rng = np.random.default_rng(seed)
    R = diameter / 2.0
    ang = rng.uniform(0, 2*np.pi, num_points)
    rad = np.sqrt(rng.uniform(0, 1, num_points)) * R
    x = rad * np.cos(ang)
    y = rad * np.sin(ang)
    triang = tri.Triangulation(x, y)
    r = np.sqrt(x*x + y*y)
    return triang, x, y, r


def downsample_mask(X, Y, mask, U, V, density):
    step = max(1, int(X.shape[0] // int(density)))
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ms = mask[::step, ::step]
    Us = U[::step, ::step]
    Vs = V[::step, ::step]
    return Xs, Ys, Ms, Us, Vs


def make_segments(X, Y, mask, eps1_plus, theta_axis, diameter, ref_mag, R):
    r = np.hypot(X, Y)
    m = mask.copy()
    m &= (r <= (1.0 - boundary_exclude_frac) * R)
    m &= (np.abs(Y) <= (1.0 - cap_exclude_frac) * R)

    step = max(1, int(X.shape[0] // density))
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ms = m[::step, ::step]
    e1 = eps1_plus[::step, ::step]
    th = theta_axis[::step, ::step]

    ok = Ms & np.isfinite(e1) & np.isfinite(th)
    Xs = Xs[ok]; Ys = Ys[ok]
    e1 = e1[ok]; th = th[ok]

    if e1.size == 0:
        return np.zeros((0, 2, 2)), np.zeros((0,)), np.zeros((0,))

    Lc = diameter / 4.0
    mag = e1 * Lc

    thr = float(min_show_frac) * float(ref_mag)
    keep = mag >= thr
    Xs = Xs[keep]; Ys = Ys[keep]
    mag = mag[keep]; th = th[keep]
    if mag.size == 0:
        return np.zeros((0, 2, 2)), np.zeros((0,)), np.zeros((0,))

    ratio = np.clip(mag / (float(ref_mag) + 1e-30), 0.0, 1.0)
    segL = (ratio ** float(length_exponent)) * (float(max_seg_frac) * R)

    dx = 0.5 * segL * np.cos(th)
    dy = 0.5 * segL * np.sin(th)

    x1 = Xs - dx; y1 = Ys - dy
    x2 = Xs + dx; y2 = Ys + dy

    epsb = 9.9e-4 * R
    def clip(x, y):
        rr = np.sqrt(x*x + y*y)
        over = rr > (R - epsb)
        if np.any(over):
            fac = (R - epsb) / (rr[over] + 1e-30)
            x[over] *= fac
            y[over] *= fac
        return x, y

    x1, y1 = clip(x1, y1)
    x2, y2 = clip(x2, y2)

    segs = np.stack([np.stack([x1, y1], axis=1),
                     np.stack([x2, y2], axis=1)], axis=1)

    colors_deg = np.rad2deg(th)
    return segs, colors_deg, mag


def reconstruct_grid(diameter, grid_N):
    R = diameter / 2.0
    x = np.linspace(-R, R, grid_N)
    y = np.linspace(-R, R, grid_N)
    X, Y = np.meshgrid(x, y)
    mask = (X*X + Y*Y) <= R*R
    return X, Y, mask, R


def reconstruct_grid_and_mask(diameter, grid_N):
    R = diameter / 2.0
    x = np.linspace(-R, R, grid_N)
    y = np.linspace(-R, R, grid_N)
    X, Y = np.meshgrid(x, y)
    mask = (X*X + Y*Y) <= R*R
    return X, Y, mask, R


def sample_line(triang, field_pa, R, n_samples, smooth_sigma_pts):
    x_line = np.linspace(-R, R, n_samples)
    y_line = np.zeros_like(x_line)

    interp = tri.LinearTriInterpolator(triang, field_pa)
    vals = interp(x_line, y_line)
    vals = np.array(vals.filled(np.nan), float)

    m = np.isfinite(vals)
    x_line = x_line[m]
    vals = vals[m]

    vals_sm = gaussian_smooth_1d(vals, smooth_sigma_pts) if smooth_sigma_pts > 0 else vals.copy()
    return x_line, vals, vals_sm


def snap_path_to_platen_arcs(xs, ys, R, beta_deg, r_end_frac=0.999, n_ext=10):
    """
    Extend BOTH ends of a mirrored crack polyline so they land on the platen loading arcs:
      top arc centered at +pi/2, bottom arc at -pi/2, halfwidth = beta_deg.

    - Keeps the interior hybrid shape
    - Only extends ends (adds short line segments)
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    beta = np.deg2rad(float(beta_deg))
    rtar = float(r_end_frac) * float(R)

    def wrap_pi(a):
        return ((a + np.pi) % (2*np.pi)) - np.pi

    def clamp_to_arc(theta, center):
        d = wrap_pi(theta - center)
        d = np.clip(d, -beta, beta)
        return center + d

    # endpoints of the mirrored path:
    # xs[0],ys[0]  -> bottom end
    # xs[-1],ys[-1] -> top end
    th_bot = np.arctan2(ys[0], xs[0])
    th_top = np.arctan2(ys[-1], xs[-1])

    th_bot_c = clamp_to_arc(th_bot, -np.pi/2)
    th_top_c = clamp_to_arc(th_top, +np.pi/2)

    xb, yb = rtar*np.cos(th_bot_c), rtar*np.sin(th_bot_c)
    xt, yt = rtar*np.cos(th_top_c), rtar*np.sin(th_top_c)

    # prepend a short segment from (xb,yb) to current start (xs[0],ys[0])
    if n_ext and n_ext > 0:
        tpre = np.linspace(0.0, 1.0, int(n_ext), endpoint=False)
        xpre = xb + (xs[0] - xb) * tpre
        ypre = yb + (ys[0] - yb) * tpre

        # append a short segment from current end (xs[-1],ys[-1]) to (xt,yt)
        tapp = np.linspace(0.0, 1.0, int(n_ext) + 1)[1:]
        xapp = xs[-1] + (xt - xs[-1]) * tapp
        yapp = ys[-1] + (yt - ys[-1]) * tapp

        xs2 = np.r_[xpre, xs, xapp]
        ys2 = np.r_[ypre, ys, yapp]
        return xs2, ys2

    # simple replace if no extension requested
    xs2 = xs.copy(); ys2 = ys.copy()
    xs2[0], ys2[0] = xb, yb
    xs2[-1], ys2[-1] = xt, yt
    return xs2, ys2


def valid_mask_from_fields(*arrays):
    mask = np.ones(np.shape(np.asarray(arrays[0])), dtype=bool)
    for a in arrays:
        aa = np.asarray(a, dtype=float)
        mask &= np.isfinite(aa)
    return mask

# --------------------------------------------------------------------------
# io: Field archive read/write, path discovery and solver binding.
# --------------------------------------------------------------------------

def _search_dirs(meta_csv=None):
    dirs = []
    try:
        dirs.append(os.getcwd())
    except Exception:
        pass
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    if meta_csv:
        try:
            dirs.append(os.path.dirname(os.path.abspath(meta_csv)))
        except Exception:
            pass
    out = []
    seen = set()
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            out.append(d)
            seen.add(d)
    return out


def bind_sif_hooks(mod):
    """Bind the solver module and its stress-intensity hooks into this package.

    ``bd`` matters as much as the hooks. While these functions lived in the
    notebook, the caller's ``bd = import_or_load_solver(...)`` and the code that
    read ``bd.compute_psi_pref_field`` shared one namespace. Moving them into
    this package separated the two: the caller kept binding its own ``bd`` and
    this module kept reading a ``bd`` that was still ``None``, so every
    crack-path run failed with ``'NoneType' object has no attribute
    'compute_psi_pref_field'``.

    Binding it here, in the function every caller already invokes immediately
    after obtaining the solver, is what keeps the two in step.
    """
    global SIF_FUN, TRY_SIF, bd
    bd = mod
    SIF_FUN = getattr(mod, "sif_two_tips_from_crack", None)
    TRY_SIF = getattr(mod, "_try_call_sif_two_tips", None)
    if SIF_FUN is None or TRY_SIF is None:
        try:
            from tools.cracked_disk_ddm import sif_two_tips_from_crack as _s, _try_call_sif_two_tips as _t
            if SIF_FUN is None:
                SIF_FUN = _s
            if TRY_SIF is None:
                TRY_SIF = _t
        except Exception:
            pass
    if SIF_FUN is None or TRY_SIF is None:
        raise RuntimeError("Cannot bind SIF hooks.")
    return SIF_FUN, TRY_SIF


def sif_hooks():
    """The bound ``(SIF_FUN, TRY_SIF)`` pair.

    Callers that keep their own module-level copies of these names read them
    out of their own namespace, which :func:`bind_sif_hooks` cannot reach. This
    lets them mirror the binding rather than silently hold ``None``.
    """
    if SIF_FUN is None or TRY_SIF is None:
        raise RuntimeError("SIF hooks are not bound; call bind_sif_hooks first")
    return SIF_FUN, TRY_SIF


def get_first_col(row, candidates, required=True, default=np.nan):
    for c in candidates:
        if c in row.index:
            return row[c]
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return default


def import_or_load_solver(solver_mod, meta_csv=None, debug=False):
    solver_mod = str(solver_mod).strip()
    for d in _search_dirs(meta_csv):
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        if debug:
            print(f"[solver] import_module('{solver_mod}')")
        return importlib.import_module(solver_mod)
    except Exception as e:
        if debug:
            print(f"[solver] failed: {e}")
    for d in _search_dirs(meta_csv):
        p = os.path.join(d, solver_mod + ".py")
        if os.path.isfile(p):
            if debug:
                print(f"[solver] file: {p}")
            return load_module_from_path(p, "bd")
    raise RuntimeError(f"Cannot find solver '{solver_mod}'.")


def load_module_from_path(py_path, module_name="bd"):
    py_path = os.path.abspath(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"Solver not found: {py_path}")
    d = os.path.dirname(py_path)
    if d and d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sample_fields_physical(npz_path, row):
    """
    Load the richer fields needed for local anisotropic softening.

    Required exported keys:
      s1, s3, U_MPa, M, tensile_w, wp_weight, conf_w
    """
    d = np.load(npz_path, allow_pickle=False)

    needed = ["s1", "s3", "U_MPa", "M", "tensile_w", "wp_weight", "conf_w"]
    missing = [k for k in needed if k not in d]
    if missing:
        raise KeyError(f"{npz_path} is missing keys: {missing}")

    s1 = np.asarray(d["s1"], dtype=float)
    s3 = np.asarray(d["s3"], dtype=float)
    U = np.asarray(d["U_MPa"], dtype=float)  # MPa numerically == MJ/m^3
    M = np.asarray(d["M"]).astype(bool)
    tensile_w = np.asarray(d["tensile_w"], dtype=float)
    wp_weight = np.asarray(d["wp_weight"], dtype=float)
    conf_w = np.asarray(d["conf_w"], dtype=float)

    if not (s1.shape == s3.shape == U.shape == M.shape == tensile_w.shape == wp_weight.shape == conf_w.shape):
        raise ValueError(
            f"Shape mismatch in {npz_path}: "
            f"s1={s1.shape}, s3={s3.shape}, U={U.shape}, M={M.shape}, "
            f"tensile_w={tensile_w.shape}, wp_weight={wp_weight.shape}, conf_w={conf_w.shape}"
        )

    valid = (
        M &
        valid_mask_from_fields(s1, s3, U, tensile_w, wp_weight, conf_w)
    )

    if not np.any(valid):
        raise RuntimeError(f"No valid points found in {npz_path}")

    s1v = s1[valid]
    s3v = s3[valid]
    Uv = U[valid]
    wtv = np.clip(tensile_w[valid], 0.0, 1.0)
    wwpv = np.clip(wp_weight[valid], 0.0, 1.0)
    wcv = np.clip(conf_w[valid], 0.0, 1.0)

    T0 = float(row["Tensile_strength_Mpa"])
    C0 = float(row["Cohesion"])
    phi0 = np.deg2rad(float(row["Friction_Angle"]))

    Tv = np.full_like(s1v, T0, dtype=float)
    Cv = np.full_like(s1v, C0, dtype=float)
    phiv = np.full_like(s1v, phi0, dtype=float)

    return s1v, s3v, Uv, wtv, wwpv, wcv, Tv, Cv, phiv


def npz_path_for_sample(data_dir, sid):
    return os.path.join(data_dir, f"sample_{int(sid):04d}_full_fields.npz")


def output_prefix_from_records(sample_records):
    rocks = sorted(set(str(rec["rock"]).strip() for rec in sample_records))
    if len(rocks) == 1:
        r = rocks[0].strip().lower()
        if r == "augen gneiss":
            return "gneiss"
        # both spellings map to the same prefix, so regenerating the strength
        # table cannot silently rename the outputs
        if r in ("psammitic schist", "psammatic schist"):
            return "schist"
        return sanitize_name(rocks[0])
    return "mixed_rocks"


def replot_from_csvs(out_dir=None, sample_ids=None,
                     meta_csv="tensile_samples_data.csv"):
    # All fourteen specimens by default. This used to be range(8, 16), the
    # schist range, so a replot silently rebuilt only the schist figures and
    # left the gneiss ones as they were. The plots are grouped by rock when
    # they are written, so covering both lithologies is the correct default
    # and a caller wanting one rock can still say so.
    out_dir = out_dir or output_dirs.fields()
    if sample_ids is None:
        sample_ids = list(range(1, 15))
    labels = {}
    if os.path.isfile(meta_csv):
        try:
            dm = pd.read_csv(meta_csv, index_col=0)
            dm.columns = [str(c).strip() for c in dm.columns]
            for sid in sample_ids:
                if sid in dm.index:
                    labels[sid] = (str(dm.loc[sid, "Rock_type"]).strip(), float(dm.loc[sid, "Angle"]))
        except Exception:
            pass
    results_list = []
    for sid in sample_ids:
        trace_csv = os.path.join(out_dir, f"smoke_trace_sid{sid}.csv")
        path_csv = os.path.join(out_dir, f"smoke_path_sid{sid}.csv")
        if not os.path.isfile(trace_csv):
            print(f"  [replot] sid={sid}: not found — skip")
            continue
        trace = pd.read_csv(trace_csv)
        rock, ang = labels.get(sid, ("sid" + str(sid), ""))
        if os.path.isfile(path_csv):
            pc = pd.read_csv(path_csv)
            xs = pc.iloc[:, 0].to_numpy(float)
            ys = pc.iloc[:, 1].to_numpy(float)
        else:
            xs = np.array([0.0])
            ys = np.array([0.0])
        R = float(np.max(np.hypot(xs, ys))) if len(xs) > 1 else 1.0
        gc_vals = trace["Gc"].dropna()
        Gc0 = float(gc_vals.iloc[0]) if len(gc_vals) > 0 else np.nan
        results_list.append(dict(
            sample_id=sid, rock=rock, angle_deg=ang, R=R, xs=xs, ys=ys,
            trace=trace, stop_reason="loaded_from_csv", Gc0_internal_used=Gc0
        ))
    if not results_list:
        print("[replot] No CSVs found in:", out_dir)
        return
    _make_all_combined_plots(results_list)
    print(f"\n[replot] Done. Folder: {os.path.abspath(out_dir)}")


def sanitize_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "output"


def _show_in_notebook(fig, name=""):
    """Display a figure inline when running under a notebook kernel.

    These plot writers save to disk and close the figure, which is right for a
    batch script but leaves a notebook cell showing nothing but a filename. The
    guard keeps that behaviour outside a kernel, so the same function serves
    both callers.
    """
    try:
        from IPython import get_ipython
        from IPython.display import display, Markdown
    except ImportError:
        return
    if get_ipython() is None:
        return
    if name:
        display(Markdown(f"**{name}**"))
    display(fig)


def save_all_energy_plot(results_list, out_path):
    all_steps = []
    for res in results_list:
        t = res["trace"].dropna(subset=["step"])
        if len(t) > 0:
            all_steps.extend(t["step"].tolist())
    x_max = max(all_steps) if all_steps else 50
    x_max = max(x_max, 10)
    # Taller than the axes need, to hold a three-column legend of fourteen
    # entries below them without squeezing the plot.
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    legend_items = []
    for i, res in enumerate(results_list):
        t = res["trace"].dropna(subset=["step"]).copy()
        if len(t) == 0:
            continue
        G = t["G"].to_numpy(float)
        Gc = t["Gc"].to_numpy(float)
        steps = t["step"].to_numpy(float)
        c = _COLORS[i % len(_COLORS)]
        mrk = _get_marker(res)
        mev = max(1, len(steps) // 12)
        lbl = _label_short(res)
        lG, = ax.plot(steps, G, color=c, lw=1.5, ls="-", marker=mrk, ms=4.5, markevery=mev,
                      markerfacecolor=c, markeredgewidth=0.4, markeredgecolor="w", zorder=3)
        lGc, = ax.plot(steps, Gc, color=c, lw=1.0, ls="--", marker="", zorder=2)
        legend_items.append((lG, lGc, lbl, c, mrk))
    ax.set_yscale("log")
    ax.set_xlim(-x_max * 0.03, x_max * 1.06)
    ax.set_xlabel("Propagation step")
    ax.set_ylabel(r"Energy release rate (MPa$\cdot$m)")
    ax.grid(True, which="major", ls=":", lw=0.55, color="0.78", zorder=0)
    ax.grid(True, which="minor", ls=":", lw=0.25, color="0.90", zorder=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    leg_handles = []
    leg_labels = []
    for lG, lGc, lbl, c, mrk in legend_items:
        leg_handles.append(Line2D([0], [0], color=c, lw=1.5, ls="-", marker=mrk, ms=4.5,
                                  markerfacecolor=c, markeredgewidth=0.4, markeredgecolor="w"))
        leg_labels.append(f"{lbl}: $G$")
    for lG, lGc, lbl, c, mrk in legend_items:
        leg_handles.append(Line2D([0], [0], color=c, lw=1.0, ls="--"))
        leg_labels.append(f"{lbl}: $G_c$")
    # Anchored to the figure rather than the axes, so the block sits in the
    # space reserved below and its height does not depend on the axes box.
    fig.legend(leg_handles, leg_labels, fontsize=7.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), frameon=True, labelspacing=0.35,
               handlelength=1.8, columnspacing=1.0, borderaxespad=0.0)
    # Fixed margins rather than tight_layout: the two lithologies must come out
    # the same size. Their legend labels differ in width, so a tight bounding
    # box cropped each panel differently -- 539 against 623 pt -- and
    # \includegraphics[width=\textwidth] then scaled them by different factors,
    # which is what made one panel of the manuscript figure look smaller than
    # the other.
    fig.subplots_adjust(left=0.105, right=0.98, top=0.965, bottom=0.255)
    # The whole canvas, explicitly. Passing bbox_inches=None would not do it:
    # matplotlib reads None as "use rcParams['savefig.bbox']", which this
    # project sets to "tight". Only an explicit Bbox pins the output size.
    full = Bbox([[0.0, 0.0], list(fig.get_size_inches())])
    fig.savefig(out_path, bbox_inches=full)
    _show_in_notebook(fig, os.path.basename(str(out_path)))
    plt.close(fig)
    print(f"  Energy : {out_path}")


def save_all_paths_plot(results_list, out_path):
    fig, ax = plt.subplots(figsize=(4.8, 5.6))
    R_max = max(float(r["R"]) for r in results_list)
    R_mm = R_max * 1e3
    theta = np.linspace(0, 2 * np.pi, 720)
    ax.plot(R_mm * np.cos(theta), R_mm * np.sin(theta), color="0.15", lw=1.0, zorder=5)
    stub = R_mm * 0.10
    lw_stub = 3.0
    ax.plot([0, 0], [R_mm, R_mm + stub], color="0.15", lw=lw_stub, solid_capstyle="round")
    ax.plot([0, 0], [-R_mm, -R_mm - stub], color="0.15", lw=lw_stub, solid_capstyle="round")
    ax.annotate("", xy=(0, R_mm + stub * 0.3), xytext=(0, R_mm + stub * 1.0),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.0))
    ax.annotate("", xy=(0, -R_mm - stub * 0.3), xytext=(0, -R_mm - stub * 1.0),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.0))
    lstyles = ["-", "-", "--", "-.", "-", "--", "-."]
    for i, res in enumerate(results_list):
        n = int(len(res["trace"].dropna(subset=["step"])))
        xs = np.asarray(res["xs"], float) * 1e3
        ys = np.asarray(res["ys"], float) * 1e3
        lbl = _label_full(res)
        ax.plot(xs, ys, color=_COLORS[i % len(_COLORS)], lw=1.6,
                ls=lstyles[i % len(lstyles)], label=f"{lbl}  ($n$={n})", zorder=4 + i)
    ax.axhline(0, color="0.80", lw=0.4, ls=":", zorder=1)
    ax.axvline(0, color="0.80", lw=0.4, ls=":", zorder=1)
    pad = 1.18 * R_mm
    ax.set_aspect("equal", "box")
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad - stub * 1.8, pad + stub * 1.8)
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$y$ (mm)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=4, frameon=True, handlelength=2.0, labelspacing=0.3,
              columnspacing=0.8, borderaxespad=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path)
    _show_in_notebook(fig, os.path.basename(str(out_path)))
    plt.close(fig)
    print(f"  Paths  : {out_path}")


def save_all_ratio_plot(results_list, out_path):
    all_steps = []
    for res in results_list:
        t = res["trace"].dropna(subset=["step"])
        if len(t) > 0:
            all_steps.extend(t["step"].tolist())
    x_max = max(all_steps) if all_steps else 50
    x_max = max(x_max, 10)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhspan(1.0, 10.0, color="#E8E8E8", zorder=0)
    ax.axhline(1.0, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.text(x_max * 0.98, 4.5, r"$G/G_c \in [1,10]$", ha="right", va="center",
            fontsize=7.5, color="0.45", style="italic")
    for i, res in enumerate(results_list):
        t = res["trace"].dropna(subset=["step"]).copy()
        if len(t) == 0:
            continue
        G = t["G"].to_numpy(float)
        Gc = t["Gc"].to_numpy(float)
        ratio = G / (Gc + 1e-30)
        steps = t["step"].to_numpy(float)
        c = _COLORS[i % len(_COLORS)]
        mrk = _get_marker(res)
        mev = max(1, len(steps) // 12)
        ax.plot(steps, ratio, color=c, lw=1.5, ls="-", marker=mrk, ms=4.5, markevery=mev,
                markerfacecolor=c, markeredgewidth=0.4, markeredgecolor="w",
                label=_label_full(res), zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(-x_max * 0.03, x_max * 1.06)
    ax.set_xlabel("Propagation step")
    ax.set_ylabel(r"$G\,/\,G_c$")
    ax.grid(True, which="major", ls=":", lw=0.55, color="0.78", zorder=0)
    ax.grid(True, which="minor", ls=":", lw=0.25, color="0.90", zorder=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              frameon=True, labelspacing=0.3, handlelength=1.8, columnspacing=0.7,
              borderaxespad=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(out_path)
    _show_in_notebook(fig, os.path.basename(str(out_path)))
    plt.close(fig)
    print(f"  G/Gc   : {out_path}")


def save_full_fields_npz(
        npz_path, *, sid, rock, ang_deg,
        D, t, R, P, alpha_const, alpha_wp_line, sigma_ref_MPa,
        E1, E2, nu12, G12,
        X, Y, M, xg, yg,
        s11_m, s22_m, s12_m,
        sxx, syy, txy,
        s1, s3, th,
        sigma_vm, U,
        Rt_eff, Rs_eff, beta_crit,
        psi_pref_img, psi_pref_guided_img,
        tensile_w_img, conf_img, wp_weight_img, w_load_img, w_guid_img,
        xs, ys,
        cell_area, U_mean, U_max, U_total_J,
        modes,
):
    mode_code_img = _mode_to_code_img(M, modes)
    np.savez_compressed(
        npz_path,
        sample_id=np.int32(sid), rock=np.array(str(rock), dtype="U128"),
        angle_deg=np.float64(ang_deg),
        D_m=np.float64(D), t_m=np.float64(t), R_m=np.float64(R), P_N=np.float64(P),
        alpha_const_rad=np.float64(alpha_const), alpha_wp_line_rad=np.float64(alpha_wp_line),
        sigma_ref_MPa=np.float64(sigma_ref_MPa),
        E1_MPa=np.float64(E1), E2_MPa=np.float64(E2), nu12=np.float64(nu12), G12_MPa=np.float64(G12),
        cell_area_m2=np.float64(cell_area), U_mean_MPa=np.float64(U_mean),
        U_max_MPa=np.float64(U_max), U_total_J=np.float64(U_total_J),
        X=X, Y=Y, M=M.astype(np.uint8), xg=xg, yg=yg,
        s11_m=_img_from_mask(M, s11_m), s22_m=_img_from_mask(M, s22_m), s12_m=_img_from_mask(M, s12_m),
        sxx=_img_from_mask(M, sxx), syy=_img_from_mask(M, syy), txy=_img_from_mask(M, txy),
        s1=_img_from_mask(M, s1), s3=_img_from_mask(M, s3), th_rad=_img_from_mask(M, th),
        sigma_vm_MPa=_img_from_mask(M, sigma_vm), U_MPa=_img_from_mask(M, U),
        Rt_eff=_img_from_mask(M, Rt_eff), Rs_eff=_img_from_mask(M, Rs_eff),
        beta_crit_rad=_img_from_mask(M, beta_crit),
        psi_pref_rad=psi_pref_img, psi_pref_guided_rad=psi_pref_guided_img,
        tensile_w=tensile_w_img, conf_w=conf_img, wp_weight=wp_weight_img,
        load_weight=w_load_img, guide_weight=w_guid_img,
        mode_code=mode_code_img,
        mode_labels=np.array(["no_failure", "tensile", "shear", "mixed"], dtype="U16"),
        crack_x_m=np.asarray(xs, dtype=float), crack_y_m=np.asarray(ys, dtype=float),
    )


def save_sample_fields_npz(npz_path, *, sid, rock, ang_deg, alpha_rad,
                           D, t, R,
                           E1_base, E2_base, nu12, G12_base,
                           X, Y, mask,
                           combined, sxx, syy, txy, s1, s3,
                           w_band, w_active,
                           angle_local, warp_m,
                           p_peak_mpa, w_comp, b_contact_m,
                           band_angle_rad, phase_plot):
    np.savez_compressed(
        npz_path,
        sample_id=np.int32(sid),
        rock=np.array(str(rock), dtype="U128"),
        angle_deg=np.float64(ang_deg),
        alpha_rad=np.float64(alpha_rad),
        D_m=np.float64(D),
        t_m=np.float64(t),
        R_m=np.float64(R),
        E1_base_MPa=np.float64(E1_base),
        E2_base_MPa=np.float64(E2_base),
        nu12=np.float64(nu12),
        G12_base_MPa=np.float64(G12_base),
        X=X,
        Y=Y,
        mask=mask.astype(np.uint8),
        combined_MPa=combined,
        sxx_MPa=sxx,
        syy_MPa=syy,
        txy_MPa=txy,
        s1_MPa=s1,
        s3_MPa=s3,
        w_band=w_band,
        w_active=w_active,
        angle_local_rad=angle_local,
        warp_m=warp_m,
        p_peak_mpa=np.float64(p_peak_mpa),
        w_comp=np.float64(w_comp),
        b_contact_m=np.float64(b_contact_m),
        band_angle_rad=np.float64(band_angle_rad),
        phase_plot_m=np.float64(phase_plot),
    )


def validate_solver_api(mod):
    req = [
        "map_angle_to_alpha", "eval_stress_field_material", "fit_orthotropic_airy_disk",
        "solve_cracked_disk_correction_ddm", "_extract_tip_sifs", "_seg_intersect",
        "G_from_K_orthotropic", "Gc_theta_weak_plane",
        "points_in_disk", "rot_to_material", "stress_material_to_global",
        "principal_from_components", "failure_mode_map", "compute_psi_pref_field"
    ]
    miss = [n for n in req if not hasattr(mod, n)]
    if miss:
        raise RuntimeError("Solver missing: " + ", ".join(miss))

# --------------------------------------------------------------------------
# smoothing: Smoothing, kernel density, bootstrap bands and curve collapse.
# --------------------------------------------------------------------------

def compute_poly_ci(angles, poly_coeffs, angles_data, observed_data, n_params=5):
    angles = np.asarray(angles)
    angles_data = np.asarray(angles_data)
    observed_data = np.asarray(observed_data)

    # Fit polynomial and get residuals
    poly_pred = np.polyval(poly_coeffs, angles_data)
    residuals = observed_data - poly_pred

    # Residual variance estimate
    if len(angles_data) > n_params:
        sigma_squared = np.sum(residuals**2) / (len(angles_data) - n_params)
    else:
        sigma_squared = 0.0

    # Design matrix for original data
    X = np.vander(angles_data, N=len(poly_coeffs), increasing=True)

    if sigma_squared > 0:
        cov_matrix = sigma_squared * np.linalg.inv(X.T @ X)
    else:
        cov_matrix = np.zeros((len(poly_coeffs), len(poly_coeffs)))

    # Design matrix for prediction points
    X_dense = np.vander(angles, N=len(poly_coeffs), increasing=True)

    # Prediction variance
    pred_var = np.sum((X_dense @ cov_matrix) * X_dense, axis=1)
    pred_var = np.clip(pred_var, a_min=0, a_max=None)
    pred_std = np.sqrt(pred_var)

    # 95% CI using t-distribution
    df = len(angles_data) - n_params
    t_val = stats.t.ppf(0.975, df) if df > 0 else 0.0   # ← key fix

    ci_half_width = t_val * pred_std
    poly_fit = np.polyval(poly_coeffs, angles)
    ci_lower = poly_fit - ci_half_width
    ci_upper = poly_fit + ci_half_width

    return poly_fit, ci_lower, ci_upper


def gaussian_kde_manual(x, grid, bw):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)

    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z ** 2).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))
    return dens


def gaussian_smooth_1d(y, sigma_pts=2.0):
    sigma_pts = float(sigma_pts)
    if sigma_pts <= 0:
        return y.copy()
    radius = int(np.ceil(4.0 * sigma_pts))
    k = np.arange(-radius, radius + 1)
    g = np.exp(-0.5 * (k / (sigma_pts + 1e-30))**2)
    g /= np.sum(g)
    y_pad = np.pad(y, (radius, radius), mode="edge")
    return np.convolve(y_pad, g, mode="valid")


def mean_profile_midband(X, Y, U, diameter, band_frac=0.02, nbins=80):
    R = diameter / 2.0
    band_half = float(band_frac) * float(diameter)

    sel = np.abs(Y) <= band_half
    Xb = X[sel]
    Ub = U[sel]

    good = np.isfinite(Xb) & np.isfinite(Ub)
    Xb = Xb[good]
    Ub = Ub[good]
    if Xb.size == 0:
        return np.array([]), np.array([])

    edges = np.linspace(-R, R, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    inds = np.digitize(Xb, edges) - 1

    means = np.full(int(nbins), np.nan)
    for i in range(int(nbins)):
        m = inds == i
        if np.any(m):
            means[i] = np.nanmean(Ub[m])

    ok = np.isfinite(means)
    return centers[ok], means[ok]


def nan_box_blur(a, iters=1):
    out = np.array(a, float)
    H, W = out.shape
    for _ in range(int(iters)):
        p = np.pad(out, 1, mode="constant", constant_values=np.nan)
        acc = np.zeros((H, W), float)
        cnt = np.zeros((H, W), float)
        for di in range(3):
            for dj in range(3):
                patch = p[di:di+H, dj:dj+W]
                m = ~np.isnan(patch)
                acc += np.where(m, patch, 0.0)
                cnt += m.astype(float)
        out = acc / np.maximum(cnt, 1.0)
    return out


def nearest_mean(stats, target_angle, default_value):
    if len(stats) == 0:
        return default_value
    idx = np.argmin(np.abs(stats.index.to_numpy(float) - target_angle))
    return float(stats["mean"].iloc[idx])


def robust_bandwidth(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 3.0

    std = np.std(x, ddof=1)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    sigma = min(std, iqr / 1.349) if iqr > 0 else std

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(std, 1.0)

    bw = 1.06 * sigma * n ** (-1.0 / 5.0)
    if not np.isfinite(bw) or bw <= 0:
        bw = 3.0

    return max(bw, 2.0)

# --------------------------------------------------------------------------
# solver: Finite-difference operators, tractions and the energy-guided stepper.
# --------------------------------------------------------------------------

def _adaptive_N_outer(n_segs, N_outer_max):
    if n_segs <= 5:
        return 20
    if n_segs <= 10:
        return max(20, N_outer_max // 3)
    if n_segs <= 20:
        return max(30, N_outer_max // 2)
    return int(N_outer_max)


def _adaptive_kink_n(kink_n_full, rn):
    if rn < 0.25:
        return max(21, kink_n_full // 3)
    if rn < 0.50:
        return max(31, kink_n_full // 2)
    return int(kink_n_full)


def _normalize_rock_name(name):
    s = str(name).strip().lower()
    s = " ".join(s.split())
    return s


def bilinear_sample_scalar(img, X, Y, x, y, fill=np.nan):
    x1d, y1d = _grid_axes_from_mesh(X, Y)
    nx = len(x1d)
    ny = len(y1d)
    if not (x1d[0] <= x <= x1d[-1]) or not (y1d[0] <= y <= y1d[-1]):
        return float(fill)
    ix = int(np.searchsorted(x1d, x) - 1)
    iy = int(np.searchsorted(y1d, y) - 1)
    ix = max(0, min(ix, nx - 2))
    iy = max(0, min(iy, ny - 2))
    x0, x1 = x1d[ix], x1d[ix + 1]
    y0, y1 = y1d[iy], y1d[iy + 1]
    if (x1 - x0) == 0 or (y1 - y0) == 0:
        return float(fill)
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    f00 = img[iy, ix]
    f10 = img[iy, ix + 1]
    f01 = img[iy + 1, ix]
    f11 = img[iy + 1, ix + 1]
    if not (np.isfinite(f00) and np.isfinite(f10) and np.isfinite(f01) and np.isfinite(f11)):
        return float(fill)
    f0 = (1 - tx) * f00 + tx * f10
    f1 = (1 - tx) * f01 + tx * f11
    return float((1 - ty) * f0 + ty * f1)


def bilinear_scalar(img, X, Y, x, y, fill=np.nan):
    x1d, y1d = _grid_axes(X, Y)
    if not (x1d[0] <= x <= x1d[-1]) or not (y1d[0] <= y <= y1d[-1]):
        return float(fill)
    ix = int(np.clip(np.searchsorted(x1d, x) - 1, 0, len(x1d) - 2))
    iy = int(np.clip(np.searchsorted(y1d, y) - 1, 0, len(y1d) - 2))
    x0, x1e, y0, y1e = x1d[ix], x1d[ix + 1], y1d[iy], y1d[iy + 1]
    if (x1e - x0) == 0 or (y1e - y0) == 0:
        return float(fill)
    tx, ty = (x - x0) / (x1e - x0), (y - y0) / (y1e - y0)
    f = img[iy, ix], img[iy, ix + 1], img[iy + 1, ix], img[iy + 1, ix + 1]
    if not all(np.isfinite(v) for v in f):
        return float(fill)
    return float((1 - ty) * ((1 - tx) * f[0] + tx * f[1]) + ty * ((1 - tx) * f[2] + tx * f[3]))


def check_force_equilibrium(sigma_xx, sigma_yy, tau_xy, x, y, diameter, applied_force=None):
    """
    Enhanced equilibrium check that prints out numeric values and returns them.
    :param sigma_xx, sigma_yy, tau_xy: stress component arrays
    :param x, y: coordinate arrays
    :param diameter: diameter of sample
    :param applied_force: optional, the force intended to apply (N). 
                          If not provided, the function can attempt an estimate.
    :return: dict with computed net forces and whether they are in tolerance
    """
    radius = diameter / 2
    num_points = len(x)
    
    # Approximate area per point if the mesh is somewhat uniform:
    total_area = np.pi * radius**2
    area_per_point = total_area / num_points
    
    # Force in x / y from normal stresses
    force_x = sigma_xx * area_per_point  # N
    force_y = sigma_yy * area_per_point  # N
    
    # If there's shear, we might also integrate tau_xy over the boundary,
    # but for a simple check, we just do the normal components' net effect.
    net_force_x = np.sum(force_x)
    net_force_y = np.sum(force_y)
    
    # If user does not provide an 'applied_force', estimate from max stress:
    if applied_force is None:
        max_stress = np.max(np.abs([sigma_xx, sigma_yy]))
        applied_force = max_stress * total_area
    
    # For demonstration, we define tolerance as 5% of the applied_force
    tolerance = 0.05 * applied_force
    
    # Print or log the results:
    print(f"---- Stress Equilibrium Check ----")
    print(f"Calculated Net Force in X: {net_force_x:.3e} N")
    print(f"Calculated Net Force in Y: {net_force_y:.3e} N")
    print(f"User-Specified (or Estimated) Applied Force: {applied_force:.3e} N")
    print(f"±5% Tolerance of Applied Force: ±{tolerance:.3e} N")
    
    balanced_x = (abs(net_force_x) <= tolerance)
    balanced_y = (abs(net_force_y) <= tolerance)
    pass_equilibrium = balanced_x and balanced_y
    
    if pass_equilibrium:
        print("Result: Forces are balanced within ±5% tolerance.\n")
    else:
        print("Result: Forces exceed ±5% tolerance.\n")
    
    return {
        'net_force_x': net_force_x,
        'net_force_y': net_force_y,
        'applied_force': applied_force,
        'tolerance': tolerance,
        'is_equilibrium': pass_equilibrium
    }


def integrate_line_force(x, sigma_pa, thickness):
    idx = np.argsort(x)
    x = x[idx]
    s = sigma_pa[idx]
    return float(np.trapezoid(s * thickness, x))


def traction_from_coeffs_on_boundary(theta, R, p1, p2, a1, a2):
    """
    Compute boundary tractions (t_r, t_t) on circle r=R given polynomial
    coefficients for Lekhnitskii potentials.

    a1, a2: complex arrays length M+1 (only m>=2 used)
    """
    theta = np.asarray(theta, float)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z1 = x + p1 * y
    z2 = x + p2 * y

    M = len(a1) - 1
    f1pp = np.zeros_like(z1, dtype=complex)
    f2pp = np.zeros_like(z2, dtype=complex)

    for m in range(2, M + 1):
        mm = m * (m - 1)
        f1pp += a1[m] * mm * (z1 ** (m - 2))
        f2pp += a2[m] * mm * (z2 ** (m - 2))

    # Stress components in global axes (the convention)
    sxx = np.real(p1**2 * f1pp + p2**2 * f2pp)
    syy = np.real(f1pp + f2pp)
    txy = -np.real(p1 * f1pp + p2 * f2pp)

    nx = np.cos(theta)
    ny = np.sin(theta)

    tx = sxx * nx + txy * ny
    ty = txy * nx + syy * ny

    tr = tx * nx + ty * ny
    tt = -tx * ny + ty * nx
    return tr, tt
