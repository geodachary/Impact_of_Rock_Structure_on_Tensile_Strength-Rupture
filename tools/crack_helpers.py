# crack_helpers.py
# -*- coding: utf-8 -*-
"""
Helpers for Brazilian disk crack modeling (Airy-fit driven).

This revision is physics-driven and avoids the “always vertical” artifact:

What changed vs your overly-vertical version:
- NO hard centerline corridor forcing.
- Graph solver now follows a physics preferred direction field (tensile/shear/weak-plane mix)
  using an alignment penalty instead of forcing x≈0.
- Start is still on TOP platen arc and goal on BOTTOM platen arc, but not forced to x=0.
- Includes a local physics stepper (often more realistic than global graph).
  Use start_mode="platen_to_platen" for pure physics growth.

Default start_mode is set to "platen_to_platen" (physics growth).
You can switch to graph with start_mode="graph_platen".

Requirements:
- geometry_helpers: angle_diff_periodic, _wrap_pi
- rotation_helpers: rot_to_material
- stress_helpers: stress_material_to_global, principal_from_components, eval_stress_field_material
- failure_mapping_helpers: failure_mode_map
- cracked_disk_ddm: _try_call_sif_two_tips, sif_two_tips_from_crack (optional, kept)
"""

from __future__ import annotations

import numpy as np
import heapq

from.geometry_helpers import angle_diff_periodic, _wrap_pi

from.rotation_helpers import rot_to_material
from.stress_helpers import (
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
)
from.failure_mapping_helpers import failure_mode_map

# Optional (kept)
from.cracked_disk_ddm import (_try_call_sif_two_tips, sif_two_tips_from_crack, traction_from_element_u,
                              total_traction_at_point_material, kink_angle_pls_anisotropic_from_traction, solve_cracked_disk_correction_ddm, traction_from_element, orthotropic_Q_plane_stress, rotate_Q, p_roots_quartic,
                              compliance_from_Q, reduced_compliance_components, stroh_LA_inplane)


# -----------------------------
# Toughness model Gc(theta)
# -----------------------------
def Gc_theta_weak_plane(
    theta_line,              # crack LINE direction in global radians
    alpha_wp,                # weak-plane LINE direction in global radians
    Gc_matrix=1.0,           # baseline matrix toughness (relative units ok)
    weak_reduction=0.35,     # along weak plane: Gc_min = (1-weak_reduction)*Gc_matrix
    eta_deg=10.0,            # angular width of weak plane effect
):
    """
    Directional toughness model (line direction: theta == theta+pi).
    Produces a smooth "valley" in Gc when crack aligns with weak plane.

    Gc(theta) = Gc_matrix * [1 - weak_reduction * exp( -sin^2(theta-alpha_wp)/eta^2 )]
    """
    # line-angle difference using sin^2 periodicity
    eta = np.deg2rad(float(eta_deg)) + 1e-30
    d = float(theta_line - alpha_wp)
    s2 = np.sin(d) ** 2
    valley = np.exp(-s2 / (eta ** 2))
    return float(Gc_matrix * (1.0 - float(weak_reduction) * valley))


# =============================================================================
# Vector-safe angle helpers
# =============================================================================

def _wrap_pi(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi



def wrap_pi_half(a):
    """Wrap line-angle(s) to (-pi/2, pi/2]. Scalars or numpy arrays."""
    a = np.asarray(a, dtype=float)
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def line_angle_diff(a: float, b: float) -> float:
    # smallest difference for LINE angles
    return abs(wrap_pi_half(float(a) - float(b)))

def Eprime_equiv(E1, E2):
    # pragmatic equivalent modulus (plane stress-ish scaling)
    return float(np.sqrt(max(E1, 1e-30) * max(E2, 1e-30)))

def G_from_K_equiv(KI, KII, Eprime):
    KI = float(KI); KII = float(KII)
    return float((KI*KI + KII*KII) / (float(Eprime) + 1e-30))


def G_from_K_modular(KI, KII, *, E1, E2, nu12, G12, mode="iso_equiv", Eprime_user=None):
    """
    mode:
      - "iso_equiv": G = (KI^2+KII^2)/Eprime, with Eprime = sqrt(E1*E2)
      - "user_Eprime": same but uses Eprime_user
    """
    KI = float(KI); KII = float(KII)
    if mode == "user_Eprime":
        Eprime = float(Eprime_user)
    else:
        Eprime = float(np.sqrt(max(E1,1e-30)*max(E2,1e-30)))
    return float((KI*KI + KII*KII) / (Eprime + 1e-30))

def _align_line_direction(psi_target, psi_ref):
    """Align a LINE direction so psi and psi+pi are equivalent; pick closest to psi_ref (scalars)."""
    psi_target = _wrap_pi(float(psi_target))
    psi_ref = _wrap_pi(float(psi_ref))
    d1 = abs(angle_diff_periodic(psi_target, psi_ref))
    d2 = abs(angle_diff_periodic(psi_target + np.pi, psi_ref))
    return _wrap_pi(psi_target if d1 <= d2 else (psi_target + np.pi))


def _angle_mean_line_scalar(a, b, w):
    """Mean for LINE angles (psi == psi+pi) using doubled-angle averaging. Scalars."""
    a = float(a)
    b = float(b)
    w = float(w)
    za = np.exp(2j * a)
    zb = np.exp(2j * b)
    z = (1.0 - w) * za + w * zb
    if abs(z) < 1e-20:
        return float(wrap_pi_half(a))
    return float(wrap_pi_half(0.5 * np.angle(z)))


def _angle_mean_line_vec(a, b, w):
    """
    Vectorized mean for LINE angles via doubled-angle averaging.
    a,b,w can be arrays broadcastable.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.asarray(w, float)
    za = np.exp(2j * a)
    zb = np.exp(2j * b)
    z = (1.0 - w) * za + w * zb
    ang = 0.5 * np.angle(z)
    return wrap_pi_half(ang)


def _line_angle_diff(a, b):
    """
    Smallest difference between two LINE angles (a == a+pi).
    Returns value in [0, pi/2].
    """
    d = wrap_pi_half(np.asarray(a, float) - np.asarray(b, float))
    return np.abs(d)


# =============================================================================
# Grid bilinear sampler
# =============================================================================

class UniformGridSampler:
    def __init__(self, X, Y):
        self.xmin = float(X[0, 0])
        self.xmax = float(X[0, -1])
        self.ymin = float(Y[0, 0])
        self.ymax = float(Y[-1, 0])
        self.nx = int(X.shape[1])
        self.ny = int(X.shape[0])
        self.dx = (self.xmax - self.xmin) / max(self.nx - 1, 1)
        self.dy = (self.ymax - self.ymin) / max(self.ny - 1, 1)

    def sample(self, F, x, y, fill=np.nan):
        x = float(x)
        y = float(y)
        if (x < self.xmin) or (x > self.xmax) or (y < self.ymin) or (y > self.ymax):
            return float(fill)

        fx = (x - self.xmin) / self.dx
        fy = (y - self.ymin) / self.dy
        j0 = int(np.floor(fx))
        i0 = int(np.floor(fy))
        j1 = min(j0 + 1, self.nx - 1)
        i1 = min(i0 + 1, self.ny - 1)
        tx = fx - j0
        ty = fy - i0

        f00 = F[i0, j0]
        f10 = F[i0, j1]
        f01 = F[i1, j0]
        f11 = F[i1, j1]

        if not all(np.isfinite(v) for v in [f00, f10, f01, f11]):
            vals = np.array([f00, f10, f01, f11], dtype=float)
            ok = np.isfinite(vals)
            return float(vals[ok][0]) if np.any(ok) else float(fill)

        return float(
            (1 - tx) * (1 - ty) * f00
            + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01
            + tx * ty * f11
        )


# =============================================================================
# Stress evaluation (vectorized)
# =============================================================================

def stress_global_from_fit(xg, yg, R, alpha_const, fit):
    """
    Vectorized stress evaluation in global coordinates.
    xg, yg can be scalars or arrays.
    """
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    xm, ym = rot_to_material(xg, yg, float(alpha_const))
    sxx_m, syy_m, txy_m = eval_stress_field_material(
        xm,
        ym,
        float(R),
        fit["p1"],
        fit["p2"],
        fit["a1"],
        fit["a2"],
    )
    return stress_material_to_global(sxx_m, syy_m, txy_m, float(alpha_const))


# =============================================================================
# Pointwise ratios + directions (safe: accepts **_unused)
# =============================================================================

def ratios_and_dirs_at_point(
    x,
    y,
    R,
    alpha_const,
    fit,
    sampler,
    Teff_img,
    Coh_img,
    Phi_img,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12,
    **_unused,
):
    Teff = sampler.sample(Teff_img, x, y, fill=np.nan)
    Coh = sampler.sample(Coh_img, x, y, fill=np.nan)
    Phi = sampler.sample(Phi_img, x, y, fill=np.nan)
    if (not np.isfinite(Teff)) or (not np.isfinite(Coh)) or (not np.isfinite(Phi)):
        return np.nan, np.nan, np.nan, np.nan

    sxx, syy, txy = stress_global_from_fit(x, y, R, alpha_const, fit)
    sxx = float(np.asarray(sxx))
    syy = float(np.asarray(syy))
    txy = float(np.asarray(txy))

    s1, _, th = principal_from_components(
        np.array([sxx]), np.array([syy]), np.array([txy])
    )
    s1 = float(s1[0])
    th = float(th[0])

    weak_phi_arr = None
    if weak_phi is not None:
        weak_phi_arr = np.array([float(weak_phi)], dtype=float)

    _, Rt_eff, Rs_eff, beta_crit = failure_mode_map(
        np.array([sxx]),
        np.array([syy]),
        np.array([txy]),
        np.array([s1]),
        Tm=np.array([Teff]),
        Coh=np.array([Coh]),
        Phi=np.array([Phi]),
        alpha_const=float(alpha_const),
        basis="first",
        util_min=0.0,
        mixed_band=0.45,
        margin=0.0,
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=str(strength_model),
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi_arr,
        eps=float(eps),
    )

    Rt = float(Rt_eff[0])
    Rs = float(Rs_eff[0])
    bc = float(beta_crit[0])

    # tensile plane direction (line direction)
    psi_t = _wrap_pi(th + np.pi / 2.0)
    # shear plane direction (line direction)
    psi_s = _wrap_pi(bc + np.pi / 2.0)

    return Rt, Rs, psi_t, psi_s


def util_at_point(
    x,
    y,
    R,
    alpha_const,
    fit,
    sampler,
    Teff_img,
    Coh_img,
    Phi_img,
    **failure_kwargs,
):
    Rt, Rs, _, _ = ratios_and_dirs_at_point(
        x,
        y,
        R,
        alpha_const,
        fit,
        sampler,
        Teff_img,
        Coh_img,
        Phi_img,
        **failure_kwargs,
    )
    if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
        return np.nan
    return float(max(Rt, Rs))


# =============================================================================
# Endpoint enforcement
# =============================================================================

def straighten_endcaps_parallel_to_load(
    xs, ys, R, beta_deg, extra_deg=6.0, ncap=14, power=2.0
):
    xs2 = np.asarray(xs, float).copy()
    ys2 = np.asarray(ys, float).copy()
    N = len(xs2)
    if N < 6:
        return xs2, ys2

    beta = np.deg2rad(float(beta_deg) + float(extra_deg))
    ncap = int(max(3, min(int(ncap), N - 1)))
    power = float(max(power, 0.5))

    def _do_end(end_idx):
        x_end = float(xs2[end_idx])
        y_end = float(ys2[end_idx])
        th = float(np.arctan2(y_end, x_end))
        center = (np.pi / 2.0) if (y_end >= 0) else (-np.pi / 2.0)
        if abs(float(angle_wrap_pi(th - center))) > beta:
            return

        if end_idx == 0:
            idxs = np.arange(0, ncap, dtype=int)
        else:
            idxs = np.arange(N - 1, N - 1 - ncap, -1, dtype=int)

        s = np.linspace(1.0, 0.0, len(idxs))
        w = s ** power
        xs2[idxs] = w * x_end + (1.0 - w) * xs2[idxs]

    _do_end(0)
    _do_end(-1)
    return xs2, ys2


def _extend_point_to_circle(p0, p1, R):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    vx, vy = x1 - x0, y1 - y0
    if vx * vx + vy * vy < 1e-20:
        rr = np.hypot(x1, y1) + 1e-30
        return (R * x1 / rr, R * y1 / rr)

    A = vx * vx + vy * vy
    B = 2.0 * (x1 * vx + y1 * vy)
    C = x1 * x1 + y1 * y1 - R * R
    disc = B * B - 4 * A * C
    if disc < 0:
        rr = np.hypot(x1, y1) + 1e-30
        return (R * x1 / rr, R * y1 / rr)

    t1 = (-B + np.sqrt(disc)) / (2 * A)
    t2 = (-B - np.sqrt(disc)) / (2 * A)
    t = max(t1, t2)
    return (x1 + t * vx, y1 + t * vy)


def enforce_platen_endpoints(
    xs, ys, R, beta_deg, snap_extra_deg=8.0, snap_to_platen_center=False
):
    xs = np.asarray(xs, float).copy()
    ys = np.asarray(ys, float).copy()
    if len(xs) < 3:
        return xs, ys

    if np.hypot(xs[0], ys[0]) < 0.98 * R:
        xs[0], ys[0] = _extend_point_to_circle((xs[1], ys[1]), (xs[0], ys[0]), R)
    if np.hypot(xs[-1], ys[-1]) < 0.98 * R:
        xs[-1], ys[-1] = _extend_point_to_circle(
            (xs[-2], ys[-2]), (xs[-1], ys[-1]), R
        )

    beta = np.deg2rad(float(beta_deg) + float(snap_extra_deg))
    for idx in [0, -1]:
        x, y = float(xs[idx]), float(ys[idx])
        th = float(np.arctan2(y, x))
        center = (np.pi / 2.0) if (y >= 0) else (-np.pi / 2.0)

        if snap_to_platen_center:
            xs[idx] = 0.0
            ys[idx] = float(R * np.sin(center))
        else:
            dth = float(angle_wrap_pi(th - center))
            dth = float(np.clip(dth, -beta, beta))
            th2 = center + dth
            xs[idx] = float(R * np.cos(th2))
            ys[idx] = float(R * np.sin(th2))

    return xs, ys


# =============================================================================
# Platen masks
# =============================================================================

def _make_platen_masks(X, Y, R, beta_deg, extra_deg=8.0, r_frac=0.94):
    r = np.hypot(X, Y)
    inside = r <= (0.999 * float(R))
    beta = np.deg2rad(float(beta_deg) + float(extra_deg))
    th = np.arctan2(Y, X)

    top_arc = (
        inside
        & (Y > 0)
        & (np.abs(angle_wrap_pi(th - np.pi / 2.0)) <= beta)
        & (r >= r_frac * R)
    )
    bot_arc = (
        inside
        & (Y < 0)
        & (np.abs(angle_wrap_pi(th + np.pi / 2.0)) <= beta)
        & (r >= r_frac * R)
    )
    return inside, top_arc, bot_arc


# =============================================================================
# Physics grids: utility + preferred propagation direction (vectorized)
# =============================================================================

def _compute_utility_and_dir_grids(
    R,
    alpha_const,
    fit,
    X,
    Y,
    Teff_img,
    Coh_img,
    Phi_img,
    mixed_band=0.45,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12,
    # direction mixing (physics)
    dir_tensile_base=0.35,      # base weight toward tensile direction
    dir_tensile_gain=0.55,      # additional weight proportional to tensile fraction
    weak_plane_follow=0.35,     # pull psi_pref toward weak-plane when shear dominates (0..1)
):
    """
    Returns:
      U:        utility grid (higher is easier to fail/crack)
      psi_pref: preferred crack-line direction at each cell (line angle in (-pi/2, pi/2])
      tfrac:    tensile fraction Rt/(Rt+Rs)
    """
    ny, nx = X.shape
    U = np.full((ny, nx), np.nan, dtype=float)
    psi_pref = np.full((ny, nx), np.nan, dtype=float)
    tfrac_grid = np.full((ny, nx), np.nan, dtype=float)

    r = np.hypot(X, Y)
    inside = (
        (r <= 0.999 * float(R))
        & np.isfinite(Teff_img)
        & np.isfinite(Coh_img)
        & np.isfinite(Phi_img)
    )
    ii, jj = np.where(inside)
    if ii.size < 10:
        return U, psi_pref, tfrac_grid

    x = X[ii, jj].ravel()
    y = Y[ii, jj].ravel()

    sxx, syy, txy = stress_global_from_fit(x, y, float(R), float(alpha_const), fit)
    sxx = np.asarray(sxx, float).ravel()
    syy = np.asarray(syy, float).ravel()
    txy = np.asarray(txy, float).ravel()

    s1, _, th = principal_from_components(sxx, syy, txy)
    th = np.asarray(th, float).ravel()

    Tm = np.asarray(Teff_img[ii, jj], float).ravel()
    Coh = np.asarray(Coh_img[ii, jj], float).ravel()
    Phi = np.asarray(Phi_img[ii, jj], float).ravel()

    weak_phi_arr = None
    if weak_phi is not None:
        weak_phi_arr = np.full_like(Tm, float(weak_phi), dtype=float)

    _, Rt_eff, Rs_eff, beta_crit = failure_mode_map(
        sxx,
        syy,
        txy,
        s1,
        Tm=Tm,
        Coh=Coh,
        Phi=Phi,
        alpha_const=float(alpha_const),
        basis="first",
        util_min=0.0,
        mixed_band=float(mixed_band),
        margin=0.0,
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=str(strength_model),
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi_arr,
        eps=float(eps),
    )

    Rt_eff = np.asarray(Rt_eff, float).ravel()
    Rs_eff = np.asarray(Rs_eff, float).ravel()
    beta_crit = np.asarray(beta_crit, float).ravel()

    # utility (keep your original spirit: max tensile/shear with mild tensile-favor if present)
    rtmax = np.nanmax(Rt_eff) if np.any(np.isfinite(Rt_eff)) else 0.0
    rsmax = np.nanmax(Rs_eff) if np.any(np.isfinite(Rs_eff)) else 0.0
    shear_weight = 0.85 if (rtmax >= 0.55 * rsmax) else 1.0
    Uloc = np.maximum(Rt_eff, shear_weight * Rs_eff)

    # direction fields (line directions)
    psi_t = wrap_pi_half(th + np.pi / 2.0)  # tensile plane
    psi_s = wrap_pi_half(beta_crit + np.pi / 2.0)  # shear plane

    tf = Rt_eff / (Rt_eff + Rs_eff + float(eps))
    tf = np.clip(tf, 0.0, 1.0)

    # mix tensile vs shear direction
    w = float(dir_tensile_base) + float(dir_tensile_gain) * tf
    w = np.clip(w, 0.10, 0.98)
    psi = _angle_mean_line_vec(psi_s, psi_t, w)

    # optionally pull toward weak plane when shear dominates
    if str(strength_model).lower() == "weak_plane" and float(weak_plane_follow) > 0.0:
        psi_wp = wrap_pi_half(float(alpha_const))  # weak-plane line direction in global
        wwp = float(weak_plane_follow) * (1.0 - tf)
        wwp = np.clip(wwp, 0.0, 0.95)
        psi = _angle_mean_line_vec(psi, psi_wp, wwp)

    U[ii, jj] = Uloc
    psi_pref[ii, jj] = psi
    tfrac_grid[ii, jj] = tf
    return U, psi_pref, tfrac_grid


# =============================================================================
# Graph (Dijkstra) solver with PHYSICS directional alignment (not forced vertical)
# =============================================================================

def crack_path_platen_to_platen_graph(
    R,
    alpha_const,
    fit,
    X,
    Y,
    Teff_img,
    Coh_img,
    Phi_img,
    platen_beta_deg=10.0,
    platen_snap_extra_deg=10.0,
    snap_to_platen_center=False,
    mixed_band=0.45,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    # mild geometric penalties only (keep small to avoid forcing vertical)
    rim_weight=0.08,
    rim_power=6.0,
    # directional physics weight (main control)
    dir_weight=1.8,             # higher => path follows psi_pref more
    u_floor=1e-3,
    # seed selection
    n_seeds=60,
    seed_center_weight=0.25,    # mild preference for top-center, not a hard force
    seed_center_sigma_frac=0.25,
    seed_tensile_weight=0.35,   # prefer tensile-dominated starts
    # smoothing
    smooth_passes=10,
    smooth_lambda=0.28,
    eps=1e-12,
    **_unused,
):
    """
    Global shortest path through resistance field, but with physics:
      cost ∝ (1/U) * (rim_pen) * (1 + dir_weight * misalignment^2)
    """
    R = float(R)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    inside, startM, goalM = _make_platen_masks(
        X,
        Y,
        R,
        beta_deg=float(platen_beta_deg),
        extra_deg=float(platen_snap_extra_deg),
        r_frac=0.94,
    )

    U, psi_pref, tfrac = _compute_utility_and_dir_grids(
        R=float(R),
        alpha_const=float(alpha_const),
        fit=fit,
        X=X,
        Y=Y,
        Teff_img=Teff_img,
        Coh_img=Coh_img,
        Phi_img=Phi_img,
        mixed_band=float(mixed_band),
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=strength_model,
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi,
        eps=float(eps),
    )

    valid = inside & np.isfinite(U) & np.isfinite(psi_pref)
    start = valid & startM
    goal = valid & goalM

    if (np.count_nonzero(start) < 3) or (np.count_nonzero(goal) < 3):
        xs = np.array([0.0, 0.0], float)
        ys = np.array([0.97 * R, -0.97 * R], float)
        xs, ys = enforce_platen_endpoints(
            xs,
            ys,
            R,
            beta_deg=platen_beta_deg,
            snap_extra_deg=platen_snap_extra_deg,
            snap_to_platen_center=snap_to_platen_center,
        )
        return {"xs": xs, "ys": ys, "psi0": -np.pi / 2, "model": "graph_fallback"}

    ny, nx = U.shape
    dx = float(X[0, 1] - X[0, 0]) if nx > 1 else (2 * R / max(nx, 1))
    dy = float(Y[1, 0] - Y[0, 0]) if ny > 1 else (2 * R / max(ny, 1))

    rnorm = np.hypot(X, Y) / (R + 1e-30)
    rim_pen = 1.0 + float(rim_weight) * (rnorm ** float(rim_power))

    Uc = np.maximum(U, float(u_floor))
    base_res = (1.0 / (Uc + 1e-30)) * rim_pen
    base_res[~valid] = np.inf

    # --- pick start seeds (physics-based) ---
    start_flat = np.flatnonzero(start.ravel())
    if start_flat.size == 0:
        return {
            "xs": np.array([]),
            "ys": np.array([]),
            "psi0": 0.0,
            "model": "graph_no_start",
        }

    Uf = U.ravel()[start_flat]
    Xf = X.ravel()[start_flat]
    Tf = tfrac.ravel()[start_flat]

    sig = float(seed_center_sigma_frac) * R + 1e-30
    score = Uf + float(seed_tensile_weight) * Tf - float(seed_center_weight) * (
        Xf / sig
    ) ** 2
    score = np.where(np.isfinite(score), score, -np.inf)
    order = start_flat[np.argsort(-score)]
    seed_idx = order[: min(int(n_seeds), order.size)]

    # --- Dijkstra ---
    N = ny * nx
    dist = np.full(N, np.inf, dtype=float)
    prev = np.full(N, -1, dtype=int)
    visited = np.zeros(N, dtype=bool)

    goal_idx = np.flatnonzero(goal.ravel())
    goal_set = set(goal_idx.tolist())

    heap = []
    for si in seed_idx:
        dist[int(si)] = 0.0
        heapq.heappush(heap, (0.0, int(si)))

    neigh = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    def lin(i, j):
        return i * nx + j

    goal_hit = -1
    while heap:
        dcur, uidx = heapq.heappop(heap)
        if visited[uidx]:
            continue
        visited[uidx] = True

        if uidx in goal_set:
            goal_hit = uidx
            break
        if dcur > dist[uidx]:
            continue

        i = uidx // nx
        j = uidx - i * nx

        psi0 = float(psi_pref[i, j])

        for di, dj in neigh:
            ii = i + di
            jj = j + dj
            if (ii < 0) or (ii >= ny) or (jj < 0) or (jj >= nx):
                continue
            if not valid[ii, jj]:
                continue

            vidx = lin(ii, jj)

            # move direction angle (line)
            mv = float(np.arctan2(di * dy, dj * dx))
            # misalignment in [0, pi/2]
            delta = float(_line_angle_diff(mv, psi0))
            # directional penalty (physics): follow psi_pref but not forced
            dir_pen = 1.0 + float(dir_weight) * (delta / (np.pi / 2.0)) ** 2

            step = float(np.hypot(dj * dx, di * dy))
            wcost = 0.5 * (base_res[i, j] + base_res[ii, jj]) * step * dir_pen
            nd = dcur + wcost

            if nd < dist[vidx]:
                dist[vidx] = nd
                prev[vidx] = uidx
                heapq.heappush(heap, (nd, vidx))

    if goal_hit < 0:
        cand = np.flatnonzero((valid & (Y < 0)).ravel())
        goal_hit = int(cand[np.nanargmin(dist[cand])]) if cand.size else int(
            seed_idx[0]
        )

    # backtrack
    path = []
    cur = int(goal_hit)
    for _ in range(N):
        path.append(cur)
        cur = int(prev[cur])
        if cur < 0:
            break
    path = path[::-1]

    xs = np.array([X.flat[k] for k in path], dtype=float)
    ys = np.array([Y.flat[k] for k in path], dtype=float)

    # smooth (keep endpoints fixed)
    if len(xs) >= 5 and smooth_passes > 0:
        for _ in range(int(smooth_passes)):
            xnew = xs.copy()
            ynew = ys.copy()
            xnew[1:-1] = xs[1:-1] + float(smooth_lambda) * (
                0.5 * (xs[:-2] + xs[2:]) - xs[1:-1]
            )
            ynew[1:-1] = ys[1:-1] + float(smooth_lambda) * (
                0.5 * (ys[:-2] + ys[2:]) - ys[1:-1]
            )

            rr = np.hypot(xnew, ynew)
            mask = rr > 0.999 * R
            xnew[mask] *= (0.999 * R) / (rr[mask] + 1e-30)
            ynew[mask] *= (0.999 * R) / (rr[mask] + 1e-30)
            xs, ys = xnew, ynew

    xs, ys = enforce_platen_endpoints(
        xs,
        ys,
        R,
        beta_deg=float(platen_beta_deg),
        snap_extra_deg=float(platen_snap_extra_deg),
        snap_to_platen_center=bool(snap_to_platen_center),
    )

    xs, ys = straighten_endcaps_parallel_to_load(
        xs,
        ys,
        R,
        beta_deg=float(platen_beta_deg),
        extra_deg=6.0,
        ncap=14,
        power=2.0,
    )

    psi0_out = float(
        np.arctan2(ys[min(1, len(ys) - 1)] - ys[0], xs[min(1, len(xs) - 1)] - xs[0])
    )
    return {"xs": xs, "ys": ys, "psi0": psi0_out, "model": "physics_graph_directional"}


# =============================================================================
# Local physics platen-to-platen growth (often more realistic)
# =============================================================================

def _clamp_outward(tip_x, tip_y, psi, eps_ang=1e-4):
    """Limit direction so it does not point too far inward (legacy center-start)."""
    r2 = float(tip_x) * float(tip_x) + float(tip_y) * float(tip_y)
    if r2 < 1e-16:
        return _wrap_pi(psi)
    theta_r = np.arctan2(tip_y, tip_x)
    d = angle_diff_periodic(psi, theta_r)
    lim = (np.pi / 2.0) - float(eps_ang)
    if abs(d) > lim:
        psi = _wrap_pi(theta_r + np.sign(d) * lim)
    return _wrap_pi(psi)


def _seg_intersect(p1, p2, q1, q2, eps=1e-12):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def onseg(a, b, c):
        return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps) and (
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    if abs(o1) <= eps and onseg(p1, p2, q1):
        return True
    if abs(o2) <= eps and onseg(p1, p2, q2):
        return True
    if abs(o3) <= eps and onseg(q1, q2, p1):
        return True
    if abs(o4) <= eps and onseg(q1, q2, p2):
        return True
    return False


def _on_platen_arc(x, y, beta, top=True):
    th = np.arctan2(y, x)
    center = (np.pi / 2.0) if top else (-np.pi / 2.0)
    return abs(float(angle_wrap_pi(th - center))) <= float(beta)


def _pick_start_on_top_platen(
    R,
    alpha_const,
    fit,
    X,
    Y,
    Teff_img,
    Coh_img,
    Phi_img,
    beta_deg,
    r_frac=0.97,
    n=81,
    extra_deg=6.0,
    **failure_kwargs,
):
    sampler = UniformGridSampler(X, Y)
    beta = np.deg2rad(float(beta_deg) + float(extra_deg))
    ths = np.linspace(np.pi / 2.0 - beta, np.pi / 2.0 + beta, int(n))
    rr = float(r_frac) * float(R)

    best = (-np.inf, 0.0, rr)
    for th in ths:
        x = rr * np.cos(th)
        y = rr * np.sin(th)
        Rt, Rs, _, _ = ratios_and_dirs_at_point(
            x,
            y,
            R,
            alpha_const,
            fit,
            sampler,
            Teff_img,
            Coh_img,
            Phi_img,
            **failure_kwargs,
        )
        if not (np.isfinite(Rt) and np.isfinite(Rs)):
            continue
        # physics: start where failure is easiest, slightly prefer tensile
        tf = Rt / (Rt + Rs + 1e-12)
        u = max(Rt, Rs) + 0.20 * tf
        if np.isfinite(u) and (u > best[0]):
            best = (float(u), float(x), float(y))
    return best[1], best[2]


def crack_path_platen_to_platen_physics(
    R,
    alpha_const,
    fit,
    X,
    Y,
    Teff_img,
    Coh_img,
    Phi_img,
    ds_frac=0.005,
    max_steps=6500,
    mixed_band=0.45,
    relax=0.75,
    n_theta_mc=361,
    fail_margin=0.0,
    max_turn_deg=18.0,
    hysteresis=1.25,
    target_smooth=0.55,
    crack_path_util_min=0.98,
    seg_check_n=7,
    ds_retry_factors=(1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.22),
    cand_n=61,
    # goal attraction helps connect to bottom platen but does NOT force vertical
    goal_weight=0.45,
    goal_power=1.25,
    platen_beta_deg=10.0,
    platen_snap_extra_deg=8.0,
    snap_to_platen_center=False,
    self_intersection_stop=True,
    eps=1e-12,
    **failure_kwargs,
):
    sampler = UniformGridSampler(X, Y)
    failure_kwargs = dict(failure_kwargs)
    failure_kwargs.setdefault("n_theta_mc", int(n_theta_mc))

    # start on top platen
    sx, sy = _pick_start_on_top_platen(
        R,
        alpha_const,
        fit,
        X,
        Y,
        Teff_img,
        Coh_img,
        Phi_img,
        beta_deg=float(platen_beta_deg),
        r_frac=0.97,
        n=81,
        extra_deg=float(platen_snap_extra_deg),
        **failure_kwargs,
    )

    goal_pt = np.array([0.0, -0.97 * float(R)], dtype=float)

    ds0 = float(ds_frac) * float(R)
    m = float(fail_margin)

    # initial direction toward goal
    psi = float(np.arctan2(goal_pt[1] - sy, goal_pt[0] - sx))
    psi0 = float(psi)

    xs = [float(sx)]
    ys = [float(sy)]

    cap = np.deg2rad(float(max_turn_deg))
    hys = float(max(hysteresis, 1.0))
    seg_check_n = int(max(seg_check_n, 3))
    seg_t = np.linspace(0.15, 1.0, seg_check_n)
    cand_n = int(max(cand_n, 9))

    psi_target_prev = psi
    drive_mode_prev = None
    beta_arc = np.deg2rad(float(platen_beta_deg) + float(platen_snap_extra_deg))

    for step_i in range(int(max_steps)):
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
        r = float(np.hypot(tip_x, tip_y))
        rn = r / (float(R) + float(eps))

        # stop if reached bottom platen zone
        if (rn >= 0.92) and (tip_y < 0.0) and _on_platen_arc(
            tip_x, tip_y, beta_arc, top=False
        ):
            break

        Rt, Rs, psi_t, psi_s = ratios_and_dirs_at_point(
            tip_x,
            tip_y,
            R,
            alpha_const,
            fit,
            sampler,
            Teff_img,
            Coh_img,
            Phi_img,
            **failure_kwargs,
        )
        if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
            break

        if (Rt < 1.0 + m) and (Rs < 1.0 + m):
            if rn >= 0.25:
                break

        psi_t = _align_line_direction(psi_t, psi)
        psi_s = _align_line_direction(psi_s, psi)

        tfrac = float(Rt / (Rt + Rs + eps))
        rel = abs(Rt - Rs) / max(max(Rt, Rs), eps)

        if rel <= float(mixed_band):
            drive_mode = "mixed"
        else:
            drive_mode = "tensile" if (Rt >= Rs) else "shear"

        # hysteresis
        if drive_mode_prev is not None and drive_mode != drive_mode_prev and drive_mode != "mixed":
            if drive_mode_prev == "tensile" and not (Rs > Rt * hys):
                drive_mode = drive_mode_prev
            if drive_mode_prev == "shear" and not (Rt > Rs * hys):
                drive_mode = drive_mode_prev
        drive_mode_prev = drive_mode

        # physics target direction (line mix)
        if drive_mode == "tensile":
            psi_target = psi_t
        elif drive_mode == "shear":
            psi_target = _angle_mean_line_scalar(psi_s, psi_t, 0.25 + 0.25 * tfrac)
        else:
            w = float(np.clip(0.35 + 0.55 * tfrac, 0.20, 0.95))
            psi_target = _angle_mean_line_scalar(psi_s, psi_t, w)

        # goal attraction (soft)
        psi_goal = float(np.arctan2(goal_pt[1] - tip_y, goal_pt[0] - tip_x))
        dgoal = float(np.hypot(goal_pt[0] - tip_x, goal_pt[1] - tip_y))
        gsoft = float(
            np.clip(1.0 - dgoal / (2.0 * float(R) + eps), 0.0, 1.0)
        )
        wg = float(goal_weight) * (gsoft ** float(goal_power))
        psi_target = _angle_mean_line_scalar(psi_target, psi_goal, wg)

        # smooth target
        psi_target = _angle_mean_line_scalar(
            psi_target_prev, psi_target, float(target_smooth)
        )
        psi_target_prev = psi_target

        # relax turn
        dpsi0 = float(np.clip(angle_diff_periodic(psi_target, psi), -cap, cap))
        psi_center = _wrap_pi(psi + float(relax) * dpsi0)

        # candidate set around current direction
        cand = psi + np.linspace(-cap, cap, cand_n)
        cand = np.array([_wrap_pi(ci) for ci in cand], dtype=float)

        best = None
        for f in ds_retry_factors:
            ds = ds0 * float(f)
            best = None

            for pc in cand:
                pc = _align_line_direction(pc, psi)

                nx = tip_x + ds * np.cos(pc)
                ny = tip_y + ds * np.sin(pc)
                if nx * nx + ny * ny >= (0.999 * float(R)) ** 2:
                    continue

                # segment utility check
                utils = []
                ok = True
                for tt in seg_t:
                    xi = tip_x + (tt * ds) * np.cos(pc)
                    yi = tip_y + (tt * ds) * np.sin(pc)
                    if xi * xi + yi * yi >= (0.999 * float(R)) ** 2:
                        ok = False
                        break
                    u = util_at_point(
                        xi,
                        yi,
                        R,
                        alpha_const,
                        fit,
                        sampler,
                        Teff_img,
                        Coh_img,
                        Phi_img,
                        **failure_kwargs,
                    )
                    if (not np.isfinite(u)) or (u < float(crack_path_util_min)):
                        ok = False
                        break
                    utils.append(u)

                if not ok:
                    continue

                u_avg = float(np.mean(utils))
                dturn = abs(angle_diff_periodic(pc, psi))
                dtarget = abs(angle_diff_periodic(pc, psi_center))

                # physics scoring: prefer high utility but allow curvature
                score = (
                    u_avg
                    - 0.12 * (dturn / max(cap, 1e-9)) ** 2
                    - 0.10 * (dtarget / max(cap, 1e-9)) ** 2
                )

                if (best is None) or (score > best[0]):
                    best = (score, nx, ny, pc)

            if best is not None:
                break

        if best is None:
            break

        _, nx, ny, psi_new = best

        if self_intersection_stop and (len(xs) > 25) and (step_i % 2 == 0):
            p1 = (xs[-1], ys[-1])
            p2 = (nx, ny)
            for j in range(1, len(xs) - 10):
                q1 = (xs[j - 1], ys[j - 1])
                q2 = (xs[j], ys[j])
                if _seg_intersect(p1, p2, q1, q2):
                    best = None
                    break
            if best is None:
                break

        xs.append(float(nx))
        ys.append(float(ny))
        psi = float(psi_new)

    Xs = np.array(xs, float)
    Ys = np.array(ys, float)

    Xs, Ys = enforce_platen_endpoints(
        Xs,
        Ys,
        R=float(R),
        beta_deg=float(platen_beta_deg),
        snap_extra_deg=float(platen_snap_extra_deg),
        snap_to_platen_center=bool(snap_to_platen_center),
    )

    Xs, Ys = straighten_endcaps_parallel_to_load(
        Xs,
        Ys,
        R=float(R),
        beta_deg=float(platen_beta_deg),
        extra_deg=6.0,
        ncap=14,
        power=2.0,
    )

    return {
        "xs": Xs,
        "ys": Ys,
        "psi0": float(psi0),
        "model": "physics_platen_to_platen_local",
    }


# =============================================================================
# Dispatcher (main entry)
# =============================================================================

def pick_initial_psi0_by_first(
    R,
    alpha_const,
    fit,
    sampler,
    Teff_img,
    Coh_img,
    Phi_img,
    n_dir=181,
    eps=1e-12,
    **failure_kwargs,
):
    r0 = 0.01 * float(R)
    psi_grid = np.linspace(0.0, np.pi, int(n_dir), endpoint=True)

    best = (-np.inf, np.pi / 2.0)
    for psi in psi_grid:
        x = r0 * np.cos(psi)
        y = r0 * np.sin(psi)
        u = util_at_point(
            x,
            y,
            R,
            alpha_const,
            fit,
            sampler,
            Teff_img,
            Coh_img,
            Phi_img,
            **failure_kwargs,
        )
        if not np.isfinite(u):
            continue
        if u > best[0]:
            best = (float(u), float(psi))
    return float(best[1])


def crack_path_single_mirror_physics(
    R,
    alpha_const,
    fit,
    X,
    Y,
    Teff_img,
    Coh_img,
    Phi_img,
    # modes:
    #   "platen_to_platen"  -> local physics growth (recommended for realistic curved cracks)
    #   "graph_platen"      -> global graph w/ physics directional alignment
    start_mode="platen_to_platen",
    platen_beta_deg=10.0,
    platen_snap_extra_deg=10.0,
    snap_to_platen_center=False,
    mixed_band=0.45,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    **kwargs,
):
    mode = str(start_mode).lower().strip()

    if mode in (
        "graph",
        "graph_platen",
        "platen_graph",
        "platen_to_platen_graph",
    ):
        return crack_path_platen_to_platen_graph(
            R=float(R),
            alpha_const=float(alpha_const),
            fit=fit,
            X=X,
            Y=Y,
            Teff_img=Teff_img,
            Coh_img=Coh_img,
            Phi_img=Phi_img,
            platen_beta_deg=float(platen_beta_deg),
            platen_snap_extra_deg=float(platen_snap_extra_deg),
            snap_to_platen_center=bool(snap_to_platen_center),
            mixed_band=float(mixed_band),
            n_theta_mc=int(n_theta_mc),
            mc_compression_only=bool(mc_compression_only),
            mc_sigma_comp_min=float(mc_sigma_comp_min),
            strength_model=strength_model,
            weak_T_ratio=float(weak_T_ratio),
            weak_C_ratio=float(weak_C_ratio),
            weak_phi=weak_phi,
            **kwargs,
        )

    # default: local physics growth
    return crack_path_platen_to_platen_physics(
        R=float(R),
        alpha_const=float(alpha_const),
        fit=fit,
        X=X,
        Y=Y,
        Teff_img=Teff_img,
        Coh_img=Coh_img,
        Phi_img=Phi_img,
        mixed_band=float(mixed_band),
        n_theta_mc=int(n_theta_mc),
        platen_beta_deg=float(platen_beta_deg),
        platen_snap_extra_deg=float(platen_snap_extra_deg),
        snap_to_platen_center=bool(snap_to_platen_center),
        strength_model=strength_model,
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi,
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        **kwargs,
    )


# --- Optional DDM path (extra; unchanged logic) ---

def max_hoop_kink_angle(KI, KII, eps=1e-18):
    KI = float(KI)
    KII = float(KII)
    if abs(KII) < eps:
        return 0.0
    root = np.sqrt(KI * KI + 8.0 * KII * KII)
    return 2.0 * np.arctan2(2.0 * KII, KI + root)


def pick_initial_psi0_by_ddm(
    R,
    alpha_const,
    fit,
    E1,
    E2,
    nu12,
    G12,
    a0_frac=0.008,
    n_dir=181,
    enforce_outward=True,
    ddm_sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    **ddm_kwargs,
):
    a0 = float(a0_frac) * float(R)
    psi_grid = np.linspace(0.0, np.pi, int(n_dir), endpoint=True)

    best = (-np.inf, np.pi / 2.0)
    for psi in psi_grid:
        xs = np.array([0.0, a0 * np.cos(psi)], dtype=float)
        ys = np.array([0.0, a0 * np.sin(psi)], dtype=float)

        psi_use = float(psi)
        if enforce_outward:
            psi_use = _clamp_outward(xs[-1], ys[-1], psi_use)

        sif_res = _try_call_sif_two_tips(
            sif_two_tips_from_crack,
            xs,
            ys,
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=fit,
            eval_stress_field_material=eval_stress_field_material,
            E1=float(E1),
            E2=float(E2),
            nu12=float(nu12),
            G12=float(G12),
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
            **ddm_kwargs,
        )
        KI0, KII0, KI1, KII1 = _extract_tip_sifs(sif_res)
        Keq = float(np.sqrt(KI1 * KI1 + KII1 * KII1))
        if np.isfinite(Keq) and (Keq > best[0]):
            best = (Keq, psi_use)

    return float(best[1])


def crack_path_single_mirror_ddm(
    R,
    alpha_const,
    fit,
    E1,
    E2,
    nu12,
    G12,
    ds_frac=0.005,
    a0_frac=0.008,
    max_steps=2500,
    max_turn_deg=12.0,
    relax=0.85,
    enforce_outward=True,
    self_intersection_stop=True,
    kmin_frac=0.02,
    ddm_sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    platen_beta_deg=10.0,
    platen_endcap_n=14,
    platen_endcap_power=2.0,
    platen_endcap_extra_deg=6.0,
    **ddm_kwargs,
):
    psi0 = pick_initial_psi0_by_ddm(
        R,
        alpha_const,
        fit,
        E1,
        E2,
        nu12,
        G12,
        a0_frac=a0_frac,
        n_dir=181,
        enforce_outward=enforce_outward,
        ddm_sample_rs=ddm_sample_rs,
        cod_offset=cod_offset,
        **ddm_kwargs,
    )

    ds0 = float(ds_frac) * float(R)
    a0 = float(a0_frac) * float(R)
    cap = np.deg2rad(float(max_turn_deg))

    xs = [0.0, a0 * np.cos(psi0)]
    ys = [0.0, a0 * np.sin(psi0)]
    psi = float(psi0)

    KI_hist = []
    KII_hist = []
    Keq_hist = []
    Kref = None

    for _ in range(int(max_steps)):
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
        if tip_x * tip_x + tip_y * tip_y >= (0.999 * float(R)) ** 2:
            break

        sif_res = _try_call_sif_two_tips(
            sif_two_tips_from_crack,
            np.array(xs, float),
            np.array(ys, float),
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=fit,
            eval_stress_field_material=eval_stress_field_material,
            E1=float(E1),
            E2=float(E2),
            nu12=float(nu12),
            G12=float(G12),
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
            **ddm_kwargs,
        )
        KI0, KII0, KI, KII = _extract_tip_sifs(sif_res)
        Keq = float(np.sqrt(KI * KI + KII * KII))

        KI_hist.append(float(KI))
        KII_hist.append(float(KII))
        Keq_hist.append(float(Keq))

        if not np.isfinite(Keq):
            break
        if Kref is None:
            Kref = max(Keq, 1e-30)
        if Keq < float(kmin_frac) * float(Kref):
            break

        dtheta = float(max_hoop_kink_angle(KI, KII))
        dtheta = float(np.clip(dtheta, -cap, cap))

        psi_target = _wrap_pi(psi + dtheta)
        if enforce_outward:
            psi_target = _clamp_outward(tip_x, tip_y, psi_target)

        dpsi = angle_diff_periodic(psi_target, psi)
        dpsi = float(np.clip(dpsi, -cap, cap))

        psi_new = _wrap_pi(psi + float(relax) * dpsi)
        if enforce_outward:
            psi_new = _clamp_outward(tip_x, tip_y, psi_new)

        nx = tip_x + ds0 * np.cos(psi_new)
        ny = tip_y + ds0 * np.sin(psi_new)

        if nx * nx + ny * ny >= (0.999 * float(R)) ** 2:
            break

        if self_intersection_stop and (len(xs) > 25):
            p1, p2 = (xs[-1], ys[-1]), (nx, ny)
            for j in range(1, len(xs) - 10):
                if _seg_intersect(p1, p2, (xs[j - 1], ys[j - 1]), (xs[j], ys[j])):
                    nx = xs[-1]
                    ny = ys[-1]
                    break

        xs.append(float(nx))
        ys.append(float(ny))
        psi = float(psi_new)

    xu = np.array(xs, float)
    yu = np.array(ys, float)
    Xs = np.concatenate([-xu[::-1], xu[1:]])
    Ys = np.concatenate([-yu[::-1], yu[1:]])

    Xs, Ys = straighten_endcaps_parallel_to_load(
        Xs,
        Ys,
        R=float(R),
        beta_deg=float(platen_beta_deg),
        extra_deg=float(platen_endcap_extra_deg),
        ncap=int(platen_endcap_n),
        power=float(platen_endcap_power),
    )

    return {
        "xs": Xs,
        "ys": Ys,
        "psi0": float(psi0),
        "KI_tip": np.array(KI_hist, float),
        "KII_tip": np.array(KII_hist, float),
        "Keq_tip": np.array(Keq_hist, float),
        "model": "ddm",
    }


# -----------------------------
# Compute G from KI,KII
# -----------------------------

def G_from_K_isotropic_equiv(KI, KII, Eprime=1.0):
    """
    Isotropic-equivalent energy release rate:
        G = (KI^2 + KII^2)/E'
    This is NOT anisotropic, but is a useful baseline.
    """
    KI = float(KI)
    KII = float(KII)
    return float((KI * KI + KII * KII) / (float(Eprime) + 1e-30))


def Eprime_from_orthotropic_plane_stress(E1, E2, nu12, G12, alpha_const=0.0):
    """
    A pragmatic isotropic-equivalent E' for scaling G when you don't yet have
    the full orthotropic H-matrix. For plane stress, E' ~ E.
    Here we use geometric mean as a stable choice: sqrt(E1*E2).
    """
    E1 = float(E1)
    E2 = float(E2)
    return float(np.sqrt(max(E1, 1e-30) * max(E2, 1e-30)))


# -----------------------------
# Robust SIF extraction helper
# -----------------------------

def _extract_tip_sifs(sif_res):
    if sif_res is None:
        return np.nan, np.nan, np.nan, np.nan
    if isinstance(sif_res, (tuple, list, np.ndarray)) and len(sif_res) >= 4:
        KI0, KII0, KI1, KII1 = (
            sif_res[0],
            sif_res[1],
            sif_res[2],
            sif_res[3],
        )
        return float(KI0), float(KII0), float(KI1), float(KII1)
    if isinstance(sif_res, dict):
        for keys in [
            ("KI0", "KII0", "KI1", "KII1"),
            ("KI_tip0", "KII_tip0", "KI_tip1", "KII_tip1"),
            ("KI_left", "KII_left", "KI_right", "KII_right"),
        ]:
            if all(k in sif_res for k in keys):
                KI0, KII0, KI1, KII1 = (
                    sif_res[keys[0]],
                    sif_res[keys[1]],
                    sif_res[keys[2]],
                    sif_res[keys[3]],
                )
                return float(KI0), float(KII0), float(KI1), float(KII1)
    raise TypeError(f"Unsupported sif result format: {type(sif_res)}")


# =============================================================================
# DDM crack growth by maximizing  G(theta) - Gc(theta)
# =============================================================================

def crack_path_single_mirror_ddm_energy(
    R,
    alpha_const,
    fit,
    E1,
    E2,
    nu12,
    G12,
    ds_frac=0.005,
    a0_frac=0.008,
    max_steps=2500,
    relax=0.85,
    enforce_outward=True,
    self_intersection_stop=True,
    # energy-based kink search
    kink_scan_deg=20.0,
    kink_n=61,
    objective="diff",  # "diff" -> G - Gc ; "ratio" -> G / Gc
    # toughness parameters
    use_weak_plane_toughness=True,
    Gc_matrix=1.0,            # relative ok, cancels if objective="ratio"
    weak_reduction=0.35,      # lower toughness along weak plane
    eta_deg=10.0,             # angular width of weak plane valley
    alpha_wp_global=None,     # weak-plane line direction in global; default uses alpha_const
    # stopping
    gmin_frac=0.02,           # stop if G falls below this fraction of initial G
    ddm_sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    # fallback to physics-based path
    physics_fallback_fn=None,     # e.g. crack_path_single_mirror_physics
    physics_fallback_kwargs=None, # kwargs for fallback
    **ddm_kwargs,
):
    """
    Energy-based DDM growth:
      - scan candidate kink angles δ
      - compute KI,KII for trial-kinked crack
      - compute G(δ) (isotropic-equiv scaling by default)
      - compute Gc(θ_tip+δ)
      - choose δ maximizing (G-Gc) or (G/Gc)

    Notes:
      - G here uses isotropic-equivalent E' scaling unless you replace it with a true anisotropic relation.
      - This is still a major upgrade vs max hoop stress because toughness enters directionally as Gc(θ).
    """
    R = float(R)
    ds = float(ds_frac) * R
    a0 = float(a0_frac) * R
    kink_scan = np.deg2rad(float(kink_scan_deg))
    kink_n = int(max(9, kink_n))
    obj = str(objective).lower().strip()

    # choose weak-plane direction
    if alpha_wp_global is None:
        alpha_wp_global = float(alpha_const)  # line direction of weak plane in global (common choice)

    # isotropic-equivalent scaling (stable baseline)
    Eprime = Eprime_from_orthotropic_plane_stress(
        E1, E2, nu12, G12, alpha_const=alpha_const
    )

    # initial straight short crack (mirror handled later)
    psi0 = float(np.pi / 2.0)  # default; you can also keep your pick_initial_psi0_by_ddm if you want
    xs = [0.0, a0 * np.cos(psi0)]
    ys = [0.0, a0 * np.sin(psi0)]
    psi = float(psi0)

    Gref = None

    for step_i in range(int(max_steps)):
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
        if tip_x * tip_x + tip_y * tip_y >= (0.999 * R) ** 2:
            break

        # scan kinks around current direction
        dscan = np.linspace(-kink_scan, kink_scan, kink_n)
        best = None

        for dth in dscan:
            # trial direction with relaxation
            psi_trial = float(psi + float(relax) * dth)

            nx = tip_x + ds * np.cos(psi_trial)
            ny = tip_y + ds * np.sin(psi_trial)
            if nx * nx + ny * ny >= (0.999 * R) ** 2:
                continue

            # build trial crack geometry for SIF call
            xtrial = np.array(xs + [nx], dtype=float)
            ytrial = np.array(ys + [ny], dtype=float)

            sif_res = _try_call_sif_two_tips(
                sif_two_tips_from_crack,
                xtrial,
                ytrial,
                R=float(R),
                alpha_const=float(alpha_const),
                airy_fit=fit,
                eval_stress_field_material=eval_stress_field_material,
                E1=float(E1),
                E2=float(E2),
                nu12=float(nu12),
                G12=float(G12),
                sample_rs=ddm_sample_rs,
                cod_offset=float(cod_offset),
                **ddm_kwargs,
            )
            KI0, KII0, KI, KII = _extract_tip_sifs(sif_res)
            if not (np.isfinite(KI) and np.isfinite(KII)):
                continue

            # driving force
            G = G_from_K_isotropic_equiv(KI, KII, Eprime=Eprime)
            if not np.isfinite(G):
                continue

            # set reference (for termination)
            if Gref is None:
                Gref = max(G, 1e-30)

            # direction-dependent toughness
            theta_line = float(psi_trial)  # crack line direction in global
            if use_weak_plane_toughness:
                Gc = Gc_theta_weak_plane(
                    theta_line=theta_line,
                    alpha_wp=float(alpha_wp_global),
                    Gc_matrix=float(Gc_matrix),
                    weak_reduction=float(weak_reduction),
                    eta_deg=float(eta_deg),
                )
            else:
                Gc = float(Gc_matrix)

            # objective
            if obj == "ratio":
                score = G / (Gc + 1e-30)
            else:
                score = G - Gc

            # intersection stop (optional)
            if self_intersection_stop and (len(xs) > 25) and (step_i % 2 == 0):
                p1 = (xs[-1], ys[-1])
                p2 = (nx, ny)
                bad = False
                for j in range(1, len(xs) - 10):
                    q1 = (xs[j - 1], ys[j - 1])
                    q2 = (xs[j], ys[j])
                    if _seg_intersect(p1, p2, q1, q2):
                        bad = True
                        break
                if bad:
                    continue

            if (best is None) or (score > best[0]):
                best = (score, nx, ny, psi_trial, G)

        # no valid kink -> fallback or stop
        if best is None:
            if physics_fallback_fn is not None:
                kw = {} if physics_fallback_kwargs is None else dict(physics_fallback_kwargs)
                return physics_fallback_fn(
                    R=float(R),
                    alpha_const=float(alpha_const),
                    fit=fit,
                    X=kw.pop("X"),
                    Y=kw.pop("Y"),
                    Teff_img=kw.pop("Teff_img"),
                    Coh_img=kw.pop("Coh_img"),
                    Phi_img=kw.pop("Phi_img"),
                    **kw,
                )
            break

        score, nx, ny, psi_new, Gbest = best

        # propagation condition (energy-based): require G >= Gc (diff objective) or ratio >= 1
        if obj == "ratio":
            if score < 1.0:
                break
        else:
            if score <= 0.0:
                break

        # stop if driving force collapses
        if (Gref is not None) and (Gbest < float(gmin_frac) * float(Gref)):
            break

        xs.append(float(nx))
        ys.append(float(ny))
        psi = float(psi_new)

    # mirror about origin to get "single mirror" disk crack
    xu = np.array(xs, float)
    yu = np.array(ys, float)
    Xs = np.concatenate([-xu[::-1], xu[1:]])
    Ys = np.concatenate([-yu[::-1], yu[1:]])

    return {
        "xs": Xs,
        "ys": Ys,
        "psi0": float(psi0),
        "model": "ddm_energy_G_minus_Gc"
        if obj != "ratio"
        else "ddm_energy_G_over_Gc",
    }



def H_matrix_from_stroh(
    E1, E2, nu12, G12,
    alpha_const=0.0,
    psi_tip_global=0.0,
    dtheta_deg=0.0,   # <- OPTIONAL, kept for backward compatibility
    **_unused
):
    """
    Build the 2×2 anisotropic energy matrix H such that:
        G = [KI, KII] @ H @ [KI, KII]^T

    Key points (energy-consistent):
    - H depends on orthotropic elastic constants and the crack-tip direction.
    - dtheta_deg is NOT required. If older code passes it, we accept it but
      do not need it for the physics.
    - Uses a stable orthotropic plane-stress stiffness and a conservative,
      physically-motivated construction:
        * rotate Q into the crack-local frame
        * compute local effective moduli (En, Et, Gnt)
        * map KI and KII into an energy form consistent with orthotropic scaling

    Notes:
    - If you already have a validated Stroh-based H from literature, replace
      the core section below with that exact closed form. This version is
      designed to be robust and avoids any required dtheta_deg plumbing.
    """

    # -----------------------------
    # 1) Orthotropic plane-stress Q (material axes)
    # -----------------------------
    E1 = float(E1); E2 = float(E2); nu12 = float(nu12); G12 = float(G12)
    nu21 = nu12 * E2 / max(E1, 1e-30)
    den = 1.0 - nu12 * nu21

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    Q = np.array([[Q11, Q12, 0.0],
                  [Q12, Q22, 0.0],
                  [0.0, 0.0, Q66]], dtype=float)

    # -----------------------------
    # 2) Rotate Q into crack-local frame
    #    material->global rotation alpha_const; crack direction psi_tip_global in global.
    #    So crack direction in material is psi_m = psi_tip_global - alpha_const
    # -----------------------------
    psi_m = float(psi_tip_global) - float(alpha_const)
    c = np.cos(psi_m); s = np.sin(psi_m)
    c2 = c*c; s2 = s*s; cs = c*s

    # Standard Voigt rotation operators (plane stress)
    T_sigma = np.array([
        [c2,  s2,  2*cs],
        [s2,  c2, -2*cs],
        [-cs, cs,  c2 - s2]
    ], dtype=float)

    T_eps = np.array([
        [c2,  s2,  cs],
        [s2,  c2, -cs],
        [-2*cs, 2*cs, c2 - s2]
    ], dtype=float)

    # Qbar = T_sigma^{-1} Q (T_eps^{-T})
    Qbar = np.linalg.solve(T_sigma, Q @ np.linalg.inv(T_eps).T)

    # -----------------------------
    # 3) Get local effective moduli from compliance in the crack-local frame
    # -----------------------------
    Sbar = np.linalg.inv(Qbar)
    S11, S22, S12, S16, S26, S66 = (
        float(Sbar[0, 0]),
        float(Sbar[1, 1]),
        float(Sbar[0, 1]),
        float(Sbar[0, 2]),
        float(Sbar[1, 2]),
        float(Sbar[2, 2]),
    )

    # Effective moduli in local frame (plane stress)
    # n = local-1, t = local-2 convention (you can swap if your KI/KII convention differs)
    En = 1.0 / max(S11, 1e-30)   # normal-direction effective modulus
    Et = 1.0 / max(S22, 1e-30)   # tangential-direction effective modulus
    Gnt = 1.0 / max(S66, 1e-30)  # in-plane shear modulus

    # -----------------------------
    # 4) Build H matrix (robust, symmetric, positive-definite)
    # -----------------------------
    # In isotropy: G = (KI^2 + KII^2)/E'  -> H = (1/E')*I
    # For orthotropy, we scale mode I and II by directional moduli.
    # A conservative, physically sensible choice:
    #   H11 ~ 1/En_eff
    #   H22 ~ 1/Gnt_eff
    # and allow mild coupling via S16/S26 if desired (often small; keep stable).
    #
    # If you want *no coupling* (simplest & very stable): set H12 = 0.
    #
    # Here we include a bounded coupling term built from S16,S26 but clipped so SPD remains.
    H11 = 1.0 / max(En, 1e-30)
    H22 = 1.0 / max(Gnt, 1e-30)

    # bounded coupling (optional); keep extremely mild to avoid non-SPD
    # Using nondimensional measure of extension-shear coupling:
    coup = 0.5 * (abs(S16) + abs(S26)) / max(np.sqrt(S11*S66) + np.sqrt(S22*S66), 1e-30)
    coup = float(np.clip(coup, 0.0, 0.15))  # hard cap for robustness

    # choose sign so coupling doesn't bias; symmetric magnitude only
    H12 = coup * np.sqrt(H11 * H22)

    H = np.array([[H11, H12],
                  [H12, H22]], dtype=float)

    # Ensure SPD numerically (tiny jitter if needed)
    # (Cholesky test)
    try:
        _ = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        # force diagonal-only if coupling made it indefinite (rare due to clipping)
        H = np.array([[H11, 0.0],
                      [0.0, H22]], dtype=float)

    return H






def G_from_K_orthotropic(
    KI, KII,
    *,
    E1, E2, nu12, G12,
    alpha_const=0.0,
    psi_tip_global=0.0,
    dtheta_deg=None,     # legacy, optional
    H=None,              # optional
    return_H=True,       # optional
    **_unused
):
    """
    Orthotropic energy release rate:
        G = [KI, KII] H [KI, KII]^T
    where H depends on orthotropic stiffness and crack direction.
    """

    KI = float(KI); KII = float(KII)

    if H is None:
        # call the function directly (NO "from crack_helpers import ...")
        H = H_matrix_from_stroh(
            E1, E2, nu12, G12,
            alpha_const=float(alpha_const),
            psi_tip_global=float(psi_tip_global),
            dtheta_deg=dtheta_deg,
        )

    H = np.asarray(H, dtype=float).reshape(2, 2)
    K = np.array([KI, KII], dtype=float)
    G = float(K @ H @ K)

    if return_H:
        return G, H
    return G
