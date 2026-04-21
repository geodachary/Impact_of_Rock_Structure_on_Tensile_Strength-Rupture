#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brazilian disk — Lekhnitskii Airy + PHYSICS failure + SINGLE crack path
v8.0: crack path fixed + improved (physics-guided)

Main improvements vs your v7.3:
  1) psi_pref is treated as a *line* orientation (θ ≡ θ+π), not an arrow direction.
     -> direction penalty uses line-angle difference
  2) Removed "force outward" on psi_pref field (was biasing Dijkstra into S-shapes)
  3) Added centerline preference using a clearance (distance-to-edge) field inside the failed band
  4) Added snap-back after smoothing so the drawn polyline stays on the allowed failed band
  5) weak_plane_phi_deg is applied consistently in the DISC failure map too

Notes:
- Still a heuristic crack surrogate (graph search on failure utilization field), not LEFM.
- Physics enters via the Airy stress field + your strength model and mode selection.

Default crack method remains: grid_phys
"""

import os, sys, argparse
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------
# Plot styling
# -------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.linewidth"] = 1.2

minorTick = {'which': 'minor', 'direction': 'out', 'length': 3, 'width': 1}
majorTick = {'which': 'major', 'direction': 'out', 'length': 5, 'width': 1.5}
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}

def close_box(ax, lw=1.2):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)

# =========================
# Geometry helpers
# =========================
def angle_diff_periodic(a, b):
    return ((a-b+np.pi)%(2*np.pi))-np.pi

def _wrap_pi(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi

def _wrap_pi_half(a):
    # wrap a *line* orientation to [-pi/2, pi/2)
    return ((a + np.pi/2) % np.pi) - np.pi/2

def angle_diff_line(a, b):
    """
    Smallest angular difference when direction is a LINE:
    b ≡ b + π.
    """
    da = abs(angle_diff_periodic(a, b))
    db = abs(angle_diff_periodic(a, b + np.pi))
    return min(da, db)

def _line_angle_mean(a, b, w):
    """
    Weighted mean of two *line* orientations using doubled angles.
    """
    a = float(a); b = float(b); w = float(w)
    z = (1.0 - w) * np.exp(2j * a) + w * np.exp(2j * b)
    if abs(z) < 1e-20:
        return _wrap_pi_half(a)
    return _wrap_pi_half(0.5*np.angle(z))

def create_disk_grid(D, n=101):
    R = D/2
    xs = np.linspace(-R, R, n)
    ys = np.linspace(-R, R, n)
    X, Y = np.meshgrid(xs, ys)
    M = (X**2 + Y**2) <= R**2
    return X, Y, M

def points_in_disk(D, n=101):
    X, Y, M = create_disk_grid(D, n)
    return X[M], Y[M], X, Y, M

# =========================
# Rotation helpers
# =========================
def rot_to_material(X, Y, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    x = c*X + s*Y
    y = -s*X + c*Y
    return x, y

def stress_material_to_global(sxx, syy, txy, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    Sxx = c*c*sxx + s*s*syy - 2*s*c*txy
    Syy = s*s*sxx + c*c*syy + 2*s*c*txy
    Txy = s*c*(sxx - syy) + (c*c - s*s)*txy
    return Sxx, Syy, Txy

def principal_from_components(sxx, syy, txy):
    th = 0.5*np.arctan2(2*txy, (sxx - syy))
    rad = np.sqrt(((sxx - syy)/2.0)**2 + txy**2)
    s1 = (sxx + syy)/2.0 + rad
    s3 = (sxx + syy)/2.0 - rad
    return s1, s3, th

# =========================
# Orthotropic compliance + Lekhnitskii roots
# =========================
def orthotropic_compliances_plane_stress(E1, E2, nu12, G12):
    s11 = 1.0/E1
    s22 = 1.0/E2
    s12 = -nu12/E1
    s66 = 1.0/G12
    return s11, s22, s12, s66

def lekh_roots_p(E1, E2, nu12, G12):
    s11, s22, s12, s66 = orthotropic_compliances_plane_stress(E1, E2, nu12, G12)
    a = float(s11)
    b = float(2*s12 + s66)
    c = float(s22)
    u = np.roots([a, b, c]).astype(np.complex128)

    p_candidates = []
    for ui in u:
        p = np.sqrt(ui)
        p_candidates.extend([p, -p])

    p_candidates = [p for p in p_candidates if np.isfinite(p.real) and np.isfinite(p.imag)]

    uniq = []
    for p in p_candidates:
        if all(abs(p-q) > 1e-10 for q in uniq):
            uniq.append(p)

    p_pos = [p for p in uniq if np.imag(p) > 1e-12]
    if len(p_pos) >= 2:
        p_pos = sorted(p_pos, key=lambda z: np.imag(z), reverse=True)
        return p_pos[0], p_pos[1]

    uniq = sorted(uniq, key=lambda z: (abs(np.imag(z)), abs(np.real(z))), reverse=True)
    if len(uniq) < 2:
        raise RuntimeError("Could not obtain two valid Lekhnitskii roots p1,p2.")
    return uniq[0], uniq[1]

# =========================
# Boundary traction model (two platen arcs)
# =========================
def smooth_arc_window(d, half_width, smooth):
    d = np.asarray(d)
    w = np.zeros_like(d, dtype=float)
    a = float(half_width)
    s = max(float(smooth), 0.0)

    if s <= 0:
        w[d <= a] = 1.0
        return w

    core = d <= (a - s)
    trans = (d > (a - s)) & (d < (a + s))
    w[core] = 1.0
    xi = (d[trans] - (a - s)) / (2*s)
    w[trans] = 0.5*(1 + np.cos(np.pi*xi))
    return w

def platen_tractions_theta(theta, beta, smooth, p0, mu=0.0):
    theta = np.asarray(theta)
    d_top = np.abs(angle_diff_periodic(theta, np.pi/2))
    d_bot = np.abs(angle_diff_periodic(theta, -np.pi/2))
    w_top = smooth_arc_window(d_top, beta, smooth)
    w_bot = smooth_arc_window(d_bot, beta, smooth)
    p = p0*(w_top + w_bot)  # MPa
    tr = -p
    if mu <= 0:
        return tr, np.zeros_like(tr)
    sign_top = -np.sign(angle_diff_periodic(theta, np.pi/2))
    sign_bot = np.sign(angle_diff_periodic(theta, -np.pi/2))
    tt = mu*p*(w_top*sign_top + w_bot*sign_bot)
    return tr, tt

def pressure_amplitude_from_load_tapered(P_N, t_m, R_m, beta, smooth, nint=6000):
    th = np.linspace(np.pi/2 - (beta+smooth), np.pi/2 + (beta+smooth), nint)
    d_top = np.abs(angle_diff_periodic(th, np.pi/2))
    w_top = smooth_arc_window(d_top, beta, smooth)
    I = np.trapezoid(w_top*np.sin(th), th)
    denom = 1e6 * t_m * R_m * max(I, 1e-12)
    return float(P_N / denom)  # MPa

# ==========================================================
# Airy evaluation: normalized z/R basis
# ==========================================================
def eval_boundary_tractions(theta_m, R, p1, p2, a1, a2):
    x = R*np.cos(theta_m)
    y = R*np.sin(theta_m)
    z1 = x + p1*y
    z2 = x + p2*y
    zh1 = z1 / R
    zh2 = z2 / R
    f1pp = np.zeros_like(z1, dtype=complex)
    f2pp = np.zeros_like(z2, dtype=complex)
    Mdeg = len(a1)-1
    invR2 = 1.0/(R*R)
    for m in range(2, Mdeg+1):
        f1pp += a1[m] * m*(m-1) * (zh1**(m-2)) * invR2
        f2pp += a2[m] * m*(m-1) * (zh2**(m-2)) * invR2
    sxx = np.real(p1**2*f1pp + p2**2*f2pp)
    syy = np.real(f1pp + f2pp)
    txy = -np.real(p1*f1pp + p2*f2pp)
    nx = np.cos(theta_m); ny = np.sin(theta_m)
    tx = sxx*nx + txy*ny
    ty = txy*nx + syy*ny
    tr = tx*nx + ty*ny
    tt = -tx*ny + ty*nx
    return tr, tt

def eval_stress_field_material(x, y, R, p1, p2, a1, a2):
    z1 = x + p1*y
    z2 = x + p2*y
    zh1 = z1 / R
    zh2 = z2 / R
    f1pp = np.zeros_like(z1, dtype=complex)
    f2pp = np.zeros_like(z2, dtype=complex)
    Mdeg = len(a1)-1
    invR2 = 1.0/(R*R)
    for m in range(2, Mdeg+1):
        f1pp += a1[m] * m*(m-1) * (zh1**(m-2)) * invR2
        f2pp += a2[m] * m*(m-1) * (zh2**(m-2)) * invR2
    sxx = np.real(p1**2*f1pp + p2**2*f2pp)
    syy = np.real(f1pp + f2pp)
    txy = -np.real(p1*f1pp + p2*f2pp)
    return sxx, syy, txy

# ==========================================================
# Arc oversampling for boundary collocation
# ==========================================================
def _build_theta_m_with_arc_oversample(alpha, beta, smooth, N_base=360, N_arc_each=900):
    th_base = np.linspace(-np.pi, np.pi, int(N_base), endpoint=False)
    span = float(beta + smooth)
    thg_top = np.linspace(np.pi/2 - span, np.pi/2 + span, int(N_arc_each), endpoint=True)
    thg_bot = np.linspace(-np.pi/2 - span, -np.pi/2 + span, int(N_arc_each), endpoint=True)
    thm_top = _wrap_pi(thg_top - alpha)
    thm_bot = _wrap_pi(thg_bot - alpha)
    th = np.concatenate([th_base, thm_top, thm_bot])
    th = np.unique(np.round(th, 14))
    th.sort()
    return th

# ==========================================================
# Robust fit (lstsq + degree-weighted Tikhonov)
# ==========================================================
def fit_orthotropic_airy_disk(
    E1, E2, nu12, G12,
    R, t, P,
    alpha,
    M=24,
    Nbd=480,
    beta_deg=10.0,
    smooth_deg=4.0,
    mu=0.0,
    lam=1e-10,
    w_arc=20.0,
    Nbd_arc_each=900,
    rcond=1e-12,
):
    p1, p2 = lekh_roots_p(E1, E2, nu12, G12)
    beta = np.deg2rad(beta_deg)
    smooth = np.deg2rad(smooth_deg)
    p0 = pressure_amplitude_from_load_tapered(P, t, R, beta, smooth)
    theta_m = _build_theta_m_with_arc_oversample(alpha, beta, smooth, N_base=Nbd, N_arc_each=Nbd_arc_each)
    theta_g = theta_m + alpha
    tr_tgt, tt_tgt = platen_tractions_theta(theta_g, beta, smooth, p0, mu=mu)

    d_top = np.abs(angle_diff_periodic(theta_g, np.pi/2))
    d_bot = np.abs(angle_diff_periodic(theta_g, -np.pi/2))
    on_arc = (d_top <= beta + smooth) | (d_bot <= beta + smooth)
    w = np.ones_like(theta_m, dtype=float)
    w[on_arc] *= float(w_arc)

    x = R*np.cos(theta_m)
    y = R*np.sin(theta_m)  # IMPORTANT
    z1 = x + p1*y
    z2 = x + p2*y
    zh1 = z1 / R
    zh2 = z2 / R
    invR2 = 1.0/(R*R)
    nx = np.cos(theta_m); ny = np.sin(theta_m)

    ncoef = M - 1
    nunk = 4*ncoef
    nb = len(theta_m)
    A = np.zeros((2*nb, nunk), dtype=float)
    b = np.zeros((2*nb,), dtype=float)
    b[:nb] = tr_tgt
    b[nb:] = tt_tgt

    def tr_tt_from_fpp(p, fpp):
        sxx = np.real((p**2)*fpp)
        syy = np.real(fpp)
        txy = -np.real(p*fpp)
        tx = sxx*nx + txy*ny
        ty = txy*nx + syy*ny
        tr = tx*nx + ty*ny
        tt = -tx*ny + ty*nx
        return tr, tt

    col = 0
    for m in range(2, M+1):
        t1 = (m*(m-1)) * (zh1**(m-2)) * invR2
        t2 = (m*(m-1)) * (zh2**(m-2)) * invR2
        tr1, tt1 = tr_tt_from_fpp(p1, t1)
        tr1i, tt1i = tr_tt_from_fpp(p1, 1j*t1)
        tr2, tt2 = tr_tt_from_fpp(p2, t2)
        tr2i, tt2i = tr_tt_from_fpp(p2, 1j*t2)
        A[:nb, col+0] = tr1;  A[nb:, col+0] = tt1
        A[:nb, col+1] = tr1i; A[nb:, col+1] = tt1i
        A[:nb, col+2] = tr2;  A[nb:, col+2] = tt2
        A[:nb, col+3] = tr2i; A[nb:, col+3] = tt2i
        col += 4

    Wsqrt = np.sqrt(np.concatenate([w, w]))
    Aw = A * Wsqrt[:, None]
    bw = b * Wsqrt

    if lam and lam > 0:
        weights = []
        for m in range(2, M+1):
            w_m = (m / M)**4
            weights.extend([w_m, w_m, w_m, w_m])
        reg = np.sqrt(float(lam)) * np.diag(weights)
        Aw_aug = np.vstack([Aw, reg])
        bw_aug = np.concatenate([bw, np.zeros(nunk)])
    else:
        Aw_aug = Aw
        bw_aug = bw

    c, *_ = np.linalg.lstsq(Aw_aug, bw_aug, rcond=rcond)
    a1 = np.zeros(M+1, dtype=complex)
    a2 = np.zeros(M+1, dtype=complex)
    col = 0
    for m in range(2, M+1):
        a1[m] = c[col+0] + 1j*c[col+1]
        a2[m] = c[col+2] + 1j*c[col+3]
        col += 4

    tr_fit, tt_fit = eval_boundary_tractions(theta_m, R, p1, p2, a1, a2)
    res_all = np.sqrt(np.mean((tr_fit-tr_tgt)**2 + (tt_fit-tt_tgt)**2))
    res_arc = np.sqrt(np.mean((tr_fit[on_arc]-tr_tgt[on_arc])**2 + (tt_fit[on_arc]-tt_tgt[on_arc])**2)) if np.any(on_arc) else float("nan")
    free = ~on_arc
    res_free = np.sqrt(np.mean((tr_fit[free]-tr_tgt[free])**2 + (tt_fit[free]-tt_tgt[free])**2)) if np.any(free) else float("nan")

    return dict(
        p1=p1, p2=p2, a1=a1, a2=a2, p0=p0, beta=beta, smooth=smooth,
        res_rms_all=float(res_all), res_rms_arc=float(res_arc), res_rms_free=float(res_free)
    )

def fit_orthotropic_airy_disk_auto(
    E1, E2, nu12, G12,
    R, t, P,
    alpha,
    beta_deg=10.0,
    smooth_deg=4.0,
    mu=0.0,
    M_list=(16, 20, 24, 28),
    lam_list=(1e-12, 1e-10, 1e-8, 1e-6),
    w_arc_list=(8.0, 12.0, 20.0),
    Nbd=480,
    Nbd_arc_each=1200,
):
    best = None
    best_score = np.inf
    for M in M_list:
        for lam in lam_list:
            for w_arc in w_arc_list:
                fit = fit_orthotropic_airy_disk(
                    E1, E2, nu12, G12,
                    R=R, t=t, P=P, alpha=alpha,
                    M=int(M), Nbd=int(Nbd),
                    beta_deg=float(beta_deg), smooth_deg=float(smooth_deg),
                    mu=float(mu), lam=float(lam), w_arc=float(w_arc),
                    Nbd_arc_each=int(Nbd_arc_each),
                )
                score = 0.45*fit["res_rms_all"] + 0.35*fit["res_rms_free"] + 0.20*fit["res_rms_arc"]
                if score < best_score:
                    best_score = score
                    best = fit
                    best["M_best"] = int(M)
                    best["lam_best"] = float(lam)
                    best["w_arc_best"] = float(w_arc)
                    best["score"] = float(score)
    return best

# =========================
# Failure: tensile + Mohr–Coulomb + weak plane option
# =========================
def plane_sigma_tau(sxx, syy, txy, beta):
    s_avg = 0.5*(sxx + syy)
    s_diff = 0.5*(sxx - syy)
    cb = np.cos(2.0*beta)
    sb = np.sin(2.0*beta)
    sigma_n = s_avg + s_diff*cb + txy*sb
    tau = -s_diff*sb + txy*cb
    return sigma_n, tau

def mc_scan_ratio(
    sxx, syy, txy,
    c, phi,
    n_theta=361,
    eps=1e-12,
    compression_only=True,
    sigma_comp_min=0.0
):
    beta = np.linspace(0.0, np.pi, int(n_theta), endpoint=False)[:, None]
    cb = np.cos(2*beta); sb = np.sin(2*beta)

    s_avg = 0.5*(sxx + syy)[None, :]
    s_diff = 0.5*(sxx - syy)[None, :]
    txyN = txy[None, :]

    sigma_n = s_avg + s_diff*cb + txyN*sb
    tau = -s_diff*sb + txyN*cb

    scmin = float(max(sigma_comp_min, 0.0))
    if compression_only:
        mask = sigma_n <= (-scmin)
        sigma_n_comp = np.where(mask, -sigma_n, 0.0)
        tau_allow = c[None, :] + sigma_n_comp*np.tan(phi[None, :])
        ratio = np.where(mask, np.abs(tau)/np.maximum(tau_allow, eps), 0.0)
    else:
        if scmin > 0:
            mask = sigma_n <= (-scmin)
            sigma_n_comp = np.where(mask, -sigma_n, 0.0)
            tau_allow = c[None, :] + sigma_n_comp*np.tan(phi[None, :])
            ratio = np.where(mask, np.abs(tau)/np.maximum(tau_allow, eps), 0.0)
        else:
            sigma_n_comp = np.maximum(0.0, -sigma_n)
            tau_allow = c[None, :] + sigma_n_comp*np.tan(phi[None, :])
            ratio = np.abs(tau) / np.maximum(tau_allow, eps)

    i_max = np.argmax(ratio, axis=0)
    rmax = ratio[i_max, np.arange(ratio.shape[1])]
    beta_crit = beta[i_max, 0]
    Is = rmax - 1.0
    return Is, beta_crit


def failure_ratios_pointwise(
    sxx, syy, txy, s1,
    Tm, Coh, Phi,
    alpha_const,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12,
):
    s1_pos = np.maximum(s1, 0.0)
    Rt_mat = s1_pos / np.maximum(Tm, eps)

    Is_mat, beta_crit = mc_scan_ratio(
        sxx, syy, txy, Coh, Phi,
        n_theta=int(n_theta_mc),
        eps=eps,
        compression_only=bool(mc_compression_only),
        sigma_comp_min=float(mc_sigma_comp_min)
    )
    Rs_mat = Is_mat + 1.0

    Rt_wp = np.zeros_like(Rt_mat, dtype=float)
    Rs_wp = np.zeros_like(Rs_mat, dtype=float)

    if str(strength_model).lower() == "weak_plane":
        beta_plane = float(alpha_const) + np.pi/2.0
        sigma_n, tau = plane_sigma_tau(sxx, syy, txy, beta_plane)
        T_plane = float(weak_T_ratio) * Tm
        C_plane = float(weak_C_ratio) * Coh
        phi_plane = Phi if (weak_phi is None) else weak_phi

        Rt_wp = np.maximum(sigma_n, 0.0) / np.maximum(T_plane, eps)

        scmin = float(max(mc_sigma_comp_min, 0.0))
        if mc_compression_only:
            mask = sigma_n <= (-scmin)
            sigma_comp = np.where(mask, -sigma_n, 0.0)
            tau_allow = C_plane + sigma_comp*np.tan(phi_plane)
            Rs_wp = np.where(mask, np.abs(tau)/np.maximum(tau_allow, eps), 0.0)
        else:
            if scmin > 0:
                mask = sigma_n <= (-scmin)
                sigma_comp = np.where(mask, -sigma_n, 0.0)
                tau_allow = C_plane + sigma_comp*np.tan(phi_plane)
                Rs_wp = np.where(mask, np.abs(tau)/np.maximum(tau_allow, eps), 0.0)
            else:
                sigma_comp = np.maximum(0.0, -sigma_n)
                tau_allow = C_plane + sigma_comp*np.tan(phi_plane)
                Rs_wp = np.abs(tau) / np.maximum(tau_allow, eps)

    Rt_eff = np.maximum(Rt_mat, Rt_wp)
    Rs_eff = np.maximum(Rs_mat, Rs_wp)
    return Rt_eff, Rs_eff, beta_crit

def failure_mode_map(
    sxx, syy, txy, s1,
    Tm, Coh, Phi,
    alpha_const,
    basis="first",
    util_min=0.98,
    mixed_band=0.45,
    margin=0.0,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12
):
    Rt_eff, Rs_eff, beta_crit = failure_ratios_pointwise(
        sxx, syy, txy, s1, Tm, Coh, Phi, alpha_const,
        n_theta_mc=n_theta_mc,
        mc_compression_only=mc_compression_only,
        mc_sigma_comp_min=mc_sigma_comp_min,
        strength_model=strength_model,
        weak_T_ratio=weak_T_ratio,
        weak_C_ratio=weak_C_ratio,
        weak_phi=weak_phi,
        eps=eps
    )

    mode = np.full_like(Rt_eff, "no_failure", dtype=object)

    if str(basis).lower() == "now":
        thr = 1.0 + float(margin)
        t_on = Rt_eff >= thr
        s_on = Rs_eff >= thr
        mode[t_on & ~s_on] = "tensile"
        mode[s_on & ~t_on] = "shear"
        both = t_on & s_on
        if np.any(both):
            Rt = Rt_eff[both]; Rs = Rs_eff[both]
            denom = np.maximum(np.maximum(Rt, Rs), eps)
            rel = np.abs(Rt - Rs) / denom
            is_mixed = rel <= float(mixed_band)
            idx = np.where(both)[0]
            mode[idx[is_mixed]] = "mixed"
            nm = ~is_mixed
            idx_nm = idx[nm]
            mode[idx_nm[Rt[nm] >= Rs[nm]]] = "tensile"
            mode[idx_nm[Rs[nm] > Rt[nm]]] = "shear"
        return mode, Rt_eff, Rs_eff, beta_crit

    util = np.maximum(Rt_eff, Rs_eff)
    active = util >= float(util_min)

    kt = np.full_like(Rt_eff, np.inf, dtype=float)
    ks = np.full_like(Rs_eff, np.inf, dtype=float)
    np.divide(1.0, Rt_eff, out=kt, where=(Rt_eff > eps))
    np.divide(1.0, Rs_eff, out=ks, where=(Rs_eff > eps))

    idx = np.where(active)[0]
    if idx.size > 0:
        kti = kt[idx]; ksi = ks[idx]
        both = np.isfinite(kti) & np.isfinite(ksi)
        if np.any(both):
            j = idx[both]
            kti2 = kt[j]; ksi2 = ks[j]
            kmin = np.minimum(kti2, ksi2)
            rel = np.abs(kti2 - ksi2) / np.maximum(kmin, eps)
            is_mixed = rel <= float(mixed_band)
            mode[j[is_mixed]] = "mixed"
            nm = ~is_mixed
            jnm = j[nm]
            mode[jnm[kti2[nm] <= ksi2[nm]]] = "tensile"
            mode[jnm[ksi2[nm] < kti2[nm]]] = "shear"

        only_t = np.isfinite(kti) & ~np.isfinite(ksi)
        only_s = ~np.isfinite(kti) & np.isfinite(ksi)
        mode[idx[only_t]] = "tensile"
        mode[idx[only_s]] = "shear"

    return mode, Rt_eff, Rs_eff, beta_crit

# =========================
# DISC failure-point stats
# =========================
def disc_failure_point_stats(modes, r_pts, R, rmax_frac=0.985):
    modes = np.asarray(modes, dtype=object)
    keep = np.asarray(r_pts) <= (float(rmax_frac) * float(R))
    fail_mask = keep & (modes != "no_failure")
    total_fail = int(np.sum(fail_mask))
    counts = {
        "tensile": int(np.sum(fail_mask & (modes == "tensile"))),
        "mixed":   int(np.sum(fail_mask & (modes == "mixed"))),
        "shear":   int(np.sum(fail_mask & (modes == "shear"))),
    }
    if total_fail > 0:
        pct = {k: 100.0 * counts[k] / total_fail for k in counts}
    else:
        pct = {k: 0.0 for k in counts}
    return total_fail, counts, pct

# =========================
# Crack helpers (used by both local and grid_phys)
# =========================
def _angle_mean(a, b, w):
    a = float(a); b = float(b); w = float(w)
    z = (1.0 - w) * np.exp(1j * a) + w * np.exp(1j * b)
    if abs(z) < 1e-20:
        return _wrap_pi(a)
    return _wrap_pi(np.angle(z))

def _align_line_direction(psi_target, psi_ref):
    psi_target = _wrap_pi(float(psi_target))
    psi_ref = _wrap_pi(float(psi_ref))
    d1 = abs(angle_diff_periodic(psi_target, psi_ref))
    d2 = abs(angle_diff_periodic(psi_target + np.pi, psi_ref))
    return _wrap_pi(psi_target if d1 <= d2 else (psi_target + np.pi))

def _make_outward_direction(x, y, psi):
    # ensure psi points outward relative to radial direction at (x,y)
    x = float(x); y = float(y); psi = _wrap_pi(float(psi))
    r2 = x*x + y*y
    if r2 < 1e-16:
        return psi
    rx, ry = x/np.sqrt(r2), y/np.sqrt(r2)
    vx, vy = np.cos(psi), np.sin(psi)
    if (vx*rx + vy*ry) < 0:
        psi = _wrap_pi(psi + np.pi)
    return psi

def _seg_intersect(p1, p2, q1, q2, eps=1e-12):
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def onseg(a,b,c):
        return (min(a[0],b[0])-eps <= c[0] <= max(a[0],b[0])+eps) and (min(a[1],b[1])-eps <= c[1] <= max(a[1],b[1])+eps)
    o1 = orient(p1,p2,q1)
    o2 = orient(p1,p2,q2)
    o3 = orient(q1,q2,p1)
    o4 = orient(q1,q2,p2)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if abs(o1) <= eps and onseg(p1,p2,q1): return True
    if abs(o2) <= eps and onseg(p1,p2,q2): return True
    if abs(o3) <= eps and onseg(q1,q2,p1): return True
    if abs(o4) <= eps and onseg(q1,q2,p2): return True
    return False

class UniformGridSampler:
    def __init__(self, X, Y):
        self.xmin = float(X[0, 0]); self.xmax = float(X[0, -1])
        self.ymin = float(Y[0, 0]); self.ymax = float(Y[-1, 0])
        self.nx = X.shape[1]; self.ny = X.shape[0]
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)

    def sample(self, F, x, y, fill=np.nan):
        x = float(x); y = float(y)
        if (x < self.xmin) or (x > self.xmax) or (y < self.ymin) or (y > self.ymax):
            return float(fill)
        fx = (x - self.xmin) / self.dx
        fy = (y - self.ymin) / self.dy
        j0 = int(np.floor(fx)); i0 = int(np.floor(fy))
        j1 = min(j0 + 1, self.nx - 1); i1 = min(i0 + 1, self.ny - 1)
        tx = fx - j0; ty = fy - i0
        f00 = F[i0, j0]; f10 = F[i0, j1]; f01 = F[i1, j0]; f11 = F[i1, j1]
        if (not np.isfinite(f00)) or (not np.isfinite(f10)) or (not np.isfinite(f01)) or (not np.isfinite(f11)):
            vals = np.array([f00, f10, f01, f11], dtype=float)
            ok = np.isfinite(vals)
            return float(vals[ok][0]) if np.any(ok) else float(fill)
        return float((1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11)

def stress_global_from_fit(xg, yg, R, alpha_const, fit):
    xg = np.asarray(xg, dtype=float); yg = np.asarray(yg, dtype=float)
    xm, ym = rot_to_material(xg, yg, alpha_const)
    sxx_m, syy_m, txy_m = eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
    return stress_material_to_global(sxx_m, syy_m, txy_m, alpha_const)

def ratios_and_dirs_at_point(
    x, y,
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12
):
    Teff = sampler.sample(Teff_img, x, y, fill=np.nan)
    Coh = sampler.sample(Coh_img, x, y, fill=np.nan)
    Phi = sampler.sample(Phi_img, x, y, fill=np.nan)
    if (not np.isfinite(Teff)) or (not np.isfinite(Coh)) or (not np.isfinite(Phi)):
        return np.nan, np.nan, np.nan, np.nan

    sxx, syy, txy = stress_global_from_fit(x, y, R, alpha_const, fit)
    sxx = float(np.asarray(sxx)); syy = float(np.asarray(syy)); txy = float(np.asarray(txy))
    s1, _, th = principal_from_components(np.array([sxx]), np.array([syy]), np.array([txy]))
    s1 = float(s1[0]); th = float(th[0])

    Rt_eff, Rs_eff, beta_crit = failure_ratios_pointwise(
        np.array([sxx]), np.array([syy]), np.array([txy]), np.array([s1]),
        np.array([Teff]), np.array([Coh]), np.array([Phi]),
        alpha_const,
        n_theta_mc=n_theta_mc,
        mc_compression_only=mc_compression_only,
        mc_sigma_comp_min=mc_sigma_comp_min,
        strength_model=strength_model,
        weak_T_ratio=weak_T_ratio,
        weak_C_ratio=weak_C_ratio,
        weak_phi=(np.array([weak_phi]) if weak_phi is not None and np.isscalar(weak_phi) else weak_phi),
        eps=eps
    )
    Rt = float(Rt_eff[0]); Rs = float(Rs_eff[0]); bc = float(beta_crit[0])

    psi_t = _wrap_pi(th + np.pi/2.0)
    psi_s = _wrap_pi(bc + np.pi/2.0)
    return Rt, Rs, psi_t, psi_s

# ==========================================================
# Local crack path (unchanged from your script)
# ==========================================================
def util_at_point(
    x, y,
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    n_theta_mc,
    mc_compression_only,
    mc_sigma_comp_min,
    strength_model,
    weak_T_ratio,
    weak_C_ratio,
    weak_phi,
    eps=1e-12
):
    Rt, Rs, _, _ = ratios_and_dirs_at_point(
        x, y, R, alpha_const, fit,
        sampler, Teff_img, Coh_img, Phi_img,
        n_theta_mc=n_theta_mc,
        mc_compression_only=mc_compression_only,
        mc_sigma_comp_min=mc_sigma_comp_min,
        strength_model=strength_model,
        weak_T_ratio=weak_T_ratio,
        weak_C_ratio=weak_C_ratio,
        weak_phi=weak_phi,
        eps=eps
    )
    if not np.isfinite(Rt) or not np.isfinite(Rs):
        return np.nan
    return float(max(Rt, Rs))

def pick_initial_psi0_by_first(
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    n_dir=181,
    n_theta_mc=361,
    eps=1e-12
):
    r0 = 0.015 * float(R)
    psi_grid = np.linspace(0.0, np.pi, int(n_dir), endpoint=True)
    best = (np.inf, np.pi/2.0)
    for psi in psi_grid:
        x = r0*np.cos(psi); y = r0*np.sin(psi)
        Rt, Rs, *_ = ratios_and_dirs_at_point(
            x, y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            n_theta_mc=n_theta_mc,
            mc_compression_only=mc_compression_only,
            mc_sigma_comp_min=mc_sigma_comp_min,
            strength_model=strength_model,
            weak_T_ratio=weak_T_ratio,
            weak_C_ratio=weak_C_ratio,
            weak_phi=weak_phi,
            eps=eps
        )
        if not np.isfinite(Rt) or not np.isfinite(Rs):
            continue
        kt = (1.0/Rt) if Rt > eps else np.inf
        ks = (1.0/Rs) if Rs > eps else np.inf
        kmin = min(kt, ks)
        if kmin < best[0]:
            best = (kmin, float(psi))
    return float(best[1])

def crack_path_single_mirror(
    R, alpha_const, fit,
    X, Y, M,
    Teff_img, Coh_img, Phi_img,
    drive_basis="first",
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.35,
    weak_C_ratio=0.60,
    weak_phi=None,
    weak_plane_prefer_path=True,
    ds_frac=0.008,
    a0_frac=0.008,
    max_steps=2500,
    mixed_band=0.45,
    relax=0.75,
    n_theta_mc=361,
    fail_margin=0.0,

    max_turn_deg=10.0,
    hysteresis=1.25,
    target_smooth=0.55,

    crack_path_util_min=0.98,
    seg_check_n=5,
    ds_retry_factors=(1.0, 0.75, 0.55, 0.40, 0.30, 0.22),

    cand_n=21,
    target_penalty=0.30,
    turn_penalty=0.30,

    enforce_outward=True,
    self_intersection_stop=True,
    eps=1e-12
):
    sampler = UniformGridSampler(X, Y)

    psi0 = pick_initial_psi0_by_first(
        R, alpha_const, fit,
        sampler, Teff_img, Coh_img, Phi_img,
        mc_compression_only=mc_compression_only,
        mc_sigma_comp_min=mc_sigma_comp_min,
        strength_model=strength_model,
        weak_T_ratio=weak_T_ratio,
        weak_C_ratio=weak_C_ratio,
        weak_phi=weak_phi,
        n_theta_mc=n_theta_mc,
        eps=eps
    )

    ds0 = float(ds_frac)*float(R)
    a0 = float(a0_frac)*float(R)
    m = float(fail_margin)

    xs = [0.0, a0*np.cos(psi0)]
    ys = [0.0, a0*np.sin(psi0)]
    psi = float(psi0)

    drive_mode_prev = None
    psi_target_prev = psi

    cap = np.deg2rad(float(max_turn_deg))
    hys = float(max(hysteresis, 1.0))
    cand_n = int(max(cand_n, 7))
    seg_check_n = int(max(seg_check_n, 3))

    seg_t = np.linspace(0.15, 1.0, seg_check_n)

    for step_i in range(int(max_steps)):
        tip_x, tip_y = xs[-1], ys[-1]
        if tip_x*tip_x + tip_y*tip_y >= (0.999*R)**2:
            break

        Rt, Rs, psi_t, psi_s = ratios_and_dirs_at_point(
            tip_x, tip_y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            n_theta_mc=n_theta_mc,
            mc_compression_only=mc_compression_only,
            mc_sigma_comp_min=mc_sigma_comp_min,
            strength_model=strength_model,
            weak_T_ratio=weak_T_ratio,
            weak_C_ratio=weak_C_ratio,
            weak_phi=weak_phi,
            eps=eps
        )
        if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
            break

        if str(drive_basis).lower() == "now":
            if (Rt < 1.0 + m) and (Rs < 1.0 + m):
                break

        psi_t = _align_line_direction(psi_t, psi)
        psi_s = _align_line_direction(psi_s, psi)
        psi_wp = _align_line_direction(_wrap_pi(alpha_const), psi)

        if str(drive_basis).lower() == "first":
            kt = (1.0/Rt) if Rt > eps else np.inf
            ks = (1.0/Rs) if Rs > eps else np.inf
            kmin = min(kt, ks)
            rel = abs(kt - ks) / max(kmin, eps)
            if rel <= float(mixed_band):
                drive_mode = "mixed"
            else:
                drive_mode = "tensile" if kt <= ks else "shear"

            if drive_mode_prev is not None and drive_mode != drive_mode_prev and drive_mode != "mixed":
                if drive_mode_prev == "tensile" and not (ks < kt / hys):
                    drive_mode = drive_mode_prev
                if drive_mode_prev == "shear" and not (kt < ks / hys):
                    drive_mode = drive_mode_prev

            if drive_mode == "mixed":
                w = Rt / (Rt + Rs + eps)
                psi_target = _angle_mean(psi_s, psi_t, w)
            elif drive_mode == "tensile":
                psi_target = psi_t
            else:
                psi_target = psi_s
        else:
            rel = abs(Rt - Rs) / max(max(Rt, Rs), eps)
            if rel <= float(mixed_band):
                drive_mode = "mixed"
            else:
                drive_mode = "tensile" if Rt >= Rs else "shear"

            if drive_mode_prev is not None and drive_mode != drive_mode_prev and drive_mode != "mixed":
                if drive_mode_prev == "tensile" and not (Rs > Rt * hys):
                    drive_mode = drive_mode_prev
                if drive_mode_prev == "shear" and not (Rt > Rs * hys):
                    drive_mode = drive_mode_prev

            if drive_mode == "mixed":
                w = Rt / (Rt + Rs + eps)
                psi_target = _angle_mean(psi_s, psi_t, w)
            elif drive_mode == "tensile":
                psi_target = psi_t
            else:
                psi_target = psi_s

        drive_mode_prev = drive_mode

        if str(strength_model).lower() == "weak_plane" and weak_plane_prefer_path:
            if abs(angle_diff_periodic(psi_wp, psi_target)) < np.deg2rad(15.0):
                psi_target = psi_wp

        psi_target = _angle_mean(psi_target_prev, psi_target, float(target_smooth))
        psi_target_prev = psi_target

        if enforce_outward:
            psi_target = _make_outward_direction(tip_x, tip_y, psi_target)

        dpsi0 = angle_diff_periodic(psi_target, psi)
        dpsi0 = np.clip(dpsi0, -cap, cap)
        psi_center = _wrap_pi(psi + float(relax)*dpsi0)
        if enforce_outward:
            psi_center = _make_outward_direction(tip_x, tip_y, psi_center)

        cand = psi + np.linspace(-cap, cap, cand_n)
        cand = np.array([_wrap_pi(ci) for ci in cand], dtype=float)

        best = None

        for f in ds_retry_factors:
            ds = ds0 * float(f)

            best = None
            for pc in cand:
                pc = _align_line_direction(pc, psi)
                if enforce_outward:
                    pc = _make_outward_direction(tip_x, tip_y, pc)

                nx = tip_x + ds*np.cos(pc)
                ny = tip_y + ds*np.sin(pc)

                if nx*nx + ny*ny >= (0.999*R)**2:
                    continue

                utils = []
                ok = True
                for tt in seg_t:
                    xi = tip_x + (tt*ds)*np.cos(pc)
                    yi = tip_y + (tt*ds)*np.sin(pc)
                    if xi*xi + yi*yi >= (0.999*R)**2:
                        ok = False
                        break
                    u = util_at_point(
                        xi, yi, R, alpha_const, fit,
                        sampler, Teff_img, Coh_img, Phi_img,
                        n_theta_mc=n_theta_mc,
                        mc_compression_only=mc_compression_only,
                        mc_sigma_comp_min=mc_sigma_comp_min,
                        strength_model=strength_model,
                        weak_T_ratio=weak_T_ratio,
                        weak_C_ratio=weak_C_ratio,
                        weak_phi=weak_phi,
                        eps=eps
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
                score = u_avg - float(turn_penalty)*(dturn/max(cap,1e-9))**2 - float(target_penalty)*(dtarget/max(cap,1e-9))**2

                if (best is None) or (score > best[0]):
                    best = (score, nx, ny, pc)

            if best is not None:
                break

        if best is None:
            break

        _, nx, ny, psi_new = best

        if self_intersection_stop and len(xs) > 25 and (step_i % 2 == 0):
            p1s = (xs[-1], ys[-1])
            p2s = (nx, ny)
            hit = False
            for j in range(1, len(xs)-10):
                q1 = (xs[j-1], ys[j-1])
                q2 = (xs[j], ys[j])
                if _seg_intersect(p1s, p2s, q1, q2):
                    hit = True
                    break
            if hit:
                break

        psi = float(psi_new)
        xs.append(float(nx)); ys.append(float(ny))

    xu = np.array(xs, float); yu = np.array(ys, float)
    xd = -xu; yd = -yu
    Xs = np.concatenate([xd[::-1], xu[1:]])
    Ys = np.concatenate([yd[::-1], yu[1:]])
    return Xs, Ys, psi0

# ==========================================================
# Grid-phys crack path (FIXED + IMPROVED)
# ==========================================================
def _smooth_polyline(x, y, window=11, iters=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 5:
        return x, y
    w = int(window)
    if w < 3:
        return x, y
    if w % 2 == 0:
        w += 1
    half = w // 2
    ker = np.ones(w, float) / w

    xs = x.copy()
    ys = y.copy()

    for _ in range(int(iters)):
        x0, y0 = xs[0], ys[0]
        xN, yN = xs[-1], ys[-1]

        xpad = np.r_[xs[half:0:-1], xs, xs[-2:-half-2:-1]]
        ypad = np.r_[ys[half:0:-1], ys, ys[-2:-half-2:-1]]

        xs2 = np.convolve(xpad, ker, mode="valid")
        ys2 = np.convolve(ypad, ker, mode="valid")

        xs2[0], ys2[0] = x0, y0
        xs2[-1], ys2[-1] = xN, yN
        xs, ys = xs2, ys2

    return xs, ys

def _pick_start_centered(X, Y, M, util_img, start_r_frac=0.03):
    R = float(np.nanmax(np.hypot(X[M], Y[M])))
    r = np.hypot(X, Y)
    inside = (M.astype(bool)) & np.isfinite(util_img) & (util_img > 0.0)

    cand = inside & (r <= float(start_r_frac)*R)
    if np.any(cand):
        score = util_img - 0.35*(r/(float(start_r_frac)*R + 1e-12))**2
        score[~cand] = -np.inf
        imax = int(np.nanargmax(score))
        return np.unravel_index(imax, util_img.shape)

    score = (r/(R+1e-12)) + 0.05*(1.0/np.maximum(util_img, 1e-6))
    score[~inside] = np.inf
    imin = int(np.nanargmin(score))
    return np.unravel_index(imin, util_img.shape)

def _grid_connects_to_boundary(start_ij, allowed, r, r_end):
    from collections import deque
    ny, nx = allowed.shape
    si, sj = start_ij
    if not allowed[si, sj]:
        return False
    q = deque()
    q.append((si, sj))
    seen = np.zeros_like(allowed, dtype=bool)
    seen[si, sj] = True
    neigh = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        i, j = q.popleft()
        if r[i, j] >= r_end:
            return True
        for di, dj in neigh:
            ni, nj = i+di, j+dj
            if ni<0 or ni>=ny or nj<0 or nj>=nx:
                continue
            if not allowed[ni, nj] or seen[ni, nj]:
                continue
            seen[ni, nj] = True
            q.append((ni, nj))
    return False

def _widest_threshold(start_ij, base_mask, util_img, r, r_end, iters=26):
    uvals = util_img[base_mask]
    if uvals.size == 0:
        return 0.0
    lo = 0.0
    hi = float(np.nanmax(uvals))
    for _ in range(int(iters)):
        mid = 0.5*(lo+hi)
        allowed = base_mask & (util_img >= mid)
        if _grid_connects_to_boundary(start_ij, allowed, r, r_end):
            lo = mid
        else:
            hi = mid
    return float(lo)

def _clearance_distance(allowed, dx, dy):
    """
    Distance-to-edge field inside allowed region (True).
    Multi-source Dijkstra starting at boundary cells of the allowed set.
    """
    allowed = allowed.astype(bool)
    ny, nx = allowed.shape
    dist = np.full((ny, nx), np.inf, float)

    neigh = [(-1,0),( -1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    steps = [float(np.hypot(di*dy, dj*dx)) for di,dj in neigh]

    def is_edge(i, j):
        if not allowed[i, j]:
            return False
        for di, dj in neigh:
            ni, nj = i+di, j+dj
            if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                return True
            if not allowed[ni, nj]:
                return True
        return False

    heap = []
    for i in range(ny):
        for j in range(nx):
            if is_edge(i, j):
                dist[i, j] = 0.0
                heap.append((0.0, i, j))
    heapq.heapify(heap)

    while heap:
        d, i, j = heapq.heappop(heap)
        if d != dist[i, j]:
            continue
        for step, (di, dj) in zip(steps, neigh):
            ni, nj = i+di, j+dj
            if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                continue
            if not allowed[ni, nj]:
                continue
            nd = d + step
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                heapq.heappush(heap, (nd, ni, nj))

    dist[~allowed] = np.nan
    return dist

def _snap_polyline_to_grid(x, y, X, Y, allowed, util_img, clearance=None, radius=2):
    """
    Snap interior points to best nearby allowed grid node (keeps path on band).
    """
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    if len(x) < 3:
        return x, y

    xmin = float(X[0,0]); ymin = float(Y[0,0])
    dx = float(X[0,1] - X[0,0])
    dy = float(Y[1,0] - Y[0,0])
    ny, nx = X.shape

    rad = int(max(radius, 1))
    for k in range(1, len(x)-1):
        j0 = int(np.round((x[k] - xmin)/dx))
        i0 = int(np.round((y[k] - ymin)/dy))
        best = None
        for di in range(-rad, rad+1):
            for dj in range(-rad, rad+1):
                i = i0 + di
                j = j0 + dj
                if i < 0 or i >= ny or j < 0 or j >= nx:
                    continue
                if not allowed[i, j]:
                    continue
                u = float(util_img[i, j]) if np.isfinite(util_img[i, j]) else -np.inf
                if clearance is not None and np.isfinite(clearance[i, j]):
                    u = u + 0.03*float(clearance[i, j]/(min(dx,dy)+1e-12))
                if (best is None) or (u > best[0]):
                    best = (u, i, j)
        if best is not None:
            _, bi, bj = best
            x[k] = float(X[bi, bj])
            y[k] = float(Y[bi, bj])
    return x, y

def crack_path_grid_phys_mirror(
    R, X, Y, M,
    util_img,
    psi_pref_img=None,

    start_r_frac=0.03,
    r_end_frac=0.99,

    util_soft_min=0.98,
    util_hard_min=0.20,
    widest_relax=0.98,
    util_power=6.0,
    dip_penalty=25.0,

    center_weight=4.0,      # NEW

    dir_penalty=1.5,
    turn_penalty=1.0,
    backtrack_penalty=1.0,

    smooth_window=11,
    smooth_iters=2,
    snap_radius=2,          # NEW

    hard_min_tries=(1.0, 0.8, 0.6, 0.45, 0.3, 0.2),
):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    util_img = np.asarray(util_img, float)
    ny, nx = X.shape

    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    r = np.hypot(X, Y)
    r_end = float(r_end_frac) * float(R)

    neigh = [(-1,0),( -1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    steps = [float(np.hypot(di*dy, dj*dx)) for di,dj in neigh]
    move_angle = [float(np.arctan2(di*dy, dj*dx)) for di,dj in neigh]

    base_inside = (M.astype(bool)) & np.isfinite(util_img) & (util_img > 0.0)
    start_ij = _pick_start_centered(X, Y, M, util_img, start_r_frac=start_r_frac)

    best_path = None

    for hm_fac in hard_min_tries:
        hard_min = float(util_hard_min) * float(hm_fac)
        base_mask = base_inside & (util_img >= hard_min)

        if not base_mask[start_ij]:
            rr = r.copy()
            rr[~base_mask] = np.inf
            if not np.any(np.isfinite(rr)):
                continue
            imin = int(np.nanargmin(rr))
            start_ij = np.unravel_index(imin, rr.shape)

        t_star = _widest_threshold(start_ij, base_mask, util_img, r, r_end)
        thr = max(float(hard_min), float(widest_relax)*float(t_star))

        allowed = base_inside & (util_img >= thr)
        if not _grid_connects_to_boundary(start_ij, allowed, r, r_end):
            continue

        clearance = _clearance_distance(allowed, dx, dy)
        c_eps = 0.5*min(dx, dy) + 1e-12

        u_soft = max(float(util_soft_min), 1e-12)
        cost_node = np.full((ny, nx), np.inf, float)
        ok = allowed
        u = util_img[ok]

        base = (1.0 / np.maximum(u, u_soft)) ** float(util_power)
        dip = np.clip(u_soft - u, 0.0, None)
        pen = float(dip_penalty) * (dip / u_soft)**2

        cl = clearance[ok]
        center_pen = float(center_weight) * (c_eps / np.maximum(cl + c_eps, 1e-12))

        cost_node[ok] = base + pen + center_pen

        nstate = 9
        dist = np.full((ny, nx, nstate), np.inf, float)
        prev = np.full((ny, nx, nstate, 3), -1, int)

        si, sj = start_ij
        if not np.isfinite(cost_node[si, sj]):
            continue

        dist[si, sj, 8] = 0.0
        heap = [(0.0, si, sj, 8)]
        found = None

        while heap:
            du, i, j, pdir = heapq.heappop(heap)
            if du != dist[i, j, pdir]:
                continue

            if r[i, j] >= r_end:
                found = (i, j, pdir)
                break

            psi_pref = None
            if psi_pref_img is not None:
                pp = psi_pref_img[i, j]
                if np.isfinite(pp):
                    psi_pref = float(pp)

            ri = r[i, j]

            for ndir, (di, dj) in enumerate(neigh):
                ni, nj = i+di, j+dj
                if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                    continue
                if not np.isfinite(cost_node[ni, nj]):
                    continue

                step = steps[ndir]
                w = step * 0.5 * (cost_node[i, j] + cost_node[ni, nj])

                # direction guidance: LINE angle difference
                if psi_pref is not None:
                    ang = move_angle[ndir]
                    dpsi = angle_diff_line(ang, psi_pref)
                    w += float(dir_penalty) * (dpsi/(0.5*np.pi))**2 * step

                # turn penalty: actual angle difference
                if pdir != 8:
                    dth = abs(angle_diff_periodic(move_angle[ndir], move_angle[pdir]))
                    w += float(turn_penalty) * (dth/np.pi)**2 * step

                # backtrack penalty
                rj = r[ni, nj]
                if rj < ri - 1e-12:
                    w += float(backtrack_penalty) * ((ri - rj)/max(dx,dy)) * step

                nd = du + w
                if nd < dist[ni, nj, ndir]:
                    dist[ni, nj, ndir] = nd
                    prev[ni, nj, ndir, :] = (i, j, pdir)
                    heapq.heappush(heap, (nd, ni, nj, ndir))

        if found is None:
            continue

        i, j, ddir = found
        path = [(i, j)]
        while True:
            pi, pj, pd = prev[i, j, ddir, :]
            if pi < 0 or pj < 0:
                break
            i, j, ddir = int(pi), int(pj), int(pd)
            path.append((i, j))
            if i == si and j == sj and ddir == 8:
                break
        path = path[::-1]

        ii = np.array([p[0] for p in path], int)
        jj = np.array([p[1] for p in path], int)
        xu = X[ii, jj].astype(float)
        yu = Y[ii, jj].astype(float)

        xu[0] = 0.0
        yu[0] = 0.0

        xu_s, yu_s = _smooth_polyline(xu, yu, window=int(smooth_window), iters=int(smooth_iters))

        if int(snap_radius) > 0:
            xu_s, yu_s = _snap_polyline_to_grid(
                xu_s, yu_s, X, Y,
                allowed=allowed, util_img=util_img,
                clearance=clearance, radius=int(snap_radius)
            )

        xd = -xu_s
        yd = -yu_s
        Xs = np.concatenate([xd[::-1], xu_s[1:]])
        Ys = np.concatenate([yd[::-1], yu_s[1:]])

        psi0 = float(np.arctan2(yu_s[1]-yu_s[0], xu_s[1]-xu_s[0])) if len(xu_s) >= 2 else np.pi/2.0
        best_path = (Xs, Ys, psi0)
        break

    if best_path is None:
        xs = np.array([0.0, 0.0], float)
        ys = np.array([0.0, 0.9*float(R)], float)
        xd = -xs; yd = -ys
        Xs = np.concatenate([xd[::-1], xs[1:]])
        Ys = np.concatenate([yd[::-1], ys[1:]])
        return Xs, Ys, np.pi/2.0

    return best_path

def compute_psi_pref_field(
    X, Y, M,
    Rt_eff, Rs_eff,
    sxx, syy, txy,
    beta_crit,
    drive_basis="first",
    mixed_band=0.45,
    eps=1e-12
):
    """
    Build psi_pref as a *LINE* orientation (θ ≡ θ+π), no outward forcing.
    """
    _, _, th = principal_from_components(sxx, syy, txy)

    psi_t = _wrap_pi_half(th + np.pi/2.0)
    psi_s = _wrap_pi_half(beta_crit + np.pi/2.0)

    Rt = Rt_eff
    Rs = Rs_eff

    psi_pref = np.full_like(Rt, np.nan, dtype=float)

    if str(drive_basis).lower() == "first":
        kt = np.full_like(Rt, np.inf, dtype=float)
        ks = np.full_like(Rs, np.inf, dtype=float)
        np.divide(1.0, Rt, out=kt, where=(Rt > eps))
        np.divide(1.0, Rs, out=ks, where=(Rs > eps))

        kmin = np.minimum(kt, ks)
        rel = np.abs(kt - ks) / np.maximum(kmin, eps)
        is_mixed = rel <= float(mixed_band)

        w = Rt / (Rt + Rs + eps)
        pm = np.array([_line_angle_mean(a, b, ww) for a, b, ww in zip(psi_s, psi_t, w)], dtype=float)
        psi_pref[is_mixed] = pm[is_mixed]

        nm = ~is_mixed
        psi_pref[nm] = np.where(kt[nm] <= ks[nm], psi_t[nm], psi_s[nm])

    else:
        rel = np.abs(Rt - Rs) / np.maximum(np.maximum(Rt, Rs), eps)
        is_mixed = rel <= float(mixed_band)

        w = Rt / (Rt + Rs + eps)
        pm = np.array([_line_angle_mean(a, b, ww) for a, b, ww in zip(psi_s, psi_t, w)], dtype=float)
        psi_pref[is_mixed] = pm[is_mixed]

        nm = ~is_mixed
        psi_pref[nm] = np.where(Rt[nm] >= Rs[nm], psi_t[nm], psi_s[nm])

    return np.array([_wrap_pi_half(v) if np.isfinite(v) else np.nan for v in psi_pref], dtype=float)

# =========================
# Specimen-level mode from crack path
# =========================
def specimen_mode_from_crack_path(
    xs, ys,
    R, alpha_const, fit,
    X, Y, Teff_img, Coh_img, Phi_img,
    specimen_mode_basis="first",
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.35,
    weak_C_ratio=0.60,
    weak_phi=None,
    n_theta_mc=361,
    mixed_band=0.45,
    eps=1e-12
):
    sampler = UniformGridSampler(X, Y)
    if len(xs) < 5:
        return "no_failure", dict(tensile=0.0, mixed=0.0, shear=0.0), np.nan

    pts = np.vstack([xs, ys]).T
    ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    L = float(s[-1])
    if L <= 1e-12:
        return "no_failure", dict(tensile=0.0, mixed=0.0, shear=0.0), np.nan

    ns = 80
    s_query = np.linspace(0.05*L, 0.95*L, ns)
    xq = np.interp(s_query, s, xs)
    yq = np.interp(s_query, s, ys)

    counts = dict(tensile=0, mixed=0, shear=0)
    usable = 0

    for x, y in zip(xq, yq):
        Rt, Rs, *_ = ratios_and_dirs_at_point(
            x, y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            n_theta_mc=n_theta_mc,
            mc_compression_only=mc_compression_only,
            mc_sigma_comp_min=mc_sigma_comp_min,
            strength_model=strength_model,
            weak_T_ratio=weak_T_ratio,
            weak_C_ratio=weak_C_ratio,
            weak_phi=weak_phi,
            eps=eps
        )
        if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
            continue
        usable += 1

        if str(specimen_mode_basis).lower() == "first":
            kt = (1.0/Rt) if Rt > eps else np.inf
            ks = (1.0/Rs) if Rs > eps else np.inf
            kmin = min(kt, ks)
            rel = abs(kt - ks) / max(kmin, eps)
            if rel <= float(mixed_band):
                counts["mixed"] += 1
            else:
                counts["tensile" if kt <= ks else "shear"] += 1
        else:
            rel = abs(Rt - Rs) / max(max(Rt, Rs), eps)
            if rel <= float(mixed_band):
                counts["mixed"] += 1
            else:
                counts["tensile" if Rt >= Rs else "shear"] += 1

    if usable == 0:
        return "no_failure", dict(tensile=0.0, mixed=0.0, shear=0.0), L

    fr = {k: 100.0*v/usable for k,v in counts.items()}

    if fr["tensile"] >= 60.0:
        mode = "tensile"
    elif fr["shear"] >= 60.0:
        mode = "shear"
    else:
        mode = "mixed"
    return mode, fr, L

# =========================
# Utilities
# =========================
def reduce_angle_0_90(theta_deg):
    th = float(theta_deg) % 180.0
    if th > 90.0:
        th = 180.0 - th
    return abs(th)

def map_angle_to_alpha(theta_deg, angle_map="direct"):
    th = np.deg2rad(float(theta_deg))
    if angle_map == "direct":
        a = th
    elif angle_map == "neg":
        a = -th
    elif angle_map == "pi2_minus":
        a = (np.pi/2.0) - th
    elif angle_map == "pi2_plus":
        a = (np.pi/2.0) + th
    else:
        a = th
    return _wrap_pi(a)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Brazilian disk (Lekhnitskii Airy) + physics-only failure + crack path")

    p.add_argument("--meta_csv", default="tensile_samples_data.csv")
    p.add_argument("--out_dir", default="stress_tensors_orthotropic_airy_physics_only")
    p.add_argument("--csv_out", default="specimen_stats.csv")
    p.add_argument("--save_plots", action="store_true", default=True)
    p.add_argument("--sample_ids", type=str, default="8-15")

    p.add_argument("--angle_map", type=str, default="pi2_minus",
                   choices=["direct","neg","pi2_minus","pi2_plus"])

    p.add_argument("--points_per_row", type=int, default=201)

    p.add_argument("--E1_GPa", type=float, default=50.0)
    p.add_argument("--E2_GPa", type=float, default=30.0)
    p.add_argument("--nu12", type=float, default=0.25)
    p.add_argument("--G12_GPa", type=float, default=12.0)

    p.add_argument("--platen_half_angle_deg", type=float, default=10.0)
    p.add_argument("--platen_smooth_deg", type=float, default=4.0)
    p.add_argument("--platen_mu", type=float, default=0.0)

    p.add_argument("--airy_auto_tune", action="store_true", default=True)
    p.add_argument("--airy_Nbd_base", type=int, default=480)
    p.add_argument("--airy_Nbd_arc_each", type=int, default=1200)

    p.add_argument("--fail_mixed_band", type=float, default=0.45)
    p.add_argument("--fail_mc_nplanes", type=int, default=361)
    p.add_argument("--fail_margin", type=float, default=0.0)
    p.add_argument("--fail_mode_basis", type=str, default="first", choices=["first","now"])
    p.add_argument("--fail_util_min", type=float, default=0.98)

    p.add_argument("--specimen_mode_basis", type=str, default="first", choices=["first","now"])
    p.add_argument("--crack_drive_basis", type=str, default="first", choices=["first","now"])

    p.add_argument("--mc_compression_only", action="store_true", default=True)
    p.add_argument("--mc_sigma_comp_min_MPa", type=float, default=None)
    p.add_argument("--mc_sigma_comp_min_frac_p0", type=float, default=0.20)

    p.add_argument("--tensile_factor", type=float, default=1.0)
    p.add_argument("--coh_factor", type=float, default=1.0)
    p.add_argument("--phi_offset_deg", type=float, default=0.0)

    p.add_argument("--strength_model", type=str, default="weak_plane", choices=["matrix","weak_plane"])
    p.add_argument("--weak_plane_T_ratio", type=float, default=0.35)
    p.add_argument("--weak_plane_C_ratio", type=float, default=0.60)
    p.add_argument("--weak_plane_phi_deg", type=float, default=None)
    p.add_argument("--weak_plane_prefer_path", action="store_true", default=True)

    p.add_argument("--plot_rmax_frac", type=float, default=0.985)
    p.add_argument("--stats_rmax_frac", type=float, default=0.985)

    p.add_argument("--crack_on", action="store_true", default=True)
    p.add_argument("--crack_method", type=str, default="grid_phys", choices=["grid_phys","local"])

    # local walker params
    p.add_argument("--crack_ds_frac", type=float, default=0.008)
    p.add_argument("--crack_a0_frac", type=float, default=0.008)
    p.add_argument("--crack_max_steps", type=int, default=2500)
    p.add_argument("--crack_relax", type=float, default=0.75)

    p.add_argument("--crack_max_turn_deg", type=float, default=10.0)
    p.add_argument("--crack_hysteresis", type=float, default=1.25)
    p.add_argument("--crack_target_smooth", type=float, default=0.55)

    p.add_argument("--crack_path_util_min", type=float, default=None,
                   help="(local walker) Minimum util=max(Rt,Rs) along crack step. Default=fail_util_min.")
    p.add_argument("--crack_seg_check_n", type=int, default=5)
    p.add_argument("--crack_cand_n", type=int, default=21)
    p.add_argument("--crack_target_penalty", type=float, default=0.30)
    p.add_argument("--crack_turn_penalty", type=float, default=0.30)
    p.add_argument("--crack_enforce_outward", action="store_true", default=True)
    p.add_argument("--crack_self_intersection_stop", action="store_true", default=True)

    # grid_phys params
    p.add_argument("--grid_start_r_frac", type=float, default=0.03)
    p.add_argument("--grid_r_end_frac", type=float, default=0.99)
    p.add_argument("--grid_util_hard_min", type=float, default=0.20)
    p.add_argument("--grid_widest_relax", type=float, default=0.98)
    p.add_argument("--grid_util_power", type=float, default=6.0)
    p.add_argument("--grid_dip_penalty", type=float, default=25.0)
    p.add_argument("--grid_dir_penalty", type=float, default=1.5)
    p.add_argument("--grid_turn_penalty", type=float, default=1.0)
    p.add_argument("--grid_backtrack_penalty", type=float, default=1.0)
    p.add_argument("--grid_smooth_window", type=int, default=11)
    p.add_argument("--grid_smooth_iters", type=int, default=2)

    # NEW knobs (v8)
    p.add_argument("--grid_center_weight", type=float, default=4.0)
    p.add_argument("--grid_snap_radius", type=int, default=2)

    p.add_argument("--load_factor", type=float, default=1.0)

    in_ipy = any(m in sys.modules for m in ("ipykernel","IPython"))
    args,_ = p.parse_known_args([] if in_ipy else None)

    if args.crack_path_util_min is None:
        args.crack_path_util_min = float(args.fail_util_min)

    dfmeta = pd.read_csv(args.meta_csv, index_col=0)

    def parse_sample_ids(spec, valid_index):
        wanted=set()
        if spec and isinstance(spec, str):
            for tok in spec.split(","):
                tok=tok.strip()
                if not tok: continue
                if "-" in tok:
                    a,b = tok.split("-",1)
                    try:
                        a=int(a); b=int(b)
                        wanted.update(range(min(a,b), max(a,b)+1))
                    except:
                        pass
                else:
                    try:
                        wanted.add(int(tok))
                    except:
                        pass
        return [i for i in valid_index if i in wanted]

    samples = parse_sample_ids(args.sample_ids, list(dfmeta.index))
    if not samples:
        raise RuntimeError("No matching sample IDs found.")

    os.makedirs(args.out_dir, exist_ok=True)
    crack_dir_root = os.path.join(args.out_dir, "crack_paths_physics")
    os.makedirs(crack_dir_root, exist_ok=True)

    E1 = args.E1_GPa*1e3
    E2 = args.E2_GPa*1e3
    G12 = args.G12_GPa*1e3
    nu12 = args.nu12

    results = []
    raw = {}

    for sid in samples:
        required = ['Rock_type','Angle','Diameter_mm','Thickness_mm','Load_(KN)',
                    'Tensile_strength_Mpa','Cohesion','Friction_Angle']
        missing = [c for c in required if c not in dfmeta.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in CSV: {missing}")

        rock, ang_deg, Dmm, tmm, PkN, Tm_in, coh_in, phi_deg_in = dfmeta.loc[sid, required]
        alpha_const = map_angle_to_alpha(float(ang_deg), angle_map=str(args.angle_map))

        D = float(Dmm)*1e-3
        t = float(tmm)*1e-3
        P = float(PkN)*1e3 * float(args.load_factor)
        R = D/2

        xg, yg, X, Y, M = points_in_disk(D, n=args.points_per_row)
        r_pts = np.hypot(xg, yg)

        if args.airy_auto_tune:
            fit = fit_orthotropic_airy_disk_auto(
                E1, E2, nu12, G12,
                R=R, t=t, P=P,
                alpha=alpha_const,
                beta_deg=args.platen_half_angle_deg,
                smooth_deg=args.platen_smooth_deg,
                mu=args.platen_mu,
                Nbd=args.airy_Nbd_base,
                Nbd_arc_each=args.airy_Nbd_arc_each,
            )
        else:
            fit = fit_orthotropic_airy_disk(
                E1, E2, nu12, G12,
                R=R, t=t, P=P,
                alpha=alpha_const,
                M=24,
                Nbd=args.airy_Nbd_base,
                beta_deg=args.platen_half_angle_deg,
                smooth_deg=args.platen_smooth_deg,
                mu=args.platen_mu,
                lam=1e-8,
                w_arc=12.0,
                Nbd_arc_each=args.airy_Nbd_arc_each,
            )

        if args.mc_sigma_comp_min_MPa is not None:
            mc_scmin = float(args.mc_sigma_comp_min_MPa)
        else:
            mc_scmin = float(max(args.mc_sigma_comp_min_frac_p0, 0.0)) * float(fit["p0"])

        xm, ym = rot_to_material(xg, yg, alpha_const)
        sxx_m, syy_m, txy_m = eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
        sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, alpha_const)
        s1, _, _ = principal_from_components(sxx, syy, txy)

        Tm = float(Tm_in) * float(args.tensile_factor)
        Coh0 = float(coh_in) * float(args.coh_factor)
        phi0 = np.deg2rad(float(phi_deg_in) + float(args.phi_offset_deg))

        Phi = np.full_like(s1, phi0, dtype=float)
        Coh = np.full_like(s1, Coh0, dtype=float)
        Teff = np.full_like(s1, Tm, dtype=float)

        Teff_img = np.full_like(X, np.nan, dtype=float)
        Coh_img  = np.full_like(X, np.nan, dtype=float)
        Phi_img  = np.full_like(X, np.nan, dtype=float)
        Teff_img[M] = float(Tm)
        Coh_img[M]  = float(Coh0)
        Phi_img[M]  = float(phi0)

        weak_phi_scalar = None
        if args.weak_plane_phi_deg is not None:
            weak_phi_scalar = float(np.deg2rad(args.weak_plane_phi_deg))

        raw[sid] = dict(
            rock=str(rock), ang=float(ang_deg), alpha=float(alpha_const),
            R=R, x=xg, y=yg, r=r_pts, X=X, Y=Y, M=M,
            sxx=sxx, syy=syy, txy=txy, s1=s1,
            Coh=Coh, Phi=Phi, Teff=Teff,
            Teff_img=Teff_img, Coh_img=Coh_img, Phi_img=Phi_img,
            fit=fit, mc_scmin=mc_scmin, weak_phi_scalar=weak_phi_scalar
        )

    n = len(samples)
    ncols = 2
    nrows = int(np.ceil(n/ncols))
    figF, axsF = plt.subplots(nrows, ncols, figsize=(14, 4.8*nrows))
    if nrows == 1:
        axsF = np.array([axsF])

    mode_map = {'no_failure':0,'tensile':1,'shear':2,'mixed':3}
    cmap = mcolors.ListedColormap(['purple','gold','darkgreen','red'])
    norm = mcolors.BoundaryNorm([0,1,2,3,4], 4)

    for k, sid in enumerate(samples):
        d = raw[sid]
        row, col = divmod(k, ncols)
        ax = axsF[row, col]

        # apply weak_plane_phi_deg if provided
        weak_phi_arr = None
        if d["weak_phi_scalar"] is not None:
            weak_phi_arr = np.full_like(d["s1"], float(d["weak_phi_scalar"]), dtype=float)

        modes, Rt_eff, Rs_eff, beta_crit = failure_mode_map(
            d["sxx"], d["syy"], d["txy"], d["s1"],
            Tm=d["Teff"], Coh=d["Coh"], Phi=d["Phi"],
            alpha_const=d["alpha"],
            basis=args.fail_mode_basis,
            util_min=args.fail_util_min,
            mixed_band=args.fail_mixed_band,
            margin=args.fail_margin,
            n_theta_mc=args.fail_mc_nplanes,
            mc_compression_only=args.mc_compression_only,
            mc_sigma_comp_min=d["mc_scmin"],
            strength_model=args.strength_model,
            weak_T_ratio=args.weak_plane_T_ratio,
            weak_C_ratio=args.weak_plane_C_ratio,
            weak_phi=weak_phi_arr,
        )

        util = np.maximum(Rt_eff, Rs_eff)
        util_img = np.full_like(d["X"], np.nan, dtype=float)
        util_img[d["M"]] = util

        psi_pref = compute_psi_pref_field(
            d["X"], d["Y"], d["M"],
            Rt_eff, Rs_eff,
            d["sxx"], d["syy"], d["txy"],
            beta_crit,
            drive_basis=args.crack_drive_basis,
            mixed_band=args.fail_mixed_band
        )
        psi_pref_img = np.full_like(d["X"], np.nan, dtype=float)
        psi_pref_img[d["M"]] = psi_pref

        disc_total, disc_counts, disc_pct = disc_failure_point_stats(
            modes=modes,
            r_pts=d["r"],
            R=d["R"],
            rmax_frac=args.stats_rmax_frac
        )

        mode_numeric = np.array([mode_map[m] for m in modes])
        keep = d["r"] <= (args.plot_rmax_frac * d["R"])

        ax.scatter(
            d["x"][keep], d["y"][keep],
            c=mode_numeric[keep], s=18, cmap=cmap, norm=norm,
            edgecolors='k', linewidths=0.15, zorder=1
        )
        ax.add_artist(plt.Circle((0,0), d["R"], fill=False, color='k', lw=1.2, zorder=2))

        if args.crack_on:
            if str(args.crack_method).lower() == "grid_phys":
                xs, ys, psi0 = crack_path_grid_phys_mirror(
                    R=d["R"], X=d["X"], Y=d["Y"], M=d["M"],
                    util_img=util_img,
                    psi_pref_img=psi_pref_img,
                    start_r_frac=float(args.grid_start_r_frac),
                    r_end_frac=float(args.grid_r_end_frac),
                    util_soft_min=float(args.fail_util_min),
                    util_hard_min=float(args.grid_util_hard_min),
                    widest_relax=float(args.grid_widest_relax),
                    util_power=float(args.grid_util_power),
                    dip_penalty=float(args.grid_dip_penalty),
                    center_weight=float(args.grid_center_weight),
                    dir_penalty=float(args.grid_dir_penalty),
                    turn_penalty=float(args.grid_turn_penalty),
                    backtrack_penalty=float(args.grid_backtrack_penalty),
                    smooth_window=int(args.grid_smooth_window),
                    smooth_iters=int(args.grid_smooth_iters),
                    snap_radius=int(args.grid_snap_radius),
                )
            else:
                xs, ys, psi0 = crack_path_single_mirror(
                    R=d["R"], alpha_const=d["alpha"], fit=d["fit"],
                    X=d["X"], Y=d["Y"], M=d["M"],
                    Teff_img=d["Teff_img"], Coh_img=d["Coh_img"], Phi_img=d["Phi_img"],
                    drive_basis=args.crack_drive_basis,
                    mc_compression_only=args.mc_compression_only,
                    mc_sigma_comp_min=d["mc_scmin"],
                    strength_model=args.strength_model,
                    weak_T_ratio=args.weak_plane_T_ratio,
                    weak_C_ratio=args.weak_plane_C_ratio,
                    weak_phi=d["weak_phi_scalar"],
                    weak_plane_prefer_path=args.weak_plane_prefer_path,
                    ds_frac=args.crack_ds_frac,
                    a0_frac=args.crack_a0_frac,
                    max_steps=args.crack_max_steps,
                    mixed_band=args.fail_mixed_band,
                    relax=args.crack_relax,
                    n_theta_mc=args.fail_mc_nplanes,
                    fail_margin=args.fail_margin,
                    max_turn_deg=args.crack_max_turn_deg,
                    hysteresis=args.crack_hysteresis,
                    target_smooth=args.crack_target_smooth,
                    crack_path_util_min=args.crack_path_util_min,
                    seg_check_n=args.crack_seg_check_n,
                    cand_n=args.crack_cand_n,
                    target_penalty=args.crack_target_penalty,
                    turn_penalty=args.crack_turn_penalty,
                    enforce_outward=args.crack_enforce_outward,
                    self_intersection_stop=args.crack_self_intersection_stop
                )

            if len(xs) > 2:
                ax.plot(xs, ys, 'k-', lw=2.2, zorder=10)

            spec_mode, fr_path, L = specimen_mode_from_crack_path(
                xs, ys,
                R=d["R"], alpha_const=d["alpha"], fit=d["fit"],
                X=d["X"], Y=d["Y"],
                Teff_img=d["Teff_img"], Coh_img=d["Coh_img"], Phi_img=d["Phi_img"],
                specimen_mode_basis=args.specimen_mode_basis,
                mc_compression_only=args.mc_compression_only,
                mc_sigma_comp_min=d["mc_scmin"],
                strength_model=args.strength_model,
                weak_T_ratio=args.weak_plane_T_ratio,
                weak_C_ratio=args.weak_plane_C_ratio,
                weak_phi=d["weak_phi_scalar"],
                n_theta_mc=args.fail_mc_nplanes,
                mixed_band=args.fail_mixed_band
            )

            rock_dir = os.path.join(args.out_dir, "crack_paths_physics", str(d["rock"]).replace(" ", "_"))
            os.makedirs(rock_dir, exist_ok=True)
            pd.DataFrame({"order":np.arange(len(xs)), "x_m":xs, "y_m":ys}).to_csv(
                os.path.join(rock_dir, f"crack_path_sample_{sid}.csv"), index=False
            )
        else:
            spec_mode, fr_path, L = "no_failure", dict(tensile=0.0, mixed=0.0, shear=0.0), np.nan

        ax.set_aspect('equal', 'box')
        ax.set_xlim(-d["R"]*1.05, d["R"]*1.05)
        ax.set_ylim(-d["R"]*1.05, d["R"]*1.05)

        ax.set_title(
            f"{d['rock']} (θ={d['ang']:.0f}°) AiryRMS={d['fit']['res_rms_all']:.2e} MPa\n"
            f"DISC fail pts={disc_total} | DISC% T={disc_pct['tensile']:.0f} M={disc_pct['mixed']:.0f} S={disc_pct['shear']:.0f} "
            f"| path SpecMode={spec_mode}"
        )
        ax.set_xlabel("X (m)", fontdict=font)
        ax.set_ylabel("Y (m)", fontdict=font)
        ax.tick_params(axis='x', **majorTick); ax.tick_params(axis='x', **minorTick)
        ax.tick_params(axis='y', **majorTick); ax.tick_params(axis='y', **minorTick)
        close_box(ax)

        results.append(dict(
            Sample=int(sid),
            Rock=str(d["rock"]),
            Angle_deg=float(d["ang"]),
            Angle_0_90_deg=float(reduce_angle_0_90(d["ang"])),
            angle_map=str(args.angle_map),
            alpha_rad=float(d["alpha"]),
            StrengthModel=str(args.strength_model),
            mc_sigma_comp_min_MPa=float(d["mc_scmin"]),
            SpecimenMode=str(spec_mode),
            CrackMethod=str(args.crack_method),
            CrackPath_Tensile_pct=float(fr_path["tensile"]),
            CrackPath_Mixed_pct=float(fr_path["mixed"]),
            CrackPath_Shear_pct=float(fr_path["shear"]),
            CrackPath_Length_m=float(L) if np.isfinite(L) else np.nan,
            DiscFailPts_Total=int(disc_total),
            DiscFailPts_Tensile=int(disc_counts["tensile"]),
            DiscFailPts_Mixed=int(disc_counts["mixed"]),
            DiscFailPts_Shear=int(disc_counts["shear"]),
            DiscFailPct_Tensile=float(disc_pct["tensile"]),
            DiscFailPct_Mixed=float(disc_pct["mixed"]),
            DiscFailPct_Shear=float(disc_pct["shear"]),
            p0_MPa=float(d["fit"]["p0"]),
            AiryTractionRMS_MPa=float(d["fit"]["res_rms_all"])
        ))

    for k in range(n, nrows*ncols):
        row, col = divmod(k, ncols)
        axsF[row, col].axis("off")

    handlesF = [
        plt.Line2D([0],[0], marker='o', color='w', label='no_failure', markerfacecolor='purple', markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='tensile', markerfacecolor='gold', markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='shear', markerfacecolor='darkgreen', markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='mixed', markerfacecolor='red', markersize=8),
        plt.Line2D([0],[0], color='k', lw=2, label='predicted crack path'),
    ]
    figF.legend(handles=handlesF, loc="lower center", ncol=3, frameon=True, edgecolor="black")
    figF.tight_layout(rect=[0,0.05,1,1])

    dfR = pd.DataFrame(results)
    dfR.to_csv(os.path.join(args.out_dir, args.csv_out), index=False)

    bins = [0, 15, 35, 60, 75, 90]
    labels = ["0–15°", "15–35°", "35–60°", "60–75°", "75–90°"]
    dfR["AngleBin"] = pd.cut(dfR["Angle_0_90_deg"], bins=bins, labels=labels, include_lowest=True, right=True)

    grp = dfR.groupby("AngleBin", observed=False)[["DiscFailPts_Tensile", "DiscFailPts_Mixed", "DiscFailPts_Shear"]].sum()
    tot = grp.sum(axis=1).replace(0, np.nan)

    tab = (grp.div(tot, axis=0) * 100.0).round(1)
    tab = tab.rename(columns={
        "DiscFailPts_Tensile": "tensile",
        "DiscFailPts_Mixed": "mixed",
        "DiscFailPts_Shear": "shear"
    }).fillna(0.0)

    tab.to_csv(os.path.join(args.out_dir, "angle_bin_mode_table_percent_DISC_POINTS.csv"))

    print("\n=== DISC failure-point mode (%) by angle bin ===")
    print(tab)

    if args.save_plots:
        figF.savefig(os.path.join(args.out_dir, "failure_modes_and_crack_physics_ONLY.pdf"),
                     dpi=300, bbox_inches="tight", transparent=True)
        print(f"\n✓ Saved to: {args.out_dir}")

    plt.show()
