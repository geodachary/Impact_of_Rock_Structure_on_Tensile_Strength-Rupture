#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brazilian disk — TRUE ORTHOTROPIC AIRY (Lekhnitskii) + PHYSICS-only failure + PHYSICS crack path
================================================================================================

Key physics reasons you may see "no tensile failure":
- You define tensile index It = (σ1+/Teff) - 1, and classify tensile only if It > fail_margin.
- If Teff (Tm) is derived from the same peak load P (common in Brazilian-test data),
  then at that load σ1_max is often ~ Teff, so It_max ~ 0.
  With fail_margin=0.05 and strict ">", you get no tensile points.
Fixes (still physics):
1) Use >= instead of > for failure thresholds (equality means "at failure surface").
2) Allow load scaling via --load_factor (physics: simulate slightly above/below the recorded load).
3) Print utilization diagnostics per sample: max(σ1/Teff) and max(MC ratio).

No heterogeneity, no spatial strength fields.
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------
# Optional SciPy smoothing
# -------------------------
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------------
# Plot styling
# -------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]   = 13
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
    """GLOBAL -> MATERIAL coordinates using constant alpha (uniform orthotropy)"""
    c, s = np.cos(alpha), np.sin(alpha)
    x =  c*X + s*Y
    y = -s*X + c*Y
    return x, y

def stress_material_to_global(sxx, syy, txy, alpha):
    """Rotate stress from MATERIAL -> GLOBAL (alpha is material x-axis angle in global)"""
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
    """
    Solve a*p^4 + b*p^2 + c = 0 in the standard plane-stress orthotropic form via p^2 roots.
    """
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
    """
    theta in GLOBAL, beta and smooth in radians.
    returns (tr, tt) where tr is radial traction (negative compressive inward), tt tangential.
    """
    theta = np.asarray(theta)
    d_top = np.abs(angle_diff_periodic(theta,  np.pi/2))
    d_bot = np.abs(angle_diff_periodic(theta, -np.pi/2))
    w_top = smooth_arc_window(d_top, beta, smooth)
    w_bot = smooth_arc_window(d_bot, beta, smooth)
    p = p0*(w_top + w_bot)  # MPa
    tr = -p

    if mu <= 0:
        return tr, np.zeros_like(tr)

    sign_top = -np.sign(angle_diff_periodic(theta,  np.pi/2))
    sign_bot =  np.sign(angle_diff_periodic(theta, -np.pi/2))
    tt = mu*p*(w_top*sign_top + w_bot*sign_bot)
    return tr, tt

def pressure_amplitude_from_load_tapered(P_N, t_m, R_m, beta, smooth, nint=6000):
    """
    Choose p0 so that ONE platen arc resultant vertical load equals P_N.
    (If your dataset defines P as the load cell reading, this is typically correct.)
    """
    th = np.linspace(np.pi/2 - (beta+smooth), np.pi/2 + (beta+smooth), nint)
    d_top = np.abs(angle_diff_periodic(th, np.pi/2))
    w_top = smooth_arc_window(d_top, beta, smooth)
    I = np.trapezoid(w_top*np.sin(th), th)
    denom = 1e6 * t_m * R_m * max(I, 1e-12)
    return float(P_N / denom)  # MPa

# ==========================================================
# Airy evaluation: normalized z/R basis (critical)
# ==========================================================
def eval_boundary_tractions(theta_m, R, p1, p2, a1, a2):
    """
    Normalized basis: f(z)=sum a[m]*(z/R)^m  => f''=sum a[m]*m(m-1)*(z/R)^(m-2)/R^2
    theta_m in MATERIAL coords
    """
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
    """
    Return material stress components at points (x,y) in MATERIAL coords.
    """
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
    """
    boundary angles in MATERIAL coords (theta_m), with dense points on GLOBAL platen arcs.
    platen arcs: theta_g = ±pi/2 with span (beta+smooth)
    theta_g = theta_m + alpha  => theta_m = theta_g - alpha
    """
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
    E1, E2, nu12, G12,      # MPa
    R, t, P,                # meters, meters, Newton
    alpha,                  # material axis angle in GLOBAL coords (radians)
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

    beta   = np.deg2rad(beta_deg)
    smooth = np.deg2rad(smooth_deg)
    p0 = pressure_amplitude_from_load_tapered(P, t, R, beta, smooth)

    theta_m = _build_theta_m_with_arc_oversample(alpha, beta, smooth, N_base=Nbd, N_arc_each=Nbd_arc_each)
    theta_g = theta_m + alpha

    tr_tgt, tt_tgt = platen_tractions_theta(theta_g, beta, smooth, p0, mu=mu)

    d_top = np.abs(angle_diff_periodic(theta_g,  np.pi/2))
    d_bot = np.abs(angle_diff_periodic(theta_g, -np.pi/2))
    on_arc = (d_top <= beta + smooth) | (d_bot <= beta + smooth)
    w = np.ones_like(theta_m, dtype=float)
    w[on_arc] *= float(w_arc)

    x = R*np.cos(theta_m)
    y = R*np.sin(theta_m)
    z1 = x + p1*y
    z2 = x + p2*y
    zh1 = z1 / R
    zh2 = z2 / R
    invR2 = 1.0/(R*R)

    nx = np.cos(theta_m); ny = np.sin(theta_m)

    ncoef = M - 1
    nunk  = 4*ncoef
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

        tr1,  tt1  = tr_tt_from_fpp(p1, t1)
        tr1i, tt1i = tr_tt_from_fpp(p1, 1j*t1)
        tr2,  tt2  = tr_tt_from_fpp(p2, t2)
        tr2i, tt2i = tr_tt_from_fpp(p2, 1j*t2)

        A[:nb, col+0] = tr1
        A[nb:, col+0] = tt1
        A[:nb, col+1] = tr1i
        A[nb:, col+1] = tt1i
        A[:nb, col+2] = tr2
        A[nb:, col+2] = tt2
        A[:nb, col+3] = tr2i
        A[nb:, col+3] = tt2i
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
        p1=p1, p2=p2, a1=a1, a2=a2,
        p0=p0, beta=beta, smooth=smooth,
        res_rms_all=float(res_all),
        res_rms_arc=float(res_arc),
        res_rms_free=float(res_free),
        theta_m=theta_m
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
                    M=int(M),
                    Nbd=int(Nbd),
                    beta_deg=float(beta_deg),
                    smooth_deg=float(smooth_deg),
                    mu=float(mu),
                    lam=float(lam),
                    w_arc=float(w_arc),
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
# Energy (plane stress)
# =========================
def orthotropic_D_plane_stress(E1, E2, nu12, G12):
    nu21 = (E2/E1)*nu12
    denom = 1.0 - nu12*nu21
    D11 = E1/denom
    D22 = E2/denom
    D12 = nu12*E2/denom
    D66 = G12
    return np.array([[D11, D12, 0.0],[D12, D22, 0.0],[0.0,0.0,D66]])

def rotate_stress_to_local(sxx, syy, txy, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    s11 = c*c*sxx + s*s*syy + 2*s*c*txy
    s22 = s*s*sxx + c*c*syy - 2*s*c*txy
    s12 = (syy - sxx)*s*c + txy*(c*c - s*s)
    return s11, s22, s12

def strain_energy_density_ortho(sxx, syy, txy, alpha_const, E1, E2, nu12, G12):
    D = orthotropic_D_plane_stress(E1, E2, nu12, G12)
    S = np.linalg.inv(D)
    s11, s22, s12 = rotate_stress_to_local(sxx, syy, txy, alpha_const)
    sig = np.vstack([s11, s22, s12]).T
    Uc = 0.5*np.einsum('ij,jk,ik->i', sig, S, sig)
    return Uc

# =========================
# Failure (physics-normalized): tensile index It and MC shear index Is
# =========================
def mc_scan_ratio(sxx, syy, txy, c, phi, n_theta=361, eps=1e-12):
    """
    Scan planes and compute maximum shear utilization ratio:
        ratio = |tau| / (c + sigma_n_comp tan(phi))
    Return:
        Is = ratio_max - 1
        beta_crit (plane NORMAL angle in [0,pi))
    """
    beta = np.linspace(0.0, np.pi, int(n_theta), endpoint=False)[:, None]
    cb = np.cos(2*beta); sb = np.sin(2*beta)

    s_avg  = 0.5*(sxx + syy)[None, :]
    s_diff = 0.5*(sxx - syy)[None, :]
    txyN   = txy[None, :]

    sigma_n = s_avg + s_diff*cb + txyN*sb
    tau     = -s_diff*sb + txyN*cb

    sigma_n_comp = np.maximum(0.0, -sigma_n)
    tau_allow = c[None, :] + sigma_n_comp*np.tan(phi[None, :])
    ratio = np.abs(tau) / np.maximum(tau_allow, eps)

    i_max = np.argmax(ratio, axis=0)
    rmax = ratio[i_max, np.arange(ratio.shape[1])]
    beta_crit = beta[i_max, 0]
    Is = rmax - 1.0
    return Is, beta_crit

def failure_mode_physics_normalized(
    sxx, syy, txy,
    s1, T_eff, Coh, Phi,
    mixed_band=0.30,
    n_theta_mc=361,
    eps=1e-12,
    margin=0.0
):
    """
    margin: require It >= margin or Is >= margin to count as failure.
    NOTE: using >= means "on the failure surface" is counted as failure (physically consistent).
    """
    s1_pos = np.maximum(s1, 0.0)
    It = s1_pos / np.maximum(T_eff, eps) - 1.0
    Is, beta_crit = mc_scan_ratio(sxx, syy, txy, Coh, Phi, n_theta=n_theta_mc, eps=eps)

    m = float(margin)
    t_on = It >= m
    s_on = Is >= m

    mode = np.full_like(It, 'no_failure', dtype=object)
    mode[t_on & ~s_on] = 'tensile'
    mode[s_on & ~t_on] = 'shear'

    both = t_on & s_on
    if np.any(both):
        Itb = It[both]
        Isb = Is[both]
        denom = np.maximum(np.maximum(Itb, Isb), eps)
        rel = np.abs(Itb - Isb) / denom
        is_mixed = rel <= float(mixed_band)

        idx = np.where(both)[0]
        mode[idx[is_mixed]] = 'mixed'
        nm = ~is_mixed
        idx_nm = idx[nm]
        It_nm = Itb[nm]
        Is_nm = Isb[nm]
        mode[idx_nm[It_nm >= Is_nm]] = 'tensile'
        mode[idx_nm[Is_nm >  It_nm]] = 'shear'

    return mode, It, Is, beta_crit

# =========================
# Stress evaluation helper for crack path (from Airy fit)
# =========================
def stress_global_from_fit(xg, yg, R, alpha_const, fit):
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    xm, ym = rot_to_material(xg, yg, alpha_const)
    sxx_m, syy_m, txy_m = eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
    sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, alpha_const)
    return sxx, syy, txy

# ==========================================================
# Crack path driven by PHYSICS failure indices
# ==========================================================
def _angle_mean(a, b, w):
    a = float(a); b = float(b); w = float(w)
    z = (1.0 - w) * np.exp(1j * a) + w * np.exp(1j * b)
    if abs(z) < 1e-20:
        return _wrap_pi(a)
    return _wrap_pi(np.angle(z))

def _align_line_direction(psi_target, psi_ref):
    psi_target = _wrap_pi(float(psi_target))
    psi_ref    = _wrap_pi(float(psi_ref))
    d1 = abs(angle_diff_periodic(psi_target, psi_ref))
    d2 = abs(angle_diff_periodic(psi_target + np.pi, psi_ref))
    return _wrap_pi(psi_target if d1 <= d2 else (psi_target + np.pi))

class UniformGridSampler:
    def __init__(self, X, Y):
        self.xmin = float(X[0, 0])
        self.xmax = float(X[0, -1])
        self.ymin = float(Y[0, 0])
        self.ymax = float(Y[-1, 0])
        self.nx = X.shape[1]
        self.ny = X.shape[0]
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)

    def sample(self, F, x, y, fill=np.nan):
        x = float(x); y = float(y)
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

        if (not np.isfinite(f00)) or (not np.isfinite(f10)) or (not np.isfinite(f01)) or (not np.isfinite(f11)):
            vals = np.array([f00, f10, f01, f11], dtype=float)
            ok = np.isfinite(vals)
            if np.any(ok):
                return float(vals[ok][0])
            return float(fill)

        return float((1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11)

def mc_scan_ratio_single(sxx, syy, txy, c, phi, n_theta=361, eps=1e-12):
    beta = np.linspace(0.0, np.pi, int(n_theta), endpoint=False)
    cb = np.cos(2.0 * beta)
    sb = np.sin(2.0 * beta)

    s_avg  = 0.5 * (sxx + syy)
    s_diff = 0.5 * (sxx - syy)

    sigma_n = s_avg + s_diff * cb + txy * sb
    tau     = -s_diff * sb + txy * cb

    sigma_n_comp = np.maximum(0.0, -sigma_n)
    tau_allow = c + sigma_n_comp * np.tan(phi)
    ratio = np.abs(tau) / np.maximum(tau_allow, eps)

    i = int(np.argmax(ratio))
    rmax = float(ratio[i])
    beta_crit = float(beta[i])  # plane normal angle
    Is = rmax - 1.0
    return Is, beta_crit

def failure_indices_at_point(
    x, y,
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    n_theta_mc=361,
    eps=1e-12
):
    Teff = sampler.sample(Teff_img, x, y, fill=np.nan)
    Coh  = sampler.sample(Coh_img,  x, y, fill=np.nan)
    Phi  = sampler.sample(Phi_img,  x, y, fill=np.nan)
    if (not np.isfinite(Teff)) or (not np.isfinite(Coh)) or (not np.isfinite(Phi)):
        return np.nan, np.nan, np.nan, np.nan

    sxx, syy, txy = stress_global_from_fit(x, y, R, alpha_const, fit)
    sxx = float(np.asarray(sxx))
    syy = float(np.asarray(syy))
    txy = float(np.asarray(txy))

    s1, _, th = principal_from_components(np.array([sxx]), np.array([syy]), np.array([txy]))
    s1 = float(s1[0]); th = float(th[0])

    It = (max(s1, 0.0) / max(Teff, eps)) - 1.0
    Is, beta_crit = mc_scan_ratio_single(sxx, syy, txy, Coh, Phi, n_theta=n_theta_mc, eps=eps)
    return It, Is, th, beta_crit

def pick_initial_psi0_by_failure(
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    r0_frac=0.015,
    n_dir=181,
    n_theta_mc=361,
    fail_margin=0.0
):
    r0 = float(r0_frac) * float(R)
    psi_grid = np.linspace(0.0, np.pi, int(n_dir), endpoint=True)

    best_score = -1e30
    best_psi = np.pi/2
    m = float(fail_margin)

    for psi in psi_grid:
        x = r0 * np.cos(psi)
        y = r0 * np.sin(psi)
        It, Is, _, _ = failure_indices_at_point(
            x, y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            n_theta_mc=n_theta_mc
        )
        if not np.isfinite(It) or not np.isfinite(Is):
            continue
        score = max(It, Is)
        if score > best_score:
            best_score = float(score)
            best_psi = float(psi)

    if best_score < m:
        best_psi = np.pi/2
    return float(best_psi), float(best_score)

def crack_path_physics_failure_bidirectional(
    R, alpha_const, fit,
    X, Y, M,
    Teff_img, Coh_img, Phi_img,
    ds_frac=0.010,
    a0_frac=0.008,
    max_steps=3000,
    mixed_band=0.30,
    relax=0.75,
    n_theta_mc=361,
    fail_margin=0.0,
    psi0=None,
    auto_pick_psi0=True
):
    sampler = UniformGridSampler(X, Y)
    ds = float(ds_frac) * float(R)
    a0 = float(a0_frac) * float(R)
    m = float(fail_margin)

    if auto_pick_psi0 or (psi0 is None):
        psi0, _ = pick_initial_psi0_by_failure(
            R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            r0_frac=0.015,
            n_dir=181,
            n_theta_mc=n_theta_mc,
            fail_margin=m
        )
    else:
        psi0 = float(psi0)

    def grow_one_side(psi_start):
        xs = [0.0, a0*np.cos(psi_start)]
        ys = [0.0, a0*np.sin(psi_start)]
        psi = float(psi_start)

        for _ in range(int(max_steps)):
            tip_x, tip_y = xs[-1], ys[-1]
            if tip_x*tip_x + tip_y*tip_y >= (0.999*R)**2:
                break

            It, Is, th, beta_crit = failure_indices_at_point(
                tip_x, tip_y, R, alpha_const, fit,
                sampler, Teff_img, Coh_img, Phi_img,
                n_theta_mc=n_theta_mc
            )
            if (not np.isfinite(It)) or (not np.isfinite(Is)):
                break

            # STOP only when strictly below margin (so equality continues)
            if (It < m) and (Is < m):
                break

            psi_t = _wrap_pi(th + np.pi/2.0)         # tensile: tangent ⟂ sigma1
            psi_s = _wrap_pi(beta_crit + np.pi/2.0)  # shear: tangent along MC plane

            psi_t = _align_line_direction(psi_t, psi)
            psi_s = _align_line_direction(psi_s, psi)

            t_on = (It >= m)
            s_on = (Is >= m)

            if t_on and (not s_on):
                psi_target = psi_t
            elif s_on and (not t_on):
                psi_target = psi_s
            else:
                denom = max(max(It, Is), 1e-12)
                rel = abs(It - Is) / denom
                if rel <= float(mixed_band):
                    w = float(It / (It + Is + 1e-12))  # weight toward tensile
                    psi_target = _angle_mean(psi_s, psi_t, w)
                else:
                    psi_target = psi_t if It >= Is else psi_s

            dpsi = angle_diff_periodic(psi_target, psi)
            psi = _wrap_pi(psi + float(relax) * dpsi)

            nx = tip_x + ds*np.cos(psi)
            ny = tip_y + ds*np.sin(psi)

            if nx*nx + ny*ny > R*R:
                dx = np.cos(psi); dy = np.sin(psi)
                A = dx*dx + dy*dy
                B = 2.0*(tip_x*dx + tip_y*dy)
                C = tip_x*tip_x + tip_y*tip_y - R*R
                disc = B*B - 4*A*C
                if disc <= 0:
                    break
                t1 = (-B + np.sqrt(disc))/(2*A)
                t2 = (-B - np.sqrt(disc))/(2*A)
                t_hit = max(t1, t2)
                if t_hit <= 0:
                    break
                nx = tip_x + t_hit*dx
                ny = tip_y + t_hit*dy
                xs.append(float(nx)); ys.append(float(ny))
                break

            xs.append(float(nx)); ys.append(float(ny))

        return np.array(xs, float), np.array(ys, float)

    xu, yu = grow_one_side(psi0)
    xd, yd = grow_one_side(_wrap_pi(psi0 + np.pi))

    xs = np.concatenate([xd[::-1], xu[1:]]) if len(xd) > 0 else xu
    ys = np.concatenate([yd[::-1], yu[1:]]) if len(yd) > 0 else yu
    return xs, ys, psi0

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Brazilian disk (Lekhnitskii Airy) + physics-only failure + crack path")

    # I/O
    p.add_argument("--meta_csv", default="tensile_samples_data.csv")
    p.add_argument("--out_dir", default="stress_tensors_orthotropic_airy_physics_only")
    p.add_argument("--csv_out", default="specimen_stats.csv")
    p.add_argument("--save_plots", action="store_true", default=True)
    p.add_argument("--sample_ids", type=str, default="8-15")

    # grid
    p.add_argument("--points_per_row", type=int, default=121)

    # orthotropic constants (GPa -> MPa)
    p.add_argument("--E1_GPa", type=float, default=50.0)
    p.add_argument("--E2_GPa", type=float, default=30.0)
    p.add_argument("--nu12", type=float, default=0.25)
    p.add_argument("--G12_GPa", type=float, default=12.0)

    # platen arc model
    p.add_argument("--platen_half_angle_deg", type=float, default=10.0)
    p.add_argument("--platen_smooth_deg", type=float, default=4.0)
    p.add_argument("--platen_mu", type=float, default=0.0)

    # Airy fit controls
    p.add_argument("--airy_auto_tune", action="store_true", default=True)
    p.add_argument("--airy_Nbd_base", type=int, default=480)
    p.add_argument("--airy_Nbd_arc_each", type=int, default=1200)

    # PHYSICS failure settings
    p.add_argument("--fail_mixed_band", type=float, default=0.30,
                   help="Mixed if |It-Is|/max(It,Is) <= band when both active.")
    p.add_argument("--fail_mc_nplanes", type=int, default=361)

    # IMPORTANT: default margin set to 0.0 so equality is included (It>=0 means σ1>=Teff)
    p.add_argument("--fail_margin", type=float, default=0.0,
                   help="Failure if It>=margin or Is>=margin. Use 0.0 for on-surface; "
                        "use -0.02 for 98% utilization; use +0.05 for 5% above.")

    # plotting stabilization
    p.add_argument("--plot_rmax_frac", type=float, default=0.985,
                   help="Only plot r <= plot_rmax_frac*R (hides edge ringing/speckle).")

    # Crack path settings (physics-driven)
    p.add_argument("--crack_on", action="store_true", default=True)
    p.add_argument("--crack_ds_frac", type=float, default=0.010)
    p.add_argument("--crack_a0_frac", type=float, default=0.008)
    p.add_argument("--crack_max_steps", type=int, default=3000)
    p.add_argument("--crack_relax", type=float, default=0.75)

    # energy cutoff (optional)
    p.add_argument("--Ucrit", type=float, default=None)

    # physics scaling
    p.add_argument("--load_factor", type=float, default=1.0,
                   help="Multiply applied load P by this factor (physics: simulate below/above recorded load).")

    in_ipy = any(m in sys.modules for m in ("ipykernel","IPython"))
    args,_ = p.parse_known_args([] if in_ipy else None)

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
                    except: pass
                else:
                    try: wanted.add(int(tok))
                    except: pass
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
    all_U = []
    raw = {}

    for sid in samples:
        rock, ang, Dmm, tmm, PkN, Tm, fol_base_loading, coh, phi_deg = \
            dfmeta.loc[sid, ['Rock_type','Angle','Diameter_mm','Thickness_mm',
                             'Load_(KN)','Tensile_strength_Mpa','Radians',
                             'Cohesion','Friction_Angle']]

        # CSV foliation angle is with respect to loading axis (y). Convert to x-axis angle:
        fol_base_x = _wrap_pi(np.pi/2.0 - float(fol_base_loading))

        D = float(Dmm)*1e-3
        t = float(tmm)*1e-3
        P = float(PkN)*1e3 * float(args.load_factor)   # <--- physics scaling
        R = D/2

        xg, yg, X, Y, M = points_in_disk(D, n=args.points_per_row)
        r_pts = np.hypot(xg, yg)

        xm, ym = rot_to_material(xg, yg, fol_base_x)

        if args.airy_auto_tune:
            fit = fit_orthotropic_airy_disk_auto(
                E1, E2, nu12, G12,
                R=R, t=t, P=P,
                alpha=fol_base_x,
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
                alpha=fol_base_x,
                M=24,
                Nbd=args.airy_Nbd_base,
                beta_deg=args.platen_half_angle_deg,
                smooth_deg=args.platen_smooth_deg,
                mu=args.platen_mu,
                lam=1e-8,
                w_arc=12.0,
                Nbd_arc_each=args.airy_Nbd_arc_each,
            )
            fit["M_best"] = 24
            fit["lam_best"] = 1e-8
            fit["w_arc_best"] = 12.0
            fit["score"] = 0.45*fit["res_rms_all"] + 0.35*fit["res_rms_free"] + 0.20*fit["res_rms_arc"]

        sxx_m, syy_m, txy_m = eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
        sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, fol_base_x)
        s1, s3, th = principal_from_components(sxx, syy, txy)

        Phi = np.full_like(s1, np.deg2rad(float(phi_deg)), dtype=float)
        Coh = np.full_like(s1, float(coh), dtype=float)
        Teff_fail = np.full_like(s1, float(Tm), dtype=float)

        # energy
        U = strain_energy_density_ortho(sxx, syy, txy, fol_base_x, E1, E2, nu12, G12)
        all_U.append(U)

        # grid images for crack-path sampling
        Teff_img = np.full_like(X, np.nan, dtype=float)
        Coh_img  = np.full_like(X, np.nan, dtype=float)
        Phi_img  = np.full_like(X, np.nan, dtype=float)
        Teff_img[M] = float(Tm)
        Coh_img[M]  = float(coh)
        Phi_img[M]  = np.deg2rad(float(phi_deg))

        raw[sid] = dict(
            rock=str(rock), ang=float(ang), D=D, t=t, P=P, R=R,
            x=xg, y=yg, r=r_pts, X=X, Y=Y, M=M,
            sxx=sxx, syy=syy, txy=txy, s1=s1, s3=s3, th=th,
            alpha_const=fol_base_x,
            Coh=Coh, Phi=Phi, Teff_fail=Teff_fail,
            Teff_img=Teff_img, Coh_img=Coh_img, Phi_img=Phi_img,
            U=U,
            airy_res_rms=fit["res_rms_all"],
            airy_res_rms_arc=fit["res_rms_arc"],
            airy_res_rms_free=fit["res_rms_free"],
            airy_p0=fit["p0"],
            beta_deg=args.platen_half_angle_deg,
            airy_M=fit.get("M_best", None),
            airy_lam=fit.get("lam_best", None),
            airy_w_arc=fit.get("w_arc_best", None),
            fit=fit
        )

        np.savez_compressed(
            os.path.join(args.out_dir, f"stress_fields_sample_{sid}.npz"),
            x=xg, y=yg, r=r_pts,
            sxx=sxx, syy=syy, txy=txy,
            s1=s1, s3=s3, th=th,
            U=U,
            alpha_const=fol_base_x,
            Teff_fail=Teff_fail,
            Coh=Coh, Phi=Phi,
            D=D, t=t, P=P,
            airy_res_rms_all=fit["res_rms_all"],
            airy_res_rms_arc=fit["res_rms_arc"],
            airy_res_rms_free=fit["res_rms_free"],
            airy_p0=fit["p0"],
            platen_half_angle_deg=args.platen_half_angle_deg,
            airy_M_best=fit.get("M_best", np.nan),
            airy_lam_best=fit.get("lam_best", np.nan),
            airy_w_arc_best=fit.get("w_arc_best", np.nan),
        )

        # --- diagnostics: peak utilization ---
        eps = 1e-12
        Rt = np.maximum(s1, 0.0) / np.maximum(Teff_fail, eps)   # tensile ratio (>=1 means tensile failure)
        Is_tmp, _ = mc_scan_ratio(sxx, syy, txy, Coh, Phi, n_theta=args.fail_mc_nplanes, eps=eps)
        Rs = Is_tmp + 1.0                                       # shear ratio (>=1 means shear failure)

        iRt = int(np.nanargmax(Rt))
        iRs = int(np.nanargmax(Rs))
        print(
            f"[sid {sid}] {rock} θ={ang}°  load_factor={args.load_factor:.3f}  "
            f"AiryRMS(all)={fit['res_rms_all']:.4e} MPa  free={fit['res_rms_free']:.4e} MPa  arc={fit['res_rms_arc']:.4e} MPa  "
            f"p0={fit['p0']:.3f} MPa\n"
            f"           max tensile ratio Rt=max(σ1/Teff)={Rt[iRt]:.4f} at (x,y)=({xg[iRt]:+.4e},{yg[iRt]:+.4e}), "
            f"max shear ratio Rs=max(MC)={Rs[iRs]:.4f} at (x,y)=({xg[iRs]:+.4e},{yg[iRs]:+.4e})"
        )

    # energy cutoff
    if args.Ucrit is None:
        Uall = np.hstack(all_U)
        Uall = Uall[np.isfinite(Uall)]
        if Uall.size == 0:
            raise RuntimeError("All energy values are NaN/inf. Check Airy root/fit.")
        args.Ucrit = float(np.percentile(Uall, 90))
        print(f"Ucrit = {args.Ucrit:.4f} MPa (90th percentile)")

    # figures
    n = len(samples)
    ncols = 2
    nrows = int(np.ceil(n/ncols))

    figF, axsF = plt.subplots(nrows, ncols, figsize=(14, 4.8*nrows))
    figE, axsE = plt.subplots(nrows, ncols, figsize=(14, 4.8*nrows))
    if nrows == 1:
        axsF = np.array([axsF])
        axsE = np.array([axsE])

    mode_map = {'no_failure':0,'tensile':1,'shear':2,'mixed':3}
    cmap = mcolors.ListedColormap(['purple','gold','darkgreen','red'])
    norm = mcolors.BoundaryNorm([0,1,2,3,4], 4)

    for k, sid in enumerate(samples):
        d = raw[sid]
        row, col = divmod(k, ncols)

        axF = axsF[row, col]
        axE = axsE[row, col]

        xg, yg = d["x"], d["y"]
        R = d["R"]

        sxx, syy, txy = d["sxx"], d["syy"], d["txy"]
        s1 = d["s1"]
        Coh, Phi = d["Coh"], d["Phi"]
        Teff_fail = d["Teff_fail"]
        U = d["U"]

        modes, It, Is, beta_crit = failure_mode_physics_normalized(
            sxx, syy, txy, s1, Teff_fail, Coh, Phi,
            mixed_band=args.fail_mixed_band,
            n_theta_mc=args.fail_mc_nplanes,
            margin=args.fail_margin
        )
        mode_numeric = np.array([mode_map[m] for m in modes])

        keep = d["r"] <= (args.plot_rmax_frac * d["R"])

        axF.scatter(
            xg[keep], yg[keep], c=mode_numeric[keep], s=18, cmap=cmap, norm=norm,
            edgecolors='k', linewidths=0.15, zorder=1
        )
        axF.add_artist(plt.Circle((0,0), R, fill=False, color='k', lw=1.2, zorder=2))

        xs = np.array([]); ys = np.array([])
        if args.crack_on:
            xs, ys, psi0_used = crack_path_physics_failure_bidirectional(
                R=R,
                alpha_const=d["alpha_const"],
                fit=d["fit"],
                X=d["X"], Y=d["Y"], M=d["M"],
                Teff_img=d["Teff_img"],
                Coh_img=d["Coh_img"],
                Phi_img=d["Phi_img"],
                ds_frac=args.crack_ds_frac,
                a0_frac=args.crack_a0_frac,
                max_steps=args.crack_max_steps,
                mixed_band=args.fail_mixed_band,
                relax=args.crack_relax,
                n_theta_mc=args.fail_mc_nplanes,
                fail_margin=args.fail_margin,
                auto_pick_psi0=True
            )

            if len(xs) > 2:
                axF.plot(xs, ys, 'k-', lw=2.2, zorder=10)

                rock_dir = os.path.join(crack_dir_root, str(d["rock"]).replace(" ", "_"))
                os.makedirs(rock_dir, exist_ok=True)
                pd.DataFrame({"order":np.arange(len(xs)), "x_m":xs, "y_m":ys}).to_csv(
                    os.path.join(rock_dir, f"crack_path_sample_{sid}.csv"), index=False
                )

        axF.set_aspect('equal', 'box')
        axF.set_xlim(-R*1.05, R*1.05)
        axF.set_ylim(-R*1.05, R*1.05)
        axF.set_title(
            f"{d['rock']} (θ={d['ang']}°) "
            f"AiryRMS={d['airy_res_rms']:.2e} MPa  free={d['airy_res_rms_free']:.2e} MPa  physics"
        )
        axF.set_xlabel("X (m)", fontdict=font)
        axF.set_ylabel("Y (m)", fontdict=font)
        axF.tick_params(axis='x', **majorTick); axF.tick_params(axis='x', **minorTick)
        axF.tick_params(axis='y', **majorTick); axF.tick_params(axis='y', **minorTick)
        close_box(axF)

        vmax = np.percentile(U[keep], 95) if np.any(keep) else np.percentile(U, 95)
        im = axE.scatter(xg[keep], yg[keep], c=U[keep], s=14, cmap="inferno", vmin=0, vmax=vmax)
        axE.add_artist(plt.Circle((0,0), R, fill=False, color='k', lw=1.2))
        axE.set_aspect('equal', 'box')
        axE.set_xlim(-R*1.05, R*1.05)
        axE.set_ylim(-R*1.05, R*1.05)
        axE.set_title(f"Energy U* (θ={d['ang']}°)")
        axE.set_xlabel("X (m)", fontdict=font)
        axE.set_ylabel("Y (m)", fontdict=font)
        axE.tick_params(axis='x', **majorTick); axE.tick_params(axis='x', **minorTick)
        axE.tick_params(axis='y', **majorTick); axE.tick_params(axis='y', **minorTick)
        close_box(axE)
        cb = plt.colorbar(im, ax=axE, fraction=0.046, pad=0.04)
        cb.set_label("U* (MPa)")

        highU = int(np.sum(U >= args.Ucrit))
        results.append(dict(
            Sample=int(sid),
            Rock=str(d["rock"]),
            Theta_deg=float(d["ang"]),
            LoadFactor=float(args.load_factor),
            AiryTractionRMS_MPa=float(d["airy_res_rms"]),
            AiryTractionRMS_free_MPa=float(d["airy_res_rms_free"]),
            AiryTractionRMS_arc_MPa=float(d["airy_res_rms_arc"]),
            p0_MPa=float(d["airy_p0"]),
            PlatenHalfAngle_deg=float(d["beta_deg"]),
            Airy_M_best=float(d["airy_M"]) if d["airy_M"] is not None else np.nan,
            Airy_lam_best=float(d["airy_lam"]) if d["airy_lam"] is not None else np.nan,
            Airy_w_arc_best=float(d["airy_w_arc"]) if d["airy_w_arc"] is not None else np.nan,
            Sigma1_max=float(np.max(d["s1"])),
            Sigma3_min=float(np.min(d["s3"])),
            EnergyPts_ge_Ucrit=highU
        ))

        vals, counts = np.unique(modes, return_counts=True)
        fr = {v: 100.0*c/len(modes) for v,c in zip(vals,counts)}
        print(f"[sid {sid}] mode%: no_failure={fr.get('no_failure',0):.1f} "
              f"tensile={fr.get('tensile',0):.1f} shear={fr.get('shear',0):.1f} mixed={fr.get('mixed',0):.1f}")

    for k in range(n, nrows*ncols):
        row, col = divmod(k, ncols)
        axsF[row, col].axis("off")
        axsE[row, col].axis("off")

    handlesF = [
        plt.Line2D([0],[0], marker='o', color='w', label='no_failure', markerfacecolor='purple', markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='tensile',   markerfacecolor='gold', markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='shear',     markerfacecolor='darkgreen', markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='mixed',     markerfacecolor='red', markersize=8),
        plt.Line2D([0],[0], color='k', lw=2, label='predicted crack path (physics failure)'),
    ]
    figF.legend(handles=handlesF, loc="lower center", ncol=3, frameon=True, edgecolor="black")
    figF.tight_layout(rect=[0,0.05,1,1])
    figE.tight_layout()

    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, args.csv_out), index=False)

    if args.save_plots:
        figF.savefig(os.path.join(args.out_dir, "failure_modes_and_crack_physics_ONLY.pdf"),
                     dpi=300, bbox_inches="tight", transparent=True)
        figE.savefig(os.path.join(args.out_dir, "energy_maps_physics_ONLY.pdf"),
                     dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved figures + CSV to: {args.out_dir}")

    plt.show()
