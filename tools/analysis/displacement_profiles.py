"""Mean mid-section horizontal displacement profiles.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 19 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from tools.data_io import load_specimen_table

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    band_open_factor_from_sigma_n, get_anisotropy_ratio, get_material_axes, get_first_col,
    hertz_contact_halfwidth, mean_profile_midband,
)
from tools.lithology import repo_relative
from tools.ddm._toolkit import ROCK_ANISO_RATIO as _CANONICAL_ANISO_RATIO



# --- implementation ---------------------------------------------------


def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def to_MPa_modulus(val):
    val = float(val)
    return val * 1e3 if val < 1e3 else val


def harmonic_mix(a, b, w):
    return 1.0 / ((1.0 - w)/(a + 1e-30) + w/(b + 1e-30) + 1e-30)


def kn_ks_from_min_fractions(E2_base, G12_base, fracE2, fracG, t_band):
    fracE2 = max(float(fracE2), 1e-6)
    fracG  = max(float(fracG), 1e-6)
    kn = (fracE2 * float(E2_base)) / (t_band + 1e-30)
    ks = (fracG  * float(G12_base)) / (t_band + 1e-30)
    return kn, ks


def gaussian_random_field(shape, dx, corr_len_m, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)

    ky = np.fft.fftfreq(shape[0], d=dx) * 2*np.pi
    kx = np.fft.fftfreq(shape[1], d=dx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX*KX + KY*KY

    L = max(float(corr_len_m), 1e-12)
    H = np.exp(-0.5 * K2 * (L**2))

    W = np.fft.fft2(w)
    f = np.fft.ifft2(W * H).real
    f -= np.mean(f)
    s = np.std(f)
    if s > 1e-12:
        f /= s
    return f


def weak_band_weight_phasewarp(X, Y, angle_local, spacing, warp_m, band_halfwidth_m, phase, sharp_power):
    bw = float(band_halfwidth_m)
    s = float(spacing)
    a = np.asarray(angle_local, float)
    warp = np.asarray(warp_m, float)

    nx = -np.sin(a)
    ny =  np.cos(a)

    d = nx*X + ny*Y + float(phase)
    d_eff = d + warp
    dist_eff = (s / np.pi) * np.abs(np.sin(np.pi * d_eff / (s + 1e-30)))
    w = np.exp(-sharp_power * (dist_eff / (bw + 1e-30))**2)
    return np.clip(w, 0.0, 1.0)


def qbar_from_Es(E1, E2, nu12, G12, theta):
    E1 = np.asarray(E1, float)
    E2 = np.asarray(E2, float)
    G12 = np.asarray(G12, float)
    nu12 = float(nu12)

    nu21 = nu12 * (E2 / (E1 + 1e-30))
    den = 1.0 - nu12*nu21

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    m = np.cos(theta); n = np.sin(theta)
    m2 = m*m; n2 = n*n
    m3 = m2*m; n3 = n2*n
    m4 = m2*m2; n4 = n2*n2

    Qbar11 = Q11*m4 + 2.0*(Q12 + 2.0*Q66)*m2*n2 + Q22*n4
    Qbar22 = Q11*n4 + 2.0*(Q12 + 2.0*Q66)*m2*n2 + Q22*m4
    Qbar12 = (Q11 + Q22 - 4.0*Q66)*m2*n2 + Q12*(m4 + n4)
    Qbar16 = (Q11 - Q12 - 2.0*Q66)*m3*n - (Q22 - Q12 - 2.0*Q66)*m*n3
    Qbar26 = (Q11 - Q12 - 2.0*Q66)*m*n3 - (Q22 - Q12 - 2.0*Q66)*m3*n
    Qbar66 = (Q11 + Q22 - 2.0*Q12 - 2.0*Q66)*m2*n2 + Q66*(m4 + n4)
    return Qbar11, Qbar22, Qbar12, Qbar16, Qbar26, Qbar66


def ddx(f, mask, h):
    out = np.zeros_like(f)
    central = mask[:, 1:-1] & mask[:, 0:-2] & mask[:, 2:]
    forward = mask[:, 1:-1] & (~mask[:, 0:-2]) & mask[:, 2:]
    backward = mask[:, 1:-1] & mask[:, 0:-2] & (~mask[:, 2:])
    core = out[:, 1:-1]
    core[:] = 0.0
    core += central * ((f[:, 2:] - f[:, 0:-2]) / (2*h))
    core += forward * ((f[:, 2:] - f[:, 1:-1]) / h)
    core += backward * ((f[:, 1:-1] - f[:, 0:-2]) / h)
    eL = mask[:, 0] & mask[:, 1]
    eR = mask[:, -1] & mask[:, -2]
    out[eL, 0] = (f[eL, 1] - f[eL, 0]) / h
    out[eR, -1] = (f[eR, -1] - f[eR, -2]) / h
    out[~mask] = 0.0
    return out


def ddy(f, mask, h):
    out = np.zeros_like(f)
    central = mask[1:-1, :] & mask[0:-2, :] & mask[2:, :]
    forward = mask[1:-1, :] & (~mask[0:-2, :]) & mask[2:, :]
    backward = mask[1:-1, :] & mask[0:-2, :] & (~mask[2:, :])
    core = out[1:-1, :]
    core[:] = 0.0
    core += central * ((f[2:, :] - f[0:-2, :]) / (2*h))
    core += forward * ((f[2:, :] - f[1:-1, :]) / h)
    core += backward * ((f[1:-1, :] - f[0:-2, :]) / h)
    eB = mask[0, :] & mask[1, :]
    eT = mask[-1, :] & mask[-2, :]
    out[0, eB] = (f[1, eB] - f[0, eB]) / h
    out[-1, eT] = (f[-1, eT] - f[-2, eT]) / h
    out[~mask] = 0.0
    return out


def boundary_segments_cut_cells(X, Y, mask, R, h):
    mR = np.zeros_like(mask); mR[:, :-1] = mask[:, 1:]
    mL = np.zeros_like(mask); mL[:, 1:]  = mask[:, :-1]
    mU = np.zeros_like(mask); mU[:-1, :] = mask[1:, :]
    mD = np.zeros_like(mask); mD[1:, :]  = mask[:-1, :]

    boundary = mask & (~(mR & mL & mU & mD))
    I0, J0 = np.where(boundary)

    Iseg, Jseg = [], []
    xm_list, ym_list, nx_list, ny_list, L_list = [], [], [], [], []

    def phi_circle(x, y):
        return x*x + y*y - R*R

    edges = [(0,1),(1,2),(2,3),(3,0)]

    for i, j in zip(I0, J0):
        xc = float(X[i, j]); yc = float(Y[i, j])
        c = np.array([
            [xc - 0.5*h, yc - 0.5*h],
            [xc + 0.5*h, yc - 0.5*h],
            [xc + 0.5*h, yc + 0.5*h],
            [xc - 0.5*h, yc + 0.5*h],
        ], float)

        phi = phi_circle(c[:, 0], c[:, 1])

        pts = []
        for a, b in edges:
            pa = phi[a]; pb = phi[b]
            if (pa <= 0 and pb >= 0) or (pa >= 0 and pb <= 0):
                denom = (pa - pb)
                if abs(denom) < 1e-30:
                    continue
                t = pa / (pa - pb)
                pts.append(c[a] + t*(c[b] - c[a]))

        if len(pts) < 2:
            continue

        pts = np.array(pts, float)
        if len(pts) > 2:
            dmax = -1.0
            p1 = pts[0]; p2 = pts[1]
            for a in range(len(pts)):
                for b in range(a+1, len(pts)):
                    d = float(np.sum((pts[a] - pts[b])**2))
                    if d > dmax:
                        dmax = d
                        p1 = pts[a]
                        p2 = pts[b]
        else:
            p1, p2 = pts[0], pts[1]

        th1 = float(np.arctan2(p1[1], p1[0]))
        th2 = float(np.arctan2(p2[1], p2[0]))
        dth = (th2 - th1) % (2*np.pi)
        dth = min(dth, 2*np.pi - dth)
        L = R * dth
        if (not np.isfinite(L)) or (L <= 0):
            continue

        u1 = np.array([np.cos(th1), np.sin(th1)])
        u2 = np.array([np.cos(th2), np.sin(th2)])
        um = u1 + u2
        if np.hypot(um[0], um[1]) < 1e-12:
            pm = 0.5*(p1 + p2)
            thm = float(np.arctan2(pm[1], pm[0]))
        else:
            thm = float(np.arctan2(um[1], um[0]))

        xb = R*np.cos(thm); yb = R*np.sin(thm)
        nx = xb / (R + 1e-30); ny = yb / (R + 1e-30)

        Iseg.append(int(i)); Jseg.append(int(j))
        xm_list.append(xb); ym_list.append(yb)
        nx_list.append(nx); ny_list.append(ny)
        L_list.append(L)

    Iseg = np.asarray(Iseg, dtype=int)
    Jseg = np.asarray(Jseg, dtype=int)
    xb = np.asarray(xm_list, dtype=float)
    yb = np.asarray(ym_list, dtype=float)
    nbx = np.asarray(nx_list, dtype=float)
    nby = np.asarray(ny_list, dtype=float)
    Lseg = np.asarray(L_list, dtype=float)

    if Iseg.size == 0:
        raise RuntimeError("No cut-cell boundary segments found. Increase GRID_N.")
    return Iseg, Jseg, xb, yb, nbx, nby, Lseg



#: Physics switches read by :func:`compute_uv_for_sample`. ``main()`` assigns
#: these as module globals, which is fine while the caller is this module's own
#: ``main``. It is not fine for ``direction_circles``, which calls the solver
#: directly when its cache misses: the name was then undefined and the run died
#: rather than recomputing. Defining them here, with the values ``main`` uses,
#: makes the solver callable in any order. ``main`` still overwrites them, so
#: nothing it does changes.
enable_heterogeneity = True
enable_joint_closure = True
use_hertz_contact_width = True


def compute_uv_for_sample(
    diameter_m, thickness_m, E1_in, E2_in, nu12, alpha_in_rad, G_in=np.nan,
    *,
    target_sigma_xx_center_mpa=0.63,
    spacing_m=None,
    grid_N=201,
):
    if spacing_m is None:
        spacing_m = _rock().spacing_m
    band_halfwidth_m = 0.00045
    band_sharp_power = 12
    phase_mode = "center_safe"
    phase_shift_m = 0.0
    band_angle_deg = None

    joint_kn_MPa_per_m = None
    joint_ks_MPa_per_m = None
    band_min_E2_fraction = 0.20
    band_min_G12_fraction = 0.15
    band_min_E1_fraction = 0.95

    closure_sigma0_mpa = 0.20
    closed_compliance_fraction = 0.08
    closure_fixed_point_iters_max = 3
    closure_rel_change_tol = 2e-3

    hetero_seed = 123
    hetero_corr_len_m = 0.008
    stiffness_cv = 0.10
    angle_hetero_deg = 3.5
    spacing_warp_amp_m= _rock().spacing_warp_amp_m

    nu_contact = nu12
    hertz_outer_iters_max = 6
    hertz_rel_b_tol = 2e-3
    platen_mu = 0.0

    tol_rel_inner = 2e-6
    maxiter_inner = 800
    tol_rel_final = 1e-7
    maxiter_final = 2500

    use_precond = True
    precond_floor = 1e-6
    solver_mode = "pcg_then_bicgstab"

    diameter = float(diameter_m)
    thickness = float(thickness_m)
    R = diameter / 2.0

    E1_base = to_MPa_modulus(E1_in)
    E2_base = to_MPa_modulus(E2_in)
    G12_base = to_MPa_modulus(G_in) if np.isfinite(G_in) else np.sqrt(E1_base * E2_base) / (2.0 * (1.0 + float(nu12)))
    nu12_loc = float(nu12)

    xlin = np.linspace(-R, R, grid_N)
    ylin = np.linspace(-R, R, grid_N)
    h = float(xlin[1] - xlin[0])
    X, Y = np.meshgrid(xlin, ylin)
    mask = (X*X + Y*Y) <= R*R

    mask_idx = np.flatnonzero(mask.ravel())
    Xf = X.ravel()
    Yf = Y.ravel()

    mask_x = np.zeros_like(mask, dtype=bool)
    mask_x[:, :-1] = mask[:, :-1] & mask[:, 1:]
    mask_y = np.zeros_like(mask, dtype=bool)
    mask_y[:-1, :] = mask[:-1, :] & mask[1:, :]

    phase_plot = 0.5*spacing_m if phase_mode == "center_safe" else float(phase_shift_m)
    alpha = float(alpha_in_rad)
    band_angle0 = alpha if (band_angle_deg is None) else np.deg2rad(float(band_angle_deg))

    if enable_heterogeneity:
        fE = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=hetero_seed + 1)
        fA = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=hetero_seed + 2)
        fW = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=hetero_seed + 3)

        sigma_ln = np.sqrt(np.log(1.0 + float(stiffness_cv)**2)) if stiffness_cv > 0 else 0.0
        stiff_factor = np.exp(sigma_ln * fE)

        angle_local = band_angle0 + np.deg2rad(float(angle_hetero_deg)) * fA
        warp_m = float(spacing_warp_amp_m) * fW
    else:
        stiff_factor = np.ones_like(X)
        angle_local = band_angle0 * np.ones_like(X)
        warp_m = np.zeros_like(X)

    stiff_factor[~mask] = 1.0
    angle_local[~mask] = band_angle0
    warp_m[~mask] = 0.0

    w_band = weak_band_weight_phasewarp(
        X, Y, angle_local=angle_local, spacing=spacing_m, warp_m=warp_m,
        band_halfwidth_m=band_halfwidth_m, phase=phase_plot, sharp_power=band_sharp_power
    )
    w_band[~mask] = 0.0

    t_band = 2.0 * float(band_halfwidth_m)

    if joint_kn_MPa_per_m is None or joint_ks_MPa_per_m is None:
        kn0, ks0 = kn_ks_from_min_fractions(E2_base, G12_base,
                                            band_min_E2_fraction, band_min_G12_fraction, t_band)
        joint_kn_MPa_per_m = kn0 if joint_kn_MPa_per_m is None else joint_kn_MPa_per_m
        joint_ks_MPa_per_m = ks0 if joint_ks_MPa_per_m is None else joint_ks_MPa_per_m

    E2_band = joint_kn_MPa_per_m * t_band
    G_band  = joint_ks_MPa_per_m * t_band
    E1_band = band_min_E1_fraction * E1_base

    def build_moduli_fields(w_active):
        w = np.clip(w_active, 0.0, 1.0)
        E1_intact = E1_base * stiff_factor
        E2_intact = E2_base * stiff_factor
        G12_intact = G12_base * stiff_factor

        E1_weak = E1_band * np.ones_like(E1_intact)
        E2_weak = E2_band * np.ones_like(E2_intact)
        G12_weak = G_band  * np.ones_like(G12_intact)

        E1 = harmonic_mix(E1_intact, E1_weak, w)
        E2 = harmonic_mix(E2_intact, E2_weak, w)
        G12 = harmonic_mix(G12_intact, G12_weak, w)
        return np.maximum(E1, 1e-6), np.maximum(E2, 1e-6), np.maximum(G12, 1e-6)

    Q11 = Q22 = Q12 = Q16 = Q26 = Q66 = None
    _precond_du = None
    _precond_dv = None

    def update_Q_from_moduli(E1, E2, G12):
        nonlocal Q11, Q22, Q12, Q16, Q26, Q66, _precond_du, _precond_dv
        Q11, Q22, Q12, Q16, Q26, Q66 = qbar_from_Es(E1, E2, nu12_loc, G12, theta=float(alpha))
        if use_precond:
            mag = (np.abs(Q11) + np.abs(Q22) + 2*np.abs(Q12) + np.abs(Q66) +
                   np.abs(Q16) + np.abs(Q26))
            du = (mag / (h*h + 1e-30)) + float(precond_floor)
            dv = du.copy()
            du[~mask] = 1.0
            dv[~mask] = 1.0
            _precond_du = du
            _precond_dv = dv
        else:
            _precond_du = None
            _precond_dv = None

    def stresses(u, v):
        exx = ddx(u, mask, h)
        eyy = ddy(v, mask, h)
        gxy = ddy(u, mask, h) + ddx(v, mask, h)
        sxx = Q11*exx + Q12*eyy + Q16*gxy
        syy = Q12*exx + Q22*eyy + Q26*gxy
        txy = Q16*exx + Q26*eyy + Q66*gxy
        sxx[~mask] = 0.0
        syy[~mask] = 0.0
        txy[~mask] = 0.0
        return sxx, syy, txy

    _work = {
        "sxx_fx": np.zeros_like(X),
        "txy_fx": np.zeros_like(X),
        "syy_fy": np.zeros_like(X),
        "txy_fy": np.zeros_like(X),
        "tmpL":   np.zeros_like(X),
        "tmpD":   np.zeros_like(X),
        "divx":   np.zeros_like(X),
        "divy":   np.zeros_like(X),
    }

    def apply_K_inplace(u, v, fx_out, fy_out):
        sxx, syy, txy = stresses(u, v)

        sxx_fx = _work["sxx_fx"]; txy_fx = _work["txy_fx"]
        syy_fy = _work["syy_fy"]; txy_fy = _work["txy_fy"]
        tmpL   = _work["tmpL"];   tmpD   = _work["tmpD"]
        divx   = _work["divx"];   divy   = _work["divy"]

        sxx_fx.fill(0.0); txy_fx.fill(0.0); syy_fy.fill(0.0); txy_fy.fill(0.0)

        sxx_fx[:, :-1] = 0.5*(sxx[:, :-1] + sxx[:, 1:]) * mask_x[:, :-1]
        txy_fx[:, :-1] = 0.5*(txy[:, :-1] + txy[:, 1:]) * mask_x[:, :-1]

        syy_fy[:-1, :] = 0.5*(syy[:-1, :] + syy[1:, :]) * mask_y[:-1, :]
        txy_fy[:-1, :] = 0.5*(txy[:-1, :] + txy[1:, :]) * mask_y[:-1, :]

        tmpL.fill(0.0); tmpL[:, 1:] = sxx_fx[:, :-1]
        tmpD.fill(0.0); tmpD[1:, :] = txy_fy[:-1, :]
        divx[:] = (sxx_fx - tmpL + txy_fy - tmpD) / h

        tmpL.fill(0.0); tmpL[:, 1:] = txy_fx[:, :-1]
        tmpD.fill(0.0); tmpD[1:, :] = syy_fy[:-1, :]
        divy[:] = (txy_fx - tmpL + syy_fy - tmpD) / h

        fx_out[:] = -divx
        fy_out[:] = -divy
        fx_out[~mask] = 0.0
        fy_out[~mask] = 0.0

    def dot_mask(ax, ay, bx, by):
        axf = ax.ravel(); ayf = ay.ravel()
        bxf = bx.ravel(); byf = by.ravel()
        return float(np.dot(axf[mask_idx], bxf[mask_idx]) + np.dot(ayf[mask_idx], byf[mask_idx]))

    def norm_mask(ax, ay):
        return float(np.sqrt(dot_mask(ax, ay, ax, ay))) + 1e-30

    def project_rigid(u, v):
        uf = u.ravel(); vf = v.ravel()
        n = mask_idx.size
        if n <= 0:
            return u, v
        um = float(np.mean(uf[mask_idx]))
        vm = float(np.mean(vf[mask_idx]))
        uf[mask_idx] -= um
        vf[mask_idx] -= vm
        denom = float(np.sum((Xf[mask_idx]**2 + Yf[mask_idx]**2))) + 1e-30
        omega = float(np.sum(Xf[mask_idx]*vf[mask_idx] - Yf[mask_idx]*uf[mask_idx]) / denom)
        uf[mask_idx] += omega * Yf[mask_idx]
        vf[mask_idx] -= omega * Xf[mask_idx]
        return u, v

    def project_force_moment(bx, by):
        bxf = bx.ravel(); byf = by.ravel()
        A = mask_idx.size
        if A <= 0:
            return bx, by
        bx0 = float(np.mean(bxf[mask_idx]))
        by0 = float(np.mean(byf[mask_idx]))
        bxf[mask_idx] -= bx0
        byf[mask_idx] -= by0
        denom = float(np.sum((Xf[mask_idx]**2 + Yf[mask_idx]**2))) + 1e-30
        moment = float(np.sum(Xf[mask_idx]*byf[mask_idx] - Yf[mask_idx]*bxf[mask_idx]))
        c = moment / denom
        bxf[mask_idx] += c * Yf[mask_idx]
        byf[mask_idx] -= c * Xf[mask_idx]
        return bx, by

    def Minv(ax, ay, zx, zy):
        if _precond_du is None:
            zx[:] = ax
            zy[:] = ay
        else:
            zx[:] = ax / (_precond_du + 1e-30)
            zy[:] = ay / (_precond_dv + 1e-30)
        zx[~mask] = 0.0
        zy[~mask] = 0.0

    def solve_pcg(bx, by, u0=None, v0=None, maxiter=1000, tol_rel=1e-7):
        if u0 is None:
            u = np.zeros_like(bx); v = np.zeros_like(by)
        else:
            u = u0.copy(); v = v0.copy()
        u, v = project_rigid(u, v)

        Ax = np.zeros_like(bx); Ay = np.zeros_like(by)
        apply_K_inplace(u, v, Ax, Ay)

        rx = bx - Ax; ry = by - Ay
        rx[~mask] = 0.0; ry[~mask] = 0.0
        r0 = norm_mask(rx, ry)
        if r0 < 1e-30:
            return u, v, True

        zx = np.zeros_like(rx); zy = np.zeros_like(ry)
        Minv(rx, ry, zx, zy)
        px = zx.copy(); py = zy.copy()
        rz_old = dot_mask(rx, ry, zx, zy)

        Ap_x = np.zeros_like(rx); Ap_y = np.zeros_like(ry)

        for _ in range(maxiter):
            apply_K_inplace(px, py, Ap_x, Ap_y)
            pAp = dot_mask(px, py, Ap_x, Ap_y)
            if not np.isfinite(pAp) or pAp <= 0.0:
                return u, v, False

            alpha_c = rz_old / (pAp + 1e-30)
            u[mask] += alpha_c * px[mask]
            v[mask] += alpha_c * py[mask]
            u, v = project_rigid(u, v)

            rx[mask] -= alpha_c * Ap_x[mask]
            ry[mask] -= alpha_c * Ap_y[mask]
            rx[~mask] = 0.0; ry[~mask] = 0.0

            if norm_mask(rx, ry) / r0 < tol_rel:
                return u, v, True

            Minv(rx, ry, zx, zy)
            rz_new = dot_mask(rx, ry, zx, zy)
            beta = rz_new / (rz_old + 1e-30)
            px[mask] = zx[mask] + beta * px[mask]
            py[mask] = zy[mask] + beta * py[mask]
            px[~mask] = 0.0; py[~mask] = 0.0
            rz_old = rz_new

        return u, v, True

    def solve_bicgstab(bx, by, u0=None, v0=None, maxiter=2500, tol_rel=1e-7):
        if u0 is None:
            u = np.zeros_like(bx); v = np.zeros_like(by)
        else:
            u = u0.copy(); v = v0.copy()
        u, v = project_rigid(u, v)

        Ku = np.zeros_like(bx); Kv = np.zeros_like(by)
        apply_K_inplace(u, v, Ku, Kv)
        rx = bx - Ku; ry = by - Kv
        rx[~mask] = 0.0; ry[~mask] = 0.0

        rhatx = rx.copy(); rhaty = ry.copy()
        rho_old = 1.0; alpha_c = 1.0; omega = 1.0

        px = np.zeros_like(rx); py = np.zeros_like(ry)
        vx = np.zeros_like(rx); vy = np.zeros_like(ry)
        sx = np.zeros_like(rx); sy = np.zeros_like(ry)
        tx = np.zeros_like(rx); ty = np.zeros_like(ry)
        yx = np.zeros_like(rx); yy = np.zeros_like(ry)
        zx = np.zeros_like(rx); zy = np.zeros_like(ry)

        r0 = norm_mask(rx, ry)

        for _ in range(maxiter):
            rho_new = dot_mask(rhatx, rhaty, rx, ry)
            if abs(rho_new) < 1e-30:
                break

            beta_c = (rho_new/rho_old) * (alpha_c/omega)

            px[mask] = rx[mask] + beta_c*(px[mask] - omega*vx[mask])
            py[mask] = ry[mask] + beta_c*(py[mask] - omega*vy[mask])
            px[~mask] = 0.0; py[~mask] = 0.0

            Minv(px, py, yx, yy)
            apply_K_inplace(yx, yy, vx, vy)

            denom = dot_mask(rhatx, rhaty, vx, vy) + 1e-30
            alpha_c = rho_new / denom

            sx[:] = rx - alpha_c*vx
            sy[:] = ry - alpha_c*vy
            sx[~mask] = 0.0; sy[~mask] = 0.0

            if norm_mask(sx, sy) / r0 < tol_rel:
                u[mask] += alpha_c*yx[mask]
                v[mask] += alpha_c*yy[mask]
                u, v = project_rigid(u, v)
                return u, v

            Minv(sx, sy, zx, zy)
            apply_K_inplace(zx, zy, tx, ty)

            tt = dot_mask(tx, ty, tx, ty) + 1e-30
            omega = dot_mask(tx, ty, sx, sy) / tt

            u[mask] += alpha_c*yx[mask] + omega*zx[mask]
            v[mask] += alpha_c*yy[mask] + omega*zy[mask]
            u, v = project_rigid(u, v)

            rx[:] = sx - omega*tx
            ry[:] = sy - omega*ty
            rx[~mask] = 0.0; ry[~mask] = 0.0

            if norm_mask(rx, ry) / r0 < tol_rel:
                return u, v

            rho_old = rho_new

        return u, v

    def solve_system(bx, by, u0, v0, tol_rel, maxiter):
        if solver_mode == "pcg_then_bicgstab":
            u, v, ok = solve_pcg(bx, by, u0=u0, v0=v0, maxiter=maxiter, tol_rel=tol_rel)
            if ok:
                return u, v
            return solve_bicgstab(bx, by, u0=u, v0=v, maxiter=maxiter, tol_rel=tol_rel)
        return solve_bicgstab(bx, by, u0=u0, v0=v0, maxiter=maxiter, tol_rel=tol_rel)

    Iseg, Jseg, xb, yb, nbx, nby, Lseg = boundary_segments_cut_cells(X, Y, mask, R, h)

    def build_rhs_contact_unit_load(b_contact_m):
        bx_rhs = np.zeros_like(X)
        by_rhs = np.zeros_like(Y)

        th = np.arctan2(yb, xb)
        s_top = R * np.abs(wrap_pi(th - np.pi/2))
        s_bot = R * np.abs(wrap_pi(th + np.pi/2))

        b = float(max(b_contact_m, 1e-6))
        wtop = np.sqrt(np.maximum(1.0 - (s_top/b)**2, 0.0))
        wbot = np.sqrt(np.maximum(1.0 - (s_bot/b)**2, 0.0))
        wload = wtop + wbot

        denom = 1e6 * thickness * float(np.sum(wload * Lseg))
        if denom <= 0:
            raise RuntimeError("Contact width too small. Increase b_contact_m or grid_N.")
        p0_mpa = 1.0 / denom

        p = p0_mpa * wload
        tx = -p * nbx
        ty = -p * nby

        if platen_mu != 0.0:
            tx_t = -nby
            ty_t =  nbx
            tau = float(platen_mu) * p
            tx += -tau * tx_t
            ty += -tau * ty_t

        wfac = (Lseg / (h*h))
        np.add.at(bx_rhs, (Iseg, Jseg), tx * wfac)
        np.add.at(by_rhs, (Iseg, Jseg), ty * wfac)

        bx_rhs[~mask] = 0.0
        by_rhs[~mask] = 0.0
        bx_rhs, by_rhs = project_force_moment(bx_rhs, by_rhs)
        return bx_rhs, by_rhs, p0_mpa

    def hertz_contact_halfwidth(P_total_N, thickness_m, R_m, E_eff_MPa, nu):
        Pprime = float(P_total_N) / (float(thickness_m) + 1e-30)
        EeffPa = float(E_eff_MPa) * 1e6
        Eprime = EeffPa / (1.0 - float(nu)**2 + 1e-30)
        b = np.sqrt(4.0 * Pprime * float(R_m) / (np.pi * Eprime + 1e-30))
        return float(b)

    def band_open_factor_from_sigma_n(sigma_n_mpa, sigma0_mpa):
        s0 = float(max(sigma0_mpa, 1e-9))
        return 0.5 * (1.0 + np.tanh(np.asarray(sigma_n_mpa, float) / s0))

    def sigma_normal_to_plane(sxx, syy, txy, plane_angle):
        a = np.asarray(plane_angle, float)
        nx = -np.sin(a); ny = np.cos(a)
        return sxx*(nx*nx) + 2.0*txy*(nx*ny) + syy*(ny*ny)

    E_eff_MPa = float(np.sqrt(E1_base * E2_base))
    b_contact = 0.0010

    u_prev = None
    v_prev = None

    final_w_active = w_band.copy()
    final_b_contact = b_contact
    final_p0_unit_mpa = 1.0

    n_outer = int(hertz_outer_iters_max) if use_hertz_contact_width else 1
    scale_tmp = 1.0

    for _outer in range(n_outer):
        bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(b_contact)

        w_active = w_band.copy()
        w_prev = None

        n_fp = int(closure_fixed_point_iters_max) if enable_joint_closure else 1
        n_fp = max(n_fp, 1)

        for _it in range(n_fp):
            E1, E2, G12 = build_moduli_fields(w_active)
            update_Q_from_moduli(E1, E2, G12)

            u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev, tol_rel=tol_rel_inner, maxiter=maxiter_inner)
            u_prev, v_prev = u, v

            sxx0, syy0, txy0 = stresses(u, v)
            sxx0 = np.where(mask, sxx0, np.nan)

            ic = (grid_N - 1)//2
            jc = (grid_N - 1)//2
            st = 2
            sxx_center0 = float(np.nanmean(sxx0[ic-st:ic+st+1, jc-st:jc+st+1]))
            if (not np.isfinite(sxx_center0)) or abs(sxx_center0) < 1e-12:
                raise RuntimeError("Center σxx invalid/too small.")

            sign_tmp = -1.0 if sxx_center0 < 0 else 1.0
            scale_tmp = float(target_sigma_xx_center_mpa) / abs(sxx_center0)

            if not enable_joint_closure:
                break

            syy0 = np.where(mask, syy0, np.nan)
            txy0 = np.where(mask, txy0, np.nan)

            sxx_sc = sign_tmp * scale_tmp * sxx0
            syy_sc = sign_tmp * scale_tmp * syy0
            txy_sc = sign_tmp * scale_tmp * txy0

            sigma_n = sigma_normal_to_plane(sxx_sc, syy_sc, txy_sc, plane_angle=angle_local)
            open_fac = band_open_factor_from_sigma_n(sigma_n, sigma0_mpa=closure_sigma0_mpa)

            w_new = w_band * (closed_compliance_fraction + (1.0 - closed_compliance_fraction) * open_fac)
            w_new[~mask] = 0.0

            if w_prev is not None:
                num = np.nanmax(np.abs(w_new[mask] - w_active[mask]))
                den = np.nanmax(np.abs(w_active[mask])) + 1e-30
                if (num / den) < closure_rel_change_tol:
                    w_active = w_new
                    break

            w_prev = w_active
            w_active = w_new

        final_w_active = w_active.copy()
        final_b_contact = b_contact
        final_p0_unit_mpa = float(p0_unit_mpa)

        if not use_hertz_contact_width:
            break

        P_calib_N = abs(scale_tmp) * 1.0
        b_old = b_contact
        b_new = hertz_contact_halfwidth(P_calib_N, thickness, R, E_eff_MPa, nu_contact)
        b_new = float(np.clip(b_new, 0.00015, 0.0060))
        b_contact = 0.55 * b_contact + 0.45 * b_new

        rel = abs(b_contact - b_old) / (abs(b_old) + 1e-30)
        if rel < hertz_rel_b_tol:
            break

    bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(final_b_contact)
    E1, E2, G12 = build_moduli_fields(final_w_active)
    update_Q_from_moduli(E1, E2, G12)

    u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev, tol_rel=tol_rel_final, maxiter=maxiter_final)

    sxx0, syy0, txy0 = stresses(u, v)
    sxx0 = np.where(mask, sxx0, np.nan)

    ic = (grid_N - 1)//2
    jc = (grid_N - 1)//2
    st = 2
    sxx_center0 = float(np.nanmean(sxx0[ic-st:ic+st+1, jc-st:jc+st+1]))
    sign = -1.0 if sxx_center0 < 0 else 1.0
    scale = float(target_sigma_xx_center_mpa) / (abs(sxx_center0) + 1e-30)

    u_scaled = sign * scale * u
    v_scaled = sign * scale * v
    u_scaled[~mask] = np.nan
    v_scaled[~mask] = np.nan

    meta = {
        "R": R,
        "h": h,
        "scale": scale,
        "sign": sign,
        "b_contact": final_b_contact,
        "p0_unit_mpa": float(final_p0_unit_mpa),
        "E1_MPa": float(E1_base),
        "E2_MPa": float(E2_base),
        "G12_MPa": float(G12_base),
    }
    return X, Y, mask, u_scaled, v_scaled, meta


def get_uv_cached(idx, row):
    D = float(get_first_col(row, ["Diameter_mm"])) * 1e-3
    t = float(get_first_col(row, ["Thickness_mm"])) * 1e-3
    rock = str(get_first_col(row, ["Rock_type"]))
    anis_ratio = get_anisotropy_ratio(rock)

    # Foliation-frame constants, per lithology; see get_material_axes.
    _mx = get_material_axes(rock)
    E1_in = float(_mx["E1"])
    E2_in = float(_mx["E2"])
    nu = float(_mx["nu12"])
    G_in = float(_mx["G12"])

    phi_load_rad = float(get_first_col(row, ["Radians"]))
    if ANGLE_FROM_LOADING_AXIS:
        alpha = wrap_pi(np.pi/2.0 - phi_load_rad)
    else:
        alpha = float(phi_load_rad)

    spacing_m = float(row["Spacing_m"]) if "Spacing_m" in row.index else float(SPACING_DEFAULT_M)

    cache_key = (
        f"idx{idx}_N{GRID_N}_sp{spacing_m:.6f}_t{TARGET_SIGMA_XX_CENTER_MPA:.3f}_"
        f"het{int(enable_heterogeneity)}_cl{int(enable_joint_closure)}_hz{int(use_hertz_contact_width)}_v2.npz"
    )
    cache_path = os.path.join(cache_dir, cache_key)

    if USE_CACHE and os.path.exists(cache_path):
        z = np.load(cache_path)
        X = z["X"]; Y = z["Y"]; mask = z["mask"].astype(bool)
        u = z["u"]; v = z["v"]
        return D, X, Y, mask, u, v

    X, Y, mask, u, v, meta = compute_uv_for_sample(
        D, t, E1_in, E2_in, nu, alpha, G_in,
        target_sigma_xx_center_mpa=TARGET_SIGMA_XX_CENTER_MPA,
        spacing_m=spacing_m,
        grid_N=GRID_N,
    )

    if USE_CACHE:
        np.savez_compressed(cache_path, X=X, Y=Y, mask=mask.astype(np.uint8), u=u, v=v)

    return D, X, Y, mask, u, v



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global ANGLE_FROM_LOADING_AXIS, BAND_FRAC, CSV_PATH, D, GRAND_MEAN, \
        GRID_N, NBINS, ROCK_ANISO_RATIO, SAMPLE_IDS, SPACING_DEFAULT_M, \
        TARGET_SIGMA_XX_CENTER_MPA, USE_CACHE, X, Y, ang_deg, ax, \
        cache_dir, color, colors, df, enable_heterogeneity, \
        enable_joint_closure, fig, half, i, idx, label, mask, ok, outpath, \
        output_dir, profiles_for_grand, row, sel, tick_params, u, u_m, \
        u_mm, use_hertz_contact_width, v, vals, x_common, x_m, x_max, \
        x_min, x_mm, xc, xx, y_common, yy
    _bind(rock)
    # One dict, defined in _toolkit and derived from the replicate table.
    ROCK_ANISO_RATIO = dict(_CANONICAL_ANISO_RATIO)
    CSV_PATH = "tensile_samples_data.csv"
    SAMPLE_IDS = list(_rock().sample_ids)
    BAND_FRAC = 0.02
    NBINS = 90
    TARGET_SIGMA_XX_CENTER_MPA = 0.63
    GRID_N = 201
    ANGLE_FROM_LOADING_AXIS = True
    SPACING_DEFAULT_M = _rock().spacing_m
    USE_CACHE = True
    enable_heterogeneity = True
    enable_joint_closure = True
    use_hertz_contact_width = True
    output_dir = os.path.join(os.getcwd(), output_dirs.FIGURE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dirs.fields(), "_cache_uv_profiles_v2")
    os.makedirs(cache_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.linewidth": 1.5,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
    })
    tick_params = {
        "major": {"which": "major", "direction": "out", "length": 5, "width": 1.5},
        "minor": {"which": "minor", "direction": "out", "length": 3, "width": 1.0},
    }
    df = load_specimen_table()
    df.columns = [str(c).strip() for c in df.columns]
    print("Loaded:", CSV_PATH)
    print("Columns:", df.columns.tolist())
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(list(SAMPLE_IDS))))
    profiles_for_grand = []
    for color, idx in zip(colors, SAMPLE_IDS):
        if idx not in df.index:
            print(f"idx={idx} not found. Skipping.")
            continue

        row = df.loc[idx]
        D, X, Y, mask, u, v = get_uv_cached(idx, row)

        x_m, u_m = mean_profile_midband(X, Y, u, D, band_frac=BAND_FRAC, nbins=NBINS)
        if x_m.size == 0:
            print(f"idx={idx}: no mid-band samples.")
            continue

        x_mm = x_m * 1e3
        u_mm = u_m * 1e3

        rock = str(row["Rock_type"]) if "Rock_type" in row.index else f"Sample {idx}"
        ang_deg = float(row["Angle"]) if "Angle" in row.index else float(np.rad2deg(float(row["Radians"])))
        label = f"{rock} ({ang_deg:.0f}°)"

        ax.plot(x_mm, u_mm, lw=2.2, color=color, label=label)
        profiles_for_grand.append((x_mm, u_mm))
    GRAND_MEAN = False
    if GRAND_MEAN and profiles_for_grand:
        x_min = min(p[0].min() for p in profiles_for_grand)
        x_max = max(p[0].max() for p in profiles_for_grand)
        x_common = np.linspace(x_min, x_max, 400)
        y_common = np.full_like(x_common, np.nan, dtype=float)
        half = (x_common[1] - x_common[0]) / 2.0

        for i, xc in enumerate(x_common):
            vals = []
            for xx, yy in profiles_for_grand:
                sel = (xx >= xc - half) & (xx < xc + half)
                if np.any(sel):
                    vals.append(np.nanmean(yy[sel]))
            if vals:
                y_common[i] = float(np.nanmean(vals))
        ok = np.isfinite(y_common)
        if np.any(ok):
            ax.plot(x_common[ok], y_common[ok], "k-", lw=3.0, label="Grand mean")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Horizontal displacement u (mm)")
    ax.set_title("Mean mid-section horizontal displacement profiles")
    ax.tick_params(**tick_params["major"])
    ax.tick_params(**tick_params["minor"])
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=12, frameon=True, ncol=3)
    outpath = os.path.join(output_dir, f"{_rock().key}_mid_disp_profiles.pdf")
    plt.savefig(outpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved: {repo_relative(outpath)}")
    print(f"Cache dir: {repo_relative(cache_dir)}")
