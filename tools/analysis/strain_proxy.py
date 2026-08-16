"""Tensile-strain proxy field and its principal direction.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 13 during the
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
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from tools.data_io import load_specimen_table

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    axis_angle, band_open_factor_from_sigma_n, build_segments_from_axis,
    get_anisotropy_ratio, hertz_contact_halfwidth, reconstruct_grid,
)



# --- implementation ---------------------------------------------------


def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def compact_axis_labels(ax, row, col, nrows, ncols,
                        xlabel="X (m)", ylabel="Y (m)",
                        tick_pad=2, label_pad=2):
    ax.tick_params(axis="both", which="both", pad=tick_pad)

    if row == nrows - 1:
        ax.set_xlabel(xlabel, labelpad=label_pad)
        ax.tick_params(labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if col == 0:
        ax.set_ylabel(ylabel, labelpad=label_pad)
        ax.tick_params(labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def gaussian_smooth_nan_fft(field, mask, sigma_pix):
    sigma_pix = float(sigma_pix)
    if sigma_pix <= 0:
        out = np.array(field, float)
        out[~mask] = np.nan
        return out

    H, W = field.shape
    key = (H, W, sigma_pix)
    if key in _fft_cache:
        Gk = _fft_cache[key]
    else:
        ky = np.fft.fftfreq(H, d=1.0) * 2*np.pi
        kx = np.fft.fftfreq(W, d=1.0) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX*KX + KY*KY
        Gk = np.exp(-0.5 * (sigma_pix**2) * K2)
        _fft_cache[key] = Gk

    f0 = np.where(mask, np.nan_to_num(field, nan=0.0), 0.0)
    m0 = mask.astype(float)

    num = np.fft.ifft2(np.fft.fft2(f0) * Gk).real
    den = np.fft.ifft2(np.fft.fft2(m0) * Gk).real
    out = num / (den + 1e-30)
    out[~mask] = np.nan
    return out


def nematic_smooth_theta(theta, mask, sigma_pix):
    if sigma_pix <= 0:
        return theta
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    c2s = gaussian_smooth_nan_fft(c2, mask, sigma_pix=sigma_pix)
    s2s = gaussian_smooth_nan_fft(s2, mask, sigma_pix=sigma_pix)
    ths = 0.5 * np.arctan2(s2s, c2s)
    ths[~mask] = np.nan
    return ths


def plot_panel(ax, segments, colors, R, title, norm):
    lc = LineCollection(
        segments,
        array=colors,
        cmap=seg_cmap,
        norm=norm,
        linewidths=seg_linewidth,
        alpha=seg_alpha
    )
    ax.add_collection(lc)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, linewidth=1.5, color="k"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    ax.set_title(title, pad=10)
    ax.tick_params(**tick_params["major"])
    ax.tick_params(**tick_params["minor"])
    ax.minorticks_on()
    return lc


def compute_sigma1plus_and_theta_for_sample(
    D, t, E1_in, E2_in, nu12, alpha, G_in=np.nan,
    target_sigma_xx_center_mpa=0.63,
    spacing_m=None,
    grid_N=201,
    **opts
):
    if spacing_m is None:
        spacing_m = _rock().spacing_m
    band_halfwidth_m = float(opts.get("band_halfwidth_m", 0.0))
    band_sharp_power = float(opts.get("band_sharp_power", 12.0))
    phase_mode = str(opts.get("phase_mode", "center_safe"))
    phase_shift_m = float(opts.get("phase_shift_m", 0.0))
    band_angle_rad = opts.get("band_angle_rad", None)

    joint_kn_MPa_per_m = opts.get("joint_kn_MPa_per_m", None)
    joint_ks_MPa_per_m = opts.get("joint_ks_MPa_per_m", None)
    band_min_E2_fraction = float(opts.get("band_min_E2_fraction", 0.20))
    band_min_G12_fraction = float(opts.get("band_min_G12_fraction", 0.15))
    band_min_E1_fraction = float(opts.get("band_min_E1_fraction", 0.95))

    enable_joint_closure = bool(opts.get("enable_joint_closure", True))
    closure_sigma0_mpa = float(opts.get("closure_sigma0_mpa", 0.20))
    closed_compliance_fraction = float(opts.get("closed_compliance_fraction", 0.08))
    closure_fixed_point_iters_max = int(opts.get("closure_fixed_point_iters_max", 3))
    closure_rel_change_tol = float(opts.get("closure_rel_change_tol", 2e-3))

    enable_heterogeneity = bool(opts.get("enable_heterogeneity", False))
    hetero_seed = int(opts.get("hetero_seed", 123))
    hetero_corr_len_m = float(opts.get("hetero_corr_len_m", 0.008))
    stiffness_cv = float(opts.get("stiffness_cv", 0.10))
    angle_hetero_deg = float(opts.get("angle_hetero_deg", 3.5))
    spacing_warp_amp_m = float(opts.get("spacing_warp_amp_m", 0.002))

    use_hertz_contact_width = bool(opts.get("use_hertz_contact_width", True))
    nu_contact = opts.get("nu_contact", None)
    hertz_outer_iters_max = int(opts.get("hertz_outer_iters_max", 6))
    hertz_rel_b_tol = float(opts.get("hertz_rel_b_tol", 2e-3))
    b_contact_init = float(opts.get("b_contact_init", 0.0010))
    platen_mu = float(opts.get("platen_mu", 0.0))

    tol_rel_inner = float(opts.get("tol_rel_inner", 2e-6))
    maxiter_inner = int(opts.get("maxiter_inner", 800))
    tol_rel_final = float(opts.get("tol_rel_final", 1e-7))
    maxiter_final = int(opts.get("maxiter_final", 2500))
    use_precond = bool(opts.get("use_precond", True))
    precond_floor = float(opts.get("precond_floor", 1e-6))
    solver_mode = str(opts.get("solver_mode", "pcg_then_bicgstab"))

    def to_MPa_modulus(val):
        val = float(val)
        return val * 1e3 if val < 1e3 else val

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

    def principal_sigma1_theta(sxx, syy, txy):
        s_avg = 0.5*(sxx + syy)
        rad = np.sqrt((0.5*(sxx - syy))**2 + txy**2)
        s1 = s_avg + rad
        theta = 0.5*np.arctan2(2.0*txy, (sxx - syy))
        return s1, theta

    def harmonic_mix(a, b, w):
        return 1.0 / ((1.0 - w)/(a + 1e-30) + w/(b + 1e-30) + 1e-30)

    def kn_ks_from_min_fractions(E2_base, G12_base, fracE2, fracG, t_band):
        fracE2 = max(float(fracE2), 1e-6)
        fracG  = max(float(fracG), 1e-6)
        kn = (fracE2 * float(E2_base)) / (t_band + 1e-30)
        ks = (fracG  * float(G12_base)) / (t_band + 1e-30)
        return kn, ks

    def band_open_factor_from_sigma_n(sigma_n_mpa, sigma0_mpa):
        s0 = float(max(sigma0_mpa, 1e-9))
        return 0.5 * (1.0 + np.tanh(np.asarray(sigma_n_mpa, float) / s0))

    def sigma_normal_to_plane(sxx, syy, txy, plane_angle):
        a = np.asarray(plane_angle, float)
        nx = -np.sin(a); ny = np.cos(a)
        return sxx*(nx*nx) + 2.0*txy*(nx*ny) + syy*(ny*ny)

    def gaussian_random_field(shape, dx, corr_len_m, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(shape)

        ky = np.fft.fftfreq(shape[0], d=dx) * 2*np.pi
        kx = np.fft.fftfreq(shape[1], d=dx) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX*KX + KY*KY

        L = max(float(corr_len_m), 1e-12)
        H = np.exp(-0.5 * K2 * (L**2))

        f = np.fft.ifft2(np.fft.fft2(w) * H).real
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
                    tt_ = pa / (pa - pb)
                    pts.append(c[a] + tt_*(c[b] - c[a]))

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
            raise RuntimeError("No cut-cell boundary segments found. Increase grid_N.")
        return Iseg, Jseg, xb, yb, nbx, nby, Lseg

    X, Y, mask, R = reconstruct_grid(float(D), int(grid_N))
    h = float(X[0, 1] - X[0, 0])

    if nu_contact is None:
        nu_contact = float(nu12)

    mask_idx = np.flatnonzero(mask.ravel())
    Xf = X.ravel()
    Yf = Y.ravel()

    mask_x = np.zeros_like(mask, dtype=bool)
    mask_x[:, :-1] = mask[:, :-1] & mask[:, 1:]
    mask_y = np.zeros_like(mask, dtype=bool)
    mask_y[:-1, :] = mask[:-1, :] & mask[1:, :]

    E1_base = to_MPa_modulus(E1_in)
    E2_base = to_MPa_modulus(E2_in)
    G12_base = to_MPa_modulus(G_in) if np.isfinite(G_in) else np.sqrt(E1_base * E2_base) / (2.0 * (1.0 + float(nu12)))
    Eeff_MPa = float(np.sqrt(E1_base * E2_base))

    band_angle0 = float(alpha) if (band_angle_rad is None) else float(band_angle_rad)
    phase_plot = 0.5*spacing_m if phase_mode == "center_safe" else float(phase_shift_m)

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

    if band_halfwidth_m > 0:
        w_band = weak_band_weight_phasewarp(
            X, Y, angle_local=angle_local, spacing=spacing_m, warp_m=warp_m,
            band_halfwidth_m=band_halfwidth_m, phase=phase_plot, sharp_power=band_sharp_power
        )
        w_band[~mask] = 0.0
    else:
        w_band = np.zeros_like(X)

    t_band = 2.0 * float(band_halfwidth_m)

    if (joint_kn_MPa_per_m is None) or (joint_ks_MPa_per_m is None):
        if t_band > 0:
            kn0, ks0 = kn_ks_from_min_fractions(
                E2_base, G12_base, band_min_E2_fraction, band_min_G12_fraction, t_band
            )
            if joint_kn_MPa_per_m is None: joint_kn_MPa_per_m = kn0
            if joint_ks_MPa_per_m is None: joint_ks_MPa_per_m = ks0
        else:
            joint_kn_MPa_per_m = 0.0
            joint_ks_MPa_per_m = 0.0

    if t_band > 0:
        E2_band = float(joint_kn_MPa_per_m) * t_band
        G_band  = float(joint_ks_MPa_per_m) * t_band
        E1_band = float(band_min_E1_fraction) * E1_base
    else:
        E2_band = E2_base
        G_band  = G12_base
        E1_band = E1_base

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
    pre_du = pre_dv = None

    def update_Q_from_moduli(E1, E2, G12):
        nonlocal Q11, Q22, Q12, Q16, Q26, Q66, pre_du, pre_dv
        Q11, Q22, Q12, Q16, Q26, Q66 = qbar_from_Es(E1, E2, nu12, G12, theta=float(alpha))
        if use_precond:
            mag = (np.abs(Q11) + np.abs(Q22) + 2*np.abs(Q12) + np.abs(Q66) +
                   np.abs(Q16) + np.abs(Q26))
            du = (mag / (h*h + 1e-30)) + float(precond_floor)
            dv = du.copy()
            du[~mask] = 1.0
            dv[~mask] = 1.0
            pre_du, pre_dv = du, dv
        else:
            pre_du = pre_dv = None

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

    work = {
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

        sxx_fx = work["sxx_fx"]; txy_fx = work["txy_fx"]
        syy_fy = work["syy_fy"]; txy_fy = work["txy_fy"]
        tmpL   = work["tmpL"];   tmpD   = work["tmpD"]
        divx   = work["divx"];   divy   = work["divy"]

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
        if mask_idx.size <= 0:
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
        if mask_idx.size <= 0:
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
        if pre_du is None:
            zx[:] = ax
            zy[:] = ay
        else:
            zx[:] = ax / (pre_du + 1e-30)
            zy[:] = ay / (pre_dv + 1e-30)
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

        for _k in range(maxiter):
            apply_K_inplace(px, py, Ap_x, Ap_y)
            pAp = dot_mask(px, py, Ap_x, Ap_y)
            if (not np.isfinite(pAp)) or (pAp <= 0.0):
                return u, v, False

            a_c = rz_old / (pAp + 1e-30)

            u[mask] += a_c * px[mask]
            v[mask] += a_c * py[mask]
            u, v = project_rigid(u, v)

            rx[mask] -= a_c * Ap_x[mask]
            ry[mask] -= a_c * Ap_y[mask]
            rx[~mask] = 0.0; ry[~mask] = 0.0

            if norm_mask(rx, ry) / r0 < tol_rel:
                return u, v, True

            Minv(rx, ry, zx, zy)
            rz_new = dot_mask(rx, ry, zx, zy)

            b_c = rz_new / (rz_old + 1e-30)
            px[mask] = zx[mask] + b_c * px[mask]
            py[mask] = zy[mask] + b_c * py[mask]
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

        rho_old = 1.0
        alpha_c = 1.0
        omega = 1.0

        px = np.zeros_like(rx); py = np.zeros_like(ry)
        vx = np.zeros_like(rx); vy = np.zeros_like(ry)

        sx = np.zeros_like(rx); sy = np.zeros_like(ry)
        tx = np.zeros_like(rx); ty = np.zeros_like(ry)

        yx = np.zeros_like(rx); yy = np.zeros_like(ry)
        zx = np.zeros_like(rx); zy = np.zeros_like(ry)

        r0 = norm_mask(rx, ry)

        for _k in range(maxiter):
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

        denom = 1e6 * float(t) * float(np.sum(wload * Lseg))
        if denom <= 0:
            raise RuntimeError("Contact width too small -> no segments loaded. Increase b_contact or grid_N.")
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
        return bx_rhs, by_rhs, float(p0_mpa)

    def hertz_contact_halfwidth(P_total_N, thickness_m, R_m, E_eff_MPa, nu):
        Pprime = float(P_total_N) / (float(thickness_m) + 1e-30)
        EeffPa = float(E_eff_MPa) * 1e6
        Eprime = EeffPa / (1.0 - float(nu)**2 + 1e-30)
        b = np.sqrt(4.0 * Pprime * float(R_m) / (np.pi * Eprime + 1e-30))
        return float(b)

    b_contact = float(b_contact_init)
    u_prev = None
    v_prev = None

    final_w_active = w_band.copy()
    final_b_contact = b_contact
    final_p0_unit_mpa = 1.0

    n_outer = int(hertz_outer_iters_max) if use_hertz_contact_width else 1
    for _outer in range(max(n_outer, 1)):
        bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(b_contact)

        w_active = w_band.copy()
        w_prev = None

        n_fp = int(closure_fixed_point_iters_max) if enable_joint_closure else 1
        n_fp = max(n_fp, 1)

        scale_tmp = 1.0
        sign_tmp = 1.0

        for _it in range(n_fp):
            E1, E2, G12 = build_moduli_fields(w_active)
            update_Q_from_moduli(E1, E2, G12)

            u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev, tol_rel=tol_rel_inner, maxiter=maxiter_inner)
            u_prev, v_prev = u, v

            sxx0, syy0, txy0 = stresses(u, v)
            sxx0 = np.where(mask, sxx0, np.nan)
            syy0 = np.where(mask, syy0, np.nan)
            txy0 = np.where(mask, txy0, np.nan)

            ic = (grid_N - 1)//2
            jc = (grid_N - 1)//2
            st = 2
            sxx_center0 = float(np.nanmean(sxx0[ic-st:ic+st+1, jc-st:jc+st+1]))
            if (not np.isfinite(sxx_center0)) or abs(sxx_center0) < 1e-12:
                raise RuntimeError("Center σxx invalid/too small; contact too narrow or solver unstable.")

            sign_tmp = -1.0 if sxx_center0 < 0 else 1.0
            scale_tmp = float(target_sigma_xx_center_mpa) / (abs(sxx_center0) + 1e-30)

            if not enable_joint_closure:
                break

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
        final_b_contact = float(b_contact)
        final_p0_unit_mpa = float(p0_unit_mpa)

        if not use_hertz_contact_width:
            break

        P_calib_N = abs(scale_tmp) * 1.0
        b_old = b_contact
        b_new = hertz_contact_halfwidth(P_calib_N, t, R, Eeff_MPa, nu_contact)
        b_new = float(np.clip(b_new, 0.00015, 0.0060))
        b_contact = 0.55*b_contact + 0.45*b_new

        rel = abs(b_contact - b_old) / (abs(b_old) + 1e-30)
        if rel < hertz_rel_b_tol:
            break

    bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(final_b_contact)
    E1, E2, G12 = build_moduli_fields(final_w_active)
    update_Q_from_moduli(E1, E2, G12)

    u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev, tol_rel=tol_rel_final, maxiter=maxiter_final)

    sxx0, syy0, txy0 = stresses(u, v)
    sxx0 = np.where(mask, sxx0, np.nan)
    syy0 = np.where(mask, syy0, np.nan)
    txy0 = np.where(mask, txy0, np.nan)

    ic = (grid_N - 1)//2
    jc = (grid_N - 1)//2
    st = 2
    sxx_center0 = float(np.nanmean(sxx0[ic-st:ic+st+1, jc-st:jc+st+1]))
    sign = -1.0 if sxx_center0 < 0 else 1.0
    scale = float(target_sigma_xx_center_mpa) / (abs(sxx_center0) + 1e-30)

    sxx = sign * scale * sxx0
    syy = sign * scale * syy0
    txy = sign * scale * txy0

    s1, theta_p = principal_sigma1_theta(sxx, syy, txy)
    sigma1_plus = np.maximum(s1, 0.0)
    sigma1_plus[~mask] = np.nan
    theta_p[~mask] = np.nan

    p_peak_mpa = abs(scale) * float(p0_unit_mpa)
    meta = {
        "Eeff_MPa": float(Eeff_MPa),
        "b_contact_m": float(final_b_contact),
        "p_peak_mpa": float(p_peak_mpa),
        "scale_to_target": float(sign * scale),
        "E1_MPa": float(E1_base),
        "E2_MPa": float(E2_base),
        "G12_MPa": float(G12_base),
    }
    return X, Y, mask, sigma1_plus, theta_p, meta



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global CBAR_RATIO, D, E1_in, E2_in, Eeff_MPa, G_in, HFIG, HSPACE, \
        NCOLS, NROWS, PHYS, R, ROCK_ANISO_RATIO, ROW_HEIGHT_IN, WFIG, \
        WIDTH_PAD_IN, WSPACE, X, Xtmp, Y, Ytmp, _, _fft_cache, \
        all_mag_vals, alpha, anis_ratio, ax, axs, boundary_exclude_frac, c, \
        cache_dir, cache_path, cap_exclude_frac, cax, cbar, cc, color_vmax, \
        color_vmax_percentile, colors, data, density, df, fig, grid_N, gs, \
        i, idx, j, last_mappable, length_exponent, length_ref_percentile, \
        mag, mask, masktmp, max_seg_frac, meta, min_show_frac, norm, nu12, \
        outpath, output_dir, panels, phi_deg, phi_load_rad, r, ref_mag, \
        row, rr, sample_indices, seg_alpha, seg_cmap, seg_linewidth, \
        segments, sigma1_plus, smooth_mag_sigma_pix, \
        smooth_theta_sigma_pix, spacing_m, t, target_sigma_xx_center_mpa, \
        theta_axis, theta_p, tick_params, title, vals
    _bind(rock)
    ROCK_ANISO_RATIO = {
        "augen gneiss": 2.037,
        "psammitic schist": 3.763,
        "psammatic schist": 3.763,   # historical spelling, still accepted
    }
    output_dir = os.path.join(os.getcwd(), output_dirs.FIGURE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dirs.fields(), "_cache_sigma1_theta_physics_v2")
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
    sample_indices = list(_rock().sample_ids)
    target_sigma_xx_center_mpa = 0.63
    spacing_m = _rock().spacing_m
    grid_N = 201
    density = 18
    seg_linewidth = 1.15
    seg_alpha = 0.95
    seg_cmap = "inferno"
    length_ref_percentile = 95.0
    max_seg_frac = 0.22
    length_exponent = 0.85
    min_show_frac = 0.02
    color_vmax_percentile = 99.0
    boundary_exclude_frac = 0.05
    cap_exclude_frac = 0.10
    smooth_theta_sigma_pix = 1.4
    smooth_mag_sigma_pix = 1.0
    NROWS = 4
    NCOLS = 2
    ROW_HEIGHT_IN = 3.20
    WIDTH_PAD_IN = 1.10
    CBAR_RATIO = 0.045
    WSPACE = 0.02
    HSPACE = 0.22
    PHYS = dict(
        band_halfwidth_m=0.00045,
        band_sharp_power=12.0,
        phase_mode="center_safe",
        phase_shift_m=0.0,
        band_angle_rad=None,

        joint_kn_MPa_per_m=None,
        joint_ks_MPa_per_m=None,
        band_min_E2_fraction=0.20,
        band_min_G12_fraction=0.15,
        band_min_E1_fraction=0.95,

        enable_joint_closure=True,
        closure_sigma0_mpa=0.20,
        closed_compliance_fraction=0.08,
        closure_fixed_point_iters_max=3,
        closure_rel_change_tol=2e-3,

        enable_heterogeneity=False,
        hetero_seed=123,
        hetero_corr_len_m=0.008,
        stiffness_cv=0.10,
        angle_hetero_deg=3.5,
        spacing_warp_amp_m= _rock().spacing_warp_amp_m,

        use_hertz_contact_width=True,
        nu_contact=None,
        hertz_outer_iters_max=6,
        hertz_rel_b_tol=2e-3,
        b_contact_init=0.0010,
        platen_mu=0.0,

        tol_rel_inner=2e-6,
        maxiter_inner=800,
        tol_rel_final=1e-7,
        maxiter_final=2500,
        use_precond=True,
        precond_floor=1e-6,
        solver_mode="pcg_then_bicgstab",
    )
    _fft_cache = {}
    panels = []
    all_mag_vals = []
    for idx in sample_indices:
        row = df.loc[idx]
        D = float(row["Diameter_mm"]) * 1e-3
        t = float(row["Thickness_mm"]) * 1e-3

        phi_load_rad = float(row["Radians"])
        alpha = wrap_pi(np.pi/2.0 - phi_load_rad)

        rock = str(row["Rock_type"])
        anis_ratio = get_anisotropy_ratio(rock)
        E1_in = float(row["Modulus_of_Elasticity"])
        E2_in = float(row["Modulus_of_Elasticity"]) / anis_ratio
        G_in = float(row["Shear_Modulus"]) if "Shear_Modulus" in df.columns else np.nan
        nu12 = float(row["Poisson_Ratio"])

        cache_path = os.path.join(
            cache_dir,
            f"physics_v2_idx{idx}_N{grid_N}_sp{spacing_m:.6f}_t{target_sigma_xx_center_mpa:.3f}.npz"
        )

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            sigma1_plus = data["sigma1_plus"]
            theta_p = data["theta_p"]
            Eeff_MPa = float(data["Eeff_MPa"])
        else:
            Xtmp, Ytmp, masktmp, sigma1_plus, theta_p, meta = compute_sigma1plus_and_theta_for_sample(
                D, t,
                E1_in,
                E2_in,
                nu12,
                alpha,
                G_in,
                target_sigma_xx_center_mpa=target_sigma_xx_center_mpa,
                spacing_m=spacing_m,
                grid_N=grid_N,
                **PHYS
            )
            Eeff_MPa = float(meta["Eeff_MPa"])
            np.savez_compressed(
                cache_path,
                sigma1_plus=sigma1_plus,
                theta_p=theta_p,
                Eeff_MPa=Eeff_MPa,
                E1_MPa=float(meta["E1_MPa"]),
                E2_MPa=float(meta["E2_MPa"]),
                G12_MPa=float(meta["G12_MPa"]),
            )

        X, Y, mask, R = reconstruct_grid(D, grid_N)

        sigma1_plus = np.nan_to_num(sigma1_plus, nan=0.0, posinf=0.0, neginf=0.0)
        theta_p = np.nan_to_num(theta_p, nan=0.0, posinf=0.0, neginf=0.0)

        mag = (sigma1_plus / (Eeff_MPa + 1e-30))
        mag[~mask] = np.nan

        if smooth_mag_sigma_pix > 0:
            mag = gaussian_smooth_nan_fft(mag, mask, sigma_pix=smooth_mag_sigma_pix)

        theta_axis = axis_angle(theta_p)
        if smooth_theta_sigma_pix > 0:
            theta_axis = nematic_smooth_theta(theta_axis, mask, sigma_pix=smooth_theta_sigma_pix)

        vals = mag[np.isfinite(mag)]
        vals = vals[vals > 0]
        if vals.size:
            all_mag_vals.append(vals)

        phi_deg = float(row["Angle"]) if "Angle" in row.index else float(np.rad2deg(phi_load_rad))
        title = f"{rock} ({phi_deg:.0f}°)"

        panels.append((title, D, X, Y, mask, theta_axis, mag, R))
    if len(all_mag_vals) == 0:
        raise RuntimeError("No finite magnitudes found for σ1+/Eeff.")
    all_mag_vals = np.concatenate(all_mag_vals)
    ref_mag = float(np.nanpercentile(all_mag_vals, length_ref_percentile))
    ref_mag = max(ref_mag, 1e-30)
    color_vmax = float(np.nanpercentile(all_mag_vals, color_vmax_percentile))
    color_vmax = max(color_vmax, ref_mag)
    norm = Normalize(vmin=0.0, vmax=color_vmax)
    WFIG = float(NCOLS * ROW_HEIGHT_IN + WIDTH_PAD_IN)
    HFIG = float(NROWS * ROW_HEIGHT_IN)
    fig = plt.figure(figsize=(WFIG, HFIG), constrained_layout=False)
    gs = fig.add_gridspec(
        NROWS, 3,
        width_ratios=[1.0, 1.0, CBAR_RATIO],
        left=0.07, right=0.985, bottom=0.06, top=0.94,
        wspace=WSPACE, hspace=HSPACE
    )
    axs = np.empty((NROWS, NCOLS), dtype=object)
    for r in range(NROWS):
        for c in range(NCOLS):
            axs[r, c] = fig.add_subplot(gs[r, c])
    cax = fig.add_subplot(gs[:, 2])
    last_mappable = None
    for i, (title, D, X, Y, mask, theta_axis, mag, R) in enumerate(panels):
        segments, colors, _ = build_segments_from_axis(X, Y, mask, theta_axis, mag, R, ref_mag, color_vmax)
        rr, cc = divmod(i, 2)
        ax = axs[rr, cc]
        last_mappable = plot_panel(ax, segments, colors, R, title, norm)
        compact_axis_labels(ax, rr, cc, NROWS, NCOLS)
    if len(panels) < NROWS * NCOLS:
        for j in range(len(panels), NROWS * NCOLS):
            rr, cc = divmod(j, NCOLS)
            axs[rr, cc].axis("off")
    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, cax=cax)
        cbar.set_label(
            r"tensile strain proxy $\varepsilon_1^{+}$",
            rotation=270, labelpad=20)
        cbar.locator = MaxNLocator(nbins=6)
        cbar.formatter = FormatStrFormatter("%.2e")
        cbar.update_ticks()
    outpath = os.path.join(output_dir, f"{_rock().key}_tensile_strain_proxy.pdf")
    plt.savefig(outpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved: {outpath}")
    print(f"Cache dir: {cache_dir}")
    print(f"Length ref (p{length_ref_percentile:.0f}) = {ref_mag:.3e}")
    print(f"Color vmax (p{color_vmax_percentile:.0f}) = {color_vmax:.3e}")
