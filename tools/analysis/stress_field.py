"""Full-disk stress-distribution panels.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 12 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    band_open_factor_from_sigma_n, get_anisotropy_ratio, get_first_col,
    hertz_contact_halfwidth, sigma1_plus,
)



# --- implementation ---------------------------------------------------


def sanitize_argv(argv):
    out = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "-f":
            skip_next = True
            continue
        if a.startswith("--f=") or a.startswith("-f="):
            continue
        out.append(a)
    return out


def wrap_pi(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi_scalar(a: float) -> float:
    return float(((float(a) + np.pi) % (2.0 * np.pi)) - np.pi)


def wrap_pi_half(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def wrap_pi_half_scalar(a: float) -> float:
    return float(((float(a) + np.pi / 2.0) % np.pi) - np.pi / 2.0)


def map_angle_to_alpha_deg(angle_deg, angle_map="direct"):
    ang = np.deg2rad(float(angle_deg))
    mode = str(angle_map).strip().lower()
    if mode == "direct":
        return wrap_pi_scalar(ang)
    if mode == "neg":
        return wrap_pi_scalar(-ang)
    if mode == "pi2_minus":
        return wrap_pi_scalar(np.pi / 2.0 - ang)
    if mode == "pi2_plus":
        return wrap_pi_scalar(np.pi / 2.0 + ang)
    raise ValueError(f"Unknown angle_map: {angle_map}")


def parse_sample_ids(spec, valid_index):
    wanted = set()
    if spec and isinstance(spec, str):
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    a = int(a)
                    b = int(b)
                    wanted.update(range(min(a, b), max(a, b) + 1))
                except Exception:
                    pass
            else:
                try:
                    wanted.add(int(tok))
                except Exception:
                    pass
    return [i for i in valid_index if i in wanted]


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


def to_MPa_modulus(val):
    val = float(val)
    return val * 1e3 if val < 1e3 else val


def principal_stresses_2d(sxx, syy, txy):
    s_avg = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5 * (sxx - syy)) ** 2 + txy ** 2)
    return s_avg + rad, s_avg - rad


def harmonic_mix(a, b, w):
    return 1.0 / ((1.0 - w) / (a + 1e-30) + w / (b + 1e-30) + 1e-30)


def kn_ks_from_min_fractions(E2_base, G12_base, fracE2, fracG, t_band):
    fracE2 = max(float(fracE2), 1e-6)
    fracG = max(float(fracG), 1e-6)
    kn = (fracE2 * float(E2_base)) / (t_band + 1e-30)
    ks = (fracG * float(G12_base)) / (t_band + 1e-30)
    return kn, ks


def sigma_normal_to_plane(sxx, syy, txy, plane_angle):
    a = np.asarray(plane_angle, float)
    nx = -np.sin(a)
    ny = np.cos(a)
    return sxx * (nx * nx) + 2.0 * txy * (nx * ny) + syy * (ny * ny)


def qbar_from_Es(E1, E2, nu12, G12, theta):
    E1 = np.asarray(E1, float)
    E2 = np.asarray(E2, float)
    G12 = np.asarray(G12, float)
    nu12 = float(nu12)

    nu21 = nu12 * (E2 / (E1 + 1e-30))
    den = 1.0 - nu12 * nu21

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    m = np.cos(theta)
    n = np.sin(theta)
    m2 = m * m
    n2 = n * n
    m3 = m2 * m
    n3 = n2 * n
    m4 = m2 * m2
    n4 = n2 * n2

    Qbar11 = Q11 * m4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * n4
    Qbar22 = Q11 * n4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * m4
    Qbar12 = (Q11 + Q22 - 4.0 * Q66) * m2 * n2 + Q12 * (m4 + n4)
    Qbar16 = (Q11 - Q12 - 2.0 * Q66) * m3 * n - (Q22 - Q12 - 2.0 * Q66) * m * n3
    Qbar26 = (Q11 - Q12 - 2.0 * Q66) * m * n3 - (Q22 - Q12 - 2.0 * Q66) * m3 * n
    Qbar66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * m2 * n2 + Q66 * (m4 + n4)
    return Qbar11, Qbar22, Qbar12, Qbar16, Qbar26, Qbar66


def gaussian_random_field(shape, dx, corr_len_m, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)

    ky = np.fft.fftfreq(shape[0], d=dx) * 2 * np.pi
    kx = np.fft.fftfreq(shape[1], d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX * KX + KY * KY

    L = max(float(corr_len_m), 1e-12)
    H = np.exp(-0.5 * K2 * (L ** 2))

    W = np.fft.fft2(w)
    f = np.fft.ifft2(W * H).real
    f -= np.mean(f)
    s = np.std(f)
    if s > 1e-12:
        f /= s
    return f


def weak_band_weight_phasewarp(X, Y, angle_local, spacing, warp_m,
                               band_halfwidth_m, phase, sharp_power):
    bw = float(band_halfwidth_m)
    s = float(spacing)
    a = np.asarray(angle_local, float)
    warp = np.asarray(warp_m, float)

    nx = -np.sin(a)
    ny = np.cos(a)

    d = nx * X + ny * Y + float(phase)
    d_eff = d + warp
    dist_eff = (s / np.pi) * np.abs(np.sin(np.pi * d_eff / (s + 1e-30)))
    w = np.exp(-sharp_power * (dist_eff / (bw + 1e-30)) ** 2)
    return np.clip(w, 0.0, 1.0)


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
        ky = np.fft.fftfreq(H, d=1.0) * 2 * np.pi
        kx = np.fft.fftfreq(W, d=1.0) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX * KX + KY * KY
        Gk = np.exp(-0.5 * (sigma_pix ** 2) * K2)
        _fft_cache[key] = Gk

    f0 = np.where(mask, field, 0.0)
    m0 = mask.astype(float)

    num = np.fft.ifft2(np.fft.fft2(f0) * Gk).real
    den = np.fft.ifft2(np.fft.fft2(m0) * Gk).real
    out = num / (den + 1e-30)
    out[~mask] = np.nan
    return out


def add_band_lines(ax, R, band_angle, spacing, phase, halfwidth=None):
    a = float(band_angle)
    s = float(spacing)
    ph = float(phase)
    n = np.array([-np.sin(a), np.cos(a)], float)
    t = np.array([np.cos(a), np.sin(a)], float)

    def draw(offset, lw, ls, alpha=0.25):
        kmin = int(np.ceil((ph - R - offset) / s))
        kmax = int(np.floor((ph + R - offset) / s))
        for k in range(kmin, kmax + 1):
            d0 = (k * s - ph) + offset
            if abs(d0) > R:
                continue
            x0 = d0 * n
            L = np.sqrt(max(R * R - d0 * d0, 0.0))
            p1 = x0 - L * t
            p2 = x0 + L * t
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k", lw=lw, ls=ls, alpha=alpha, zorder=5)

    draw(0.0, 1.0, "-")
    if halfwidth is not None and halfwidth > 0:
        draw(+halfwidth, 0.8, "--", alpha=0.18)
        draw(-halfwidth, 0.8, "--", alpha=0.18)


def compute_combined_field_for_sample(
    diameter_m,
    thickness_m,
    E1_in,
    E2_in,
    nu12,
    alpha_in_rad,
    *,
    G_in=np.nan,
    target_sigma_xx_center_mpa=0.63,
    spacing_m=None,
    grid_N=401,
    band_halfwidth_m=0.00045,
    band_sharp_power=12.0,
    phase_mode="center_safe",
    phase_shift_m=0.0,
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
    enable_heterogeneity=True,
    hetero_seed=123,
    hetero_corr_len_m=0.008,
    stiffness_cv=0.10,
    angle_hetero_deg=3.5,
    spacing_warp_amp_m=None,
    use_hertz_contact_width=True,
    nu_contact=None,
    hertz_outer_iters_max=6,
    hertz_rel_b_tol=2e-3,
    platen_mu=0.0,
    tol_rel_inner=2e-6,
    maxiter_inner=800,
    tol_rel_final=1e-7,
    maxiter_final=2500,
    use_precond=True,
    precond_floor=1e-6,
    solver_mode="pcg_then_bicgstab",
):
    if spacing_m is None:
        spacing_m = _rock().spacing_m
    if spacing_warp_amp_m is None:
        spacing_warp_amp_m = _rock().spacing_warp_amp_m
    def ddx(f, mask, h):
        out = np.zeros_like(f)
        central = mask[:, 1:-1] & mask[:, 0:-2] & mask[:, 2:]
        forward = mask[:, 1:-1] & (~mask[:, 0:-2]) & mask[:, 2:]
        backward = mask[:, 1:-1] & mask[:, 0:-2] & (~mask[:, 2:])
        core = out[:, 1:-1]
        core[:] = 0.0
        core += central * ((f[:, 2:] - f[:, 0:-2]) / (2 * h))
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
        core += central * ((f[2:, :] - f[0:-2, :]) / (2 * h))
        core += forward * ((f[2:, :] - f[1:-1, :]) / h)
        core += backward * ((f[1:-1, :] - f[0:-2, :]) / h)
        eB = mask[0, :] & mask[1, :]
        eT = mask[-1, :] & mask[-2, :]
        out[0, eB] = (f[1, eB] - f[0, eB]) / h
        out[-1, eT] = (f[-1, eT] - f[-2, eT]) / h
        out[~mask] = 0.0
        return out

    def boundary_segments_cut_cells(X, Y, mask, R, h):
        mR = np.zeros_like(mask)
        mR[:, :-1] = mask[:, 1:]
        mL = np.zeros_like(mask)
        mL[:, 1:] = mask[:, :-1]
        mU = np.zeros_like(mask)
        mU[:-1, :] = mask[1:, :]
        mD = np.zeros_like(mask)
        mD[1:, :] = mask[:-1, :]

        boundary = mask & (~(mR & mL & mU & mD))
        I0, J0 = np.where(boundary)

        Iseg, Jseg = [], []
        xm_list, ym_list, nx_list, ny_list, L_list = [], [], [], [], []

        def phi_circle(x, y):
            return x * x + y * y - R * R

        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for i, j in zip(I0, J0):
            xc = float(X[i, j])
            yc = float(Y[i, j])
            c = np.array([
                [xc - 0.5 * h, yc - 0.5 * h],
                [xc + 0.5 * h, yc - 0.5 * h],
                [xc + 0.5 * h, yc + 0.5 * h],
                [xc - 0.5 * h, yc + 0.5 * h],
            ], float)

            phi = phi_circle(c[:, 0], c[:, 1])
            pts = []
            for a, b in edges:
                pa = phi[a]
                pb = phi[b]
                if (pa <= 0 and pb >= 0) or (pa >= 0 and pb <= 0):
                    denom = pa - pb
                    if abs(denom) < 1e-30:
                        continue
                    t = pa / (pa - pb)
                    pts.append(c[a] + t * (c[b] - c[a]))

            if len(pts) < 2:
                continue

            pts = np.array(pts, float)
            if len(pts) > 2:
                dmax = -1.0
                p1 = pts[0]
                p2 = pts[1]
                for a in range(len(pts)):
                    for b in range(a + 1, len(pts)):
                        d = float(np.sum((pts[a] - pts[b]) ** 2))
                        if d > dmax:
                            dmax = d
                            p1 = pts[a]
                            p2 = pts[b]
            else:
                p1, p2 = pts[0], pts[1]

            th1 = float(np.arctan2(p1[1], p1[0]))
            th2 = float(np.arctan2(p2[1], p2[0]))
            dth = (th2 - th1) % (2 * np.pi)
            dth = min(dth, 2 * np.pi - dth)
            L = R * dth
            if (not np.isfinite(L)) or (L <= 0):
                continue

            u1 = np.array([np.cos(th1), np.sin(th1)])
            u2 = np.array([np.cos(th2), np.sin(th2)])
            um = u1 + u2
            if np.hypot(um[0], um[1]) < 1e-12:
                pm = 0.5 * (p1 + p2)
                thm = float(np.arctan2(pm[1], pm[0]))
            else:
                thm = float(np.arctan2(um[1], um[0]))

            xb = R * np.cos(thm)
            yb = R * np.sin(thm)
            nx = xb / (R + 1e-30)
            ny = yb / (R + 1e-30)

            Iseg.append(int(i))
            Jseg.append(int(j))
            xm_list.append(xb)
            ym_list.append(yb)
            nx_list.append(nx)
            ny_list.append(ny)
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

    diameter = float(diameter_m)
    thickness = float(thickness_m)
    R = diameter / 2.0

    E1_base = to_MPa_modulus(E1_in)
    E2_base = to_MPa_modulus(E2_in)
    G12_base = to_MPa_modulus(G_in) if np.isfinite(G_in) else np.sqrt(E1_base * E2_base) / (2.0 * (1.0 + float(nu12)))
    nu12_loc = float(nu12)

    xlin = np.linspace(-R, R, int(grid_N))
    ylin = np.linspace(-R, R, int(grid_N))
    h = float(xlin[1] - xlin[0])
    X, Y = np.meshgrid(xlin, ylin)
    mask = (X * X + Y * Y) <= R * R

    mask_idx = np.flatnonzero(mask.ravel())
    Xf = X.ravel()
    Yf = Y.ravel()

    mask_x = np.zeros_like(mask, dtype=bool)
    mask_x[:, :-1] = mask[:, :-1] & mask[:, 1:]
    mask_y = np.zeros_like(mask, dtype=bool)
    mask_y[:-1, :] = mask[:-1, :] & mask[1:, :]

    phase_plot = 0.5 * spacing_m if str(phase_mode) == "center_safe" else float(phase_shift_m)
    alpha = float(alpha_in_rad)
    band_angle0 = wrap_pi_half_scalar(alpha)

    if enable_heterogeneity:
        fE = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=int(hetero_seed) + 1)
        fA = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=int(hetero_seed) + 2)
        fW = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=int(hetero_seed) + 3)

        sigma_ln = np.sqrt(np.log(1.0 + float(stiffness_cv) ** 2)) if stiffness_cv > 0 else 0.0
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
        X, Y,
        angle_local=angle_local,
        spacing=spacing_m,
        warp_m=warp_m,
        band_halfwidth_m=band_halfwidth_m,
        phase=phase_plot,
        sharp_power=band_sharp_power,
    )
    w_band[~mask] = 0.0
    t_band = 2.0 * float(band_halfwidth_m)

    local_joint_kn = joint_kn_MPa_per_m
    local_joint_ks = joint_ks_MPa_per_m
    if local_joint_kn is None or local_joint_ks is None:
        kn0, ks0 = kn_ks_from_min_fractions(
            E2_base, G12_base, band_min_E2_fraction, band_min_G12_fraction, t_band
        )
        local_joint_kn = kn0 if local_joint_kn is None else local_joint_kn
        local_joint_ks = ks0 if local_joint_ks is None else local_joint_ks

    E2_band = local_joint_kn * t_band
    G_band = local_joint_ks * t_band
    E1_band = band_min_E1_fraction * E1_base

    def build_moduli_fields(w_active):
        w = np.clip(w_active, 0.0, 1.0)
        E1_intact = E1_base * stiff_factor
        E2_intact = E2_base * stiff_factor
        G12_intact = G12_base * stiff_factor

        E1_weak = E1_band * np.ones_like(E1_intact)
        E2_weak = E2_band * np.ones_like(E2_intact)
        G12_weak = G_band * np.ones_like(G12_intact)

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
            mag = (np.abs(Q11) + np.abs(Q22) + 2 * np.abs(Q12) + np.abs(Q66) +
                   np.abs(Q16) + np.abs(Q26))
            du = (mag / (h * h + 1e-30)) + float(precond_floor)
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
        sxx = Q11 * exx + Q12 * eyy + Q16 * gxy
        syy = Q12 * exx + Q22 * eyy + Q26 * gxy
        txy = Q16 * exx + Q26 * eyy + Q66 * gxy
        sxx[~mask] = 0.0
        syy[~mask] = 0.0
        txy[~mask] = 0.0
        return sxx, syy, txy

    _work = {
        "sxx_fx": np.zeros_like(X),
        "txy_fx": np.zeros_like(X),
        "syy_fy": np.zeros_like(X),
        "txy_fy": np.zeros_like(X),
        "tmpL": np.zeros_like(X),
        "tmpD": np.zeros_like(X),
        "divx": np.zeros_like(X),
        "divy": np.zeros_like(X),
    }

    def apply_K_inplace(u, v, fx_out, fy_out):
        sxx, syy, txy = stresses(u, v)

        sxx_fx = _work["sxx_fx"]
        txy_fx = _work["txy_fx"]
        syy_fy = _work["syy_fy"]
        txy_fy = _work["txy_fy"]
        tmpL = _work["tmpL"]
        tmpD = _work["tmpD"]
        divx = _work["divx"]
        divy = _work["divy"]

        sxx_fx.fill(0.0)
        txy_fx.fill(0.0)
        syy_fy.fill(0.0)
        txy_fy.fill(0.0)

        sxx_fx[:, :-1] = 0.5 * (sxx[:, :-1] + sxx[:, 1:]) * mask_x[:, :-1]
        txy_fx[:, :-1] = 0.5 * (txy[:, :-1] + txy[:, 1:]) * mask_x[:, :-1]
        syy_fy[:-1, :] = 0.5 * (syy[:-1, :] + syy[1:, :]) * mask_y[:-1, :]
        txy_fy[:-1, :] = 0.5 * (txy[:-1, :] + txy[1:, :]) * mask_y[:-1, :]

        tmpL.fill(0.0)
        tmpL[:, 1:] = sxx_fx[:, :-1]
        tmpD.fill(0.0)
        tmpD[1:, :] = txy_fy[:-1, :]
        divx[:] = (sxx_fx - tmpL + txy_fy - tmpD) / h

        tmpL.fill(0.0)
        tmpL[:, 1:] = txy_fx[:, :-1]
        tmpD.fill(0.0)
        tmpD[1:, :] = syy_fy[:-1, :]
        divy[:] = (txy_fx - tmpL + syy_fy - tmpD) / h

        fx_out[:] = -divx
        fy_out[:] = -divy
        fx_out[~mask] = 0.0
        fy_out[~mask] = 0.0

    def dot_mask(ax, ay, bx, by):
        axf = ax.ravel()
        ayf = ay.ravel()
        bxf = bx.ravel()
        byf = by.ravel()
        return float(np.dot(axf[mask_idx], bxf[mask_idx]) + np.dot(ayf[mask_idx], byf[mask_idx]))

    def norm_mask(ax, ay):
        return float(np.sqrt(dot_mask(ax, ay, ax, ay))) + 1e-30

    def project_rigid(u, v):
        uf = u.ravel()
        vf = v.ravel()
        n = mask_idx.size
        if n <= 0:
            return u, v
        um = float(np.mean(uf[mask_idx]))
        vm = float(np.mean(vf[mask_idx]))
        uf[mask_idx] -= um
        vf[mask_idx] -= vm
        denom = float(np.sum((Xf[mask_idx] ** 2 + Yf[mask_idx] ** 2))) + 1e-30
        omega = float(np.sum(Xf[mask_idx] * vf[mask_idx] - Yf[mask_idx] * uf[mask_idx]) / denom)
        uf[mask_idx] += omega * Yf[mask_idx]
        vf[mask_idx] -= omega * Xf[mask_idx]
        return u, v

    def project_force_moment(bx, by):
        bxf = bx.ravel()
        byf = by.ravel()
        A = mask_idx.size
        if A <= 0:
            return bx, by
        bx0 = float(np.mean(bxf[mask_idx]))
        by0 = float(np.mean(byf[mask_idx]))
        bxf[mask_idx] -= bx0
        byf[mask_idx] -= by0
        denom = float(np.sum((Xf[mask_idx] ** 2 + Yf[mask_idx] ** 2))) + 1e-30
        moment = float(np.sum(Xf[mask_idx] * byf[mask_idx] - Yf[mask_idx] * bxf[mask_idx]))
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
            u = np.zeros_like(bx)
            v = np.zeros_like(by)
        else:
            u = u0.copy()
            v = v0.copy()

        u, v = project_rigid(u, v)

        Ax = np.zeros_like(bx)
        Ay = np.zeros_like(by)
        apply_K_inplace(u, v, Ax, Ay)

        rx = bx - Ax
        ry = by - Ay
        rx[~mask] = 0.0
        ry[~mask] = 0.0

        r0 = norm_mask(rx, ry)
        if r0 < 1e-30:
            return u, v, True

        zx = np.zeros_like(rx)
        zy = np.zeros_like(ry)
        Minv(rx, ry, zx, zy)

        px = zx.copy()
        py = zy.copy()
        rz_old = dot_mask(rx, ry, zx, zy)

        Ap_x = np.zeros_like(rx)
        Ap_y = np.zeros_like(ry)

        for _ in range(int(maxiter)):
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
            rx[~mask] = 0.0
            ry[~mask] = 0.0

            if norm_mask(rx, ry) / r0 < tol_rel:
                return u, v, True

            Minv(rx, ry, zx, zy)
            rz_new = dot_mask(rx, ry, zx, zy)

            beta = rz_new / (rz_old + 1e-30)
            px[mask] = zx[mask] + beta * px[mask]
            py[mask] = zy[mask] + beta * py[mask]
            px[~mask] = 0.0
            py[~mask] = 0.0
            rz_old = rz_new

        return u, v, True

    def solve_bicgstab(bx, by, u0=None, v0=None, maxiter=2500, tol_rel=1e-7):
        if u0 is None:
            u = np.zeros_like(bx)
            v = np.zeros_like(by)
        else:
            u = u0.copy()
            v = v0.copy()

        u, v = project_rigid(u, v)

        Ku = np.zeros_like(bx)
        Kv = np.zeros_like(by)
        apply_K_inplace(u, v, Ku, Kv)

        rx = bx - Ku
        ry = by - Kv
        rx[~mask] = 0.0
        ry[~mask] = 0.0

        rhatx = rx.copy()
        rhaty = ry.copy()

        rho_old = 1.0
        alpha_c = 1.0
        omega = 1.0

        px = np.zeros_like(rx)
        py = np.zeros_like(ry)
        vx = np.zeros_like(rx)
        vy = np.zeros_like(ry)

        sx = np.zeros_like(rx)
        sy = np.zeros_like(ry)
        tx = np.zeros_like(rx)
        ty = np.zeros_like(ry)

        yx = np.zeros_like(rx)
        yy = np.zeros_like(ry)
        zx = np.zeros_like(rx)
        zy = np.zeros_like(ry)

        r0 = norm_mask(rx, ry)

        for _ in range(int(maxiter)):
            rho_new = dot_mask(rhatx, rhaty, rx, ry)
            if abs(rho_new) < 1e-30:
                break

            beta_c = (rho_new / rho_old) * (alpha_c / omega)
            px[mask] = rx[mask] + beta_c * (px[mask] - omega * vx[mask])
            py[mask] = ry[mask] + beta_c * (py[mask] - omega * vy[mask])
            px[~mask] = 0.0
            py[~mask] = 0.0

            Minv(px, py, yx, yy)
            apply_K_inplace(yx, yy, vx, vy)

            denom = dot_mask(rhatx, rhaty, vx, vy) + 1e-30
            alpha_c = rho_new / denom

            sx[:] = rx - alpha_c * vx
            sy[:] = ry - alpha_c * vy
            sx[~mask] = 0.0
            sy[~mask] = 0.0

            if norm_mask(sx, sy) / r0 < tol_rel:
                u[mask] += alpha_c * yx[mask]
                v[mask] += alpha_c * yy[mask]
                u, v = project_rigid(u, v)
                return u, v

            Minv(sx, sy, zx, zy)
            apply_K_inplace(zx, zy, tx, ty)

            tt = dot_mask(tx, ty, tx, ty) + 1e-30
            omega = dot_mask(tx, ty, sx, sy) / tt

            u[mask] += alpha_c * yx[mask] + omega * zx[mask]
            v[mask] += alpha_c * yy[mask] + omega * zy[mask]
            u, v = project_rigid(u, v)

            rx[:] = sx - omega * tx
            ry[:] = sy - omega * ty
            rx[~mask] = 0.0
            ry[~mask] = 0.0

            if norm_mask(rx, ry) / r0 < tol_rel:
                return u, v

            rho_old = rho_new

        return u, v

    def solve_system(bx, by, u0, v0, tol_rel, maxiter):
        if str(solver_mode) == "pcg_then_bicgstab":
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
        s_top = R * np.abs(wrap_pi(th - np.pi / 2))
        s_bot = R * np.abs(wrap_pi(th + np.pi / 2))

        b = float(max(b_contact_m, 1e-6))
        wtop = np.sqrt(np.maximum(1.0 - (s_top / b) ** 2, 0.0))
        wbot = np.sqrt(np.maximum(1.0 - (s_bot / b) ** 2, 0.0))
        wload = wtop + wbot

        denom = 1e6 * thickness * float(np.sum(wload * Lseg))
        if denom <= 0:
            raise RuntimeError("Contact width too small -> no loaded boundary segments. Increase b_contact_m or grid_N.")
        p0_mpa = 1.0 / denom

        p = p0_mpa * wload
        tx = -p * nbx
        ty = -p * nby

        if platen_mu != 0.0:
            tx_t = -nby
            ty_t = nbx
            tau = float(platen_mu) * p
            tx += -tau * tx_t
            ty += -tau * ty_t

        wfac = (Lseg / (h * h))
        np.add.at(bx_rhs, (Iseg, Jseg), tx * wfac)
        np.add.at(by_rhs, (Iseg, Jseg), ty * wfac)

        bx_rhs[~mask] = 0.0
        by_rhs[~mask] = 0.0
        bx_rhs, by_rhs = project_force_moment(bx_rhs, by_rhs)
        return bx_rhs, by_rhs, p0_mpa

    def hertz_contact_halfwidth(P_total_N, thickness_m, R_m, E_eff_MPa, nu):
        Pprime = float(P_total_N) / (float(thickness_m) + 1e-30)
        EeffPa = float(E_eff_MPa) * 1e6
        Eprime = EeffPa / (1.0 - float(nu) ** 2 + 1e-30)
        b = np.sqrt(4.0 * Pprime * float(R_m) / (np.pi * Eprime + 1e-30))
        return float(b)

    def robust_center_value(field, mask, ic, jc):
        for st in (2, 3, 4, 6, 8, 10):
            i0 = max(0, ic - st)
            i1 = min(field.shape[0], ic + st + 1)
            j0 = max(0, jc - st)
            j1 = min(field.shape[1], jc + st + 1)

            sub = np.asarray(field[i0:i1, j0:j1], float)
            msub = np.asarray(mask[i0:i1, j0:j1], bool) & np.isfinite(sub)
            if not np.any(msub):
                continue

            vals = sub[msub]
            meanv = float(np.nanmean(vals))
            if np.isfinite(meanv) and abs(meanv) >= 1e-12:
                return meanv

            vmax = float(vals[np.argmax(np.abs(vals))])
            if np.isfinite(vmax) and abs(vmax) >= 1e-12:
                return vmax

        return np.nan

    local_nu_contact = float(nu12_loc) if (nu_contact is None) else float(nu_contact)
    E_eff_MPa = float(np.sqrt(E1_base * E2_base))
    b_contact = max(0.0010, 4.0 * h)

    u_prev = None
    v_prev = None

    final_w_active = w_band.copy()
    final_b_contact = b_contact
    final_p0_unit_mpa = 1.0

    n_outer = int(hertz_outer_iters_max) if use_hertz_contact_width else 1
    max_contact_retries = max(n_outer, 6)
    scale_tmp = 1.0

    outer_iter = 0
    while outer_iter < max_contact_retries:
        try:
            bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(b_contact)
        except RuntimeError:
            b_contact = float(np.clip(max(1.35 * b_contact, 4.0 * h), 0.00015, 0.0060))
            outer_iter += 1
            continue

        w_active = w_band.copy()
        w_prev = None
        n_fp = int(closure_fixed_point_iters_max) if enable_joint_closure else 1
        n_fp = max(n_fp, 1)
        center_ok = False

        for _ in range(n_fp):
            E1, E2, G12 = build_moduli_fields(w_active)
            update_Q_from_moduli(E1, E2, G12)

            u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev,
                                tol_rel=tol_rel_inner, maxiter=maxiter_inner)
            u_prev, v_prev = u, v

            sxx0, syy0, txy0 = stresses(u, v)
            sxx0 = np.where(mask, sxx0, np.nan)
            syy0 = np.where(mask, syy0, np.nan)
            txy0 = np.where(mask, txy0, np.nan)

            ic = (grid_N - 1) // 2
            jc = (grid_N - 1) // 2
            sxx_center0 = robust_center_value(sxx0, mask, ic, jc)
            if (not np.isfinite(sxx_center0)) or abs(sxx_center0) < 1e-12:
                center_ok = False
                break

            center_ok = True
            sign_tmp = -1.0 if sxx_center0 < 0 else 1.0
            scale_tmp = float(target_sigma_xx_center_mpa) / abs(sxx_center0)

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

        if not center_ok:
            b_contact = float(np.clip(max(1.35 * b_contact, 4.0 * h), 0.00015, 0.0060))
            outer_iter += 1
            continue

        final_w_active = w_active.copy()
        final_b_contact = b_contact
        final_p0_unit_mpa = float(p0_unit_mpa)

        if not use_hertz_contact_width:
            break

        P_calib_N = abs(scale_tmp) * 1.0
        b_old = b_contact
        b_new = hertz_contact_halfwidth(P_calib_N, thickness, R, E_eff_MPa, local_nu_contact)
        b_new = float(np.clip(b_new, 0.00015, 0.0060))
        b_contact = 0.55 * b_contact + 0.45 * b_new

        rel = abs(b_contact - b_old) / (abs(b_old) + 1e-30)
        outer_iter += 1
        if rel < hertz_rel_b_tol:
            break

    bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(final_b_contact)
    E1, E2, G12 = build_moduli_fields(final_w_active)
    update_Q_from_moduli(E1, E2, G12)

    u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev,
                        tol_rel=tol_rel_final, maxiter=maxiter_final)

    sxx0, syy0, txy0 = stresses(u, v)
    sxx0 = np.where(mask, sxx0, np.nan)
    syy0 = np.where(mask, syy0, np.nan)
    txy0 = np.where(mask, txy0, np.nan)

    ic = (grid_N - 1) // 2
    jc = (grid_N - 1) // 2
    sxx_center0 = robust_center_value(sxx0, mask, ic, jc)
    if (not np.isfinite(sxx_center0)) or abs(sxx_center0) < 1e-12:
        ref_vals = np.abs(sxx0[mask])
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        ref = float(np.nanpercentile(ref_vals, 95)) if ref_vals.size else 1.0
        ref = max(ref, 1e-12)
        sign = 1.0
        scale = float(target_sigma_xx_center_mpa) / ref
    else:
        sign = -1.0 if sxx_center0 < 0 else 1.0
        scale = float(target_sigma_xx_center_mpa) / abs(sxx_center0)

    sxx = sign * scale * sxx0
    syy = sign * scale * syy0
    txy = sign * scale * txy0

    s1, s3 = principal_stresses_2d(sxx, syy, txy)
    sigma1_plus = np.maximum(s1, 0.0)
    syy_minus = np.minimum(syy, 0.0)

    p_peak_mpa = abs(scale) * float(p0_unit_mpa)
    w_comp = float(target_sigma_xx_center_mpa) / (p_peak_mpa + 1e-30)
    w_comp = float(np.clip(w_comp, 0.0, 1.0))

    syy_cap = np.maximum(syy_minus, -p_peak_mpa)
    combined = sigma1_plus + w_comp * syy_cap
    combined = np.where(mask, combined, np.nan)

    meta = {
        "R": R,
        "phase_plot": phase_plot,
        "band_angle0": band_angle0,
        "p_peak_mpa": p_peak_mpa,
        "w_comp": w_comp,
        "b_contact_m": final_b_contact,
        "alpha_rad": alpha,
        "E1_base": E1_base,
        "E2_base": E2_base,
        "G12_base": G12_base,
    }

    fields = {
        "X": X,
        "Y": Y,
        "mask": mask,
        "combined": combined,
        "sxx": sxx,
        "syy": syy,
        "txy": txy,
        "s1": s1,
        "s3": s3,
        "w_band": w_band,
        "w_active": final_w_active,
        "angle_local": angle_local,
        "warp_m": warp_m,
    }
    return combined, mask, xlin, ylin, meta, fields


def plot_combined_on_axis(ax, combined, mask, xlin, ylin, title, meta, norm,
                          *, plot_interpolation="bicubic",
                          contour_levels=16,
                          contour_sigma_pix=1.35,
                          contour_corner_mask=True,
                          contour_linewidth=0.55,
                          contour_alpha=0.60,
                          show_band_lines=True,
                          show_band_edges=True,
                          spacing_m=None,
                          band_halfwidth_m=0.00045):
    if spacing_m is None:
        spacing_m = _rock().spacing_m
    R = float(meta["R"])
    extent = [xlin[0], xlin[-1], ylin[0], ylin[-1]]
    Xp, Yp = np.meshgrid(xlin, ylin)

    f = np.ma.array(combined, mask=~mask)
    im = ax.imshow(
        f,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=norm,
        interpolation=str(plot_interpolation),
        alpha=0.95,
    )

    fs = gaussian_smooth_nan_fft(combined, mask, sigma_pix=contour_sigma_pix)
    vmin, vmax = norm.vmin, norm.vmax
    try:
        ax.contour(
            Xp,
            Yp,
            fs,
            levels=np.linspace(vmin, vmax, int(contour_levels)),
            colors="k",
            linewidths=float(contour_linewidth),
            alpha=float(contour_alpha),
            corner_mask=bool(contour_corner_mask),
            zorder=4,
        )
    except Exception:
        pass

    if show_band_lines:
        add_band_lines(
            ax,
            R,
            band_angle=meta["band_angle0"],
            spacing=spacing_m,
            phase=meta["phase_plot"],
            halfwidth=(band_halfwidth_m if show_band_edges else None),
        )

    ax.add_patch(plt.Circle((0, 0), R, fill=False, linewidth=1.2, color="k", zorder=6))
    ax.set_title(title, pad=6)
    ax.set_xlim(-R * 1.05, R * 1.05)
    ax.set_ylim(-R * 1.05, R * 1.05)
    ax.set_aspect("equal", "box")
    ax.minorticks_on()
    return im



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global D, E1_in, E2_in, G_in, HFIG, HROW, ROCK_ANISO_RATIO, WFIG, \
        _fft_cache, _unknown, abs_lim, all_abs, alpha, ang_deg, anis_ratio, \
        args, argv, ax, axs, c, cax, cb, cc, coli, combined, computed, df, \
        field_dir, fields, fig, global_abs, gs, joint_kn, joint_ks, k, \
        last_mappable, mask, meta, n, ncols, norm, nrows, nu12, nu_contact, \
        out_pdf, p, payload, r, row, rowi, rr, samples, sid, slots, \
        stats_path, stats_rows, t, title, vals, xlin, ylin
    _bind(rock)
    """
    Brazilian disk — publication-style orthotropic combined-stress field plots
    aligned to the conventions used in the reference DDM crack-growth script.

    Main alignments to the first script:
      - argparse + Jupyter-safe argv handling
      - sample selection via --sample_ids (e.g., 1-7 or 1,3,5)
      - angle mapping via --angle_map with the same interface style
      - compact 2-column journal layout with square panels
      - outer-axis labeling only (prevents overlap)
      - thin dedicated shared colorbar column
      - per-sample NPZ export + summary CSV

    Physics/solver core remains the orthotropic combined-field model from the
    second script, but the layout, sample handling, export structure, and angle
    conventions are revised to better match the first script.
    """
    ROCK_ANISO_RATIO = {
        "augen gneiss": 2.037,
        "psammitic schist": 3.763,
        "psammatic schist": 3.763,   # historical spelling, still accepted
    }
    _fft_cache = {}
    p = argparse.ArgumentParser(
        "Brazilian disk — publication-style orthotropic combined field",
        allow_abbrev=False,
    )

    p.add_argument("--meta_csv", default="tensile_samples_data.csv")
    p.add_argument("--out_dir", default=output_dirs.FIGURE_DIR)
    p.add_argument("--sample_ids", type=str, default=_rock().sample_id_spec())
    p.add_argument("--angle_map", type=str, default="direct",
                   choices=["direct", "neg", "pi2_minus", "pi2_plus"])
    p.add_argument("--stats_csv", default="combined_field_summary.csv")
    p.add_argument("--save_plots", action="store_true", default=True)
    # p.add_argument("--save_fields_npz", action="store_true", default=True)

    p.add_argument("--hspace", type=float, default=0.14)
    p.add_argument("--wspace", type=float, default=0.02)
    p.add_argument("--row_height_in", type=float, default=3.20)
    p.add_argument("--width_pad_in", type=float, default=0.95)
    p.add_argument("--cbar_ratio", type=float, default=0.045)

    p.add_argument("--font_family", default="Times New Roman")
    p.add_argument("--font_size", type=float, default=14)
    p.add_argument("--axes_title_size", type=float, default=16)
    p.add_argument("--axes_label_size", type=float, default=14)
    p.add_argument("--tick_label_size", type=float, default=12)
    p.add_argument("--axes_linewidth", type=float, default=1.5)
    p.add_argument("--figure_dpi", type=int, default=300)

    p.add_argument("--target_sigma_xx_center_mpa", type=float, default=0.63)
    p.add_argument("--spacing_m", type=float, default=_rock().spacing_m)
    p.add_argument("--grid_N", type=int, default=401)
    p.add_argument("--disk_percentile_abs", type=float, default=97.0)
    p.add_argument("--colorbar_nbins", type=int, default=9)
    p.add_argument("--colorbar_fmt", default="%.2f")
    p.add_argument("--nu12", type=float, default=0.25)

    p.add_argument("--band_halfwidth_m", type=float, default=0.00045)
    p.add_argument("--band_sharp_power", type=float, default=12.0)
    p.add_argument("--phase_mode", default="center_safe")
    p.add_argument("--phase_shift_m", type=float, default=0.0)

    p.add_argument("--joint_kn_MPa_per_m", type=float, default=np.nan)
    p.add_argument("--joint_ks_MPa_per_m", type=float, default=np.nan)
    p.add_argument("--band_min_E2_fraction", type=float, default=0.20)
    p.add_argument("--band_min_G12_fraction", type=float, default=0.15)
    p.add_argument("--band_min_E1_fraction", type=float, default=0.95)

    p.add_argument("--enable_joint_closure", action="store_true", default=True)
    p.add_argument("--disable_joint_closure", action="store_false", dest="enable_joint_closure")
    p.add_argument("--closure_sigma0_mpa", type=float, default=0.20)
    p.add_argument("--closed_compliance_fraction", type=float, default=0.08)
    p.add_argument("--closure_fixed_point_iters_max", type=int, default=3)
    p.add_argument("--closure_rel_change_tol", type=float, default=2e-3)

    p.add_argument("--enable_heterogeneity", action="store_true", default=True)
    p.add_argument("--disable_heterogeneity", action="store_false", dest="enable_heterogeneity")
    p.add_argument("--hetero_seed", type=int, default=123)
    p.add_argument("--hetero_corr_len_m", type=float, default=0.008)
    p.add_argument("--stiffness_cv", type=float, default=0.10)
    p.add_argument("--angle_hetero_deg", type=float, default=3.5)
    p.add_argument("--spacing_warp_amp_m", type=float, default=_rock().spacing_warp_amp_m)

    p.add_argument("--use_hertz_contact_width", action="store_true", default=True)
    p.add_argument("--disable_hertz_contact_width", action="store_false", dest="use_hertz_contact_width")
    p.add_argument("--nu_contact", type=float, default=np.nan)
    p.add_argument("--hertz_outer_iters_max", type=int, default=6)
    p.add_argument("--hertz_rel_b_tol", type=float, default=2e-3)
    p.add_argument("--platen_mu", type=float, default=0.0)

    p.add_argument("--tol_rel_inner", type=float, default=2e-6)
    p.add_argument("--maxiter_inner", type=int, default=800)
    p.add_argument("--tol_rel_final", type=float, default=1e-7)
    p.add_argument("--maxiter_final", type=int, default=2500)
    p.add_argument("--use_precond", action="store_true", default=True)
    p.add_argument("--disable_precond", action="store_false", dest="use_precond")
    p.add_argument("--precond_floor", type=float, default=1e-6)
    p.add_argument("--solver_mode", default="pcg_then_bicgstab")

    p.add_argument("--plot_interpolation", default="bicubic")
    p.add_argument("--contour_levels", type=int, default=16)
    p.add_argument("--contour_sigma_pix", type=float, default=1.35)
    p.add_argument("--contour_corner_mask", action="store_true", default=True)
    p.add_argument("--contour_linewidth", type=float, default=0.55)
    p.add_argument("--contour_alpha", type=float, default=0.60)
    p.add_argument("--show_band_lines", action="store_true", default=True)
    p.add_argument("--show_band_edges", action="store_true", default=True)

    argv = sanitize_argv(sys.argv[1:])
    args, _unknown = p.parse_known_args(argv)

    plt.rcParams.update({
        "font.family": str(args.font_family),
        "font.size": float(args.font_size),
        "axes.linewidth": float(args.axes_linewidth),
        "axes.titlesize": float(args.axes_title_size),
        "axes.labelsize": float(args.axes_label_size),
        "xtick.labelsize": float(args.tick_label_size),
        "ytick.labelsize": float(args.tick_label_size),
        "legend.fontsize": float(args.tick_label_size),
        "figure.dpi": int(args.figure_dpi),
        "savefig.dpi": int(args.figure_dpi),
        "text.usetex": False,
    })

    os.makedirs(args.out_dir, exist_ok=True)
    field_dir = os.path.join(args.out_dir, "fields_npz")
    os.makedirs(field_dir, exist_ok=True)

    df = pd.read_csv(args.meta_csv, delimiter=",", header=0, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]

    samples = parse_sample_ids(args.sample_ids, list(df.index))
    if not samples:
        raise RuntimeError("No matching sample IDs found.")

    n = len(samples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    slots = nrows * ncols

    HROW = float(args.row_height_in)
    WFIG = float(ncols * HROW + args.width_pad_in)
    HFIG = float(nrows * HROW)

    fig = plt.figure(figsize=(WFIG, HFIG), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows, 3,
        width_ratios=[1.0, 1.0, float(args.cbar_ratio)],
        left=0.07, right=0.985, bottom=0.06, top=0.94,
        wspace=float(args.wspace), hspace=float(args.hspace),
    )
    axs = np.empty((nrows, 2), dtype=object)
    for r in range(nrows):
        for c in range(2):
            axs[r, c] = fig.add_subplot(gs[r, c])
    cax = fig.add_subplot(gs[:, 2])

    if nrows == 1:
        axs = np.array([axs[0]])

    all_abs = []
    computed = []
    stats_rows = []

    for sid in samples:
        row = df.loc[sid]

        D = float(get_first_col(row, ["Diameter_mm"])) * 1e-3
        t = float(get_first_col(row, ["Thickness_mm"])) * 1e-3

        rock = get_first_col(row, ["Rock_type"], required=True)
        anis_ratio = get_anisotropy_ratio(rock)

        E1_in = to_MPa_modulus(get_first_col(row, ["Modulus_of_Elasticity", "Youngs_Modulus", "E"]))
        G_in = float(get_first_col(row, ["Shear_Modulus", "G"], required=False, default=np.nan))
        G_in = to_MPa_modulus(G_in) if np.isfinite(G_in) else np.nan
        E2_in = E1_in / anis_ratio

        nu12 = float(get_first_col(row, ["Poisson_Ratio", "nu", "PoissonRatio"], required=False, default=args.nu12))

        if "Angle" in row.index:
            ang_deg = float(row["Angle"])
        elif "Radians" in row.index:
            ang_deg = float(np.rad2deg(float(row["Radians"])))
        else:
            raise KeyError("Need either 'Angle' or 'Radians' in the CSV.")

        alpha = map_angle_to_alpha_deg(ang_deg, angle_map=str(args.angle_map))

        joint_kn = None if not np.isfinite(args.joint_kn_MPa_per_m) else float(args.joint_kn_MPa_per_m)
        joint_ks = None if not np.isfinite(args.joint_ks_MPa_per_m) else float(args.joint_ks_MPa_per_m)
        nu_contact = None if not np.isfinite(args.nu_contact) else float(args.nu_contact)

        combined, mask, xlin, ylin, meta, fields = compute_combined_field_for_sample(
            D, t, E1_in, E2_in, nu12, alpha,
            G_in=G_in,
            target_sigma_xx_center_mpa=float(args.target_sigma_xx_center_mpa),
            spacing_m=float(args.spacing_m),
            grid_N=int(args.grid_N),
            band_halfwidth_m=float(args.band_halfwidth_m),
            band_sharp_power=float(args.band_sharp_power),
            phase_mode=str(args.phase_mode),
            phase_shift_m=float(args.phase_shift_m),
            joint_kn_MPa_per_m=joint_kn,
            joint_ks_MPa_per_m=joint_ks,
            band_min_E2_fraction=float(args.band_min_E2_fraction),
            band_min_G12_fraction=float(args.band_min_G12_fraction),
            band_min_E1_fraction=float(args.band_min_E1_fraction),
            enable_joint_closure=bool(args.enable_joint_closure),
            closure_sigma0_mpa=float(args.closure_sigma0_mpa),
            closed_compliance_fraction=float(args.closed_compliance_fraction),
            closure_fixed_point_iters_max=int(args.closure_fixed_point_iters_max),
            closure_rel_change_tol=float(args.closure_rel_change_tol),
            enable_heterogeneity=bool(args.enable_heterogeneity),
            hetero_seed=int(args.hetero_seed) + int(sid),
            hetero_corr_len_m=float(args.hetero_corr_len_m),
            stiffness_cv=float(args.stiffness_cv),
            angle_hetero_deg=float(args.angle_hetero_deg),
            spacing_warp_amp_m=float(args.spacing_warp_amp_m),
            use_hertz_contact_width=bool(args.use_hertz_contact_width),
            nu_contact=nu_contact,
            hertz_outer_iters_max=int(args.hertz_outer_iters_max),
            hertz_rel_b_tol=float(args.hertz_rel_b_tol),
            platen_mu=float(args.platen_mu),
            tol_rel_inner=float(args.tol_rel_inner),
            maxiter_inner=int(args.maxiter_inner),
            tol_rel_final=float(args.tol_rel_final),
            maxiter_final=int(args.maxiter_final),
            use_precond=bool(args.use_precond),
            precond_floor=float(args.precond_floor),
            solver_mode=str(args.solver_mode),
        )

        vals = np.asarray(combined[mask], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            abs_lim = float(np.nanpercentile(np.abs(vals), float(args.disk_percentile_abs)))
            all_abs.append(abs_lim)

        computed.append((sid, rock, ang_deg, alpha, D, t, nu12, anis_ratio, combined, mask, xlin, ylin, meta, fields))

        stats_rows.append({
            "Sample": int(sid),
            "Rock": str(rock),
            "Angle_deg": float(ang_deg),
            "alpha_rad": float(alpha),
            "D_m": float(D),
            "t_m": float(t),
            "E1_MPa": float(E1_in),
            "E2_MPa": float(E2_in),
            "G12_MPa": float(G_in) if np.isfinite(G_in) else float(meta["G12_base"]),
            "Anisotropy_Ratio": float(anis_ratio),
            "b_contact_m": float(meta["b_contact_m"]),
            "p_peak_mpa": float(meta["p_peak_mpa"]),
            "w_comp": float(meta["w_comp"]),
            "Combined_min_MPa": float(np.nanmin(combined)),
            "Combined_max_MPa": float(np.nanmax(combined)),
            "Combined_mean_MPa": float(np.nanmean(combined)),
        })

    global_abs = max(max(all_abs) if all_abs else 1e-6, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-global_abs, vcenter=0.0, vmax=global_abs)

    last_mappable = None
    for k, payload in enumerate(computed):
        sid, rock, ang_deg, alpha, D, t, nu12, anis_ratio, combined, mask, xlin, ylin, meta, fields = payload
        rowi, coli = divmod(k, 2)
        ax = axs[rowi, coli]

        title = f"{rock} | Angle={float(ang_deg):.0f}°"
        last_mappable = plot_combined_on_axis(
            ax, combined, mask, xlin, ylin, title, meta, norm,
            plot_interpolation=str(args.plot_interpolation),
            contour_levels=int(args.contour_levels),
            contour_sigma_pix=float(args.contour_sigma_pix),
            contour_corner_mask=bool(args.contour_corner_mask),
            contour_linewidth=float(args.contour_linewidth),
            contour_alpha=float(args.contour_alpha),
            show_band_lines=bool(args.show_band_lines),
            show_band_edges=bool(args.show_band_edges),
            spacing_m=float(args.spacing_m),
            band_halfwidth_m=float(args.band_halfwidth_m),
        )
        compact_axis_labels(ax, rowi, coli, nrows, ncols)

        # if args.save_fields_npz:
        #     out_npz = os.path.join(field_dir, f"sample_{int(sid):04d}_combined_fields.npz")
        #     save_sample_fields_npz(
        #         out_npz,
        #         sid=sid,
        #         rock=rock,
        #         ang_deg=ang_deg,
        #         alpha_rad=alpha,
        #         D=D,
        #         t=t,
        #         R=meta["R"],
        #         E1_base=meta["E1_base"],
        #         E2_base=meta["E2_base"],
        #         nu12=nu12,
        #         G12_base=meta["G12_base"],
        #         X=fields["X"],
        #         Y=fields["Y"],
        #         mask=fields["mask"],
        #         combined=fields["combined"],
        #         sxx=fields["sxx"],
        #         syy=fields["syy"],
        #         txy=fields["txy"],
        #         s1=fields["s1"],
        #         s3=fields["s3"],
        #         w_band=fields["w_band"],
        #         w_active=fields["w_active"],
        #         angle_local=fields["angle_local"],
        #         warp_m=fields["warp_m"],
        #         p_peak_mpa=meta["p_peak_mpa"],
        #         w_comp=meta["w_comp"],
        #         b_contact_m=meta["b_contact_m"],
        #         band_angle_rad=meta["band_angle0"],
        #         phase_plot=meta["phase_plot"],
        #     )

    if n < slots:
        rr, cc = divmod(n, ncols)
        axs[rr, cc].axis("off")

    if last_mappable is not None:
        cb = fig.colorbar(last_mappable, cax=cax)
        cb.set_label("Combined field (MPa)")
        cb.locator = MaxNLocator(nbins=int(args.colorbar_nbins))
        cb.formatter = FormatStrFormatter(str(args.colorbar_fmt))
        cb.update_ticks()
        cb.ax.tick_params(length=4, width=1)

    stats_path = os.path.join(args.out_dir, str(args.stats_csv))
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    if args.save_plots:
        out_pdf = os.path.join(args.out_dir, f"{_rock().key}_stress_distribution.pdf")
        fig.savefig(out_pdf, dpi=int(args.figure_dpi), bbox_inches="tight", pad_inches=0.08, transparent=True)
        print(f"\n✓ Saved: {out_pdf}")
        print(f"✓ Stats: {stats_path}")
        print(f"✓ Shared color scale: ±{global_abs:.3f} MPa (percentile={float(args.disk_percentile_abs):.1f}%)")

    # if args.save_fields_npz:
    #     print(f"✓ Full fields (.npz) saved in: {field_dir}")

    plt.show()
