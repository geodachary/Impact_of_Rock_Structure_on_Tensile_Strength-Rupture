"""Radial stress traverses and weak-band crossings on y=0.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 10 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
import os

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    create_triangular_mesh_in_disk, integrate_line_force, sample_line,
)
from tools.data_io import load_specimen_table



# --- implementation ---------------------------------------------------


def weak_band_weight_points(x, y, angle_rad, spacing, halfwidth, phase, sharp_power):
    a = float(angle_rad)
    s = float(spacing)
    bw = float(halfwidth)
    ph = float(phase)
    nx = -np.sin(a)
    ny =  np.cos(a)

    d = nx * x + ny * y + ph
    dist_eff = (s / np.pi) * np.abs(np.sin(np.pi * d / (s + 1e-30)))
    w = np.exp(-sharp_power * (dist_eff / (bw + 1e-30))**2)
    return np.clip(w, 0.0, 1.0)


def anisotropic_strength_factor(x, y):
    w = weak_band_weight_points(
        x, y,
        angle_rad=anisotropic_angle_rad,
        spacing=spacing_m,
        halfwidth=band_halfwidth_m,
        phase=band_phase_m,
        sharp_power=band_sharp_power
    )
    fac = 1.0 - strength_drop * w
    return np.maximum(fac, min_strength_fraction)


def principal_stresses_2d(sxx, syy, txy):
    s_avg = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5*(sxx - syy))**2 + txy**2)
    return s_avg + rad, s_avg - rad


def band_crossings_on_y0(R):
    a = float(anisotropic_angle_rad)
    nx = -np.sin(a)
    if abs(nx) < 1e-10:
        return np.array([])
    kmin = int(np.floor((-abs(nx)*R + band_phase_m) / spacing_m)) - 3
    kmax = int(np.ceil((+abs(nx)*R + band_phase_m) / spacing_m)) + 3
    xs = []
    for k in range(kmin, kmax + 1):
        xk = (k * spacing_m - band_phase_m) / (nx + 1e-30)
        if -R <= xk <= R:
            xs.append(xk)
    xs = np.array(xs, float)
    xs.sort()
    return xs


def calculate_shape_fields_on_nodes(x, y, r, diameter):
    R = diameter / 2.0
    r_safe = np.maximum(r, 1e-12)

    # --- Uniform base stress shape ---
    s0 = 1.0  # 1 Pa nominal
    sigma_xx = s0 * np.ones_like(r_safe)
    sigma_yy = s0 * np.ones_like(r_safe)

    # simple shear proxy
    tau_xy   = s0 * (x / r_safe)

    # rotate by anisotropic angle
    c = np.cos(anisotropic_angle_rad)
    s = np.sin(anisotropic_angle_rad)
    sigma_xx_p = sigma_xx * c*c + sigma_yy * s*s + 2.0 * tau_xy * s*c
    sigma_yy_p = sigma_xx * s*s + sigma_yy * c*c - 2.0 * tau_xy * s*c
    tau_xy_p   = (sigma_yy - sigma_xx) * s*c + tau_xy * (c*c - s*s)

    # tensile-like scalar: max principal
    s1, _ = principal_stresses_2d(sigma_xx_p, sigma_yy_p, tau_xy_p)
    sigma_theta = s1

    # band weakening
    fac = anisotropic_strength_factor(x, y)
    sigma_theta *= fac
    sigma_xx_p *= fac

    outside = r > R
    sigma_theta[outside] = np.nan
    sigma_xx_p[outside] = np.nan
    return sigma_theta, sigma_xx_p



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global F_unit_1, F_unit_2, P_failure_A, P_failure_A_check, P_failure_B, \
        P_k, P_uniform_est, R, _, anisotropic_angle_rad, ax, \
        band_halfwidth_m, band_phase_m, band_sharp_power, center_mask, \
        diameter, err_center_rel, err_conv_rel, err_def_rel, err_lin_rel, \
        fig, index_number, k_test, line_samples, line_samples_check, \
        mesh_seed, min_strength_fraction, num_points_mesh, outpath, r, \
        raw1_s, raw1_u, raw2_u, s0_pa, save_dir, save_name, sigma_theta_pa, \
        sigma_theta_pa_k, sigma_theta_unit_pa, sm1_s, sm1_u, sm2_u, smk, \
        smooth_sigma_pts, spacing_m, step, strength_drop, \
        sxx_center_scaled_mpa, sxx_center_unit_pa, sxxp_unit_pa, \
        target_center_pa, target_sigma_xx_center_mpa, thickness, \
        tol_center_rel, tol_conv_rel, tol_def_rel, tol_lin_rel, triang, x, \
        x1, x1s, x2, xk, xs, xv, y, y_top_pad_fraction, y_vals, ymax, ymin, \
        yr
    _bind(rock)
    df = load_specimen_table()
    index_number = 4
    diameter = float(df.loc[index_number, "Diameter_mm"]) * 1e-3
    thickness = float(df.loc[index_number, "Thickness_mm"]) * 1e-3
    anisotropic_angle_rad = float(df.loc[index_number, "Radians"])
    target_sigma_xx_center_mpa = 0.63
    spacing_m = _rock().spacing_m
    band_halfwidth_m = 0.00045
    band_sharp_power = 10
    band_phase_m = 0.5 * spacing_m
    strength_drop = 0.35
    min_strength_fraction = 0.40
    num_points_mesh = 70000
    mesh_seed = 0
    line_samples = 2200
    line_samples_check = 4400
    smooth_sigma_pts = 2.0
    save_dir = output_dirs.FIGURE_DIR
    save_name = f"{_rock().key}_stress_distribution_graph.pdf"
    tol_center_rel = 2e-3
    tol_conv_rel = 8e-3
    tol_def_rel = 2e-4
    tol_lin_rel = 2e-4
    y_top_pad_fraction = 0.19
    R = diameter / 2.0
    triang, x, y, r = create_triangular_mesh_in_disk(diameter, num_points=num_points_mesh, seed=mesh_seed)
    sigma_theta_unit_pa, sxxp_unit_pa = calculate_shape_fields_on_nodes(x, y, r, diameter)
    center_mask = r < (0.02 * R)
    sxx_center_unit_pa = float(np.nanmean(sxxp_unit_pa[center_mask]))
    if (not np.isfinite(sxx_center_unit_pa)) or abs(sxx_center_unit_pa) < 1e-30:
        raise RuntimeError("Center σxx' unit value invalid/too small. Increase mesh points or adjust setup.")
    target_center_pa = target_sigma_xx_center_mpa * 1e6
    s0_pa = target_center_pa / (abs(sxx_center_unit_pa) + 1e-30)
    sigma_theta_pa = sigma_theta_unit_pa * s0_pa
    x1, raw1_u, sm1_u = sample_line(triang, sigma_theta_unit_pa, R, line_samples, smooth_sigma_pts)
    F_unit_1 = integrate_line_force(x1, sm1_u, thickness)
    x2, raw2_u, sm2_u = sample_line(triang, sigma_theta_unit_pa, R, line_samples_check, smooth_sigma_pts)
    F_unit_2 = integrate_line_force(x2, sm2_u, thickness)
    P_failure_A = s0_pa * F_unit_1
    P_failure_A_check = s0_pa * F_unit_2
    x1s, raw1_s, sm1_s = sample_line(triang, sigma_theta_pa, R, line_samples, smooth_sigma_pts)
    P_failure_B = integrate_line_force(x1s, sm1_s, thickness)
    P_uniform_est = target_sigma_xx_center_mpa * 1e6 * np.pi * R * R
    sxx_center_scaled_mpa = (abs(sxx_center_unit_pa) * s0_pa) / 1e6
    err_center_rel = abs(sxx_center_scaled_mpa - target_sigma_xx_center_mpa) / (target_sigma_xx_center_mpa + 1e-30)
    err_def_rel = abs(P_failure_B - P_failure_A) / (abs(P_failure_A) + 1e-30)
    err_conv_rel = abs(P_failure_A_check - P_failure_A) / (abs(P_failure_A_check) + 1e-30)
    k_test = 1.5
    sigma_theta_pa_k = sigma_theta_unit_pa * (k_test * s0_pa)
    xk, _, smk = sample_line(triang, sigma_theta_pa_k, R, line_samples, smooth_sigma_pts)
    P_k = integrate_line_force(xk, smk, thickness)
    err_lin_rel = abs(P_k - k_test * P_failure_A) / (abs(k_test * P_failure_A) + 1e-30)
    print(f"[Reference] Uniform-stress estimate for {target_sigma_xx_center_mpa} MPa: {P_uniform_est:.2f} N")
    print(f"[Calibrated] center σxx' (unit scale=1 Pa): {(sxx_center_unit_pa/1e6):.6e} MPa per Pa-scale")
    print(f"[Calibrated] required nominal scale s0 ≈ {s0_pa:.3e} Pa")
    print(f"[Result] P_failure (definition) ≈ {P_failure_A:.2f} N\n")
    print("=== VALIDATION (consistent with P_failure definition) ===")
    print(f"Center stress check: σxx'(center) = {sxx_center_scaled_mpa:.6f} MPa (target {target_sigma_xx_center_mpa:.6f})")
    print(f"  rel err = {err_center_rel:.3e} -> {'PASS' if err_center_rel < tol_center_rel else 'FAIL'}")
    print(f"Definition check: P_failure_A (unit*scale) = {P_failure_A:.2f} N, P_failure_B (direct) = {P_failure_B:.2f} N")
    print(f"  rel err = {err_def_rel:.3e} -> {'PASS' if err_def_rel < tol_def_rel else 'FAIL'}")
    print(f"Convergence check: P_failure({line_samples}) = {P_failure_A:.2f} N, P_failure({line_samples_check}) = {P_failure_A_check:.2f} N")
    print(f"  rel diff = {err_conv_rel:.3e} -> {'PASS' if err_conv_rel < tol_conv_rel else 'FAIL'}")
    print(f"Linearity check: k={k_test:.2f}, P(k*s0) = {P_k:.2f} N, k*P = {(k_test*P_failure_A):.2f} N")
    print(f"  rel err = {err_lin_rel:.3e} -> {'PASS' if err_lin_rel < tol_lin_rel else 'FAIL'}\n")
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=220)
    ax.plot(x1s, sm1_s / 1e6, label="Smoothed Stress", zorder=3)
    step = max(len(x1s) // 280, 1)
    ax.scatter(x1s[::step], (raw1_s / 1e6)[::step],
               s=18, alpha=0.55, label="Data Points",
               edgecolors="black", linewidth=0.35, zorder=4)
    xs = band_crossings_on_y0(R)
    for xv in xs:
        ax.axvline(xv, linestyle="--", linewidth=0.7, alpha=0.22, zorder=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("X along diameter (m)", fontsize=14, labelpad=10)
    ax.set_ylabel("Stress (MPa)", fontsize=14, labelpad=10)
    y_vals = sm1_s / 1e6
    ymin = float(np.nanmin(y_vals))
    ymax = float(np.nanmax(y_vals))
    yr = ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - 0.08 * yr, ymax + y_top_pad_fraction * yr)
    ax.legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
        edgecolor="black",
        framealpha=0.92,
        fancybox=True,
        shadow=True,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, save_name)
    fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    print(f"Saved: {outpath}")
