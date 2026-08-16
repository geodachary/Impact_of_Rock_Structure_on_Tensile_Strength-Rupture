"""Principal tensile strain panels from the solver fields.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 17 during the
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
from tools.data_io import load_specimen_table

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    angle_axis, get_anisotropy_ratio, make_segments,
    modulus_to_Pa_from_csv, principal_strain_and_angle,
    reconstruct_grid_and_mask, strains_from_stress_TIsotropic,
    tensile_stress_from_sigma1_theta,
)
from tools.analysis.strain_proxy import compute_sigma1plus_and_theta_for_sample



# --- implementation ---------------------------------------------------


def wrap_pi(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi


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
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    c2s = gaussian_smooth_nan_fft(c2, mask, sigma_pix=sigma_pix)
    s2s = gaussian_smooth_nan_fft(s2, mask, sigma_pix=sigma_pix)
    ths = 0.5 * np.arctan2(s2s, c2s)
    ths[~mask] = np.nan
    return ths


def plot_panel(ax, segments, colors_deg, R, title, norm_angle):
    lc = LineCollection(
        segments,
        array=colors_deg,
        cmap=cmap_angle,
        norm=norm_angle,
        linewidths=segment_linewidth,
        alpha=segment_alpha
    )
    ax.add_collection(lc)
    ax.add_patch(plt.Circle((0, 0), R, color="black", fill=False, linewidth=1.5))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    ax.set_title(title, pad=10)
    ax.tick_params(**tick_params["major"])
    ax.tick_params(**tick_params["minor"])
    ax.minorticks_on()
    ax.grid(False)
    return lc



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global CBAR_RATIO, E1, E2, G12, HFIG, HSPACE, Lc, NCOLS, NROWS, PHYS, \
        R, ROCK_ANISO_RATIO, ROW_HEIGHT_IN, WFIG, WIDTH_PAD_IN, WSPACE, X, \
        Xtmp, Y, Ytmp, _, _fft_cache, all_mag_vals, ang_deg, anis_ratio, \
        ax, axs, boundary_exclude_frac, c, cache_dir, cache_path, \
        cap_exclude_frac, cax, cbar, cc, cmap_angle, colors_deg, data, \
        density, df, diameter, eps1, eps1_plus, ex, ey, fig, grid_N, gs, \
        gxy, i, idx, j, last_lc, length_exponent, mag_full, mask, masktmp, \
        max_seg_frac, meta, min_show_frac, norm_angle, nu12_local, outpath, \
        output_dir, panels, phi_load_rad, r, ref_mag, row, rr, \
        sample_indices, scale_percentile, segment_alpha, segment_linewidth, \
        segs, sigma1_plus, smooth_orient_sigma_pix, smooth_sigma_pix, \
        spacing_m, sxx_pa, syy_pa, target_sigma_xx_center_mpa, theta_axis, \
        theta_e, theta_p, theta_rad, thickness, tick_params, title, txy_pa, \
        vals
    _bind(rock)
    ROCK_ANISO_RATIO = {
        "augen gneiss": 2.037,
        "psammitic schist": 3.763,
        "psammatic schist": 3.763,   # historical spelling, still accepted
    }
    output_dir = os.path.join(os.getcwd(), output_dirs.FIGURE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dirs.fields(), "_cache_solver_fields_physics_v2")
    os.makedirs(cache_dir, exist_ok=True)
    df = load_specimen_table()
    df.columns = [str(c).strip() for c in df.columns]
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
    sample_indices = list(_rock().sample_ids)
    target_sigma_xx_center_mpa = 0.63
    spacing_m = _rock().spacing_m
    grid_N = 201
    density = 18
    cmap_angle = "twilight"
    segment_linewidth = 1.1
    segment_alpha = 0.95
    scale_percentile = 95.0
    max_seg_frac = 0.20
    length_exponent = 0.85
    min_show_frac = 0.02
    boundary_exclude_frac = 0.05
    cap_exclude_frac = 0.10
    smooth_sigma_pix = 1.25
    smooth_orient_sigma_pix = 1.50
    NROWS = 4
    NCOLS = 2
    ROW_HEIGHT_IN = 3.20
    WIDTH_PAD_IN = 0.95
    CBAR_RATIO = 0.045
    WSPACE = 0.02
    HSPACE = 0.14
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
        diameter = float(row["Diameter_mm"]) * 1e-3
        thickness = float(row["Thickness_mm"]) * 1e-3

        phi_load_rad = float(row["Radians"])
        theta_rad = wrap_pi(np.pi/2.0 - phi_load_rad)

        rock = str(row["Rock_type"])
        anis_ratio = get_anisotropy_ratio(rock)

        E1 = modulus_to_Pa_from_csv(float(row["Modulus_of_Elasticity"]))
        E2 = E1 / anis_ratio
        nu12_local = float(row["Poisson_Ratio"])
        G12 = modulus_to_Pa_from_csv(float(row["Shear_Modulus"])) if "Shear_Modulus" in df.columns else np.sqrt(E1 * E2) / (2.0 * (1.0 + nu12_local))

        cache_path = os.path.join(
            cache_dir,
            f"physics_v2_idx{idx}_N{grid_N}_sp{spacing_m:.6f}_t{target_sigma_xx_center_mpa:.3f}.npz"
        )

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            sigma1_plus = data["sigma1_plus"]
            theta_p = data["theta_p"]
        else:
            Xtmp, Ytmp, masktmp, sigma1_plus, theta_p, meta = compute_sigma1plus_and_theta_for_sample(
                diameter,
                thickness,
                float(row["Modulus_of_Elasticity"]),
                float(row["Modulus_of_Elasticity"]) / anis_ratio,
                float(row["Poisson_Ratio"]),
                theta_rad,
                float(row["Shear_Modulus"]) if "Shear_Modulus" in df.columns else np.nan,
                target_sigma_xx_center_mpa=target_sigma_xx_center_mpa,
                spacing_m=spacing_m,
                grid_N=grid_N,
                **PHYS
            )
            np.savez_compressed(cache_path, sigma1_plus=sigma1_plus, theta_p=theta_p)

        X, Y, mask, R = reconstruct_grid_and_mask(diameter, grid_N)

        sigma1_plus = np.nan_to_num(sigma1_plus, nan=0.0, posinf=0.0, neginf=0.0)
        theta_p = np.nan_to_num(theta_p, nan=0.0, posinf=0.0, neginf=0.0)

        sxx_pa, syy_pa, txy_pa = tensile_stress_from_sigma1_theta(sigma1_plus, theta_p)
        sxx_pa[~mask] = 0.0
        syy_pa[~mask] = 0.0
        txy_pa[~mask] = 0.0

        ex, ey, gxy = strains_from_stress_TIsotropic(
            sxx_pa, syy_pa, txy_pa, E1, E2, nu12_local, G12, theta_rad
        )
        ex[~mask] = np.nan
        ey[~mask] = np.nan
        gxy[~mask] = np.nan

        eps1, theta_e = principal_strain_and_angle(ex, ey, gxy)
        eps1_plus = np.maximum(eps1, 0.0)

        if smooth_sigma_pix > 0:
            eps1_plus = gaussian_smooth_nan_fft(eps1_plus, mask, sigma_pix=smooth_sigma_pix)

        if smooth_orient_sigma_pix > 0:
            theta_e = nematic_smooth_theta(theta_e, mask, sigma_pix=smooth_orient_sigma_pix)

        theta_axis = angle_axis(theta_e)

        Lc = diameter / 4.0
        mag_full = eps1_plus * Lc
        vals = mag_full[np.isfinite(mag_full)]
        vals = vals[vals > 0]
        if vals.size:
            all_mag_vals.append(vals)

        ang_deg = float(row["Angle"]) if "Angle" in row.index else float(np.rad2deg(phi_load_rad))
        title = f"{rock} ({ang_deg:.0f}°)"
        panels.append((title, diameter, X, Y, mask, eps1_plus, theta_axis, R))
    if len(all_mag_vals) == 0:
        raise RuntimeError("No finite tensile-strain magnitudes found. Check solver outputs.")
    all_mag_vals = np.concatenate(all_mag_vals)
    ref_mag = float(np.nanpercentile(all_mag_vals, scale_percentile))
    ref_mag = max(ref_mag, 1e-30)
    norm_angle = Normalize(vmin=-90.0, vmax=90.0)
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
    last_lc = None
    for i, (title, diameter, X, Y, mask, eps1_plus, theta_axis, R) in enumerate(panels):
        segs, colors_deg, _ = make_segments(X, Y, mask, eps1_plus, theta_axis, diameter, ref_mag, R)
        rr, cc = divmod(i, 2)
        ax = axs[rr, cc]
        last_lc = plot_panel(ax, segs, colors_deg, R, title, norm_angle)
        compact_axis_labels(ax, rr, cc, NROWS, NCOLS)
    if len(panels) < NROWS * NCOLS:
        for j in range(len(panels), NROWS * NCOLS):
            rr, cc = divmod(j, NCOLS)
            axs[rr, cc].axis("off")
    if last_lc is not None:
        cbar = fig.colorbar(last_lc, cax=cax)
        cbar.set_label("Principal tensile strain axis angle (deg)", rotation=270, labelpad=18)
        cbar.set_ticks([-90, -45, 0, 45, 90])
        cbar.ax.tick_params(length=4, width=1)
    outpath = os.path.join(output_dir, f"{_rock().key}_principal_tensile_strain.pdf")
    plt.savefig(outpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved: {outpath}")
    print(f"Cache dir: {cache_dir}")
    print(f"Length reference magnitude (p{scale_percentile:.0f}) = {ref_mag:.3e}")
