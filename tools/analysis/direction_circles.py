"""Stress-direction circles around the disk.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 21 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import hashlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.data_io import load_specimen_table

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    clip_to_disk, downsample_mask, get_anisotropy_ratio,
    plot_direction_circle, reconstruct_grid,
)



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


def get_uv_cached(idx, row):
    D = float(row["Diameter_mm"]) * 1e-3
    t = float(row["Thickness_mm"]) * 1e-3

    phi_load = float(row["Radians"])
    alpha = wrap_pi(np.pi/2.0 - phi_load)

    rock = str(row["Rock_type"])
    anis_ratio = get_anisotropy_ratio(rock)

    E1_in = float(row["Modulus_of_Elasticity"])
    E2_in = float(E1_in) / float(anis_ratio)
    nu = float(row["Poisson_Ratio"])
    G_in = float(row["Shear_Modulus"]) if "Shear_Modulus" in row.index else np.nan

    spacing_m = float(row["Spacing_m"]) if "Spacing_m" in row.index else float(SPACING_DEFAULT_M)

    # The cached field is only valid for the elastic constants and geometry it
    # was solved with, so those go into the key. A key built from the specimen
    # number alone survives any edit to the specimen table and silently returns
    # a field belonging to material properties that no longer apply.
    stamp = hashlib.sha1(
        np.array([D, t, E1_in, E2_in, nu, G_in, alpha], float).tobytes()
    ).hexdigest()[:10]
    cache_path = os.path.join(
        cache_dir,
        f"idx{idx}_N{GRID_N}_sp{spacing_m:.6f}_t{TARGET_SIGMA_XX_CENTER_MPA:.3f}"
        f"_m{stamp}_v3.npz"
    )
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return D, z["u"], z["v"]

    if "compute_uv_for_sample" not in globals():
        raise RuntimeError(
            "compute_uv_for_sample(...) not found. "
            "Run 'TRUE solver u' code first (the one that defines compute_uv_for_sample)."
        )

    X, Y, mask, u, v, meta = compute_uv_for_sample(
        D, t, E1_in, E2_in, nu, alpha, G_in,
        target_sigma_xx_center_mpa=TARGET_SIGMA_XX_CENTER_MPA,
        spacing_m=spacing_m,
        grid_N=GRID_N,
    )

    np.savez_compressed(cache_path, u=u, v=v)
    return D, u, v



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global BOUNDARY_EXCLUDE_FRAC, CAP_EXCLUDE_FRAC, CBAR_RATIO, COLOR_MODE, \
        COLOR_PERCENTILE, D, DENSITY, GRID_N, HFIG, HSPACE, L, \
        MAX_ARROW_FRAC, Ms, NCOLS, NROWS, R, ROCK_ANISO_RATIO, \
        ROW_HEIGHT_IN, SAMPLE_IDS, SPACING_DEFAULT_M, \
        TARGET_SIGMA_XX_CENTER_MPA, Up, Uplot, Us, Vp, Vplot, Vs, WFIG, \
        WIDTH_PAD_IN, WSPACE, X, Xp, Xs, Y, Yp, Ys, all_colors, ang_deg, \
        ax, axs, c, cache_dir, cax, cbar, cc, col_id, colors, df, dx, dy, \
        epsb, fig, gs, i, idx, j, last_mappable, mag, mag_plot, mag_safe, \
        mask, norm, ok, outpath, output_dir, r, ratio, ref_global, row, \
        row_id, rr, sample_list, tick_params, title, u, v, valid, vals, \
        vmax
    _bind(rock)
    ROCK_ANISO_RATIO = {
        "augen gneiss": 2.037,
        "psammitic schist": 3.763,
        "psammatic schist": 3.763,   # historical spelling, still accepted
    }
    output_dir = os.path.join(os.getcwd(), output_dirs.FIGURE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dirs.fields(), "_cache_direction_circles_uv_v2")
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
    SAMPLE_IDS = list(_rock().sample_ids)
    GRID_N = 201
    TARGET_SIGMA_XX_CENTER_MPA = 0.63
    SPACING_DEFAULT_M = _rock().spacing_m
    DENSITY = 20
    MAX_ARROW_FRAC = 0.20
    BOUNDARY_EXCLUDE_FRAC = 0.05
    CAP_EXCLUDE_FRAC = 0.08
    COLOR_PERCENTILE = 99.0
    COLOR_MODE = "uv_mag_mm"
    NROWS = 4
    NCOLS = 2
    ROW_HEIGHT_IN = 3.20
    WIDTH_PAD_IN = 0.95
    CBAR_RATIO = 0.045
    WSPACE = 0.02
    HSPACE = 0.14
    all_colors = []
    for idx in SAMPLE_IDS:
        if idx not in df.index:
            continue
        row = df.loc[idx]
        D, u, v = get_uv_cached(idx, row)
        X, Y, mask, R = reconstruct_grid(D, GRID_N)

        if COLOR_MODE == "u_only_mm":
            c = np.abs(u) * 1e3
        else:
            c = np.hypot(u, v) * 1e3

        c = np.where(mask, c, np.nan)
        vals = c[np.isfinite(c)]
        if vals.size:
            all_colors.append(vals)
    if len(all_colors) == 0:
        raise RuntimeError("No finite displacement values found to build color scale.")
    all_colors = np.concatenate(all_colors)
    vmax = float(np.nanpercentile(all_colors, COLOR_PERCENTILE))
    vmax = max(vmax, 1e-12)
    norm = Normalize(vmin=0.0, vmax=vmax)
    ref_global = max(vmax, 1e-12)
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
    for i, idx in enumerate(SAMPLE_IDS):
        if idx not in df.index:
            continue
        row = df.loc[idx]

        D, u, v = get_uv_cached(idx, row)
        X, Y, mask, R = reconstruct_grid(D, GRID_N)

        r = np.hypot(X, Y)
        valid = mask.copy()
        valid &= (r <= (1.0 - BOUNDARY_EXCLUDE_FRAC) * R)
        valid &= (np.abs(Y) <= (1.0 - CAP_EXCLUDE_FRAC) * R)

        Xs, Ys, Ms, Us, Vs = downsample_mask(X, Y, valid, u, v, density=DENSITY)

        ok = Ms & np.isfinite(Us) & np.isfinite(Vs)
        Xp = Xs[ok]
        Yp = Ys[ok]
        Up = Us[ok]
        Vp = Vs[ok]

        mag = np.hypot(Up, Vp)
        mag_safe = np.where(mag == 0.0, 1.0, mag)
        dx = Up / mag_safe
        dy = Vp / mag_safe

        if COLOR_MODE == "u_only_mm":
            mag_plot = np.abs(Up) * 1e3
        else:
            mag_plot = mag * 1e3

        # One length scale for every panel. Normalising each panel by its own 95th
        # percentile makes all seven look alike whatever the displacements actually
        # are, which is the opposite of what the comparison is for.
        ratio = np.clip(mag_plot / ref_global, 0.0, 1.0)
        L = ratio * (MAX_ARROW_FRAC * R)

        Uplot = dx * L
        Vplot = dy * L

        epsb = 9.9e-4 * R
        Uplot, Vplot = clip_to_disk(Xp, Yp, Uplot, Vplot, R, epsb)

        colors = np.clip(mag_plot, 0.0, vmax)

        rock = str(row["Rock_type"]) if "Rock_type" in row.index else f"Sample {idx}"
        ang_deg = float(row["Angle"]) if "Angle" in row.index else float(np.rad2deg(float(row["Radians"])))
        title = f"{rock} ({ang_deg:.0f}°)"

        row_id, col_id = divmod(i, 2)
        ax = axs[row_id, col_id]

        last_mappable = plot_direction_circle(
            ax, Xp, Yp, Uplot, Vplot, colors, R, title, norm, cmap="magma"
        )
        compact_axis_labels(ax, row_id, col_id, NROWS, NCOLS)
    sample_list = [idx for idx in SAMPLE_IDS if idx in df.index]
    if len(sample_list) < NROWS * NCOLS:
        for j in range(len(sample_list), NROWS * NCOLS):
            rr, cc = divmod(j, NCOLS)
            axs[rr, cc].axis("off")
    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, cax=cax)
        cbar.set_label("Displacement magnitude (mm)", rotation=270, labelpad=20)
        cbar.locator = MaxNLocator(nbins=6)
        cbar.formatter = FormatStrFormatter("%.2e")
        cbar.update_ticks()
    outpath = os.path.join(output_dir, f"{_rock().key}_direction_circles.pdf")
    plt.savefig(outpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved: {outpath}")
    print(f"Color vmax (p{COLOR_PERCENTILE:.0f}) = {vmax:.3e} mm")
    print(f"Cache dir: {cache_dir}")
