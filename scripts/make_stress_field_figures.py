#!/usr/bin/env python3
"""Stress-field figures rebuilt from the exported DDM fields.

Two things are fixed here relative to the panels the solver draws inline.

First, the glyph panels now share one length scale. Normalising each panel by
its own 95th percentile rescales every specimen to fill its own axes, so a
specimen carrying half the stress of its neighbour draws glyphs the same length.
Amplitude is where most of the orientation dependence lives -- the normalised
stress-field shape differs by up to 21% across gneiss orientations and 41%
across schist -- and per-panel normalisation divides exactly that out.

Second, a companion figure resolves the same stress state onto the foliation
plane. Principal directions rotate by under 2 degrees across the gneiss series
and under 5 across the schist, which is why the glyph panels look alike however
they are scaled; the fabric-resolved tractions vary over the full range and are
what the failure argument actually rests on.

    python scripts/make_stress_field_figures.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fabric_tractions as ft          # noqa: E402
from tools import output_dirs  # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
SLUG = {"Augen gneiss": "augen_gneiss", "Psammitic schist": "psammitic_schist"}
COLOR = {"Augen gneiss": "#4C72B0", "Psammitic schist": "#DD4B39"}

GLYPH_STEP = 9          # grid stride between glyphs
GLYPH_FRAC = 0.085      # half-length of a glyph at the reference stress, in radii
LEN_CAP_FRAC = 0.16


def save(fig, stem):
    for d in OUT_DIRS:
        d.mkdir(exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def glyph_segments(x, y, mag, th, ref, R):
    """Line segments for one principal direction, length proportional to |sigma|."""
    L = np.clip(GLYPH_FRAC * R * (np.abs(mag) / ref), 0.0, LEN_CAP_FRAC * R)
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([x + L * c, y + L * s], axis=1),
                     np.stack([x - L * c, y - L * s], axis=1)], axis=1)


def stress_glyph_figure(fields, rock, ref):
    """Seven orientations of one lithology, all drawn at the same stress scale."""
    grp = sorted((f for f in fields if f["rock"] == rock), key=lambda f: f["angle_deg"])
    nrows = int(np.ceil(len(grp) / 2))
    fig, axs = plt.subplots(nrows, 2, figsize=(7.4, 3.55 * nrows))
    axs = np.atleast_2d(axs)

    for k, f in enumerate(grp):
        ax = axs[k // 2, k % 2]
        R = float(f["R_m"])
        m = ft.core_mask(f, 0.90)
        thin = np.zeros_like(m)
        thin[::GLYPH_STEP, ::GLYPH_STEP] = True
        m = m & thin

        x, y = f["X"][m], f["Y"][m]
        s1, s3, th = f["s1"][m], f["s3"][m], f["th_rad"][m]

        for mag, ang, col in ((s1, th, "#c1121f"), (s3, th + np.pi / 2, "#1f5f9e")):
            seg = glyph_segments(x, y, mag, ang, ref, R)
            for sel, ls in ((mag >= 0, "solid"), (mag < 0, "dashed")):
                if np.any(sel):
                    ax.add_collection(LineCollection(seg[sel], colors=col, linewidths=0.95,
                                                     linestyles=ls, alpha=0.75, zorder=2))

        ax.add_artist(plt.Circle((0, 0), R, fill=False, color="k", lw=1.2, zorder=3))
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.05 * R, 1.05 * R)
        ax.set_ylim(-1.05 * R, 1.05 * R)
        ax.set_title(f"{rock} ({f['angle_deg']:.0f}°)", pad=6)
        if k // 2 == nrows - 1:
            ax.set_xlabel("X (m)")
        else:
            ax.tick_params(labelbottom=False)
        if k % 2 == 0:
            ax.set_ylabel("Y (m)")
        else:
            ax.tick_params(labelleft=False)

    for j in range(len(grp), nrows * 2):
        axs[j // 2, j % 2].axis("off")

    handles = [Line2D([], [], color="#c1121f", lw=1.6, label=r"$\sigma_1$"),
               Line2D([], [], color="#1f5f9e", lw=1.6, label=r"$\sigma_3$"),
               Line2D([], [], color="0.35", lw=1.6, ls="solid", label="tension"),
               Line2D([], [], color="0.35", lw=1.6, ls="dashed", label="compression")]
    axs[-1, -1].legend(handles=handles, loc="center", frameon=True, ncol=2,
                  title=f"common scale: full-length glyph = {ref:.1f} MPa")
    fig.tight_layout()
    save(fig, f"{SLUG[rock]}_stress_tensors")


def traction_figures(tab):
    """Foliation-resolved normal and shear traction against orientation."""
    for stem, col, ylab, title in (
        ("fabric_traction_normal", "sigma_n_mean_MPa",
         r"Normal traction on foliation, $\sigma_n$ (MPa)", "opening"),
        ("fabric_traction_shear", "tau_abs_mean_MPa",
         r"Shear traction on foliation, $|\tau|$ (MPa)", "sliding")):
        fig, ax = plt.subplots(figsize=(4.6, 3.9))
        for rock, g in tab.groupby("rock"):
            g = g.sort_values("angle_deg")
            ax.plot(g.angle_deg, g[col], "-o", color=COLOR[rock], lw=2.0,
                    ms=6, label=rock, zorder=3)
        if col.startswith("sigma"):
            ax.axhline(0.0, color="0.35", ls=":", lw=1.2, zorder=1)
            ax.text(2, 0.0, "foliation opens above this line", fontsize=ANNOT_FS,
                    color="0.35", va="bottom")
        ax.set_xlabel(r"Fabric angle, $\alpha$ (deg)")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(0, 91, 15))
        ax.set_xlim(-3, 93)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=LEGEND_FS)
        for s in ax.spines.values():
            s.set_linewidth(1.5)
        fig.tight_layout()
        save(fig, stem)


def main():
    apply_plot_style()
    files = ft.field_files()
    if not files:
        raise SystemExit("no exported fields found; run the notebooks first")
    fields = [ft.load_field(p) for p in files]

    # one reference stress for every panel of every lithology
    pool = np.concatenate([np.concatenate([np.abs(f["s1"][ft.core_mask(f, 0.90)]),
                                           np.abs(f["s3"][ft.core_mask(f, 0.90)])])
                           for f in fields])
    ref = float(np.percentile(pool, 95.0))
    print(f"  common glyph reference (95th percentile of |sigma| over all "
          f"{len(fields)} specimens) = {ref:.2f} MPa")

    for rock in sorted({f["rock"] for f in fields}):
        stress_glyph_figure(fields, rock, ref)
        print(f"  wrote {SLUG[rock]}_stress_tensors.pdf")

    tab = ft.traction_table()
    traction_figures(tab)
    output_dirs.tables()
    tab.to_csv(REPO / output_dirs.TABLE_DIR / "fabric_tractions.csv", index=False)

    rot = ft.principal_rotation_table()
    rot.to_csv(REPO / output_dirs.TABLE_DIR / "principal_rotation.csv", index=False)

    het = ft.heterogeneity_table()
    het.to_csv(REPO / output_dirs.TABLE_DIR / "principal_heterogeneity.csv", index=False)

    # These two are read back by scripts/patch_stress_field_text.py to build the
    # captions. They had no writer, so they aged while their siblings above were
    # refreshed on every run, and the captions were being built partly from
    # current tables and partly from stale ones. Writing them here keeps the
    # whole set in step.
    grd = ft.orientation_gradient_table()
    grd.to_csv(REPO / output_dirs.TABLE_DIR / "orientation_gradient.csv", index=False)

    from tools import displacement_corridor as dc
    cor = dc.corridor_table()
    cor.to_csv(REPO / output_dirs.TABLE_DIR / "displacement_corridor.csv", index=False)

    print("\n  foliation-resolved tractions (disk core):")
    for rock, g in tab.groupby("rock"):
        g = g.sort_values("angle_deg")
        print(f"    {rock}:")
        print("      sigma_n  " + "  ".join(f"{a:.0f}d={v:+.2f}"
              for a, v in zip(g.angle_deg, g.sigma_n_mean_MPa)))
        print("      |tau|    " + "  ".join(f"{a:.0f}d={v:.2f}"
              for a, v in zip(g.angle_deg, g.tau_abs_mean_MPa)))
    print("\n  principal-axis rotation vs the 0 degree specimen (median, deg):")
    for rock, g in rot.groupby("rock"):
        print(f"    {rock}: max {g.rotation_median_deg.max():.2f}")
    print("\n  written: figures + fabric_tractions, principal_rotation,\n           principal_heterogeneity, orientation_gradient,\n           displacement_corridor (outputs/tables/*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
