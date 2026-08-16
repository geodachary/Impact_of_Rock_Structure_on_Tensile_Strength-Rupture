#!/usr/bin/env python3
"""Show what a corridor and a lobe are, rather than asserting the difference.

The strain-energy panels were read as showing continuous corridors in one
lithology and localized hotspots in the other, and the distinction was not
visible to a reader comparing two colour maps. The reason is that the elevated
region is not outlined in those panels: the eye is asked to judge connectivity
from a continuous colour scale, which it cannot do reliably.

This figure draws the region the words refer to. The upper quartile of
strain-energy density inside the disc interior is filled, its separate connected
components are given different shades, and each panel is annotated with the
component count and the elongation of the largest component. Whether the two
platen lobes have merged into one elongated corridor is then something the
reader sees outlined rather than infers from shading.

    python scripts/make_energy_localization_figure.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import energy_localization as elz          # noqa: E402
from tools import output_dirs                          # noqa: E402
from tools import fabric_tractions as ft              # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
SLUG = {"Augen gneiss": "augen_gneiss", "Psammitic schist": "psammitic_schist"}
SHADES = ["#c6dbef", "#4C72B0", "#2f4b7c", "#8c9ec4"]


def panel(ax, f, show_angle=True):
    """Elevated-energy region of one specimen, components shaded separately."""
    R = float(f["R_m"])
    hot = elz.elevated_set(f)
    lab, n = ndimage.label(hot, structure=np.ones((3, 3)))
    sh = elz.component_shape(f)

    ax.add_artist(plt.Circle((0, 0), R, fill=False, color="0.35", lw=1.0, zorder=3))
    shown = np.where(lab > 0, ((lab - 1) % len(SHADES)) + 1, 0).astype(float)
    ax.pcolormesh(f["X"], f["Y"], np.ma.masked_where(shown == 0, shown),
                  cmap=ListedColormap(SHADES), vmin=0.5, vmax=len(SHADES) + 0.5,
                  shading="nearest", zorder=2)

    corridor = elz.is_corridor(sh)
    ax.text(0.03, 0.03,
            f"n = {sh['n_components']}\ne = {sh['elongation']:.2f}",
            transform=ax.transAxes, fontsize=ANNOT_FS, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.6))
    ax.set_title(f"{f['angle_deg']:.0f}°" + ("  corridor" if corridor else "  lobes"),
                 fontsize=ANNOT_FS, pad=2.5,
                 color="#1a5c2a" if corridor else "0.25")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.08 * R, 1.08 * R)
    ax.set_ylim(-1.08 * R, 1.08 * R)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.0)


#: Sized so that at 0.48\\textwidth the graphic is reproduced at close to its
#: natural size, which keeps the annotations legible in a side-by-side pair
#: rather than shrinking them the way a wide figure would.
FIGSIZE = (3.15, 6.4)


def main():
    output_dirs.ensure()
    apply_plot_style()
    fields = {}
    for p in ft.field_files():
        f = ft.load_field(p)
        fields.setdefault(f["rock"], []).append(f)

    for rock, group in fields.items():
        group.sort(key=lambda f: f["angle_deg"])
        fig, axs = plt.subplots(4, 2, figsize=FIGSIZE)
        for k, f in enumerate(group):
            panel(axs[k // 2, k % 2], f)
        for j in range(len(group), 8):
            axs[j // 2, j % 2].axis("off")
        axs[-1, -1].text(0.5, 0.5,
                         "filled: upper quartile\nof strain-energy density\n\n"
                         "shades: separate\nconnected components\n\n"
                         "$n$: components\n$e$: elongation of\nthe largest",
                         transform=axs[-1, -1].transAxes, ha="center", va="center",
                         fontsize=ANNOT_FS)
        fig.tight_layout(pad=0.4, h_pad=0.7, w_pad=0.4)
        for d in OUT_DIRS:
            fig.savefig(d / f"fig_S_energy_localization_{SLUG[rock]}.pdf",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

        n_corr = sum(elz.is_corridor(elz.component_shape(f)) for f in group)
        e = [elz.component_shape(f)["elongation"] for f in group]
        print(f"  {rock:17s} corridor at {n_corr}/7 orientations, "
              f"elongation {min(e):.2f}-{max(e):.2f}")
    print("\n  written: fig_S_energy_localization_<rock>.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
