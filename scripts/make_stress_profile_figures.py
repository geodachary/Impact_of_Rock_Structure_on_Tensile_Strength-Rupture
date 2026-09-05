#!/usr/bin/env python3
"""Stress and resistance profiles along the loading diameter, from the solved field.

The panel this replaces was not built from the Lekhnitskii solution. It formed a
uniform placeholder field (``sigma_xx = s0`` everywhere), took its maximum
principal value, multiplied by a periodic *strength* reduction factor, and
plotted the product with a stress axis. The flat baseline interrupted by sharp
notches is the signature of that construction; a Brazilian disc has no such
profile.

The distinction matters because the two quantities behave differently. The
elastic field is homogeneous orthotropic and its equilibrium and compatibility
conditions contain no periodic term, so the stress carries no spacing signature
whatever. The spacing enters the tensile resistance, and therefore the tensile
utility, which is where the periodicity actually lives. Plotting them on one
axis as though they were the same quantity is what produced the confusion.

Panel (a) is the maximum principal stress and panel (b) the tensile utility
:math:`R_t`, both sampled along the loading diameter of the :math:`0^\\circ`
specimen, where the foliation lies normal to the sampling line and every trace
of the spacing should appear if it appears anywhere.

    python scripts/make_stress_profile_figures.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fabric_tractions as ft             # noqa: E402
from tools import output_dirs                          # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)        # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
SLUG = {"Augen gneiss": "augen_gneiss", "Psammitic schist": "psammitic_schist"}
SPACING = {"Augen gneiss": 0.010, "Psammitic schist": 0.002}
COLOR = {"Augen gneiss": "#4C72B0", "Psammitic schist": "#DD4B39"}


def profile(f, key):
    """Values of ``key`` along the loading diameter, with the arc-length coordinate."""
    X, Y, M = f["X"], f["Y"], f["M"].astype(bool)
    i = int(np.argmin(np.abs(X[0, :])))
    col = M[:, i]
    return Y[:, i][col], np.asarray(f[key][:, i][col], float)


def count_extrema(v):
    rng = float(np.nanmax(v) - np.nanmin(v))
    pk, _ = find_peaks(v, prominence=0.02 * (rng + 1e-12))
    return len(pk)


def main():
    output_dirs.ensure()
    apply_plot_style()
    fields = {}
    for p in ft.field_files():
        f = ft.load_field(p)
        if abs(f["angle_deg"]) < 1e-9:
            fields[f["rock"]] = f
    if len(fields) < 2:
        raise SystemExit("0 degree fields not found; run the notebooks first")

    for rock, f in fields.items():
        R = float(f["R_m"])
        s = SPACING[rock]
        y, s1 = profile(f, "s1")
        # The spacing is written into the proximity weight, not into R_t: at
        # 0 degrees the foliation is clamped and R_t is the smooth matrix
        # term, so plotting it here annotated the figure with a band count
        # that contradicted the text on the facing page.
        _, wt = profile(f, "wp_weight")
        n_rt = count_extrema(wt)
        expected = 2.0 * R / s

        fig, ax = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
        ax[0].plot(y, s1, color=COLOR[rock], lw=1.8)
        ax[0].set_ylabel(r"$\sigma_1$ (MPa)")
        ax[0].set_title(f"{rock}, fabric angle $0^\\circ$", pad=6)
        ax[0].text(0.02, 0.06,
                   "elastic field is homogeneous:\nno spacing signature",
                   transform=ax[0].transAxes, fontsize=ANNOT_FS, va="bottom",
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))

        ax[1].plot(y, wt, color=COLOR[rock], lw=1.4)
        ax[1].set_ylabel(r"Weak-plane proximity weight $w$")
        ax[1].set_xlabel("Position along loading diameter (m)")
        ax[1].text(0.02, 0.92,
                   f"{n_rt} maxima; $2R/s$ = {expected:.0f}\n"
                   f"spacing $s$ = {s * 1e3:.0f} mm",
                   transform=ax[1].transAxes, fontsize=ANNOT_FS, va="top",
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))

        for a in ax:
            a.grid(True, alpha=0.3)
            for sp in a.spines.values():
                sp.set_linewidth(1.5)
        fig.tight_layout()
        for d in OUT_DIRS:
            fig.savefig(d / f"{SLUG[rock]}_stress_distribution_graph.pdf",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"  {rock:17s} sigma_1 maxima {count_extrema(s1):2d} "
              f"(range {s1.min():.2f} to {s1.max():.2f} MPa);  "
              f"w maxima {n_rt:2d} against 2R/s = {expected:.0f}")

    print("\n  written: <rock>_stress_distribution_graph.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
