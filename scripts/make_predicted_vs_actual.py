#!/usr/bin/env python3
"""Predicted-versus-measured tensile strength for both lithologies.

Driven by :mod:`tools.ati_model`, so the panel always shows the same envelope
that the Methods section defines and the parameter table reports. The earlier
notebook version carried its own copy of the model and fell out of step when the
envelope was revised.

    python scripts/make_predicted_vs_actual.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ati_model as A                   # noqa: E402
from tools import output_dirs                          # noqa: E402
from tools.data_io import load_replicate_table     # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)      # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STYLE = {"Augen gneiss": dict(color="#4C72B0", marker="o"),
         "Psammitic schist": dict(color="#DD4B39", marker="s")}


def main():
    output_dirs.ensure()
    apply_plot_style()
    d = load_replicate_table()
    # drawn at the width it is reproduced at, so labels keep their size
    # square canvas: the axes are equal-aspect, and a smaller one leaves the
    # rotated y-label taller than the page once the tight box is applied
    fig, ax = plt.subplots(figsize=(4.6, 5.0))

    obs_all, pred_all, lines = [], [], []
    for rock, sty in STYLE.items():
        sub = d[d.Rock_type == rock].dropna(subset=["Angle", "Tensile_strength_Mpa"])
        r = A.fit(sub)
        obs = sub.Tensile_strength_Mpa.to_numpy(float)
        pred = A.strength(np.deg2rad(sub.Angle.to_numpy(float)),
                          r["sigma0"], r["sigma90"], r["eta"], r["beta_peak_deg"])
        obs_all.append(obs)
        pred_all.append(pred)
        ax.scatter(obs, pred, s=70, alpha=0.75, edgecolors=sty["color"],
                   facecolors=sty["color"], marker=sty["marker"], label=rock, zorder=3)
        # rock names written in full, not abbreviated
        lines.append(f"{rock}: RMSE {r['RMSE']:.3f} MPa, "
                     f"$R^2$ {r['R2']:.3f}, "
                     r"$\chi^2_{\mathrm{red}}$ " f"{r['chi2_red']:.2f}")

    obs = np.concatenate(obs_all)
    pred = np.concatenate(pred_all)
    lo, hi = 0.9 * min(obs.min(), pred.min()), 1.05 * max(obs.max(), pred.max())
    slope, intercept = np.polyfit(obs, pred, 1)
    xs = np.linspace(lo, hi, 100)

    ax.plot([lo, hi], [lo, hi], ls="--", color="#c1121f", lw=1.6,
            label="1:1 line", zorder=2)
    ax.plot(xs, slope * xs + intercept, color="#2a7f3f", lw=2.0,
            label=f"OLS: $y$ = {slope:.2f}$x$ + {intercept:.2f}", zorder=2)

    # Statistics in a box at the bottom right: that corner is below the 1:1
    # diagonal and carries no points.
    ax.text(0.97, 0.03, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=ANNOT_FS,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.55", alpha=0.95))
    ax.set_xlabel("Measured tensile strength (MPa)")
    ax.set_ylabel("Predicted tensile strength (MPa)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Legend below the axes: with the statistics occupying the lower right and
    # points along the whole diagonal, no interior placement stays clear.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=True, fontsize=LEGEND_FS)
    for s in ax.spines.values():
        s.set_linewidth(1.5)

    fig.tight_layout()
    for stem in (REPO / output_dirs.FIGURE_DIR / "predicted_vs_actual_both_rocks",
                 REPO / "manuscript/predicted_vs_actual_both_rocks"):
        fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  ordinary least squares: slope {slope:.3f}, intercept {intercept:.3f} MPa")
    for l in lines:
        print("  " + l.replace("$R^2$", "R2").replace(r"$\chi^2_{\mathrm{red}}$", "chi2_red"))
    print("\nwritten: manuscript/predicted_vs_actual_both_rocks.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
