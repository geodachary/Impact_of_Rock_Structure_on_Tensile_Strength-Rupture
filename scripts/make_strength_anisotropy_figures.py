#!/usr/bin/env python3
"""The two panels of the strength-anisotropy figure, from the replicate data.

Both were orphans. ``ucs_vs_loading_angle_with_ci.pdf`` and
``combined_strength_anisotropy.pdf`` existed only as PDFs inside
``manuscript/``, last written on 2026-08-11, with no script, tool or notebook
anywhere in the repository able to regenerate them. The manuscript cites both.
A reader could not check them, a data revision could not reach them, and the
figure-sync stage had nothing in ``outputs/figures`` to copy, so the staleness
guards could not see them either.

They are rebuilt here from ``selected_all_samples.csv``, the same file every
other strength statistic is computed from, so they now follow the data.

Panel (a) is uniaxial compressive strength against fabric angle with 95%
confidence intervals on the per-angle mean. Panel (b) is the strength
anisotropy ratio, the largest per-angle mean over the smallest across the seven
tested orientations, for both loading modes.

    python scripts/make_strength_anisotropy_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import output_dirs                          # noqa: E402
from tools.plot_style import apply_plot_style, LEGEND_FS   # noqa: E402

RAW = REPO / "selected_all_samples.csv"
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
STYLE = {"Augen gneiss": dict(color="#4C72B0", ls="-", marker="o"),
         "Psammitic schist": dict(color="#DD4B39", ls="--", marker="s")}
UCS, TEN = "UCS_(Mpa)", "Tensile_strength_Mpa"


def per_angle(d, col):
    """Mean, standard error and 95% half-width of the mean, by angle."""
    g = d.groupby("Angle")[col]
    out = g.agg(n="count", mean="mean", sd="std")
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    # Student t, because three to five replicates is not a normal approximation
    out["ci95"] = out["sem"] * stats.t.ppf(0.975, out["n"] - 1)
    return out


def ratios(d, col):
    """Largest per-angle mean over the smallest, per lithology."""
    return {rock: float(g.groupby("Angle")[col].mean().max()
                        / g.groupby("Angle")[col].mean().min())
            for rock, g in d.groupby("Rock_type")}


def panel_ucs(ax, d):
    for rock, g in d.groupby("Rock_type"):
        st, s = STYLE[rock], per_angle(g, UCS)
        ax.errorbar(s.index, s["mean"], yerr=s["ci95"], color=st["color"],
                    ls=st["ls"], marker=st["marker"], lw=2.0, ms=5, capsize=3,
                    elinewidth=1.2, label=rock)
        lo = int(s["mean"].idxmin())
        ax.annotate(f"min {lo}$^\\circ$", (lo, s["mean"].loc[lo]),
                    textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=LEGEND_FS, color=st["color"])
    ax.set_xlabel(r"Fabric angle $\alpha$ (deg)")
    ax.set_ylabel("Uniaxial compressive strength (MPa)")
    ax.set_xticks(range(0, 91, 15))


def panel_ratios(ax, d):
    t, u = ratios(d, TEN), ratios(d, UCS)
    rocks = ["Augen gneiss", "Psammitic schist"]
    x = np.arange(len(rocks)); w = 0.34
    ax.bar(x - w / 2, [t[r] for r in rocks], w, label="Tensile",
           color="#4C72B0", edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, [u[r] for r in rocks], w, label="Compressive",
           color="#BBBBBB", edgecolor="black", linewidth=0.8, hatch="//")
    for i, r in enumerate(rocks):
        ax.text(i - w / 2, t[r] + 0.04, f"{t[r]:.2f}", ha="center", fontsize=LEGEND_FS)
        ax.text(i + w / 2, u[r] + 0.04, f"{u[r]:.2f}", ha="center", fontsize=LEGEND_FS)
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(rocks)
    ax.set_ylabel(r"Anisotropy ratio $\sigma_{\max}/\sigma_{\min}$")
    ax.set_ylim(0, max(list(t.values()) + list(u.values())) * 1.18)


def main() -> int:
    apply_plot_style()
    output_dirs.ensure(output_dirs.DOC_DIR, output_dirs.FIGURE_DIR,
                       output_dirs.TABLE_DIR)
    d = pd.read_csv(RAW)

    for name, draw, figsize in (("ucs_vs_loading_angle_with_ci", panel_ucs, (5.2, 4.0)),
                                ("combined_strength_anisotropy", panel_ratios, (5.2, 4.0))):
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax, d)
        ax.grid(True, alpha=0.3)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.legend(fontsize=LEGEND_FS, frameon=False)
        fig.tight_layout()
        for out in OUT_DIRS:
            fig.savefig(out / f"{name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    t, u = ratios(d, TEN), ratios(d, UCS)
    rows = [dict(rock=r, tensile_ratio=t[r], compressive_ratio=u[r],
                 ucs_min_angle_deg=int(d[d.Rock_type == r].groupby("Angle")[UCS].mean().idxmin()),
                 tensile_min_angle_deg=int(d[d.Rock_type == r].groupby("Angle")[TEN].mean().idxmin()))
            for r in t]
    tab = pd.DataFrame(rows)
    tab.to_csv(Path(output_dirs.tables()) / "strength_anisotropy_ratios.csv", index=False)

    for r in rows:
        print(f"  {r['rock']:17s} tensile {r['tensile_ratio']:.3f} "
              f"(min {r['tensile_min_angle_deg']} deg)   compressive "
              f"{r['compressive_ratio']:.3f} (min {r['ucs_min_angle_deg']} deg)")
    print("\n  written: ucs_vs_loading_angle_with_ci.pdf, "
          "combined_strength_anisotropy.pdf,\n           "
          "outputs/tables/strength_anisotropy_ratios.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
