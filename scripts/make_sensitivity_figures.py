#!/usr/bin/env python3
"""Sensitivity of the failed area to weak-plane spacing and to strength disorder.

Both panels were previously drawn from per-specimen values written into a
notebook as literal arrays. They are computed here from the exported fields, so
the curves follow the model.

    python scripts/make_sensitivity_figures.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import sensitivity_sweeps as sw            # noqa: E402
from tools import output_dirs  # noqa: E402
from tools.plot_style import apply_plot_style         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
STYLE = {"Augen gneiss": dict(color="#4C72B0", ls="-", marker="o"),
         "Psammitic schist": dict(color="#DD4B39", ls="--", marker="s")}


def panel(ax, df, xcol, xlabel):
    """Mean +/- one standard deviation across the seven specimens of a lithology."""
    for rock, g in df.groupby("rock"):
        st = STYLE[rock]
        m = g.groupby(xcol).delta_failure_pct.mean()
        s = g.groupby(xcol).delta_failure_pct.std(ddof=1)
        ax.fill_between(m.index, m - s, m + s, color=st["color"], alpha=0.15, lw=0)
        ax.plot(m.index, m, ls=st["ls"], marker=st["marker"], lw=2.0, ms=5,
                color=st["color"], label=rock)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta$Failure (%)")
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)


def main():
    apply_plot_style()
    sp = sw.spacing_sweep()
    het = sw.heterogeneity_sweep()
    output_dirs.tables()
    sp.to_csv(REPO / output_dirs.TABLE_DIR / "sensitivity_spacing.csv", index=False)
    het.to_csv(REPO / output_dirs.TABLE_DIR / "sensitivity_heterogeneity.csv", index=False)

    for df, xcol, xlabel, stem in (
            (sp, "spacing_m", "Weak-plane spacing, $s$ (m)", "delta_failure_vs_spacing"),
            (het, "heterogeneity", "Strength disorder amplitude, $h$ (--)",
             "delta_failure_vs_heterogeneity")):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        panel(ax, df, xcol, xlabel)
        # Below the axes rather than inside it: these panels are 3.6 inches
        # wide and the curves reach the corners, so an in-axes legend sat on
        # the data at every placement matplotlib chose.
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        for d in OUT_DIRS:
            fig.savefig(d / f"{stem}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    for name, df, xcol in (("spacing", sp, "spacing_m"),
                           ("heterogeneity", het, "heterogeneity")):
        print(f"  {name}:")
        for rock, g in df.groupby("rock"):
            m = g.groupby(xcol).delta_failure_pct.mean()
            print(f"    {rock:17s} " + "  ".join(f"{k:g}:{v:+.1f}%" for k, v in m.items()))
    # do the two controls act alike on the two lithologies?
    for name, df, xcol in (("spacing", sp, "spacing_m"),
                           ("heterogeneity", het, "heterogeneity")):
        a, b = [g.groupby(xcol).delta_failure_pct.mean() for _, g in df.groupby("rock")]
        print(f"  max |gneiss - schist| in {name}: {float((a - b).abs().max()):.1f} "
              "percentage points")
    print("\n  written: delta_failure_vs_spacing.pdf, delta_failure_vs_heterogeneity.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
