#!/usr/bin/env python3
"""Angle dependence of the failure statistics, counted from the exported fields.

The two panels were previously drawn from mode counts written into the notebook
as literal lists. Those lists cannot have come from the classifier: the number of
points in the disc is fixed by the grid and cannot change with orientation, yet
their per-angle totals run 4580, 4548, 4564, 3684, 2980, 1876, 1876, and the
mixed count coincides exactly with the tensile count at three of the seven
angles and with the shear count at two more. They also disagree with the fields
as computed, most visibly at 75 degrees where they give a failed fraction of 7
per cent against 28 per cent counted directly.

This script counts the four outcomes from ``mode_code`` in the exported fields,
so the panels move when the model does.

    python scripts/make_failure_statistics_figures.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fabric_tractions as ft             # noqa: E402
from tools import output_dirs  # noqa: E402
from tools import failure_classification as fc       # noqa: E402
from tools import strain_partitioning as sp          # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)        # noqa: E402
from tools.plotting import CLASS_COLORS              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
STYLE = {"Augen gneiss": dict(color="#4C72B0", ls="-", marker="o"),
         "Psammitic schist": dict(color="#DD4B39", ls="--", marker="s")}


def counts() -> pd.DataFrame:
    """Mode counts over the disc interior for every specimen."""
    rows = []
    for p in ft.field_files():
        f = ft.load_field(p)
        m = ft.core_mask(f)
        labels = [str(x) for x in f["mode_labels"]]
        idx = {l: i for i, l in enumerate(labels)}
        c = f["mode_code"][m]
        n = {k: int(np.sum(c == v)) for k, v in idx.items()}
        failed = n["tensile"] + n["shear"] + n["mixed"]
        rows.append(dict(rock=f["rock"], angle_deg=f["angle_deg"], total=int(c.size),
                         failed=failed, **n,
                         p_fail=100.0 * failed / c.size,
                         tensile_cond=100.0 * n["tensile"] / max(failed, 1),
                         shear_cond=100.0 * n["shear"] / max(failed, 1),
                         mixed_cond=100.0 * n["mixed"] / max(failed, 1)))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)


def class_fractions() -> pd.DataFrame:
    """Area fraction of each of the four mechanism-locus classes, per specimen."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("ra", REPO / "scripts/reproduce_all.py")
    ra = ilu.module_from_spec(spec); spec.loader.exec_module(ra)
    strengths = sp.specimen_strengths()
    rows = []
    for sid in range(1, 15):
        r = ra._classify_panel(sid, strengths)
        if r.get("status") != "computed":
            continue
        mc = r["mode_code"][r["mask"]]
        rows.append(dict(rock=r["lithology"], angle_deg=float(r["angle_deg"]),
                         **{c: float(np.mean(mc == fc.CLASS_CODES[c]))
                            for c in fc.CLASS_ORDER},
                         none=float(np.mean(mc == fc.CLASS_CODES["none"]))))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)


def main():
    apply_plot_style()
    t = counts()
    cls = class_fractions()
    cls.to_csv(REPO / output_dirs.TABLE_DIR / "fourclass_area_fractions.csv", index=False)

    # the grid is the same for every specimen, so this must hold
    assert t.total.nunique() == 1, f"point totals differ between specimens: {sorted(t.total.unique())}"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for rock, g in cls.groupby("rock"):
        g = g.sort_values("angle_deg")
        ls = STYLE[rock]["ls"]
        for c in fc.CLASS_ORDER:
            ax.plot(g.angle_deg, g[c], marker=STYLE[rock]["marker"], ls=ls, lw=1.7,
                    ms=4.5, color=CLASS_COLORS[c],
                    label=f"{c} ({rock})")
    ax.set_xlabel(r"Foliation angle, $\alpha$ (deg)")
    ax.set_ylabel("Area fraction of disc interior")
    ax.set_xticks(range(0, 91, 15)); ax.grid(True, alpha=0.3)
    # Below the axes. Eight entries inside the box sat over the top-left of the
    # plot, which is exactly where the schist MS curve starts at 0.76; moving
    # the legend out returns that corner to the data.
    #
    # Two columns rather than four: these labels are long, and four columns made
    # the legend wider than the axes, which then shrank to fit it. The anchor
    # clears the x-axis label rather than sitting on it.
    ax.legend(fontsize=LEGEND_FS, ncol=2, frameon=True,
              loc="upper center", bbox_to_anchor=(0.5, -0.20),
              columnspacing=1.4, handlelength=2.2, borderaxespad=0.0)
    for sp_ in ax.spines.values():
        sp_.set_linewidth(1.5)
    fig.tight_layout()
    for d in OUT_DIRS:
        # bbox_inches="tight" keeps the legend, which now sits outside the axes.
        fig.savefig(d / "conditional_failure_modes_vs_theta.pdf", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for rock, g in t.groupby("rock"):
        g = g.sort_values("angle_deg")
        st = STYLE[rock]
        ax.plot(g.angle_deg, g.p_fail, marker=st["marker"], ls=st["ls"], lw=2.0,
                ms=6, color=st["color"], label=rock)
    ax.set_xlabel(r"Foliation angle, $\alpha$ (deg)")
    ax.set_ylabel("Failed fraction of disc interior (%)")
    ax.set_xticks(range(0, 91, 15)); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=LEGEND_FS)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "pfailure_vs_theta.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_dirs.tables()
    t.to_csv(REPO / output_dirs.TABLE_DIR / "failure_statistics.csv", index=False)
    for rock, g in t.groupby("rock"):
        g = g.sort_values("angle_deg")
        print(f"  {rock}")
        print("     p_fail  " + "  ".join(f"{a:.0f}d={v:.0f}%" for a, v in zip(g.angle_deg, g.p_fail)))
    for rock, g in cls.groupby("rock"):
        g = g.sort_values("angle_deg")
        print(f"  {rock} four-class peaks: " + ", ".join(
            f"{c} {g[c].max():.3f} at {g.loc[g[c].idxmax(),'angle_deg']:.0f}deg"
            for c in fc.CLASS_ORDER))
    print("\n  written: conditional_failure_modes_vs_theta.pdf, pfailure_vs_theta.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
