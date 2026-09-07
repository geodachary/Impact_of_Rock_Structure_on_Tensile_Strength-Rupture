#!/usr/bin/env python3
"""Strength envelope against fabric angle, with bootstrap intervals and residuals.

The panel this replaces was fitted with four free parameters -- both end-member
strengths as well as the depth and position of the weakening. The envelope the
paper adopts fits two: the end members are measured, not estimated, and the
curve is required to pass through them. The extra freedom is why the earlier fit
reported a higher coefficient of determination for the schist (0.931 against
0.912), and it left the figure disagreeing with the statistics quoted in the
text and with the claim that only two parameters are fitted.

Everything here comes from :mod:`tools.ati_model`, so the curve, the intervals
and the reported statistics are the ones the manuscript quotes.

Intervals are bootstrap: replicates are resampled within each fabric angle, the
end members are recomputed from the resample as measured quantities, and the two
free parameters are refitted. That propagates measurement scatter through both
the fixed and the fitted parts of the model.

    python scripts/make_envelope_fit_figure.py [--boot N]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ati_model as A                       # noqa: E402
from tools import output_dirs                          # noqa: E402
from tools.data_io import load_replicate_table         # noqa: E402
from tools.plot_style import apply_plot_style, ANNOT_FS, LEGEND_FS   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
STYLE = {"Augen gneiss": dict(color="#4C72B0", marker="o", ls="-"),
         "Psammitic schist": dict(color="#DD4B39", marker="s", ls="--")}
SEED = 20260807


def bootstrap_band(sub, beta, n_boot, seed=SEED):
    """Pointwise 95% band from resampling replicates within each fabric angle."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in sub.groupby(sub["Angle"].round().astype(int))]
    curves = []
    for _ in range(int(n_boot)):
        draw = [g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31)))
                for g in groups]
        rs = pd.concat(draw)
        try:
            f = A.fit(rs)
        except Exception:
            continue
        curves.append(A.strength(beta, f["sigma0"], f["sigma90"],
                                 f["eta"], f["beta_peak_deg"]))
    c = np.asarray(curves, float)
    return np.percentile(c, 2.5, axis=0), np.percentile(c, 97.5, axis=0), len(c)


def main(n_boot):
    # a clean checkout carries no manuscript directory, so create the
    # output paths before writing into them
    output_dirs.ensure(output_dirs.DOC_DIR, output_dirs.DOC_TABLE_DIR,
                       output_dirs.FIGURE_DIR, output_dirs.TABLE_DIR)
    output_dirs.ensure()
    apply_plot_style()
    d = load_replicate_table()
    beta = np.linspace(0.0, np.pi / 2, 361)
    deg = np.degrees(beta)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    labels = []

    for rock, st in STYLE.items():
        sub = d[d.Rock_type == rock].dropna(subset=["Angle", "Tensile_strength_Mpa"])
        f = A.fit(sub)
        curve = A.strength(beta, f["sigma0"], f["sigma90"], f["eta"], f["beta_peak_deg"])
        lo, hi, n_ok = bootstrap_band(sub, beta, n_boot)

        axes[0].scatter(sub.Angle, sub.Tensile_strength_Mpa, s=26, alpha=0.25,
                        color=st["color"], lw=0, zorder=2)
        stats = A.per_angle_statistics(sub)
        axes[0].errorbar(stats.index, stats["mean"], yerr=stats["sem"], fmt=st["marker"],
                         ms=6, capsize=3, color=st["color"], zorder=4,
                         label=f"{rock} mean $\\pm$ SEM")
        axes[0].fill_between(deg, lo, hi, color=st["color"], alpha=0.18, lw=0, zorder=1)
        axes[0].plot(deg, curve, color=st["color"], ls=st["ls"], lw=2.2, zorder=3)
        axes[0].axvline(f["model_min_deg"], color=st["color"], ls=":", lw=1.1, alpha=0.7)

        resid = sub.Tensile_strength_Mpa.to_numpy(float) - A.strength(
            np.deg2rad(sub.Angle.to_numpy(float)), f["sigma0"], f["sigma90"],
            f["eta"], f["beta_peak_deg"])
        axes[1].scatter(sub.Angle, resid, s=34, alpha=0.75, color=st["color"],
                        marker=st["marker"], lw=0)

        labels.append(f"{rock}: $R^2$ {f['R2']:.3f}, "
                      r"$\chi^2_{\mathrm{red}}$ " f"{f['chi2_red']:.2f}, "
                      f"minimum {f['model_min_deg']:.1f}$^\\circ$")
        print(f"  {rock:17s} R2={f['R2']:.3f}  chi2red={f['chi2_red']:.2f}  "
              f"eta={f['eta']:.3f}  beta_p={f['beta_peak_deg']:.1f}  "
              f"({n_ok}/{n_boot} bootstrap fits converged)")

    axes[0].set_xlabel(r"Fabric angle, $\alpha$ (deg)")
    axes[0].set_ylabel("Tensile strength (MPa)")
    axes[0].legend(frameon=True, fontsize=LEGEND_FS, loc="lower left")
    axes[1].axhline(0.0, color="0.3", ls="--", lw=1.2)
    axes[1].set_xlabel(r"Fabric angle, $\alpha$ (deg)")
    axes[1].set_ylabel("Residual (MPa)")

    for a in axes:
        a.set_xticks(range(0, 91, 15))
        a.set_xlim(-4, 94)
        a.grid(True, alpha=0.3)
        for s in a.spines.values():
            s.set_linewidth(1.5)

    fig.text(0.5, 0.005, "   |   ".join(labels), ha="center", va="bottom",
             fontsize=ANNOT_FS,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.55"))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    for out in OUT_DIRS:
        fig.savefig(out / "Anisotropic_model_smooth_endpoint_both_rocks.pdf", dpi=300)
    plt.close(fig)
    print("\n  written: Anisotropic_model_smooth_endpoint_both_rocks.pdf")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    raise SystemExit(main(ap.parse_args().boot))
