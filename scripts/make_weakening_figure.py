#!/usr/bin/env python3
"""Foliation weakening factor, drawn from the adopted strength envelope.

The panel this replaces was computed from the earlier activation function
``asym_activation(beta, u)``, which the envelope no longer uses. It therefore
showed a model the paper had already moved away from, and its axis label
(:math:`\\sigma/\\sigma_{\\mathrm{TI}}`) did not match the definition given in the
caption (:math:`\\sigma(\\alpha)/\\sigma(0^\\circ)`).

Both quantities are drawn here, because they answer different questions and the
distinction is easy to lose. The weakening factor

.. math::

    \\mathcal{W}(\\beta) = \\frac{\\sigma(\\beta)}{\\sigma_{\\mathrm{TI}}(\\beta)}
                        = \\bigl[1 + \\eta\\,\\mathcal{H}(\\beta;\\beta_p)\\bigr]^{-1/2}

isolates the weakening from the transversely isotropic interpolation between the
two measured end members. It equals unity at both, so it shows what the fabric
does beyond the stiffness contrast alone, and its minimum lies at
:math:`\\beta_p` with depth :math:`(1+\\eta)^{-1/2}`. The normalised strength
:math:`\\sigma(\\beta)/\\sigma(0^\\circ)` is the directly comparable quantity, and
its minimum need not coincide with :math:`\\beta_p` because it also carries the
end-member contrast.

Bands are the envelope of the curve over the 95% profile-likelihood intervals of
the two fitted parameters, which is the same uncertainty reported for them in
the parameter table.

    python scripts/make_weakening_figure.py
"""
from __future__ import annotations

from pathlib import Path
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ati_model as A                      # noqa: E402
from tools import output_dirs                          # noqa: E402
from tools.data_io import load_replicate_table        # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
COLOR = {"Augen gneiss": "#4C72B0", "Psammitic schist": "#DD4B39"}


def fit_with_intervals(sub):
    r = A.fit(sub)
    e_lo, e_hi, *_ = A.profile_interval(r, "eta", 0.0, 6.0, 0.05)
    b_lo, b_hi, *_ = A.profile_interval(r, "beta_peak", 45.0, 89.0, 1.0)
    r.update(eta_lo=e_lo, eta_hi=e_hi, beta_lo=b_lo, beta_hi=b_hi)
    return r


def main():
    # a clean checkout carries no manuscript directory, so create the
    # output paths before writing into them
    output_dirs.ensure(output_dirs.DOC_DIR, output_dirs.DOC_TABLE_DIR,
                       output_dirs.FIGURE_DIR, output_dirs.TABLE_DIR)
    output_dirs.ensure()
    apply_plot_style()
    d = load_replicate_table()
    beta = np.linspace(0.0, np.pi / 2, 721)
    deg = np.degrees(beta)

    # Two panels side by side, sized so the graphic is reproduced at close to its
    # natural size when included at the full text width.
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    summary = {}

    for rock in ("Augen gneiss", "Psammitic schist"):
        sub = d[d.Rock_type == rock].dropna(subset=["Angle", "Tensile_strength_Mpa"])
        r = fit_with_intervals(sub)
        c = COLOR[rock]

        W = A.weakening_factor(beta, r["eta"], r["beta_peak_deg"])
        corners = [A.weakening_factor(beta, e, b) for e, b in
                   itertools.product((r["eta_lo"], r["eta_hi"]),
                                     (r["beta_lo"], r["beta_hi"]))]
        axes[0].fill_between(deg, np.min(corners, axis=0), np.max(corners, axis=0),
                             color=c, alpha=0.15, lw=0)
        axes[0].plot(deg, W, color=c, lw=2.0, label=rock)
        axes[0].plot(r["beta_peak_deg"], A.weakening_factor(
            np.deg2rad(r["beta_peak_deg"]), r["eta"], r["beta_peak_deg"]),
            "o", color=c, ms=6, zorder=4)

        s = A.strength(beta, r["sigma0"], r["sigma90"], r["eta"], r["beta_peak_deg"])
        sn = s / r["sigma0"]
        sc = [A.strength(beta, r["sigma0"], r["sigma90"], e, b) / r["sigma0"]
              for e, b in itertools.product((r["eta_lo"], r["eta_hi"]),
                                            (r["beta_lo"], r["beta_hi"]))]
        axes[1].fill_between(deg, np.min(sc, axis=0), np.max(sc, axis=0),
                             color=c, alpha=0.15, lw=0)
        axes[1].plot(deg, sn, color=c, lw=2.0, label=rock)

        st = A.per_angle_statistics(sub)
        axes[1].plot(st.index, st["mean"] / r["sigma0"], "o", color=c, ms=5,
                     mfc="white", mew=1.4, zorder=4)

        summary[rock] = dict(
            W_min=float(W.min()), beta_p=r["beta_peak_deg"],
            norm_min=float(sn.min()), norm_min_deg=float(deg[int(np.argmin(sn))]),
            eta=r["eta"])

    axes[0].set_ylabel(r"$\mathcal{W}(\alpha)=\sigma/\sigma_{\mathrm{TI}}$")
    axes[1].set_ylabel(r"$\sigma(\alpha)/\sigma(0^\circ)$")

    for a in axes:
        a.set_xlabel(r"Fabric angle, $\alpha$ (deg)")
        a.set_xticks(range(0, 91, 15)); a.set_xlim(0, 90)
        a.axhline(1.0, color="0.4", ls=":", lw=1.1)
        a.grid(True, alpha=0.3)
        for s_ in a.spines.values():
            s_.set_linewidth(1.5)
    axes[0].legend(frameon=True, fontsize=LEGEND_FS, loc="lower left")

    # panel labels below the axes, matching the placement used by the
    # minipage figures elsewhere in the manuscript
    for a, lab in zip(axes, ("(a)", "(b)")):
        a.text(0.5, -0.30, lab, transform=a.transAxes, ha="center", va="top",
               fontsize=PANEL_LABEL_FS)

    fig.tight_layout(pad=0.8)
    for out in OUT_DIRS:
        # saved on the exact canvas rather than a tight box: matplotlib
        # mismeasures the extent of the calligraphic W in the y-label, and a
        # tight crop shaves its left edge
        fig.savefig(out / "Smooth_weakening_factor_both_rocks.pdf", dpi=300)
    plt.close(fig)

    for rock, v in summary.items():
        print(f"  {rock:17s} W_min = {v['W_min']:.3f} at beta_p = {v['beta_p']:.1f} deg "
              f"(eta = {v['eta']:.3f});  sigma/sigma_0 minimum {v['norm_min']:.3f} "
              f"at {v['norm_min_deg']:.1f} deg")
    print("\n  written: Smooth_weakening_factor_both_rocks.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
