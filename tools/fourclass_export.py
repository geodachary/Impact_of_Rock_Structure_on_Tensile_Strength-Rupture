#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export helpers for the locus-aware 4-class failure classifier
(failure_mode_map_4class in failure_mapping_helpers.py).

ADDITIVE revision module (2026-07): consumed by the new marker-delimited
"4CLASS" blocks in Tensile_augen_gneiss.ipynb (cells 38/42) and by the
standalone verification harness. It reads the
per-specimen record dicts produced by the adjacent failure_mode_map_4class
calls and writes:
  - classifier mapping table  (old->new label mapping table,
    aggregated per specimen x old_label x new_primary_mode; pointwise export
    of ~20k grid points x 7 specimens was deliberately aggregated)
  - figures/

Nothing in the legacy pipeline imports this module.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Okabe-Ito (Wong 2011, Nature Methods) colourblind-safe palette, fixed order.
# Hue family encodes LOCUS (blues = weak plane, warm = matrix);
# lightness within a family separates MECHANISM (tensile dark, shear light).
MODE4_ORDER = ["WT", "WS", "MT", "MS", "no_failure"]
MODE4_COLORS = {
    "WT": "#0072B2",          # weak-plane tensile  (blue)
    "WS": "#56B4E9",          # weak-plane shear    (sky blue)
    "MT": "#D55E00",          # matrix tensile      (vermillion)
    "MS": "#E69F00",          # matrix shear        (orange)
    "no_failure": "#E3E3E3",  # neutral, recessive
}
MODE4_LONG = {
    "WT": "WT: weak-plane tensile",
    "WS": "WS: weak-plane shear",
    "MT": "MT: matrix tensile",
    "MS": "MS: matrix shear",
    "no_failure": "no failure",
}
_LOCUS = {"WT": "weak_plane", "WS": "weak_plane", "MT": "matrix", "MS": "matrix",
          "no_failure": "none"}
_MECH = {"WT": "tensile", "WS": "shear", "MT": "tensile", "MS": "shear",
         "no_failure": "none"}
OLD_ORDER = ["tensile", "shear", "mixed", "no_failure"]


def _mapping_rule(old, new):
    if old == "no_failure" and new == "no_failure":
        return ("old='no_failure' -> new='no_failure'; both classifiers agree the "
                "point-wise utility stays below util_min on the same stress state")
    if old in ("tensile", "shear", "mixed"):
        return (f"old='{old}' (mechanism-only; legacy classifier collapsed locus via "
                f"max(R_mat, R_wp) before labeling) -> new='{new}': locus/mechanism "
                "resolved directly from the same underlying stress state by "
                "failure_mode_map_4class; old label alone was locus-ambiguous")
    return (f"old='{old}' -> new='{new}': set difference from classifier structure "
            "(4-class argmax with hard wp_active gate vs legacy max-collapse with "
            "primary 'mixed' band), same stress input")


def fourclass_mapping_rows(records, source="cell38_production_grid",
                           rmax_frac=0.985):
    """Aggregate per-point old->new labels to contingency rows.

    records: list of dicts with keys sid, rock, ang_deg, R, r_pts,
             modes_old, mode4, mixed_flag, secondary_descriptor.
    Granularity: one row per (specimen x old_label x new_primary_mode) cell
    with point counts and fractions (points restricted to r <= rmax_frac*R,
    matching disk_failure_stats_all_points in cell 38).
    """
    rows = []
    for rec in records:
        keep = np.asarray(rec["r_pts"], float) <= float(rmax_frac) * float(rec["R"])
        old = np.asarray(rec["modes_old"], dtype=object)[keep].astype(str)
        new = np.asarray(rec["mode4"], dtype=object)[keep].astype(str)
        mflag = np.asarray(rec["mixed_flag"], bool)[keep]
        sdesc = np.asarray(rec["secondary_descriptor"], dtype=object)[keep].astype(str)
        n_tot = int(keep.sum())
        for ol in OLD_ORDER:
            m_ol = old == ol
            n_ol = int(m_ol.sum())
            if n_ol == 0:
                continue
            for nl in MODE4_ORDER:
                m = m_ol & (new == nl)
                n = int(m.sum())
                if n == 0:
                    continue
                sd_vals, sd_cnt = np.unique(sdesc[m], return_counts=True)
                rows.append(dict(
                    sample_id=int(rec["sid"]),
                    rock_type=str(rec["rock"]),
                    angle_deg=float(rec["ang_deg"]),
                    old_label=ol,
                    new_primary_mode=nl,
                    locus=_LOCUS[nl],
                    mechanism=_MECH[nl],
                    n_points=n,
                    frac_of_specimen=n / max(n_tot, 1),
                    frac_of_old_label=n / max(n_ol, 1),
                    mixed_flag=bool(np.mean(mflag[m]) > 0.5),
                    mixed_flag_frac=float(np.mean(mflag[m])),
                    secondary_descriptor=str(sd_vals[np.argmax(sd_cnt)]),
                    ambiguous=not (ol == "no_failure" and nl == "no_failure"),
                    mapping_rule=_mapping_rule(ol, nl),
                    source=source,
                    granularity=f"per-specimen aggregate over grid points with r<={rmax_frac}R",
                ))
    return rows


def write_classifier_mapping_csv(records, path,
                                 source="cell38_production_grid", rmax_frac=0.985):
    rows = fourclass_mapping_rows(records, source=source, rmax_frac=rmax_frac)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def plot_fourclass_maps(records, out_pdf, out_png,
                        lith_title=None, rmax_frac=0.985,
                        point_size=15, ncols=2):
    """Spatial WT/WS/MT/MS/no-failure maps, one panel per specimen.

    ``lith_title`` names the rock in the figure title. It defaulted to a
    literal "Augen gneiss" and no caller overrode it, so the psammitic schist
    map was published under the other rock's name while its filename and every
    panel title said schist. Taking the name from the records removes the
    second source of truth; passing a string explicitly is still allowed.
    """
    if lith_title is None:
        names = {str(r["rock"]) for r in records}
        lith_title = names.pop() if len(names) == 1 else "Brazilian disc"
    n = len(records)
    nrows = int(np.ceil((n + 1) / ncols))  # +1 slot for the legend panel
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols + 0.6, 4.3 * nrows),
                            constrained_layout=False)
    axs = np.atleast_2d(axs)
    for k, rec in enumerate(records):
        ax = axs.flat[k]
        R = float(rec["R"])
        xg = np.asarray(rec["xg"], float) * 1e3  # m -> mm
        yg = np.asarray(rec["yg"], float) * 1e3
        keep = np.asarray(rec["r_pts"], float) <= rmax_frac * R
        m4 = np.asarray(rec["mode4"], dtype=object).astype(str)
        # draw recessive class first, then the four failure classes on top
        for lab in ["no_failure", "MS", "MT", "WS", "WT"]:
            sel = keep & (m4 == lab)
            if not np.any(sel):
                continue
            ax.scatter(xg[sel], yg[sel], s=point_size, c=MODE4_COLORS[lab],
                       edgecolors="k" if lab != "no_failure" else "none",
                       linewidths=0.10, zorder=2 if lab != "no_failure" else 1)
        ax.add_artist(plt.Circle((0, 0), R * 1e3, fill=False, color="k",
                                 lw=1.1, zorder=3))
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.06 * R * 1e3, 1.06 * R * 1e3)
        ax.set_ylim(-1.06 * R * 1e3, 1.06 * R * 1e3)
        ax.set_title(f"{rec['rock']} | fabric angle = {float(rec['ang_deg']):.0f}°",
                     fontsize=10, pad=5)
        row, col = divmod(k, ncols)
        if row == nrows - 1 or (k + ncols) >= n:
            ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")
    # legend panel
    axL = axs.flat[n]
    axL.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=MODE4_COLORS[lab],
                          markeredgecolor="k" if lab != "no_failure" else "none",
                          markeredgewidth=0.3, markersize=9,
                          label=MODE4_LONG[lab])
               for lab in MODE4_ORDER]
    axL.legend(handles=handles, loc="center", frameon=True, edgecolor="0.6",
               fontsize=9.5, title="4-class failure mode (locus + mechanism)",
               title_fontsize=10)
    for k in range(n + 1, nrows * ncols):
        axs.flat[k].axis("off")
    fig.suptitle(
        f"Locus-resolved failure-mode classification, {lith_title} Brazilian discs: "
        "weak-plane (WT/WS) vs matrix (MT/MS) failure from failure_mode_map_4class "
        "on the production Airy/Lekhnitskii stress fields",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_pdf, out_png
