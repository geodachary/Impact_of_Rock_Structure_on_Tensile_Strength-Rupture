"""Shared publication style and the seven-panel composite figure builders.

The established visual language of the original figures is preserved — disc
outline, equal aspect, loading arrows on the vertical diameter, faint foliation
rulings, blue observed / red predicted traces — while type sizes and line
widths are raised to remain legible at journal column width, which is the
readability concern raised in review.

Both lithologies use the same functions, the same class-colour mapping and the
same panel geometry, so gneiss and schist figures are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .conventions import RADIUS_M, MM_TO_M
from .failure_classification import CLASS_ORDER, NO_FAILURE
from .plot_style import (PLOT_STYLE, TICK_PARAMS, apply_plot_style, draw_disc, style_axes,
                         panel_label, assert_no_suptitle)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
#: Publication rcParams. Sizes are chosen so a 7-panel composite reproduced at
#: full page width keeps all text at or above 7 pt.
#: Retained for backward compatibility; the authoritative style is
#: tools.plot_style.PLOT_STYLE, which this mirrors.
PUBLICATION_RC = dict(PLOT_STYLE)

#: Trace styles — one definition, used by every overlay panel.
OBSERVED_STYLE = dict(color="#1b6ca8", lw=1.6, marker="o", ms=2.2,
                      mfc="#1b6ca8", mec="none", zorder=4)
PREDICTED_STYLE = dict(color="#c1121f", lw=1.9, zorder=3)
OBSERVED_FULL_STYLE = dict(color="0.72", lw=0.8, zorder=2)
DISC_STYLE = dict(color="0.30", lw=1.3, zorder=1)
FOLIATION_STYLE = dict(color="#c8b89a", lw=0.7, zorder=0)

#: Categorical colours for the four primary mechanisms. Chosen so the two
#: weak-plane classes share a warm family and the two matrix classes a cool one,
#: and so the four remain distinguishable in greyscale by lightness.
#: Class fills. Matrix shear covers most of the disc at low fabric angles, so it
#: carries a mid-tone rather than a dark one: the secondary mixed-mode hatch is
#: drawn in black on top of these fills and was invisible over the previous
#: near-black navy. Matrix tensile is a light green, kept clearly separate in
#: lightness from the mid-blue of matrix shear.
CLASS_COLORS = {
    "WT": "#d1495b",   # weak-plane tensile opening
    "WS": "#edae49",   # weak-plane shear sliding
    "MT": "#a8d5a2",   # matrix tensile
    "MS": "#7ba7d1",   # matrix shear
    NO_FAILURE: "#e9e9e9",
}
CLASS_LABELS = {
    "WT": "WT: weak-plane tensile opening",
    "WS": "WS: weak-plane shear sliding",
    "MT": "MT: matrix tensile",
    "MS": "MS: matrix shear",
    NO_FAILURE: "below failure threshold",
}


def use_publication_style():
    """Apply the mandated publication style. Call once at the top of a notebook."""
    return apply_plot_style()


def _mm(a):
    return np.asarray(a, float) / MM_TO_M


def _draw_disc(ax, foliation_deg=None, show_loading=True, central_frac=None):
    """Disc, clipped foliation lines and loading reference, in millimetres."""
    return draw_disc(ax, RADIUS_M / MM_TO_M, angle_deg=foliation_deg,
                     show_loading=show_loading, central_frac=central_frac)


def _panel_grid(n=7, figsize=(9.0, 12.6)):
    """4x2 grid holding seven specimen panels; the eighth cell carries the legend.

    The canvas is deliberately small. These composites are reproduced two to a
    row at 0.48 of the text block, so a figure drawn at full text width would
    have its labels reduced to about 3 pt on the page. Drawing near the printed
    size keeps the nominal type sizes intact after reduction.
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    return fig, axes.ravel()


# ---------------------------------------------------------------------------
# Seven-panel fracture-trace overlay
# ---------------------------------------------------------------------------
def seven_panel_trace_overlay(rows, lithology, out_stem, formats=("pdf", "png")):
    """Composite observed-vs-predicted overlay for one lithology's seven specimens.

    ``rows`` are :func:`tools.traces.compare_specimen` results for that
    lithology, in fabric-angle order. Panels share equal aspect and common
    physical limits, and a single shared legend replaces seven redundant ones.
    """
    use_publication_style()
    rows = sorted(rows, key=lambda r: r["experimental_angle_deg"])
    if len(rows) != 7:
        raise ValueError(f"expected 7 specimens for {lithology.display_name}, got {len(rows)}")

    fig, axes = _panel_grid()
    for k, (ax, r) in enumerate(zip(axes[:7], rows)):
        # No fit-window ring: orientation is fitted over the whole primary
        # segment, so a circle here would depict a domain nothing uses.
        _draw_disc(ax, foliation_deg=r["experimental_angle_deg"], central_frac=None)
        if r.get("status") == "computed":
            ox, oy = r["_obs_full"]
            ax.plot(_mm(ox), _mm(oy), **OBSERVED_FULL_STYLE)
            oxp, oyp = r["_obs_xy"]
            pxp, pyp = r["_pred_xy"]
            ax.plot(_mm(oxp), _mm(oyp), **OBSERVED_STYLE)
            ax.plot(_mm(pxp), _mm(pyp), **PREDICTED_STYLE)
            sub = (f"obs {r['observed_orientation_deg']:.1f}°, "
                   f"pred {r['predicted_orientation_deg']:.1f}°, "
                   r"$|\Delta\theta|$ = "
                   f"{r['abs_axial_angular_error_deg']:.1f}°")
        else:
            ax.text(0, 0, "not computable", ha="center", va="center", fontsize=8)
            sub = r.get("reason", "")[:48]
        panel_label(ax, f"({chr(97+k)})")
        # same title font and rock-type prefix as the classification composite,
        # so the two seven-panel figures read as one family
        ax.set_title(f"{lithology.display_name},  "
                     r"$\alpha_{\mathrm{exp}}$ = "
                     f"{r['experimental_angle_deg']}°\n{sub}", fontsize=12, pad=6)
        if k >= 5:
            ax.set_xlabel("x (mm)")
        if k % 2 == 0:
            ax.set_ylabel("y (mm)")

    # shared legend in the unused eighth cell
    axes[7].axis("off")
    handles = [
        Line2D([], [], **{**OBSERVED_STYLE, "marker": "o"}, label="Observed primary trace"),
        Line2D([], [], **PREDICTED_STYLE, label="Predicted (DDM) trace"),
        Line2D([], [], **OBSERVED_FULL_STYLE, label="Observed, all digitized segments"),
        Line2D([], [], color="0.55", lw=1.0, ls="-.", label="Loading diameter"),
        Line2D([], [], **FOLIATION_STYLE, label="Foliation trace"),
    ]
    axes[7].legend(handles=handles, loc="center", fontsize=11.5, frameon=True,
                   title=f"{lithology.display_name}", title_fontsize=13)

    # No suptitle: manuscript and supplementary figures take their title from
    # the caption. Verified by assert_no_suptitle below.
    fig.tight_layout()
    assert_no_suptitle(fig)
    paths = _save(fig, out_stem, formats)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Seven-panel WT/WS/MT/MS classification
# ---------------------------------------------------------------------------
def _crack_path_mm(sample_id):
    """Model-derived crack trace for one specimen, in millimetres, or None."""
    from . import lithology as lith
    try:
        path = lith.predicted_trace_path(int(sample_id))
    except Exception:
        return None
    if not Path(path).exists():
        return None
    import pandas as pd
    d = pd.read_csv(path)
    if not {"x_m", "y_m"} <= set(d.columns) or len(d) < 2:
        return None
    if "order" in d.columns:
        d = d.sort_values("order")
    return _mm(d["x_m"].to_numpy()), _mm(d["y_m"].to_numpy())


def seven_panel_classification(panels, lithology, out_stem, formats=("pdf", "png"),
                               show_mixed_overlay=True, show_crack_path=True):
    """Composite four-mechanism classification map for one lithology.

    ``panels`` is a list of dicts with ``sample``, ``angle_deg``, ``X``, ``Y``,
    ``mode_code``, ``mask`` and optionally ``mixed_flag``, in angle order.
    Mixed behaviour appears only as a hatched overlay — never as a fifth colour.

    The model-derived crack trace is drawn over each map, so the mechanism
    field and the path it produces are read together rather than from two
    separate figures.
    """
    use_publication_style()
    panels = sorted(panels, key=lambda p: p["angle_deg"])
    if len(panels) != 7:
        raise ValueError(f"expected 7 panels for {lithology.display_name}, got {len(panels)}")

    from matplotlib.colors import ListedColormap, BoundaryNorm
    order = [NO_FAILURE] + list(CLASS_ORDER)
    cmap = ListedColormap([CLASS_COLORS[c] for c in order])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = _panel_grid()
    drew_path = False
    for k, (ax, p) in enumerate(zip(axes[:7], panels)):
        code = np.array(p["mode_code"], dtype=float)
        mask = np.asarray(p.get("mask", np.ones_like(code, bool)), bool)
        code = np.where(mask, code, np.nan)
        ax.pcolormesh(_mm(p["X"]), _mm(p["Y"]), np.ma.masked_invalid(code),
                      cmap=cmap, norm=norm, shading="auto", zorder=0)
        if show_mixed_overlay and p.get("mixed_flag") is not None:
            mf = np.asarray(p["mixed_flag"], bool) & mask
            if mf.any():
                ax.contourf(_mm(p["X"]), _mm(p["Y"]), mf.astype(float), levels=[0.5, 1.5],
                            colors="none", hatches=["////"], zorder=1)
                ax.contour(_mm(p["X"]), _mm(p["Y"]), mf.astype(float), levels=[0.5],
                           colors="k", linewidths=0.4, zorder=2)
        _draw_disc(ax, foliation_deg=p["angle_deg"], central_frac=None)

        if show_crack_path:
            xy = _crack_path_mm(p["sample"])
            if xy is not None:
                # white casing first so the trace reads over every class fill
                ax.plot(xy[0], xy[1], color="white", lw=2.6, solid_capstyle="round",
                        zorder=5)
                ax.plot(xy[0], xy[1], color="black", lw=1.4, solid_capstyle="round",
                        zorder=6)
                drew_path = True

        panel_label(ax, f"({chr(97+k)})")
        ax.set_title(f"{lithology.display_name},  "
                     r"$\alpha_{\mathrm{exp}}$ = " f"{p['angle_deg']}°",
                     fontsize=12, pad=6)
        if k >= 5:
            ax.set_xlabel("x (mm)")
        if k % 2 == 0:
            ax.set_ylabel("y (mm)")

    axes[7].axis("off")

    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="0.3", label=CLASS_LABELS[c])
               for c in CLASS_ORDER]
    handles.append(Patch(facecolor=CLASS_COLORS[NO_FAILURE], edgecolor="0.3",
                         label=CLASS_LABELS[NO_FAILURE]))
    if show_mixed_overlay:
        handles.append(Patch(facecolor="white", edgecolor="k", hatch="////",
                             label="secondary mixed-mode competition"))
    if drew_path:
        handles.append(Line2D([], [], color="black", lw=1.4,
                              label="model-derived crack path"))

    # boxed legend in the free eighth cell, beside the last panel
    axes[7].legend(handles=handles, loc="center", fontsize=11.5, frameon=True,
                   title=f"{lithology.display_name}", title_fontsize=13)

    fig.tight_layout()
    assert_no_suptitle(fig)
    paths = _save(fig, out_stem, formats)
    plt.close(fig)
    return paths


def _save(fig, out_stem, formats):
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_stem.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt)
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# Strain-energy partitioning across regimes, both lithologies
# ---------------------------------------------------------------------------
def strain_partition_two_rocks(df, out_stem, formats=("pdf", "png")):
    """Stacked WT/WS/MT/MS energy split, both lithologies, three regimes.

    Driven entirely by ``df`` (the output of
    :func:`tools.strain_partitioning.partition_all`) — no hard-coded values, so
    the figure tracks the calculation instead of a pasted snapshot. Schist bars
    are hatched, matching the established figure.
    """
    from .failure_classification import CLASS_ORDER
    from .strain_partitioning import REGIME_OFFSETS_MPA

    use_publication_style()
    g = df[df["status"] == "computed"]
    regimes = list(REGIME_OFFSETS_MPA)
    rocks = ["Augen gneiss", "Psammitic schist"]
    angles = sorted(g["angle_deg"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=True)
    width = 0.38
    for ax, regime in zip(axes, regimes):
        sub = g[g["regime"] == regime]
        x = np.arange(len(angles), dtype=float)
        for k, rock in enumerate(rocks):
            r = sub[sub["lithology"] == rock].set_index("angle_deg").reindex(angles)
            bottom = np.zeros(len(angles))
            off = (k - 0.5) * width
            for cls in CLASS_ORDER:
                vals = r[f"{cls}_pct"].to_numpy(float)
                ax.bar(x + off, vals, width, bottom=bottom,
                       color=CLASS_COLORS[cls], edgecolor="0.25", linewidth=0.6,
                       hatch="///" if rock == "Psammitic schist" else None,
                       label=f"{cls}" if (ax is axes[0] and k == 0) else None)
                bottom += np.nan_to_num(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(a)}" for a in angles])
        ax.set_xlabel(r"$\alpha_{\mathrm{exp}}$ (deg)")
        ax.set_title(f"{regime} ({REGIME_OFFSETS_MPA[regime]:+.2f} MPa)", fontsize=14)
        style_axes(ax)
    axes[0].set_ylabel("share of stored energy (%)")

    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="0.25", label=CLASS_LABELS[c])
               for c in CLASS_ORDER]
    handles += [Patch(facecolor="white", edgecolor="0.25", label="Augen Gneiss (plain)"),
                Patch(facecolor="white", edgecolor="0.25", hatch="///",
                      label="Psammitic Schist (hatched)")]
    # legend across the foot of the figure in a single row: at 15 in wide there
    # is room for all six entries, and the three panels keep equal width instead
    # of losing space to a legend column on the right
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, 0.012), columnspacing=1.4,
               handlelength=1.7, handletextpad=0.6, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    assert_no_suptitle(fig)
    paths = _save(fig, out_stem, formats)
    plt.close(fig)
    return paths
