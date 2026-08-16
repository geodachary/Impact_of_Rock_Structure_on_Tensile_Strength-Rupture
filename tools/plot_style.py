"""Authoritative publication plot style and disc-geometry plotting helpers.

Every manuscript, supplementary and validation figure imports its style from
here, so a single edit changes all of them consistently.

Two rules that are easy to violate accidentally and are therefore enforced by
helpers rather than by convention:

* **No main title.** Manuscript and supplementary figures carry their title in
  the caption, never in the artwork. :func:`apply_plot_style` does not set one,
  and :func:`assert_no_suptitle` can be used in tests.
* **Foliation guide lines are clipped to the disc.** A foliation line drawn at
  an angle must terminate on the specimen boundary, never overshoot it.
  :func:`foliation_chord` returns the chord endpoints analytically.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Mandatory style
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.linewidth": 1.5,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False,
}

TICK_PARAMS = {
    "major": {"which": "major", "direction": "out", "length": 5, "width": 1.5},
    "minor": {"which": "minor", "direction": "out", "length": 3, "width": 1},
}

#: Additional settings that do not belong in the mandated dict but are required
#: for a reproducible, publication-grade vector export.
_EXPORT_DEFAULTS = {
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,     # embed TrueType so text stays selectable
    "ps.fonttype": 42,
}


def apply_plot_style(strict_font: bool = False):
    """Apply the mandated style. Returns :data:`TICK_PARAMS` for convenience.

    Parameters
    ----------
    strict_font
        If True, raise when Times New Roman is unavailable instead of allowing
        Matplotlib's silent fallback. Default False so a missing system font
        degrades gracefully rather than breaking a batch run; the substitution
        is reported by :func:`resolved_font`.
    """
    mpl.rcParams.update(PLOT_STYLE)
    mpl.rcParams.update(_EXPORT_DEFAULTS)
    if strict_font and not times_new_roman_available():
        raise RuntimeError(
            "Times New Roman is not available to Matplotlib; install the font "
            "or call apply_plot_style(strict_font=False) to accept a fallback.")
    return TICK_PARAMS


def times_new_roman_available() -> bool:
    """Whether Matplotlib can resolve Times New Roman on this machine."""
    from matplotlib import font_manager
    try:
        path = font_manager.findfont("Times New Roman", fallback_to_default=False)
        return bool(path)
    except Exception:
        return False


def resolved_font() -> str:
    """The font family actually used, so a silent fallback is visible."""
    from matplotlib import font_manager
    try:
        return font_manager.FontProperties(
            family=mpl.rcParams["font.family"]).get_name()
    except Exception:
        return "unknown"


def style_axes(ax, tick_params=None):
    """Apply the mandated major and minor tick styling to one axis."""
    tp = tick_params or TICK_PARAMS
    ax.tick_params(**tp["major"])
    ax.tick_params(**tp["minor"])
    ax.minorticks_on()
    return ax


# ---------------------------------------------------------------------------
# Disc geometry
# ---------------------------------------------------------------------------
def foliation_chord(angle_deg, radius, offset=0.0):
    """Endpoints of a foliation guide line **clipped to the disc boundary**.

    The line has direction ``angle_deg`` (degrees counterclockwise from +x) and
    is displaced by ``offset`` along its own normal. Returns the chord of the
    circle of radius ``radius`` centred at the origin, so the line never
    extends beyond the specimen.

    Returns
    -------
    (x0, x1), (y0, y1) or ``None`` when ``|offset| >= radius`` (the line misses
    the disc entirely and must not be drawn).

    Geometry: a chord at perpendicular distance ``d`` from the centre has
    half-length ``sqrt(R^2 - d^2)``.
    """
    a = np.radians(float(angle_deg))
    d = float(offset)
    R = float(radius)
    if abs(d) >= R:
        return None
    half = np.sqrt(R * R - d * d)
    ux, uy = np.cos(a), np.sin(a)          # along the line
    nx, ny = -np.sin(a), np.cos(a)         # normal to the line
    cx, cy = d * nx, d * ny                # closest point to the centre
    return ((cx - half * ux, cx + half * ux), (cy - half * uy, cy + half * uy))


def draw_foliation_lines(ax, angle_deg, radius, n_lines=9, spacing_frac=0.30, **kw):
    """Draw evenly spaced foliation guide lines, each clipped to the disc."""
    style = dict(color="#c8b89a", lw=0.9, zorder=0)
    style.update(kw)
    drawn = 0
    for k in range(-(n_lines // 2), n_lines // 2 + 1):
        chord = foliation_chord(angle_deg, radius, k * spacing_frac * radius)
        if chord is None:
            continue
        (x0, x1), (y0, y1) = chord
        ax.plot([x0, x1], [y0, y1], **style)
        drawn += 1
    return drawn


def draw_disc(ax, radius, angle_deg=None, show_loading=True, central_frac=None,
              n_foliation=9):
    """Specimen boundary, clipped foliation lines and the loading reference.

    Establishes equal aspect and symmetric limits, so every disc panel in the
    package shares one geometry.
    """
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(radius * np.cos(th), radius * np.sin(th), color="0.25", lw=1.5, zorder=2)
    if central_frac:
        ax.add_patch(plt.Circle((0, 0), central_frac * radius, fill=False, ls=":",
                                lw=1.0, ec="0.55", zorder=2))
    if angle_deg is not None:
        draw_foliation_lines(ax, angle_deg, radius, n_lines=n_foliation)
    if show_loading:
        ax.plot([0, 0], [-radius * 1.16, radius * 1.16], color="0.45", lw=1.2,
                ls="-.", zorder=1)
        for sgn in (1, -1):
            ax.annotate("", xy=(0, sgn * radius * 1.02),
                        xytext=(0, sgn * radius * 1.26),
                        arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.4))
    ax.set_aspect("equal", adjustable="box")
    lim = radius * 1.32
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    style_axes(ax)
    return ax


def panel_label(ax, text, loc=(0.02, 0.97), **kw):
    """Consistent ``(a)``-style panel label in axes coordinates."""
    style = dict(ha="left", va="top", fontsize=14, fontweight="bold",
                 transform=ax.transAxes, zorder=10)
    style.update(kw)
    return ax.text(loc[0], loc[1], text, **style)


def assert_no_suptitle(fig):
    """Manuscript figures must not carry a main title; the caption supplies it."""
    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_text().strip():
        raise AssertionError(f"figure carries a suptitle: {st.get_text()!r}")
    return True


#: Sizes for elements drawn inside the axes. Axis labels and ticks come from the
#: rcParams above; these cover the parts matplotlib does not style globally, so
#: that legends and annotations match across every figure in the manuscript
#: rather than drifting per script.
LEGEND_FS = 10
ANNOT_FS = 9
PANEL_LABEL_FS = 11
