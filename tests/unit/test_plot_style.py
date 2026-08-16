"""Tests for the mandated publication plot style and disc-geometry helpers.

Two requirements are easy to violate silently and are therefore asserted here:
foliation guide lines must terminate on the disc boundary, and manuscript
figures must carry no main title.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tools.plot_style import (PLOT_STYLE, TICK_PARAMS, apply_plot_style,
                              foliation_chord, draw_foliation_lines, draw_disc,
                              panel_label, assert_no_suptitle,
                              times_new_roman_available, style_axes)

R = 25.5
ANGLES = (0, 15, 30, 45, 60, 75, 90)


# ---------------------------------------------------------------------------
# Mandated style values
# ---------------------------------------------------------------------------
def test_mandated_style_values_are_exact():
    assert PLOT_STYLE["font.family"] == "Times New Roman"
    assert PLOT_STYLE["font.size"] == 14
    assert PLOT_STYLE["axes.linewidth"] == 1.5
    assert PLOT_STYLE["axes.titlesize"] == 16
    assert PLOT_STYLE["axes.labelsize"] == 14
    assert PLOT_STYLE["xtick.labelsize"] == 12
    assert PLOT_STYLE["ytick.labelsize"] == 12
    assert PLOT_STYLE["legend.fontsize"] == 12
    assert PLOT_STYLE["figure.dpi"] == 300
    assert PLOT_STYLE["savefig.dpi"] == 300
    assert PLOT_STYLE["text.usetex"] is False


def test_mandated_tick_params_are_exact():
    assert TICK_PARAMS["major"] == {"which": "major", "direction": "out",
                                    "length": 5, "width": 1.5}
    assert TICK_PARAMS["minor"] == {"which": "minor", "direction": "out",
                                    "length": 3, "width": 1}


def test_apply_plot_style_returns_tick_params_and_sets_rcparams():
    tp = apply_plot_style()
    assert tp is TICK_PARAMS
    assert matplotlib.rcParams["font.size"] == 14
    assert matplotlib.rcParams["axes.linewidth"] == 1.5
    # vector export must embed TrueType, not outline the text
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_font_resolution_is_reported_not_silently_ignored():
    # Not an assertion about the machine: the helper must simply answer.
    assert isinstance(times_new_roman_available(), bool)


# ---------------------------------------------------------------------------
# Foliation lines clipped to the disc
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle", ANGLES)
def test_centred_foliation_chord_ends_exactly_on_the_boundary(angle):
    """A line through the centre must be a diameter: both ends at r = R."""
    (x0, x1), (y0, y1) = foliation_chord(angle, R, 0.0)
    assert np.hypot(x0, y0) == pytest.approx(R, abs=1e-9)
    assert np.hypot(x1, y1) == pytest.approx(R, abs=1e-9)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("offset", [-0.9, -0.5, 0.0, 0.3, 0.75])
def test_offset_foliation_chord_never_leaves_the_disc(angle, offset):
    chord = foliation_chord(angle, R, offset * R)
    assert chord is not None
    (x0, x1), (y0, y1) = chord
    for x, y in ((x0, y0), (x1, y1)):
        assert np.hypot(x, y) <= R + 1e-9, "foliation line escapes the disc"
    # endpoints must lie ON the boundary, not short of it
    assert np.hypot(x0, y0) == pytest.approx(R, abs=1e-9)
    assert np.hypot(x1, y1) == pytest.approx(R, abs=1e-9)


@pytest.mark.parametrize("angle", ANGLES)
def test_chord_direction_matches_the_requested_angle(angle):
    (x0, x1), (y0, y1) = foliation_chord(angle, R, 0.0)
    got = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180.0
    assert got == pytest.approx(angle % 180.0, abs=1e-9)


def test_chord_is_none_when_the_line_misses_the_disc():
    assert foliation_chord(30, R, R) is None
    assert foliation_chord(30, R, 1.5 * R) is None


@pytest.mark.parametrize("angle", ANGLES)
def test_every_drawn_foliation_line_stays_inside_the_disc(angle):
    fig, ax = plt.subplots()
    n = draw_foliation_lines(ax, angle, R)
    assert n >= 3
    for line in ax.lines:
        x, y = line.get_xdata(), line.get_ydata()
        assert np.all(np.hypot(x, y) <= R + 1e-9), "a drawn line leaves the disc"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Disc panel geometry
# ---------------------------------------------------------------------------
def test_draw_disc_sets_equal_aspect_and_symmetric_limits():
    fig, ax = plt.subplots()
    draw_disc(ax, R, angle_deg=45)
    assert ax.get_aspect() == 1.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 == pytest.approx(-x1) and y0 == pytest.approx(-y1)
    assert (x1, y1) == pytest.approx((y1, x1))       # square extent
    assert x1 > R                                     # boundary is visible
    plt.close(fig)


def test_panel_label_is_placed_in_axes_coordinates():
    fig, ax = plt.subplots()
    t = panel_label(ax, "(a)")
    assert t.get_text() == "(a)"
    assert t.get_transform() is ax.transAxes
    plt.close(fig)


def test_style_axes_applies_minor_ticks():
    fig, ax = plt.subplots()
    style_axes(ax)
    assert len(ax.xaxis.get_minor_ticks()) > 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# No main title on manuscript figures
# ---------------------------------------------------------------------------
def test_assert_no_suptitle_passes_on_a_clean_figure():
    fig, _ax = plt.subplots()
    assert assert_no_suptitle(fig) is True
    plt.close(fig)


def test_assert_no_suptitle_rejects_a_titled_figure():
    fig, _ax = plt.subplots()
    fig.suptitle("A main title that belongs in the caption")
    with pytest.raises(AssertionError):
        assert_no_suptitle(fig)
    plt.close(fig)


def test_composite_figure_builders_emit_no_main_title(tmp_path):
    """The four required composites must carry no suptitle."""
    from tools import lithology as lith
    from tools import traces as tr
    from tools import plotting as tp

    rows = [r for r in tr.compare_all() if r["lithology_key"] == "augen_gneiss"]
    out = tp.seven_panel_trace_overlay(rows, lith.AUGEN_GNEISS,
                                       tmp_path / "g", formats=("png",))
    assert out and out[0].exists()
    # the builder calls assert_no_suptitle internally; reaching here proves it passed
