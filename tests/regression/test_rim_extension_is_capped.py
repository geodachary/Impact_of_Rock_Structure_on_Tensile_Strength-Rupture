"""A predicted trace may not be joined to the rim by a segment nothing computed.

The stepper is trimmed back to the furthest radius the tip actually reached,
because the energy criterion can steer the path inward and orbit there. On five
of the fourteen specimens that furthest radius is well inside the disc, as low
as 0.77R. The endpoint was then extended to the boundary unconditionally, which
drew a straight terminal segment up to 31 times the step length and swung the
end of the path as much as 8.7 mm off axis. Two of those, the augen gneiss at
15 and 30 degrees, are visible as a hook at the bottom of the overlay panels.

Nothing computed that segment. It is the ray from the last real point to the
circle, and when the trimmed path ended on a repeated point the direction was
degenerate and the fallback projected radially instead, which is why the hook
points somewhere different in each panel.

The published orientations never saw it: they are fitted inside
``PREDICTED_CORE_FRAC`` (0.85R) and every fabricated segment lies outside that.
Capping the extension therefore changes no reported number, and these tests pin
both halves of that statement, the cap and the invariance, so that a future
change cannot quietly reintroduce the geometry or start letting it into a fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import lithology as lith
from tools import traces as tr
from tools.analysis import crack_energy_suite as ces
from tools.analysis import crack_path_suite as cps

R = tr.RADIUS_M
CAP = ces.MAX_RIM_EXTENSION_FRAC * R


def _predicted_primary(sample_id):
    d = tr.load_predicted_trace(lith.predicted_trace_path(sample_id))
    px, py, *_ = tr.primary_segment(d["x"], d["y"])
    return np.asarray(px, float), np.asarray(py, float)


def _predicted_raw(sample_id):
    """The whole stored polyline, which is what several figures plot.

    ``primary_segment`` splits on long jumps and keeps the longest piece, so a
    fabricated terminal segment is silently discarded by any check that goes
    through it. The overlay panels take that route and looked clean; the
    strain-energy panels plot the raw array and showed the psammitic schist at
    0 degrees with two 36 mm spikes running out to the rim near the equator.
    Anything asserting the geometry has to read the raw array.
    """
    d = tr.load_predicted_trace(lith.predicted_trace_path(sample_id))
    return np.asarray(d["x"], float), np.asarray(d["y"], float)


def test_the_two_copies_of_the_helper_agree():
    """Both modules carry their own extension helper; a cap on one is not a fix.

    ``crack_energy_suite`` writes the traces the figures use and
    ``crack_path_suite`` writes the smoke paths. They were separately authored
    and are separately maintained, so the constant is asserted equal here
    rather than trusted to stay so.
    """
    assert cps.MAX_RIM_EXTENSION_FRAC == ces.MAX_RIM_EXTENSION_FRAC


@pytest.mark.parametrize("mod", [ces, cps])
def test_a_stalled_path_is_left_where_it_stalled(mod):
    """The case that produced the hook: last point far inside, cap in force."""
    p0, p1 = (0.0, 0.019), (0.0, 0.0195)          # heading out, still at 0.76R
    out = mod.extend_to_circle(p0, p1, R, max_extend=CAP)
    assert out == pytest.approx(p1), (
        "an endpoint 6 mm short of the rim was extended to it; the cap is not "
        "being applied")


@pytest.mark.parametrize("mod", [ces, cps])
def test_a_degenerate_direction_does_not_become_a_radial_jump(mod):
    """The repeated-endpoint case, which took the radial fallback."""
    p = (0.004, 0.0205)
    out = mod.extend_to_circle(p, p, R, max_extend=CAP)
    assert out == pytest.approx(p)


@pytest.mark.parametrize("mod", [ces, cps])
def test_a_path_that_did_reach_the_rim_is_still_closed(mod):
    """The cap must not stop the extension it exists to allow.

    The stepper halts at ``r_end_frac`` = 0.995R, so the genuine gap is a
    fraction of one step and has to keep being closed.
    """
    p0, p1 = (0.0, 0.0250), (0.0, 0.02536)
    x, y = mod.extend_to_circle(p0, p1, R, max_extend=CAP)
    assert np.hypot(x, y) == pytest.approx(R, rel=1e-9)


@pytest.mark.parametrize("mod", [ces, cps])
def test_omitting_the_cap_keeps_the_old_behaviour(mod):
    """``max_extend=None`` is still a plain ray-circle intersection."""
    p0, p1 = (0.0, 0.019), (0.0, 0.0195)
    x, y = mod.extend_to_circle(p0, p1, R)
    assert np.hypot(x, y) == pytest.approx(R, rel=1e-9)


def test_no_published_trace_ends_in_a_fabricated_segment():
    offenders = {}
    for sid in range(1, 15):
        x, y = _predicted_primary(sid)
        seg = np.hypot(np.diff(x), np.diff(y))
        if seg.size and max(seg[0], seg[-1]) > CAP:
            offenders[sid] = float(max(seg[0], seg[-1]) / np.median(seg))
    assert not offenders, (
        f"terminal segments exceed the cap on {offenders} (as multiples of the "
        "median step). Either the traces predate the cap and need regenerating, "
        "or the extension is being called uncapped again.")


def test_no_raw_trace_ends_in_a_fabricated_segment():
    """The check that would have caught the psammitic schist at 0 degrees.

    Read the raw array, not the primary segment: ``primary_segment`` splits on
    long jumps and keeps the longest piece, so it discarded both 36 mm spikes
    and the overlay panels looked clean while the strain-energy panels drew
    them. The cap governs the *terminal* extension, so that is what is
    asserted here.
    """
    offenders = {}
    for sid in range(1, 15):
        x, y = _predicted_raw(sid)
        seg = np.hypot(np.diff(x), np.diff(y))
        if not seg.size:
            continue
        worst = float(max(seg[0], seg[-1]))
        if worst > CAP:
            offenders[sid] = round(worst * 1e3, 2)
    assert not offenders, (
        f"the stored polyline begins or ends with a jump of {offenders} (mm), "
        f"longer than the {CAP * 1e3:.2f} mm cap. No step the model takes is "
        "that long, so these are extensions to the rim that were never "
        "computed; the strain-energy and path panels draw them.")


def test_no_interior_jump_is_a_discarded_spike():
    """An oversized step in the middle is a different fault, so measure it so.

    The arc-median collapse re-bins the path along its principal axis, and a
    path that reverses often puts several points in one bin and leaves the
    next empty, which spreads the output points unevenly. On the psammitic
    schist at 15 degrees, which reverses in y twenty times against four and
    zero on its neighbours, that legitimately produces steps of 0.7 mm against
    a 0.105 mm median. That is binning, not invented geometry.

    A discarded spike is a different size entirely: the ones this module was
    written for were 36 mm, over a hundred times the median. Judge against the
    specimen's own step length rather than against the rim cap, which governs
    the ends only.
    """
    offenders = {}
    for sid in range(1, 15):
        x, y = _predicted_raw(sid)
        seg = np.hypot(np.diff(x), np.diff(y))
        if seg.size < 3:
            continue
        interior = seg[1:-1]
        ratio = float(interior.max() / np.median(seg))
        if ratio > 15.0:
            offenders[sid] = round(ratio, 1)
    assert not offenders, (
        f"interior steps reach {offenders} times the median. Uneven collapse "
        "binning does not do that; a jump this size is a segment the stepper "
        "never took.")


def test_capping_the_extension_changes_no_reported_orientation():
    """The invariance that makes this a figure fix and not a results change."""
    for sid in range(1, 15):
        x, y = _predicted_primary(sid)
        seg = np.hypot(np.diff(x), np.diff(y))
        keep = np.ones(x.size, bool)
        if seg.size and seg[-1] > CAP:
            keep[-1] = False
        if seg.size and seg[0] > CAP:
            keep[0] = False

        def fit(xx, yy):
            m = tr.core_mask_radial(xx, yy, R)
            if m.sum() >= 3:
                return tr.fit_orientation_deg(xx[m], yy[m])[0]
            return tr.fit_orientation_deg(xx, yy)[0]

        assert fit(x, y) == pytest.approx(fit(x[keep], y[keep]), abs=1e-6), (
            f"specimen {sid}: dropping the terminal segment moved the fitted "
            "orientation, so it is reaching inside PREDICTED_CORE_FRAC and the "
            "cap is no longer a figure-only change")
