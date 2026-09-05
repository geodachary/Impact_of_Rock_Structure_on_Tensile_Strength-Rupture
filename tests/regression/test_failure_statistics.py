"""Failure statistics must be counted from the fields, not carried as literals.

Both panels of the failure-statistics figure were drawn from mode counts written
into a notebook as literal lists. Those lists could not have come from the
classifier: the disc grid is fixed, so the number of points cannot change with
orientation, yet their per-angle totals varied from 4580 to 1876, and the mixed
count coincided exactly with the tensile count at three of seven angles.

The constant-total check below is the one that catches this class of error
immediately, and it is why it runs first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import fabric_tractions as ft

REPO = Path(__file__).resolve().parents[2]
FIELDS = ft.field_files()
needs_fields = pytest.mark.skipif(not FIELDS, reason="fields not exported yet")


@pytest.fixture(scope="module")
def stats():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mfs", REPO / "scripts/make_failure_statistics_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.counts()


@needs_fields
def test_point_totals_are_identical_across_specimens(stats):
    """A varying total is impossible on a fixed grid and marks fabricated counts."""
    assert stats.total.nunique() == 1, sorted(stats.total.unique())


@needs_fields
def test_modes_sum_to_the_total(stats):
    s = stats
    assert ((s.tensile + s.shear + s.mixed + s.no_failure) == s.total).all()
    assert ((s.tensile + s.shear + s.mixed) == s.failed).all()


@needs_fields
def test_mixed_is_not_a_duplicate_of_another_mode(stats):
    """The literal arrays repeated the tensile count as the mixed count.

    Restricted to specimens that actually record mixed points. With the stress
    sign convention declared, tensile and mixed are both zero below
    75 degrees, where the foliation is clamped and the matrix fails in shear;
    comparing two zeros says nothing about duplication and would fail this
    test for the wrong reason.
    """
    s = stats[(stats.failed > 0) & (stats.mixed > 0)]
    assert not s.empty, "no specimen records any mixed points"
    assert not (s.mixed == s.tensile).any()
    assert not (s.mixed == s.shear).any()


@needs_fields
def test_failed_fraction_minimum_matches_the_final_data(stats):
    """Both lithologies now dip at 75 degrees and recover slightly at 90.

    This assertion has moved twice. It first expected a minimum at 75 degrees
    in both rocks; the 2026-08 strength revision moved the gneiss minimum to
    90, and the corrected cohesion has moved it back to 75. What survives all
    three states, and is what Section 4.6 argues from, is that the minimum
    coincides with the measured strength minimum at 75 degrees and that there
    is a partial recovery at 90. That is asserted rather than the bare angle,
    so the test tracks the claim instead of the last run.
    """
    for rock, g in stats.groupby("rock"):
        g = g.sort_values("angle_deg")
        assert g.loc[g.p_fail.idxmin(), "angle_deg"] == 75.0, (
            f"{rock}: failed fraction now bottoms at "
            f"{g.loc[g.p_fail.idxmin(), 'angle_deg']:.0f} deg, not at the "
            "75 deg strength minimum that Section 4.6 ties it to")
        assert g.p_fail.iloc[-1] > g.p_fail.min(), (
            f"{rock}: no recovery at 90 deg; the reactivation argument in "
            "Section 4.6 depends on one")


@needs_fields
def test_mixed_competition_is_confined_to_low_fabric_angles(stats):
    """Where the governing and runner-up utilities are close, and where not.

    This once asserted that no specimen exceeded 40% mixed, which was true of
    the earlier strength calibration. With the corrected cohesion the two
    matrix utilities run much closer at low fabric angle and the conditional
    mixed fraction reaches 76%. That is a real change in the competition, not
    a leak: what has to stay true for Section 4.6 is that it is confined to
    the low-angle end, where the fabric is clamped and neither matrix
    criterion is clearly ahead.
    """
    lo = stats[stats.angle_deg <= 30.0]
    hi = stats[stats.angle_deg >= 45.0]
    assert lo.mixed_cond.max() > hi.mixed_cond.max(), (
        "mixed competition is no longer strongest at low fabric angle")
    assert hi.mixed_cond.max() < 40.0, (
        f"mixed competition reaches {hi.mixed_cond.max():.1f}% at or above "
        "45 deg, where the text describes one utility as clearly governing")


def test_notebook_literals_are_no_longer_the_source():
    nb = REPO / "Tensile_general_plots.ipynb"
    if not nb.exists():
        pytest.skip("general-plots notebook absent")
    d = json.loads(nb.read_text(encoding="utf-8"))
    for c in d["cells"]:
        src = "".join(c["source"])
        if "pfailure_vs_theta" in src:
            assert "mixed_ag   = [1352" not in src, (
                "the hard-coded mode counts are still driving the figure; "
                "run scripts/make_failure_statistics_figures.py instead")
