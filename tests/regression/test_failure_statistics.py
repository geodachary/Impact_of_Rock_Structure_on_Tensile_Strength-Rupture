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
def test_three_mode_mixed_competition_peaks_where_the_branches_meet(stats):
    """Where the governing and runner-up utilities are close, per lithology.

    These are the superseded three-mode ``mixed_cond`` values. The manuscript
    does not quote them: Section 4.6 declares the four-class classifier the one
    reported, and test_mixed_fraction_has_one_definition holds the text to it.
    They are still checked because the failed fraction in Fig. C.26b is read
    from this same file, so the file has to stay sound.

    An earlier version asserted the competition was confined to low fabric
    angle in both rocks. That is true of the gneiss, whose peak is at 0 deg
    with the fabric clamped and neither matrix criterion ahead, but not of the
    schist, which peaks at 60 deg where the weak-plane and matrix branches
    cross. The two lithologies are therefore checked separately.
    """
    g = stats[stats.rock == "Augen gneiss"]
    sch = stats[stats.rock == "Psammitic schist"]
    assert int(g.loc[g.mixed_cond.idxmax(), "angle_deg"]) == 0, (
        "the gneiss three-mode mixed competition no longer peaks with the "
        "fabric clamped at 0 deg")
    assert g[g.angle_deg <= 30].mixed_cond.max() > \
        g[g.angle_deg >= 45].mixed_cond.max(), (
        "gneiss mixed competition is no longer strongest at low fabric angle")
    assert int(sch.loc[sch.mixed_cond.idxmax(), "angle_deg"]) == 60, (
        "the schist three-mode mixed competition no longer peaks at 60 deg, "
        "where the weak-plane and matrix branches cross")
    assert sch[sch.angle_deg <= 15].mixed_cond.max() == 0.0, (
        "the schist now shows mixed competition at 0 and 15 deg, where matrix "
        "shear governs without a rival")


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
