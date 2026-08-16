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
    """The literal arrays repeated the tensile count as the mixed count."""
    s = stats[stats.failed > 0]
    assert not (s.mixed == s.tensile).any()
    assert not (s.mixed == s.shear).any()


@needs_fields
def test_failed_fraction_minimum_is_at_seventy_five_degrees(stats):
    for rock, g in stats.groupby("rock"):
        g = g.sort_values("angle_deg")
        assert g.loc[g.p_fail.idxmin(), "angle_deg"] == 75.0, rock
        assert g.p_fail.iloc[-1] > g.p_fail.min(), f"{rock}: no recovery at 90 deg"


@needs_fields
def test_mixed_fraction_never_reaches_forty_percent(stats):
    """The text once described specimens above 40%; none exist."""
    assert stats.mixed_cond.max() < 40.0


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
