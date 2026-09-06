"""Figure C.26 quotes class fractions and failed fractions; nothing pinned them.

The four-class panel and the failed-fraction panel both carry numbers in their
caption, and the surrounding body text repeats two of them. No guard read that
caption, so when the anisotropy ratios moved from the adopted constants to the
measured end-member ratio the caption kept quoting the superseded values: matrix
shear peaking at 0.48 and 0.54, weak-plane opening at 0.13 and 0.10, a failed
fraction of 0.57 and 0.66 at 30 degrees, and matrix tension "confined to the two
lowest angles" when the class had become empty. The full test suite passed
throughout, because every other guard reads tables rather than captions.

This file closes that gap. It reads the caption and the body sentence and checks
them against the same CSVs the figure is drawn from.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"

pytestmark = pytest.mark.skipif(
    not (TEX.is_file() and (TAB / "fourclass_area_fractions.csv").is_file()),
    reason="manuscript or class-fraction table absent")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def four():
    return pd.read_csv(TAB / "fourclass_area_fractions.csv")


@pytest.fixture(scope="module")
def stats():
    return pd.read_csv(TAB / "failure_statistics.csv")


def _at(df, rock, angle, col):
    v = df[(df.rock == rock) & (df.angle_deg == angle)][col]
    assert len(v) == 1
    return float(v.iloc[0])


def test_the_caption_quotes_the_matrix_shear_peak(tex, four):
    m = re.search(r"peaking at \$([\d.]+)\$ in the\s*\n?\s*gneiss and \$([\d.]+)\$ in the schist", tex)
    assert m, "the matrix-shear peak sentence has changed shape"
    for said, rock in ((m.group(1), "Augen gneiss"), (m.group(2), "Psammitic schist")):
        g = four[four.rock == rock]
        assert float(said) == pytest.approx(g.MS.max(), abs=0.005), (
            f"{rock}: caption says matrix shear peaks at {said}, the table "
            f"gives {g.MS.max():.3f}")


def test_the_caption_quotes_the_weak_plane_opening_at_ninety(tex, four):
    m = re.search(r"opening grows toward \$90\^\\circ\$ \(\$([\d.]+)\$ and \$([\d.]+)\$\)", tex)
    assert m, "the weak-plane opening sentence has changed shape"
    for said, rock in ((m.group(1), "Augen gneiss"), (m.group(2), "Psammitic schist")):
        assert float(said) == pytest.approx(_at(four, rock, 90, "WT"), abs=0.005), (
            f"{rock}: caption says WT = {said} at 90 deg, the table gives "
            f"{_at(four, rock, 90, 'WT'):.3f}")


def test_the_caption_reports_matrix_tension_correctly(tex, four):
    """It held 0.120 of the gneiss under the adopted ratios.

    Under the measured ratios the class is empty in the gneiss but not in the
    schist, where it reaches 0.008 at 60 degrees. The caption says exactly
    that, so both halves are checked rather than the blanket "empty everywhere"
    this test used to assert.
    """
    assert "Matrix tension is empty in the gneiss at every" in tex, (
        "the caption no longer states that matrix tension is empty in the gneiss")
    g = four[four.rock == "Augen gneiss"]
    assert g.MT.max() == 0.0, (
        f"matrix tension has reappeared in the gneiss (max {g.MT.max():.4f}); "
        "the caption describes the class as empty there")
    sch = four[four.rock == "Psammitic schist"]
    assert sch.MT.max() == pytest.approx(0.008, abs=0.0005), (
        f"schist matrix tension is now {sch.MT.max():.4f}; the caption says 0.008")
    assert int(sch.loc[sch.MT.idxmax(), "angle_deg"]) == 60


def test_the_caption_quotes_the_failed_fraction_extremes(tex, stats):
    m = re.search(r"greatest at \$30\^\\circ\$ in\s*\n?\s*the gneiss \(\$([\d.]+)\$\) and at "
                  r"\$45\^\\circ\$ in the schist \(\$([\d.]+)\$\), falls to a\s*\n?\s*"
                  r"minimum at \$75\^\\circ\$ \(\$([\d.]+)\$ and \$([\d.]+)\$\)", tex)
    assert m, "the failed-fraction sentence has changed shape"
    want = [("Augen gneiss", 30), ("Psammitic schist", 45),
            ("Augen gneiss", 75), ("Psammitic schist", 75)]
    for said, (rock, ang) in zip(m.groups(), want):
        got = _at(stats, rock, ang, "p_fail") / 100.0
        assert float(said) == pytest.approx(got, abs=0.005), (
            f"{rock} at {ang} deg: caption says {said}, the table gives {got:.3f}")


@pytest.mark.parametrize("rock,peak_deg", [("Augen gneiss", 30),
                                           ("Psammitic schist", 45)])
def test_the_failed_fraction_has_an_interior_maximum(stats, rock, peak_deg):
    """The interior maximum Reviewer 2 asked about.

    What the response turns on is that the maximum is interior, not that it
    sits at one particular angle. Under the measured ratios the two lithologies
    peak at different angles, 30 degrees in the gneiss and 45 in the schist,
    which is what the caption now states; both are still interior and both
    still fall to their minimum at 75 degrees.
    """
    g = stats[stats.rock == rock]
    peak = int(g.loc[g.p_fail.idxmax(), "angle_deg"])
    assert peak == peak_deg, (
        f"{rock}: the failed-point fraction now peaks at {peak} deg, not "
        f"{peak_deg}; the caption and Section 4.6 must be updated with it.")
    assert peak not in (0, 90), "the maximum must be interior"
    assert int(g.loc[g.p_fail.idxmin(), "angle_deg"]) == 75


def test_the_body_and_caption_agree_about_the_seventy_five_degree_values(tex, stats):
    m = re.search(r"\$75\^\\circ\$ their failed fractions are close, \$([\d.]+)\$ in the gneiss against\s*\n?\s*\$([\d.]+)\$ in the schist", tex)
    assert m, "the body sentence comparing the two rocks at 75 deg has changed shape"
    for said, rock in ((m.group(1), "Augen gneiss"), (m.group(2), "Psammitic schist")):
        got = _at(stats, rock, 75, "p_fail") / 100.0
        assert float(said) == pytest.approx(got, abs=0.005), (
            f"{rock}: body says {said} at 75 deg, the table gives {got:.3f}")
