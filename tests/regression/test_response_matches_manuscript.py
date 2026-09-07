"""The response letter must quote the same numbers as the manuscript.

The letter is submitted alongside the paper and cites the same quantities, so a
reviewer reads them side by side. It had fallen a long way behind: it still
described the asymmetric fitting domain that protocol B replaced, quoted the
best-segment metric that was withdrawn, reported the framework as behind the
null when it is nominally ahead, and carried superseded class fractions,
anisotropy ratios, failed fractions, elongations and symmetry indices.

The mirror-symmetry numbers were wrong in the manuscript as well, which the
earlier section-by-section audit missed: the median departure was given as 0.50
against 0.18, and the four specimens below the visibility threshold were called
"precisely the fabric end members" when one of them is the schist at 45 degrees
and the schist at 0 degrees is above it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
RSP = REPO / "manuscript" / "response_to_reviewers.tex"
TAB = REPO / "outputs" / "tables"
G, P = "Augen gneiss", "Psammitic schist"

pytestmark = pytest.mark.skipif(not (TEX.is_file() and RSP.is_file()),
                                reason="manuscript or response absent")


@pytest.fixture(scope="module")
def rsp():
    return RSP.read_text(encoding="utf-8")


def test_the_symmetry_statistics(rsp):
    m = pd.read_csv(TAB / "displacement_mirror_symmetry.csv")
    assert m.max_index.median() == pytest.approx(0.18, abs=0.005)
    assert int((m.max_index > 0.10).sum()) == 10
    below = m[m.max_index < 0.10]
    got = sorted((r.lithology, int(r.angle_deg)) for _, r in below.iterrows())
    assert got == [(G, 0), (G, 90), (P, 45), (P, 90)], (
        f"the specimens below the visibility threshold are now {got}; both the "
        "manuscript and the response name them explicitly")
    assert m.max_index.max() == pytest.approx(0.577, abs=0.002)
    tex = TEX.read_text(encoding="utf-8")
    for doc, name in ((tex, "manuscript"), (rsp, "response")):
        assert "$0.50$" not in doc or "median departure" not in doc, name
        assert "0.036" in doc, f"{name} no longer reports the schist 45 deg outlier"
    assert "$0.83$" not in tex, "the superseded symmetry pair is back"


def test_the_response_class_fractions(rsp):
    f = pd.read_csv(TAB / "fourclass_area_fractions.csv").set_index(["rock", "angle_deg"])
    assert f.loc[(G, 90.0)].WT == pytest.approx(0.144, abs=0.002)
    assert f.loc[(P, 90.0)].WT == pytest.approx(0.151, abs=0.002)
    for v in ("$0.144$", "$0.151$", "$0.215$", "$0.208$", "$0.319$", "$0.372$"):
        assert v in rsp, f"the response no longer quotes {v}"
    for stale in ("$0.163$", "$0.186$", "$0.219$", "$0.185$", "$0.345$", "$0.362$"):
        assert stale not in rsp, f"a superseded class fraction is back: {stale}"


def test_the_response_trace_statistics(rsp):
    t = pd.read_csv(TAB / "trace_comparison_metrics.csv")
    e = t.abs_axial_angular_error_deg
    o = t.observed_orientation_deg
    assert e.mean() == pytest.approx(3.35, abs=0.02)
    assert (o.min(), o.max()) == pytest.approx((85.1, 97.1), abs=0.05)
    # the letter must not still say the framework is behind the null
    assert "framework is marginally behind it" not in rsp
    assert "nominally ahead of the null" in rsp
    for stale in ("$3.42^\\circ$", "$9.55^\\circ$", "$85.7^\\circ$", "$99.5^\\circ$",
                  "$+0.04^\\circ$", "$p = 0.91$"):
        assert stale not in rsp, f"a superseded trace statistic is back: {stale}"
    assert "$-0.30^\\circ$" in rsp and "$p = 0.54$" in rsp


def test_the_response_does_not_claim_the_withdrawn_metric(rsp):
    assert "gives $2.6^\\circ$ for both" not in rsp, (
        "the response again presents the best-segment metric as reported; the "
        "manuscript carries one error metric only")
    assert "reports a single error metric" in rsp


def test_the_response_anisotropy_and_failed_fractions(rsp):
    a = pd.read_csv(TAB / "strength_anisotropy_ratios.csv").set_index("rock")
    assert a.loc[G, "compressive_ratio"] == pytest.approx(1.60, abs=0.005)
    assert a.loc[P, "tensile_ratio"] == pytest.approx(2.57, abs=0.005)
    for stale in ("compressive $1.75$", "tensile $2.76$", "compressive $1.69$"):
        assert stale not in rsp, f"a superseded ratio is back: {stale}"
    s = pd.read_csv(TAB / "failure_statistics.csv")
    g = s[s.rock == G].set_index("angle_deg").p_fail
    c = s[s.rock == P].set_index("angle_deg").p_fail
    assert int(g.idxmax()) == 30 and int(c.idxmax()) == 45
    assert "peaks at $30^\\circ$ in the gneiss" in rsp
    assert "$45^\\circ$ in the\nschist" in rsp


def test_the_response_elongations(rsp):
    e = pd.read_csv(TAB / "energy_localization.csv")
    s = e[e.rock == P].set_index("angle_deg")
    assert sorted(int(a) for a in s.index if s.loc[a].corridor) == [0, 15, 90]
    assert "$1.17$ to $1.55$" in rsp
    for stale in ("$1.06$ and $1.43$", "$4.85$", "$1.20$ and $1.30$"):
        assert stale not in rsp, f"a superseded elongation is back: {stale}"


def test_the_response_states_the_same_mechanism_routes(rsp):
    """Both documents must describe the gneiss as skipping the sliding stage."""
    tex = TEX.read_text(encoding="utf-8")
    assert "sliding never leads" in rsp
    assert "different routes" in tex
    assert "through sliding on the fabric to weak-plane opening" not in rsp
