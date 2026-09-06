"""The mixed fraction must mean the same thing in Section 4.8 and Section 5.2.

Two classifiers compute a mixed fraction and the manuscript quoted both under
names that read alike. ``failure_statistics.csv`` carries ``mixed_cond`` from
the superseded three-mode scheme; the four-class classifier carries
``mixed_flag``. For the augen gneiss at 0 degrees they give 75.6% and 52.9%.

Section 4.8 quoted the four-class values and Section 5.2 the three-mode ones,
while Section 4.6 states that "the four-class scheme is the classification
reported in every failure-mode map and failure statistic". The two
subsections disagreed about one quantity and one of them contradicted that
declaration.

The three-mode columns are still written, because the failed fraction in
Fig. C.26b is read from the same file, so this cannot be fixed by deleting
them. It has to be held by a test.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import export
from tools import strain_partitioning as sp

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
STATS = REPO / "outputs" / "tables" / "failure_statistics.csv"

pytestmark = pytest.mark.skipif(not STATS.is_file(), reason="statistics absent")


@pytest.fixture(scope="module")
def fourclass():
    st = sp.specimen_strengths()
    out = {}
    for sid in range(1, 15):
        p = export.classification_panel(sid, st)
        m = np.asarray(p["mixed_flag"])[p["core_mask"]]
        c = np.asarray(p["mode_code"])[p["core_mask"]]
        f = c >= 0
        out.setdefault(p["lithology"], {})[int(p["angle_deg"])] = (
            100.0 * m[f].mean() if f.any() else 0.0)
    return out


def test_the_two_classifiers_really_do_differ(fourclass):
    """Guard the guard: if they ever agree, this file proves nothing."""
    fs = pd.read_csv(STATS)
    worst = 0.0
    for _, r in fs.iterrows():
        worst = max(worst, abs(r.mixed_cond - fourclass[r.rock][int(r.angle_deg)]))
    assert worst > 10.0, (
        "the three-mode and four-class mixed fractions now agree to within "
        f"{worst:.1f} points, so quoting either is harmless and this test is "
        "no longer needed")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_manuscript_quotes_only_the_four_class_values(fourclass):
    t = TEX.read_text(encoding="utf-8")
    # Under the measured ratios each rock has an interior peak at a different
    # angle, and Sections 4.8, 5.2 and the Conclusion all quote that pair. The
    # peaks are what the manuscript states, so the peaks are what is matched.
    g = fourclass["Augen gneiss"]
    sch = fourclass["Psammitic schist"]
    g_peak = max(g.values())
    s_peak = max(sch.values())
    assert max(g, key=g.get) == 45, "the gneiss mixed peak has moved off 45 deg"
    assert max(sch, key=sch.get) == 60, "the schist mixed peak has moved off 60 deg"
    assert f"${g_peak:.1f}$\\%" in t, f"the gneiss peak {g_peak:.1f}% is not stated"
    assert f"${s_peak:.1f}$\\%" in t, f"the schist peak {s_peak:.1f}% is not stated"
    # Match the values, not one phrasing of them. The first version of this
    # test looked for "$76$\\% and $68$\\%" and missed the Conclusion, which
    # wrote "reaches $76$\\% in the gneiss and $68$\\% in the schist".
    for stale in ("$76$\\%", "$68$\\%", "$28$\\% in the gneiss",
                  "$4.4$\\% in the gneiss", "$35$\\% in the schist"):
        assert stale not in t, (
            f"a three-mode or superseded mixed figure is back: {stale}. The "
            "four-class classifier is the one Section 4.6 declares reported.")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_mixed_fraction_is_never_claimed_to_vanish_at_forty_five(fourclass):
    """Only the three-mode scheme gives zero there; four-class peaks there."""
    # The gneiss peak sits at 45 deg, so the mixed fraction is at its largest
    # exactly where the superseded scheme called it zero. Above the peak the
    # decline is uneven, which Section 5.2 now states rather than claiming a
    # uniform drop below 3%.
    g = fourclass["Augen gneiss"]
    sch = fourclass["Psammitic schist"]
    assert g[45] > 5.0, f"gneiss four-class mixed fraction at 45 deg is {g[45]:.1f}%"
    assert max(g[60], g[75], g[90]) == pytest.approx(5.7, abs=0.1), (
        "the gneiss mixed fraction above the peak has moved; Section 5.2 "
        "quotes at most 5.7%")
    assert sch[75] == 0.0 and sch[90] == 0.0, (
        "the schist mixed fraction no longer vanishes from 75 deg upward")
    assert g[90] == 0.0 and sch[90] == 0.0, (
        "Sections 4.8 and 5.2 state the mixed fraction vanishes at 90 deg")
    t = TEX.read_text(encoding="utf-8")
    assert "falls to zero at $45^\\circ$" not in t, (
        "the claim that the mixed fraction vanishes at 45 degrees is back; it "
        "holds only under the superseded three-mode classifier")
