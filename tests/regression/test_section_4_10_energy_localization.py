"""Section 4.10's localization measures, colour scale and G/Gc statistics.

Four numbers were wrong, and one of them disagreed with Section 4.8 about the
same quantity.

* The elongation ranges were given as 1.13 to 1.63 in the gneiss and 1.51 to
  4.86 in the schist. The table gives 1.095 to 1.606 and 1.143 to 4.861, so
  the schist's lower bound was out by a third. The same paragraph then said
  the gneiss elongations lie between 1.1 and 1.6, contradicting its own
  1.13-to-1.63 a few lines earlier.
* The shared colour scale was quoted as 0.062 MPa for the schist against 0.046
  for the gneiss; ``energy_colour_scale.csv`` gives 0.078 and 0.050.
* The G/Gc band coefficients of variation were given here as 3.12, 3.55 and
  3.01 and in Section 4.8 as 3.0, 4.4 and 3.1, for one quantity with one
  generated value: 3.63, 3.43, 2.34. Two subsections disagreed with each other
  and both disagreed with the table.
* The smallest energy-selected G/Gc is 1.017, quoted as 1.01.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"

pytestmark = pytest.mark.skipif(not (TAB / "energy_localization.csv").is_file(),
                                reason="localization table absent")


def test_the_elongation_ranges():
    d = pd.read_csv(TAB / "energy_localization.csv")
    want = {"Augen gneiss": (1.17, 1.55), "Psammitic schist": (1.17, 4.79)}
    for rock, (lo, hi) in want.items():
        e = d[d.rock == rock].elongation
        assert e.min() == pytest.approx(lo, abs=0.01), f"{rock} min {e.min():.3f}"
        assert e.max() == pytest.approx(hi, abs=0.01), f"{rock} max {e.max():.3f}"


def test_the_corridor_forms_only_in_the_schist_at_the_fabric_end_members():
    """The corridor is a schist feature and it sits at the fabric end members.

    Under the per-specimen elastic constants it appeared at 60-90 degrees;
    with the material-axis constants it forms where the foliation is either
    normal to the loading axis or parallel to it, at 0, 15 and 90 degrees,
    and the region is two lobes at every intermediate angle. The gneiss never
    forms one.
    """
    d = pd.read_csv(TAB / "energy_localization.csv")
    corr = d[d.corridor]
    assert set(corr.rock) == {"Psammitic schist"}, (
        "a corridor now forms in the gneiss; the contrast the text draws is gone")
    assert sorted(corr.angle_deg) == [0.0, 15.0, 90.0]
    assert corr.elongation.min() == pytest.approx(4.6, abs=0.05)
    assert corr.elongation.max() == pytest.approx(4.8, abs=0.05)
    mid = d[(d.rock == "Psammitic schist") & (~d.corridor)]
    assert (mid.n_components == 2).all(), (
        "the schist no longer breaks into two lobes at intermediate angles")
    g = d[d.rock == "Augen gneiss"]
    assert (g.n_components == 2).all(), "the gneiss no longer stays two lobes"


def test_the_shared_colour_scale():
    d = pd.read_csv(TAB / "energy_colour_scale.csv").set_index("rock")
    assert d.loc["Augen gneiss", "u_95th_percentile_MPa"] == pytest.approx(0.035, abs=0.001)
    assert d.loc["Psammitic schist", "u_95th_percentile_MPa"] == pytest.approx(0.036, abs=0.001)


def test_every_energy_selected_step_clears_gc():
    d = pd.read_csv(TAB / "fig20_GGc_profile_metrics.csv")
    assert int(d.n_below_Gc_energy.sum()) == 0, (
        "an energy-selected step now falls below Gc")
    assert d.min_G_over_Gc_energy.min() == pytest.approx(1.00, abs=0.005)
    assert d.phys_fraction.mean() == pytest.approx(0.77, abs=0.005)


def test_the_driving_force_spans_about_eight_decades():
    d = pd.read_csv(TAB / "fig20_GGc_profile_metrics.csv")
    span = np.log10(d.max_G_over_Gc / d.min_G_over_Gc)
    assert np.median(span) == pytest.approx(8.0, abs=0.5), (
        f"median dynamic range is {np.median(span):.1f} decades")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_two_subsections_agree_about_the_band_coefficients():
    """One quantity, one value, wherever the manuscript states it."""
    t = TEX.read_text(encoding="utf-8")
    d = pd.read_csv(TAB / "fig20_GGc_band_summary.csv").set_index("band")
    quoted = [f"${v:.1f}$" for v in (d.loc["low (0-30 deg)", "mean_CV"],
                                     d.loc["intermediate (45-60 deg)", "mean_CV"],
                                     d.loc["high (75-90 deg)", "mean_CV"])]
    assert quoted == ["$3.7$", "$4.2$", "$2.4$"], quoted
    # both places must carry the same triple and nothing else
    for stale in ("$3.12$", "$3.55$", "$3.01$, $3.1$", "$4.4$ and $3.1$",
                  "$3.7$, $3.4$ and $2.3$",
                  "$3.6$, $3.4$ and $2.3$", "$3.2$, $4.3$ and $2.6$"):
        assert stale not in t, f"a superseded band figure is back: {stale}"
    assert t.count("$3.7$, $4.2$ and $2.4$") == 2, (
        "the two subsections no longer state the same band coefficients")
