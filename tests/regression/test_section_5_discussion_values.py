"""Section 5's reported values and the claims that rest on them.

Section 5 is mostly interpretation, and its interpretation was sound; what had
drifted were three numeric groups it leans on:

* the trace-roughness comparison still quoted the pre-warp modelled wander for
  the gneiss at 90 degrees (0.05 mm against a current 0.25) and a paired
  p-value of 0.002 where the producer computes 0.0043;
* the Griffith substitution quoted load factors of 0.90, 0.86 and 1.14 for the
  gneiss at 0 degrees, where the table gives 1.005, 1.003 and 1.209.

The surviving claims are pinned here as well, because each is an argument the
section makes rather than a number it merely reports: that opening is gated by
the sign of the resolved traction, that the mechanism ordering survives the
whole proxy sweep, and that the three matrix criteria are commensurate.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
G, P = "Augen gneiss", "Psammitic schist"

pytestmark = pytest.mark.skipif(
    not (TEX.is_file() and (TAB / "trace_roughness.csv").is_file()),
    reason="manuscript or outputs absent")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def test_the_model_is_straighter_and_by_the_stated_amount(tex):
    r = pd.read_csv(TAB / "trace_roughness.csv")
    assert int(r.model_straighter.sum()) == 13, (
        "the model is no longer straighter in thirteen of fourteen specimens")
    g90 = r[(r.lithology == G) & (r.angle_deg == 90)].iloc[0]
    s75 = r[(r.lithology == P) & (r.angle_deg == 75)].iloc[0]
    assert g90.predicted_rms_wander_mm == pytest.approx(0.25, abs=0.01), (
        "the gneiss 90 deg modelled wander has moved; the pre-warp value 0.05 "
        "was quoted in Section 5 long after the warp was wired in")
    assert g90.observed_rms_wander_mm == pytest.approx(0.77, abs=0.01)
    assert s75.predicted_rms_wander_mm == pytest.approx(0.27, abs=0.01)
    assert s75.observed_rms_wander_mm == pytest.approx(0.90, abs=0.01)
    p = stats.ttest_rel(r.predicted_rms_wander_mm, r.observed_rms_wander_mm).pvalue
    assert p == pytest.approx(0.004, abs=0.001)
    assert "$p = 0.002$" not in tex, "the superseded roughness p-value is back"


def test_the_griffith_substitution_does_not_change_the_governing_locus():
    from tools import criterion_consistency as cc
    t = cc.consistency_table()
    g0 = t[(t.rock == G) & (t.angle_deg == 0)].iloc[0]
    assert g0.lam_tension == pytest.approx(1.005, abs=0.005)
    assert g0.lam_griffith == pytest.approx(1.003, abs=0.005)
    assert g0.lam_mohr_coulomb == pytest.approx(1.21, abs=0.01)
    # the argument: swapping the tensile criterion barely moves the factor,
    # and both stay well below the shear factor, so the ordering is unchanged
    assert abs(g0.lam_griffith - g0.lam_tension) / g0.lam_tension < 0.002
    assert max(g0.lam_tension, g0.lam_griffith) < g0.lam_mohr_coulomb


def test_the_three_criteria_are_commensurate():
    from tools import criterion_consistency as cc
    t = cc.consistency_table()
    med = {}
    for c in ("lam_tension", "lam_mohr_coulomb", "lam_griffith"):
        v = t[c].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        med[c] = float(np.median(np.abs(np.log(v))))
    assert min(med.values()) == pytest.approx(0.11, abs=0.01)
    assert max(med.values()) == pytest.approx(0.17, abs=0.01)


def test_opening_is_gated_by_the_sign_of_the_resolved_traction():
    """Section 5.5's structural argument for trusting the sequence."""
    t = pd.read_csv(TAB / "fabric_tractions.csv")
    for rock in (G, P):
        g = t[t.rock == rock].set_index("angle_deg").sigma_n_mean_MPa
        assert g.loc[60] < 0 < g.loc[75], (
            f"{rock}: the traction no longer changes sign between 60 and 75 deg")
        assert (g.loc[[0, 15, 30, 45, 60]] < 0).all()
    f = pd.read_csv(TAB / "fourclass_area_fractions.csv")
    clamped = f[f.angle_deg <= 45]
    assert (clamped.WT == 0).all(), (
        "weak-plane opening has appeared where the foliation is clamped shut")


def test_the_mechanism_ordering_survives_the_whole_proxy_sweep():
    d = pd.read_csv(TAB / "mechanism_robustness_sweep.csv")
    assert len(d) == 12, "the sweep is no longer twelve combinations"
    assert set(d.T_wp_over_T0) == {0.20, 0.35, 0.50, 0.70}
    assert set(d.c_wp_over_c) == {0.4, 0.6, 0.8}
    assert not d.gneiss_WT_at_or_below_45.any()
    assert not d.schist_WT_at_or_below_45.any(), (
        "opening now appears at or below 45 deg somewhere in the sweep, which "
        "is the claim Section 5.5 rests on")
    onsets = set(d.gneiss_WT_onset_deg) | set(d.schist_WT_onset_deg)
    assert onsets <= {60, 75, 90} and min(onsets) == 60 and max(onsets) == 90, (
        f"the opening onset now spans {sorted(onsets)}; Section 5.5 says 60 to 90")
    assert int(d.production.sum()) == 1


def test_the_three_orientations_of_section_5_1():
    """Widest competition, greatest shear and least strength."""
    t = pd.read_csv(TAB / "fabric_tractions.csv")
    shear_peak = {r: int(g.loc[g.tau_abs_mean_MPa.idxmax(), "angle_deg"])
                  for r, g in t.groupby("rock")}
    assert shear_peak[G] == 45 and shear_peak[P] == 30
    g45 = t[(t.rock == G) & (t.angle_deg == 45)].iloc[0]
    assert g45.sigma_n_mean_MPa == pytest.approx(-9.0, abs=0.3), (
        "Section 5.1 quotes -9.0 MPa as the clamping traction at the gneiss "
        "competition maximum")
    f = pd.read_csv(TAB / "fourclass_area_fractions.csv").set_index(["rock", "angle_deg"])
    assert f.loc[(P, 60.0)].WS > f.loc[(P, 60.0)].MS
    assert f.loc[(G, 60.0)].WS == pytest.approx(0.215, abs=0.002)
    assert f.loc[(G, 60.0)].MS == pytest.approx(0.238, abs=0.002)
