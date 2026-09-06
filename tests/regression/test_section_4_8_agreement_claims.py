"""Section 4.8's G/Gc profile statistics and its mixed-mode diagnostic.

The profile metrics are generated into ``fig20_GGc_band_summary.csv`` and
``fig20_GGc_profile_metrics.csv``, but the manuscript quoted neither. Its band
coefficients of variation were 3.0, 4.4 and 3.1 where the table gives 3.63,
3.43 and 2.34, and it had the intermediate band as the largest when the low
band is. Its shear-bearing step fractions at 0 degrees were 65% and 78%
against 0.616 and 0.894 in the table.

The second paragraph was worse than stale. It asserted that mixed-mode
fraction correlates with fragmentation style, specimens above 20% breaking
into pieces and those below 10% splitting once. The correlation runs the other
way. Mixed-mode competition peaks at the low fabric angles, 52.9% in the
gneiss at 0 degrees and 43.0% in the schist at 15, exactly the specimens that
Fig. 3 records as splitting cleanly along the loading axis, and it is
essentially zero from 60 degrees up where the schist fragments into slabs.
The quoted peaks of 28% and 35% match nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import export
from tools import strain_partitioning as sp

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"

pytestmark = pytest.mark.skipif(
    not (TAB / "fig20_GGc_band_summary.csv").is_file(),
    reason="G/Gc metrics not generated yet")


def test_no_profile_is_monotonic_and_all_show_reversals():
    d = pd.read_csv(TAB / "fig20_GGc_profile_metrics.csv")
    assert len(d) == 14
    assert not d.monotonic.any(), "a G/Gc profile is now monotonic"
    for rock in d.lithology.unique():
        assert (d[d.lithology == rock].n_reversals > 0).all(), (
            f"{rock}: some profile no longer reverses; the text says "
            "reversals occur in both lithologies")


def test_the_band_coefficients_of_variation():
    d = pd.read_csv(TAB / "fig20_GGc_band_summary.csv").set_index("band")
    want = {"low (0-30 deg)": 3.73, "intermediate (45-60 deg)": 4.16,
            "high (75-90 deg)": 2.37}
    for band, v in want.items():
        assert d.loc[band, "mean_CV"] == pytest.approx(v, abs=0.05), (
            f"{band}: mean CV is {d.loc[band, 'mean_CV']:.2f}, paper says {v}")
    cv = d["mean_CV"]
    assert cv.max() / cv.min() < 2.0, (
        "the bands now differ more than twofold, which is the separation the "
        "text argues against")
    # Re-locked once the schist fields were rebuilt on the measured ratio; the
    # previous lock mixed a corrected gneiss with a schist still carrying the
    # abandoned 3.763. The low band is largest again, but the secondary "low
    # band is largest" argument stays out of Section 4.8: what the text argues
    # is the sub-twofold spread, asserted above, now 1.60.


def test_the_shear_bearing_step_fractions_at_zero_degrees():
    d = pd.read_csv(TAB / "fig20_GGc_profile_metrics.csv")
    at0 = d[d.angle_deg == 0].set_index("lithology")
    assert at0.loc["Augen gneiss", "shear_fraction"] == pytest.approx(0.94, abs=0.01)
    assert at0.loc["Psammitic schist", "shear_fraction"] == pytest.approx(0.89, abs=0.01)


def _mixed_of_failed():
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


def test_mixed_mode_has_an_interior_maximum_in_each_lithology():
    """Under the measured ratios each rock peaks at its own interior angle.

    With the adopted ratios both rocks peaked below 15 degrees, at 52.9 and
    43.0 per cent. The measured ratios move the gneiss peak to 45 degrees and
    the schist peak to 60, shrink both, and drive the fraction to zero at 0 and
    90 degrees in both rocks, which is what Section 4.8 now states.
    """
    f = _mixed_of_failed()
    g, sch = f["Augen gneiss"], f["Psammitic schist"]
    assert max(g.values()) == pytest.approx(13.5, abs=0.3)
    assert max(g, key=g.get) == 45
    assert max(sch.values()) == pytest.approx(19.9, abs=0.3)
    assert max(sch, key=sch.get) == 60
    for rock, v in f.items():
        assert v[0] == 0.0 and v[90] == 0.0, (
            f"{rock}: mixed-mode fraction is no longer zero at both 0 and 90 "
            "degrees; Section 4.8 rests on the interior maximum")


def test_mixed_mode_does_not_track_fragmentation():
    """The withdrawn claim. Fragmentation grows with angle; this does not.

    Fig. 3 records clean axial splitting at 0 to 30 degrees and slabbing above
    60, so fragmentation is most pronounced at 90 degrees. Section 4.8 rests on
    the mixed-mode fraction vanishing exactly there, in both lithologies. If it
    ever rose with angle instead, the correlation the text withdrew would be
    back and the paragraph would need revisiting rather than silently agreeing
    again.
    """
    f = _mixed_of_failed()
    for rock, v in f.items():
        peak = max(v, key=v.get)
        assert peak not in (0, 90), (
            f"{rock}: the mixed-mode maximum has moved to {peak} deg; the "
            "section argues it is interior")
        assert v[90] == 0.0, (
            f"{rock}: mixed-mode fraction is {v[90]:.1f}% at 90 deg, where "
            "fragmentation is greatest; the section says it vanishes there")
        assert v[peak] > v[60] or peak == 60, (
            f"{rock}: the interior maximum is no longer a maximum")


def test_the_three_matrix_criteria_are_commensurate_not_co_banded():
    """The three criteria are commensurate, not confined to a common band.

    Section 5.5 said all three fall "within the same band". Mohr-Coulomb spans
    0.74 to 2.46 against 0.84-1.42 and 0.76-1.23, so one specimen sits well
    outside the others. What carries the argument is that the median
    |ln lambda| values agree to within a factor of 1.6.
    """
    from tools import criterion_consistency as cc
    t = cc.consistency_table()
    med = {}
    for c in ("lam_tension", "lam_mohr_coulomb", "lam_griffith"):
        v = t[c].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        med[c] = float(np.median(np.abs(np.log(v))))
    lo, hi = min(med.values()), max(med.values())
    assert lo == pytest.approx(0.105, abs=0.01) and hi == pytest.approx(0.167, abs=0.01), (
        f"median |ln lambda| now spans {lo:.3f} to {hi:.3f}; Section 5.5 quotes "
        "0.11 to 0.17")
    assert hi / lo < 2.0, "the criteria are no longer commensurate"
    # and the reason the old wording was wrong must still hold
    mc = t["lam_mohr_coulomb"].to_numpy(float)
    assert mc.max() > 2.0, (
        "Mohr-Coulomb now falls inside the 0.5-2.0 band on every specimen, so "
        "the 'same band' wording would be defensible again")
