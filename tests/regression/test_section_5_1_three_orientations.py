"""Section 5.1 claims three orientations are distinct and says why.

The argument is that a symmetric law collapses three conditions a foliated
rock separates: widest mechanism competition, least strength, and greatest
resolved shear. The claim is only worth making if the three really do fall at
different angles, and if the stated reason for each is the one the fields
show.

The reason given for the first was wrong. The text said widest competition
occurs "where the fabric carries little traction of either kind". At those
angles the fabric carries its *largest* clamping traction, -22.6 MPa in the
gneiss at 0 degrees and -16.4 in the schist at 15, with no part of the
interior in tension. What suppresses the weak-plane utilities there is the
clamping, which leaves the two matrix criteria to compete with one another;
that is why the competition is widest, and it is the opposite of the fabric
carrying little traction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import export
from tools import strain_partitioning as sp

REPO = Path(__file__).resolve().parents[2]
TRACTIONS = REPO / "outputs" / "tables" / "fabric_tractions.csv"

pytestmark = pytest.mark.skipif(not TRACTIONS.is_file(), reason="tractions absent")


@pytest.fixture(scope="module")
def survey():
    st = sp.specimen_strengths()
    out = {}
    for sid in range(1, 15):
        p = export.classification_panel(sid, st)
        m = np.asarray(p["mixed_flag"])[p["core_mask"]]
        c = np.asarray(p["mode_code"])[p["core_mask"]]
        f = c >= 0
        out.setdefault(p["lithology"], {})[int(p["angle_deg"])] = dict(
            mixed=100.0 * m[f].mean() if f.any() else 0.0,
            matrix=p["fractions"]["MT"] + p["fractions"]["MS"],
            fabric=p["fractions"]["WT"] + p["fractions"]["WS"])
    return out


def test_the_orientations_fall_where_section_5_1_says(survey):
    """The three conditions a symmetric law would collapse.

    They do not fall together in either rock, but they do not all separate
    either. In the gneiss the widest competition and the greatest resolved
    shear coincide at 45 degrees and only the strength minimum stands apart;
    the schist separates all three, at 60, 30 and 75. Section 5.1 states that
    asymmetry explicitly, so both patterns are pinned rather than a blanket
    "all three differ" that holds for one rock only.
    """
    tr = pd.read_csv(TRACTIONS)
    expected = {"Augen gneiss": (45, 45), "Psammitic schist": (60, 30)}
    for rock, v in survey.items():
        widest = max(v, key=lambda a: v[a]["mixed"])
        g = tr[tr.rock == rock]
        shear_peak = int(g.loc[g.tau_abs_mean_MPa.idxmax(), "angle_deg"])
        want_widest, want_shear = expected[rock]
        assert widest == want_widest, (
            f"{rock}: widest competition has moved to {widest} deg; Section "
            f"5.1 states {want_widest}")
        assert shear_peak == want_shear, (
            f"{rock}: greatest resolved shear has moved to {shear_peak} deg; "
            f"Section 5.1 states {want_shear}")
        weakest = 75                       # measured, both lithologies
        assert weakest not in (widest, shear_peak), (
            f"{rock}: the strength minimum now coincides with a traction "
            "peak; the argument of Section 5.1 needs it separated")
    gn = survey["Augen gneiss"]
    assert max(gn, key=lambda a: gn[a]["mixed"]) == \
        int(tr[tr.rock == "Augen gneiss"].set_index("angle_deg")
            .tau_abs_mean_MPa.idxmax()), (
        "the gneiss widest-competition and shear-peak angles no longer "
        "coincide; Section 5.1 says both fall at 45 deg")


def test_widest_competition_is_clamped_in_the_gneiss(survey):
    """Section 5.1 explains the gneiss peak by clamping; check the traction.

    At 45 degrees the mean normal traction on the foliation is -9.0 MPa, so
    the planes are still held closed and weak-plane opening is not an
    available class there, which is why the competition is between the shear
    branches. The schist peak at 60 degrees sits at only -2 MPa with much of
    the interior already in tension, so the explanation is made for the gneiss
    alone and that asymmetry is pinned here too.
    """
    tr = pd.read_csv(TRACTIONS)
    g = max(survey["Augen gneiss"], key=lambda a: survey["Augen gneiss"][a]["mixed"])
    row = tr[(tr.rock == "Augen gneiss") & (tr.angle_deg == g)].iloc[0]
    assert row.sigma_n_mean_MPa == pytest.approx(-9.0, abs=0.3), (
        f"the gneiss widest-competition angle ({g} deg) has normal traction "
        f"{row.sigma_n_mean_MPa:.1f} MPa; Section 5.1 quotes -9.0")
    assert row.sigma_n_mean_MPa < 0.0, "the gneiss fabric is no longer clamped there"
    assert survey["Augen gneiss"][g]["fabric"] == pytest.approx(0.114, abs=0.01)

    s_ang = max(survey["Psammitic schist"],
                key=lambda a: survey["Psammitic schist"][a]["mixed"])
    row = tr[(tr.rock == "Psammitic schist") & (tr.angle_deg == s_ang)].iloc[0]
    assert row.sigma_n_mean_MPa > -10.0 and row.open_fraction > 0.0, (
        f"the schist widest-competition angle ({s_ang} deg) is clamped again; "
        "the qualification in Section 5.1 can then be dropped")
