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


def test_the_three_orientations_are_distinct(survey):
    tr = pd.read_csv(TRACTIONS)
    for rock, v in survey.items():
        widest = max(v, key=lambda a: v[a]["mixed"])
        g = tr[tr.rock == rock]
        shear_peak = int(g.loc[g.tau_abs_mean_MPa.idxmax(), "angle_deg"])
        weakest = 75                       # measured, both lithologies
        assert len({widest, shear_peak, weakest}) == 3, (
            f"{rock}: widest competition, greatest shear and least strength "
            f"fall at {widest}, {shear_peak} and {weakest}; the argument needs "
            "them separated")


def test_widest_competition_is_where_the_fabric_is_clamped(survey):
    tr = pd.read_csv(TRACTIONS)
    for rock, v in survey.items():
        widest = max(v, key=lambda a: v[a]["mixed"])
        row = tr[(tr.rock == rock) & (tr.angle_deg == widest)].iloc[0]
        assert row.sigma_n_mean_MPa < -10.0, (
            f"{rock}: at the widest-competition angle ({widest} deg) the "
            f"normal traction is {row.sigma_n_mean_MPa:.1f} MPa. The text says "
            "the planes are clamped there.")
        assert row.open_fraction == 0.0, (
            f"{rock}: part of the interior now has the foliation in tension at "
            f"{widest} deg")
        assert v[widest]["matrix"] > 10 * max(v[widest]["fabric"], 1e-6), (
            f"{rock}: failure at {widest} deg is no longer matrix-dominated, "
            "so the competition is not between the two matrix criteria")
