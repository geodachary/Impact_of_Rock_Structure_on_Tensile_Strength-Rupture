"""Section 4.2's fit statistics, including two with no producer.

The cosine-law comparison (minima at 85.4 and 86.6 degrees, R2 0.877 and 0.889)
is the argument that a symmetric law misplaces the minimum, but
``ati_model.cosine_law`` is called by nothing. It reproduces only under an
*unweighted* fit; the envelope beside it is weighted, which is not obvious from
the text and is pinned here.

The per-angle departures were attributed to 75 degrees. They peak at 45, where
the envelope is furthest from the end members it is pinned to.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import differential_evolution

from tools import ati_model as am

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
RAW = REPO / "selected_all_samples.csv"

pytestmark = pytest.mark.skipif(not RAW.is_file(), reason="replicate data absent")

ROCKS = ("Psammitic schist", "Augen gneiss")


@pytest.fixture(scope="module")
def data():
    return {r: g for r, g in pd.read_csv(RAW).groupby("Rock_type")}


def _cosine_fit(g):
    """Unweighted three-parameter cosine law: minimum position and R^2."""
    g = g.dropna(subset=["Angle", "Tensile_strength_Mpa"])
    a = g["Angle"].to_numpy(float)
    y = g["Tensile_strength_Mpa"].to_numpy(float)
    hi = y.max()
    res = differential_evolution(
        lambda p: float(np.sum((am.cosine_law(a, *p) - y) ** 2)),
        [(0.0, 2 * hi), (-hi, hi), (0.0, 180.0)],
        seed=42, maxiter=4000, popsize=30, tol=1e-12, polish=True,
        updating="deferred")
    dense = np.linspace(0.0, 90.0, 4001)
    ym = am.cosine_law(dense, *res.x)
    pred = am.cosine_law(a, *res.x)
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(dense[int(np.argmin(ym))]), float(r2)


def test_the_cosine_law_comparison_reproduces(data):
    """The numbers Section 4.2 quotes against the adopted envelope."""
    want = {"Augen gneiss": (85.4, 0.877), "Psammitic schist": (86.6, 0.889)}
    for rock, (deg, r2) in want.items():
        got_deg, got_r2 = _cosine_fit(data[rock])
        assert got_deg == pytest.approx(deg, abs=0.15), (
            f"{rock}: cosine-law minimum is at {got_deg:.1f} deg, the paper "
            f"says {deg}")
        assert got_r2 == pytest.approx(r2, abs=0.002), (
            f"{rock}: cosine-law R2 is {got_r2:.3f}, the paper says {r2}")


def test_the_cosine_minimum_is_displaced_by_more_than_ten_degrees(data):
    """The claim the comparison exists to support."""
    for rock in ROCKS:
        deg, _ = _cosine_fit(data[rock])
        assert deg - 75.0 > 10.0, (
            f"{rock}: the cosine law now places its minimum {deg - 75:.1f} deg "
            "from the measured one; the text says more than ten")


def test_per_angle_departures_peak_midway_between_the_end_members(data):
    """Magnitude and location, the latter having been wrong."""
    want = {"Psammitic schist": 8.6, "Augen gneiss": 3.5}
    for rock, pct in want.items():
        r = am.fit(data[rock])
        st = am.per_angle_statistics(data[rock])
        pred = am.strength(np.deg2rad(st.index.to_numpy(float)), r["sigma0"],
                           r["sigma90"], r["eta"], r["beta_peak_deg"])
        dep = 100.0 * np.abs(pred - st["mean"].to_numpy()) / st["mean"].to_numpy()
        assert dep.max() == pytest.approx(pct, abs=0.1), (
            f"{rock}: largest per-angle departure is {dep.max():.1f}%, the "
            f"paper says {pct}%")
        assert st.index[int(np.argmax(dep))] == 45, (
            f"{rock}: largest departure is now at "
            f"{st.index[int(np.argmax(dep))]} deg, not 45. The envelope is "
            "pinned at 0 and 90, which is why the worst fit sits midway.")
        assert dep[st.index.get_loc(0)] == pytest.approx(0.0, abs=1e-6)
        assert dep[st.index.get_loc(90)] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_text_no_longer_blames_the_weakest_orientation():
    said = re.search(r"Per-angle departures from the measured means reach "
                     r"\$([\d.]+)\$\\% in the schist and\s*\n?\$([\d.]+)\$\\% in "
                     r"the gneiss, both at \$(\d+)\^\\circ\$",
                     TEX.read_text(encoding="utf-8"))
    assert said, "the per-angle departure sentence is no longer in its expected form"
    assert int(said.group(3)) == 45, (
        f"the text attributes the largest departures to {said.group(3)} deg again")
