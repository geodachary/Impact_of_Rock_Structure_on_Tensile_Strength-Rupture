"""Section 4.5's deviation statistics and the mixed-mode claim beside them.

The deviations are correct but reproduce only under the production definition:
angle from the weak-plane *line*, r <= 0.985R, top 20% of points by tensile
drive. Read over the whole interior the 0 degree median is 10.1 rather than 2.6,
which looks like an error and is not. The pooled figures need
``pooled_equal_weight_abs_delta``; a median of per-angle medians gives 34.5.

The mixed-mode sentence placed widest competition near 60 degrees. Every source
puts it at the low angles, with almost none from 60 to 75.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import export, lithology as lith
from tools import strain_partitioning as sp
from tools.ddm._toolkit import deviation_arrays, pooled_equal_weight_abs_delta

_have = all(lith.field_cache_path(s).exists() for s in range(1, 15))
pytestmark = pytest.mark.skipif(not _have, reason="fields not exported yet")

GNEISS_MEDIANS = [2.6, 12.7, 26.7, 42.3, 58.5, 74.5, 86.9]


def _records():
    out = {}
    for sid in range(1, 15):
        d = np.load(lith.field_cache_path(sid), allow_pickle=True)
        M = d["M"].astype(bool)
        X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
        pack = dict(r_pts=np.hypot(X, Y)[M], R=float(d["R_m"]),
                    th=np.asarray(d["th_rad"], float)[M],
                    alpha_wp_line=float(d["alpha_wp_line_rad"]),
                    sxx=np.asarray(d["sxx"], float)[M],
                    syy=np.asarray(d["syy"], float)[M],
                    txy=np.asarray(d["txy"], float)[M])
        dd, _ = deviation_arrays(pack)
        L = lith.lithology_of_sample(sid)
        out.setdefault(L.display_name, []).append(
            dict(angle=L.angle_for_sample(sid), delta_deg=dd))
    return out


@pytest.fixture(scope="module")
def records():
    return _records()


def test_the_per_angle_medians_match_the_quoted_sequence(records):
    got = {r["angle"]: float(np.median(np.abs(r["delta_deg"])))
           for r in records["Augen gneiss"]}
    for angle, want in zip(sorted(got), GNEISS_MEDIANS):
        assert got[angle] == pytest.approx(want, abs=0.05), (
            f"gneiss at {angle} deg: {got[angle]:.1f}, the paper says {want}")


def test_the_two_lithologies_agree_to_within_the_quoted_margin(records):
    g = {r["angle"]: float(np.median(np.abs(r["delta_deg"])))
         for r in records["Augen gneiss"]}
    s = {r["angle"]: float(np.median(np.abs(r["delta_deg"])))
         for r in records["Psammitic schist"]}
    worst = max(abs(g[a] - s[a]) for a in g)
    assert worst <= 2.8, (
        f"the lithologies now differ by {worst:.2f} deg; both the caption and "
        "the text say 2.8")
    assert worst > 1.0, (
        "the caption previously claimed agreement to within a degree, which "
        f"was wrong at {worst:.2f}; if that is now true, fix the text too")


def test_the_pooled_medians_need_the_equal_weight_pooling(records):
    want = {"Augen gneiss": 33.1, "Psammitic schist": 27.1}
    for rock, expect in want.items():
        rr = [r for r in records[rock] if r["angle"] != 90]
        got = float(np.median(pooled_equal_weight_abs_delta(rr)))
        assert got == pytest.approx(expect, abs=0.15), (
            f"{rock}: pooled median excluding 90 deg is {got:.1f}, the paper "
            f"says {expect}")


def test_mixed_mode_competition_peaks_at_low_angle_not_at_the_minimum():
    """The corrected claim, asserted as a shape rather than two numbers."""
    st = sp.specimen_strengths()
    frac = {}
    for sid in range(1, 15):
        p = export.classification_panel(sid, st)
        m = np.asarray(p["mixed_flag"])[p["core_mask"]]
        frac.setdefault(p["lithology"], {})[int(p["angle_deg"])] = 100.0 * m.mean()

    assert frac["Augen gneiss"][0] == pytest.approx(20.8, abs=0.2)
    assert frac["Psammitic schist"][15] == pytest.approx(20.7, abs=0.2)
    assert frac["Augen gneiss"][75] == pytest.approx(0.6, abs=0.15)
    assert frac["Psammitic schist"][75] == pytest.approx(0.0, abs=0.05)
    for rock, f in frac.items():
        peak = max(f, key=lambda a: f[a])
        assert peak <= 15, (
            f"{rock}: mixed-mode competition now peaks at {peak} deg. The text "
            "says it is greatest at the low angles and near zero at the "
            "strength minimum.")
        assert f[75] < f[peak] / 10.0, (
            f"{rock}: mixed-mode fraction at 75 deg is no longer negligible "
            "beside its peak")
