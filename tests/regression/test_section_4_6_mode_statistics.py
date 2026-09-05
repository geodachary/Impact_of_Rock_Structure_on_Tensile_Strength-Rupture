"""Section 4.6's class fractions, criterion calibration and symmetry indices.

The fractions were right; three claims about ordering were not. The failed
fraction was said to return to the schist at 90 degrees (the gneiss leads there,
14.0% against 11.1%); the mirror-symmetry median was 0.45 rather than 0.50, with
the sub-threshold cases miscounted; and the largest displacement amplitude was
placed at 15 degrees when it is at 45 under either profile convention.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"

pytestmark = pytest.mark.skipif(
    not (TAB / "fourclass_area_fractions.csv").is_file(),
    reason="tables not generated yet")


@pytest.fixture(scope="module")
def fourclass():
    return pd.read_csv(TAB / "fourclass_area_fractions.csv")


@pytest.fixture(scope="module")
def stats():
    return pd.read_csv(TAB / "failure_statistics.csv")


def _at(df, rock, angle, col):
    v = df[(df.rock == rock) & (df.angle_deg == angle)][col]
    assert len(v) == 1
    return float(v.iloc[0])


def test_the_quoted_class_fractions(fourclass):
    want = [("Augen gneiss", 90, "WT", 0.128), ("Psammitic schist", 90, "WT", 0.101),
            ("Augen gneiss", 60, "WS", 0.212), ("Psammitic schist", 60, "WS", 0.169),
            ("Augen gneiss", 0, "MS", 0.273), ("Psammitic schist", 0, "MS", 0.541),
            ("Augen gneiss", 0, "MT", 0.120), ("Psammitic schist", 15, "MT", 0.012)]
    for rock, ang, cls, v in want:
        assert _at(fourclass, rock, ang, cls) == pytest.approx(v, abs=0.001), (
            f"{rock} {cls} at {ang} deg")


def test_matrix_shear_vanishes_above_sixty_degrees(fourclass):
    for rock in fourclass.rock.unique():
        for ang in (75, 90):
            assert _at(fourclass, rock, ang, "MS") == 0.0, (
                f"{rock}: matrix shear is no longer absent at {ang} deg")


def test_the_failed_fraction_ordering_does_not_return_to_the_schist(stats):
    g = {int(r.angle_deg): r.p_fail for r in
         stats[stats.rock == "Augen gneiss"].itertuples()}
    s = {int(r.angle_deg): r.p_fail for r in
         stats[stats.rock == "Psammitic schist"].itertuples()}
    for a in (0, 15, 30, 45):
        assert s[a] > g[a], f"the schist no longer leads at {a} deg"
    for a in (60, 75, 90):
        assert g[a] > s[a], (
            f"at {a} deg the schist leads again ({s[a]:.1f}% against "
            f"{g[a]:.1f}%); the text says the reversal from 60 deg does not "
            "return, including at 90")


def test_the_criterion_load_factors(stats):
    from tools import criterion_consistency as cc
    t = cc.consistency_table()
    want = {"lam_tension": (0.81, 1.44, 0.18, 6),
            "lam_mohr_coulomb": (0.73, 2.71, 0.14, 5),
            "lam_griffith": (0.68, 1.25, 0.20, 6)}
    for col, (lo, hi, med, early) in want.items():
        v = t[col].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        assert v.min() == pytest.approx(lo, abs=0.005), col
        assert v.max() == pytest.approx(hi, abs=0.005), col
        assert np.median(np.abs(np.log(v))) == pytest.approx(med, abs=0.005), col
        assert int((v < 1).sum()) == early, (
            f"{col}: fires early on {(v < 1).sum()} of 14, text says {early}")


def test_the_mirror_symmetry_statistics():
    d = pd.read_csv(TAB / "displacement_mirror_symmetry.csv")
    assert d.max_index.median() == pytest.approx(0.50, abs=0.005), (
        f"median departure is {d.max_index.median():.3f}")
    below = d[d.max_index <= 0.10]
    assert len(below) == 4, f"{len(below)} solutions fall below the threshold"
    assert set(below.angle_deg) == {0, 90}, (
        "the sub-threshold cases are no longer exactly the fabric end members: "
        f"{below[['lithology', 'angle_deg']].to_dict('records')}")
    g = d[(d.lithology == "Augen gneiss") & (d.angle_deg == 45)].max_index.iloc[0]
    s = d[(d.lithology == "Psammitic schist") & (d.angle_deg == 45)].max_index.iloc[0]
    assert round(float(s), 2) == 0.83 and round(float(g), 2) == 0.75, (
        f"the 45 degree pair is now {s:.3f} against {g:.3f}")


def test_the_displacement_amplitude_peaks_at_forty_five_degrees():
    """Not at 15 degrees, under either way of taking the profile."""
    from tools import lithology as lith
    files = sorted(glob.glob(str(REPO / "outputs/fields/_cache_uv_profiles_v2/*.npz")))
    if not files:
        pytest.skip("displacement profile cache absent")
    for axis in (0, 1):
        peak = {}
        for f in files:
            idx = int(re.search(r"idx(\d+)", f).group(1))
            d = np.load(f, allow_pickle=True)
            uu = np.where(d["mask"].astype(bool), d["u"], np.nan)
            L = lith.lithology_of_sample(idx)
            peak.setdefault(L.display_name, {})[L.angle_for_sample(idx)] = \
                float(np.nanmax(np.abs(np.nanmean(uu, axis=axis))))
        for rock, v in peak.items():
            assert max(v, key=v.get) != 15, (
                f"{rock}: the 15 degree case is now the largest amplitude "
                f"(axis={axis}); the text no longer claims that")
