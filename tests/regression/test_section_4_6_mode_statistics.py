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
    want = [("Augen gneiss", 90, "WT", 0.144), ("Psammitic schist", 90, "WT", 0.151),
            ("Augen gneiss", 60, "WS", 0.215), ("Psammitic schist", 45, "WS", 0.208),
            ("Augen gneiss", 0, "MS", 0.319), ("Psammitic schist", 0, "MS", 0.372),
            ("Augen gneiss", 0, "MT", 0.000), ("Psammitic schist", 15, "MT", 0.000)]
    for rock, ang, cls, v in want:
        assert _at(fourclass, rock, ang, cls) == pytest.approx(v, abs=0.001), (
            f"{rock} {cls} at {ang} deg")


def test_matrix_shear_vanishes_above_sixty_degrees(fourclass):
    for rock in fourclass.rock.unique():
        for ang in (75, 90):
            assert _at(fourclass, rock, ang, "MS") == 0.0, (
                f"{rock}: matrix shear is no longer absent at {ang} deg")


def test_the_failed_fraction_ordering_alternates(stats):
    """Neither lithology leads systematically under the measured ratios.

    With the adopted ratios the schist led from 0 to 45 degrees and the gneiss
    from 60 upward, a clean crossover the text described. That crossover is
    gone: the ordering now alternates, so Section 4.6 states the pattern
    instead of a reversal.
    """
    g = {int(r.angle_deg): r.p_fail for r in
         stats[stats.rock == "Augen gneiss"].itertuples()}
    s = {int(r.angle_deg): r.p_fail for r in
         stats[stats.rock == "Psammitic schist"].itertuples()}
    for a in (0, 30, 45):
        assert s[a] > g[a], f"the schist no longer leads at {a} deg"
    for a in (15, 60, 75, 90):
        assert g[a] > s[a], f"the gneiss no longer leads at {a} deg"


def test_the_criterion_load_factors(stats):
    from tools import criterion_consistency as cc
    t = cc.consistency_table()
    want = {"lam_tension": (0.841, 1.418, 0.105, 6),
            "lam_mohr_coulomb": (0.736, 2.462, 0.167, 6),
            "lam_griffith": (0.762, 1.225, 0.111, 6)}
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
    assert d.max_index.median() == pytest.approx(0.180, abs=0.005), (
        f"median departure is {d.max_index.median():.3f}")
    below = d[d.max_index <= 0.10]
    assert len(below) == 4, f"{len(below)} solutions fall below the threshold"
    # Three of the four are fabric end members, as the argument expects. The
    # fourth is the schist at 45 degrees, which sits an order of magnitude
    # below its own neighbours (0.036 against 0.344 and 0.341). That dip is
    # recorded rather than smoothed over: the paragraph says departures are
    # small at the end members, and this one specimen is an exception to it.
    assert set(below.angle_deg) == {0, 45, 90}, (
        "the sub-threshold cases are no longer exactly the fabric end members: "
        f"{below[['lithology', 'angle_deg']].to_dict('records')}")
    g = d[(d.lithology == "Augen gneiss") & (d.angle_deg == 45)].max_index.iloc[0]
    s = d[(d.lithology == "Psammitic schist") & (d.angle_deg == 45)].max_index.iloc[0]
    assert round(float(s), 2) == 0.04 and round(float(g), 2) == 0.20, (
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
