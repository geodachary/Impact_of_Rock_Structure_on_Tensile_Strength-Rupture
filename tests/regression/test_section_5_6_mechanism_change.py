"""Section 5.6 turns the displaced minimum into a measurement, so it must be one.

The claim is that a minimum away from 45 degrees measures the orientation at
which the governing mechanism passes from sliding on the fabric to opening it.
Two things have to hold for that to be a measurement rather than a coincidence
of wording: the governing class must really change there, and the measured
strength minimum must fall at the same angle in both rocks.

It also claims that two specimens of the same apparent strength can fail by
different mechanisms, which is the argument for reporting locus alongside peak
load. That needs at least one such pair to exist in this dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
FC = REPO / "outputs" / "tables" / "fourclass_area_fractions.csv"
RAW = REPO / "selected_all_samples.csv"

pytestmark = pytest.mark.skipif(not FC.is_file(), reason="fractions absent")

CLASSES = ("WT", "WS", "MT", "MS")


def _governing():
    fc = pd.read_csv(FC)
    out = {}
    for rock, g in fc.groupby("rock"):
        g = g.sort_values("angle_deg")
        out[rock] = {int(r.angle_deg): max(CLASSES, key=lambda c: getattr(r, c))
                     for r in g.itertuples()}
    return out


def test_the_governing_class_turns_from_sliding_to_opening_at_the_minimum():
    gov = _governing()
    T = pd.read_csv(RAW).groupby(["Rock_type", "Angle"]).Tensile_strength_Mpa.mean()
    for rock, seq in gov.items():
        angles = sorted(seq)
        assert seq[60] == "WS", f"{rock}: 60 deg is governed by {seq[60]}, not sliding"
        assert seq[75] == "WT", f"{rock}: 75 deg is governed by {seq[75]}, not opening"
        weakest = min(angles, key=lambda a: T[(rock, a)])
        assert weakest == 75, (
            f"{rock}: the measured minimum is at {weakest} deg, so it no longer "
            "coincides with the sliding-to-opening change")


def test_the_change_is_a_real_transition_not_a_relabelling():
    """Matrix shear must hold the low angles, so there is something to leave."""
    gov = _governing()
    for rock, seq in gov.items():
        for a in (0, 15, 30, 45):
            assert seq[a] == "MS", (
                f"{rock}: {a} deg is governed by {seq[a]}; the sequence the "
                "argument rests on is matrix shear, then sliding, then opening")


def test_equal_strength_specimens_can_differ_in_locus():
    gov = _governing()
    T = pd.read_csv(RAW).groupby(["Rock_type", "Angle"]).Tensile_strength_Mpa.mean()
    pairs = [(r, a, float(T[(r, a)]), gov[r][a]) for r in gov for a in gov[r]]
    found = [(x, y) for i, x in enumerate(pairs) for y in pairs[i + 1:]
             if abs(x[2] - y[2]) < 0.15 and x[3] != y[3]]
    assert found, (
        "no two specimens within 0.15 MPa of each other now differ in governing "
        "locus; Section 5.6 argues from exactly that case")
