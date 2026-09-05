"""Section 4.7's two sweeps, and the scale-invariance argument behind them.

The subsection makes a mechanical claim that no other test covers: the failed
area is insensitive to spacing because the weakened band is tied to the
spacing through ``b_w = 0.12 s``, so the weakened area fraction is
scale-invariant at 2 b_w / s = 0.24. That is asserted here against the
archived weights, and it is the reason the spacing curve is flat rather than
an observation that it happens to be.

Nothing else pins the sweep magnitudes either. They live in the figure caption
and in this paragraph, and both are prose.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import fabric_tractions as ft
from tools import lithology as lith

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"

pytestmark = pytest.mark.skipif(not (TAB / "sensitivity_spacing.csv").is_file(),
                                reason="sweeps not generated yet")


def test_the_weakened_area_fraction_is_scale_invariant():
    """The mechanism: b_w scales with s, so the band fraction does not."""
    got = {}
    for sid in (1, 8):
        d = np.load(lith.field_cache_path(sid), allow_pickle=True)
        M = d["M"].astype(bool)
        r = np.hypot(np.asarray(d["X"], float), np.asarray(d["Y"], float))
        core = M & (r <= ft.CORE_FRAC * float(d["R_m"]))
        w = np.asarray(d["wp_weight"], float)[core]
        got[lith.lithology_of_sample(sid).display_name] = float(
            np.mean(w >= np.exp(-1.0)))

    spacings = {l.display_name: l.spacing_m for l in lith.LITHOLOGIES.values()}
    assert spacings["Augen gneiss"] / spacings["Psammitic schist"] == pytest.approx(5.0)
    for rock, frac in got.items():
        assert frac == pytest.approx(2 * 0.12, abs=0.03), (
            f"{rock}: {frac:.3f} of the interior lies within one band "
            f"half-width, against 2 b_w / s = 0.24 by construction")
    spread = abs(got["Augen gneiss"] - got["Psammitic schist"])
    assert spread < 0.02, (
        f"the band fraction now differs by {spread:.3f} between rocks whose "
        "spacings differ fivefold; the scale invariance the paragraph rests "
        "on has been lost")


def test_the_spacing_response_is_flat_against_the_introduction_step():
    d = pd.read_csv(TAB / "sensitivity_spacing.csv")
    want = {"Augen gneiss": 0.17, "Psammitic schist": 0.11}
    for rock, flat in want.items():
        s = d[(d.rock == rock) & (d.spacing_m > 0)].groupby(
            "spacing_m").delta_failure_pct.mean()
        spread = float(s.max() - s.min())
        assert spread == pytest.approx(flat, abs=0.01), (
            f"{rock}: spacing moves the failed area by {spread:.3f} pp, the "
            f"paper says {flat}")
        step = float(s.mean())
        assert step / spread > 30.0, (
            f"{rock}: the flat range is now {step / spread:.0f} times smaller "
            "than the introduction step, not the fiftieth the text claims")


def test_disorder_grows_monotonically_and_does_not_saturate():
    d = pd.read_csv(TAB / "sensitivity_heterogeneity.csv")
    g = d.groupby(["rock", "heterogeneity"]).delta_failure_pct.mean().unstack(0)
    want = {"Augen gneiss": 12.8, "Psammitic schist": 9.7}
    for rock, at04 in want.items():
        v = g[rock].to_numpy(float)
        assert np.all(np.diff(v) > 0), f"{rock}: no longer monotonic"
        assert v[-1] == pytest.approx(at04, abs=0.05), (
            f"{rock}: {v[-1]:.2f} pp at amplitude 0.4, paper says {at04}")
        assert np.diff(v)[-1] >= np.diff(v)[-2], (
            f"{rock}: the response is saturating; the text says it is still "
            "accelerating at the largest amplitude sampled")
    assert (g["Augen gneiss"].iloc[1:] > g["Psammitic schist"].iloc[1:]).all(), \
        "the gneiss is no longer the more sensitive at every amplitude"
