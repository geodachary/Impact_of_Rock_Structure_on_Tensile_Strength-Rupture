"""Section 2.7 describes how the strength envelope is built; the code must match.

Nothing here is quoted as a number, so the check is structural: each stated
property of the kernel and of the fit is asserted against ``tools.ati_model``.

One claim was wrong. The text said beta_p "is bounded to (0, 90) by geometry
and therefore requires no arbitrary limit". The optimiser is bounded to
[30, 89] and the profile scan to [45, 89]. Those limits exist for a reason,
since kappa diverges at both end members, but they are limits, and the sentence
denied there were any. Both fits and both intervals sit well inside them, so
the restriction does not bind, which is what the corrected sentence says.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import ati_model as A

REPO = Path(__file__).resolve().parents[2]
PARAMS = REPO / "outputs" / "tables" / "ati_envelope_parameters.csv"


def test_the_kernel_vanishes_with_its_slope_at_both_end_members():
    for peak in (60.0, 72.6, 83.5):
        for edge in (0.0, np.pi / 2):
            assert A.weakening_kernel(edge, peak) == pytest.approx(0.0, abs=1e-12)
        h = 1e-6
        for edge in (0.0, np.pi / 2):
            near = A.weakening_kernel(edge + (h if edge == 0 else -h), peak)
            assert near < 1e-9, (
                f"kernel does not approach zero smoothly at {np.degrees(edge):.0f} "
                f"deg for beta_p={peak}: {near:.3e}")


def test_the_kernel_is_normalised_to_unit_maximum():
    for peak in (45.0, 72.6, 83.5):
        b = np.linspace(0.0, np.pi / 2, 20001)
        k = A.weakening_kernel(b, peak)
        assert k.max() == pytest.approx(1.0, abs=1e-6), f"max is {k.max():.6f}"
        at = np.degrees(b[int(np.argmax(k))])
        assert at == pytest.approx(peak, abs=0.2), (
            f"kernel peaks at {at:.2f} deg, not at beta_p={peak}")


def test_the_envelope_reaches_the_measured_end_members_exactly():
    s0, s90 = 9.57, 5.003
    for eta, peak in ((0.193, 72.6), (2.365, 83.5)):
        assert A.strength(0.0, s0, s90, eta, peak) == pytest.approx(s0, rel=1e-9)
        assert A.strength(np.pi / 2, s0, s90, eta, peak) == pytest.approx(s90, rel=1e-9)


def test_kappa_diverges_at_the_end_members():
    """The reason the search is restricted, which the text now states."""
    assert abs(A.kappa_from_peak(0.5)) > abs(A.kappa_from_peak(45.0))
    assert abs(A.kappa_from_peak(89.5)) > abs(A.kappa_from_peak(45.0))


@pytest.mark.skipif(not PARAMS.is_file(), reason="envelope parameters absent")
def test_the_search_bounds_do_not_bind():
    lo, hi = A.BETA_BOUNDS
    assert (lo, hi) == (30.0, 89.0), (
        f"the search bounds have moved to {(lo, hi)}; Section 2.7 states them")
    d = pd.read_csv(PARAMS)
    for _, r in d.iterrows():
        for v, name in ((r.beta_peak_deg, "fit"), (r.beta_lo, "lower interval"),
                        (r.beta_hi, "upper interval")):
            assert lo + 5 < v < hi - 4, (
                f"{r.rock}: beta_p {name} is {v}, close to the search bound "
                f"{(lo, hi)}; the text says the restriction does not bind")
        assert 0 < r.eta < A.ETA_BOUNDS[1], f"{r.rock}: eta at its bound"


def test_only_eta_and_beta_are_fitted():
    """The end members are held at their measured values, not optimised."""
    import inspect
    src = inspect.getsource(A.fit)
    assert "sigma0 = float(st.loc[0" in src and "sigma90 = float(st.loc[90" in src, (
        "the end members are no longer read straight from the measurements")
    assert "[ETA_BOUNDS, BETA_BOUNDS]" in src, (
        "the search no longer optimises exactly two parameters")
    assert "floor - strength(" in src, (
        "the admissibility constraint on the curve has gone")
