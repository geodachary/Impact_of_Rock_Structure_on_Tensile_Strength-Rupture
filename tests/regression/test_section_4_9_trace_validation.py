"""Section 4.9's trace-comparison statistics.

Both traces are read over one domain, r <= 0.85R, and one error metric is
reported. The best-segment score and the asymmetric full-versus-core protocol
that this file used to guard were withdrawn from the paper, so the tests that
locked them are gone; what replaces them are the two claims the section now
makes in their place, that the window costs precision and that no window in
0.70R to the full trace separates the model from the null.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tools import traces

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "outputs" / "tables" / "trace_comparison_metrics.csv"

pytestmark = pytest.mark.skipif(not CSV.is_file(), reason="trace metrics absent")


@pytest.fixture(scope="module")
def d():
    return pd.read_csv(CSV)


def _null(o):
    x = np.abs(np.asarray(o, float) - 90.0) % 180.0
    return np.minimum(x, 180.0 - x)


def test_the_headline_error_statistics(d):
    e = d.abs_axial_angular_error_deg.to_numpy(float)
    assert e.mean() == pytest.approx(3.35, abs=0.05)
    assert np.sqrt((e ** 2).mean()) == pytest.approx(4.25, abs=0.05)
    assert np.median(e) == pytest.approx(3.29, abs=0.05)
    assert e.max() == pytest.approx(8.75, abs=0.05), (
        f"maximum error is {e.max():.3f}; the text and the appendix table must "
        "agree on it")
    assert int((e <= 5).sum()) == 11 and int((e <= 10).sum()) == 14


def test_the_null_comparison_is_a_tie(d):
    e = d.abs_axial_angular_error_deg.to_numpy(float)
    n = _null(d.observed_orientation_deg.to_numpy(float))
    diff = e - n
    assert n.mean() == pytest.approx(3.65, abs=0.05)
    # Both traces are read over the same interior, r <= 0.85R. The framework is
    # marginally ahead of the null, by under a third of a degree, which the
    # confidence interval below spans.
    assert diff.mean() == pytest.approx(-0.30, abs=0.02)
    m = len(diff); se = diff.std(ddof=1) / np.sqrt(m)
    lo, hi = diff.mean() - stats.t.ppf(0.975, m - 1) * se, \
        diff.mean() + stats.t.ppf(0.975, m - 1) * se
    assert lo == pytest.approx(-1.35, abs=0.03) and hi == pytest.approx(0.74, abs=0.03)
    assert stats.ttest_rel(e, n).pvalue == pytest.approx(0.54, abs=0.01)
    assert stats.wilcoxon(e, n).pvalue == pytest.approx(1.00, abs=0.01)
    # The advantage is not consistent in sign; the section says seven and seven.
    assert int((diff < 0).sum()) == 7 and int((diff > 0).sum()) == 7
    r = stats.pearsonr(d.observed_orientation_deg, d.predicted_orientation_deg)
    assert r[0] == pytest.approx(0.22, abs=0.01) and r[1] == pytest.approx(0.45, abs=0.01)


def test_no_fitting_radius_separates_the_model_from_the_null(d):
    """The section claims the window is immaterial; this is that claim.

    The comparison is repeated, model and null together, over radii from 0.70R
    to the full trace. Section 4.9 states that the mean absolute error stays
    between 3.3 and 4.7 degrees, that the paired t-test never falls below
    p = 0.42, and that the model-minus-null difference changes sign twice
    across the range. If any radius ever separated the two, the conclusion of
    the section would be a property of the window and not of the data.
    """
    R = traces.RADIUS_M
    maes, ps, signs = [], [], []
    for frac in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        err, null = [], []
        for _, r in d.iterrows():
            o = traces.load_observed_trace(r.observed_source)
            q = traces.load_predicted_trace(r.predicted_source)
            ox, oy, *_ = traces.primary_segment(np.asarray(o["x"], float),
                                                np.asarray(o["y"], float))
            px, py, *_ = traces.primary_segment(np.asarray(q["x"], float),
                                                np.asarray(q["y"], float))
            om = np.hypot(ox, oy) <= frac * R
            pm = np.hypot(px, py) <= frac * R
            assert om.sum() >= 3 and pm.sum() >= 3
            oa = traces.fit_orientation_deg(ox[om], oy[om])[0]
            pa = traces.fit_orientation_deg(px[pm], py[pm])[0]
            err.append(_null([pa - oa + 90.0])[0])
            null.append(_null([oa])[0])
        err = np.asarray(err); null = np.asarray(null)
        maes.append(err.mean())
        ps.append(stats.ttest_rel(err, null).pvalue)
        signs.append(np.sign((err - null).mean()))
    assert min(maes) == pytest.approx(3.35, abs=0.05)
    assert max(maes) == pytest.approx(4.71, abs=0.05)
    assert min(ps) >= 0.42, (
        f"a fitting radius now separates model from null, p = {min(ps):.3f}; "
        "Section 4.9 states the window is immaterial")
    changes = int(np.sum(np.diff(signs) != 0))
    assert changes == 2, (
        f"the model-minus-null difference changes sign {changes} times across "
        "the radius range; the section says twice")


def test_no_predicted_path_wanders(d):
    """The retreat and stall guards keep every path near one diameter."""
    for _, r in d.iterrows():
        p = traces.load_predicted_trace(r.predicted_source)
        L = traces.arclength(np.asarray(p["x"], float),
                             np.asarray(p["y"], float)) * 1e3
        assert L < 1.25 * 51.0, (
            f"sample {int(r['sample'])}: predicted path is {L:.1f} mm in a "
            "51 mm disc; the excursions the guards removed are back")


def test_the_shared_window_costs_precision_by_the_stated_amount(d):
    """Section 4.9 accepts a precision cost to gain a symmetric domain.

    It states that retaining 80% of the digitized points raises the mean fit
    standard error from 0.80 degrees over the full trace to 1.00 degrees. Both
    halves are locked, because the argument of the passage is the trade and not
    either number alone.
    """
    R = traces.RADIUS_M
    se_full, se_core, kept = [], [], []
    for _, r in d.iterrows():
        o = traces.load_observed_trace(r.observed_source)
        ox, oy, *_ = traces.primary_segment(np.asarray(o["x"], float),
                                            np.asarray(o["y"], float))
        m = np.hypot(ox, oy) <= 0.85 * R
        se_full.append(traces.angle_standard_error_deg(ox, oy))
        se_core.append(traces.angle_standard_error_deg(ox[m], oy[m]))
        kept.append(m.mean())
    assert np.mean(kept) == pytest.approx(0.80, abs=0.02)
    assert np.nanmean(se_full) == pytest.approx(0.80, abs=0.03)
    assert np.nanmean(se_core) == pytest.approx(1.00, abs=0.03), (
        f"the shared window now costs {np.nanmean(se_core):.2f} deg of fit "
        "precision; the text quotes 1.00")
