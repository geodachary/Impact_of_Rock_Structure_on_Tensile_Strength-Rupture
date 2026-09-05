"""Section 4.9's trace-comparison statistics.

Most are exact. Four were not: the maximum error disagreed with its own
appendix table (6.9 against 6.847); the 0.85R restriction was called decisive
on the schist at 15 degrees, which it moves least, describing excursions the
retreat and stall guards had already removed; the central window was said to
discard two thirds of the baseline when it halves it; and the mismatched-scoring
gap is 0.79 degrees, not 0.90.
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
    assert e.mean() == pytest.approx(3.3, abs=0.05)
    assert np.sqrt((e ** 2).mean()) == pytest.approx(3.7, abs=0.05)
    assert np.median(e) == pytest.approx(3.4, abs=0.05)
    assert e.max() == pytest.approx(6.8, abs=0.05), (
        f"maximum error is {e.max():.3f}; the text and the appendix table must "
        "agree on it")
    assert int((e <= 5).sum()) == 11 and int((e <= 10).sum()) == 14


def test_the_null_comparison_is_a_tie(d):
    e = d.abs_axial_angular_error_deg.to_numpy(float)
    n = _null(d.observed_orientation_deg.to_numpy(float))
    diff = e - n
    assert n.mean() == pytest.approx(3.4, abs=0.05)
    assert diff.mean() == pytest.approx(-0.05, abs=0.01)
    m = len(diff); se = diff.std(ddof=1) / np.sqrt(m)
    lo, hi = diff.mean() - stats.t.ppf(0.975, m - 1) * se, \
        diff.mean() + stats.t.ppf(0.975, m - 1) * se
    assert lo == pytest.approx(-0.75, abs=0.02) and hi == pytest.approx(0.65, abs=0.02)
    assert stats.ttest_rel(e, n).pvalue == pytest.approx(0.88, abs=0.01)
    assert stats.wilcoxon(e, n).pvalue == pytest.approx(0.81, abs=0.01)
    assert int((diff < 0).sum()) == 6 and int((diff > 0).sum()) == 8
    r = stats.pearsonr(d.observed_orientation_deg, d.predicted_orientation_deg)
    assert r[0] == pytest.approx(0.36, abs=0.01) and r[1] == pytest.approx(0.21, abs=0.01)


def test_the_core_restriction_moves_one_specimen_and_it_is_the_gneiss_at_thirty(d):
    R = 0.0255
    shift, tag = [], []
    for _, r in d.iterrows():
        p = traces.load_predicted_trace(r.predicted_source)
        sx, sy, *_ = traces.primary_segment(np.asarray(p["x"], float),
                                            np.asarray(p["y"], float))
        full = traces.fit_orientation_deg(sx, sy)[0]
        m = np.hypot(sx, sy) <= 0.85 * R
        core = traces.fit_orientation_deg(sx[m], sy[m])[0]
        x = abs(full - core) % 180.0
        shift.append(min(x, 180.0 - x))
        tag.append((r.lithology, int(r.experimental_angle_deg)))
    shift = np.asarray(shift)
    assert int((shift < 3.0).sum()) == 13, (
        f"{(shift < 3).sum()} of 14 shift by under 3 degrees")
    worst = tag[int(np.argmax(shift))]
    assert worst == ("Augen gneiss", 30), (
        f"the exception is now {worst}; the text names the gneiss at 30")
    assert shift.max() == pytest.approx(3.2, abs=0.1)


def test_no_predicted_path_wanders(d):
    """The retreat and stall guards keep every path near one diameter."""
    for _, r in d.iterrows():
        p = traces.load_predicted_trace(r.predicted_source)
        L = traces.arclength(np.asarray(p["x"], float),
                             np.asarray(p["y"], float)) * 1e3
        assert L < 1.25 * 51.0, (
            f"sample {int(r['sample'])}: predicted path is {L:.1f} mm in a "
            "51 mm disc; the excursions the guards removed are back")


def test_the_window_halves_both_points_and_baseline(d):
    def span(x, y):
        q = np.column_stack([x, y]); q = q - q.mean(0)
        w, v = np.linalg.eigh(np.cov(q, rowvar=False))
        return (q @ v[:, -1]).std()
    pf, bf = [], []
    for _, r in d.iterrows():
        o = traces.load_observed_trace(r.observed_source)
        px, py, *_ = traces.primary_segment(np.asarray(o["x"], float),
                                            np.asarray(o["y"], float))
        m = traces.central_mask(px, py)
        if int(m.sum()) >= 3:
            pf.append(m.sum() / len(px)); bf.append(span(px[m], py[m]) / span(px, py))
    assert np.mean(pf) == pytest.approx(0.5, abs=0.08)
    assert np.mean(bf) == pytest.approx(0.5, abs=0.08), (
        f"the window retains {np.mean(bf):.2f} of the baseline; the text says "
        "it discards about half")


def test_the_best_segment_scores_tie_at_every_threshold(d):
    for f in (0.0, 0.10, 0.15, 0.20, 0.30):
        s = traces.best_segment_statistics(min_frac=f)
        assert abs(s["model_minus_null_deg"]) < 0.15, (
            f"at min_frac {f} the model leads the null by "
            f"{-s['model_minus_null_deg']:.3f} deg; the text says neither "
            "metric separates them for thresholds from 0 to 30%")
    s = traces.best_segment_statistics(min_frac=0.15)
    assert round(s["model_mae_deg"], 1) == 2.6 and round(s["null_mae_deg"], 1) == 2.6
    mismatched = _null(d.observed_orientation_deg.to_numpy(float)).mean() - s["model_mae_deg"]
    assert mismatched == pytest.approx(0.79, abs=0.02), (
        f"the mismatched-scoring gap is {mismatched:.3f} deg")
