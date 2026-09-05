"""Best-segment matching, and the trap that makes it easy to misuse.

The observed traces are multi-crack: every one of the fourteen specimens
carries between two and four connected segments, while the stepper produces a
single path. Scoring against the longest observed segment alone therefore
discards the rest, on average 44% of the digitized length, and the longest
segment is a minority of the trace in six specimens. Matching the prediction to
whichever crack it fits best is the natural generalisation and is what a reader
applying this to their own multi-crack dataset would want.

It is also the easiest metric in this file to fool. Giving the model several
candidates and the null exactly one manufactures skill out of the segment
count alone: on this dataset it turns a dead heat into an apparent 0.79 degree
win for the framework, which is the largest margin anywhere in this
comparison and is entirely an artefact of the mismatch. Scored equally the two
are separated by 0.015 degrees. These tests pin the fair comparison and the
size of the trap.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import traces as tr

pytestmark = pytest.mark.skipif(
    not tr.observed_available() if hasattr(tr, "observed_available") else False,
    reason="traces not available")


def test_every_specimen_is_multi_crack():
    """The premise of the metric: one predicted path, several observed cracks."""
    counts = [len(tr.observed_segment_orientations(sid, min_frac=0.0))
              for sid in range(1, 15)]
    assert all(c >= 2 for c in counts), (
        f"a specimen digitized as a single crack: {counts}. The best-segment "
        "metric is only needed while the observations are multi-crack.")


def test_the_length_guard_drops_fragments_but_keeps_cracks():
    for sid in range(1, 15):
        allsegs = tr.observed_segment_orientations(sid, min_frac=0.0)
        kept = tr.observed_segment_orientations(sid)
        assert 1 <= len(kept) <= len(allsegs)
        total = sum(L for _, L in allsegs)
        for _, L in kept:
            assert L / total >= tr.MIN_SEGMENT_FRAC - 1e-9 or len(kept) == 1


def test_the_single_sided_error_is_not_a_public_entry_point():
    """The mistake is prevented by the API, not only warned about.

    ``best_segment_error_pair`` returns the model and null halves together and
    there is no public way to get one without the other, so scoring the two
    sides differently takes deliberate effort rather than an oversight.
    """
    assert not hasattr(tr, "best_segment_error_deg"), (
        "the single-sided best-segment error is public again; quoting it "
        "without a null scored the same way is the error this module exists "
        "to prevent"
    )
    assert hasattr(tr, "best_segment_error_pair")

    segs = tr.observed_segment_orientations(9)
    model, null = tr.best_segment_error_pair(89.0, segs)
    assert np.isfinite(model) and np.isfinite(null)


def test_model_and_null_must_be_scored_the_same_way():
    """The comparison the manuscript reports, and the one it must not report."""
    stats = tr.best_segment_statistics()
    rows = [r for r in tr.compare_all() if r.get("status") == "computed"]

    # Fair: both sides get every candidate crack. Neither leads by anything
    # approaching the 0.2 to 2.6 degree digitization uncertainty.
    assert abs(stats["model_minus_null_deg"]) < 0.5, (
        f"model and null have separated by "
        f"{stats['model_minus_null_deg']:+.2f} deg under equal scoring; the "
        "withdrawal of orientation-predictive skill would need revisiting"
    )

    # Unfair: model gets every crack, null is pinned to the longest segment.
    # This does not make the model win, but it does close a 0.90 degree gap to
    # nothing, which is enough to be reported as agreement by anyone who does
    # not notice the two sides were scored differently.
    unfair_null = float(np.mean([
        tr.axial_angular_error_deg(90.0, r["observed_orientation_deg"])
        for r in rows]))
    unfair_gap = stats["model_mae_deg"] - unfair_null
    fair_gap = stats["model_minus_null_deg"]
    assert unfair_gap < -0.5, (
        f"the mismatched comparison no longer flatters the model "
        f"({unfair_gap:+.3f}); the warning in this docstring needs updating")
    assert abs(fair_gap) < abs(unfair_gap) / 2, (
        "equal scoring should leave the two far closer than mismatched "
        f"scoring: fair {fair_gap:+.3f} against unfair {unfair_gap:+.3f}")


def test_best_segment_does_not_change_the_conclusion():
    """Both metrics must agree that neither predictor leads."""
    rows = tr.compare_all()
    primary_gap = (tr.aggregate_statistics(rows)["mae_deg"]
                   - tr.null_model_statistics(rows)["mae_deg"])
    best_gap = tr.best_segment_statistics(rows)["model_minus_null_deg"]
    assert abs(primary_gap) < 0.5 and abs(best_gap) < 0.5, (
        f"the two metrics no longer agree that the comparison is a tie: "
        f"primary {primary_gap:+.2f}, best-segment {best_gap:+.2f}")


@pytest.mark.parametrize("floor", [0.0, 0.10, 0.15, 0.20, 0.30])
def test_the_conclusion_is_insensitive_to_the_length_guard(floor):
    s = tr.best_segment_statistics(min_frac=floor)
    assert abs(s["model_minus_null_deg"]) < 0.5, (
        f"at min_frac={floor} the gap reached {s['model_minus_null_deg']:+.2f}; "
        "the reported conclusion would then depend on the guard value")


def test_the_null_comparison_is_a_statistical_tie_not_a_narrow_win():
    """The model leads the null by 0.05 deg. That is not skill, and the text
    must not be allowed to drift into saying it is.

    A bare mean difference invites the reading that the model is slightly
    better. Three things forbid it, and Section 4.9 and the response letter now
    state all three, so they are pinned here:

    * the paired 95% confidence interval contains zero by a wide margin
    * neither a paired t-test nor a Wilcoxon signed-rank test is significant
    * the advantage is not consistent in sign, the model being closer on six
      of the fourteen specimens and further on eight

    If a future change produces a genuine separation this test fails, which is
    the point: that would be a result to argue for deliberately, not to inherit.
    """
    from scipy import stats

    rows = [r for r in tr.compare_all() if r.get("status") == "computed"]
    obs = np.array([r["observed_orientation_deg"] for r in rows])
    e_model = np.array([r["abs_axial_angular_error_deg"] for r in rows])
    e_null = np.array([tr.axial_angular_error_deg(90.0, o) for o in obs])
    d = e_model - e_null

    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    lo, hi = d.mean() - half, d.mean() + half
    assert lo < 0 < hi, (
        f"the paired 95% interval [{lo:+.3f}, {hi:+.3f}] no longer contains "
        "zero; one predictor now genuinely beats the other and Section 4.9 "
        "needs rewriting rather than adjusting")

    assert stats.ttest_rel(e_model, e_null).pvalue > 0.05
    assert stats.wilcoxon(e_model, e_null).pvalue > 0.05

    better = int((d < 0).sum())
    assert 4 <= better <= 10, (
        f"the model is closer on {better} of {n} specimens. Away from an even "
        "split the advantage has become consistent in sign, which is what "
        "would make the mean difference meaningful")
