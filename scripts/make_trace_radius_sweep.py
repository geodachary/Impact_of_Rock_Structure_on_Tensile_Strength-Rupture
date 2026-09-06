"""Sensitivity of the trace-orientation comparison to the fitting radius.

The reported comparison uses the predefined analysis interior, r <= 0.85R
(CORE_FRAC), which is the domain every stress, traction, failure and energy
statistic in this study already uses. This script repeats the identical
comparison over a range of radii so that the choice of window can be shown
to be immaterial rather than argued for. Both traces are clipped by the same
radius at every step, and the null predictor is scored against the same
clipped observations, so model and null are always compared on equal terms.

The radius is not selected here. Nothing downstream reads the best row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import traces as tr        # noqa: E402
from tools import lithology as lith   # noqa: E402

OUT_DIR = REPO / "outputs" / "tables"

FRACTIONS = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
N_SPECIMENS = 14


def _axial(a, b):
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def _primary_traces():
    out = []
    for sid in range(1, N_SPECIMENS + 1):
        o = tr.load_observed_trace(lith.observed_trace_path(sid))
        p = tr.load_predicted_trace(lith.predicted_trace_path(sid))
        ox, oy, *_ = tr.primary_segment(o["x"], o["y"])
        px, py, *_ = tr.primary_segment(p["x"], p["y"])
        out.append((sid, np.asarray(ox, float), np.asarray(oy, float),
                    np.asarray(px, float), np.asarray(py, float)))
    return out


def sweep():
    R = tr.RADIUS_M
    traces = _primary_traces()
    rows = []
    for frac in FRACTIONS:
        err, null, se, kept = [], [], [], []
        for sid, ox, oy, px, py in traces:
            om = np.hypot(ox, oy) <= frac * R
            pm = np.hypot(px, py) <= frac * R
            if om.sum() < 3 or pm.sum() < 3:
                raise RuntimeError(
                    f"specimen {sid} retains too few points at frac={frac}")
            oa, _ = tr.fit_orientation_deg(ox[om], oy[om])
            pa, _ = tr.fit_orientation_deg(px[pm], py[pm])
            err.append(_axial(pa, oa))
            null.append(_axial(90.0, oa))
            se.append(tr.angle_standard_error_deg(ox[om], oy[om]))
            kept.append(float(om.mean()))
        err = np.asarray(err)
        null = np.asarray(null)
        diff = err - null
        boot = stats.bootstrap((diff,), np.mean, n_resamples=10000,
                               random_state=20260807).confidence_interval
        rows.append(dict(
            radius_frac=frac,
            mae_deg=err.mean(),
            rmse_deg=float(np.sqrt((err ** 2).mean())),
            median_deg=float(np.median(err)),
            max_deg=err.max(),
            n_within_5=int((err <= 5.0).sum()),
            n_within_10=int((err <= 10.0).sum()),
            null_mae_deg=null.mean(),
            model_minus_null_deg=diff.mean(),
            ci_low_deg=float(boot.low),
            ci_high_deg=float(boot.high),
            p_paired_t=float(stats.ttest_rel(err, null).pvalue),
            p_wilcoxon=float(stats.wilcoxon(err, null).pvalue),
            mean_fit_se_deg=float(np.nanmean(se)),
            mean_points_retained=float(np.mean(kept)),
        ))
    return pd.DataFrame(rows)


def main():
    df = sweep()
    out = OUT_DIR / "trace_radius_sweep.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"  written: {out.name}")
    with pd.option_context("display.width", 200,
                           "display.max_columns", None):
        print(df.round(3).to_string(index=False))
    worst = df["p_paired_t"].min()
    print(f"\nminimum paired-t p across all radii: {worst:.3f}")
    print("sign of model-minus-null: "
          + ", ".join(f"{f:.2f}:{'-' if d < 0 else '+'}"
                      for f, d in zip(df.radius_frac,
                                      df.model_minus_null_deg)))


if __name__ == "__main__":
    main()
