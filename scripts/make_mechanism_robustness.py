#!/usr/bin/env python3
"""Two checks that separate what the framework resolves from what it does not.

The trajectory comparison of Section 4.9 does not beat a loading-parallel null,
which invites the question of why the mechanism sequence should be trusted at
all. These two tables answer it with measurements rather than assertion.

The first asks whether the observed fracture orientations carry any fabric
signal to begin with. They do not: the deviation from the loading axis is
uncorrelated with fabric angle, and its mean is not distinguishable from zero.
The residual the model is being scored against is specimen scatter, so no
formulation could beat the null on this dataset.

The second asks whether the mechanism ordering depends on the weak-plane
proxies, which are assigned rather than measured. It does not. Opening enters
through max(sigma_n, 0), so it is identically unavailable wherever the
foliation is clamped, whatever strength is assigned; only the angle at which
it switches on moves with the proxy.
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

from tools import failure_classification as fc      # noqa: E402
from tools import fabric_tractions as ft            # noqa: E402
from tools import lithology as lith                 # noqa: E402
from tools import strain_partitioning as sp         # noqa: E402
from tools import traces as tr                      # noqa: E402

OUT = REPO / "outputs" / "tables"


def orientation_signal() -> pd.DataFrame:
    rows = tr.compare_all()
    alpha = np.array([r["experimental_angle_deg"] for r in rows], float)
    dev = np.array([((r["observed_orientation_deg"] - 90 + 90) % 180) - 90
                    for r in rows])
    pdev = np.array([((r["predicted_orientation_deg"] - 90 + 90) % 180) - 90
                     for r in rows])
    pear = stats.pearsonr(alpha, dev)
    spear = stats.spearmanr(alpha, dev)
    one = stats.ttest_1samp(dev, 0.0)
    op = stats.pearsonr(dev, pdev)
    return pd.DataFrame([dict(
        n=len(rows),
        observed_mean_deviation_deg=float(dev.mean()),
        observed_sd_deviation_deg=float(dev.std(ddof=1)),
        observed_range_deg=float(dev.max() - dev.min()),
        predicted_range_deg=float(pdev.max() - pdev.min()),
        pearson_r_deviation_vs_alpha=float(pear[0]),
        pearson_p_deviation_vs_alpha=float(pear[1]),
        spearman_rho_deviation_vs_alpha=float(spear.statistic),
        spearman_p_deviation_vs_alpha=float(spear.pvalue),
        t_mean_deviation_zero=float(one.statistic),
        p_mean_deviation_zero=float(one.pvalue),
        pearson_r_observed_vs_predicted_deviation=float(op[0]),
        pearson_p_observed_vs_predicted_deviation=float(op[1]),
    )])


def _fractions(sid, strengths, t_ratio, c_ratio):
    d = np.load(lith.field_cache_path(sid), allow_pickle=True)
    p = strengths[int(sid)]
    res = fc.classify(
        d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=t_ratio * p["T_m"], c_wp=c_ratio * p["c_m"], phi_wp=p["phi_m"],
        T_m=p["T_m"], c_m=p["c_m"], phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=sp.ACTIVATION_FLOOR,
        threshold=sp.THRESHOLD, eta_mix=sp.ETA_MIX, n_theta=sp.N_THETA)
    radius = np.hypot(np.asarray(d["X"], float), np.asarray(d["Y"], float))
    core = d["M"].astype(bool) & (radius <= ft.CORE_FRAC * float(d["R_m"]))
    return fc.class_fractions(res["mode_code"], core)


def mechanism_robustness() -> pd.DataFrame:
    strengths = sp.specimen_strengths()
    out = []
    for t_ratio in (0.20, 0.35, 0.50, 0.70):
        for c_ratio in (0.40, 0.60, 0.80):
            row = dict(T_wp_over_T0=t_ratio, c_wp_over_c=c_ratio,
                       production=(t_ratio == sp.WEAK_T_RATIO
                                   and c_ratio == sp.WEAK_C_RATIO))
            for key, sids in (("gneiss", range(1, 8)), ("schist", range(8, 15))):
                wt, ws, ang = [], [], []
                for sid in sids:
                    f = _fractions(sid, strengths, t_ratio, c_ratio)
                    wt.append(f.get("WT", 0.0))
                    ws.append(f.get("WS", 0.0))
                    ang.append(lith.lithology_of_sample(sid)
                               .angle_for_sample(sid))
                opening = [a for a, v in zip(ang, wt) if v > 1e-6]
                row[f"{key}_WT_onset_deg"] = min(opening) if opening else np.nan
                row[f"{key}_WS_peak_deg"] = ang[int(np.argmax(ws))]
                row[f"{key}_WT_at_or_below_45"] = bool(
                    any(v > 1e-6 for a, v in zip(ang, wt) if a <= 45))
            out.append(row)
    return pd.DataFrame(out)


def trace_roughness() -> pd.DataFrame:
    """How straight the modelled path is compared with the digitized crack.

    The stepper advances through a homogeneous orthotropic field, so nothing
    at grain scale deflects it and the path it produces is close to a straight
    line. Real cracks meet individual augen, grain boundaries and microcracks
    and wander. The comparison is reported because the difference is visible
    in the overlays and is a limit of the homogenized field rather than of the
    trajectory rule.
    """
    rows = []
    for sid in range(1, 15):
        lit = lith.lithology_of_sample(sid)
        out = dict(sample=sid, lithology=lit.display_name,
                   angle_deg=lit.angle_for_sample(sid))
        for label, loader, path in (
                ("observed", tr.load_observed_trace, lith.observed_trace_path(sid)),
                ("predicted", tr.load_predicted_trace, lith.predicted_trace_path(sid))):
            t = loader(path)
            x = np.asarray(t["x"], float)
            y = np.asarray(t["y"], float)
            if label == "observed":
                segs = list(tr.split_segments(x, y))
                x, y = max(segs, key=lambda s: float(
                    np.sum(np.hypot(np.diff(s[0]), np.diff(s[1])))))
            X = np.c_[x - x.mean(), y - y.mean()]
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            resid = X @ np.array([-vt[0, 1], vt[0, 0]])
            out[f"{label}_rms_wander_mm"] = float(resid.std() * 1e3)
            out[f"{label}_max_wander_mm"] = float(np.abs(resid).max() * 1e3)
        rows.append(out)
    df = pd.DataFrame(rows)
    df["model_straighter"] = (df.predicted_rms_wander_mm
                              < df.observed_rms_wander_mm)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a = orientation_signal()
    a.to_csv(OUT / "orientation_signal_test.csv", index=False)
    b = mechanism_robustness()
    b.to_csv(OUT / "mechanism_robustness_sweep.csv", index=False)
    c = trace_roughness()
    c.to_csv(OUT / "trace_roughness.csv", index=False)
    tt = stats.ttest_rel(c.predicted_rms_wander_mm, c.observed_rms_wander_mm)
    print(f"  written: trace_roughness.csv")
    print(f"  model straighter in {int(c.model_straighter.sum())}/{len(c)} "
          f"specimens, paired p = {tt.pvalue:.4f}")
    print("  written: orientation_signal_test.csv, mechanism_robustness_sweep.csv")
    print(f"  deviation vs alpha: r = {a.pearson_r_deviation_vs_alpha[0]:+.3f}, "
          f"p = {a.pearson_p_deviation_vs_alpha[0]:.3f}")
    print(f"  ordering preserved in all {len(b)} proxy combinations: "
          f"{not b.gneiss_WT_at_or_below_45.any() and not b.schist_WT_at_or_below_45.any()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
