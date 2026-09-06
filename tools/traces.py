"""Fracture-trace loading, primary-segment selection, orientation fitting, comparison.

Consolidates the digitized-trace loading conventions audited from
``crack_data_plot.py`` with the orientation-extraction rule, so that observed
and predicted traces are treated identically. See
the module docstring below for the audited conventions.

Coordinate handling
-------------------
Both datasets are natively in the global frame of :mod:`tools.conventions`
(metres, disc-centred, ``+y`` loading), so the transform between them is the
identity. The only operation applied to observed data is the radial projection
of points digitized marginally outside the specimen boundary, which reproduces
``crack_data_plot.py``'s own behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .conventions import (MAX_TRACE_GAP_M, RADIUS_M, DEFAULT_CENTRAL_FRAC,
                          axial_angular_error_deg, signed_axial_difference_deg)
from . import lithology as lith


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_observed_trace(path):
    """Load a digitized laboratory trace.

    Headerless two-column CSV, ``x`` and ``y`` in metres, disc centre at the
    origin. Non-numeric rows are coerced and dropped. Points outside ``r = R``
    are projected radially onto the boundary.

    Returns
    -------
    dict with ``x``, ``y`` (used arrays), ``x_src``, ``y_src`` (untransformed),
    and provenance counters.
    """
    path = lith.resolve_repo_path(path)
    df = pd.read_csv(path, header=None, names=["x", "y"])
    for c in ("x", "y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_raw = len(df)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    x_src = df["x"].to_numpy(float)
    y_src = df["y"].to_numpy(float)
    x, y = x_src.copy(), y_src.copy()
    r = np.hypot(x, y)
    n_outside = int(np.sum(r > RADIUS_M))
    outside = r > RADIUS_M
    if np.any(outside):
        scale = RADIUS_M / r[outside]
        x[outside] *= scale
        y[outside] *= scale
    if len(x) and np.max(np.hypot(x_src, y_src)) > 1.0:
        raise ValueError(f"{path.name}: coordinates look like millimetres, not metres")
    return dict(x=x, y=y, x_src=x_src, y_src=y_src, n_rows_raw=n_raw,
                n_points=len(x), n_dropped=n_raw - len(x), n_outside_disc=n_outside,
                source=str(path))


def load_predicted_trace(path):
    """Load a DDM predicted trace (header ``order,x_m,y_m``, metres)."""
    path = lith.resolve_repo_path(path)
    df = pd.read_csv(path).dropna(subset=["x_m", "y_m"]).reset_index(drop=True)
    if "order" in df.columns:
        df = df.sort_values("order").reset_index(drop=True)
    x = df["x_m"].to_numpy(float)
    y = df["y_m"].to_numpy(float)
    return dict(x=x, y=y, x_src=x.copy(), y_src=y.copy(), n_rows_raw=len(df),
                n_points=len(x), n_dropped=0,
                n_outside_disc=int(np.sum(np.hypot(x, y) > RADIUS_M)),
                source=str(path))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def split_segments(x, y, max_gap=MAX_TRACE_GAP_M):
    """Split a point sequence wherever consecutive spacing exceeds ``max_gap``."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return [(x, y)]
    d = np.hypot(np.diff(x), np.diff(y))
    brk = np.where(d > max_gap)[0] + 1
    return [(x[i], y[i]) for i in np.split(np.arange(len(x)), brk) if len(i)]


def arclength(x, y):
    """Cumulative polyline length."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(x) > 1 else 0.0


def primary_segment(x, y, max_gap=MAX_TRACE_GAP_M):
    """Objective primary-fracture rule: the connected segment of greatest arc length.

    Returns ``(x, y, n_segments, primary_length, total_length)``.
    """
    segs = split_segments(x, y, max_gap)
    lens = [arclength(sx, sy) for sx, sy in segs]
    k = int(np.argmax(lens))
    return segs[k][0], segs[k][1], len(segs), lens[k], float(np.sum(lens))


def central_mask(x, y, frac=DEFAULT_CENTRAL_FRAC):
    """Points within ``frac * R`` of the disc centre."""
    return np.hypot(np.asarray(x, float), np.asarray(y, float)) <= frac * RADIUS_M


def fit_orientation_deg(x, y):
    """Total-least-squares axial orientation, degrees CCW from +x, in ``[0, 180)``.

    Principal axis of the mean-centred covariance — appropriate because both
    coordinates carry error. Also returns a collinearity measure
    ``1 - lambda_min/lambda_max`` (1 = perfectly line-like, 0 = isotropic).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan, np.nan
    pts = np.column_stack([x, y])
    pts = pts - pts.mean(axis=0)
    if not np.all(np.isfinite(pts)):
        return np.nan, np.nan
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return np.nan, np.nan
    w, v = np.linalg.eigh(cov)
    major = v[:, int(np.argmax(w))]
    ang = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    lmax, lmin = float(np.max(w)), float(np.min(w))
    return ang, (1.0 - lmin / lmax if lmax > 0 else np.nan)


def angle_standard_error_deg(x, y):
    """Standard error of the total-least-squares orientation, in degrees.

    For points scattered about a line, the fitted angle has

        sigma_theta = sigma_perp / (sqrt(n) * s_parallel)

    with ``sigma_perp`` the RMS perpendicular residual (here, digitization
    scatter), ``s_parallel`` the RMS spread along the line and ``n`` the point
    count. Precision therefore depends on the *baseline* as much as on the
    number of points: halving the span costs as much as quartering the sample.

    This is what makes a windowed fit dangerous on a sparsely digitized trace.
    Restricting to the central disc keeps roughly half the points and half
    the baseline, so it can only inflate the variance. The measure is
    reported alongside every orientation as :data:`ANGLE_SE_ADVISORY_DEG`.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.inf
    pts = np.column_stack([x, y])
    pts = pts - pts.mean(axis=0)
    if not np.all(np.isfinite(pts)):
        return np.inf
    w, v = np.linalg.eigh(np.cov(pts, rowvar=False))
    along = pts @ v[:, int(np.argmax(w))]
    perp = pts @ v[:, int(np.argmin(w))]
    s_par = float(np.sqrt(np.mean(along ** 2)))
    if s_par <= 0:
        return np.inf
    s_perp = float(np.sqrt(np.mean(perp ** 2)))
    return float(np.degrees(s_perp / (np.sqrt(len(x)) * s_par)))


#: Standard error above which a fitted orientation should not be read as a
#: measurement. Reported per specimen, not used to select an estimator: the
#: orientation is fitted over the full primary segment for every specimen, so
#: there is no per-specimen decision for a threshold to make. Kept because a
#: reader comparing a 2 deg error against a 5 deg criterion is entitled to know
#: which orientations the digitization actually determines.
ANGLE_SE_ADVISORY_DEG = 2.0


def orientation_central(x, y, frac=DEFAULT_CENTRAL_FRAC, min_pts=3):
    """Fit orientation over the central disc region, falling back to the full segment.

    Returns ``(angle_deg, collinearity, n_central_points, fallback_used)``.
    """
    m = central_mask(x, y, frac)
    if int(np.sum(m)) >= min_pts:
        a, c = fit_orientation_deg(np.asarray(x)[m], np.asarray(y)[m])
        return a, c, int(np.sum(m)), False
    a, c = fit_orientation_deg(x, y)
    return a, c, int(np.sum(m)), True



#: The predicted path is fitted inside this fraction of the disc radius.
#:
#: Every field diagnostic in this work is taken over ``fabric_tractions.CORE_FRAC``
#: = 0.85 R, "the disc interior away from the platen contacts", because the
#: contact zone is where the orthotropic solution is least trustworthy. The
#: stepper was never given the same restriction, and it walks wherever its
#: guidance field points. On thirteen specimens that costs nothing: 15 to 21% of
#: each path lies outside 0.85 R and the fitted orientation moves by under three
#: degrees. On the psammitic schist at 15 degrees it is decisive. There the
#: foliation is near-horizontal, the compliance anisotropy is the largest in the
#: set and the spacing the finest, so beneath the platens, where the tensile
#: drive is weak, the fabric guidance wins and the tip runs 12.5 mm sideways
#: along the foliation before returning. Two such excursions make the path
#: 99.5 mm long inside a 51 mm disc, put 47% of it in the contact zone, and pull
#: the fitted orientation to 76.8 degrees against 85 to 92 for every other
#: specimen.
#:
#: Restricting the fit to the same interior the rest of the paper uses is not a
#: repair of that path, which is still what the stepper produced; it is a
#: statement that the model's output is read only where the model is posed.
#: The observed traces are not restricted: they are data, and Section 4.9 sets
#: out why discarding digitized points inflates the error.
PREDICTED_CORE_FRAC = 0.85


def core_mask_radial(x, y, R, frac=PREDICTED_CORE_FRAC):
    """Points inside ``frac * R`` of the disc centre."""
    return np.hypot(np.asarray(x, float), np.asarray(y, float)) <= float(frac) * float(R)


def orientation_pair(ox, oy, px, py, frac=DEFAULT_CENTRAL_FRAC, min_pts=3,
                     domain="full_primary_segment"):
    """Fit both traces of one specimen over the whole primary segment.

    Orientation is taken over the entire primary segment, identically for the
    observed and the predicted trace. ``frac`` is accepted so a caller can
    still ask for the windowed variant as a sensitivity check, but it does not
    select the domain.

    Why not the central window
    --------------------------
    Restricting the fit to the central half of the radius was meant to keep it
    clear of the platen contacts, where a trace curves toward the loading
    points. It fails in two different ways, and between them they accounted for
    both of the large orientation errors previously reported.

    * **Imprecision.** The window discards roughly half the points and half
      the baseline, and the standard error of a fitted angle goes as
      ``sigma_perp / (sqrt(n) * s_parallel)``. On a sparsely digitized trace
      that is ruinous: specimen 12 keeps 8 of 15 points inside the window and
      its orientation carries a standard error of 4.0 deg.
    * **Domain-dependent bias.** Where a trace is curved, the middle third has
      a genuinely different tangent from the chord of the whole segment, and
      the fit to that sub-piece can be *tight* while pointing somewhere else.
      Specimen 5 is the case in point: 102.63 deg windowed with a comfortable
      standard error of 1.93 deg, against 93.68 deg over the full segment. A
      standard error measures scatter about the fitted line and cannot see this
      at all, which is why a precision test alone does not catch it.

    Fitting the full segment removes both. It uses all the digitized evidence,
    it makes the estimate independent of where the window is drawn, and it is
    one uniform rule for all fourteen specimens rather than a per-specimen
    choice. The window does remove a small systematic curvature from the
    *predicted* paths, which bend as the stepper drives the tip to the boundary
    (mean full-minus-central shift -1.63 deg, t = -2.92, p = 0.012), but that
    is far smaller than the 9 to 13 deg of domain-dependence it imports on the
    observed side, and it is common to both traces here because the same rule
    is applied to each.

    Returns ``(o_ang, o_col, p_ang, p_col, domain, o_se, p_se)``.
    """
    if domain == "central_window":
        # Retained only so the reported window-sensitivity check is
        # reproducible. Not the estimator; see above for why.
        om, pm = central_mask(ox, oy, frac), central_mask(px, py, frac)
        if int(np.sum(om)) >= min_pts and int(np.sum(pm)) >= min_pts:
            ox, oy = np.asarray(ox)[om], np.asarray(oy)[om]
            px, py = np.asarray(px)[pm], np.asarray(py)[pm]
        else:
            domain = "full_primary_segment"

    o_ang, o_col = fit_orientation_deg(ox, oy)
    p_ang, p_col = fit_orientation_deg(px, py)
    o_se = angle_standard_error_deg(ox, oy)
    p_se = angle_standard_error_deg(px, py)
    return o_ang, o_col, p_ang, p_col, domain, o_se, p_se


def bootstrap_orientation_sd(x, y, frac=DEFAULT_CENTRAL_FRAC, n=2000, seed=20260807,
                             use_central=True):
    """Bootstrap SD of the fitted orientation: digitization and fit scatter.

    This is **not** a replicate standard deviation: one specimen was tested per
    (lithology, fabric angle) pair, so between-specimen scatter is not
    measurable from this dataset.

    ``use_central`` must match the domain the orientation itself was fitted on
    (see :func:`orientation_pair`). Quoting a scatter measured inside the
    window beside an angle measured over the whole segment would describe a fit
    that was never performed.
    """
    m = central_mask(x, y, frac)
    xs, ys = (np.asarray(x)[m], np.asarray(y)[m]) if (
        use_central and int(np.sum(m)) >= 3) else (np.asarray(x), np.asarray(y))
    if len(xs) < 3:
        return np.nan
    base, _ = fit_orientation_deg(xs, ys)
    if not np.isfinite(base):
        return np.nan
    rng = np.random.default_rng(seed)
    devs = []
    for _ in range(n):
        idx = rng.integers(0, len(xs), len(xs))
        a, _c = fit_orientation_deg(xs[idx], ys[idx])
        if np.isfinite(a):
            devs.append(signed_axial_difference_deg(a, base))
    return float(np.std(devs, ddof=1)) if len(devs) > 2 else np.nan


def resample_polyline(x, y, n=400):
    """Arc-length resampling so shape metrics are not biased by point density."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return x, y
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    if s[-1] <= 0:
        return x, y
    t = np.linspace(0.0, s[-1], n)
    return np.interp(t, s, x), np.interp(t, s, y)


def shape_metrics(ax, ay, bx, by, n=400):
    """Supplementary shape agreement: symmetric mean nearest-neighbour and Hausdorff."""
    if len(ax) < 2 or len(bx) < 2:
        return dict(symmetric_mean_nn_m=np.nan, hausdorff_m=np.nan)
    arx, ary = resample_polyline(ax, ay, n)
    brx, bry = resample_polyline(bx, by, n)
    d = np.hypot(arx[:, None] - brx[None, :], ary[:, None] - bry[None, :])
    d_ab, d_ba = d.min(axis=1), d.min(axis=0)
    return dict(symmetric_mean_nn_m=float(0.5 * (d_ab.mean() + d_ba.mean())),
                hausdorff_m=float(max(d_ab.max(), d_ba.max())))


# ---------------------------------------------------------------------------
# Specimen-level comparison
# ---------------------------------------------------------------------------

def bootstrap_orientation_sd_points(x, y, n=2000, seed=20260807):
    """Bootstrap SD of the fitted orientation for an already-clipped trace.

    Resamples the digitized points of the trace as it was actually fitted, so
    the quoted scatter describes the fit that was performed. It measures
    digitization and fit scatter only. It is **not** a replicate standard
    deviation: one specimen was tested per lithology-angle pair, so
    between-specimen variability is not measurable from this dataset.

    The points along one crack are spatially correlated, so an independent
    resample of individual points understates the spread somewhat. A block
    bootstrap along arc length would be stricter. It is not used here because
    the quantity is reported as a descriptive fit uncertainty and no claim in
    the paper turns on its exact width.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    base, _ = fit_orientation_deg(x, y)
    if not np.isfinite(base):
        return np.nan
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(n)):
        idx = rng.integers(0, len(x), len(x))
        a, _c = fit_orientation_deg(x[idx], y[idx])
        if np.isfinite(a):
            out.append(((a - base + 90.0) % 180.0) - 90.0)
    return float(np.std(out, ddof=1)) if len(out) > 2 else np.nan


def compare_specimen(sample_id, frac=DEFAULT_CENTRAL_FRAC, root=None, seed=20260807,
                     domain="full_primary_segment"):
    """Full observed-vs-predicted orientation comparison for one specimen."""
    lithology = lith.lithology_of_sample(sample_id)
    angle = lithology.angle_for_sample(sample_id)
    obs_p = lith.observed_trace_path(sample_id, root)
    pred_p = lith.predicted_trace_path(sample_id, root)

    base = dict(sample=int(sample_id), lithology=lithology.display_name,
                lithology_key=lithology.key, experimental_angle_deg=angle,
                observed_source=lith.repo_relative(obs_p),
                predicted_source=lith.repo_relative(pred_p))

    if not obs_p.exists() or not pred_p.exists():
        base.update(status="blocked",
                    reason=f"missing file(s): observed={obs_p.exists()}, "
                           f"predicted={pred_p.exists()}")
        return base

    o = load_observed_trace(obs_p)
    p = load_predicted_trace(pred_p)
    oxp, oyp, o_nseg, o_len, o_tot = primary_segment(o["x"], o["y"])
    pxp, pyp, p_nseg, p_len, p_tot = primary_segment(p["x"], p["y"])

    # Both traces are clipped to the same predefined interior before either is
    # fitted. r <= 0.85 R is the domain the stress, failure and energy
    # statistics already use; it is not chosen to minimise angular error. The
    # primary observed segment is identified above from the complete digitized
    # trace, so clipping changes the fitting domain and never crack identity.
    # Previously only the predicted trace was clipped, which compared the two
    # over different domains.
    oxp_f, oyp_f, pxp_f, pyp_f = (np.asarray(oxp, float), np.asarray(oyp, float),
                                  np.asarray(pxp, float), np.asarray(pyp, float))
    o_core = core_mask_radial(oxp_f, oyp_f, RADIUS_M)
    p_core = core_mask_radial(pxp_f, pyp_f, RADIUS_M)
    predicted_frac_outside_core = float(1.0 - p_core.mean())
    observed_frac_outside_core = float(1.0 - o_core.mean())

    if int(o_core.sum()) >= 3 and int(p_core.sum()) >= 3:
        oxp_d, oyp_d = oxp_f[o_core], oyp_f[o_core]
        pxp_d, pyp_d = pxp_f[p_core], pyp_f[p_core]
        fit_domain = "interior_0.85R"
    else:
        # Too few points survive the clip to fit an axis; fall back to the
        # whole segment for both, and say so, rather than mixing domains.
        oxp_d, oyp_d, pxp_d, pyp_d = oxp_f, oyp_f, pxp_f, pyp_f
        fit_domain = "full_primary_segment"

    o_ang, o_col = fit_orientation_deg(oxp_d, oyp_d)
    p_ang, p_col = fit_orientation_deg(pxp_d, pyp_d)
    o_se = angle_standard_error_deg(oxp_d, oyp_d)
    p_se = angle_standard_error_deg(pxp_d, pyp_d)
    observed_fit_domain = predicted_fit_domain = domain = fit_domain
    assert observed_fit_domain == predicted_fit_domain, (
        "observed and predicted orientations must share one fitting domain")

    # Domain-sensitivity check: the same estimator over both full traces.
    o_ang_full, _ = fit_orientation_deg(oxp_f, oyp_f)
    p_ang_full, _ = fit_orientation_deg(pxp_f, pyp_f)

    o_sd = bootstrap_orientation_sd_points(oxp_d, oyp_d, seed=seed)

    om = central_mask(oxp, oyp, frac)
    pm = central_mask(pxp, pyp, frac)
    o_ncen, p_ncen = int(np.sum(om)), int(np.sum(pm))
    sm = shape_metrics(np.asarray(oxp)[om], np.asarray(oyp)[om],
                       np.asarray(pxp)[pm], np.asarray(pyp)[pm])

    base.update(
        observed_orientation_deg=o_ang, observed_bootstrap_sd_deg=o_sd,
        predicted_orientation_deg=p_ang,
        abs_axial_angular_error_deg=float(axial_angular_error_deg(p_ang, o_ang)),
        signed_wrapped_diff_deg=float(signed_axial_difference_deg(p_ang, o_ang)),
        observed_n_points=o["n_points"], predicted_n_points=p["n_points"],
        observed_n_segments=o_nseg, predicted_n_segments=p_nseg,
        observed_primary_seg_length_m=o_len,
        observed_primary_frac_of_total_length=(o_len / o_tot) if o_tot else np.nan,
        predicted_primary_seg_length_m=p_len,
        observed_n_points_central=o_ncen, predicted_n_points_central=p_ncen,
        observed_fit_collinearity=o_col, predicted_fit_collinearity=p_col,
        # Which domain the pair was fitted on, the precision that decided it,
        # and the tolerance in force. Reported for every specimen so a reader
        # can see which orientations the data actually determines rather than
        # only which ones triggered the fallback.
        orientation_fit_domain=domain,
        observed_fit_domain=observed_fit_domain,
        predicted_fit_domain=predicted_fit_domain,
        observed_frac_outside_core=observed_frac_outside_core,
        # Domain-sensitivity check, both traces over their full extent.
        observed_orientation_full_deg=o_ang_full,
        predicted_orientation_full_deg=p_ang_full,
        abs_axial_angular_error_full_deg=float(
            axial_angular_error_deg(p_ang_full, o_ang_full)),
        observed_orientation_se_deg=(None if not np.isfinite(o_se) else o_se),
        predicted_orientation_se_deg=(None if not np.isfinite(p_se) else p_se),
        orientation_se_advisory_deg=ANGLE_SE_ADVISORY_DEG,
        observed_central_fallback_used=(domain == "full"),
        predicted_central_fallback_used=(domain == "full"),
        observed_n_outside_disc=o["n_outside_disc"],
        predicted_frac_outside_core=predicted_frac_outside_core,
        predicted_core_frac=PREDICTED_CORE_FRAC,
        supp_symmetric_mean_nn_distance_m=sm["symmetric_mean_nn_m"],
        supp_hausdorff_distance_m=sm["hausdorff_m"],
        central_fraction_used=frac, status="computed", reason="",
        _obs_xy=(oxp_d, oyp_d), _pred_xy=(pxp_d, pyp_d),
        # Full extent of the same two traces, for context in the overlay.
        _obs_primary_full=(oxp_f, oyp_f), _pred_full=(pxp_f, pyp_f),
        _obs_full=(o["x"], o["y"]),
    )
    return base


def compare_all(frac=DEFAULT_CENTRAL_FRAC, root=None, seed=20260807,
                domain="full_primary_segment"):
    """Comparison for all 14 specimens, in specimen order."""
    return [compare_specimen(r["sample"], frac, root, seed, domain)
            for r in lith.pairing_table()]


def aggregate_statistics(rows):
    """MAE, RMSE, median, max and tolerance counts from computed rows only."""
    e = np.array([r["abs_axial_angular_error_deg"] for r in rows
                  if r.get("status") == "computed"], float)
    e = e[np.isfinite(e)]
    if not len(e):
        return dict(n=0, status="not_computable",
                    reason="no specimen produced a finite angular error")
    return dict(n=int(len(e)), mae_deg=float(e.mean()),
                rmse_deg=float(np.sqrt((e ** 2).mean())),
                median_deg=float(np.median(e)), max_deg=float(e.max()),
                n_within_5=int((e <= 5).sum()), pct_within_5=100 * float((e <= 5).mean()),
                n_within_10=int((e <= 10).sum()), pct_within_10=100 * float((e <= 10).mean()),
                status="computed", reason="")


def null_model_statistics(rows, null_orientation_deg=None):
    """Error of a trivial predictor, for assessing whether the model has skill.

    ``None`` uses the loading-axis orientation. Brazilian-disc fractures are
    loading-subparallel almost by construction, so error statistics are only
    evidence of skill if they beat this null.
    """
    from .conventions import LOADING_AXIS_DEG
    null = LOADING_AXIS_DEG if null_orientation_deg is None else float(null_orientation_deg)
    obs = np.array([r["observed_orientation_deg"] for r in rows
                    if r.get("status") == "computed"], float)
    e = axial_angular_error_deg(np.full_like(obs, null), obs)
    return dict(n=int(len(e)), null_orientation_deg=null, mae_deg=float(e.mean()),
                rmse_deg=float(np.sqrt((e ** 2).mean())),
                n_within_5=int((e <= 5).sum()), n_within_10=int((e <= 10).sum()))

#: A digitized segment shorter than this fraction of the total trace length is
#: a fragment rather than a crack, and is excluded from best-segment matching.
#: The model-minus-null gap is unchanged from 0 to 15% and shifts by 0.2 deg at
#: 30%, so the conclusion does not rest on this value.
MIN_SEGMENT_FRAC = 0.15


def observed_segment_orientations(sample_id, root=None,
                                  min_frac=MIN_SEGMENT_FRAC):
    """Orientation and length of every digitized crack in one specimen.

    The observed traces are not single cracks. Every specimen carries between
    two and four connected segments, while the stepper produces one path, so a
    comparison against the longest segment alone discards the rest: on this
    dataset it sets aside 44% of the digitized length on average, and the
    longest segment is a minority of the trace in six of the fourteen.

    Returns a list of ``(orientation_deg, length_m)``, longest first, with
    fragments below ``min_frac`` of the total length dropped. If that would
    empty the list the longest segment is kept, so the function always returns
    at least one candidate.
    """
    from . import lithology as _lith
    obs = load_observed_trace(_lith.observed_trace_path(sample_id, root))
    out = []
    for sx, sy in split_segments(obs["x"], obs["y"]):
        if len(sx) < 3:
            continue
        ang, _ = fit_orientation_deg(sx, sy)
        if np.isfinite(ang):
            out.append((float(ang), float(arclength(sx, sy))))
    if not out:
        return []
    total = sum(L for _, L in out)
    kept = [(a, L) for a, L in out if L / total >= float(min_frac)]
    if not kept:
        kept = [max(out, key=lambda t: t[1])]
    return sorted(kept, key=lambda t: -t[1])


def _best_segment_error_deg(orientation_deg, segments):
    """Smallest axial error between one orientation and any crack.

    Private. Scoring a single predictor this way is not interpretable on its
    own, so the public entry points always return a paired null; see
    :func:`best_segment_error_pair`.
    """
    if not segments or not np.isfinite(orientation_deg):
        return float("nan")
    return float(min(axial_angular_error_deg(float(orientation_deg), a)
                     for a, _ in segments))


def best_segment_error_pair(predicted_deg, segments, null_orientation_deg=None):
    """Best-segment error of the model **and** of the null, together.

    The multi-crack generalisation of the primary-segment error: the prediction
    is scored against whichever digitized crack it matches best. That is the
    right comparison when the specimen broke more than once, and it is the form
    to use on other datasets.

    It returns a pair, and there is no public way to obtain the model half
    alone, because the single number is not interpretable. Matching the model
    to any of several cracks while holding the null to one manufactures skill
    out of the segment count: on this dataset that mismatch alone turns a
    0.015 degree dead heat into an apparent 0.79 degree win for the framework,
    the largest margin anywhere in the comparison and entirely an artefact.
    Returning both halves together is what makes that mistake require
    deliberate effort rather than an oversight.

    Returns ``(model_error_deg, null_error_deg)``.
    """
    from .conventions import LOADING_AXIS_DEG
    null = (LOADING_AXIS_DEG if null_orientation_deg is None
            else float(null_orientation_deg))
    return (_best_segment_error_deg(predicted_deg, segments),
            _best_segment_error_deg(null, segments))


def best_segment_statistics(rows=None, root=None, min_frac=MIN_SEGMENT_FRAC,
                            null_orientation_deg=None):
    """Model and null error under best-segment matching, scored identically."""
    from .conventions import LOADING_AXIS_DEG
    null = LOADING_AXIS_DEG if null_orientation_deg is None else float(null_orientation_deg)
    rows = compare_all(root=root) if rows is None else rows
    model, nulls = [], []
    for r in rows:
        if r.get("status") != "computed":
            continue
        segs = observed_segment_orientations(r["sample"], root, min_frac)
        m_err, n_err = best_segment_error_pair(
            r["predicted_orientation_deg"], segs, null)
        model.append(m_err)
        nulls.append(n_err)
    m = np.array(model, float); n = np.array(nulls, float)
    m = m[np.isfinite(m)]; n = n[np.isfinite(n)]
    return dict(n=int(len(m)), min_segment_frac=float(min_frac),
                model_mae_deg=float(m.mean()), null_mae_deg=float(n.mean()),
                model_minus_null_deg=float(m.mean() - n.mean()),
                model_n_within_5=int((m <= 5).sum()),
                model_n_within_10=int((m <= 10).sum()))
