"""Orientation-dependent tensile strength envelope for foliated rock.

The envelope is a transversely isotropic interpolation between the two measured
end members, attenuated by a weakening term that is active only at intermediate
fabric angles:

.. math::

    \\sigma(\\beta) = \\frac{\\sigma_{\\mathrm{TI}}(\\beta)}
                          {\\sqrt{1 + \\eta\\, \\mathcal{H}(\\beta;\\beta_p)}},
    \\qquad
    \\sigma_{\\mathrm{TI}}(\\beta) =
      \\left[\\frac{\\cos^2\\beta}{\\sigma_0^2}
           + \\frac{\\sin^2\\beta}{\\sigma_{90}^2}\\right]^{-1/2}

with the weakening kernel

.. math::

    \\mathcal{H}(\\beta;\\beta_p) \\propto
        \\sin^2(\\pi x)\\,\\exp[\\kappa (x - \\tfrac12)],
    \\qquad x = \\beta/90^\\circ,

normalised so that :math:`\\max \\mathcal{H} = 1`.

Why this kernel
---------------
``sin^2`` vanishes together with its first derivative at both end members, so
the envelope returns to :math:`\\sigma_0` and :math:`\\sigma_{90}` smoothly. An
earlier form used ``cos^{1-\\xi}``, whose exponent tends to zero as the
asymmetry grows; the kernel then stays near unity across the whole range and
collapses only in an infinitesimal neighbourhood of an end member, producing a
step. The exponential here carries the asymmetry instead, leaving the endpoint
behaviour untouched.

What is fitted
--------------
``sigma_0`` and ``sigma_90`` are **measured**, not fitted: they are the mean
strengths at 0 and 90 degrees. Only two quantities are fitted, and neither can
be measured directly:

``eta``
    depth of the weakening; the deepest attenuation is
    :math:`W_{\\min} = (1+\\eta)^{-1/2}`.
``beta_peak``
    fabric angle at which the weakening is strongest, parameterised directly
    rather than through :math:`\\kappa`, because it is bounded to
    :math:`(0^\\circ, 90^\\circ)` by geometry and needs no arbitrary limit.

One physical constraint applies: the envelope may not predict a strength below
the lowest specimen actually broken. That is a statement about the curve, so it
does not depend on choosing a reference angle.

Uncertainties are profile-likelihood intervals — each parameter is scanned while
the other is re-optimised — which report what the data constrain rather than a
linearised approximation around the optimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

EPS = 1e-12

#: Chi-square rise defining a 95% interval for one parameter.
DELTA_CHI2_95 = 3.84

#: Search bounds. beta_peak is bounded by geometry; eta is generous and is not
#: expected to bind once the admissibility constraint is applied.
ETA_BOUNDS = (0.0, 20.0)
BETA_BOUNDS = (30.0, 89.0)


def kappa_from_peak(beta_peak_deg):
    """Skew parameter that places the kernel maximum at ``beta_peak_deg``."""
    x = np.clip(float(beta_peak_deg), 1e-6, 90.0 - 1e-6) / 90.0
    return -2.0 * np.pi / np.tan(np.pi * x)


def weakening_kernel(beta_rad, beta_peak_deg):
    """Normalised weakening kernel; zero at both end members, unit maximum."""
    x = np.clip(2.0 * np.asarray(beta_rad, float) / np.pi, 0.0, 1.0)
    k = kappa_from_peak(beta_peak_deg)
    raw = np.sin(np.pi * x) ** 2 * np.exp(k * (x - 0.5))
    xp = np.clip(float(beta_peak_deg) / 90.0, 1e-9, 1.0 - 1e-9)
    raw_max = max(float(np.sin(np.pi * xp) ** 2 * np.exp(k * (xp - 0.5))), EPS)
    out = np.where((x <= EPS) | (x >= 1.0 - EPS), 0.0, raw / raw_max)
    return np.clip(out, 0.0, 1.0 + 1e-10)


def ti_envelope(beta_rad, sigma0, sigma90):
    """Transversely isotropic interpolation between the two end members."""
    c, s = np.cos(beta_rad), np.sin(beta_rad)
    return 1.0 / np.sqrt(c ** 2 / float(sigma0) ** 2 + s ** 2 / float(sigma90) ** 2)


def strength(beta_rad, sigma0, sigma90, eta, beta_peak_deg):
    """Predicted tensile strength at fabric angle ``beta_rad`` (radians)."""
    return (ti_envelope(beta_rad, sigma0, sigma90)
            / np.sqrt(1.0 + float(eta) * weakening_kernel(beta_rad, beta_peak_deg)))


def weakening_factor(beta_rad, eta, beta_peak_deg):
    """:math:`W = \\sigma/\\sigma_{\\mathrm{TI}}`; unity at both end members."""
    return 1.0 / np.sqrt(1.0 + float(eta) * weakening_kernel(beta_rad, beta_peak_deg))


def cosine_law(angle_deg, S1, S2, beta):
    """The classical cosine law, for comparison against the envelope above.

    :math:`S_1 + S_2\\cos[2(\\beta - \\beta_0)]` -- the standard single-harmonic
    description of strength anisotropy. It is fitted to the same measurements
    so that the two minima can be quoted side by side; it is *not* part of the
    ATI model and shares none of its parameters.
    """
    return S1 + S2 * np.cos(2.0 * np.deg2rad(np.asarray(angle_deg, float) - beta))


def per_angle_statistics(df_rock, angle_col="Angle", value_col="Tensile_strength_Mpa"):
    """Mean, standard deviation, count and standard error at each fabric angle."""
    g = df_rock.groupby(df_rock[angle_col].round().astype(int))[value_col]
    st = g.agg(["mean", "std", "count", "min"])
    st["sem"] = st["std"] / np.sqrt(st["count"])
    return st


def _weights(df_rock, angle_col="Angle", value_col="Tensile_strength_Mpa"):
    """Per-angle standard deviation, falling back to the global value."""
    st = df_rock.groupby(angle_col)[value_col].agg(["std"])["std"].to_dict()
    y = df_rock[value_col].to_numpy(float)
    glob = float(np.nanstd(y, ddof=1)) if len(y) > 1 else 1.0
    if not np.isfinite(glob) or glob <= 0:
        glob = 1.0
    w = np.array([st.get(a, np.nan) for a in df_rock[angle_col]], float)
    w[~np.isfinite(w) | (w <= 0)] = glob
    return np.clip(w, 1e-6, None)


def fit(df_rock, angle_col="Angle", value_col="Tensile_strength_Mpa", seed=42):
    """Fit ``eta`` and ``beta_peak`` with the end members held at their measured values."""
    d = df_rock.dropna(subset=[angle_col, value_col]).copy()
    st = per_angle_statistics(d, angle_col, value_col)
    if 0 not in st.index or 90 not in st.index:
        raise ValueError("both 0 and 90 degree end members are required")

    sigma0 = float(st.loc[0, "mean"])
    sigma90 = float(st.loc[90, "mean"])
    floor = float(d[value_col].min())          # lowest specimen actually broken

    beta = np.deg2rad(d[angle_col].to_numpy(float))
    y = d[value_col].to_numpy(float)
    w = _weights(d, angle_col, value_col)
    dense = np.linspace(0.0, np.pi / 2, 4001)

    def objective(p):
        r = (strength(beta, sigma0, sigma90, p[0], p[1]) - y) / w
        # penalise any excursion below the lowest measured specimen
        below = np.maximum(0.0, floor - strength(dense, sigma0, sigma90, p[0], p[1])).max()
        return float(np.sum(r ** 2) + 1e4 * below ** 2)

    res = differential_evolution(objective, [ETA_BOUNDS, BETA_BOUNDS], seed=seed,
                                 maxiter=2500, popsize=22, tol=1e-10, polish=True,
                                 updating="deferred")
    eta, beta_peak = float(res.x[0]), float(res.x[1])

    pred = strength(beta, sigma0, sigma90, eta, beta_peak)
    resid = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    dof = max(1, len(y) - 2)
    ym = strength(dense, sigma0, sigma90, eta, beta_peak)
    k = int(np.argmin(ym))

    return dict(
        sigma0=sigma0, sigma90=sigma90, specimen_floor=floor,
        eta=eta, beta_peak_deg=beta_peak,
        W_min=float(1.0 / np.sqrt(1.0 + eta)),
        max_weakening_pct=float(100.0 * (1.0 - 1.0 / np.sqrt(1.0 + eta))),
        model_min_MPa=float(ym[k]), model_min_deg=float(np.degrees(dense[k])),
        observed_min_MPa=float(st["mean"].min()), observed_min_deg=int(st["mean"].idxmin()),
        R2=float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        RMSE=float(np.sqrt(np.mean(resid ** 2))),
        chi2_red=float(np.sum((resid / w) ** 2) / dof),
        n_points=int(len(y)), n_angles=int(len(st)),
        _objective=objective,
    )


def profile_interval(result, which, lo, hi, step, seed=1):
    """95% profile-likelihood interval: scan one parameter, re-optimise the other."""
    objective = result["_objective"]
    grid = np.arange(lo, hi + 1e-9, step)
    vals = []
    for v in grid:
        if which == "eta":
            f, bounds = (lambda z: objective([v, z[0]])), [BETA_BOUNDS]
        elif which == "beta_peak":
            f, bounds = (lambda z: objective([z[0], v])), [ETA_BOUNDS]
        else:
            raise ValueError("which must be 'eta' or 'beta_peak'")
        o = differential_evolution(f, bounds, seed=seed, maxiter=300, popsize=10,
                                   tol=1e-8, polish=True)
        vals.append(float(o.fun))
    vals = np.asarray(vals)
    inside = grid[(vals - vals.min()) <= DELTA_CHI2_95]
    return float(inside.min()), float(inside.max()), grid, vals
