"""Four-mechanism failure classification: WT, WS, MT, MS.

The four primary classes distinguish both the **mechanism** (tensile or shear)
and the **structural locus** (weak plane or intact matrix):

======  ==========================================================
 WT     tensile opening along the weak plane (foliation)
 WS     shear sliding along the weak plane
 MT     tensile cracking through the intact matrix
 MS     shear cracking through the intact matrix
======  ==========================================================

Utilities, under a tension-positive convention with
``<x>_+ = max(x, 0)``, unit foliation normal ``n`` and tangent ``t``:

.. math::

    \\sigma_n^{wp} = n^T \\sigma n, \\qquad \\tau^{wp} = t^T \\sigma n

    R_{WT} = \\frac{\\langle \\sigma_n^{wp} \\rangle_+}{T_{wp}}

    R_{WS} = \\frac{|\\tau^{wp}|}{c_{wp} + \\langle -\\sigma_n^{wp}\\rangle_+ \\tan\\phi_{wp}}

    R_{MT} = \\frac{\\langle \\sigma_1 \\rangle_+}{T_{m}}

    R_{MS} = \\max_{\\beta}\\;
      \\frac{|\\tau_\\beta|}{c_m + \\langle -\\sigma_{n\\beta}\\rangle_+ \\tan\\phi_m}

``beta`` denotes a candidate plane **NORMAL**, scanned over ``[0, pi)``.

Weak-plane utilities are admissible only where a weak plane is actually
modelled. Where the activation weight is below ``activation_floor`` the WT and
WS utilities are returned as NaN and are **excluded from the argmax**, so a
matrix point can never be labelled a weak-plane failure.

"Mixed" is not a primary class. It is a secondary flag raised when the
second-largest *active* utility is within ``eta_mix`` of the largest.
"""

from __future__ import annotations

import numpy as np

#: Primary class order, used consistently for colour mapping and legends.
CLASS_ORDER = ("WT", "WS", "MT", "MS")
NO_FAILURE = "none"

#: Integer codes for array storage. -1 marks a point below the failure threshold.
CLASS_CODES = {"WT": 0, "WS": 1, "MT": 2, "MS": 3, NO_FAILURE: -1}
CODE_TO_CLASS = {v: k for k, v in CLASS_CODES.items()}

_EPS = 1e-12


def _positive_part(a):
    return np.maximum(np.asarray(a, float), 0.0)


def _validate_strength(name, value, allow_zero=False):
    v = np.asarray(value, float)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains non-finite values.")
    if allow_zero:
        if np.any(v < 0):
            raise ValueError(f"{name} must be non-negative.")
    elif np.any(v <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return v


def _validate_friction(name, phi):
    p = np.asarray(phi, float)
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values; friction angle must be finite.")
    if np.any(p < 0) or np.any(p >= np.pi / 2):
        raise ValueError(f"{name} out of physical range: need 0 <= phi < pi/2 rad "
                         f"(got min={float(p.min()):.5f}, max={float(p.max()):.5f}).")
    return p


def resolve_traction(sxx, syy, txy, beta):
    """Normal and shear traction on a plane whose NORMAL is at angle ``beta`` (rad).

    ``sigma_n = n^T sigma n`` and ``tau = t^T sigma n`` with
    ``n = (cos b, sin b)`` and ``t = (-sin b, cos b)``.
    """
    sxx, syy, txy = (np.asarray(v, float) for v in (sxx, syy, txy))
    c, s = np.cos(beta), np.sin(beta)
    sigma_n = sxx * c * c + syy * s * s + 2.0 * txy * c * s
    tau = (syy - sxx) * s * c + txy * (c * c - s * s)
    return sigma_n, tau


def principal_stresses(sxx, syy, txy):
    """Tension-positive principal stresses ``(s1 >= s3)`` and the ``s1`` direction (rad)."""
    sxx, syy, txy = (np.asarray(v, float) for v in (sxx, syy, txy))
    mean = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5 * (sxx - syy)) ** 2 + txy ** 2)
    return mean + rad, mean - rad, 0.5 * np.arctan2(2.0 * txy, sxx - syy)


def check_symmetry(txy_a, txy_b, tol=1e-9):
    """Verify stress-tensor symmetry within tolerance."""
    d = np.max(np.abs(np.asarray(txy_a, float) - np.asarray(txy_b, float)))
    if d > tol:
        raise ValueError(f"stress tensor is not symmetric within {tol}: max |txy-tyx| = {d}")
    return True


def matrix_shear_utility(sxx, syy, txy, c_m, phi_m, n_theta=361):
    """Maximum matrix-shear utility over candidate plane normals, and the governing beta.

    The candidate interval is ``[0, pi)`` because a plane normal is an axial
    quantity. ``n_theta`` sets the angular resolution; convergence is tested in
    ``tools/tests``.
    """
    c_m = _validate_strength("c_matrix", c_m, allow_zero=True)
    phi_m = _validate_friction("phi_matrix", phi_m)
    betas = np.linspace(0.0, np.pi, int(n_theta), endpoint=False)
    best = None
    best_beta = None
    for b in betas:
        sn, tau = resolve_traction(sxx, syy, txy, b)
        denom = c_m + _positive_part(-sn) * np.tan(phi_m)
        denom = np.where(np.abs(denom) < _EPS, _EPS, denom)
        r = np.abs(tau) / denom
        if best is None:
            best = r
            best_beta = np.full_like(r, b)
        else:
            upd = r > best
            best = np.where(upd, r, best)
            best_beta = np.where(upd, b, best_beta)
    return best, best_beta


def classify(sxx, syy, txy, *, alpha_f, T_wp, c_wp, phi_wp, T_m, c_m, phi_m,
             weak_plane_weight=None, activation_floor=0.0, threshold=1.0,
             eta_mix=0.9, n_theta=361, stress_convention="tension_positive"):
    """Classify each point into WT / WS / MT / MS with a secondary mixed flag.

    Parameters
    ----------
    sxx, syy, txy : array_like
        Stress components in the global frame.
    alpha_f : float or array_like
        Foliation trace direction (radians).
    T_wp, c_wp, phi_wp : weak-plane tensile strength, cohesion, friction angle (rad)
    T_m, c_m, phi_m : matrix tensile strength, cohesion, friction angle (rad)
    weak_plane_weight : array_like or None
        Activation weight. Where it is ``<= activation_floor`` the weak-plane
        utilities are inactive and excluded from ranking. ``None`` means the
        weak plane is active everywhere.
    threshold : float
        A point is 'failed' when the governing utility reaches this value.
    eta_mix : float
        Secondary mixed flag raised when ``R_second / R_max >= eta_mix``.

    Returns
    -------
    dict of arrays: the four utilities, ``mode`` (string codes), ``mode_code``,
    ``R_max``, ``R_second``, ``mixed_flag``, ``mixed_ratio``,
    ``secondary_descriptor``, resolved weak-plane tractions, the governing
    matrix-shear plane normal, activation state and failure state.
    """
    from .conventions import to_tension_positive

    sxx, syy, txy = to_tension_positive(sxx, syy, txy, stress_convention)
    T_wp = _validate_strength("T_wp", T_wp)
    T_m = _validate_strength("T_m", T_m)
    c_wp = _validate_strength("c_wp", c_wp, allow_zero=True)
    c_m = _validate_strength("c_matrix", c_m, allow_zero=True)
    phi_wp = _validate_friction("phi_wp", phi_wp)
    phi_m = _validate_friction("phi_matrix", phi_m)

    # weak-plane tractions: beta is the foliation NORMAL
    beta_n = np.asarray(alpha_f, float) + np.pi / 2.0
    sn_wp, tau_wp = resolve_traction(sxx, syy, txy, beta_n)

    R_WT = _positive_part(sn_wp) / np.where(np.abs(T_wp) < _EPS, _EPS, T_wp)
    den_ws = c_wp + _positive_part(-sn_wp) * np.tan(phi_wp)
    R_WS = np.abs(tau_wp) / np.where(np.abs(den_ws) < _EPS, _EPS, den_ws)

    s1, s3, theta_p = principal_stresses(sxx, syy, txy)
    R_MT = _positive_part(s1) / np.where(np.abs(T_m) < _EPS, _EPS, T_m)
    R_MS, beta_crit = matrix_shear_utility(sxx, syy, txy, c_m, phi_m, n_theta)

    # weak-plane activation
    if weak_plane_weight is None:
        active = np.ones_like(R_WT, dtype=bool)
        w = np.ones_like(R_WT)
    else:
        w = np.asarray(weak_plane_weight, float)
        active = w > activation_floor
    R_WT_eff = np.where(active, R_WT, np.nan)
    R_WS_eff = np.where(active, R_WS, np.nan)

    stack = np.stack([R_WT_eff, R_WS_eff,
                      np.broadcast_to(R_MT, R_WT.shape).astype(float),
                      np.broadcast_to(R_MS, R_WT.shape).astype(float)])
    # NaN utilities are inactive and must never win the argmax
    filled = np.where(np.isfinite(stack), stack, -np.inf)
    idx = np.argmax(filled, axis=0)
    R_max = np.take_along_axis(filled, idx[None], axis=0)[0]

    srt = np.sort(filled, axis=0)
    R_second = srt[-2]
    R_second = np.where(np.isfinite(R_second), R_second, np.nan)

    failed = R_max >= float(threshold)
    mode_code = np.where(failed, idx, CLASS_CODES[NO_FAILURE]).astype(np.int8)

    with np.errstate(divide="ignore", invalid="ignore"):
        mixed_ratio = np.where(R_max > 0, R_second / R_max, np.nan)
    mixed_flag = failed & np.isfinite(mixed_ratio) & (mixed_ratio >= float(eta_mix))

    names = np.array(CLASS_ORDER)
    mode = np.where(failed, names[idx], NO_FAILURE)

    second_idx = np.argsort(filled, axis=0)[-2]
    descriptor = _secondary_descriptor(idx, second_idx, mixed_flag)

    return dict(
        R_WT=R_WT_eff, R_WS=R_WS_eff, R_MT=np.broadcast_to(R_MT, R_WT.shape),
        R_MS=np.broadcast_to(R_MS, R_WT.shape),
        R_max=np.where(np.isfinite(R_max), R_max, np.nan), R_second=R_second,
        utility_difference=np.where(np.isfinite(R_second), R_max - R_second, np.nan),
        mode=mode, mode_code=mode_code, failed=failed,
        mixed_ratio=mixed_ratio, mixed_flag=mixed_flag,
        secondary_descriptor=descriptor,
        sigma_n_wp=sn_wp, tau_wp=tau_wp, sigma_1=s1, sigma_3=s3, theta_p=theta_p,
        beta_crit_matrix_shear=beta_crit,
        weak_plane_weight=w, weak_plane_active=active,
        locus=np.where(failed, np.where(idx < 2, "weak_plane", "matrix"), NO_FAILURE),
        mechanism=np.where(failed, np.where((idx == 0) | (idx == 2), "tensile", "shear"),
                           NO_FAILURE),
        threshold=float(threshold), eta_mix=float(eta_mix), n_theta=int(n_theta),
    )


def _secondary_descriptor(first_idx, second_idx, mixed_flag):
    """Name the competing pair where the secondary mixed flag is raised."""
    out = np.full(np.shape(first_idx), "", dtype=object)
    pairs = {
        frozenset((0, 1)): "weak-plane opening-sliding",
        frozenset((2, 3)): "matrix tensile-shear",
        frozenset((0, 2)): "tensile competition across loci",
        frozenset((1, 3)): "shear competition across loci",
        frozenset((0, 3)): "weak-plane versus matrix competition",
        frozenset((1, 2)): "weak-plane versus matrix competition",
    }
    fi = np.asarray(first_idx).ravel()
    si = np.asarray(second_idx).ravel()
    mf = np.asarray(mixed_flag).ravel()
    flat = out.ravel()
    for k in range(flat.size):
        if mf[k] and fi[k] != si[k]:
            flat[k] = pairs.get(frozenset((int(fi[k]), int(si[k]))), "")
    return flat.reshape(np.shape(first_idx))


def class_fractions(mode_code, mask=None):
    """Area fraction of each primary class among in-domain points."""
    code = np.asarray(mode_code)
    if mask is not None:
        code = code[np.asarray(mask, bool)]
    n = code.size
    out = {c: (float(np.mean(code == CLASS_CODES[c])) if n else np.nan) for c in CLASS_ORDER}
    out[NO_FAILURE] = float(np.mean(code == CLASS_CODES[NO_FAILURE])) if n else np.nan
    out["n_points"] = int(n)
    out["failed_point_fraction"] = (float(np.mean(code != CLASS_CODES[NO_FAILURE]))
                                    if n else np.nan)
    return out
