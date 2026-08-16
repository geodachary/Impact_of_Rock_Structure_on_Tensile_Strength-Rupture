"""Coordinate, angle and stress-sign conventions — stated once for the project.

Global frame
------------
Right-handed ``(x, y)`` in metres, specimen centre at the origin.
``+x`` is the horizontal diameter; ``+y`` is the loading axis.
Angles increase counterclockwise from ``+x``.

Loading
-------
The load is applied diametrically along ``+y``: platens are centred at global
polar angle ``+90`` and ``-90`` degrees. Verified against
``boundary_traction_model.platen_tractions_theta``.

Angle inventory
---------------
``alpha_exp``    experimental fabric angle, degrees, measured from the
                 horizontal diameter ``+x``. Axial, physically ``[0, 90]``.
                 ``0``  -> foliation trace parallel to ``+x``, perpendicular to loading.
                 ``90`` -> foliation trace parallel to the loading direction.
``alpha_f``      global foliation trace direction, radians, axial, ``[0, pi)``.
``alpha_n``      global foliation normal direction, radians = ``alpha_f + pi/2``.
``theta_p``      maximum-principal-stress direction, radians, axial.
``beta_plane``   candidate failure-plane **NORMAL** angle (not a trace), radians,
                 scanned over ``[0, pi)``.
``theta_crack``  fracture-trace orientation, degrees, axial, ``[0, 180)``.

Stress sign convention
----------------------
**Tension-positive** at the classifier interface. A compression-positive field
must be converted once, at that single interface, rather than by scattering sign
flips through the pipeline.
"""

from __future__ import annotations

import numpy as np

# Specimen geometry (from crack_data_plot.py; the digitized data are in metres)
MM_TO_M = 1.0e-3
DIAMETER_M = 51.0 * MM_TO_M
RADIUS_M = DIAMETER_M / 2.0

#: Consecutive-point spacing above which a digitized trace is treated as broken.
MAX_TRACE_GAP_M = 7.0 * MM_TO_M

#: Loading axis orientation, degrees CCW from +x.
LOADING_AXIS_DEG = 90.0

#: Default portion of the disc radius over which trace orientation is fitted.
DEFAULT_CENTRAL_FRAC = 0.5


def alpha_f_from_alpha_exp_deg(alpha_exp_deg: float) -> float:
    """Global foliation trace direction (radians) from the experimental angle."""
    return float(np.radians(float(alpha_exp_deg) % 180.0))


def alpha_n_from_alpha_f(alpha_f: float) -> float:
    """Global foliation normal direction (radians) = ``alpha_f + pi/2``."""
    return float((float(alpha_f) + np.pi / 2.0) % np.pi)


def foliation_trace_tangent_normal(alpha_f: float):
    """Unit tangent along the foliation trace and unit normal, in the global frame."""
    t_hat = (float(np.cos(alpha_f)), float(np.sin(alpha_f)))
    beta = alpha_n_from_alpha_f(alpha_f)
    n_hat = (float(np.cos(beta)), float(np.sin(beta)))
    return t_hat, n_hat


def wrap_axial_deg(angle_deg):
    """Reduce a line orientation to the canonical ``[0, 180)`` range."""
    return np.mod(np.asarray(angle_deg, dtype=float), 180.0)


def axial_angular_error_deg(theta_pred_deg, theta_obs_deg):
    """Smallest axial angular error between two line orientations, in ``[0, 90]``.

    ``delta = |theta_pred - theta_obs|``;
    ``error = min(delta mod 180, 180 - (delta mod 180))``.
    """
    delta = np.abs(np.asarray(theta_pred_deg, float) - np.asarray(theta_obs_deg, float))
    dmod = np.mod(delta, 180.0)
    return np.minimum(dmod, 180.0 - dmod)


def signed_axial_difference_deg(theta_a_deg, theta_b_deg):
    """Signed axial difference wrapped into ``(-90, 90]``."""
    return np.mod(np.asarray(theta_a_deg, float) - np.asarray(theta_b_deg, float) + 90.0,
                  180.0) - 90.0


def to_tension_positive(sxx, syy, txy, input_convention: str = "tension_positive"):
    """Convert a stress field to the tension-positive convention, once.

    Parameters
    ----------
    input_convention : {'tension_positive', 'compression_positive'}
    """
    if input_convention == "tension_positive":
        return np.asarray(sxx, float), np.asarray(syy, float), np.asarray(txy, float)
    if input_convention == "compression_positive":
        return -np.asarray(sxx, float), -np.asarray(syy, float), -np.asarray(txy, float)
    raise ValueError(f"unknown stress convention: {input_convention!r}")


def describe_end_member(alpha_exp_deg: float) -> str:
    """Physical description of an end-member configuration, derived not hard-coded."""
    alpha_f = alpha_f_from_alpha_exp_deg(alpha_exp_deg)
    t_hat, n_hat = foliation_trace_tangent_normal(alpha_f)
    trace_along_loading = abs(t_hat[1]) > abs(t_hat[0])
    relation = ("parallel to the loading direction" if trace_along_loading
                else "parallel to the horizontal diameter (perpendicular to loading)")
    return (f"alpha_exp={float(alpha_exp_deg):.1f} deg -> alpha_f={alpha_f:.4f} rad; "
            f"trace tangent={t_hat}, normal={n_hat}; trace is {relation}.")
