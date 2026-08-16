# stress_helpers.py
# -*- coding: utf-8 -*-
"""
stress_helpers.py

Small, dependency-light helpers for 2D stress transformation + principal stresses.

Conventions:
- (sxx, syy, txy) are engineering stress components in a Cartesian basis.
- alpha is the material rotation angle used in your project.
- Functions accept scalars or numpy arrays.
"""

from __future__ import annotations
import numpy as np


def stress_material_to_global(sxx, syy, txy, alpha):
    """
    Rotate stress components from material coords -> global coords by angle alpha.

    Parameters
    ----------
    sxx, syy, txy : float or np.ndarray
        Stress components in material coordinates.
    alpha : float
        Rotation angle (radians).

    Returns
    -------
    Sxx, Syy, Txy : same shape as inputs
        Stress components in global coordinates.
    """
    sxx = np.asarray(sxx, dtype=float)
    syy = np.asarray(syy, dtype=float)
    txy = np.asarray(txy, dtype=float)

    c = np.cos(alpha)
    s = np.sin(alpha)

    # Standard 2D tensor rotation (Voigt form)
    Sxx = c*c*sxx + s*s*syy - 2.0*s*c*txy
    Syy = s*s*sxx + c*c*syy + 2.0*s*c*txy
    Txy = s*c*(sxx - syy) + (c*c - s*s)*txy
    return Sxx, Syy, Txy


def stress_global_to_material(Sxx, Syy, Txy, alpha):
    """
    Rotate stress components from global coords -> material coords.
    This is the inverse of stress_material_to_global, i.e. use -alpha.

    Returns sxx, syy, txy in material coords.
    """
    return stress_material_to_global(Sxx, Syy, Txy, -float(alpha))


def principal_from_components(sxx, syy, txy):
    """
    Principal stresses and principal angle from 2D stress components.

    Returns
    -------
    s1 : maximum principal stress
    s3 : minimum principal stress
    th : principal angle (radians), where the s1 direction is at angle th
         relative to the x-axis of the input coordinate system.
    """
    sxx = np.asarray(sxx, dtype=float)
    syy = np.asarray(syy, dtype=float)
    txy = np.asarray(txy, dtype=float)

    th = 0.5 * np.arctan2(2.0*txy, (sxx - syy))
    rad = np.sqrt(((sxx - syy) * 0.5)**2 + txy**2)

    s_avg = 0.5 * (sxx + syy)
    s1 = s_avg + rad
    s3 = s_avg - rad
    return s1, s3, th


def eval_stress_field_material(x, y, R, p1, p2, a1, a2):
    z1 = x + p1*y
    z2 = x + p2*y
    zh1 = z1 / R
    zh2 = z2 / R
    f1pp = np.zeros_like(z1, dtype=complex)
    f2pp = np.zeros_like(z2, dtype=complex)
    Mdeg = len(a1)-1
    invR2 = 1.0/(R*R)
    for m in range(2, Mdeg+1):
        f1pp += a1[m] * m*(m-1) * (zh1**(m-2)) * invR2
        f2pp += a2[m] * m*(m-1) * (zh2**(m-2)) * invR2
    sxx = np.real(p1**2*f1pp + p2**2*f2pp)
    syy = np.real(f1pp + f2pp)
    txy = -np.real(p1*f1pp + p2*f2pp)
    return sxx, syy, txy