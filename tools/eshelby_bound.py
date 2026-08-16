"""How much the homogeneous elastic field neglects by not resolving the augen.

The stress field used throughout is a homogeneous orthotropic solution. The
augen gneiss is not homogeneous at the grain scale: stiff feldspar--quartz augen
sit in a more compliant micaceous matrix. This module bounds what that
simplification costs, so the choice can be defended with a number rather than an
assertion.

Two separate things are worth keeping apart.

The **mean** effect is already in the model. ``Modulus_of_Elasticity`` and the
other constants are measured on the whole specimen, so they are effective moduli
of the two-phase aggregate, not matrix properties. Mori--Tanaka homogenisation is
included here to show the size of that difference: it is what the measurement
already absorbs.

The **fluctuation** about that mean is what is genuinely omitted. Eshelby's
result gives it exactly for one ellipsoidal inhomogeneity: the strain, and hence
the stress, inside the inclusion is uniform, and

.. math::

    \\boldsymbol{\\varepsilon}_I =
      \\left[\\mathbf{I} + \\mathbf{S}\\,\\mathbf{C}_m^{-1}
      (\\mathbf{C}_i - \\mathbf{C}_m)\\right]^{-1} \\boldsymbol{\\varepsilon}^\\infty ,

with :math:`\\mathbf{S}` the Eshelby tensor of the ellipse. Outside, the
perturbation decays as :math:`(a/r)^2` in two dimensions, so it is local.

Neither the phase moduli nor the augen size, shape and volume fraction were
measured for these specimens, so nothing here is presented as a property of the
rock. The functions take those quantities as arguments and the accompanying
script sweeps them over ranges wide enough to bracket a feldspar-in-mica
aggregate. The result is an upper bound on a neglected term, which is all it is
used for.

Plane strain throughout; the matrix is treated as isotropic for this estimate,
which is conservative in the sense that it puts the whole modulus contrast into
the inclusion rather than sharing it with the fabric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Components are ordered (11, 22, 12) with engineering shear, so that the
#: stiffness maps (eps11, eps22, gamma12) to (sig11, sig22, sig12).
VOIGT = ("11", "22", "12")


def stiffness_plane_strain(E, nu) -> np.ndarray:
    """Isotropic plane-strain stiffness in the engineering-shear convention."""
    E, nu = float(E), float(nu)
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array([[c * (1.0 - nu), c * nu, 0.0],
                     [c * nu, c * (1.0 - nu), 0.0],
                     [0.0, 0.0, E / (2.0 * (1.0 + nu))]], float)


def eshelby_ellipse(aspect, nu) -> np.ndarray:
    """Eshelby tensor of an elliptical cylinder, engineering-shear convention.

    ``aspect`` is the ratio of the semi-axis along 2 to the semi-axis along 1,
    so ``aspect = 1`` is a circle and larger values are an inclusion elongated
    across the 1-direction. Components follow Mura, *Micromechanics of Defects
    in Solids*, eq. 11.22; the circular limit reproduces the familiar
    :math:`(5-4\\nu)/[8(1-\\nu)]` and :math:`(4\\nu-1)/[8(1-\\nu)]`.
    """
    a, b = 1.0, float(aspect)
    nu = float(nu)
    d = (a + b) ** 2
    k = 1.0 / (2.0 * (1.0 - nu))

    S1111 = k * ((b * b + 2.0 * a * b) / d + (1.0 - 2.0 * nu) * b / (a + b))
    S2222 = k * ((a * a + 2.0 * a * b) / d + (1.0 - 2.0 * nu) * a / (a + b))
    S1122 = k * (b * b / d - (1.0 - 2.0 * nu) * b / (a + b))
    S2211 = k * (a * a / d - (1.0 - 2.0 * nu) * a / (a + b))
    S1212 = k * ((a * a + b * b) / (2.0 * d) + (1.0 - 2.0 * nu) / 2.0)

    # gamma^c = 2 eps^c_12 = 2 S1212 gamma*, hence the factor two on the shear row
    return np.array([[S1111, S1122, 0.0],
                     [S2211, S2222, 0.0],
                     [0.0, 0.0, 2.0 * S1212]], float)


def strain_concentration(E_m, nu_m, E_i, nu_i, aspect) -> np.ndarray:
    """Dilute strain-concentration tensor ``A`` with ``eps_inclusion = A eps_remote``."""
    Cm = stiffness_plane_strain(E_m, nu_m)
    Ci = stiffness_plane_strain(E_i, nu_i)
    S = eshelby_ellipse(aspect, nu_m)
    return np.linalg.inv(np.eye(3) + S @ np.linalg.inv(Cm) @ (Ci - Cm))


def interior_stress(E_m, nu_m, E_i, nu_i, aspect, remote_stress) -> np.ndarray:
    """Uniform stress inside the inclusion for a given remote stress."""
    Cm = stiffness_plane_strain(E_m, nu_m)
    Ci = stiffness_plane_strain(E_i, nu_i)
    A = strain_concentration(E_m, nu_m, E_i, nu_i, aspect)
    eps_inf = np.linalg.solve(Cm, np.asarray(remote_stress, float))
    return Ci @ (A @ eps_inf)


def concentration_factor(E_m, nu_m, E_i, nu_i, aspect, remote_stress) -> float:
    """Largest stress component inside the inclusion, relative to the remote one."""
    s_inf = np.asarray(remote_stress, float)
    s_in = interior_stress(E_m, nu_m, E_i, nu_i, aspect, s_inf)
    ref = np.max(np.abs(s_inf))
    return float(np.max(np.abs(s_in)) / ref) if ref > 0 else np.nan


def mori_tanaka_modulus(E_m, nu_m, E_i, nu_i, aspect, f) -> np.ndarray:
    """Effective stiffness of the two-phase aggregate at inclusion fraction ``f``.

    This is the quantity a modulus measured on the whole specimen returns, which
    is why the homogeneous model is not missing the mean stiffening.
    """
    Cm = stiffness_plane_strain(E_m, nu_m)
    Ci = stiffness_plane_strain(E_i, nu_i)
    A = strain_concentration(E_m, nu_m, E_i, nu_i, aspect)
    f = float(f)
    return Cm + f * (Ci - Cm) @ A @ np.linalg.inv((1.0 - f) * np.eye(3) + f * A)


def decay_radius(threshold=0.10) -> float:
    """Distance, in inclusion radii, at which the perturbation falls below ``threshold``.

    The exterior disturbance of a circular inhomogeneity in two dimensions falls
    off as :math:`(a/r)^2`, so the radius follows directly.
    """
    return float(np.sqrt(1.0 / float(threshold)))


def bound_table(E_m, nu_m, nu_i, contrasts, aspects, remote_stress) -> pd.DataFrame:
    """Concentration factors over a grid of modulus contrast and inclusion shape."""
    rows = []
    for c in contrasts:
        for ar in aspects:
            rows.append(dict(
                contrast=float(c), aspect=float(ar),
                E_inclusion_GPa=float(E_m * c) / 1e3,
                concentration=concentration_factor(
                    E_m, nu_m, E_m * c, nu_i, ar, remote_stress),
            ))
    return pd.DataFrame(rows)
