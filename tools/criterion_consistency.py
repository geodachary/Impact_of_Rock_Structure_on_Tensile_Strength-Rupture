"""Is each failure criterion calibrated for the test it is being applied to?

A pointwise classifier reports whichever criterion is closest to failure. That
is only informative if the criteria are commensurate: if one of them fires far
too early it will win the argmax everywhere, and the resulting map describes
the miscalibration rather than the material.

The check needed is cheap and does not depend on the criterion, the geometry or
the material. Each specimen was loaded until it broke, and the model is scaled
to that load, so a criterion calibrated for this stress state should be reached
at a load factor

.. math::

    \\lambda = 1

where ``lambda`` is the fraction of the observed failure load at which the
criterion is first met. ``lambda`` far below one means the criterion predicts
failure that did not happen; far above one means it cannot explain the failure
that did.

On these specimens the maximum-principal-stress criterion sits at
``lambda = 0.76`` to ``1.44`` with a median ``|ln lambda|`` of 0.19, while
linear Mohr-Coulomb built from the triaxial cohesion and friction angle sits at
``0.22`` to ``1.37`` with a median of 0.74, and fires before the observed
failure load on eleven of the fourteen. The schist at 45 degrees should have
broken at 22% of the load it carried.

That is why matrix shear governs almost every point of the failure maps and
matrix tensile is empty: not because the fabric drives shear, but because the
shear criterion is being evaluated far outside the confinement range it was
measured in and fires first almost everywhere. Re-deriving the tensile strength
so the tensile criterion is exact by construction does not change it, because
Mohr-Coulomb still fires first on eleven specimens.

Run this before reading a classification, on any dataset. A criterion with
``lambda`` of 0.2 is not evidence about a mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fabric_tractions as ft
from . import strain_partitioning as sp

#: A criterion whose load factor falls outside this band is not calibrated for
#: the stress state it is being applied to. Deliberately wide: the point is to
#: catch a criterion firing at a fifth of the observed load, not to police the
#: second decimal.
CONSISTENT_BAND = (0.5, 2.0)


def _centre_index(X, Y):
    return np.unravel_index(np.argmin(np.asarray(X) ** 2 + np.asarray(Y) ** 2),
                            np.asarray(X).shape)


def load_factors(field, strengths) -> dict:
    """Load factor at which each criterion is first met, at the disc centre.

    ``field`` is a loaded field archive and ``strengths`` the per-specimen
    strength record. The centre is used because it is where the Brazilian test
    is interpreted and where the tensile stress is greatest.
    """
    sid = int(field["sample_id"])
    X, Y = np.asarray(field["X"]), np.asarray(field["Y"])
    i = _centre_index(X, Y)
    s1 = float(np.asarray(field["s1"])[i])
    s3 = float(np.asarray(field["s3"])[i])

    p = strengths[sid]
    T_m, c_m, phi_m = p["T_m"], p["c_m"], p["phi_m"]

    lam_tension = T_m / s1 if s1 > 0 else np.inf

    # Griffith, tension-positive: the plain tensile branch where s1 + 3 s3 > 0,
    # the combined branch otherwise, which is the Brazilian case.
    if s1 + 3.0 * s3 > 0:
        lam_griffith = lam_tension
    else:
        num = (s1 - s3) ** 2
        lam_griffith = (-8.0 * T_m * (s1 + s3) / num) if num > 0 else np.inf

    # Mohr-Coulomb on the critical plane through the centre.
    tau = 0.5 * (s1 - s3)
    sigma_n = 0.5 * (s1 + s3)
    denom = tau + sigma_n * np.tan(phi_m)
    lam_mc = (c_m / denom) if denom > 0 else np.inf

    return dict(sample=sid, rock=str(field["rock"]),
                angle_deg=float(field["angle_deg"]),
                sigma_1=s1, sigma_3=s3, T_m=T_m, c_m=c_m,
                phi_deg=float(np.degrees(phi_m)),
                lam_tension=float(lam_tension),
                lam_griffith=float(lam_griffith),
                lam_mohr_coulomb=float(lam_mc))


def consistency_table(root=None) -> pd.DataFrame:
    """Load factors for every specimen, one row each."""
    strengths = sp.specimen_strengths()
    rows = [load_factors(ft.load_field(p), strengths)
            for p in ft.field_files(root)]
    return (pd.DataFrame(rows)
            .sort_values(["rock", "angle_deg"])
            .reset_index(drop=True))


def summary(table=None) -> pd.DataFrame:
    """Per-criterion calibration, as a distance from lambda = 1."""
    t = consistency_table() if table is None else table
    out = []
    for col, name in (("lam_tension", "maximum principal stress"),
                      ("lam_griffith", "Griffith"),
                      ("lam_mohr_coulomb", "Mohr-Coulomb")):
        v = t[col].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        lo, hi = CONSISTENT_BAND
        out.append(dict(criterion=name,
                        median_abs_log_lambda=float(np.median(np.abs(np.log(v)))),
                        lam_min=float(v.min()), lam_max=float(v.max()),
                        n_fires_early=int((v < 1.0).sum()),
                        n_outside_band=int(((v < lo) | (v > hi)).sum()),
                        n=int(v.size)))
    return pd.DataFrame(out)


def is_consistent(table=None, criterion="lam_mohr_coulomb") -> bool:
    """Whether a criterion is calibrated for this stress state on most specimens."""
    t = consistency_table() if table is None else table
    lo, hi = CONSISTENT_BAND
    v = t[criterion].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    return bool(((v >= lo) & (v <= hi)).mean() > 0.5)
