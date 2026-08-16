"""Tractions resolved on the foliation plane, from the exported DDM fields.

The principal-direction field of a Brazilian disk is almost independent of
orientation. Loading is traction-controlled on a simply connected domain, so the
stress field depends on the compliances only through the compatibility equation,
and across the full 0-90 degree range the principal axes rotate by under two
degrees in the gneiss and under five in the schist. Plotting principal directions
therefore produces seven panels that look the same whatever the fabric is doing.

Resolving the same stress state onto the foliation plane recovers the
orientation dependence, because it asks a different question: not which way the
principal axes point, but what the stress does *to the fabric*. For a foliation
at angle ``alpha`` to the loading axis, with unit normal
:math:`\\mathbf{n} = (-\\sin\\alpha, \\cos\\alpha)` and in-plane direction
:math:`\\mathbf{s} = (\\cos\\alpha, \\sin\\alpha)`,

.. math::

    \\sigma_n = \\mathbf{n} \\cdot \\boldsymbol{\\sigma} \\mathbf{n},
    \\qquad
    \\tau = \\mathbf{s} \\cdot \\boldsymbol{\\sigma} \\mathbf{n}.

:math:`\\sigma_n` is the traction trying to pull the foliation apart and
:math:`\\tau` the traction trying to slide it. Both vary strongly and
monotonically with orientation, and together they separate the two ways a
foliation plane can fail.

Sign convention follows the exported fields: tension positive, so
:math:`\\sigma_n < 0` means the foliation is clamped shut and cannot open.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import lithology as lith
from . import output_dirs
from .data_io import canonical_rock_type

FIELD_DIR = output_dirs.FIELDS_NPZ

#: Fraction of the radius kept when averaging, to stay clear of the platen
#: contacts where the stress concentration is a boundary-condition artefact
#: rather than a property of the specimen.
CORE_FRAC = 0.85


def field_files(root=None):
    """Exported field files, ordered by specimen number."""
    root = Path(root or lith.REPO_ROOT)
    return sorted((root / FIELD_DIR).glob("sample_*_full_fields.npz"))


def load_field(path) -> dict:
    """One exported field, with the rock name in its canonical spelling."""
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["rock"] = canonical_rock_type(str(out["rock"]))
    out["sample_id"] = int(out["sample_id"])
    out["angle_deg"] = float(out["angle_deg"])
    return out


def core_mask(f, frac=CORE_FRAC):
    """Disk interior, away from the platen contacts."""
    return f["M"].astype(bool) & (np.hypot(f["X"], f["Y"]) < float(frac) * float(f["R_m"]))


def resolve_on_foliation(f):
    """``(sigma_n, tau)`` fields resolved on the foliation plane."""
    a = float(f["alpha_const_rad"])
    n = (-np.sin(a), np.cos(a))
    s = (np.cos(a), np.sin(a))
    sxx, syy, txy = f["sxx"], f["syy"], f["txy"]
    sigma_n = n[0] ** 2 * sxx + 2.0 * n[0] * n[1] * txy + n[1] ** 2 * syy
    tau = s[0] * (n[0] * sxx + n[1] * txy) + s[1] * (n[0] * txy + n[1] * syy)
    return sigma_n, tau


def traction_table(root=None, frac=CORE_FRAC) -> pd.DataFrame:
    """Mean and peak foliation-resolved tractions for every specimen."""
    rows = []
    for p in field_files(root):
        f = load_field(p)
        m = core_mask(f, frac)
        sn, tau = resolve_on_foliation(f)
        sn, tau = sn[m], tau[m]
        rows.append(dict(
            sample_id=f["sample_id"], rock=f["rock"], angle_deg=f["angle_deg"],
            sigma_n_mean_MPa=float(np.mean(sn)),
            sigma_n_max_MPa=float(np.max(sn)),
            tau_abs_mean_MPa=float(np.mean(np.abs(tau))),
            tau_abs_max_MPa=float(np.max(np.abs(tau))),
            open_fraction=float(np.mean(sn > 0.0)),
        ))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)


def principal_rotation_table(root=None, frac=CORE_FRAC) -> pd.DataFrame:
    """How far the principal axes rotate away from the 0 degree specimen.

    This is the quantity the stress-glyph figure draws. Tabulating it makes the
    near-invariance measurable instead of something the reader has to take on
    trust from panels that look alike.
    """
    fields = [load_field(p) for p in field_files(root)]
    rows = []
    for rock in sorted({f["rock"] for f in fields}):
        grp = sorted((f for f in fields if f["rock"] == rock), key=lambda f: f["angle_deg"])
        ref = grp[0]
        m = core_mask(ref, frac)
        th0 = ref["th_rad"][m]
        for f in grp:
            d = np.degrees(np.abs(((f["th_rad"][m] - th0 + np.pi / 2) % np.pi) - np.pi / 2))
            rows.append(dict(rock=rock, angle_deg=f["angle_deg"],
                             rotation_median_deg=float(np.median(d)),
                             rotation_p95_deg=float(np.percentile(d, 95))))
    return pd.DataFrame(rows)


def heterogeneity_table(root=None, frac=CORE_FRAC) -> pd.DataFrame:
    """Spread of principal orientation *within* each specimen.

    Principal directions are axial data with period 180 degrees, so the spread
    is a circular statistic on the doubled angle. Reported because the earlier
    reading of these panels -- that orientation heterogeneity is markedly higher
    in the gneiss -- is a claim about this quantity and can be checked against it.
    """
    rows = []
    for p in field_files(root):
        f = load_field(p)
        th = f["th_rad"][core_mask(f, frac)]
        z = np.abs(np.mean(np.exp(2j * th)))
        rows.append(dict(rock=f["rock"], angle_deg=f["angle_deg"],
                         circ_sd_deg=float(np.degrees(np.sqrt(-2.0 * np.log(z))) / 2.0)))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)


def sign_change_angle(tab, rock):
    """Fabric angle at which the mean normal traction crosses zero."""
    g = tab[tab.rock == rock].sort_values("angle_deg")
    a = g.angle_deg.to_numpy(float)
    v = g.sigma_n_mean_MPa.to_numpy(float)
    k = int(np.argmax(v > 0.0))
    if k == 0:
        return float("nan")
    return float(a[k - 1] + (a[k] - a[k - 1]) * (-v[k - 1]) / (v[k] - v[k - 1]))


def orientation_gradient_table(root=None, frac=CORE_FRAC) -> pd.DataFrame:
    """How sharply principal orientation changes from place to place.

    "Abrupt stress reorientation" is a statement about the spatial gradient of
    principal direction, so this measures exactly that: the magnitude of the
    gradient of the orientation field, in degrees per millimetre. Orientation is
    axial with period 180 degrees, so the gradient is taken on the doubled angle
    through its complex representation and halved, which avoids the wrap
    artefacts that a plain finite difference on ``th_rad`` would produce.
    """
    rows = []
    for p in field_files(root):
        f = load_field(p)
        m = core_mask(f, frac)
        z = np.exp(2j * f["th_rad"])
        dy, dx = np.gradient(z, f["Y"][:, 0], f["X"][0, :])
        # |d(2*theta)| = |dz| / |z| for unit-modulus z; halve to get d(theta)
        g = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2) / 2.0
        g = np.degrees(g[m]) * 1e-3          # deg per metre -> deg per mm
        rows.append(dict(rock=f["rock"], angle_deg=f["angle_deg"],
                         grad_median_deg_per_mm=float(np.median(g)),
                         grad_p90_deg_per_mm=float(np.percentile(g, 90))))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)
