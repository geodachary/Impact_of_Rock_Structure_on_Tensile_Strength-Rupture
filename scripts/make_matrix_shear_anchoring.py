#!/usr/bin/env python3
"""How the matrix shear envelope is anchored, and what the alternative costs.

The four-class classification reads matrix shear against matrix tensile, so
the anchoring of the shear envelope decides which of the two governs. This
records the check that makes the adopted anchoring defensible and measures the
alternative, in a table rather than a figure.

Adopted: cohesion and friction angle measured in triaxial compression. The
consistency test is whether that pair reproduces the uniaxial compressive
strength measured on the same specimen, since for a Mohr-Coulomb material
``UCS = 2 c cos(phi) / (1 - sin(phi))``. It does, to 81-112%, which is the
number Sections 4.6 and 4.10 quote.

Alternative: cohesion re-anchored so the envelope reproduces that UCS exactly,

    c = UCS (1 - sin phi) / (2 cos phi),

with nothing else changed.

The history of this check is worth recording. It began as a supplementary
figure showing that matrix tensile was empty under the adopted calibration
and appeared only under re-anchoring. Corrected cohesion values withdrew that
result and matrix tensile became present, so the figure was retired. Moving
from the adopted anisotropy ratios to the measured ones has emptied the class
again, and now under both anchorings: the fractions below are zero to three
decimals throughout. The cohesion-ratio measurement the table rests on is
still what Sections 4.6 and 5.5 quote, so the table is kept.

    python scripts/make_matrix_shear_anchoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import failure_classification as fc          # noqa: E402
from tools import fabric_tractions as ft                # noqa: E402
from tools import lithology as lith                     # noqa: E402
from tools import output_dirs                           # noqa: E402
from tools import strain_partitioning as sp             # noqa: E402

#: Measured per-angle uniaxial compressive strength, MPa.
UCS_BY_SPECIMEN = {
    1: 55.472, 2: 50.884, 3: 38.514, 4: 45.0675, 5: 48.950, 6: 61.3425, 7: 67.4625,
    8: 40.5067, 9: 40.2933, 10: 28.0267, 11: 25.6667, 12: 36.830, 13: 39.6025,
    14: 43.330,
}
CLASSES = ("WT", "WS", "MT", "MS")


def cohesion_from_ucs(ucs, phi):
    """Cohesion of a Mohr-Coulomb material with this UCS and friction angle."""
    return ucs * (1.0 - np.sin(phi)) / (2.0 * np.cos(phi))


def fractions(sample_id, strengths, ucs_anchored):
    """Four-class area fractions over the analysis interior, one anchoring."""
    d = np.load(lith.field_cache_path(sample_id), allow_pickle=True)
    p = strengths[sample_id]
    c = (cohesion_from_ucs(UCS_BY_SPECIMEN[sample_id], p["phi_m"])
         if ucs_anchored else p["c_m"])
    res = fc.classify(
        d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=sp.WEAK_T_RATIO * p["T_m"], c_wp=sp.WEAK_C_RATIO * c,
        phi_wp=p["phi_m"], T_m=p["T_m"], c_m=c, phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=sp.ACTIVATION_FLOOR,
        threshold=sp.THRESHOLD, eta_mix=sp.ETA_MIX, n_theta=sp.N_THETA)
    radius = np.hypot(np.asarray(d["X"], float), np.asarray(d["Y"], float))
    mask = d["M"].astype(bool) & (radius <= ft.CORE_FRAC * float(d["R_m"]))
    code = res["mode_code"][mask]
    return {k: float(np.mean(code == fc.CLASS_CODES[k])) for k in CLASSES}


def main():
    strengths = sp.specimen_strengths()
    rows = []
    for sid in range(1, 15):
        p = strengths[sid]
        ucs = UCS_BY_SPECIMEN[sid]
        c_implied = cohesion_from_ucs(ucs, p["phi_m"])
        adopted = fractions(sid, strengths, False)
        anchored = fractions(sid, strengths, True)
        L = lith.lithology_of_sample(sid)
        rows.append(dict(
            sample_id=sid, rock=L.display_name, angle_deg=L.angle_for_sample(sid),
            measured_UCS_MPa=ucs,
            cohesion_measured_MPa=p["c_m"],
            cohesion_implied_by_UCS_MPa=c_implied,
            cohesion_ratio=p["c_m"] / c_implied,
            **{f"{k}_adopted": adopted[k] for k in CLASSES},
            **{f"{k}_ucs_anchored": anchored[k] for k in CLASSES},
        ))
    df = pd.DataFrame(rows)
    out = Path(output_dirs.tables()) / "matrix_shear_anchoring.csv"
    df.to_csv(out, index=False)

    r = df.cohesion_ratio
    print(f"  measured cohesion reproduces {100 * r.min():.0f}-{100 * r.max():.0f}% "
          "of the cohesion each specimen's own UCS implies")
    print(f"  matrix tensile, adopted     : max {df.MT_adopted.max():.3f} "
          f"(specimen {int(df.sample_id[df.MT_adopted.idxmax()])})")
    print(f"  matrix tensile, re-anchored : max {df.MT_ucs_anchored.max():.3f} "
          f"(specimen {int(df.sample_id[df.MT_ucs_anchored.idxmax()])})")
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
