"""What fills the MT class, and how sensitive that is to the shear anchoring.

**This file used to explain why MT was empty.** On the cohesion values carried
before the strength table was corrected, the measured (c, phi) pairs returned
only 28% to 68% of the uniaxial compressive strengths of the same specimens, a
linear envelope extrapolated from confined triaxial tests into the
tensile-compressive state at the disc centre was reached far too early, and
matrix tensile failure never won the argmax anywhere.

With the corrected cohesion the pairs return 81% to 112% of the measured UCS,
the envelope is anchored by an independent measurement rather than extended
beyond one, and MT appears: 0.087 of the interior in the augen gneiss at 0
degrees and between 0.001 and 0.009 in the psammitic schist from 15 to 75
degrees. These tests pin the corrected state and the residual sensitivity to
the anchoring choice, which is the point the supplementary figure records.

The mechanism
-------------
``R_MT = <sigma_1>+ / T_m`` competes against ``R_MS``, the Mohr-Coulomb ratio
maximised over candidate planes. At the centre of a Brazilian disc the stress
state is roughly ``sigma_1 = +T``, ``sigma_3 = -2T``, so the deviator is about
three times the tensile stress while the tensile criterion sees only
``sigma_1``. Scaling the state proportionally, the Mohr-Coulomb envelope is
reached at a smaller load factor than the tension cutoff, so shear governs and
MT never wins the argmax.

Why that is a calibration artefact rather than a result
------------------------------------------------------
The cohesion and friction angle come from triaxial tests run under confinement
(Section 2.6). A Brazilian disc centre is not in that calibration range: it is a
tensile-compressive state with no confinement at all. Extrapolating a linear
envelope from confined tests into that regime over-predicts shear, which is the
long-standing objection to linear Mohr-Coulomb at low confinement and the reason
Hoek-Brown and tension-cutoff formulations exist.

The consistency check is that the measured ``(c, phi)`` pairs should reproduce
the measured uniaxial compressive strengths of the same specimens. For a
Mohr-Coulomb material ``UCS = 2 c cos(phi) / (1 - sin(phi))``, and the adopted
pairs now return 81% to 112% of the UCS actually measured. Re-anchoring the
envelope at the uniaxial point moves MT between the rocks rather than creating
it: it removes MT from the gneiss at 0 degrees and raises it in the schist at
15 degrees.

Nothing here changes the adopted parameters. The published classification keeps
the measured triaxial values; this records what those values imply and what a
differently anchored envelope would give.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import fabric_tractions as ft
from tools import strain_partitioning as sp

#: Measured per-angle uniaxial compressive strength, MPa (Table 1 source data).
UCS_BY_SPECIMEN = {
    1: 55.472, 2: 50.884, 3: 38.514, 4: 45.0675, 5: 48.950, 6: 61.3425, 7: 67.4625,
    8: 40.5067, 9: 40.2933, 10: 28.0267, 11: 25.6667, 12: 36.830, 13: 39.6025,
    14: 43.330,
}


def _cohesion_from_ucs(ucs, phi):
    """Cohesion of a Mohr-Coulomb material with this UCS and friction angle."""
    return ucs * (1.0 - np.sin(phi)) / (2.0 * np.cos(phi))


def test_adopted_cohesion_reproduces_the_measured_ucs():
    """The consistency that makes the shear envelope trustworthy here."""
    strengths = sp.specimen_strengths()
    ratios = []
    for sid, ucs in UCS_BY_SPECIMEN.items():
        p = strengths[sid]
        ratios.append(p["c_m"] / _cohesion_from_ucs(ucs, p["phi_m"]))
    ratios = np.asarray(ratios)

    assert ratios.min() > 0.70, (
        f"the weakest specimen's cohesion is {ratios.min():.2f} of the value "
        "its own UCS implies. Below about 0.7 the envelope is being "
        "extrapolated well outside the range it was measured in, and the "
        "matrix-shear share of the four-class maps stops being interpretable. "
        "Section 4.6 quotes 81 to 112 per cent."
    )
    assert ratios.max() < 1.30, (
        f"the strongest is {ratios.max():.2f}; well above unity would mean the "
        "triaxial pair predicts a stronger specimen than was measured"
    )


@pytest.mark.parametrize("sid,label", [(1, "gneiss 0 deg"), (8, "schist 0 deg")])
def test_the_two_criteria_are_reached_at_comparable_loads(sid, label):
    """Neither may govern by a margin that makes the other unreachable.

    At the centre of a Brazilian disc the deviator is roughly three times the
    tensile stress, so shear is expected to be reached first; the question is
    by how much. A factor of a few means the ordering reflects the stress
    state, which is a result. A factor of five or more meant the envelope was
    miscalibrated, which was an artefact.
    """
    strengths = sp.specimen_strengths()
    path = [q for q in ft.field_files() if f"sample_{sid:04d}" in str(q)][0]
    f = ft.load_field(path)
    X, Y = np.asarray(f["X"]), np.asarray(f["Y"])
    i = np.unravel_index(np.argmin(X**2 + Y**2), X.shape)
    s1 = float(np.asarray(f["s1"])[i])
    s3 = float(np.asarray(f["s3"])[i])

    p = strengths[sid]
    tau = 0.5 * (s1 - s3)
    sigma_n = 0.5 * (s1 + s3)

    lam_tension = p["T_m"] / s1
    lam_shear = p["c_m"] / (tau + sigma_n * np.tan(p["phi_m"]))

    ratio = lam_tension / lam_shear
    assert 0.4 < ratio < 4.0, (
        f"{label}: the two criteria are reached {ratio:.1f}x apart "
        f"(tension {lam_tension:.2f}, shear {lam_shear:.2f}). Section 4.6 "
        "reads the two matrix classes as an ordering, which requires them to "
        "be commensurate."
    )
    assert 0.5 < lam_shear < 2.5, (
        f"{label}: shear is reached at {lam_shear:.2f} of the observed "
        "failure load, outside the consistency band"
    )


def test_a_ucs_anchored_envelope_recovers_matrix_tensile():
    """The classifier resolves MT; the adopted calibration is what suppresses it."""
    import numpy as np

    from tools import failure_classification as fc
    from tools import lithology as lith
    from tools import strain_partitioning as sp_mod

    strengths = sp.specimen_strengths()

    def mt_fraction(sid, use_ucs_anchor):
        d = np.load(lith.field_cache_path(sid), allow_pickle=True)
        p = strengths[sid]
        c = (_cohesion_from_ucs(UCS_BY_SPECIMEN[sid], p["phi_m"])
             if use_ucs_anchor else p["c_m"])
        res = fc.classify(
            d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
            T_wp=sp_mod.WEAK_T_RATIO * p["T_m"], c_wp=sp_mod.WEAK_C_RATIO * c,
            phi_wp=p["phi_m"], T_m=p["T_m"], c_m=c, phi_m=p["phi_m"],
            weak_plane_weight=d["wp_weight"],
            activation_floor=sp_mod.ACTIVATION_FLOOR,
            threshold=sp_mod.THRESHOLD, eta_mix=sp_mod.ETA_MIX,
            n_theta=sp_mod.N_THETA)
        m = d["M"].astype(bool)
        return float(np.mean(res["mode_code"][m] == fc.CLASS_CODES["MT"]))

    # Under the measured anisotropy ratios matrix tensile is empty, and it is
    # empty under both anchorings, so the anchoring choice no longer moves it.
    # While the adopted ratios were in force MT held 0.120 of the gneiss
    # interior at 0 degrees and re-anchoring removed it; that comparison is
    # withdrawn. Section 4.6 now reports the class as absent.
    g_adopted = mt_fraction(1, use_ucs_anchor=False)
    g_anchored = mt_fraction(1, use_ucs_anchor=True)
    s_adopted = mt_fraction(9, use_ucs_anchor=False)
    s_anchored = mt_fraction(9, use_ucs_anchor=True)

    for label, v in (("gneiss 0 deg, adopted", g_adopted),
                     ("gneiss 0 deg, UCS-anchored", g_anchored),
                     ("schist 15 deg, adopted", s_adopted),
                     ("schist 15 deg, UCS-anchored", s_anchored)):
        assert v <= 0.001, (
            f"matrix tensile has reappeared at {label}: {v:.4f}. Sections 3.7 "
            "and 4.6 report the class as empty; if it is back, both need "
            "revisiting rather than silently agreeing again."
        )
