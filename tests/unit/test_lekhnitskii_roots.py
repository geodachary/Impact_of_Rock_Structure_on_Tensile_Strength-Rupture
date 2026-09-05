"""The Lekhnitskii characteristic roots, and the isotropic limit they degenerate at.

Three claims the manuscript makes about Section 3.3 are checked here, because
two of them were wrong in an earlier state of the code and the third was
asserted in the text with nothing verifying it.

The isotropic limit is the substantive one. With s11 = s22 and
2*s12 + s66 = 2*s11 the quadratic in u = p^2 collapses to (u + 1)^2 = 0, a
repeated root at p = i. The two-potential representation needs two roots in
the upper half plane; the earlier root search found only one and fell back to
returning the conjugate pair (i, -i). With p2 = -i the stress relations give
sxx = -Re(F1'' + F2'') and syy = +Re(F1'' + F2''), so sxx = -syy identically,
for any coefficients whatever. The disc then reported a centre-stress ratio of
exactly -1 instead of the Brazilian -3, and did so silently, at every platen
arc width and every series truncation. Nothing failed; the field was simply
not the Brazilian solution.

That limit is never entered in production, because both lithologies are
strongly anisotropic, which is why the defect survived. It was reachable by
anyone applying the framework to a weakly anisotropic material.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.airy_solution import fit_orthotropic_airy_disk
from tools.lekhnitskii_root import lekh_roots_p
from tools.stress_helpers import eval_stress_field_material

# Disc geometry of the tested specimens.
R, T, P = 0.0255, 0.0226, 1.0e4

# The two production lithologies, as adopted in Table 2.
LITHOLOGIES = {
    "Augen gneiss":     (42.5e9, 20.8e9, 0.24, 18.1e9),
    "Psammitic schist": (25.0e9, 6.65e9, 0.15, 9.0e9),
}


def _centre_ratio(E1, E2, nu12, G12, alpha=0.0, **kw):
    """syy/sxx at the disc centre. The Brazilian value is -3."""
    fit = fit_orthotropic_airy_disk(E1, E2, nu12, G12, R, T, P, alpha, **kw)
    z = np.array([0.0])
    sxx, syy, _ = eval_stress_field_material(
        z, z, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
    return float(syy[0] / sxx[0])


def test_both_roots_lie_in_the_upper_half_plane():
    """A pair straddling the real axis is not a basis, isotropic case included."""
    cases = dict(LITHOLOGIES)
    E, nu = 40e9, 0.25
    cases["isotropic"] = (E, E, nu, E / (2 * (1 + nu)))
    cases["near-isotropic"] = (E, E / 1.001, nu, E / (2 * (1 + nu)))
    for name, (E1, E2, nu12, G12) in cases.items():
        p1, p2 = lekh_roots_p(E1, E2, nu12, G12)
        assert p1.imag > 0 and p2.imag > 0, f"{name}: got p1={p1}, p2={p2}"
        assert abs(p1 - p2) > 1e-9, f"{name}: roots not distinct"


def test_isotropic_limit_recovers_the_brazilian_centre_stress():
    """The check Section 3.3 claims: the isotropic limit is the Hondros field.

    The tolerance is 5%, not machine precision, because the platens are a
    finite arc rather than a line load. The classical -3 is the line-load
    value; the production 10 degree arc shifts it by 3.1%, and narrowing the
    arc to 2 degrees brings it back to 0.6%, which is what identifies the arc
    as the source of the offset rather than the degeneracy perturbation.
    """
    E, nu = 40e9, 0.25
    ratio = _centre_ratio(E, E, nu, E / (2 * (1 + nu)))
    assert ratio == pytest.approx(-3.0, rel=0.05), (
        f"isotropic limit gives syy/sxx = {ratio:.4f}, not the Brazilian -3. "
        "A value of exactly -1 means the degenerate root was not handled and "
        "the two potentials collapsed onto a conjugate pair.")


def test_the_platen_arc_is_what_offsets_the_isotropic_limit():
    """Section 3.3 attributes the offset to the arc and quotes both numbers.

    The attribution is the substantive part: an offset that did not shrink
    with the arc would mean the degeneracy perturbation, or the series
    truncation, was contaminating the limit. Nothing else checks the two
    percentages the text quotes, so they are pinned here.
    """
    E, nu = 40e9, 0.25
    wide = _centre_ratio(E, E, nu, E / (2 * (1 + nu)), beta_deg=10.0)
    narrow = _centre_ratio(E, E, nu, E / (2 * (1 + nu)), beta_deg=2.0)
    assert abs(wide + 3.0) / 3.0 == pytest.approx(0.031, abs=0.002), (
        f"the production arc now gives {wide:.4f}; Section 3.3 quotes 3.1%")
    assert abs(narrow + 3.0) / 3.0 == pytest.approx(0.006, abs=0.002), (
        f"the 2 degree arc now gives {narrow:.4f}; Section 3.3 quotes 0.6%")
    assert abs(narrow + 3.0) < abs(wide + 3.0), (
        "narrowing the platen arc no longer moves the limit toward -3, so the "
        "offset is not the arc and the explanation in Section 3.3 is wrong")


def test_isotropic_limit_is_not_the_degenerate_minus_one():
    """Pin the specific failure, which is exact and therefore unmistakable."""
    E, nu = 40e9, 0.25
    assert abs(_centre_ratio(E, E, nu, E / (2 * (1 + nu))) + 1.0) > 0.5


def test_the_limit_is_approached_continuously():
    """No discontinuity between the perturbed limit and genuine anisotropy."""
    E, nu = 40e9, 0.25
    ratios = [_centre_ratio(E, E / r, nu, E / (2 * (1 + nu)))
              for r in (1.0, 1.001, 1.01)]
    assert all(abs(x + 3.0) / 3.0 < 0.05 for x in ratios), ratios
    assert abs(ratios[0] - ratios[1]) < 0.05, (
        f"jump across the degenerate point: {ratios[0]:.4f} -> {ratios[1]:.4f}")


def test_roots_share_an_imaginary_part_and_differ_in_the_real_part():
    """Section 3.7 once said the imaginary parts were "well-separated".

    They are identical. The roots are +/-a + bi: what separates them is the
    real part, and what keeps them away from the degenerate isotropic root
    p = i is that the real part is bounded away from zero.
    """
    for name, consts in LITHOLOGIES.items():
        p1, p2 = lekh_roots_p(*consts)
        assert p1.imag == pytest.approx(p2.imag, abs=1e-12), name
        assert p1.real == pytest.approx(-p2.real, abs=1e-12), name
        assert abs(p1.real) > 0.4, f"{name}: real part too close to degeneracy"


def test_roots_do_not_depend_on_loading_orientation():
    """Orientation enters through the boundary tractions, not through the roots.

    The manuscript said the roots were solved for a compliance "rotated into
    the global frame" and were "updated for each loading orientation". They
    are not: the biquadratic form holds precisely because the compliance is
    the unrotated material-frame one, where s16 = s26 = 0.
    """
    for E1, E2, nu12, G12 in LITHOLOGIES.values():
        ref = lekh_roots_p(E1, E2, nu12, G12)
        for _ in range(3):
            assert lekh_roots_p(E1, E2, nu12, G12) == ref
    gneiss = lekh_roots_p(*LITHOLOGIES["Augen gneiss"])
    schist = lekh_roots_p(*LITHOLOGIES["Psammitic schist"])
    assert gneiss != schist, "roots must still differ between lithologies"


def test_production_roots_are_unchanged_by_the_degeneracy_handling():
    """The perturbation must not touch materials that are already separated."""
    expected = {
        "Augen gneiss":     (0.4976929465 + 1.0870747068j),
        "Psammitic schist": (0.5916197879 + 1.2605169028j),
    }
    for name, consts in LITHOLOGIES.items():
        p1, _ = lekh_roots_p(*consts)
        assert p1 == pytest.approx(expected[name], abs=1e-9), name


def test_tensile_cap_uses_the_foliation_normal_convention():
    """Delta is measured from the foliation normal, not the foliation.

    Both the closed-form cap and its two implementations once measured Delta
    from the foliation line and modulated with cos^2, which put the *strongest*
    material on the weak planes and made a crack running along a plane the
    hardest to drive. The appendix derivation requires the opposite: minimum
    resistance on an interface, intact value midway between interfaces.
    """
    from tools.analysis.mohr_coulomb_local import anisotropic_tensile_strength
    T0, spacing = 10.0, 0.002
    y = np.linspace(-0.02, 0.02, 2001)
    x = np.zeros_like(y)
    crack_vertical = np.pi / 2

    # planes horizontal: the crack crosses the array and must break the
    # matrix bridges, so the cap sweeps the whole range up to T0
    across = anisotropic_tensile_strength(T0, 0.0, crack_vertical,
                                          spacing=spacing, x=x, y=y, min_factor=0.0)
    assert across.max() == pytest.approx(T0, rel=1e-3)
    assert across.min() == pytest.approx(0.0, abs=1e-6)

    # planes vertical: the crack runs along a weak plane, so the cap vanishes
    along = anisotropic_tensile_strength(T0, np.pi / 2, crack_vertical,
                                         spacing=spacing, x=x, y=y, min_factor=0.0)
    assert np.allclose(along, 0.0, atol=1e-9), (
        "a crack following a weak plane must meet no tensile resistance; "
        "a nonzero cap here means Delta is being measured from the foliation "
        "rather than from its normal")
