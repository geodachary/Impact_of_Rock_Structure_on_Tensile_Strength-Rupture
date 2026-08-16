"""The inclusion bound must reproduce the cases where the answer is known.

This estimate is quoted in the manuscript as the reason a heterogeneous elastic
solution was not built, so it has to be right in the limits that can be checked
by hand rather than merely plausible. Each test below is a closed-form result:
the Eshelby tensor of a circle, the no-contrast identity, the vanishing stress in
a void, and the rigid-inclusion factor, which for a circular inclusion follows
from ``sigma_I = C_m S^-1 C_m^-1 sigma_inf`` and equals 3/2 at nu = 0.25.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import eshelby_bound as eb

E_M = 30.0e3
NU = 0.25
UNIAXIAL = np.array([1.0, 0.0, 0.0])


@pytest.mark.parametrize("nu", [0.15, 0.25, 0.35])
def test_circular_eshelby_tensor_matches_closed_form(nu):
    S = eb.eshelby_ellipse(1.0, nu)
    assert S[0, 0] == pytest.approx((5.0 - 4.0 * nu) / (8.0 * (1.0 - nu)), rel=1e-12)
    assert S[0, 1] == pytest.approx((4.0 * nu - 1.0) / (8.0 * (1.0 - nu)), rel=1e-12, abs=1e-15)
    assert S[2, 2] == pytest.approx(2.0 * (3.0 - 4.0 * nu) / (8.0 * (1.0 - nu)), rel=1e-12)


def test_no_contrast_leaves_the_field_untouched():
    A = eb.strain_concentration(E_M, NU, E_M, NU, 1.0)
    assert np.allclose(A, np.eye(3), atol=1e-12)


def test_void_carries_no_stress():
    s = eb.interior_stress(E_M, NU, E_M * 1e-9, NU, 1.0, UNIAXIAL)
    assert np.max(np.abs(s)) < 1e-6


def test_rigid_circular_inclusion_reaches_the_analytic_factor():
    """sigma_I -> C_m S^-1 C_m^-1 sigma_inf as C_i -> infinity; 3/2 at nu = 0.25."""
    got = eb.concentration_factor(E_M, NU, E_M * 1e8, NU, 1.0, UNIAXIAL)
    assert got == pytest.approx(1.5, abs=1e-3)


def test_stiffer_inclusion_carries_more_stress():
    prev = 0.0
    for c in (1.5, 2.0, 2.5, 3.0):
        got = eb.concentration_factor(E_M, NU, E_M * c, NU, 1.0, UNIAXIAL)
        assert got > prev
        prev = got


@pytest.mark.parametrize("f,expected", [(0.0, "matrix"), (1.0, "inclusion")])
def test_mori_tanaka_recovers_both_end_members(f, expected):
    C = eb.mori_tanaka_modulus(E_M, NU, 3.0 * E_M, NU, 1.0, f)
    target = eb.stiffness_plane_strain(E_M if expected == "matrix" else 3.0 * E_M, NU)
    assert np.allclose(C, target, rtol=1e-9, atol=1e-6)


def test_every_entry_lies_below_its_own_rigid_limit():
    """1.74 exceeds the circular limit of 1.50 because it is a 3:1 ellipse.

    The caption previously placed those two numbers side by side without saying
    so, which reads as a violated bound. Each shape has its own limit and every
    swept entry must sit under the one for its own aspect ratio.
    """
    s_inf = np.array([4.00, -20.69, 0.0])
    for ar in (1.0, 2.0, 3.0):
        limit = eb.concentration_factor(E_M, NU, E_M * 1e8, NU, ar, s_inf)
        for c in (1.5, 2.0, 2.5, 3.0):
            got = eb.concentration_factor(E_M, NU, E_M * c, NU, ar, s_inf)
            assert got < limit, f"aspect {ar}:1 contrast {c}: {got} exceeds limit {limit}"
    assert eb.concentration_factor(E_M, NU, E_M * 1e8, NU, 3.0, s_inf) > 2.5


def test_bound_is_insensitive_to_the_matrix_poisson_ratio():
    """The sweep fixes nu at 0.25 while the specimen measures 0.26."""
    s_inf = np.array([4.00, -20.69, 0.0])
    a = eb.bound_table(E_M, 0.25, 0.25, [1.5, 2, 2.5, 3], [1, 2, 3], s_inf)
    b = eb.bound_table(E_M, 0.26, 0.26, [1.5, 2, 2.5, 3], [1, 2, 3], s_inf)
    assert abs(a.concentration.max() - b.concentration.max()) < 0.01


def test_quoted_bounds_are_what_the_module_produces():
    """The numbers in the limitations paragraph must come from this code."""
    tab = eb.bound_table(E_M, NU, NU, [1.5, 2.0, 2.5, 3.0], [1.0, 2.0, 3.0],
                         np.array([4.00, -20.69, 0.0]))
    assert tab.concentration.min() == pytest.approx(1.12, abs=0.01)
    assert tab.concentration.max() == pytest.approx(1.74, abs=0.01)
    assert eb.decay_radius(0.10) == pytest.approx(3.162, abs=0.01)
