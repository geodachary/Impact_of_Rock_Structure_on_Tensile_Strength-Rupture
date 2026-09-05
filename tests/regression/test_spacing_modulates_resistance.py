"""The spacing must appear in the resistance, never in the elastic field.

The stress-profile panel was previously built from a uniform placeholder field
multiplied by a periodic strength factor and plotted on a stress axis, which is
what made it look as though the maximum principal stress carried the fabric
spacing. It cannot: the Lekhnitskii solution is homogeneous orthotropic and its
equilibrium and compatibility conditions contain no term in ``s``.

These tests pin both halves of the statement -- that sigma_1 is free of periodic
content, and that the tensile utility carries it at the expected 2R/s count --
so the two quantities cannot be conflated again.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from tools import fabric_tractions as ft

SPACING = {"Augen gneiss": 0.010, "Psammitic schist": 0.002}
FIELDS = ft.field_files()
needs_fields = pytest.mark.skipif(not FIELDS, reason="fields not exported yet")


def _zero_degree_fields():
    out = {}
    for p in FIELDS:
        f = ft.load_field(p)
        if abs(f["angle_deg"]) < 1e-9:
            out[f["rock"]] = f
    return out


def _profile(f, key):
    X, Y, M = f["X"], f["Y"], f["M"].astype(bool)
    i = int(np.argmin(np.abs(X[0, :])))
    col = M[:, i]
    return np.asarray(f[key][:, i][col], float)


def _n_maxima(v):
    rng = float(np.nanmax(v) - np.nanmin(v))
    pk, _ = find_peaks(v, prominence=0.02 * (rng + 1e-12))
    return len(pk)


@needs_fields
@pytest.mark.parametrize("rock", list(SPACING))
def test_principal_stress_carries_no_spacing_signature(rock):
    f = _zero_degree_fields().get(rock)
    if f is None:
        pytest.skip("no 0 degree field for this lithology")
    n = _n_maxima(_profile(f, "s1"))
    assert n <= 2, f"{rock}: sigma_1 shows {n} maxima; the elastic field is homogeneous"


@needs_fields
@pytest.mark.parametrize("rock", list(SPACING))
def test_weak_plane_weight_carries_the_spacing(rock):
    """The spacing enters the resistance through the proximity weight.

    This previously read ``Rt_eff`` at 0 degrees and expected 2R/s maxima. It
    got them, but only because the stress field was being sign-flipped before
    classification: the flip turned the platen compression clamping the
    horizontal foliation into apparent tension, so the periodic weight
    modulated a weak-plane tensile term that should have been identically
    zero. With the convention corrected, ``Rt_eff`` at 0 degrees is the matrix
    term sigma_1 / T_m, which is smooth, and one maximum is the right answer.

    The claim itself is unchanged and is checked here on the quantity that
    actually carries it.
    """
    f = _zero_degree_fields().get(rock)
    if f is None:
        pytest.skip("no 0 degree field for this lithology")
    expected = 2.0 * float(f["R_m"]) / SPACING[rock]
    n = _n_maxima(_profile(f, "wp_weight"))
    assert abs(n - expected) <= max(2.0, 0.1 * expected), (
        f"{rock}: weak-plane weight shows {n} maxima against {expected:.0f} "
        "expected from 2R/s")


@needs_fields
@pytest.mark.parametrize("rock", list(SPACING))
def test_clamped_foliation_carries_no_tensile_utility(rock):
    """At 0 degrees the planes are shut, so the weak-plane tensile term is flat.

    This is the other half of the correction above: a periodic signature in
    the tensile utility at 0 degrees would mean the clamping had been lost.
    """
    f = _zero_degree_fields().get(rock)
    if f is None:
        pytest.skip("no 0 degree field for this lithology")
    n = _n_maxima(_profile(f, "Rt_eff"))
    assert n <= 2, (
        f"{rock}: R_t shows {n} maxima at 0 degrees. The foliation is "
        "clamped there, so the tensile utility is the smooth matrix term; "
        "periodic content means the sign convention has been lost again.")
