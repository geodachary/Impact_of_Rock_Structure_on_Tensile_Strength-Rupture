"""Section 3's numbers, against the code constants and the convergence outputs.

Section 3 quotes the elastic identification, three convergence studies, the
isotropic-limit check, the stepper settings and the softening-cap sweep. Almost
none of it was guarded, and six statements had drifted from the data:

* the directional moduli E(theta) for both rocks, quoted from a superseded run;
* "rises monotonically" for the gneiss, which dips at 15 degrees;
* the shear-modulus range, described as measured when the table's column is
  E(theta)/(2(1+nu(theta))) at every row and so carries no independent content;
* the collocation deviation, quoted as 0.016% against a measured 0.015%;
* the series-order accuracy, quoted against an M=56 solution the study does not
  compute, and the M=80 boundary residual, quoted as 4.8e-3 against 2.6e-3;
* the sweep caption's standard deviations, which had the two lithologies
  swapped.

The constants are checked at their definition sites rather than by grepping for
a literal, because several appear as defaults that production overrides.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"

pytestmark = pytest.mark.skipif(
    not (TAB / "convergence_series_order.csv").is_file(),
    reason="convergence outputs absent")


# --- elastic identification -------------------------------------------------

def test_the_material_axis_constants_quoted_in_section_3():
    from tools.ddm._toolkit import ROCK_MATERIAL_AXES
    g = ROCK_MATERIAL_AXES["augen gneiss"]
    s = ROCK_MATERIAL_AXES["psammitic schist"]
    assert g["E1"] / g["E2"] == pytest.approx(1.39, abs=0.005)
    assert s["E1"] / s["E2"] == pytest.approx(0.86, abs=0.005)
    assert g["G12"] == pytest.approx(16.61, abs=0.01)
    assert s["G12"] == pytest.approx(6.53, abs=0.01)


def test_the_directional_moduli_quoted_in_section_3():
    d = pd.read_csv(REPO / "selected_all_samples.csv").dropna(how="all")
    E = {r: g.groupby("Angle").Modulus_of_Elasticity.mean()
         for r, g in d.groupby("Rock_type")}
    g, s = E["Augen gneiss"], E["Psammitic schist"]
    assert g.loc[0] == pytest.approx(38.4, abs=0.05)
    assert g.loc[15] == pytest.approx(37.3, abs=0.05)
    assert g.loc[90] == pytest.approx(53.4, abs=0.05)
    assert g.loc[15] < g.loc[0], (
        "the gneiss dip at 15 deg has gone; Section 3 no longer needs its "
        "'apart from a shallow dip' qualifier")
    assert s.loc[0] == pytest.approx(33.0, abs=0.05)
    assert s.loc[45] == pytest.approx(19.7, abs=0.05)
    assert s.loc[90] == pytest.approx(28.5, abs=0.05)
    assert s.max() / s.min() == pytest.approx(1.68, abs=0.005)


def test_the_shear_modulus_column_is_not_independent():
    """Section 3 says so explicitly; if it stopped being true, that goes."""
    d = pd.read_csv(REPO / "selected_all_samples.csv").dropna(how="all")
    iso = d.Modulus_of_Elasticity / (2.0 * (1.0 + d.Poisson_Ratio))
    assert (d.Shear_Modulus - iso).abs().max() < 1e-3


def test_the_lekhnitskii_root_separations():
    from tools.ddm._toolkit import ROCK_MATERIAL_AXES
    for key, want in (("augen gneiss", 0.66), ("psammitic schist", 1.46)):
        v = ROCK_MATERIAL_AXES[key]
        c = [1 / v["E1"], 0.0,
             (2 * (-v["nu12"] / v["E1"]) + 1 / v["G12"]), 0.0, 1 / v["E2"]]
        up = sorted([z for z in np.roots(c) if z.imag > 0], key=lambda z: z.imag)
        assert abs(up[0] - up[1]) == pytest.approx(want, abs=0.005)


# --- convergence ------------------------------------------------------------

def test_the_series_order_convergence_numbers():
    so = pd.read_csv(TAB / "convergence_series_order.csv")
    b = so.groupby("series_order").boundary_residual_rel.mean()
    assert b[24] == pytest.approx(6.6e-2, rel=0.02)
    assert b[48] == pytest.approx(6.3e-3, rel=0.02)
    assert b[64] == pytest.approx(5.0e-3, rel=0.03)
    assert b[80] == pytest.approx(2.6e-3, rel=0.03), (
        "Section 3 quotes the M=80 residual; it must not drift back to the "
        "M=64 value, which is what it used to quote")
    core = so[so.series_order == 48].rel_error_core.max() * 100
    assert core == pytest.approx(0.27, abs=0.02), (
        f"core field at M=48 is now {core:.2f}% of the M=96 reference")


def test_the_collocation_convergence_number():
    co = pd.read_csv(TAB / "convergence_collocation.csv")
    prod = co[co.n_boundary == 420].rel_error.max() * 100
    assert prod == pytest.approx(0.0145, abs=0.0005), (
        f"largest deviation at the production 420 points is {prod:.4f}%")


def test_the_production_discretisation_is_what_section_3_says():
    import scripts.make_mesh_sensitivity_figure as m
    assert m.PRODUCTION_NBD == (420, 900)
    assert m.SERIES_REFERENCE == 96
    from tools import airy_solution as A
    assert A.DEFAULT_M == 48
    assert A.DEFAULT_LAM == 1e-18


def test_the_grid_counts_quoted_in_section_3():
    gr = pd.read_csv(TAB / "convergence_grid.csv")
    assert 20081 in set(gr.n_interior)
    assert round(0.85 ** 2 * 20081, -2) == 14500.0


def test_the_isotropic_limit_offsets():
    import sys
    sys.path.insert(0, str(REPO / "tests" / "unit"))
    from test_lekhnitskii_roots import _centre_ratio
    E, nu = 40.0, 0.25
    G = E / (2 * (1 + nu))
    wide = _centre_ratio(E, E, nu, G, beta_deg=10.0)
    narrow = _centre_ratio(E, E, nu, G, beta_deg=2.0)
    assert 100 * abs(wide + 3) / 3 == pytest.approx(3.1, abs=0.1)
    assert 100 * abs(narrow + 3) / 3 == pytest.approx(0.6, abs=0.1)


# --- model constants --------------------------------------------------------

def test_the_classifier_and_stepper_constants():
    from tools import strain_partitioning as sp
    assert (sp.WEAK_T_RATIO, sp.WEAK_C_RATIO) == (0.35, 0.60)
    assert sp.ACTIVATION_FLOOR == 0.05
    assert sp.THRESHOLD == 1.0
    assert sp.ETA_MIX == 0.90
    assert sp.N_THETA == 361
    from tools.analysis import crack_energy_suite as ce
    assert ce.MAX_RIM_EXTENSION_FRAC == 0.02


def test_the_directional_toughness_reduction_is_one_minus_the_strength_ratio():
    """Section 3 states w_r = 1 - T_wp/T_0 = 0.65.

    The function default is 0.35, which is the strength ratio and not the
    reduction; production resolves w_r from the ratio instead of taking that
    default, so the resolution is what is checked.
    """
    from tools import strain_partitioning as sp
    assert 1.0 - sp.WEAK_T_RATIO == pytest.approx(0.65)
    from tools.crack_helpers import Gc_theta_weak_plane
    import inspect
    assert inspect.signature(Gc_theta_weak_plane).parameters["eta_deg"].default == 10.0
    assert inspect.signature(Gc_theta_weak_plane).parameters["Gc_matrix"].default == 1.0
    # along the plane the toughness drops to (1 - w_r) of the matrix value
    along = Gc_theta_weak_plane(0.0, 0.0, 1.0, weak_reduction=0.65, eta_deg=10.0)
    assert along == pytest.approx(0.35, abs=1e-9)


def test_the_permutation_offsets():
    from tools.strain_partitioning import REGIME_OFFSETS_MPA
    assert REGIME_OFFSETS_MPA == {"Thrust": -0.50, "Strike-Slip": 0.00,
                                  "Extensional": +0.50}


# --- softening-cap sweep ----------------------------------------------------

def test_the_softening_cap_sweep_as_section_3_describes_it():
    d = pd.read_csv(TAB / "kmax_softening_sweep.csv").sort_values("k_C")
    assert len(d) == 18
    assert (d.k_C.min(), d.k_C.max()) == (0.05, 0.90)
    assert np.allclose(d.k_T / d.k_C, 0.35)
    g, s = d.mixed_fraction_augen_gneiss, d.mixed_fraction_psammitic_schist
    assert g.min() == pytest.approx(0.027, abs=0.001)
    assert g.max() == pytest.approx(0.085, abs=0.001)
    assert not (np.diff(g.values) > 0).all(), (
        "the gneiss sweep is now strictly monotonic; Section 3 describes a "
        "plateau between k_C = 0.15 and 0.25 and would need updating")
    assert s.min() == pytest.approx(0.115, abs=0.001)
    assert s.max() == pytest.approx(0.150, abs=0.001)
    assert d.loc[s.idxmax(), "k_C"] == pytest.approx(0.15)
    assert d[s >= 0.98 * s.max()].k_C.min() == pytest.approx(0.10)
    # Section 3 states these as upper bounds ("moves by at most"), so they are
    # checked as bounds, and kept tight enough that a real drift still trips.
    assert 0.05 < g.max() - g.min() <= 0.06
    assert 0.03 < s.max() - s.min() <= 0.04


def test_the_sweep_standard_deviations_are_not_swapped():
    """They were. The caption gave the schist's range for the gneiss."""
    d = pd.read_csv(TAB / "kmax_softening_sweep.csv")
    assert d.sd_augen_gneiss.min() == pytest.approx(0.027, abs=0.001)
    assert d.sd_augen_gneiss.max() == pytest.approx(0.081, abs=0.001)
    assert d.sd_psammitic_schist.min() == pytest.approx(0.147, abs=0.001)
    assert d.sd_psammitic_schist.max() == pytest.approx(0.168, abs=0.001)
    assert (d.sd_psammitic_schist > d.mixed_fraction_psammitic_schist).all(), (
        "the schist standard deviation no longer exceeds its mean at every "
        "cap value, which is what the caption asserts")
