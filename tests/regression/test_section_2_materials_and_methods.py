"""Section 2's measured values, against the shipped specimen table.

Section 2 quotes specimen geometry, Table 1's petrophysical means, and the
elastic and shear-strength constants for the psammitic schist at 0 degrees.
None of it was guarded, and one pair had drifted: the compression caption
described a peak near 41 MPa at an axial strain of about 1.2e-3, while the
three archived specimens peak at 41.6 to 43.6 MPa at strains of 1.26 to
1.32e-3. Every measured number in the section is checked here against
``selected_all_samples.csv``, which is the source for all of them.

Two Section 2 statements are deliberately not tested, because the release
carries no data for them: the 0.5-1 MPa/s loading rate, and the 2.0-2.1
height-to-diameter ratio of the compression cylinders. Both are protocol,
and the cylinders are not in the archived table.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "selected_all_samples.csv"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"

pytestmark = pytest.mark.skipif(not (CSV.is_file() and TEX.is_file()),
                                reason="specimen table or manuscript absent")


@pytest.fixture(scope="module")
def d():
    return pd.read_csv(CSV).dropna(how="all")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def test_disc_geometry_matches_the_table(d):
    D = d.Diameter_mm.astype(float)
    t = d.Thickness_mm.astype(float)
    assert set(D.unique()) == {51.0}, "the discs are no longer all 51 mm"
    assert (t.min(), t.max()) == (20.0, 27.0)
    assert t.mean() == pytest.approx(22.6, abs=0.05)
    assert t.std(ddof=1) == pytest.approx(1.8, abs=0.05)
    assert (t / D).min() == pytest.approx(0.39, abs=0.005)
    assert (t / D).max() == pytest.approx(0.53, abs=0.005)


def test_the_test_matrix_is_seven_angles_and_the_stated_replicate_counts(d):
    for rock, n in (("Augen gneiss", 30), ("Psammitic schist", 22)):
        g = d[d.Rock_type == rock]
        assert sorted(g.Angle.astype(int).unique()) == [0, 15, 30, 45, 60, 75, 90]
        assert len(g) == n, (
            f"{rock} now has {len(g)} replicates; Section 2 says "
            "thirty against twenty-two")


def test_equation_1_reproduces_the_recorded_tensile_strengths(d):
    """sigma_t = 2P / (pi D t), the ISRM form quoted in Section 2."""
    import numpy as np
    P = d["Load_(KN)"].astype(float) * 1e3
    D = d.Diameter_mm.astype(float) * 1e-3
    t = d.Thickness_mm.astype(float) * 1e-3
    calc = 2.0 * P / (np.pi * D * t) / 1e6
    assert (calc - d.Tensile_strength_Mpa.astype(float)).abs().max() < 0.01


def test_every_lithology_angle_pair_has_one_shear_strength_pair(d):
    """Section 2 states each pair carries its own cohesion and friction angle."""
    for (rock, ang), g in d.groupby(["Rock_type", "Angle"]):
        assert g.Cohesion.nunique() == 1 and g.Friction_Angle.nunique() == 1, (
            f"{rock} at {ang:.0f} deg carries more than one strength pair")
    assert len(d.groupby(["Rock_type", "Angle"])) == 14


def test_the_schist_zero_degree_constants_quoted_in_section_2(d, tex):
    s0 = d[(d.Rock_type == "Psammitic schist") & (d.Angle == 0)]
    assert float(s0.Cohesion.iloc[0]) == pytest.approx(11.94, abs=0.005)
    assert float(s0.Friction_Angle.iloc[0]) == pytest.approx(28.09, abs=0.005)
    assert s0.Modulus_of_Elasticity.astype(float).mean() == pytest.approx(33.00, abs=0.005)
    assert s0.Poisson_Ratio.astype(float).mean() == pytest.approx(0.23, abs=0.005)
    for q in (r"c = 11\.94", r"\\phi = 28\.09", r"E = 33\.00", r"\\nu = 0\.23"):
        assert re.search(q, tex), f"Section 2 no longer quotes {q}"


def test_the_compression_caption_matches_the_archived_specimens(d, tex):
    """The pair that had drifted. Both halves come from the same three rows."""
    s0 = d[(d.Rock_type == "Psammitic schist") & (d.Angle == 0)]
    ucs = s0["UCS_(Mpa)"].astype(float)
    eps = s0.Axial_strain.astype(float) * 1e3
    assert ucs.min() == pytest.approx(41.6, abs=0.05)
    assert ucs.max() == pytest.approx(43.6, abs=0.05)
    assert eps.min() == pytest.approx(1.26, abs=0.005)
    assert eps.max() == pytest.approx(1.32, abs=0.005)
    m = re.search(r"peaks of\s*\n?\s*\$([\d.]+)\$--\$([\d.]+)\$~MPa, reached at axial "
                  r"strains of \$([\d.]+)\$--\$([\d.]+)\\times10\^\{-3\}\$", tex)
    assert m, "the compression caption sentence has changed shape"
    assert float(m.group(1)) == pytest.approx(ucs.min(), abs=0.05)
    assert float(m.group(2)) == pytest.approx(ucs.max(), abs=0.05)
    assert float(m.group(3)) == pytest.approx(eps.min(), abs=0.005)
    assert float(m.group(4)) == pytest.approx(eps.max(), abs=0.005)


def test_table_1_petrophysical_means(d, tex):
    want = {"Density": (2.78, 2.85), "Porosity_percent": (1.65, 1.17),
            "UCS_(Mpa)": (36.62, 53.62), "Modulus_of_Elasticity": (25.24, 43.00),
            "Poisson_Ratio": (0.18, 0.24)}
    for col, (schist, gneiss) in want.items():
        for rock, quoted in (("Psammitic schist", schist), ("Augen gneiss", gneiss)):
            got = d[d.Rock_type == rock][col].astype(float).mean()
            assert got == pytest.approx(quoted, abs=0.006), (
                f"Table 1 quotes {quoted} for {rock} {col}; the data give {got:.3f}")


def test_the_foliation_spacings_match_the_lithology_constants():
    from tools import lithology as lith
    assert lith.AUGEN_GNEISS.spacing_m == pytest.approx(0.010)
    assert lith.PSAMMITIC_SCHIST.spacing_m == pytest.approx(0.002)


def test_the_ati_fit_window_and_interval_method():
    from tools import ati_model
    assert ati_model.BETA_BOUNDS == (30.0, 89.0), (
        "Section 2 quotes a search window of [30, 89] degrees")
    assert "95%" in ati_model.profile_interval.__doc__
