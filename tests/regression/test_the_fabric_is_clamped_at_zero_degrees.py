"""At 0 degrees the foliation is held shut, and the paper must say so.

This is stated correctly in the methods, where the anisotropic tensile cap is
motivated, and it is what the tractions and the four-class fractions both
show: the mean normal traction on the foliation is about -22.6 MPa in the
gneiss and -20.2 MPa in the schist, the open fraction is exactly zero, and no
point in either specimen is classified as weak-plane tensile.

The caption of the ruptured-specimen figure said the opposite, that near-axial
splitting at 0 degrees is "consistent with loading perpendicular to foliation
maximizing tensile normal stress on the weakest planes". Loading perpendicular
to the foliation maximizes the *compressive* normal stress on it. The
conclusion drawn, axial splitting, is right; the mechanism given for it was
inverted, and it contradicted the methods section on the same point.

That went unnoticed because no result is computed from a caption. This test
asserts the physics and then checks the text against it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
TRACTIONS = REPO / "outputs" / "tables" / "fabric_tractions.csv"
FOURCLASS = REPO / "outputs" / "tables" / "fourclass_area_fractions.csv"


@pytest.mark.skipif(not TRACTIONS.is_file(), reason="tractions not exported yet")
def test_the_normal_traction_is_compressive_and_nothing_opens():
    d = pd.read_csv(TRACTIONS)
    at0 = d[d.angle_deg == 0]
    assert len(at0) == 2, "expected one row per lithology at 0 degrees"
    for _, r in at0.iterrows():
        assert r.sigma_n_mean_MPa < -10.0, (
            f"{r.rock}: mean normal traction at 0 deg is "
            f"{r.sigma_n_mean_MPa:.2f} MPa, no longer firmly compressive")
        assert r.open_fraction == 0.0, (
            f"{r.rock}: {r.open_fraction:.3f} of the interior has the foliation "
            "in tension at 0 deg; it should be none")


@pytest.mark.skipif(not FOURCLASS.is_file(), reason="fractions not exported yet")
def test_no_weak_plane_opening_occurs_at_zero_degrees():
    d = pd.read_csv(FOURCLASS)
    at0 = d[d.angle_deg == 0]
    assert (at0.WT == 0).all(), (
        f"weak-plane tensile opening is now nonzero at 0 deg: "
        f"{at0[['rock', 'WT']].to_dict('records')}. Splitting there has to "
        "break the matrix, because the planes are clamped shut.")
    assert (at0[["MT", "MS"]].sum(axis=1) > 0).all(), (
        "failure at 0 deg is no longer matrix-hosted, which is the other half "
        "of the same claim")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_text_does_not_claim_tension_on_the_planes_at_zero_degrees():
    t = TEX.read_text(encoding="utf-8")
    assert "maximizing\n\ttensile normal stress on the weakest planes" not in t, \
        "the inverted mechanism has returned to the ruptured-specimen caption"
    assert "maximizing tensile normal stress" not in t.replace("\n\t", " "), \
        "the inverted mechanism has returned somewhere in the manuscript"
