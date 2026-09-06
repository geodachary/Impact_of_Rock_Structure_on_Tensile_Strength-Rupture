"""Section 2.6's representative compression values, and how they were obtained.

The six quoted numbers for the psammitic schist at 0 degrees all come from the
replicate file and are exact. The provenance around them was not.

The subsection said the augen gneiss moduli were "estimated from compression
curves on companion cores and checked against published ranges". They are
measured per specimen exactly as the schist's are: 30 replicates carrying 29
distinct values of E, with real within-angle scatter. Table 5 lists them under
"Measured directly", so the two statements disagreed, and the weaker one was
in the methods where a reader looks for provenance.

It also called E and nu "direction-averaged values for the 0 degree
configuration". They are averaged over the three replicates at one direction,
not over directions, and each specimen uses its own pair rather than a shared
reference.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "selected_all_samples.csv"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"

pytestmark = pytest.mark.skipif(not RAW.is_file(), reason="replicate data absent")


@pytest.fixture(scope="module")
def schist0():
    d = pd.read_csv(RAW)
    return d[(d.Rock_type == "Psammitic schist") & (d.Angle == 0)]


def test_the_representative_envelope_and_moduli(schist0):
    assert len(schist0) == 3, "the 0 degree schist no longer has three replicates"
    want = {"Cohesion": 11.94, "Friction_Angle": 28.09,
            "Modulus_of_Elasticity": 33.00, "Poisson_Ratio": 0.23,
            "UCS_(Mpa)": 42.51, "Axial_strain": 0.0013}
    for col, v in want.items():
        got = float(schist0[col].mean())
        assert round(got, 4) == pytest.approx(v, abs=0.006), (
            f"{col}: measurements give {got:.4f}, the text quotes {v}")


def test_shear_strength_is_one_pair_per_lithology_and_angle():
    """The claim that each fabric angle carries its own c and phi."""
    d = pd.read_csv(RAW)
    for rock, g in d.groupby("Rock_type"):
        for col in ("Cohesion", "Friction_Angle"):
            sd = g.groupby("Angle")[col].std().fillna(0.0)
            assert (sd == 0).all(), (
                f"{rock}: {col} now varies within a fabric angle, so it is no "
                "longer one triaxial fit per lithology-angle pair")
            assert g.groupby("Angle")[col].mean().nunique() > 1, (
                f"{rock}: {col} is the same at every angle, so it is not "
                "resolved per fabric angle as Section 2.6 states")


def test_both_lithologies_moduli_are_measured_per_specimen():
    d = pd.read_csv(RAW)
    for rock, g in d.groupby("Rock_type"):
        # 17 distinct values over 22 schist replicates after the data
        # revision, which is still clearly a per-specimen measurement
        # rather than one value repeated per angle.
        assert g.Modulus_of_Elasticity.nunique() > 0.7 * len(g), (
            f"{rock}: E now takes {g.Modulus_of_Elasticity.nunique()} distinct "
            f"values over {len(g)} replicates; per-specimen measurement is what "
            "Section 2.6 and Table 5 both claim")
        assert g.groupby("Angle").Modulus_of_Elasticity.std().fillna(0).max() > 0, (
            f"{rock}: no within-angle scatter in E")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_gneiss_moduli_are_not_described_as_estimated():
    t = TEX.read_text(encoding="utf-8")
    assert "we estimated $E$ and" not in t.replace("\n", " "), (
        "Section 2.6 describes the gneiss moduli as estimated again; they are "
        "measured per specimen, and Table 5 lists them as measured directly")
    assert "direction-averaged values for the $0^\\circ$ configuration" not in t, (
        "the replicate mean is described as direction-averaged again")
