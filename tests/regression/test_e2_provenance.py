"""E2 is derived from the measured end-member modulus ratio, and stays that way.

``E1``, ``nu12`` and ``G12`` are read per specimen from the measurement file.
``E2`` is not: it is ``E1 / A_E``. ``A_E`` was for a long time a pair of
hardcoded constants, 2.037 and 3.763, that no source in the repository, no
paper in the bibliography, and no combination of the measured quantities could
reproduce. They are now computed from the replicate table as

    A_E = mean E(alpha=90) / mean E(alpha=0)

with alpha the fabric angle defined in the manuscript: alpha = 0 is foliation
normal to loading, alpha = 90 is foliation parallel. That is the ratio the
transversely isotropic E(theta) relation forces, since it passes exactly
through both end members.

This file pins that the derivation stays a derivation. The failure it exists
to catch is someone reintroducing a constant, because the compliance contrast
is the quantity Section 5.3 credits with the difference in localization
between the two rocks, and a silently reinstated magic number would change
every field in the paper.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import lithology as lith
from tools.ddm._toolkit import ROCK_ANISO_RATIO, get_anisotropy_ratio

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
RAW = REPO / "selected_all_samples.csv"

#: rock label in the measurement file -> key in ROCK_ANISO_RATIO
ROCKS = {"Augen gneiss": "augen gneiss", "Psammitic schist": "psammitic schist"}


def _measured_ratio(rock):
    """E(90)/E(0) computed from the replicate file, independently of the code.

    The production helper reads the fourteen-row specimen table through
    ``data_io``. This recomputes the same quantity from the raw replicates, so
    the test fails if those two tables ever stop agreeing rather than passing
    because both were regenerated from the same mistake.
    """
    d = pd.read_csv(RAW)
    d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
    per_angle = d[d.Rock_type == rock].groupby("Angle")["Modulus_of_Elasticity"].mean()
    return float(per_angle.loc[90] / per_angle.loc[0])


def test_no_measurement_of_e2_exists():
    """E2 is still derived, not a column someone added."""
    cols = pd.read_csv(RAW, nrows=1).columns
    assert not any("E2" in c or "modulus_2" in c.lower() for c in cols), (
        "an E2 column has appeared in the measurements; if it is now measured "
        "directly, use it instead of deriving E2 from the ratio")


@pytest.mark.parametrize("rock", sorted(ROCKS))
def test_the_ratio_is_the_measured_end_member_ratio(rock):
    """The whole point: no constant, the number comes from the data."""
    got = get_anisotropy_ratio(rock)
    want = _measured_ratio(rock)
    assert got == pytest.approx(want, rel=1e-12), (
        f"{rock}: the anisotropy ratio in force is {got:.6f} but the "
        f"measurements give {want:.6f}")


def test_the_old_hardcoded_constants_are_gone():
    """2.037 and 3.763 had no source. They must not come back."""
    for key, value in ROCK_ANISO_RATIO.items():
        for gone in (2.037, 3.763):
            assert abs(value - gone) > 1e-6, (
                f"{key} is back to the unsourced constant {gone}. The ratio is "
                "derived from the replicate table; see _toolkit.")
    # Mentioning them in the provenance note is fine; assigning them is not.
    code = [ln for ln in
            (REPO / "tools" / "ddm" / "_toolkit.py").read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    for gone in ("2.037", "3.763"):
        offenders = [ln.strip() for ln in code if gone in ln]
        assert not offenders, (
            f"_toolkit assigns the retired constant {gone}: {offenders[:2]}")


def test_the_schist_ratio_is_below_one():
    """Not a typo, and the code must tolerate it.

    The schist's apparent moduli make it stiffer across the foliation than
    along it, and its E(theta) minimum sits at 45 degrees, so its modulus
    anisotropy is carried by G12 rather than by an E1/E2 contrast. Several
    places would be tempted to assume ratio > 1; this records that the data
    does not.
    """
    assert get_anisotropy_ratio("Psammitic schist") < 1.0
    assert get_anisotropy_ratio("Augen gneiss") > 1.0


@pytest.mark.skipif(not all(lith.field_cache_path(s).exists() for s in (1, 8)),
                    reason="fields not exported")
@pytest.mark.parametrize("sid,rock", [(1, "Augen gneiss"), (8, "Psammitic schist")])
def test_the_exported_fields_used_that_ratio(sid, rock):
    """The archives must have been built with the ratio now in force."""
    d = np.load(lith.field_cache_path(sid), allow_pickle=True)
    ratio = float(d["E1_MPa"]) / float(d["E2_MPa"])
    assert ratio == pytest.approx(get_anisotropy_ratio(rock), rel=1e-6), (
        f"specimen {sid}: the stored field has E1/E2 = {ratio:.6f}, the ratio "
        f"in force is {get_anisotropy_ratio(rock):.6f}. Re-export the fields.")


def test_every_copy_of_the_ratio_agrees_with_the_canonical_one():
    """Eight modules define a shadow dict; none is read, all must still agree."""
    pat = re.compile(r"ROCK_ANISO_RATIO = \{(.*?)\}", re.S)
    found = {}
    for p in sorted((REPO / "tools").rglob("*.py")):
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            pairs = re.findall(r'"([^"]+)":\s*([\d.]+)', m.group(1))
            if pairs:
                found[str(p.relative_to(REPO))] = {k: float(v) for k, v in pairs}
    for path, d in found.items():
        for key, value in d.items():
            assert value == pytest.approx(get_anisotropy_ratio(key), rel=1e-3), (
                f"{path} hardcodes {key}={value}, the ratio in force is "
                f"{get_anisotropy_ratio(key):.4f}. Only the canonical one is "
                "read, so this edit would silently do nothing.")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_manuscript_quotes_the_ratio_in_force():
    """Every ratio the paper states must be the one the code used."""
    t = TEX.read_text(encoding="utf-8")
    m = re.search(r"anisotropy ratios \$E_1/E_2\$ are \$([\d.]+)\$ \(gneiss\) and\s*\n?"
                  r"\$([\d.]+)\$ \(schist\)", t)
    assert m, "the sentence stating both anisotropy ratios has changed shape"
    for said, rock in ((m.group(1), "Augen gneiss"), (m.group(2), "Psammitic schist")):
        want = round(get_anisotropy_ratio(rock), 2)
        assert float(said) == want, (
            f"{rock}: the paper says {said}, the ratio in force is {want}")

    # Table 5 quotes the same pair; it drifted once already.
    m = re.search(r"A_E = E\(90\^\\circ\)/E\(0\^\\circ\) = ([\d.]+)\$ \(gneiss\), "
                  r"\$([\d.]+)\$ \(schist\)", t)
    assert m, "the Table 5 A_E cell has changed shape"
    for said, rock in ((m.group(1), "Augen gneiss"), (m.group(2), "Psammitic schist")):
        want = round(get_anisotropy_ratio(rock), 2)
        assert float(said) == want, (
            f"Table 5 says A_E = {said} for the {rock}, the ratio in force is "
            f"{want}")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_parameter_table_identifies_e2_with_the_zero_degree_specimen():
    """E2 is the foliation-normal modulus, measured at 0 degrees.

    It used to be derived as E1/A_E from a per-specimen E1, which is why the
    table carried a note saying no independent measurement existed. Under the
    material-axis formulation the 0 degree test loads across the foliation and
    measures E2 directly, so the note would now be wrong.
    """
    t = TEX.read_text(encoding="utf-8")
    assert "read from the $0^\\circ$ specimen" in t, (
        "Table 5 no longer identifies E2 with the 0 degree specimen")
    assert "no\nindependent measurement of $E_2$ was made" not in t, (
        "Table 5 still carries the note that E2 was never measured; under the "
        "material-axis formulation it is the 0 degree modulus")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_methods_prose_does_not_call_e2_laboratory_derived():
    """Section 3.2 said the laboratory moduli initialize E1, E2 and nu12.

    E2 is derived from E1 and the ratio, not read from the table, and the
    sentence has to keep saying so.
    """
    t = TEX.read_text(encoding="utf-8")
    assert "initialize $E_1$,\n$E_2$, and $\\nu_{12}$" not in t, (
        "the stiffness paragraph claims E2 comes from the laboratory moduli "
        "again, which contradicts Table 5")
    assert "$A_E$ is" in t, "the definition of A_E has gone from Section 3.2"


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_elastic_moduli_row_is_one_value_per_lithology():
    """E1 and nu12 carry one value per rock, not one per fabric angle.

    The row said specimen-scale while the solver rotated the same tensor by
    the fabric angle, which applied the orientation dependence twice.
    """
    t = TEX.read_text(encoding="utf-8")
    row = re.search(r"\\rev\{\$E_1,\\nu_\{12\}\$\}\s*&(.*?)\\\\", t, re.S)
    assert row, "the elastic-moduli row of Table 5 has changed shape"
    assert "One value per lithology" in row.group(1), (
        "the elastic-moduli row no longer says the constants are per rock")
    assert "specimen-scale" not in row.group(1), (
        "Table 5 calls E1 and nu12 specimen-scale again; they are "
        "foliation-frame constants of the lithology")


@pytest.mark.skipif(not all(lith.field_cache_path(s).exists() for s in range(1, 15)),
                    reason="fields not exported")
def test_the_moduli_are_material_axis_constants_of_the_lithology():
    """E1, E2, nu12 and G12 are foliation-frame constants, one set per rock.

    They used to be re-read per specimen as the apparent modulus E(alpha).
    The solver rotates the stiffness tensor to each specimen's fabric angle,
    so doing that applied the orientation dependence twice and the tensor no
    longer returned the measured directional stiffness: 29 per cent low for
    the gneiss at 0 degrees, 21 per cent high for the schist. With the
    constants held per lithology it reproduces the measured E(alpha) to
    within 0.2 per cent at every schist angle.
    """
    from tools.ddm._toolkit import get_material_axes
    for sid in range(1, 15):
        d = np.load(lith.field_cache_path(sid), allow_pickle=True)
        rock = lith.lithology_of_sample(sid).display_name
        want = get_material_axes(rock)
        for key, col, scale in (("E1", "E1_MPa", 1e3), ("E2", "E2_MPa", 1e3),
                                ("G12", "G12_MPa", 1e3), ("nu12", "nu12", 1.0)):
            assert float(d[col]) == pytest.approx(want[key] * scale, rel=1e-9), (
                f"specimen {sid} carries {col} = {float(d[col])}, not the "
                f"{rock} material-axis value {want[key] * scale}")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_section_2_8_counts_the_parameter_kinds_in_the_table():
    """The prose count must track the category headings in Table 5."""
    t = TEX.read_text(encoding="utf-8")
    heads = re.findall(r"\\multicolumn\{5\}\{l\}\{(?:\\rev\{)?\\textit\{([^}]*)", t)
    assert len(heads) == 8, f"Table 5 now has {len(heads)} category headings: {heads}"
    said = re.search(r"They fall into (\w+) kinds", t)
    assert said, "the parameter-kind sentence has changed shape"
    words = {"five": 5, "six": 6, "seven": 7, "eight": 8}
    assert words[said.group(1)] == 6, (
        f"Section 2.8 says {said.group(1)} kinds; the table's eight headings "
        "collapse to six")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_schist_shear_modulus_reading_matches_the_measurements():
    """Section 3.2 explains why A_E under-describes the schist. Pin the numbers.

    The end-member ratio samples only 0 and 90 degrees. In the schist E(theta)
    dips at 45 degrees and the shear modulus is about half the gneiss value, so
    the anisotropy is carried by G12 rather than by an E1-E2 contrast. If any of
    these move, the explanation stops following from the data.
    """
    t = TEX.read_text(encoding="utf-8")
    d = pd.read_csv(RAW)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]

    E, G = {}, {}
    for rock in ("Augen gneiss", "Psammitic schist"):
        g = d[d.Rock_type == rock]
        E[rock] = g.groupby("Angle")["Modulus_of_Elasticity"].mean()
        G[rock] = g.groupby("Angle")["Shear_Modulus"].mean()

    gn, sc = E["Augen gneiss"], E["Psammitic schist"]
    assert round(gn.loc[0], 1) == 38.4 and round(gn.loc[90], 1) == 53.4
    assert round(sc.min(), 1) == 19.7 and sc.idxmin() == 45, (
        f"the schist E(theta) minimum is now {sc.min():.1f} GPa at "
        f"{sc.idxmin()} deg; Section 3.3 says 19.67 GPa at 45 deg")

    factor = sc.max() / sc.min()
    assert round(factor, 2) == 1.68, (
        f"E(theta) now varies by a factor of {factor:.2f} across the schist "
        "series; Section 3.3 says 1.68")
    assert round(sc.loc[0], 1) == 33.0 and round(sc.loc[90], 1) == 28.5
    ratio = sc.loc[90] / sc.loc[0]
    assert ratio < factor, (
        "the end-member ratio no longer understates the variation in "
        "E(theta), which is the point Section 3.3 makes")

    # G12 is identified from E(theta), not read from the Shear_Modulus column,
    # which in this table is E/(2(1+nu)) at every row and so carries no
    # independent shear information. The identified values must stay well
    # below the isotropic equivalent in the schist, which is what makes the
    # shear modulus, rather than the E1-E2 contrast, carry its anisotropy.
    from tools.ddm._toolkit import get_material_axes
    axes = {r: get_material_axes(r) for r in ("Augen gneiss", "Psammitic schist")}
    assert round(axes["Augen gneiss"]["G12"], 1) == 16.6
    assert round(axes["Psammitic schist"]["G12"], 1) == 6.5
    for rock in axes:
        iso = E[rock].loc[90] / (2 * (1 + axes[rock]["nu12"]))
        if rock == "Psammitic schist":
            assert axes[rock]["G12"] < 0.7 * iso, (
                "the schist G12 is no longer well below its isotropic "
                "equivalent, so the shear modulus no longer carries its "
                "elastic anisotropy")

    assert "carried mainly by its low in-plane shear" in t, (
        "the sentence explaining why A_E under-describes the schist has gone")
