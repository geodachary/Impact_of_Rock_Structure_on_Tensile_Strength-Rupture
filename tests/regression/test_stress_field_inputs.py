"""The exported stress fields must be solved with each specimen's own properties.

Three failures motivated these tests, and none of them raised anything.

Poisson's ratio was taken from a single command-line default of 0.25 for all
fourteen specimens while the measured column, spanning 0.16 to 0.29, sat unused
in the table. Two gneiss specimens carried elastic constants that no longer
matched the table, because their fields had been exported before an edit and
never re-exported. And the cached displacement solutions were keyed on the
specimen number alone, so no change to the specimen table could invalidate them.

Each test compares what the exported field says it was solved with against what
the specimen table currently holds, which is the only check that catches a
stale export.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tools import fabric_tractions as ft
from tools.data_io import load_specimen_table

REPO = Path(__file__).resolve().parents[2]

#: These assertions used to be made against notebook cells -- cell 33 of the
#: gneiss notebook and cell 35 of the schist one, which held the same code
#: twice. That code now lives in one module per section, so each check is made
#: once and holds for both lithologies by construction rather than by a pair of
#: parametrised cases that could diverge.
SECTION = REPO / "tools" / "analysis"

FIELDS = ft.field_files()
needs_fields = pytest.mark.skipif(not FIELDS, reason="fields not exported yet")


def _section(name):
    return (SECTION / f"{name}.py").read_text(encoding="utf-8")


@needs_fields
@pytest.mark.parametrize("path", FIELDS, ids=lambda p: p.stem)
def test_field_matches_specimen_table(path):
    """The export carries the foliation-frame constants of its lithology.

    These are properties of the rock, not of the specimen: the 90 degree test
    loads along the foliation and gives E1 and nu12, the 0 degree test loads
    across it and gives E2. The solver rotates this tensor to each specimen's
    fabric angle, so reading the apparent modulus E(alpha) per specimen would
    apply the orientation dependence twice.
    """
    from tools.ddm._toolkit import get_material_axes

    f = ft.load_field(path)
    want = get_material_axes(str(f["rock"]))
    assert float(f["E1_MPa"]) == pytest.approx(want["E1"] * 1e3, rel=1e-6)
    assert float(f["E2_MPa"]) == pytest.approx(want["E2"] * 1e3, rel=1e-6)
    assert float(f["G12_MPa"]) == pytest.approx(want["G12"] * 1e3, rel=1e-6)
    assert float(f["nu12"]) == pytest.approx(want["nu12"], abs=1e-9)


@needs_fields
def test_poisson_ratio_is_one_value_per_lithology():
    """nu12 is measured with E1, at 90 degrees, so it is a rock constant.

    It used to vary specimen by specimen, which is what made the tensor
    orientation-dependent before it was even rotated.
    """
    from tools.ddm._toolkit import get_material_axes

    seen = {}
    for path in FIELDS:
        f = ft.load_field(path)
        seen.setdefault(str(f["rock"]), set()).add(round(float(f["nu12"]), 9))
    assert len(seen) == 2, f"expected two lithologies, got {sorted(seen)}"
    for rock, values in seen.items():
        assert len(values) == 1, (
            f"{rock}: nu12 still varies between specimens, {sorted(values)}")
        assert values.pop() == pytest.approx(
            get_material_axes(rock)["nu12"], abs=1e-9)


def test_solver_reads_the_material_axes():
    src = _section("crack_energy_suite")
    assert "get_material_axes(rock)" in src, (
        "the solver no longer reads the foliation-frame constants")
    assert 'float(row["Poisson_Ratio"])' not in src, (
        "the solver is reading a per-specimen Poisson ratio again")


def test_displacement_cache_key_covers_the_material():
    """A cache key that omits the material cannot notice the material changing."""
    src = _section("direction_circles")
    m = re.search(r"stamp = hashlib\.sha1\(\s*np\.array\(\[(.*?)\], float\)", src, re.S)
    assert m, "cache key carries no material fingerprint"
    keyed = {t.strip() for t in m.group(1).split(",")}
    assert {"D", "t", "E1_in", "E2_in", "nu", "G_in", "alpha"} <= keyed


def test_arrow_length_uses_one_scale_for_all_panels():
    """Per-panel normalisation makes every orientation look alike by construction."""
    src = _section("direction_circles")
    assert "ref_global" in src
    assert "np.nanpercentile(mag_plot" not in src, "arrow length is still per-panel"


@needs_fields
def test_foliation_tractions_resolve_the_orientation_dependence():
    """The fabric-resolved tractions must carry the signal the glyphs do not."""
    tab = ft.traction_table()
    rot = ft.principal_rotation_table()

    for rock, g in tab.groupby("rock"):
        g = g.sort_values("angle_deg")
        sn = g.sigma_n_mean_MPa.to_numpy()
        # clamped shut when the fabric is across the load, open near 90 degrees
        assert sn[0] < -5.0, f"{rock}: foliation not clamped at 0 deg"
        assert sn[-1] > sn[0] + 5.0, f"{rock}: no orientation dependence in sigma_n"
        assert np.all(np.diff(sn) > 0), f"{rock}: sigma_n is not monotonic in angle"
        # shear peaks at an intermediate angle rather than at an end member
        tau = g.tau_abs_mean_MPa.to_numpy()
        assert 0 < int(np.argmax(tau)) < len(tau) - 1, f"{rock}: shear peak at an end member"

    # the quantity the glyph figure draws stays nearly invariant, which is the
    # reason those panels look alike and why the tractions are plotted instead
    assert rot.rotation_median_deg.max() < 10.0
