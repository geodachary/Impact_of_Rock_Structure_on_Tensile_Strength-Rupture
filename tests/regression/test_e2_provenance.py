"""E2 is derived from an adopted ratio, and the parameter table must say so.

``E1``, ``nu12`` and ``G12`` are read per specimen from the measurement file.
``E2`` is not: it is ``E1 / ROCK_ANISO_RATIO`` with the ratio a hardcoded
per-lithology constant, 2.037 and 3.763, and there is no E2 column anywhere in
the measurements. Table 5 listed all three of E1, E2 and nu12 under "Measured
directly", sourced to UCS tests.

That matters beyond bookkeeping. The compliance contrast is the quantity
Section 5.3 credits with reproducing the weaker localization of the gneiss
without discrete augen, so a reader needs to know it is an adopted input
rather than a measurement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import lithology as lith
from tools.ddm._toolkit import ROCK_ANISO_RATIO

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
RAW = REPO / "selected_all_samples.csv"


def test_no_measurement_of_e2_exists():
    cols = pd.read_csv(RAW, nrows=1).columns
    assert not any("E2" in c or "modulus_2" in c.lower() for c in cols), (
        "an E2 column has appeared in the measurements; if it is now measured, "
        "Table 5 should move it back under 'Measured directly'")


@pytest.mark.skipif(not all(lith.field_cache_path(s).exists() for s in (1, 8)),
                    reason="fields not exported")
def test_e2_is_e1_divided_by_the_adopted_ratio():
    for sid, key in ((1, "augen gneiss"), (8, "psammitic schist")):
        d = np.load(lith.field_cache_path(sid), allow_pickle=True)
        ratio = float(d["E1_MPa"]) / float(d["E2_MPa"])
        assert ratio == pytest.approx(ROCK_ANISO_RATIO[key], rel=1e-6), (
            f"specimen {sid}: E1/E2 is {ratio:.4f}, the adopted ratio is "
            f"{ROCK_ANISO_RATIO[key]}")


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_parameter_table_does_not_call_e2_measured():
    t = TEX.read_text(encoding="utf-8")
    assert "$E_1,E_2,\\nu_{12}$ & lithology-specific & Orthotropic elastic moduli" not in t, \
        "Table 5 lists E2 as measured directly again"
    assert "Adopted, not measured independently" in t, (
        "the adopted-parameter block has gone from Table 5")
    assert "no\nindependent measurement of $E_2$ was made" in t, (
        "the E2 provenance note has gone")


def test_every_copy_of_the_ratio_agrees_with_the_canonical_one():
    """Nine modules define this dict; eight are dead shadows of the first.

    Every caller goes through ``get_anisotropy_ratio``, which reads
    ``_toolkit``. Editing one of the eight therefore changes nothing, and a
    future caller reading its own copy would pick up whatever is there. Held
    equal until the shadows are removed.
    """
    import re
    from pathlib import Path

    pat = re.compile(r"ROCK_ANISO_RATIO = \{(.*?)\}", re.S)
    found = {}
    for p in sorted((REPO / "tools").rglob("*.py")):
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            pairs = re.findall(r'"([^"]+)":\s*([\d.]+)', m.group(1))
            found[str(p.relative_to(REPO))] = {k: float(v) for k, v in pairs}

    assert found, "the anisotropy ratio is no longer defined anywhere"
    canonical = found["tools/ddm/_toolkit.py"]
    assert canonical == {"augen gneiss": 2.037, "psammitic schist": 3.763,
                         "psammatic schist": 3.763}
    for path, d in found.items():
        assert d == canonical, (
            f"{path} defines a different anisotropy ratio ({d}) from the "
            f"canonical {canonical}. Only the canonical one is in force, so "
            "this edit would silently do nothing.")


def test_the_ratio_is_not_derivable_from_the_measurements():
    """Guard the provenance note: if it becomes derivable, say so instead.

    The note in ``_toolkit`` lists the combinations that were checked. If some
    measured combination ever reproduces the adopted ratios, the honest thing
    is to derive them rather than keep calling them adopted.
    """
    d = pd.read_csv(RAW)
    target = {"Augen gneiss": 2.037, "Psammitic schist": 3.763}
    for rock, g in d.groupby("Rock_type"):
        m = g.groupby("Angle")[["Modulus_of_Elasticity", "Shear_Modulus",
                                "Poisson_Ratio", "UCS_(Mpa)"]].mean()
        E, G, nu = m.Modulus_of_Elasticity, m.Shear_Modulus, m.Poisson_Ratio
        cands = [E.mean() / G.mean(), E.loc[0] / G.loc[0], E.max() / E.min(),
                 E.loc[90] / E.loc[0], 2 * (1 + nu.mean()),
                 m["UCS_(Mpa)"].max() / m["UCS_(Mpa)"].min()]
        for c in cands:
            assert abs(c - target[rock]) > 0.02, (
                f"{rock}: a measured combination now gives {c:.3f}, matching "
                f"the adopted ratio {target[rock]}. Derive it and update both "
                "the note in _toolkit and Table 5.")
