"""The local-damage softening cap must be one number, not three.

``kmax_sweep`` sweeps k_C and selects a value under a stated rule. Two other
places carry a k_C: ``local_damage.resolve_local_damage_params`` and a second
copy of the same resolver inside ``mohr_coulomb_local``. Neither read the
sweep, so the sweep's conclusion was never propagated: both resolvers returned
0.850/0.297 for the augen gneiss and 0.650/0.227 for the psammitic schist,
labelled "calibrated defaults", while the sweep selected 0.05 for both rocks
and the manuscript reported 0.05.

Nothing published broke, because ``mohr_coulomb_local`` writes no figure and no
table. But running it printed a cap seventeen times the adopted one, which is
what surfaced the split, and a reader comparing the two would have found the
paper describing a model the code does not run.

This test ties the resolvers to the sweep and the sweep to the paper.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
SWEEP = REPO / "outputs" / "tables" / "kmax_softening_sweep.csv"

ROCKS = ("Augen gneiss", "Psammitic schist")


def _resolvers():
    from tools.analysis import local_damage as ld
    from tools.analysis import mohr_coulomb_local as mc
    return {"local_damage": ld.resolve_local_damage_params,
            "mohr_coulomb_local": mc.resolve_local_damage_params}


def test_both_resolvers_return_the_same_cap():
    """Two copies of one function must not disagree about its constants."""
    got = {name: {r: fn(r) for r in ROCKS} for name, fn in _resolvers().items()}
    a, b = got["local_damage"], got["mohr_coulomb_local"]
    assert a == b, f"the two resolvers disagree: {a} against {b}"


def test_the_cap_is_the_one_the_sweep_selects():
    """The adopted value has a machine-readable provenance; use it."""
    if not SWEEP.is_file():
        pytest.skip("kmax_softening_sweep.csv not written yet")
    sweep = pd.read_csv(SWEEP)
    fn = _resolvers()["local_damage"]
    for rock, col in zip(ROCKS, ("adopted_augen_gneiss", "adopted_psammitic_schist")):
        picked = sweep.loc[sweep[col], "k_C"]
        assert len(picked) == 1, f"{rock}: the sweep marks {len(picked)} adopted rows"
        kC, kT = fn(rock)
        assert kC == pytest.approx(float(picked.iloc[0])), (
            f"{rock}: the model runs at k_C = {kC} and the sweep selects "
            f"{float(picked.iloc[0])}. The sweep is the calibration; if the "
            "model is to run elsewhere, change the rule, not just the default.")
        assert kT == pytest.approx(0.35 * kC), (
            f"{rock}: k_T = {kT} is not 0.35 k_C, the tie the sweep assumes")


def test_the_sweep_rule_is_the_one_the_paper_states():
    """Grid, ratio and tolerance, read out of the sweep module itself."""
    from tools.analysis import kmax_sweep as ks
    src = inspect.getsource(ks)
    assert "np.linspace(0.05, 0.90, 18)" in src, "the swept grid has moved"
    assert re.search(r"KT_RATIO\s*=\s*0\.35", src), "the k_T tie has moved"
    assert re.search(r"REL_TOL\s*=\s*0\.02", src), "the selection tolerance has moved"


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_paper_quotes_that_cap():
    said = re.search(r"gives \$k_C = ([\d.]+)\$ for both\s*\n?lithologies", 
                     TEX.read_text(encoding="utf-8"))
    assert said, "the adopted-cap sentence is no longer in its expected form"
    kC, _ = _resolvers()["local_damage"]("Augen gneiss")
    assert float(said.group(1)) == pytest.approx(kC), (
        f"the paper says k_C = {said.group(1)} and the model runs at {kC}")


def test_the_cap_is_not_lithology_specific():
    """The sweep finds one value; the resolvers must not reintroduce two.

    They previously returned a different pair per rock, and the manuscript
    states in the same breath that the cap is not a lithology-specific
    quantity. Both cannot be true.
    """
    fn = _resolvers()["local_damage"]
    assert fn(ROCKS[0]) == fn(ROCKS[1]), (
        f"the cap is lithology-specific again: {fn(ROCKS[0])} against "
        f"{fn(ROCKS[1])}, while the sweep selects one value for both rocks")
