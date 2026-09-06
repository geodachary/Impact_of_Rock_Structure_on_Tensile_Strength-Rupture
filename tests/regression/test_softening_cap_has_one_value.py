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


def test_the_cap_runs_at_the_stated_value_and_the_gap_is_disclosed():
    """The model runs at 0.05; under the measured ratios the sweep picks elsewhere.

    While the adopted anisotropy ratios were in force the selection rule
    returned 0.05 in both rocks and the model ran there. The material-axis
    constants change the sweep: the gneiss curve rises to the upper bound
    without turning over, and the schist has the shallow interior maximum with
    its smallest within-tolerance cap at 0.10. The model still runs at 0.05
    and Section 3.9 says so explicitly, so this test pins the disclosure
    rather than an agreement that no longer holds.
    """
    if not SWEEP.is_file():
        pytest.skip("kmax_softening_sweep.csv not written yet")
    fn = _resolvers()["local_damage"]
    for rock in ROCKS:
        kC, kT = fn(rock)
        assert kC == pytest.approx(0.05), (
            f"{rock}: the model runs at k_C = {kC}; Section 3.9 states 0.05")
        assert kT == pytest.approx(0.35 * kC), (
            f"{rock}: k_T = {kT} is not 0.35 k_C, the tie the sweep assumes")
    tex = (Path(__file__).resolve().parents[2] / "manuscript"
           / "manscript_revision_001.tex")
    if tex.is_file():
        t = tex.read_text(encoding="utf-8")
        assert "a value the sweep does not select" in t, (
            "Section 3.9 no longer discloses that the running cap differs from "
            "the one the selection rule returns")


def test_the_sweep_rule_is_the_one_the_paper_states():
    """Grid, ratio and tolerance, read out of the sweep module itself."""
    from tools.analysis import kmax_sweep as ks
    src = inspect.getsource(ks)
    assert "np.linspace(0.05, 0.90, 18)" in src, "the swept grid has moved"
    assert re.search(r"KT_RATIO\s*=\s*0\.35", src), "the k_T tie has moved"
    assert re.search(r"REL_TOL\s*=\s*0\.02", src), "the selection tolerance has moved"


@pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")
def test_the_paper_quotes_that_cap():
    """The running cap the paper states must be the one the resolvers return."""
    said = re.search(r"run at \$k_C = ([\d.]+)\$ with \$k_T = ([\d.]+)\$",
                     TEX.read_text(encoding="utf-8"))
    assert said, "the running-cap sentence is no longer in its expected form"
    kC, kT = _resolvers()["local_damage"]("Augen gneiss")
    assert float(said.group(1)) == pytest.approx(kC), (
        f"the paper says k_C = {said.group(1)} and the model runs at {kC}")
    assert float(said.group(2)) == pytest.approx(kT, abs=5e-4), (
        f"the paper says k_T = {said.group(2)} and the model runs at {kT}")


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
