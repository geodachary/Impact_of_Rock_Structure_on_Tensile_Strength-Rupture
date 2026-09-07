"""Every figure the manuscript cites, and the numbers its captions carry.

The plots themselves cannot drift silently: ``tools/__init__`` pins
SOURCE_DATE_EPOCH, so a regenerated figure is byte-identical when the data and
code are unchanged, and ``reproduce_all`` reports how many it had to copy. What
does drift is the caption text, which is written by hand.

Five caption claims had. Three were superseded numbers: the collocation
deviation (0.016% against 0.0145%), the coarsest-level error (0.32% against
0.26%) and the grid shift (0.08 against 0.007 percentage points). Two were
wrong about what the data show: the elevated-energy caption gave the schist
elongations as 4.85 and 4.62 rising "from 1.06 to 1.43" in the gneiss, where
the values are 4.79, 4.63 and 1.17 to 1.55, and concluded that corridor
development is confined to low fabric angles when the schist forms a corridor
at 90 degrees as well; and the G/Gc caption said Gc varies by under 0.3% in
eight of fourteen specimens, where only three do, and attributed the largest
variation to 75 and 90 degrees when the largest is the schist at 90 and the
gneiss at 30.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
TAB = REPO / "outputs" / "tables"
G, P = "Augen gneiss", "Psammitic schist"

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript absent")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def test_every_cited_graphic_exists(tex):
    cited = sorted({m.group(1) for m in
                    re.finditer(r"includegraphics\[[^\]]*\]\{([\w./-]+)\}", tex)})
    assert len(cited) == 54, f"the manuscript now cites {len(cited)} graphics"
    missing = [g for g in cited
               if not (REPO / "manuscript" /
                       (g if "." in Path(g).name else g + ".pdf")).is_file()]
    assert not missing, f"cited but absent from manuscript/: {missing}"


def test_figures_are_reproducible_by_construction():
    """The property that makes a byte comparison meaningful at all."""
    import tools  # noqa: F401  (import sets the variable)
    import os
    assert os.environ.get("SOURCE_DATE_EPOCH") == "1735689600", (
        "SOURCE_DATE_EPOCH is no longer pinned, so a regenerated figure gets a "
        "fresh timestamp and identical output can no longer be recognised")


def test_the_mesh_convergence_caption(tex):
    co = pd.read_csv(TAB / "convergence_collocation.csv")
    gr = pd.read_csv(TAB / "convergence_grid.csv")
    coarse = 100 * co[co.n_boundary == co.n_boundary.min()].rel_error.max()
    prod = 100 * co[co.n_boundary == 420].rel_error.max()
    piv = gr.pivot_table(index="sample_id", columns="n_interior", values="area_fraction")
    shift = 100 * (piv[20081] - piv[31417]).abs().max()
    assert coarse == pytest.approx(0.26, abs=0.01)
    assert prod == pytest.approx(0.0145, abs=0.0005)
    assert shift == pytest.approx(0.007, abs=0.001)
    assert "the error falls from $0.26$\\%" in tex
    assert "largest deviation over the fourteen specimens is $0.014$\\%" in tex
    assert "at most $0.007$ percentage" in tex
    assert "$0.016$\\%" not in tex, "the superseded collocation figure is back"


def test_the_energy_localization_caption(tex):
    e = pd.read_csv(TAB / "energy_localization.csv")
    s = e[e.rock == P].set_index("angle_deg")
    g = e[e.rock == G].set_index("angle_deg")
    assert s.loc[0].elongation == pytest.approx(4.79, abs=0.005)
    assert s.loc[15].elongation == pytest.approx(4.63, abs=0.005)
    assert s.loc[90].elongation == pytest.approx(4.62, abs=0.005)
    assert g.elongation.min() == pytest.approx(1.17, abs=0.005)
    assert g.elongation.max() == pytest.approx(1.55, abs=0.005)
    # the corridor is an end-member feature in the schist, not a low-angle one
    corridor = sorted(int(a) for a in s.index if s.loc[a].corridor)
    assert corridor == [0, 15, 90], (
        f"the schist corridor angles are now {corridor}; the caption says the "
        "two fabric end members")
    assert not g.corridor.any(), "the gneiss now forms a corridor somewhere"
    assert (g.n_components == 2).all()
    assert "confined to the two fabric end\n    members" in tex


def test_the_ggc_caption_reversal_fractions():
    """Replaces an unverifiable claim with one the band table supports.

    The caption used to quote a "normalized trend" of +0.032 and +0.042 per
    step. No producer computed it and its definition was not recorded, so it
    could not be checked; it is now stated as the reversal fractions, which are
    in outputs/tables/fig20_GGc_band_summary.csv.
    """
    import pandas as pd
    b = pd.read_csv(TAB / "fig20_GGc_band_summary.csv").set_index("band")
    low = b.loc["low (0-30 deg)", "mean_reversal_fraction"]
    mid = b.loc["intermediate (45-60 deg)", "mean_reversal_fraction"]
    assert low == pytest.approx(0.46, abs=0.005)
    assert mid == pytest.approx(0.41, abs=0.005)
    assert low > mid, (
        "the low-angle profiles now reverse less often than the intermediate "
        "ones; the caption says they are not the steadier of the two")
    assert int(b.n_monotonic.sum()) == 0
    tex = TEX.read_text(encoding="utf-8")
    assert "$0.032$ per step" not in tex, "the unverifiable trend claim is back"


def test_the_ggc_caption_gc_variation():
    from tools import fracture_energy as fe
    from tools import lithology as lith
    var, orders = [], []
    for sid in range(1, 15):
        d = fe.load_profile(sid)
        Gc = np.asarray(d["Gc"], float)
        Gv = np.asarray(d["G"], float)
        var.append(100.0 * (Gc.max() - Gc.min()) / Gc.min())
        orders.append(np.log10(Gv.max() / Gv[Gv > 0].min()))
    var = np.asarray(var)
    assert int((var < 0.3).sum()) == 3, (
        f"{(var < 0.3).sum()} of 14 specimens now vary by under 0.3%; the "
        "caption says three")
    assert var.max() == pytest.approx(84, abs=1.5)
    assert int(np.argmax(var)) == 13, "the largest Gc variation is no longer the schist at 90 deg"
    assert np.median(orders) == pytest.approx(8.0, abs=0.5)


def test_the_ati_profile_caption_thresholds():
    """Delta chi-squared 3.84 and the claim that no optimum sits at a bound."""
    from tools import ati_model as A
    a = pd.read_csv(TAB / "ati_envelope_parameters.csv").set_index("rock")
    from scipy import stats
    assert stats.chi2.ppf(0.95, 1) == pytest.approx(3.84, abs=0.005)
    for rock in (G, P):
        r = a.loc[rock]
        assert A.ETA_BOUNDS[0] < r.eta < A.ETA_BOUNDS[1]
        assert A.BETA_BOUNDS[0] < r.beta_peak_deg < A.BETA_BOUNDS[1]


def test_the_fabric_traction_caption_endpoints(tex):
    t = pd.read_csv(TAB / "fabric_tractions.csv").set_index(["rock", "angle_deg"])
    assert t.loc[(G, 0.0)].sigma_n_mean_MPa == pytest.approx(-23.36, abs=0.01)
    assert t.loc[(P, 0.0)].sigma_n_mean_MPa == pytest.approx(-21.92, abs=0.01)
    assert "$-23.36$" in tex and "$-21.92$" in tex
