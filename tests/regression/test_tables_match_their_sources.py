"""Every manuscript table, against the file or the code it comes from.

Nine of the fourteen tables are machine-generated and spliced into the
manuscript by ``scripts/inline_tables.py`` between markers. Three things can go
wrong and none of them was guarded: the spliced block can drift from the
``manuscript/tables/*.txt`` it came from, a ``.txt`` can stop being spliced at
all, and a generated table can disagree with the CSV its producer read.

The remaining five are maintained by hand. Table 1 and the model-metrics table
are covered by the Section 2 and Section 4 files; the notation table is prose.
What is checked here is the parameter-transparency table, which lists every
constant in the framework, and the principal-stress permutation table.
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
TDIR = REPO / "manuscript" / "tables"
TAB = REPO / "outputs" / "tables"

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript absent")

BLOCK = re.compile(
    r"% >>> BEGIN GENERATED TABLE: (\S+) \(scripts/inline_tables\.py\) <<<\n"
    r"% Edit the producing script, not this block; it is overwritten\.\n"
    r"(.*?)\n% >>> END GENERATED TABLE: \1 <<<", re.S)


@pytest.fixture(scope="module")
def blocks():
    return dict(BLOCK.findall(TEX.read_text(encoding="utf-8")))


def _rows(name):
    return [l.strip() for l in (TDIR / f"{name}.txt").read_text().splitlines()
            if "&" in l and "\\\\" in l and not l.lstrip().startswith("%")]


def test_every_spliced_block_matches_its_source_file(blocks):
    assert len(blocks) == 9, f"expected nine generated tables, found {len(blocks)}"
    for name, body in blocks.items():
        src = TDIR / f"{name}.txt"
        assert src.is_file(), f"{name} is spliced but has no source file"
        assert src.read_text().strip() == body.strip(), (
            f"the {name} block in the manuscript has drifted from "
            f"manuscript/tables/{name}.txt; re-run scripts/inline_tables.py")


def test_no_generated_table_is_orphaned(blocks):
    for f in sorted(TDIR.glob("*.txt")):
        assert f.stem in blocks, (
            f"{f.name} is generated but never spliced into the manuscript")


def test_strength_by_angle_matches_the_specimen_table():
    """Rows are grouped under a lithology header rather than naming it."""
    import numpy as np
    d = pd.read_csv(REPO / "selected_all_samples.csv").dropna(how="all")
    rock, seen = None, 0
    # read the raw file: the lithology headers carry no "&" and so are not
    # returned by _rows, and without them the rows cannot be attributed
    raw = (TDIR / "table_S_strength_by_angle.txt").read_text().splitlines()
    for l in (x.strip() for x in raw):
        h = re.search(r"\\textit\{([A-Za-z ]+)\}", l)
        if h:
            rock = h.group(1).strip()
            continue
        m = re.match(r"(\d+) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", l)
        if not m or rock is None:
            continue
        seen += 1
        g = d[(d.Rock_type == rock) & (d.Angle == int(m.group(1)))].Tensile_strength_Mpa
        n, mean, sd = len(g), g.mean(), g.std(ddof=1)
        assert int(m.group(2)) == n
        assert float(m.group(3)) == pytest.approx(mean, abs=0.006)
        assert float(m.group(4)) == pytest.approx(sd, abs=0.006)
        assert float(m.group(5)) == pytest.approx(sd / np.sqrt(n), abs=0.006)
        assert float(m.group(6)) == pytest.approx(100.0 * sd / mean, abs=0.06)
    assert seen == 14


def test_ati_envelope_table_matches_the_fit():
    a = pd.read_csv(TAB / "ati_envelope_parameters.csv").set_index("rock")
    for l in _rows("table_S_ati_envelope"):
        m = re.match(r"([A-Za-z ]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & "
                     r"\[([\d.]+), ([\d.]+)\] & ([\d.]+) & \[(\d+), (\d+)\] & "
                     r"([\d.]+) & ([\d.]+) & ([\d.]+)", l)
        if not m:
            continue
        r = a.loc[m.group(1).strip()]
        assert float(m.group(2)) == pytest.approx(r.sigma0, abs=0.006)
        assert float(m.group(3)) == pytest.approx(r.sigma90, abs=0.006)
        assert float(m.group(4)) == pytest.approx(r.eta, abs=0.001)
        assert float(m.group(7)) == pytest.approx(r.beta_peak_deg, abs=0.05)
        assert float(m.group(10)) == pytest.approx(r.max_weakening_pct, abs=0.05)
        assert float(m.group(11)) == pytest.approx(r.R2, abs=0.001)
        assert float(m.group(12)) == pytest.approx(r.chi2_red, abs=0.005)


def test_mirror_symmetry_table_matches_its_csv():
    m = pd.read_csv(TAB / "displacement_mirror_symmetry.csv")
    seen = 0
    for l in _rows("table_S_mirror_symmetry"):
        f = re.match(r"(\d+) & ([A-Za-z ]+) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", l)
        if not f:
            continue
        seen += 1
        r = m[m["sample"] == int(f.group(1))].iloc[0]
        assert float(f.group(4)) == pytest.approx(r.u_antisymmetry_index, abs=6e-4)
        assert float(f.group(5)) == pytest.approx(r.v_symmetry_index, abs=6e-4)
        assert float(f.group(6)) == pytest.approx(r.max_index, abs=6e-4)
    assert seen == 14


def test_ggc_profile_table_matches_its_csv():
    p = pd.read_csv(TAB / "fig20_GGc_profile_metrics.csv")
    seen = 0
    for l in _rows("table_S_fig20_GGc"):
        f = re.match(r"(\d+) & ([A-Za-z ]+) & (\d+) & (\d+) & ([\d.]+) & "
                     r"([\d.]+) & ([\d.]+) & ([\d.]+)", l)
        if not f:
            continue
        seen += 1
        r = p[p["sample"] == int(f.group(1))].iloc[0]
        assert int(f.group(4)) == int(r.n_steps)
        assert float(f.group(5)) == pytest.approx(r.reversal_fraction, abs=0.006)
        assert float(f.group(6)) == pytest.approx(r.cv, abs=0.006)
        assert float(f.group(7)) == pytest.approx(r.min_G_over_Gc_energy, abs=0.001)
        assert float(f.group(8)) == pytest.approx(r.shear_fraction, abs=0.006)
    assert seen == 14
    # the caption states the dataset minimum rather than carrying it as a column
    assert "being 1.004" in (TDIR / "table_S_fig20_GGc.txt").read_text()
    assert p.min_G_over_Gc_energy.min() == pytest.approx(1.004, abs=0.001)


def test_band_summary_table_matches_its_csv():
    b = pd.read_csv(TAB / "fig20_GGc_band_summary.csv")
    want = {"low": 0, "intermediate": 1, "high": 2}
    for l in _rows("table_S_fig20_bands"):
        f = re.match(r"([a-z]+), .* & (\d+) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", l)
        if not f:
            continue
        r = b.iloc[want[f.group(1)]]
        assert int(f.group(2)) == int(r.n_profiles)
        assert int(f.group(3)) == int(r.n_monotonic)
        assert float(f.group(4)) == pytest.approx(r.mean_reversal_fraction, abs=0.006)
        assert float(f.group(5)) == pytest.approx(r.mean_CV, abs=0.006)
        assert float(f.group(6)) == pytest.approx(r.mean_shear_fraction, abs=0.006)


def test_eshelby_table_matches_its_csv():
    e = pd.read_csv(TAB / "eshelby_bound.csv")
    seen = 0
    for l in _rows("table_S_eshelby_bound"):
        nums = re.findall(r"[\d.]+", l)
        if len(nums) < 3:
            continue
        con, asp, val = float(nums[0]), float(nums[1]), float(nums[-1])
        r = e[np.isclose(e.contrast, con) & np.isclose(e.aspect, asp)]
        if len(r) != 1:
            continue
        seen += 1
        assert val == pytest.approx(float(r.concentration.iloc[0]), abs=0.006)
    assert seen >= 6


def test_the_permutation_table_offsets_match_the_code():
    from tools.strain_partitioning import REGIME_OFFSETS_MPA
    assert REGIME_OFFSETS_MPA == {"Thrust": -0.50, "Strike-Slip": 0.00,
                                  "Extensional": +0.50}


def test_the_parameter_transparency_table_matches_the_code():
    """Every constant the table lists, read from where it is defined."""
    from tools.ddm._toolkit import ROCK_MATERIAL_AXES as MA
    from tools import airy_solution as A
    from tools import strain_partitioning as sp
    from tools import fabric_tractions as ft
    from tools.analysis import crack_energy_suite as ce
    from tools.crack_helpers import Gc_theta_weak_plane
    from tools import lithology as lith

    g, s = MA["augen gneiss"], MA["psammitic schist"]
    assert (round(g["E1"], 2), round(g["nu12"], 3)) == (53.38, 0.210)
    assert (round(s["E1"], 2), round(s["nu12"], 3)) == (28.50, 0.190)
    assert round(g["E2"], 2) == 38.40 and round(s["E2"], 2) == 33.00
    assert round(g["G12"], 2) == 16.61 and round(s["G12"], 2) == 6.53
    assert A.DEFAULT_M == 48 and A.DEFAULT_LAM == 1e-18
    assert (sp.WEAK_T_RATIO, sp.WEAK_C_RATIO) == (0.35, 0.60)
    assert sp.ACTIVATION_FLOOR == 0.05 and sp.N_THETA == 361
    assert ft.CORE_FRAC == 0.85
    assert ce.MAX_RIM_EXTENSION_FRAC == 0.02
    assert ce.RETREAT_TOL_FRAC == 0.25
    sig = inspect.signature(Gc_theta_weak_plane).parameters
    assert sig["eta_deg"].default == 10.0 and sig["Gc_matrix"].default == 1.0
    assert 1.0 - sp.WEAK_T_RATIO == pytest.approx(0.65)
    assert lith.AUGEN_GNEISS.spacing_m == 0.010
    assert lith.PSAMMITIC_SCHIST.spacing_m == 0.002


def test_the_moduli_are_read_from_the_orientations_the_table_names():
    """E1 and nu12 from the 90 deg specimen, E2 from the 0 deg specimen."""
    from tools.ddm._toolkit import ROCK_MATERIAL_AXES as MA
    d = pd.read_csv(REPO / "selected_all_samples.csv").dropna(how="all")
    for rock, key in (("Augen gneiss", "augen gneiss"),
                      ("Psammitic schist", "psammitic schist")):
        g = d[d.Rock_type == rock].groupby("Angle")
        E, nu = g.Modulus_of_Elasticity.mean(), g.Poisson_Ratio.mean()
        assert MA[key]["E1"] == pytest.approx(E.loc[90], abs=0.006)
        assert MA[key]["nu12"] == pytest.approx(nu.loc[90], abs=0.006)
        assert MA[key]["E2"] == pytest.approx(E.loc[0], abs=0.006)
