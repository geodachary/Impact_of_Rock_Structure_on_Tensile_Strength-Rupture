"""The revised analyses must live in the notebooks and stay thin.

The work of this revision is consolidated in ``tools/``. The notebooks call it:
per-lithology diagnostics in the two solver notebooks, two-rock comparisons in
the general-plots notebook. These tests pin that arrangement rather than the
numbers, which the module-level tests already cover.

The failure they guard against is analysis migrating back into a notebook, where
nothing checks it -- which is how the fabricated mode counts and the placeholder
stress profiles survived for as long as they did.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

LITHOLOGY = ("Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb")
GENERAL = "Tensile_general_plots.ipynb"
ALL = LITHOLOGY + (GENERAL,)

#: The heading that opens the consolidated block in each notebook. This used to
#: be a "REVISED ANALYSIS BLOCK" marker left over from the migration; the
#: notebooks are generated now, so the block is anchored on its real heading.
MARKERS = {
    nb: "# Classification, orientation validation and energy partitioning"
    for nb in LITHOLOGY}
MARKERS[GENERAL] = "# Consolidated cross-lithology diagnostics"


def _cells(nb):
    return json.loads((REPO / nb).read_text(encoding="utf-8"))["cells"]


def _block(nb):
    """The consolidated block, from its heading to the end of the notebook."""
    cells, marker = _cells(nb), MARKERS[nb]
    start = next((i for i, c in enumerate(cells) if marker in "".join(c["source"])), None)
    assert start is not None, f"{nb} carries no consolidated-analysis block"
    return cells[start:]


def _source(nb):
    return "\n".join("".join(c["source"]) for c in _block(nb)
                     if c["cell_type"] == "code")


@pytest.mark.parametrize("nb", ALL)
def test_block_is_present_exactly_once(nb):
    hits = sum(MARKERS[nb] in "".join(c["source"]) for c in _cells(nb))
    assert hits == 1, f"{nb} has {hits} blocks; the build should be idempotent"


@pytest.mark.parametrize("nb", ALL)
def test_every_block_cell_parses(nb):
    for c in _block(nb):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


@pytest.mark.parametrize("nb", ALL)
def test_block_is_self_contained(nb):
    """It must not lean on variables the long solver cells leave behind."""
    src = _source(nb)
    assert "REPO = Path.cwd()" in src, "the block does not establish its own paths"
    # tools/ is importable because notebooks run from the repository root; a
    # path edit here would trip test_notebooks_have_no_top_level_sys_path_manipulation
    assert "sys.path.insert" not in src, "the block should not edit sys.path"
    for leaked in ("dfmeta", "args.", "samples ="):
        assert leaked not in src, f"{nb} block reads notebook state: {leaked}"


@pytest.mark.parametrize("nb", ALL)
def test_no_literal_data_arrays(nb):
    tree = ast.parse(_source(nb))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            elts = node.value.elts
            if len(elts) >= 5 and all(isinstance(e, ast.Constant)
                                      and isinstance(e.value, (int, float)) for e in elts):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                raise AssertionError(f"{nb}: literal data array {names}")


@pytest.mark.parametrize("nb", LITHOLOGY)
def test_lithology_block_reports_only_its_own_rock(nb):
    src = _source(nb)
    # The display name and slug are derived from the bound Lithology rather
    # than written as literals, which is what stops a notebook reporting the
    # other rock's numbers after a copy-paste.
    assert "ROCK_NAME, SLUG = ROCK.display_name, ROCK.key" in src
    assert "rock == ROCK_NAME" in src, "results are not filtered to this lithology"
    for combined in ("ati_model", "sensitivity_sweeps", "softening_selection"):
        assert combined not in src, f"{combined} is a combined analysis; it belongs in {GENERAL}"


def test_general_block_carries_the_combined_analyses():
    src = _source(GENERAL)
    for stage in ("ati_model", "strength_anisotropy", "sensitivity_sweeps",
                  "softening_selection", "traces", "make_predicted_vs_actual",
                  "make_weakening_figure", "make_eshelby_supplement"):
        assert stage in src, f"{GENERAL} never reaches {stage}"


def test_four_class_scheme_is_documented_as_the_replacement():
    md = "\n".join("".join(c["source"]) for c in _block(LITHOLOGY[0])
                   if c["cell_type"] == "markdown")
    assert "supersedes" in md.lower()
    for cls in ("WT", "WS", "MT", "MS"):
        assert cls in _source(LITHOLOGY[0])


def test_inputs_are_verified():
    assert "len(ft.field_files()) == 14" in _source(GENERAL)
    assert "total.nunique() == 1" in _source(GENERAL)
