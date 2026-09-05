"""The notebooks must equal what ``build_notebooks.py`` would write.

``scripts/notebook_cells/*.json`` is the source the three publication notebooks
are generated from, and ``build_notebooks.py`` overwrites them from it. Editing
a notebook directly therefore creates a change that the next rebuild silently
reverts, and nothing warned about it: ``--check`` prints what it would write
without comparing it to what is on disk.

Both halves of that had happened. The notebooks had been edited to write into
``outputs/`` after the output tree moved, while the JSON still named the retired
``figures/`` and ``results/`` roots and still called
``pathlib.Path('results').mkdir()``, so a rebuild would have recreated the
directories the release cleanup removed and sent three tables back to a path
nothing reads. The general notebook had also gained
``run("make_matrix_tensile_sensitivity")``, which the JSON did not have, so a
rebuild would have dropped a supplementary figure generator entirely.

This test compares the two directly, which is the check ``--check`` does not do.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "scripts" / "notebook_cells"

#: Which cell file supplies the tail of which notebooks.
SOURCES = {
    "general": ["Tensile_general_plots.ipynb"],
    "lithology": ["Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb"],
}

#: Output roots retired when the tree moved to ``outputs/``. A generated cell
#: naming one of these is writing where nothing reads.
RETIRED_ROOTS = ("'figures/", '"figures/', "'results/", '"results/')


def _src(cell) -> str:
    s = cell["source"]
    return (s if isinstance(s, str) else "".join(s)).strip()


def _pairs(stem, notebook):
    spec = json.loads((CELLS / f"{stem}.json").read_text(encoding="utf-8"))
    nb = json.loads((REPO / notebook).read_text(encoding="utf-8"))
    tail = nb["cells"][len(nb["cells"]) - len(spec):]
    return list(zip(spec, tail))


@pytest.mark.parametrize(
    "stem,notebook",
    [(s, nb) for s, nbs in SOURCES.items() for nb in nbs])
def test_notebook_tail_matches_its_cell_source(stem, notebook):
    drifted = [k for k, (a, b) in enumerate(_pairs(stem, notebook))
               if _src(a) != _src(b)]
    assert not drifted, (
        f"{notebook} differs from scripts/notebook_cells/{stem}.json at cell "
        f"index {drifted}. Whichever is right, they must agree: running "
        "build_notebooks.py would overwrite the notebook with the JSON and "
        "discard the difference. Edit the JSON, then rebuild."
    )


@pytest.mark.parametrize("stem", sorted(SOURCES))
def test_the_cell_source_does_not_name_a_retired_output_root(stem):
    spec = json.loads((CELLS / f"{stem}.json").read_text(encoding="utf-8"))
    offenders = [k for k, c in enumerate(spec)
                 if any(r in _src(c) for r in RETIRED_ROOTS)]
    assert not offenders, (
        f"{stem}.json writes to the retired figures/ or results/ roots at cell "
        f"{offenders}. Everything generated now lives under outputs/; a rebuild "
        "would recreate the directories the release cleanup removed."
    )


@pytest.mark.parametrize("stem", sorted(SOURCES))
def test_no_cell_recreates_a_retired_directory(stem):
    spec = json.loads((CELLS / f"{stem}.json").read_text(encoding="utf-8"))
    offenders = [k for k, c in enumerate(spec)
                 if "mkdir" in _src(c) and "results" in _src(c)]
    assert not offenders, (
        f"{stem}.json still calls mkdir on results/ at cell {offenders}")
