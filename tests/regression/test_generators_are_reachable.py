"""Every generator must be called from a notebook, and own its output alone.

Two failures motivated these tests, and both were silent.

A script that no notebook calls is invisible: a reader cannot tell from the
notebooks that the figure exists or how it was made, and a full notebook run
does not refresh it. ``make_envelope_fit_figure.py`` was in that state, so the
strength-envelope figure in the manuscript came from a notebook cell fitting
four parameters while the text and tables quoted the script's two-parameter
fit. The plotted curve and the quoted numbers disagreed.

Two producers writing one filename is the same problem from the other side.
Which version reaches the paper then depends on execution order, so running a
section on its own could silently replace a published figure with a different
model's output. Four figures were contested this way.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb",
             "Tensile_general_plots.ipynb"]

GENERATORS = sorted(p.stem for p in (REPO / "scripts").glob("*.py")
                    if p.stem.startswith(("make_", "patch_")))

FIGURE = re.compile(r'["\']([\w{}().\-]*?\.(?:pdf|png))["\']')


def _called_from_notebooks():
    called = set()
    for nb in NOTEBOOKS:
        doc = json.loads((REPO / nb).read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            src = "".join(cell["source"])
            called |= set(re.findall(r'run(?:_all)?\(\s*\(?\s*"([a-z_0-9]+)"', src))
            called |= set(re.findall(r'"(make_[a-z_0-9]+)"', src))
    return called


@pytest.mark.parametrize("script", GENERATORS)
def test_generator_is_called_from_a_notebook(script):
    assert script in _called_from_notebooks(), (
        f"scripts/{script}.py produces manuscript output but no notebook calls "
        "it, so a full notebook run will not refresh what it writes and a "
        "reader cannot see from the notebooks that it ran.")


def test_no_figure_has_two_producers():
    """One filename, one place that writes it."""
    producers = collections.defaultdict(set)
    sources = list((REPO / "scripts").glob("*.py")) + list((REPO / "tools").rglob("*.py"))
    for path in sources:
        if path.name == "extract_analysis_sections.py":
            continue          # the migration record, not a producer
        for m in FIGURE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            name = m.group(1)
            if name.startswith(".") and name.count(".") == 1:
                # A bare extension, from a suffix constant such as
                # SUFFIXES = (".pdf", ".png") or a table keyed on suffix. Not a
                # figure name, and counting it made every module that sorts
                # files by extension look like a producer of the same figure.
                continue
            if "{" in name and not name.startswith("{"):
                continue      # a partial f-string fragment, not a whole name
            key = re.sub(r"^\{[^}]*\}_", "<rock>_", name)
            if "{" in key:
                continue
            producers[key].add(str(path.relative_to(REPO)))

    contested = {k: sorted(v) for k, v in producers.items() if len(v) > 1}
    assert not contested, (
        "these figures are written by more than one place, so which version "
        f"reaches the manuscript depends on execution order: {contested}")
