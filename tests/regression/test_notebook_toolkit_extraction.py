"""The analysis must live in one place, and the notebooks must only call it.

Both lithology notebooks used to define the same ~100 functions inline, then the
same ~15 000 lines of analysis on top of them. The functions moved to
``tools.ddm``; the analysis then moved to ``tools.analysis``, one module per
section. What is left in a notebook is markdown and one-line calls.

These tests fail if that regresses -- if a definition is pasted back into a
notebook, if a section module shadows a shared name, or if the two lithology
notebooks stop being the same notebook with a different rock bound.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb"]
ALL_NOTEBOOKS = NOTEBOOKS + ["Tensile_general_plots.ipynb"]
SECTIONS = REPO / "tools" / "analysis"


def _code_cells(nb):
    doc = json.loads((REPO / nb).read_text(encoding="utf-8"))
    return [("".join(c["source"]), i) for i, c in enumerate(doc["cells"])
            if c["cell_type"] == "code"]


def _section_files():
    return [f for f in sorted(SECTIONS.glob("*.py"))
            if f.name not in ("__init__.py", "_context.py")]


@pytest.fixture(scope="module")
def exported():
    import tools.ddm as ddm
    return set(ddm.__all__)


@pytest.mark.parametrize("nb", ALL_NOTEBOOKS)
def test_notebook_defines_no_functions(nb):
    """A definition in a notebook is a copy that can drift from the package."""
    offenders = []
    for src, i in _code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                offenders.append(f"{n.name} (cell {i})")
    assert not offenders, (
        f"{nb} defines {', '.join(offenders)}. Put it in tools/ and call it, "
        "then regenerate with scripts/build_notebooks.py.")


@pytest.mark.parametrize("path", _section_files(), ids=lambda p: p.stem)
def test_no_section_shadows_the_toolkit(path, exported):
    """A local def of an exported name would silently override the shared one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    clashes = sorted({n.name for n in tree.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name in exported})
    assert not clashes, (
        f"{path.name} redefines names owned by tools.ddm: {', '.join(clashes)}. "
        "Edit tools/ddm/_toolkit.py instead of keeping a copy.")


@pytest.mark.parametrize("path", _section_files(), ids=lambda p: p.stem)
def test_section_has_exactly_one_entry_point(path):
    """Two ``main`` definitions in one module means one silently wins.

    Several of these cells were written as command-line scripts and define
    ``def main()``. The generated entry point is also ``main``, and being
    defined later it shadows the cell's -- so the cell's own ``main()`` call
    re-enters the wrapper. That import cleanly and fails only partway through a
    half-hour run, which is why it is checked here rather than left to a run.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mains = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "main"]
    assert len(mains) <= 1, (
        f"{path.name} defines main() {len(mains)} times; the cell's own entry "
        "point should have been renamed to run_section")


def test_the_two_lithology_notebooks_run_the_same_code():
    """Their code may differ only where it names the lithology.

    This is the property the whole refactor exists to guarantee: the gneiss and
    schist workflows are the same code, and the only thing separating them is
    the ``Lithology`` bound at the top. Prose is exempt -- the two rocks are
    described differently on purpose -- so the check is on code cells.
    """
    a, b = [_code_cells(nb) for nb in NOTEBOOKS]
    assert len(a) == len(b), "the notebooks no longer have the same shape"

    tokens = ("AUGEN_GNEISS", "PSAMMITIC_SCHIST")
    for (sa, i), (sb, _) in zip(a, b):
        for t in tokens:
            sa, sb = sa.replace(t, "<ROCK>"), sb.replace(t, "<ROCK>")
        assert sa == sb, (
            f"code cell {i} differs by more than the lithology it names:\n"
            f"--- gneiss ---\n{sa[:400]}\n--- schist ---\n{sb[:400]}")


@pytest.mark.parametrize("nb", ALL_NOTEBOOKS)
def test_no_empty_cells(nb):
    """Blank cells are leftovers; they execute as no-ops and clutter the read."""
    doc = json.loads((REPO / nb).read_text(encoding="utf-8"))
    blank = [i for i, c in enumerate(doc["cells"]) if not "".join(c["source"]).strip()]
    assert not blank, f"{nb} has empty cells at {blank}"


def test_toolkit_has_no_circular_theme_imports():
    """Themed views re-export from _toolkit only, so they cannot deadlock."""
    pkg = REPO / "tools" / "ddm"
    for f in sorted(pkg.glob("*.py")):
        if f.name in ("__init__.py", "_toolkit.py"):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                assert n.module in ("_toolkit", "tools.ddm._toolkit"), (
                    f"{f.name} imports {n.module}; themed views must re-export "
                    "from _toolkit only")
