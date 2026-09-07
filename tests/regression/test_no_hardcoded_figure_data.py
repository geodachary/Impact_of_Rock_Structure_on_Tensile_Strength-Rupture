"""Figures the manuscript includes must not be drawn from values typed in.

Three figures in this paper were built from data that never touched the solved
fields: a stress profile formed from a uniform placeholder times a strength
factor, and two panels of failure statistics whose mode counts were literal
lists with per-angle totals that a fixed grid cannot produce. Each was found
only when a reviewer questioned the interpretation.

This test looks for the pattern rather than the instances: a notebook cell that
writes a figure the manuscript includes, while assigning a long literal numeric
array. Parameter sweeps and axis tick lists are legitimate, so the check is
restricted to figures that reach the manuscript and to arrays long enough to be
per-specimen data.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript/manscript_revision_001.tex"

#: The manuscript source is not part of the software release, so tests that
#: read it skip on a clean clone rather than failing there.
needs_manuscript = pytest.mark.skipif(
    not TEX.is_file(), reason="manuscript source not shipped with the release")

pytestmark = needs_manuscript


#: Lists of chosen inputs, not measurements.
#: Chosen inputs and plotting parameters, not measurements: swept values,
#: axis ticks and histogram bin edges.
ALLOWED = {"mesh_points_list", "angles", "angles_ag", "angles_ps", "kmax_list",
           "hetero", "spacing", "spacings", "edges", "GRID_LEVELS", "NBD_LEVELS"}

#: Length at or above which a literal array is per-specimen data rather than a
#: handful of styling constants.
MIN_LEN = 5


def _manuscript_figures():
    return set(re.findall(r"\{([\w./-]+)\.pdf\}", TEX.read_text(encoding="utf-8")))


def _numeric_literal(node):
    """A list/tuple of numbers, or one nested one row deep, or np.array of either.

    The first version of this check looked only for a bare list of numbers and so
    missed ``np.array([[...], [...]])``, which is how the spacing and
    heterogeneity sweeps carried their values.
    """
    if isinstance(node, ast.Call):                       # np.array([...])
        f = node.func
        name = getattr(f, "attr", getattr(f, "id", ""))
        if name == "array" and node.args:
            return _numeric_literal(node.args[0])
        return 0
    if not isinstance(node, (ast.List, ast.Tuple)):
        return 0
    elts = node.elts
    if elts and all(isinstance(e, (ast.List, ast.Tuple)) for e in elts):
        return sum(_numeric_literal(e) for e in elts)
    if all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
           for e in elts):
        return len(elts)
    return 0


def _literal_arrays(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        n = _numeric_literal(node.value)
        if n >= MIN_LEN:
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id not in ALLOWED:
                    out.append((t.id, n))
    return out


KNOWN_UNRESOLVED: dict = {}   # the sensitivity sweeps are now computed


@pytest.mark.parametrize("nb", sorted(p.name for p in REPO.glob("*.ipynb")))
def test_manuscript_figures_are_not_drawn_from_literals(nb):
    figs = _manuscript_figures()
    d = json.loads((REPO / nb).read_text(encoding="utf-8"))
    problems = []
    for i, c in enumerate(d["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "savefig" not in src:
            continue
        written = {m for m in re.findall(r'["\']([\w./-]+)\.pdf["\']', src)}
        if not (written & figs):
            continue
        names = {n for n, _ in _literal_arrays(src)}
        names -= KNOWN_UNRESOLVED.get((nb, i), set())
        if names:
            problems.append(f"{nb} cell {i} writes {sorted(written & figs)} "
                            f"from literal arrays {sorted(names)}")
    assert not problems, "\n".join(problems)


def test_known_unresolved_entries_still_apply():
    """Drop an entry from KNOWN_UNRESOLVED once its sweep is computed."""
    for (nb, i), names in KNOWN_UNRESOLVED.items():
        src = "".join(json.loads((REPO / nb).read_text(encoding="utf-8"))
                      ["cells"][i]["source"])
        found = {n for n, _ in _literal_arrays(src)}
        stale = names - found
        assert not stale, f"{nb} cell {i}: no longer literal, remove {sorted(stale)}"
