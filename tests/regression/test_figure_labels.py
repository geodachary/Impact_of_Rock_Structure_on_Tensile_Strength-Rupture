"""Axis labels must not carry LaTeX escapes.

Matplotlib renders these labels with mathtext, and ``text.usetex`` is set to
False throughout this project. A LaTeX escape such as ``\\%`` therefore reaches
the figure literally: the axis reads ``Delta Failure (\\%)`` instead of
``Delta Failure (%)``.

It is an easy mistake because the same string is correct in the manuscript, and
it survives review because the figure still builds and the label still looks
almost right. Four labels carried it before this test existed.

The check is on the source rather than on rendered output, because reading the
text back out of a PDF would not distinguish a literal backslash from a
rendering artefact.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Where figures are drawn. The manuscript-writing scripts are excluded: LaTeX
#: escapes are correct in the text they emit.
SOURCES = sorted(
    [p for p in (REPO / "scripts").glob("*.py")
     if p.name not in {"patch_stress_field_text.py", "make_ati_supplement.py",
                       "make_eshelby_supplement.py", "make_fig20_supplement.py",
                       "inline_tables.py"}]
    + list((REPO / "tools").glob("*.py"))
    + list((REPO / "tools" / "analysis").glob("*.py")))

#: A label-setting call whose argument contains a backslash-escaped percent.
LABEL_WITH_ESCAPE = re.compile(
    r"(?:set_xlabel|set_ylabel|set_title|suptitle|set_zlabel)\s*\([^)]*\\\\?%")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_latex_percent_escape_in_axis_labels(path):
    text = path.read_text(encoding="utf-8")
    hits = [m.group(0)[:70] for m in LABEL_WITH_ESCAPE.finditer(text)]
    assert not hits, (
        f"{path.name} escapes a percent sign in an axis label: {hits}. "
        "text.usetex is False, so the backslash is drawn literally; write %.")


def test_usetex_stays_disabled():
    """The fix above is only correct while LaTeX rendering is off."""
    import matplotlib
    matplotlib.use("Agg")
    from tools.plot_style import apply_plot_style

    apply_plot_style()
    assert matplotlib.rcParams["text.usetex"] is False, (
        "text.usetex was enabled; the plain % signs in axis labels would now "
        "need escaping again")
