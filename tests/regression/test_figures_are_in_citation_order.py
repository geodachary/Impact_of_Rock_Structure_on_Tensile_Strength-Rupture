"""Figures must be numbered in order of first mention, and all must be mentioned.

LaTeX numbers floats by position in the source, so nothing enforces this. Four
main-body figures and one appendix figure were out of order, and two figures
(the strength-anisotropy panels and the energy-localization panels) were
referenced only from inside another figure's caption, never in the running
text, so a reader following the argument was never sent to them.

First mention means first mention in prose: a cross-reference inside a caption
does not count, which is what hid the two unmentioned figures.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
AUX = REPO / "manuscript" / "manscript_revision_001.aux"

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")


def _survey():
    t = TEX.read_text(encoding="utf-8")
    floats = [(m.start(), m.end()) for m in
              re.finditer(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", t, re.S)]
    in_float = lambda q: any(a <= q < b for a, b in floats)
    out = []
    for m in re.finditer(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", t, re.S):
        lab = re.search(r"\\label\{(fig:[^}]+)\}", m.group(0))
        if not lab:
            continue
        lab = lab.group(1)
        first = next((r.start() for r in re.finditer(r"\\ref\{" + re.escape(lab) + r"\}", t)
                      if not in_float(r.start())), None)
        out.append((m.start(), lab, first))
    return t, out


def test_every_figure_is_mentioned_in_the_text():
    _, survey = _survey()
    silent = [lab for _, lab, first in survey if first is None]
    assert not silent, (
        f"these figures are never referred to in the running text: {silent}. "
        "A reference inside another figure's caption does not send the reader "
        "there.")


def test_figures_are_numbered_in_order_of_first_mention():
    t, survey = _survey()
    app = t.index("\\appendix")
    for name, group in (("main body", [s for s in survey if s[0] < app]),
                        ("appendix", [s for s in survey if s[0] >= app])):
        firsts = [f for _, _, f in group if f is not None]
        assert firsts == sorted(firsts), (
            f"{name} figures are out of first-mention order:\n  " +
            "\n  ".join(f"{lab} first mentioned at offset {f}"
                        for _, lab, f in group if f is not None))


@pytest.mark.skipif(not AUX.is_file(), reason="document not compiled")
def test_the_printed_numbers_ascend_with_first_mention():
    t, survey = _survey()
    num = dict(re.findall(r"\\newlabel\{(fig:[^}]+)\}\{\{([^}]+)\}",
                          AUX.read_text(errors="ignore")))
    for prefix, pat in (("main", r"\d+"), ("appendix", r"C\.\d+")):
        seq = [(f, num[lab]) for _, lab, f in survey
               if f is not None and lab in num and re.fullmatch(pat, num[lab])]
        keys = [int(n.split(".")[-1]) for _, n in seq]
        assert keys == sorted(keys), f"{prefix} figure numbers do not ascend: {seq}"
