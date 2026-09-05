"""Supplementary tables must be numbered in the order the text first cites them.

They were not. The section held the blocks in the order they had accumulated,
so a reader following the main text met Table B.5 before B.4, then jumped to
B.9 and back to B.6. Nothing enforces this in LaTeX, which numbers floats by
position in the file, so it can only be held by a test.

One case needed the producing script changed rather than the document. The
per-specimen G/Gc table and the angle-band table were emitted as a single
generated block, and they are first cited in the methods and in Section 4.10
respectively, so no placement of one block could satisfy both.
``make_fig20_supplement`` now writes them as two blocks.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
AUX = REPO / "manuscript" / "manscript_revision_001.aux"

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")


def _supplement_labels(tex):
    s = tex.index(r"\section{Supplementary tables}")
    e = tex.index(r"\section{Supplementary figures}")
    return re.findall(r"\\label\{(tab:[^}]+)\}", tex[s:e])


def test_the_supplement_follows_first_citation_order():
    tex = TEX.read_text(encoding="utf-8")
    body = tex[:tex.index("\\appendix")]
    positions = []
    for lab in _supplement_labels(tex):
        m = re.search(r"\\ref\{" + re.escape(lab) + r"\}", body)
        assert m, f"{lab} is in the supplement but never cited in the main text"
        positions.append((body[:m.start()].count("\n") + 1, lab))

    ordered = sorted(positions)
    assert positions == ordered, (
        "the supplementary tables are out of citation order.\n  as printed: "
        + ", ".join(l for _, l in positions)
        + "\n  should be:  " + ", ".join(l for _, l in ordered))


def test_the_two_fig20_tables_are_separate_blocks():
    """Guard the fix: one block again would force them adjacent."""
    tex = TEX.read_text(encoding="utf-8")
    B = "% >>> BEGIN GENERATED TABLE: {n} (scripts/inline_tables.py) <<<"
    E = "% >>> END GENERATED TABLE: {n} <<<"
    for name, label in (("table_S_fig20_GGc", "tab:fig20_GGc"),
                        ("table_S_fig20_bands", "tab:fig20_bands")):
        m = re.search(re.escape(B.format(n=name)) + r".*?" + re.escape(E.format(n=name)),
                      tex, re.S)
        assert m, f"{name} has no marker block"
        labs = re.findall(r"\\label\{(tab:[^}]+)\}", m.group(0))
        assert labs == [label], f"{name} contains {labs}, expected only {label}"

    from scripts import inline_tables
    assert "table_S_fig20_bands" in inline_tables.TABLES, (
        "inline_tables would not maintain the new block")


@pytest.mark.skipif(not AUX.is_file(), reason="document not compiled")
def test_the_rendered_numbers_ascend_with_citation():
    """The printed numbers, not just the source order."""
    tex = TEX.read_text(encoding="utf-8")
    body = tex[:tex.index("\\appendix")]
    num = dict(re.findall(r"\\newlabel\{(tab:[^}]+)\}\{\{([^}]+)\}",
                          AUX.read_text(encoding="utf-8", errors="ignore")))
    seen = []
    for lab in _supplement_labels(tex):
        m = re.search(r"\\ref\{" + re.escape(lab) + r"\}", body)
        n = num.get(lab)
        if m and n and re.fullmatch(r"B\.\d+", n):
            seen.append((body[:m.start()].count("\n") + 1, int(n.split(".")[1])))
    assert seen, "no supplementary table numbers resolved"
    assert [k for _, k in seen] == sorted(k for _, k in seen), (
        f"printed numbers do not ascend with citation: {seen}")


def test_the_document_has_no_unbalanced_braces():
    """A surplus closing brace compiles with a recoverable error and no warning.

    One survived a hand edit to a block ``patch_stress_field_text.py`` owns:
    right page count, no undefined references, "Too many }'s" buried in the
    log. Counting ``\\rev{`` openings does not catch it; this counts both.
    """
    t = TEX.read_text(encoding="utf-8")
    src = re.sub(r"(?<!\\)%.*", "", t)
    src = src.replace(r"\{", "").replace(r"\}", "")
    assert src.count("{") == src.count("}"), (
        f"unbalanced braces: {src.count('{')} open, {src.count('}')} close")


def test_the_compile_log_records_no_errors():
    log = REPO / "manuscript" / "manscript_revision_001.log"
    if not log.is_file():
        pytest.skip("document not compiled")
    errs = [l for l in log.read_text(errors="ignore").splitlines() if l.startswith("!")]
    assert not errs, f"LaTeX reported errors on the last build: {errs[:3]}"
