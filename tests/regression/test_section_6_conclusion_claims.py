"""The Conclusion, the Abstract and the Highlights, against the outputs.

Two things had gone wrong, and the second is the one that matters.

The fracture-orientation span was quoted as 85.7 to 99.5 degrees while the
table it cites gives 85.1 to 97.1; 99.5 is the full-segment orientation of the
gneiss 30 degree specimen from before both traces were clipped to 0.85R.

The mechanism sequence was stated as matrix failure, then weak-plane sliding,
then opening, in both lithologies. That is the schist's route. In the gneiss
weak-plane sliding never becomes the governing class: it reaches 0.215 against
matrix shear at 0.238 at 60 degrees and opening succeeds matrix shear
directly. Section 5.2 says so explicitly, so the Conclusion, the Abstract and
the second Highlight all contradicted the Discussion. The dominant-class
sequence is recomputed here rather than asserted.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
HL = REPO / "manuscript" / "highlights.txt"
G, P = "Augen gneiss", "Psammitic schist"
CLASSES = ("WT", "WS", "MT", "MS")

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript absent")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def _dominant_sequence():
    f = pd.read_csv(TAB / "fourclass_area_fractions.csv")
    out = {}
    for rock, g in f.groupby("rock"):
        g = g.sort_values("angle_deg")
        out[rock] = [max(CLASSES, key=lambda c: r[c]) for _, r in g.iterrows()]
    return out


def test_the_fracture_orientation_span_matches_the_cited_table(tex):
    t = pd.read_csv(TAB / "trace_comparison_metrics.csv")
    o = t.observed_orientation_deg
    assert o.min() == pytest.approx(85.1, abs=0.05)
    assert o.max() == pytest.approx(97.1, abs=0.05)
    m = re.search(r"loading-subparallel at every fabric angle,\s*\n?\s*"
                  r"spanning \$([\d.]+)\^\\circ\$ to \$([\d.]+)\^\\circ\$", tex)
    assert m, "the Conclusion's span sentence has changed shape"
    assert float(m.group(1)) == pytest.approx(o.min(), abs=0.05)
    assert float(m.group(2)) == pytest.approx(o.max(), abs=0.05), (
        "the Conclusion quotes a span the table it cites does not support")
    # every fracture is within 10 degrees of the loading diameter
    assert (o - 90.0).abs().max() < 10.0


def test_weak_plane_sliding_never_governs_in_the_gneiss():
    seq = _dominant_sequence()
    assert "WS" not in seq[G], (
        f"weak-plane sliding now governs somewhere in the gneiss: {seq[G]}. "
        "The Conclusion, Abstract and Highlights all state that it does not, "
        "and would need rewording.")
    assert "WS" in seq[P], "weak-plane sliding no longer governs in the schist"
    assert seq[P][seq[P].index("WS")] == "WS"
    f = pd.read_csv(TAB / "fourclass_area_fractions.csv").set_index(["rock", "angle_deg"])
    assert f.loc[(G, 60.0)].WS == pytest.approx(0.215, abs=0.002)
    assert f.loc[(G, 60.0)].MS == pytest.approx(0.238, abs=0.002)


def test_both_rocks_end_in_weak_plane_opening_and_start_in_matrix_shear():
    seq = _dominant_sequence()
    for rock in (G, P):
        assert seq[rock][0] == "MS", f"{rock} no longer starts in matrix shear"
        assert seq[rock][-1] == "WT", f"{rock} no longer ends in weak-plane opening"


def test_the_conclusion_does_not_generalise_the_sliding_stage(tex):
    assert "and the two\nlithologies take different routes" in tex, (
        "the Conclusion has reverted to a single sequence for both rocks")
    assert "reached through weak-plane sliding in\nthe schist" in tex, (
        "the Abstract has reverted to a single sequence for both rocks")


def test_the_anisotropy_ratios_quoted_in_the_conclusion(tex):
    a = pd.read_csv(TAB / "strength_anisotropy_ratios.csv").set_index("rock")
    assert a.loc[G, "tensile_ratio"] == pytest.approx(1.37, abs=0.005)
    assert a.loc[G, "compressive_ratio"] == pytest.approx(1.60, abs=0.005)
    assert a.loc[P, "tensile_ratio"] == pytest.approx(2.57, abs=0.005)
    assert a.loc[P, "compressive_ratio"] == pytest.approx(2.59, abs=0.005)
    # the claim that the two modes separate in one rock and not the other
    gsep = abs(a.loc[G, "compressive_ratio"] - a.loc[G, "tensile_ratio"]) / a.loc[G, "tensile_ratio"]
    psep = abs(a.loc[P, "compressive_ratio"] - a.loc[P, "tensile_ratio"]) / a.loc[P, "tensile_ratio"]
    assert gsep > 0.10 and psep < 0.02, (
        f"the mode separation is now {gsep:.2f} in the gneiss and {psep:.2f} "
        "in the schist; the Conclusion says it separates in one rock only")
    assert int(a.loc[G, "tensile_min_angle_deg"]) == 75
    assert int(a.loc[P, "tensile_min_angle_deg"]) == 75


@pytest.mark.skipif(not HL.is_file(), reason="highlights absent")
def test_the_highlights_are_elsevier_compliant_and_agree_across_formats():
    bullets = [l for l in HL.read_text().splitlines()
               if l.strip() and l.strip() != "Highlights"]
    assert 3 <= len(bullets) <= 5, "Elsevier allows three to five highlights"
    for b in bullets:
        assert len(b) <= 85, f"highlight is {len(b)} characters: {b}"
    rtf = REPO / "manuscript" / "highlights.rtf"
    docx = REPO / "manuscript" / "highlights.docx"
    if rtf.is_file():
        s = rtf.read_text(encoding="utf-8", errors="replace")
        for b in bullets:
            assert b in s, f"highlights.rtf is out of step: {b}"
    if docx.is_file():
        t = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
        for b in bullets:
            assert b in t, f"highlights.docx is out of step: {b}"
    assert not any("gives way to weak-plane sliding" in x for x in bullets)
