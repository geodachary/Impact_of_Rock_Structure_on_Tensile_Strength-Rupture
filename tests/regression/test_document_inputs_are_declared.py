"""Cited figures are either generated or declared laboratory inputs.

``stage_sync_document_figures`` skips any cited figure absent from
``outputs/figures``, treating it as an input. That is right for photographs,
schematics and records prepared outside this repository, but it also hid two
figures that should have been generated and no longer could be, dated
2026-08-11 with no producer anywhere. They are now written by
``make_strength_anisotropy_figures.py``.

This keeps the two categories apart, so a new orphan cannot hide among the
declared inputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
FIGDIR = REPO / "outputs" / "figures"

#: Cited graphics with no generated counterpart, and why each is an input.
DECLARED_INPUTS = {
    "augen_geniss_drawing.pdf":       "hand-drawn fabric schematic",
    "schist_drawing.pdf":             "hand-drawn fabric schematic",
    "disc_draw.pdf":                  "hand-drawn loading schematic",
    "MOE_lab_test.jpg":               "laboratory photograph",
    "Tensile_strength_sample_test.png": "laboratory photograph",
    # Laboratory records prepared outside this repository, by design. The
    # archived data are per-specimen scalars (UCS, axial strain, cohesion,
    # friction angle); the load-displacement traces and triaxial circles these
    # two plot are not part of the release and are not meant to be. They are
    # inputs to the document in the same sense as the photographs.
    "axial_stress_strain.pdf":        "lab record, prepared outside the repository",
    "mohr_circles_with_envelope.pdf": "triaxial envelope, prepared outside the repository",
    # Written straight into manuscript/ by their generators rather than via
    # outputs/figures, so they are generated but never synced.
    "fig_S_ati_profiles.pdf":         "written by make_ati_supplement.py",
    "fig_S_GGc_monotonicity.pdf":     "written by make_fig20_supplement.py",
}

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")


def _cited():
    tex = TEX.read_text(encoding="utf-8")
    out = set()
    for name in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
        out.add(name if "." in name else name + ".pdf")
    return out


def test_ungenerated_figures_are_exactly_the_declared_inputs():
    ungenerated = {c for c in _cited() if not (FIGDIR / c).exists()}
    undeclared = sorted(ungenerated - set(DECLARED_INPUTS))
    assert not undeclared, (
        f"these cited figures are neither generated nor declared inputs: "
        f"{undeclared}. Either a generator has stopped writing them, or they "
        "are orphans that no code can rebuild. Add a producer, or declare them "
        "here with the reason.")


def test_no_declared_input_has_quietly_become_generated():
    """Guard the guard: a stale declaration hides a real producer."""
    stale = sorted(n for n in DECLARED_INPUTS if (FIGDIR / n).exists())
    assert not stale, (
        f"{stale} are now written to outputs/figures, so they are generated "
        "and should be removed from DECLARED_INPUTS")


def test_every_cited_figure_exists_in_the_document_folder():
    missing = sorted(c for c in _cited() if not (TEX.parent / c).exists())
    assert not missing, f"cited but absent from manuscript/: {missing}"
