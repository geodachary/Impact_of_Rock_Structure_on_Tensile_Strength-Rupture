"""Numbers quoted in Results must agree with the tools that produce them.

The anisotropy ratios in Section 4 had drifted: the text carried 1.52 and 1.87
for the augen gneiss where the replicate table gives 1.37 and 1.75, and a 23%
separation where the computed value is 28%. The schist values were correct,
which is how the error survived a read-through.

Anything the manuscript states that a tool can recompute is checked here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import strength_anisotropy as sa

TEX = Path(__file__).resolve().parents[2] / "manuscript/manscript_revision_001.tex"

#: The manuscript source is not part of the software release, so tests that
#: read it skip on a clean clone rather than failing there.
needs_manuscript = pytest.mark.skipif(
    not TEX.is_file(), reason="manuscript source not shipped with the release")

pytestmark = needs_manuscript



@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ratios():
    return sa.anisotropy_ratios().set_index("lithology")


def test_tensile_anisotropy_ratios_are_quoted_correctly(tex, ratios):
    for lit in ("Augen gneiss", "Psammitic schist"):
        v = ratios.loc[lit, "tensile_anisotropy"]
        assert f"{v:.2f}" in tex, f"{lit} tensile ratio {v:.2f} not in the manuscript"


def test_compressive_anisotropy_ratios_are_quoted_correctly(tex, ratios):
    for lit in ("Augen gneiss", "Psammitic schist"):
        v = ratios.loc[lit, "compressive_anisotropy"]
        assert f"${v:.2f}$" in tex, f"{lit} compressive ratio {v:.2f} not in the manuscript"


def test_relative_separation_is_quoted_correctly(tex, ratios):
    for lit in ("Augen gneiss", "Psammitic schist"):
        v = abs(ratios.loc[lit, "relative_separation_pct"])
        assert f"${v:.0f}$\\%" in tex, f"{lit} separation {v:.0f}% not in the manuscript"


def test_superseded_anisotropy_values_are_gone(tex):
    """Match the numbers themselves, not one phrasing of them.

    The first version of this test looked for "1.52 for augen" and
    "$23$\\% relative", and missed a figure caption that wrote the same
    superseded values as "1.52 and 1.87 for augen gneiss" and
    "$23$\\% and $63$\\% relative". Anchoring on the digits closes that gap.
    """
    body = tex.split("\\section*{Appendices}")[0]
    # 1.52 and 1.87 also occur legitimately elsewhere (a G/Gc coefficient of
    # variation), so the superseded values are only flagged where they appear
    # in a sentence about anisotropy ratios.
    for m in re.finditer(r"1\.52|1\.87|\$23\$\\%", body):
        window = body[max(0, m.start() - 220):m.end() + 220].lower()
        if "anisotrop" in window or "ratios are" in window:
            raise AssertionError(
                f"superseded anisotropy value {m.group(0)} near: "
                f"...{body[max(0, m.start() - 90):m.end() + 60]}...")


def test_tensile_minimum_orientation(tex, ratios):
    """Both lithologies are weakest at 75 degrees, which the text relies on."""
    for lit in ("Augen gneiss", "Psammitic schist"):
        assert int(ratios.loc[lit, "tensile_min_angle"]) == 75
