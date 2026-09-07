"""README section 15 must quote REFERENCES.bib, not a hand-written recollection.

The first draft of that section was written from memory and three entries named
the wrong journal outright: Horgan and Simmonds was given as Applied Mechanics
Reviews rather than Composites Engineering, Barnett and Lothe as Physica
Norvegica rather than Journal of Physics F, and Jaeger as Geofisica Pura e
Applicata rather than Geological Magazine. Two more invented a publisher the
bibliography does not carry.

The strings are now generated from REFERENCES.bib. This checks that every key
the section relies on is present in that file and that the year and journal in
the README agree with it, so a later hand edit cannot reintroduce the problem.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BIB = REPO / "REFERENCES.bib"
README = REPO / "README.md"

KEYS = [
    "Lekhnitskii1964TheoryBody", "Claesson2002BrazilianSolution",
    "Hondros1959TheConcrete", "Toupin1965", "HorganSimmonds1994",
    "Roylance2000LaminatedPlates", "Eshelby1957", "MoriTanaka1973",
    "Ip2021ImpactsRocks", "Stroh1958DislocationsElasticity",
    "Barnett1973ElasticTheory", "Sih1965OnBodies", "Erdogan1963OnShear",
    "Griffith1921", "Hutchinson1991MixedMaterials", "Chen1998FractureMethod",
    "Jaeger1960ShearRocks", "Anderson1951TheBritain",
    "ISRM1977SuggestedMaterials", "BIENIAWSKI1979138", "ISRM197847",
    "Schneider2012NIHAnalysis", "Acharya2026ImpactSoftware",
]

pytestmark = pytest.mark.skipif(not BIB.is_file(), reason="REFERENCES.bib absent")


def _entries():
    text = BIB.read_text(errors="replace")
    return {m.group(2).strip(): m.group(3)
            for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S)}


def _field(body, name):
    # the trailing newline is optional: the last field in an entry is followed
    # directly by the closing brace, which is where the doi sits
    m = re.search(name + r"\s*=\s*\{+(.*?)\}+\s*,?\s*(?:\n|$)", body, re.S)
    return " ".join(m.group(1).split()).replace("{", "").replace("}", "") if m else ""


def test_every_method_reference_is_in_the_bib():
    e = _entries()
    missing = [k for k in KEYS if k not in e]
    assert not missing, f"REFERENCES.bib is missing {missing}"


def test_the_software_cites_itself_with_the_zenodo_doi():
    e = _entries()
    b = e["Acharya2026ImpactSoftware"]
    assert _field(b, "doi") == "10.5281/zenodo.22591642"
    assert "Software" in _field(b, "howpublished")


@pytest.mark.parametrize("key,journal_fragment", [
    ("HorganSimmonds1994", "Composites Engineering"),
    ("Barnett1973ElasticTheory", "Journal of Physics F"),
    ("Jaeger1960ShearRocks", "Geological Magazine"),
    ("Hondros1959TheConcrete", "Aust. J. Appl. Sci."),
])
def test_the_three_that_were_wrong_match_the_bib(key, journal_fragment):
    """These named the wrong journal when written from memory."""
    b = _entries()[key]
    assert journal_fragment in _field(b, "journal")
    readme = README.read_text(encoding="utf-8")
    assert journal_fragment in readme, (
        f"README no longer quotes the journal REFERENCES.bib gives for {key}")


def test_the_readme_years_agree_with_the_bib():
    e = _entries()
    readme = README.read_text(encoding="utf-8")
    start = readme.index("## 15. Methods implemented")
    end = readme.index("## 16. License and citation")
    section = readme[start:end]
    for k in KEYS:
        if k == "Acharya2026ImpactSoftware":
            continue
        yr = _field(e[k], "year")
        assert f"({yr})" in section, (
            f"the README section does not carry {yr} for {k}; the citation "
            "strings are generated from REFERENCES.bib and should agree")


def test_the_readme_points_at_a_file_that_exists():
    readme = README.read_text(encoding="utf-8")
    assert "`REFERENCES.bib` in the repository root" in readme
    assert "manuscript/references.bib" not in readme, (
        "the README points at the manuscript bibliography, which is not "
        "shipped with the repository")
