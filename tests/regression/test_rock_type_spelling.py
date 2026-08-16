"""Rock-type spelling must not silently drop a lithology.

The strength table has carried both ``Psammatic`` and ``Psammitic``. Notebook
cells select a lithology with an equality test and then guard the result with
``if len(...) > 0``, so the wrong spelling produces an empty frame and the
lithology disappears from a figure without raising anything. That is exactly
what happened to the schist fit in the strength-envelope figure.

These tests pin the canonical spelling, the tolerance of the loader, and the
absence of the stale literal from the notebooks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools import lithology as lith
from tools.data_io import canonical_rock_type, load_specimen_table

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb",
             "Tensile_general_plots.ipynb"]


def _code(nb):
    d = json.loads((REPO / nb).read_text(encoding="utf-8"))
    return [(i, "".join(c["source"])) for i, c in enumerate(d["cells"])
            if c["cell_type"] == "code"]


@pytest.mark.parametrize("spelling", [
    "Psammitic schist", "Psammatic Schist", "psammitic schist",
    "  PSAMMATIC   SCHIST  ", "psammitic_schist",
])
def test_every_spelling_canonicalises(spelling):
    assert canonical_rock_type(spelling) == lith.PSAMMITIC_SCHIST.display_name


def test_gneiss_canonicalises():
    assert canonical_rock_type("augen gneiss") == lith.AUGEN_GNEISS.display_name


def test_unknown_rock_type_raises():
    with pytest.raises(ValueError):
        canonical_rock_type("Slate")


def test_loaded_table_uses_canonical_spelling():
    d = load_specimen_table()
    assert set(d.Rock_type.unique()) == {lith.AUGEN_GNEISS.display_name,
                                         lith.PSAMMITIC_SCHIST.display_name}


def test_both_lithologies_select_seven_specimens():
    """The failure mode this guards: a filter that quietly returns nothing."""
    d = load_specimen_table()
    for name in (lith.AUGEN_GNEISS.display_name, lith.PSAMMITIC_SCHIST.display_name):
        n = (d.Rock_type == name).sum()
        assert n == 7, f"{name} selected {n} rows, expected 7"


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_no_stale_rock_type_literal(nb):
    """A quoted 'Psammatic Schist' would select zero rows from the current table."""
    bad = [i for i, s in _code(nb)
           if '"Psammatic Schist"' in s or "'Psammatic Schist'" in s]
    assert not bad, (
        f"{nb} compares Rock_type against the stale spelling in cells {bad}; "
        "the current table uses 'Psammitic schist' and the match would be empty")


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_output_filenames_use_canonical_spelling(nb):
    bad = {}
    for i, s in _code(nb):
        hits = re.findall(r"psammatic_schist_[A-Za-z_]*\.(?:pdf|png|svg|csv|npz)", s)
        if hits:
            bad[i] = sorted(set(hits))
    assert not bad, f"{nb} writes legacy-spelled output files: {bad}"


def test_replicate_table_is_canonicalised():
    """The replicate file carries its own capitalisation; it must be normalised."""
    from tools.data_io import load_replicate_table
    d = load_replicate_table()
    assert set(d.Rock_type.unique()) == {lith.AUGEN_GNEISS.display_name,
                                         lith.PSAMMITIC_SCHIST.display_name}
    for name in (lith.AUGEN_GNEISS.display_name, lith.PSAMMITIC_SCHIST.display_name):
        assert (d.Rock_type == name).sum() > 0, f"{name} selects no replicates"


def test_anisotropy_ratios_use_canonical_labels():
    """Notebook code indexes this result by display name; the labels must match."""
    from tools import strength_anisotropy as sa
    r = sa.anisotropy_ratios().set_index("lithology")
    for name in (lith.AUGEN_GNEISS.display_name, lith.PSAMMITIC_SCHIST.display_name):
        assert name in r.index, f"{name} missing from anisotropy_ratios"
        assert r.loc[name, "tensile_anisotropy"] > 1.0


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_no_direct_reads_of_the_data_tables(nb):
    """Direct reads bypass canonicalisation and silently empty every filter."""
    bad = [i for i, s in _code(nb)
           if re.search(r"read_csv\(\s*(?:CSV_PATH|[\"']selected_all_samples|"
                        r"[\"']tensile_samples_data)", s)]
    assert not bad, (
        f"{nb} reads a data table directly in cells {bad}; use "
        "tools.data_io.load_specimen_table / load_replicate_table so Rock_type "
        "arrives canonicalised")


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_rock_name_literals_match_the_canonical_display_names(nb):
    """A filter literal that is not a display name selects nothing."""
    canon = {lith.AUGEN_GNEISS.display_name, lith.PSAMMITIC_SCHIST.display_name}
    variants = {"Augen Gneiss", "Psammitic Schist", "Psammatic Schist",
                "Augen gneiss", "Psammitic schist", "Psammatic schist"}
    for i, s in _code(nb):
        for v in variants - canon:
            assert f'"{v}"' not in s and f"'{v}'" not in s, (
                f"{nb} cell {i} uses the non-canonical literal {v!r}")
