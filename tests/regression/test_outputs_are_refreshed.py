"""No published table may be older than the fields it is derived from.

``test_generators_are_reachable`` checks that every ``scripts/make_*.py`` is
called from a notebook, because a generator nothing invokes produces a figure
nobody refreshes. It does not cover producers living in ``tools/``, and it
cannot see through an f-string path, so a table can go stale in either of two
ways it will never notice.

Both happened. Over one revision cycle ``outputs/tables/`` accumulated eight
files that no code wrote any longer and that contradicted the manuscript:

* ``strength_anisotropy_ratios`` gave 1.52 and 1.87 where the raw measurements
  and the paper give 1.37 and 1.75, because it carried inflated replicate
  counts. Anyone auditing the paper from ``outputs/`` would have concluded the
  published ratios were wrong.
* ``ATI_fitted_parameters`` gave eta = 0.286 against the published 0.193, under
  a name one character different from the live ``ati_envelope_parameters``.
* ``strain_energy_partition`` held the three-mode tensile/shear/mixed
  classification that this revision withdrew, five months after the four-class
  scheme replaced it.
* ``combined_field_summary`` sat in ``tables/`` after the writer had moved to
  ``figures/``, so the copy a reader would find was three weeks behind.

Rather than enumerate producers, which is what the f-strings defeat, this test
asserts the invariant those failures all violate: the exported fields are the
expensive upstream artefact, and everything derived from them must be at least
as new. A file older than the fields was not refreshed by the last full run,
whatever wrote it and wherever that writer lives.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import fabric_tractions as ft
from tools import output_dirs

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / output_dirs.TABLE_DIR

#: Tables whose content does not derive from the solved fields, so their age
#: carries no information. Keep this list short and justified; an entry here is
#: an assertion that re-running the solver cannot change the file.
FIELD_INDEPENDENT = {
    # Digitized laboratory traces and the specimen pairing they imply. Inputs.
    "sample_pairing_manifest",
}

FIELDS = ft.field_files()
needs_fields = pytest.mark.skipif(
    not FIELDS, reason="fields not exported yet; nothing to date outputs against")


def _oldest_field_mtime() -> float:
    return min(os.path.getmtime(str(p)) for p in FIELDS)


def _tables():
    return sorted(TABLE_DIR.glob("*.csv"))


@needs_fields
def test_no_published_table_predates_the_fields():
    oldest = _oldest_field_mtime()
    stale = [p.name for p in _tables()
             if p.stem not in FIELD_INDEPENDENT
             and os.path.getmtime(p) < oldest]
    assert not stale, (
        "these tables are older than the exported fields, so the last full run "
        f"did not refresh them: {stale}. Either a producer is no longer being "
        "invoked, or the file is a leftover from a path change and should be "
        "removed. Do not simply touch the file: the numbers in it are the "
        "question, not its timestamp."
    )


@needs_fields
def test_every_table_is_written_by_something_in_the_tree():
    """A weaker check that still catches a file with no writer at all.

    Producers reach these paths through f-strings, so the basename may not
    appear anywhere. Matching on the stem's leading component catches the
    orphans without demanding a literal.
    """
    sources = []
    for pattern in ("tools/**/*.py", "scripts/*.py", "*.ipynb"):
        sources.extend(REPO.glob(pattern))
    blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in sources)

    from tools.lithology import LITHOLOGIES
    # A producer writes "prefix_{ROCK.key}.csv", so the literal basename never
    # appears. Strip a trailing lithology key before looking for the prefix.
    suffixes = sorted(LITHOLOGIES, key=len, reverse=True)

    orphans = []
    for path in _tables():
        stem = path.stem
        if stem in blob:
            continue
        root = next((stem[: -len(sfx) - 1] for sfx in suffixes
                     if stem.endswith("_" + sfx)), None)
        if root and root in blob:
            continue
        orphans.append(path.name)

    assert not orphans, (
        f"no file in tools/, scripts/ or the notebooks writes these: {orphans}. "
        "An unwritten table cannot be regenerated and will drift from the "
        "manuscript silently."
    )


def test_the_exemption_list_stays_justified():
    """Every exemption must name a file that exists, so the list cannot rot."""
    present = {p.stem for p in _tables()}
    missing = sorted(FIELD_INDEPENDENT - present)
    assert not missing, (
        f"exempted tables that no longer exist: {missing}. Remove them from "
        "FIELD_INDEPENDENT so the list keeps meaning what it says."
    )
