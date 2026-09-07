"""A clean checkout must fail informatively, not cryptically.

``reproduce_all`` reads the fourteen field archives, which are tracked, and the
mid-plane displacement cache, which is not: the cache is keyed on a hash of the
specimen geometry and material, so committing it would pin a stale solve. On a
fresh clone the cache is therefore absent.

``corridor_table`` used to loop over an empty file list, build an empty frame,
and fail several calls later inside ``sort_values`` with ``KeyError: 'rock'``,
which names nothing a reader could act on. It now raises immediately and says
what to run. ``displacement_symmetry`` already returned a blocked row instead,
and that behaviour is pinned here so the two stay consistent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_corridor_table_names_the_missing_cache(tmp_path):
    from tools import displacement_corridor as dc
    # tmp_path has the layout but none of the cached solutions
    (tmp_path / "outputs" / "fields").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as e:
        dc.corridor_table(root=tmp_path)
    msg = str(e.value)
    assert "execute_notebooks" in msg, (
        "the error no longer tells the reader what to run")
    assert "_cache" in msg


def test_symmetry_reports_blocked_rather_than_raising(tmp_path):
    from tools import displacement_symmetry as ds
    (tmp_path / "outputs" / "fields").mkdir(parents=True)
    row = ds.specimen_symmetry(1, root=tmp_path)
    assert row["status"] == "blocked"
    assert "no cached displacement solution" in row["reason"]


def test_the_readme_states_the_notebook_dependency():
    r = (REPO / "README.md").read_text(encoding="utf-8")
    assert "On a fresh clone, run the notebooks first" in r, (
        "the README no longer warns that reproduce_all needs the notebooks on "
        "a clean checkout")


def test_the_tracked_field_archives_are_present():
    npz = sorted((REPO / "outputs" / "fields" / "fields_npz").glob("*.npz"))
    assert len(npz) == 14, f"{len(npz)} field archives tracked, expected 14"
