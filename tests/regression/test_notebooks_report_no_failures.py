"""A notebook that ran but failed must not look like a notebook that worked.

``crack_path_suite`` catches per-specimen exceptions and prints
``[smoke_all] sid=N FAILED: ...`` instead of raising. That is deliberate, so
one bad specimen does not lose the other thirteen, but it means the cell
returns normally, the notebook exits zero, and ``execute_notebooks.py`` reports
"24/24 code cells produced output" while every specimen in fact failed.

That is exactly what happened. Adding a ``phase`` argument to
``weak_plane_weight_field`` activated a dead branch in ``_safe_wp_weight`` that
referred to an undefined name, all fourteen specimens raised NameError, and the
three combined crack-path figures silently stopped being refreshed. No guard
noticed: the freshness guards read tables, and the tables were fine.

The notebooks are the published deliverable, so their stored output is checked
here the way a reader would read it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb",
             "Tensile_general_plots.ipynb"]

# Substrings that mean a stage gave up. Kept literal and short so the test says
# what it found rather than matching a regex nobody can read.
FAILURE_MARKERS = (
    "FAILED:",
    "No successful runs",
    "is not defined",
    "Traceback (most recent call last)",
)


def _cell_text(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "error":
            out.append("Traceback (most recent call last)")
            out.append(str(o.get("evalue", "")))
            out.extend(o.get("traceback", []))
        t = o.get("text")
        if t:
            out.append("".join(t) if isinstance(t, list) else str(t))
        data = o.get("data", {})
        if "text/plain" in data:
            v = data["text/plain"]
            out.append("".join(v) if isinstance(v, list) else str(v))
    return "\n".join(out)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_cell_stored_an_error_output(name):
    path = REPO / name
    if not path.is_file():
        pytest.skip(f"{name} absent")
    doc = json.loads(path.read_text(encoding="utf-8"))
    bad = [i for i, c in enumerate(doc["cells"])
           if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    assert not bad, (
        f"{name} stores an error output in cell(s) {bad}. The notebooks are "
        "published as-is, so a reader would open this and see a traceback.")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_cell_reports_a_failed_stage(name):
    path = REPO / name
    if not path.is_file():
        pytest.skip(f"{name} absent")
    doc = json.loads(path.read_text(encoding="utf-8"))
    hits = []
    for i, c in enumerate(doc["cells"]):
        if c.get("cell_type") != "code":
            continue
        text = _cell_text(c)
        for marker in FAILURE_MARKERS:
            if marker in text:
                line = next((l.strip() for l in text.splitlines() if marker in l), marker)
                hits.append(f"cell {i}: {line[:120]}")
                break
    assert not hits, (
        f"{name} reports a failed stage in its stored output:\n  "
        + "\n  ".join(hits)
        + "\nThe cell returned normally and the notebook exited zero, which is "
          "why nothing else caught this. Re-run the notebook after fixing the "
          "underlying error; do not clear the output to silence the test.")
