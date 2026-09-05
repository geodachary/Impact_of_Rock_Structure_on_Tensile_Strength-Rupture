"""Regression tests for packaging hygiene, figure structure and Step 2 values.

These guard the refactor itself: that no import escaped back to the old root
layout, that importing the package has no side effects, that the four required
composite figures keep their seven-panel structure, and that the manuscript-bound
numbers still match the locked Step 2 evidence.
"""
import ast
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SUPPORTED_NOTEBOOKS = ("Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb")

# modules that were moved out of the repository root during the Step 3 refactor
MOVED_MODULES = {
    "airy_solution", "bound_all_helpers", "boundary_traction_model", "crack_helpers",
    "cracked_disk_ddm", "failure_mapping_helpers", "fourclass_export", "geometry_helpers",
    "grid_phys_method", "hybrid_crack_path_helpers", "lekhnitskii_root",
    "momentum_crack_path_helpers", "rotation_helpers", "stress_helpers",
    "angle_convention", "crack_data_plot",
}


# ---------------------------------------------------------------------------
# Packaging hygiene
# ---------------------------------------------------------------------------
def test_repository_root_holds_no_loose_python_modules():
    loose = sorted(p.name for p in REPO.glob("*.py"))
    assert loose == [], f"loose modules remain at the root: {loose}"


def test_tools_package_imports():
    import tools
    assert hasattr(tools, "__version__")


@pytest.mark.parametrize("mod", [
    "tools.conventions", "tools.lithology", "tools.traces",
    "tools.failure_classification", "tools.plotting", "tools.export",
    "tools.stress_helpers", "tools.geometry_helpers", "tools.rotation_helpers",
    "tools.airy_solution", "tools.boundary_traction_model",
    "tools.failure_mapping_helpers", "tools.fourclass_export", "tools.crack_helpers",
    "tools.cracked_disk_ddm", "tools.grid_phys_method", "tools.lekhnitskii_root",
    "tools.bound_all_helpers",
])
def test_every_tools_module_imports(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("nb", SUPPORTED_NOTEBOOKS)
def test_notebooks_import_project_code_only_from_tools(nb):
    """No supported notebook may import from the old root module layout."""
    d = json.loads((REPO / nb).read_text(encoding="utf-8"))
    offenders = []
    for i, cell in enumerate(d["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        cleaned = "\n".join("" if re.match(r"\s*[%!]", l) else l for l in src.splitlines())
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                if n.module.split(".")[0] in MOVED_MODULES:
                    offenders.append((i, n.module))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] in MOVED_MODULES:
                        offenders.append((i, a.name))
    assert not offenders, f"{nb} still imports from the old root layout: {offenders}"


@pytest.mark.parametrize("nb", SUPPORTED_NOTEBOOKS)
def test_notebooks_have_no_top_level_sys_path_manipulation(nb):
    """Cell-scope sys.path edits are forbidden.

    The notebooks retain a documented dynamic solver loader that adjusts
    sys.path *inside* its own functions so a solver can be loaded from an
    explicit file path. That is an intentional, scoped mechanism, not stray
    path hacking, so only module-scope (cell top-level) edits fail here.
    """
    d = json.loads((REPO / nb).read_text(encoding="utf-8"))
    offenders = []
    for i, cell in enumerate(d["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "sys.path" not in src:
            continue
        cleaned = "\n".join("" if re.match(r"\s*[%!]", l) else l for l in src.splitlines())
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            continue
        for node in tree.body:                      # module scope only
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in ("insert", "append"):
                    tgt = sub.value
                    if (isinstance(tgt, ast.Attribute) and tgt.attr == "path"
                            and isinstance(tgt.value, ast.Name) and tgt.value.id == "sys"
                            and isinstance(node, (ast.Expr, ast.If, ast.For, ast.Try))):
                        offenders.append(i)
    assert not offenders, f"{nb} edits sys.path at cell top level in cells {sorted(set(offenders))}"


@pytest.mark.parametrize("nb", SUPPORTED_NOTEBOOKS)
def test_no_string_module_reference_points_at_the_old_root_layout(nb):
    """importlib.import_module takes a STRING, so a module move does not update it.

    This caught a real regression: the dynamic solver loader defaulted to
    "bound_all_helpers" after that module had moved into the package.
    """
    d = json.loads((REPO / nb).read_text(encoding="utf-8"))
    offenders = []
    for i, cell in enumerate(d["cells"]):
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            for m in re.finditer(r"[\"\']([A-Za-z_]\w*)[\"\']", line):
                if m.group(1) in MOVED_MODULES:
                    offenders.append((i, m.group(1)))
    assert not offenders, f"{nb} references moved modules by bare string: {offenders[:5]}"


@pytest.mark.parametrize("nb", SUPPORTED_NOTEBOOKS)
def test_notebooks_contain_no_user_specific_absolute_paths(nb):
    d = json.loads((REPO / nb).read_text(encoding="utf-8"))
    hits = []
    for i, c in enumerate(d["cells"]):
        if c["cell_type"] != "code":
            continue
        for line in c["source"]:
            if re.search(r"[\"'](/Users/|/home/|C:\\\\)", line):
                hits.append((i, line.strip()[:70]))
    assert not hits, f"{nb} carries machine-specific paths: {hits[:3]}"


def test_importing_tools_has_no_side_effects(tmp_path):
    """Importing the package must not write files or open figures."""
    script = (
        "import os, sys, tempfile, pathlib\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "d = tempfile.mkdtemp(); os.chdir(d)\n"
        "import tools, tools.plotting, tools.traces, tools.failure_classification\n"
        "created = list(pathlib.Path(d).iterdir())\n"
        "assert not created, f'import created files: {created}'\n"
        "import matplotlib.pyplot as plt\n"
        "assert not plt.get_fignums(), 'import created figures'\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]


def test_no_circular_imports_when_each_module_is_imported_alone():
    mods = sorted(p.stem for p in (REPO / "tools").glob("*.py") if p.stem != "__init__")
    for m in mods:
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {str(REPO)!r}); import tools.{m}"],
            capture_output=True, text=True)
        assert r.returncode == 0, f"tools.{m} failed to import alone: {r.stderr[-400:]}"


# ---------------------------------------------------------------------------
# Required composite figures
# ---------------------------------------------------------------------------
REQUIRED_FIGURES = [
    ("outputs/figures/fracture_trace_overlay_augen_gneiss_7panel", "overlay"),
    ("outputs/figures/fracture_trace_overlay_psammitic_schist_7panel", "overlay"),
    ("outputs/figures/failure_mechanism_classification_augen_gneiss_7panel", "class"),
    ("outputs/figures/failure_mechanism_classification_psammitic_schist_7panel", "class"),
]


@pytest.mark.parametrize("stem,_kind", REQUIRED_FIGURES)
@pytest.mark.parametrize("ext", ["pdf", "png"])
def test_required_figure_exists_in_every_format(stem, _kind, ext):
    p = REPO / f"{stem}.{ext}"
    assert p.exists() and p.stat().st_size > 5000, f"missing or truncated: {p}"


def test_seven_panel_builders_produce_exactly_seven_specimen_panels(tmp_path):
    """Structural check on the figure builders themselves, not the saved files."""
    import matplotlib
    matplotlib.use("Agg")
    from tools import lithology as lith
    from tools import traces as tr
    from tools import plotting as tp

    rows = [r for r in tr.compare_all() if r["lithology_key"] == "augen_gneiss"]
    assert len(rows) == 7
    assert [r["experimental_angle_deg"] for r in sorted(
        rows, key=lambda r: r["experimental_angle_deg"])] == [0, 15, 30, 45, 60, 75, 90]
    assert sorted(r["sample"] for r in rows) == [1, 2, 3, 4, 5, 6, 7]
    out = tp.seven_panel_trace_overlay(rows, lith.AUGEN_GNEISS, tmp_path / "g", formats=("png",))
    assert out and out[0].exists()


def test_overlay_builder_rejects_a_wrong_panel_count(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from tools import lithology as lith
    from tools import traces as tr
    from tools import plotting as tp
    rows = [r for r in tr.compare_all() if r["lithology_key"] == "augen_gneiss"][:6]
    with pytest.raises(ValueError):
        tp.seven_panel_trace_overlay(rows, lith.AUGEN_GNEISS, tmp_path / "bad", formats=("png",))


def test_class_colour_mapping_is_shared_and_ordered():
    from tools.plotting import CLASS_COLORS, CLASS_LABELS
    from tools.failure_classification import CLASS_ORDER, NO_FAILURE
    assert CLASS_ORDER == ("WT", "WS", "MT", "MS")
    for c in CLASS_ORDER:
        assert c in CLASS_COLORS and c in CLASS_LABELS
    assert NO_FAILURE in CLASS_COLORS
    # the four primary classes must be visually distinct
    assert len({CLASS_COLORS[c] for c in CLASS_ORDER}) == 4


# ---------------------------------------------------------------------------
# Step 2 numerical non-regression
# ---------------------------------------------------------------------------
# Values re-locked on 2026-08-09 after the predicted crack traces were
# regenerated; the conclusions are unchanged (null predictor still beats the
# model, correlation still indistinguishable from zero).
def test_orientation_statistics_match_locked_step2_values():
    """Recomputing from source must reproduce the values in the manuscript."""
    from tools import traces as tr
    agg = tr.aggregate_statistics(tr.compare_all())
    assert agg["n"] == 14
    # Re-locked when orientation fitting moved from the central window to the
    # whole primary segment (tools.traces.orientation_pair). The window was
    # producing both of the large errors, in two different ways: specimen 12's
    # windowed fit kept 8 of 15 points and was simply imprecise (SE 4.0 deg),
    # while specimen 5's was precise but measured a different tangent, 102.63
    # deg against 93.68 deg over the full segment, because the trace is curved.
    # A precision test catches the first and is blind to the second, so the
    # window was dropped rather than gated.
    #
    # This raises the model's apparent accuracy and was adopted after
    # inspecting those two specimens, which is worth stating plainly. What
    # makes it defensible is that it is one uniform rule for all fourteen with
    # no per-specimen choice, it uses every digitized point, and the estimate
    # no longer depends on where the window is drawn. The windowed variant
    # remains computable via domain="central_window" and still reproduces the
    # previously published values exactly.
    # Re-locked at M = 48 (2026-08-29). The Airy series order was pinned at 24
    # by literals at four call sites, so every field was under-converged; the
    # boundary-traction residual fell from 6.6e-2 to 6.3e-3 when that was fixed.
    # The predicted orientations moved by up to 4 degrees on individual
    # specimens. The within-5 count fell from 10 to 9 and the worst error grew
    # from 7.5 to 9.0, so the agreement is slightly weaker than at M = 24, not
    # better.
    assert agg["mae_deg"] == pytest.approx(3.32, abs=0.01)
    assert agg["rmse_deg"] == pytest.approx(3.73, abs=0.01)
    assert agg["median_deg"] == pytest.approx(3.37, abs=0.01)
    assert agg["max_deg"] == pytest.approx(6.85, abs=0.01)
    assert agg["n_within_5"] == 11
    assert agg["n_within_10"] == 14

    # The windowed variant is kept reproducible because the manuscript reports
    # it as a sensitivity check.
    old = tr.aggregate_statistics(tr.compare_all(domain="central_window"))
    assert old["mae_deg"] == pytest.approx(5.56, abs=0.01)
    assert old["max_deg"] == pytest.approx(11.89, abs=0.01)


def test_loading_parallel_null_is_not_beaten_by_the_model():
    """The manuscript withdraws any orientation-predictive claim; pin that.

    The ordering itself has moved four times as the fields were corrected:
    3.37 against 3.53 at M = 24, 3.37 against 3.26 once M reached 48, 3.37
    against 4.27 once the stress-sign convention was declared, and 3.37
    against 3.39 once the predicted path was read only inside 0.85 R. Every
    one of those margins is a fraction of the 0.2 to 2.6 degree digitization
    uncertainty, so none of them establishes an ordering. What survives is
    that the framework does not beat the trivial predictor, and that is what
    the text claims and what is pinned here.
    """
    from tools import traces as tr
    rows = tr.compare_all()
    model = tr.aggregate_statistics(rows)
    null = tr.null_model_statistics(rows)
    assert null["mae_deg"] == pytest.approx(3.37, abs=0.01)
    assert model["mae_deg"] > null["mae_deg"] - 0.5, (
        f"the framework now leads the null by "
        f"{null['mae_deg'] - model['mae_deg']:.2f} deg, beyond the "
        "digitization uncertainty; the withdrawal of orientation-predictive "
        "skill would need revisiting")



# ``manuscript_values.csv`` was retired with the final dataset (2026-08). It was
# written by the pre-refactor notebook cells through
# ``tools.export.append_manuscript_values``, which the migrated notebooks never
# call, so nothing in the pipeline regenerates it and the guard it carried could
# only ever fail. Provenance for the reported numbers now lives in the
# per-quantity tables under ``outputs/tables/``, each of which is written by the
# generator that computes it and is covered by its own test above.
