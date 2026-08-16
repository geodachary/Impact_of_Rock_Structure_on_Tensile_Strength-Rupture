#!/usr/bin/env python3
"""Lift each long notebook cell into a section module under ``tools/analysis/``.

Why this exists
---------------
Both lithology notebooks carried the whole analysis inline: ~15 000 lines of
function definitions pasted into cells, with 137 distinct function names and
66 of them defined more than once with different bodies. Because a notebook
shares one global namespace top to bottom, a call did not mean what the nearest
definition said it meant -- it meant whatever had been defined *last*. That is
the property that made the two notebooks drift apart.

The fix is one module per analysis section rather than one module per function
name. Grouping by section is what makes the move safe: the divergent versions
cluster by section, and *within* a section the gneiss and schist cells agree
(the only exceptions are recorded in SECTIONS below). So a section module can
be shared by both lithologies while reproducing exactly the code that was live
at that point of the notebook.

What the extractor does with a cell
-----------------------------------
``split_cell`` walks the top level of the cell and sorts every statement into

  imports  -> module header
  defs     -> module body, unchanged
  the rest -> the body of ``main(rock)``, in its original order

Only definitions are hoisted. Everything else keeps notebook order inside
``main``, which declares ``global`` for every name the cell bound, because the
hoisted functions close over those names exactly as they did in the notebook.

Nested helpers are left alone. Many of these cells define their solver as
closures inside one big entry function (``compute_uv_for_sample`` and friends);
those are already encapsulated and move wholesale.

Numbers that differed between the two notebooks -- sample ids, weak-plane
spacing, phase-warp amplitude, output filename -- are *not* copied into the
module. They are read from the ``Lithology`` handed to ``main``, which is what
lets one module serve both rocks.

    python scripts/extract_analysis_sections.py [--check]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tools" / "analysis"

#: The extractor reads the *pre-refactor* notebooks, not the generated ones.
#: They are kept under legacy_notebooks/pre_refactor/ so this script stays
#: reproducible after the working notebooks have been rebuilt from it.
SOURCE_DIR = REPO / "legacy_notebooks" / "pre_refactor"

AG = "Tensile_augen_gneiss.ipynb"
PS = "Tensile_psammitic_schist.ipynb"

#: One entry per analysis section.
#:
#: ``source`` is the notebook the code is taken from. Where the two notebooks
#: genuinely diverged the gneiss cell is authoritative -- it carries the newer
#: implementation in every case (physics-guided crack postprocessing, the
#: four-class guidance block, per-rock mesh spacing) and the schist cell had
#: fallen behind. ``note`` records that decision inline.
SECTIONS = [
    dict(module="base_fields", ag=3, ps=3, scope="library",
         doc="Specimen table, triangular mesh and the local Mohr-Coulomb "
             "failure test used by the early diagnostic cells."),
    dict(module="ddm_disk", ag=8, ps=9,
         doc="Displacement-discontinuity solve on the cut-cell disk: moduli "
             "fields, contact loading, PCG/BiCGSTAB solvers, stresses.",
         note="schist cell 8 is a stale duplicate of this cell and is dropped."),
    dict(module="stress_graph", ag=10, ps=11,
         doc="Radial stress traverses and weak-band crossings on y=0."),
    dict(module="stress_field", ag=12, ps=13,
         doc="Full-disk stress-distribution panels."),
    dict(module="strain_proxy", ag=13, ps=15,
         doc="Tensile-strain proxy field and its principal direction."),
    dict(module="principal_strain", ag=17, ps=19,
         doc="Principal tensile strain panels from the solver fields."),
    dict(module="displacement_profiles", ag=19, ps=21,
         doc="Mean mid-section horizontal displacement profiles."),
    dict(module="direction_circles", ag=21, ps=23,
         doc="Stress-direction circles around the disk."),
    dict(module="crack_energy_suite", ag=33, ps=35,
         doc="Energy-guided DDM crack growth, four-class failure maps, "
             "strain-energy and stress-tensor glyph figures.",
         note="gneiss cell is authoritative: it routes through "
              "postprocess_best_crack_physics, which the schist cell lacked."),
    dict(module="crack_path_suite", ag=37, ps=37,
         doc="Crack-path studies with energy and field guidance.",
         note="gneiss cell is authoritative: build_guidance_fields carries the "
              "four-class failure-map block absent from the schist cell."),
    dict(module="model_anisotropy", ag=49, ps=50, scope="library",
         doc="Model anisotropy ratio from the solved principal stresses."),
    dict(module="metric_export", ag=55, ps=56, scope="both",
         doc="Formatted spreadsheet export of the manuscript metric table."),
    dict(module="foliation_deviation", ag=60, ps=60,
         doc="Fracture deviation from the foliation, by angle."),
    dict(module="kmax_sweep", ag=63, ps=63, scope="both",
         doc="Mixed-mode fraction against the local damage cap k_max."),
    dict(module="local_damage", ag=66, ps=66, scope="library",
         doc="Local cohesion-softening damage model."),
    dict(module="mohr_coulomb_local", ag=67, ps=67, scope="both",
         doc="Advanced stress evaluation and the local Mohr-Coulomb "
             "failure-mode classifier."),

    # --- cross-lithology strength models -------------------------------
    # These fitted both rocks together but sat inside one lithology notebook,
    # where the other rock could not see them. They belong in the general
    # notebook. `src` names the notebook each is taken from, because the two
    # notebooks carried different members of this set.
]

#: Cells deliberately not extracted, and why. Recorded so the omission is a
#: decision on the record rather than something that looks like an oversight.
DROPPED = [
    ("Tensile_augen_gneiss.ipynb cells 35 and 56",
     "byte-identical 842-line copies of an ATI fit that frees four parameters "
     "(both end members plus eta and beta_p). tools/ati_model.py fits two, "
     "holding the measured end members fixed, and is what the manuscript "
     "reports -- the inline copy is what made figure 4's R-squared disagree "
     "with the text. Superseded, not moved."),
    ("Tensile_psammitic_schist.ipynb cell 8",
     "stale duplicate of cell 9; cell 9 matches the gneiss solver cell."),
    ("Tensile_augen_gneiss.ipynb cells 5, 51 and 53",
     "mesh convergence, the smooth end-member strength model and its weakening "
     "factor. Each wrote a figure that scripts/make_mesh_sensitivity_figure.py, "
     "scripts/make_envelope_fit_figure.py and scripts/make_weakening_figure.py "
     "also write, so the same three filenames had two producers. The scripts "
     "are authoritative: both model scripts fit through tools/ati_model.py, "
     "which frees only eta and beta_p and holds the measured end members fixed, "
     "and that is the method the manuscript describes and the values it reports "
     "(eta 0.193 and 2.365, beta_p 72.6 and 83.5 deg). The cells fit four "
     "parameters instead. The stored envelope figure had in fact come from the "
     "cell, so the plotted fit and the quoted fit disagreed."),
    ("Tensile_psammitic_schist.ipynb cell 58",
     "the M7 predicted-against-measured figure. It wrote "
     "predicted_vs_actual_both_rocks.pdf, the same path as "
     "scripts/make_predicted_vs_actual.py, which the consolidated block of the "
     "general notebook runs afterwards; the script therefore won every full "
     "run and its figure is the one in the paper. They are different fits, not "
     "two renderings of one: the script gives R^2 0.887 and 0.912 with RMSE "
     "0.374 and 0.622 MPa, which the manuscript reports in the results text, "
     "the envelope table and the ATI supplement, while the cell gives 0.878 "
     "and 0.802. Keeping both left the same filename contested by two models."),
    ("Tensile_psammitic_schist.ipynb cells 53 and 54",
     "the combined (M7) strength model and its weakening-factor figure. The "
     "smooth end-member model supersedes them, and it is the one the current "
     "manuscript reports: the paper includes "
     "Anisotropic_model_smooth_endpoint_both_rocks.pdf and "
     "Smooth_weakening_factor_both_rocks.pdf, and cites neither "
     "Anistropic_model_combined_both_rocks.pdf nor "
     "Weakening_factor_both_rocks.pdf. Both families were cited by the earlier "
     "draft now in archive/superseded_manuscript/. Carrying both in the general "
     "notebook left it presenting two strength models with nothing to say which "
     "one the results rest on."),
    ("Tensile_psammitic_schist.ipynb cell 52",
     "the normalised tensile-anisotropy comparison. 250 of its 307 lines were "
     "already commented out in the source notebook, leaving only an import, so "
     "the cell did nothing wherever it ran. Its figure "
     "(normalized_tensile_anisotropy_comparison_two_rocks.pdf) came from an "
     "earlier state of the cell and is not cited by the manuscript. Extracting "
     "it produced a module whose main() body was `pass`, which reads as broken "
     "in a published notebook."),
    ("Tensile_augen_gneiss.ipynb cell 58",
     "identical to the schist cell 58 apart from writing "
     "predicted_vs_actual_M7_only.pdf, which the manuscript does not cite."),
]

# Literals that encode the lithology. None may survive into a shared module:
# each is rewritten to read from the lithology bound by main(). `_rock()` is
# resolved at call time, which is what makes one module serve both rocks.
LITHOLOGY_LITERALS = [
    (re.compile(r"list\(range\(1,\s*8\)\)|range\(1,\s*8\)"), "list(_rock().sample_ids)"),
    (re.compile(r"list\(range\(8,\s*15\)\)|range\(8,\s*15\)"), "list(_rock().sample_ids)"),
    (re.compile(r'"1-7"|"8-14"'), "_rock().sample_id_spec()"),
    (re.compile(r"(?<![\w.])spacing_m\s*=\s*0\.0(?:10|02)\b"),
     "spacing_m = _rock().spacing_m"),
    (re.compile(r"(?<![\w.])SPACING_DEFAULT_M\s*=\s*0\.0(?:10|02)\b"),
     "SPACING_DEFAULT_M = _rock().spacing_m"),
    (re.compile(r"(?<![\w.])spacing_warp_amp_m\s*([=:])\s*0\.0(?:02|004)\b"),
     r"spacing_warp_amp_m\1 _rock().spacing_warp_amp_m"),
    (re.compile(r'"(?:augen_gneiss|psammitic_schist)_([a-z0-9_]+)\.(pdf|png)"'),
     r'f"{_rock().key}_\1.\2"'),
    # argparse defaults for the same quantities. These are evaluated when the
    # parser is built, which happens inside main(), so _rock() is bound.
    #
    # Match the option names loosely. The first version of this rule named
    # --spacing_m exactly and silently missed --weak_spacing_m, which carries
    # the same quantity under a different name; the schist then inherited the
    # gneiss's 10 mm spacing and its regenerated fields showed 5 weak-plane
    # bands where 2R/s demands 25. Anything ending in `spacing_m` is the
    # spacing.
    (re.compile(r'(add_argument\("--[a-z_]*spacing_m",[^)]*?default=)'
                r'0\.0(?:10|02|1)\b'),
     r"\1_rock().spacing_m"),
    (re.compile(r'(add_argument\("--[a-z_]*spacing_warp_amp_m",[^)]*?default=)'
                r'0\.0(?:02|004)\b'),
     r"\1_rock().spacing_warp_amp_m"),
]

#: Values that must never survive into a shared module, and what they mean.
#: Checked after generation: a lithology-specific number left behind is the
#: failure this whole exercise exists to prevent, and it is invisible until
#: someone counts bands in a regenerated field.
FORBIDDEN_LITERALS = [
    (re.compile(r'add_argument\("--[a-z_]*spacing[a-z_]*",[^)]*?default=[0-9]'),
     "a spacing argparse default is still a literal"),
    (re.compile(r"^\s*(?:weak_)?spacing_m\s*=\s*0\.[0-9]+", re.M),
     "a spacing assignment is still a literal"),
    # Signature defaults are the dangerous ones: no caller has to pass the
    # argument for the literal to become the live value, so a wrong number is
    # silent. `weak_spacing=0.01` reached both rocks this way.
    (re.compile(r"[,(]\s*(?:weak_)?spacing[a-z_]*\s*=\s*0\.[0-9]+"),
     "a spacing default is still a literal in a function signature or call"),
]

#: Lithology-dependent values appearing as ``def`` signature defaults. Python
#: binds those once at import, so the value would be frozen to whichever rock
#: was extracted. Each becomes ``None`` and is resolved in the function body.
DEFAULT_RESOLUTIONS = {
    "spacing_m": "_rock().spacing_m",
    "spacing_augen_gneiss": "_rock().spacing_m",
    "spacing_other": "_rock().spacing_m",
    "spacing_warp_amp_m": "_rock().spacing_warp_amp_m",
    # The weak-plane spacing under its other name. The crack-path cells call it
    # `weak_spacing`, and because no caller passes it, the signature default is
    # the live value -- so leaving it a literal gives both rocks whichever
    # number the source cell happened to carry.
    "weak_spacing": "_rock().spacing_m",
}

HEADER = '''"""{doc}

Extracted verbatim from {src} cell {cell} by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
{note}"""
from __future__ import annotations

'''


def cells(nb_name):
    nb = json.loads((SOURCE_DIR / nb_name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def split_cell(src):
    """Sort top-level statements into imports, definitions and everything else.

    Only ``def``/``class`` are hoisted to module level. Every other top-level
    statement stays, in its original order, inside ``main`` -- which is what the
    notebook did when it ran the cell top to bottom. Sorting the assignments
    into "constants" and hoisting those was the tempting alternative, but it
    reorders them against the statements left behind, and any mistake there is
    silent. So the split is on statement kind alone.
    """
    tree = ast.parse(src)
    imports, defs, driver = [], [], []

    for node in tree.body:
        seg = ast.get_source_segment(src, node)
        if seg is None:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append(seg)
        elif _is_main_guard(node):
            # `if __name__ == "__main__":` is true in a notebook and false in a
            # module, so leaving it would turn the section into a no-op that
            # still reports success. The guarded block is the cell's driver;
            # unwrap it so main() actually runs it.
            driver.append(_unwrap(src, node))
        else:
            driver.append(seg)
    return imports, defs, driver


def _is_main_guard(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _unwrap(src, node):
    """The body of a compound statement, dedented to top level."""
    lines = src.splitlines()
    first, last = node.body[0], node.body[-1]
    block = "\n".join(lines[first.lineno - 1:last.end_lineno])
    return textwrap.dedent(block)


def _bound_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names = set()
        for e in target.elts:
            names |= _bound_names(e)
        return names
    return set()


def _bindings_in(body_nodes):
    """Names bound by a list of statements -- assignments, loops, with, etc.

    Recurses through compound statements, because a name bound inside a
    top-level ``try:`` or ``if:`` is still a notebook global -- ``aug_popt`` is
    assigned inside a ``try`` and read by the next cell. It does *not* recurse
    into function or class bodies, whose names are locals and never were
    globals.
    """
    names = set()

    def walk(body):
        for node in body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    names.update(_bound_names(t))
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                names.update(_bound_names(node.target))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                names.update(_bound_names(node.target))
                walk(node.body)
                walk(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        names.update(_bound_names(item.optional_vars))
                walk(node.body)
            elif isinstance(node, (ast.If, ast.While)):
                walk(node.body)
                walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                walk(node.orelse)
                walk(node.finalbody)
                for h in node.handlers:
                    if h.name:
                        names.add(h.name)
                    walk(h.body)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)          # the name only, not the body
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])

    walk(body_nodes)
    return names



def _toplevel_bindings(src):
    """:func:`_bindings_in` for a source string."""
    return _bindings_in(ast.parse(src).body)

def _wrap_global(names, width=76):
    """A ``global`` declaration, wrapped so the generated module stays readable."""
    if not names:
        return ""
    lines, current = [], "    global "
    for i, n in enumerate(names):
        piece = n + ("," if i < len(names) - 1 else "")
        if len(current) + len(piece) + 1 > width and current.strip() != "global":
            lines.append(current.rstrip() + " \\")
            current = "        "
        current += piece + " "
    lines.append(current.rstrip())
    return "\n".join(lines) + "\n"




#: Sections that consume the fit produced by an earlier section. In the
#: notebook these were adjacent cells sharing globals; as modules the
#: dependency has to be stated, and the dependency's ``main`` is run first.
#: Generated module text, keyed by module name, filled as the run proceeds.
#: A section that consumes another's fit resolves its names against this.
GENERATED: dict[str, str] = {}

DEPENDENCIES: dict[str, str] = {}

#: Functions one section deliberately reuses from an earlier one instead of
#: redefining. The principal-strain cell said so in as many words -- "Uses the
#: EXISTING compute_sigma1plus_and_theta_for_sample(...) (DO NOT redefine it
#: here)" -- which only worked because the earlier cell had already run. As
#: modules the reuse is an import, and the ordering is enforced by it.
SIBLING_EXPORTS = {
    "compute_sigma1plus_and_theta_for_sample": "strain_proxy",
}

#: Imports the old cells inherited from earlier cells rather than declaring.
STANDARD_IMPORTS = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "os": "import os",
    "sys": "import sys",
    "json": "import json",
    "math": "import math",
    "Path": "from pathlib import Path",
}


def free_names(text):
    """Names the module reads but never binds at module level.

    The whole module is scanned, not just ``main``. The cells define their
    solvers as deeply nested closures, and those bodies call the shared toolkit
    too -- ``band_open_factor_from_sigma_n`` is reached only from inside
    ``compute_combined_field_for_sample``. Scanning only the top level would
    miss them, and the miss would surface as a NameError partway through a
    twenty-minute solve.

    This over-approximates: a local variable that happens to share a name with
    a toolkit export is reported as free. That is the safe direction -- the
    caller only uses the result to decide what to import, and a local always
    wins over a module-level import.
    """
    tree = ast.parse(text)
    import builtins
    bound = set(dir(builtins)) | {"True", "False", "None", "__name__", "__file__"}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
    bound |= _bindings_in(tree.body)

    main = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if main is not None:
        bound |= {a.arg for a in main.args.args}
        bound |= _bindings_in(main.body)
        for x in ast.walk(main):
            if isinstance(x, ast.Global):
                bound |= set(x.names)

    reads = {x.id for x in ast.walk(tree)
             if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    return reads - bound


def build_prelude(text, module):
    """Imports and state the cell inherited from earlier cells.

    A notebook cell could read anything an earlier cell had left in the
    namespace: the shared ``tools.ddm`` toolkit imported near the top, the
    specimen table ``df``, or a fit computed by the cell immediately above. As
    a module none of that is in scope, so the inherited names are resolved
    explicitly here -- which is also what makes each section runnable on its
    own rather than only after a full top-to-bottom run.
    """
    try:
        import tools.ddm as _ddm
        shared = set(getattr(_ddm, "__all__", []))
    except Exception:                       # pragma: no cover - build-time only
        shared = set()

    missing = free_names(text)
    header, body = [], []

    for name in sorted(missing & set(STANDARD_IMPORTS)):
        header.append(STANDARD_IMPORTS[name])

    from_ddm = sorted(missing & shared)
    if from_ddm:
        header.append("from tools.ddm import (  # noqa: F401\n    "
                      + ",\n    ".join(_wrap_names(from_ddm)) + ",\n)")

    for name in sorted(missing & set(SIBLING_EXPORTS)):
        if SIBLING_EXPORTS[name] != module:
            header.append(
                f"from tools.analysis.{SIBLING_EXPORTS[name]} import {name}")

    if "df" in missing:
        header.append("from tools.data_io import load_specimen_table")
        body.append("df = load_specimen_table()")

    dep = DEPENDENCIES.get(module)
    if dep and dep in GENERATED:
        # Split what the dependency offers: functions can be imported, but the
        # fitted values only exist once its main() has run, so they are read
        # back afterwards. Resolving against the dependency's actual contents
        # keeps this from importing every name that merely looks unbound.
        dep_tree = ast.parse(GENERATED[dep])
        functions = {n.name for n in dep_tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name != "main"}
        state = set()
        for n in ast.walk(dep_tree):
            if isinstance(n, ast.Global):
                state |= set(n.names)

        wanted_fns = sorted(missing & functions)
        wanted_state = sorted(missing & state)

        if wanted_fns:
            header.append(f"from tools.analysis.{dep} import (  # noqa: F401\n    "
                          + ",\n    ".join(_wrap_names(wanted_fns)) + ",\n)")
        if wanted_state:
            header.append(f"from tools.analysis import {dep} as _fit")
            body.append(
                f"# This section plots the fit produced by {dep}, which was the\n"
                f"# cell immediately above it in the old notebook. Running it here\n"
                f"# is what that adjacency used to do implicitly.\n"
                f"_fit.main()")
            for n in wanted_state:
                body.append(f"{n} = _fit.{n}")

    return header, body


def _wrap_names(names, width=72):
    lines, current = [], ""
    for n in names:
        if current and len(current) + len(n) + 2 > width:
            lines.append(current.rstrip(", "))
            current = ""
        current += n + ", "
    if current:
        lines.append(current.rstrip(", "))
    return lines


#: What a cell's own ``main`` is renamed to. Several cells were written as
#: command-line scripts and define ``def main()``. The generated entry point is
#: also called ``main``, and being defined later it wins -- so the cell's own
#: ``main()`` call would re-enter the wrapper instead of the script body. The
#: collision is silent at import and only shows up mid-run, so the cell's
#: function is renamed rather than the wrapper, whose name is part of the
#: package's interface.
CELL_ENTRY_POINT = "run_section"

_MAIN_CALL = re.compile(r"(?<![\w.])main(?=\s*\()")


def rename_cell_main(text):
    """Rename a cell's own ``main`` so it cannot be shadowed by the wrapper."""
    return _MAIN_CALL.sub(CELL_ENTRY_POINT, text)


def defines_main(defs):
    return any(ast.parse(d).body[0].name == "main" for d in defs)


def _takes_argv(defs):
    """True when the renamed cell entry point accepts an ``argv`` argument."""
    for d in defs:
        node = ast.parse(d).body[0]
        if node.name != CELL_ENTRY_POINT:
            continue
        args = node.args
        return any(a.arg == "argv" for a in args.posonlyargs + args.args + args.kwonlyargs)
    return False


#: Corrections applied to a generated module, with the reason for each.
#:
#: The extractor's contract is that a section runs the code the cell ran. These
#: are the declared exceptions: places where moving the code out of the notebook
#: namespace broke a binding that used to work by adjacency. Each is recorded
#: here rather than hand-edited into the generated file, so it survives
#: regeneration and stays visible.
PATCHES = {
    "displacement_profiles": [
        (
            'cache_dir = os.path.join(output_dir, "_cache_uv_profiles_v2")\n',
            'cache_dir = os.path.join(output_dirs.fields(), "_cache_uv_profiles_v2")\n',
            "the solver cache hung off the figure directory because that was the only output directory the cell knew about. It is not a figure: it is recomputable solver state keyed on the specimen geometry, so it belongs with the other cached fields",
        ),
    ],
    "principal_strain": [
        (
            'cache_dir = os.path.join(output_dir, "_cache_solver_fields_physics_v2")\n',
            'cache_dir = os.path.join(output_dirs.fields(), "_cache_solver_fields_physics_v2")\n',
            "the solver cache hung off the figure directory because that was the only output directory the cell knew about. It is not a figure: it is recomputable solver state keyed on the specimen geometry, so it belongs with the other cached fields",
        ),
    ],
    "direction_circles": [
        (
            'cache_dir = os.path.join(output_dir, "_cache_direction_circles_uv_v2")\n',
            'cache_dir = os.path.join(output_dirs.fields(), "_cache_direction_circles_uv_v2")\n',
            "the solver cache hung off the figure directory because that was the only output directory the cell knew about. It is not a figure: it is recomputable solver state keyed on the specimen geometry, so it belongs with the other cached fields",
        ),
    ],
    "crack_energy_suite": [
        (
            "    # Writes the old->new classifier mapping table and the locus-resolved\n"
            "    # WT/WS/MT/MS figure to revision_outputs/. All existing outputs above\n"
            "    # (ddm_fields/*) are untouched.\n",
            "    # Writes the old->new classifier mapping table to the results\n"
            "    # directory and the locus-resolved WT/WS/MT/MS figure to the figure\n"
            "    # directory. All existing outputs above are untouched.\n",
            "the comment named revision_outputs/ and ddm_fields/, directories that "
            "no longer exist; a stale path in a comment is worse than none because "
            "a reader trusts it",
        ),
        (
            '        out_pdf_stress = os.path.join(args.out_dir, f"{_rock().key}_stress_tensors.pdf")\n'
            '        out_pdf_energy = os.path.join(args.out_dir, f"{_rock().key}_strain_energy.pdf")\n',
            '        # args.out_dir is where the field archives and the per-specimen\n'
            '        # crack traces go. These two are figures, so they follow the\n'
            '        # figures instead of being filed under the section that drew them.\n'
            '        _fig_dir = output_dirs.figures()\n'
            '        out_pdf_stress = os.path.join(_fig_dir, f"{_rock().key}_stress_tensors.pdf")\n'
            '        out_pdf_energy = os.path.join(_fig_dir, f"{_rock().key}_strain_energy.pdf")\n',
            "the stress-tensor and strain-energy panels were written beside the "
            "cached fields rather than with the other figures",
        ),
        (
            # Written against the post-delithologise text: the filename here
            # was hardcoded to augen_gneiss in the cell and the literal rule
            # has already turned it into _rock().key by the time patches run.
            '                os.path.join("revision_outputs", "figures", f"{_rock().key}_fourclass_map.pdf"),\n'
            '                os.path.join("revision_outputs", "figures", f"{_rock().key}_fourclass_map.png"),\n',
            '                os.path.join("figures", f"{_rock().key}_fourclass_map.pdf"),\n'
            '                os.path.join("figures", f"{_rock().key}_fourclass_map.png"),\n',
            "the four-class map was written to revision_outputs/figures/, a "
            "directory named after the revision process rather than after what "
            "it holds; figures belong with the other figures",
        ),
        (
            '            _fc4_csv = os.path.join("revision_outputs", "classifier_mapping.csv")\n',
            '            # Per lithology. Both rocks wrote this same path, so whichever\n'
            '            # notebook ran second replaced the first rock\'s table and the\n'
            '            # file silently held one lithology while reading as though it\n'
            '            # held both.\n'
            '            _fc4_csv = os.path.join(\n'
            '                "results", f"classifier_mapping_{_rock().key}.csv")\n',
            "the old-to-new classifier mapping was written to one unprefixed "
            "path by both lithologies, so only the last rock to run survived. "
            "It is a machine-readable result, so it belongs in results/ with "
            "the others rather than in a directory named after a revision",
        ),
        (
            "def plot_stress_tensors_glyphs(",
            "#: Stress-tensor glyph colours. Red for tension, blue for compression --\n"
            "#: the convention a reader brings to a tensor plot. These are the\n"
            "#: diverging ends of ColorBrewer RdYlBu, so they stay distinguishable in\n"
            "#: greyscale and to the common forms of colour blindness. Defined once\n"
            "#: because the glyphs and the legend must agree.\n"
            "TENSOR_TENSION_COLOR = \"#d7191c\"\n"
            "TENSOR_COMPRESSION_COLOR = \"#2c7bb6\"\n"
            "\n"
            "\n"
            "def plot_stress_tensors_glyphs(",
            "one definition of the tension/compression colours for glyphs and legend",
        ),
        (
            "                               glyph_len_frac=0.10, glyph_alpha=0.65,\n",
            "                               glyph_len_frac=0.10, glyph_alpha=0.85,\n",
            "glyphs were washed out at 0.65; the sign colours below need to read "
            "clearly at print size",
        ),
        (
            '    if sign_style == "linestyle":\n'
            '        pairs = [(seg1, m1_t, "r", "solid"), (seg1, m1_c, "r", "dashed"),\n'
            '                 (seg3, m3_t, "b", "solid"), (seg3, m3_c, "b", "dashed")]\n'
            "    else:\n"
            '        pairs = [(seg1, m1_t, "#d62728", "solid"), (seg1, m1_c, "#7f1d1d", "solid"),\n'
            '                 (seg3, m3_t, "#1f77b4", "solid"), (seg3, m3_c, "#0b3c5d", "solid")]\n'
            "    for seg, mask, col, ls in pairs:\n"
            "        if np.any(mask):\n"
            "            kw = dict(colors=col, linewidths=0.95, alpha=alpha, zorder=2)\n"
            '            if ls == "dashed":\n'
            '                kw["linestyles"] = "dashed"\n'
            '                kw["alpha"] = alpha * 0.85\n'
            "            ax.add_collection(LineCollection(seg[mask], **kw))\n",

            "    # Colour carries the SIGN of the stress, which is what a reader of a\n"
            "    # tensor-glyph plot expects: red in tension, blue in compression. The\n"
            "    # previous scheme spent colour on which principal axis a tick was\n"
            "    # (sigma_1 red, sigma_3 blue) and left the sign to a dashed linestyle,\n"
            "    # so the physically important distinction was the harder one to see.\n"
            "    #\n"
            "    # The axis is still legible: sigma_1 is drawn heavier than sigma_3.\n"
            "    # Line weight is the right carrier for it because the major and minor\n"
            "    # axes are perpendicular at every point, so the pair reads as a cross\n"
            "    # whichever way it is coloured.\n"
            "    TENSION, COMPRESSION = TENSOR_TENSION_COLOR, TENSOR_COMPRESSION_COLOR\n"
            "    pairs = [(seg1, m1_t, TENSION, 1.45), (seg1, m1_c, COMPRESSION, 1.45),\n"
            "             (seg3, m3_t, TENSION, 0.75), (seg3, m3_c, COMPRESSION, 0.75)]\n"
            "    for seg, mask, col, lw in pairs:\n"
            "        if np.any(mask):\n"
            "            ax.add_collection(LineCollection(\n"
            "                seg[mask], colors=col, linewidths=lw, alpha=alpha,\n"
            "                zorder=2))\n",
            "stress-tensor glyphs coloured by principal axis rather than by sign, "
            "against the usual red-tension / blue-compression convention",
        ),
        (
            "        handles_stress = [\n"
            '            Line2D([0], [0], color="r", lw=2.0, label=r"$\\sigma_1$"),\n'
            '            Line2D([0], [0], color="b", lw=2.0, label=r"$\\sigma_3$"),\n'
            "        ]\n",

            "        handles_stress = [\n"
            "            Line2D([0], [0], color=TENSOR_TENSION_COLOR, lw=2.0,\n"
            '                   label="tension"),\n'
            "            Line2D([0], [0], color=TENSOR_COMPRESSION_COLOR, lw=2.0,\n"
            '                   label="compression"),\n'
            '            Line2D([0], [0], color="0.35", lw=2.4, label=r"$\\sigma_1$"),\n'
            '            Line2D([0], [0], color="0.35", lw=1.0, label=r"$\\sigma_3$"),\n'
            "        ]\n",
            "the legend still said colour meant the principal axis, which the "
            "sign-based colouring above makes wrong",
        ),
        (
            '        out_pdf_fail = os.path.join(args.out_dir, f"{_rock().key}_failure_fields.pdf")\n',
            "        # The three-class failure-field panel is superseded by the\n"
            "        # WT/WS/MT/MS map written to the figure directory below,\n"
            "        # resolves mechanism and structural locus together instead of\n"
            "        # collapsing weak-plane opening into a matrix category. The\n"
            "        # current manuscript cites that figure and no longer includes\n"
            "        # *_failure_fields.pdf, so the panel is not written.\n",
            "the three-class failure-field figure is superseded by the four-class "
            "map and is no longer cited by the manuscript",
        ),
        (
            "        fig_fail.savefig(out_pdf_fail, dpi=300, bbox_inches=\"tight\", pad_inches=0.08, transparent=True)\n",
            "        plt.close(fig_fail)\n",
            "stop writing and displaying the superseded failure-field panel",
        ),
        (
            '        print(f"\\n✓ Saved: {out_pdf_fail}")\n'
            '        print(f"✓ Saved: {out_pdf_stress}")\n',
            '        print(f"\\n✓ Saved: {out_pdf_stress}")\n',
            "the superseded panel is no longer written, so it is not reported",
        ),
    ],
    "strain_proxy": [        (
            'cache_dir = os.path.join(output_dir, "_cache_sigma1_theta_physics_v2")\n',
            'cache_dir = os.path.join(output_dirs.fields(), "_cache_sigma1_theta_physics_v2")\n',
            "the solver cache hung off the figure directory because that was the only output directory the cell knew about. It is not a figure: it is recomputable solver state keyed on the specimen geometry, so it belongs with the other cached fields",
        ),

        (
            '        title = f"{rock} ({phi_deg:.0f}°) — σ1+/Eeff axis"\n',
            '        title = f"{rock} ({phi_deg:.0f}°)"\n',
            "the panel title also carried the plotted quantity, which is the "
            "same in all seven panels and made each title too long for a "
            "two-column panel; the quantity belongs on the colour bar, while "
            "the rock and angle identify the panel",
        ),
        (
            "    WIDTH_PAD_IN = 0.95\n",
            "    WIDTH_PAD_IN = 1.10\n",
            "the colour-bar label now runs to three lines, which needs more "
            "width than the old single line or it is clipped at the edge",
        ),
        (
            "    HSPACE = 0.14\n",
            "    HSPACE = 0.22\n",
            "at 0.14 the row titles sat on the x tick labels of the row above; "
            "the panels are square, so the row pitch has to clear a title and "
            "a tick row",
        ),
        (
            '        cbar.set_label("Strain proxy magnitude (σ1+/Eeff), clipped", rotation=270, labelpad=18)\n',
            '        cbar.set_label(\n'
            '            r"tensile strain proxy $\\varepsilon_1^{+}$",\n'
            '            rotation=270, labelpad=20)\n',
            "'Strain proxy magnitude (σ1+/Eeff), clipped' named neither the "
            "quantity nor what was clipped. It is the positive part of the "
            "major principal stress over the effective modulus, a "
            "dimensionless strain, and it is the colour scale that is capped, "
            "not the data: values above the percentile are drawn in the top "
            "colour rather than discarded",
        ),
    ],
    "crack_path_suite": [
        (
            "    _make_all_combined_plots(results, out_dir)\n",
            "    # out_dir holds the step traces and the run summary. The three\n"
            "    # combined plots are figures, so they are left to default to the\n"
            "    # figure directory rather than landing beside the CSVs.\n"
            "    _make_all_combined_plots(results)\n",
            "the three combined crack-path figures were written beside the step "
            "diagnostics rather than with the other figures",
        ),
        (
            "        replot_from_csvs(out_dir=args.out_dir, meta_csv=args.meta_csv)\n",
            "        # Name the specimens explicitly. The combined-plot writer names\n"
            "        # its output after the rocks it was given, so replotting all\n"
            "        # fourteen produces one cross-rock figure rather than the two\n"
            "        # per-lithology figures the manuscript cites.\n"
            "        replot_from_csvs(out_dir=args.out_dir, meta_csv=args.meta_csv,\n"
            "                         sample_ids=list(_rock().sample_ids))\n",
            "replot rebuilt whichever rock the toolkit default named, so the "
            "other lithology's figures were left stale",
        ),
        (
            "    global bd\n",
            "    global bd, SIF_FUN, TRY_SIF\n",
            "run_section binds the solver but declared only bd as global, so "
            "the hook names it set were locals",
        ),
        (
            "    bind_sif_hooks(bd)\n",
            "    bind_sif_hooks(bd)\n"
            "    # bind_sif_hooks binds the hooks inside tools.ddm. This section\n"
            "    # keeps its own module-level SIF_FUN/TRY_SIF and the stepper\n"
            "    # reads those, so mirror the binding across; otherwise\n"
            "    # energy_step_guided calls None.\n"
            "    SIF_FUN, TRY_SIF = sif_hooks()\n",
            "the stepper read this module's SIF_FUN/TRY_SIF, which stayed None "
            "after the toolkit migration, so every crack-path run failed",
        ),
        (
            "def energy_step_guided(",
            "#: How far a step may retreat toward the disc centre, as a fraction\n"
            "#: of the step length. Zero would forbid any inward motion and make\n"
            "#: the path brittle to rounding; a quarter step absorbs that without\n"
            "#: letting the tip wander back into the interior.\n"
            "RETREAT_TOL_FRAC = 0.25\n"
            "\n"
            "\n"
            "def energy_step_guided(",
            "tolerance for the radial-progress constraint below",
        ),
        (
            "                       N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10, debug=False):\n",
            "                       N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10, debug=False,\n"
            "                       force_psi=None, accept_subcritical=False):\n",
            "let the stepper evaluate one prescribed direction, so the "
            "physics-guided steps can be measured with the same machinery",
        ),
        (
            "    dscan = np.linspace(-cap, cap, kink_n)\n"
            "    cand_list = []\n"
            "    for dth in dscan:\n"
            "        psi_new = float(psi_tip + dth)\n"
            "        nx = tipx + float(ds) * np.cos(psi_new)\n"
            "        ny = tipy + float(ds) * np.sin(psi_new)\n"
            "\n"
            "        if (nx * nx + ny * ny) >= (0.999 * R) ** 2:\n"
            "            continue\n"
            "\n"
            "        cheap = 0.0\n"
            "        if np.isfinite(psi_line0):\n"
            "            dpsi = abs(wrap_pi_half_scalar(psi_new - psi_line0)) / (np.pi / 2.0)\n"
            "            cheap = -dpsi * dpsi\n"
            "\n"
            "        cand_list.append((cheap, psi_new, nx, ny, abs(dth)))\n"
            "\n"
            "    if not cand_list:\n"
            "        return None\n",

            "    dscan = np.linspace(-cap, cap, kink_n)\n"
            "    r_tip = float(np.hypot(tipx, tipy))\n"
            "    cand_list = []\n"
            "    retreating = []\n"
            "    for dth in dscan:\n"
            "        psi_new = float(psi_tip + dth)\n"
            "        nx = tipx + float(ds) * np.cos(psi_new)\n"
            "        ny = tipy + float(ds) * np.sin(psi_new)\n"
            "\n"
            "        if (nx * nx + ny * ny) >= (0.999 * R) ** 2:\n"
            "            continue\n"
            "\n"
            "        cheap = 0.0\n"
            "        if np.isfinite(psi_line0):\n"
            "            dpsi = abs(wrap_pi_half_scalar(psi_new - psi_line0)) / (np.pi / 2.0)\n"
            "            cheap = -dpsi * dpsi\n"
            "\n"
            "        # Radial progress. The scan is symmetric about the current\n"
            "        # direction and cap_deg reaches 90 degrees where the tip is\n"
            "        # shear-dominated, so a step may point back toward the disc\n"
            "        # centre. Nothing else forbade it, and over successive steps\n"
            "        # the tip then wandered instead of propagating: the schist\n"
            "        # retreated on 98 of 149 steps and never reached the\n"
            "        # boundary, while the gneiss happened to drift outward and\n"
            "        # terminate. That difference was not physics -- it was the\n"
            "        # schist's finer fabric giving a systematically wider scan\n"
            "        # (median cap 47 against 33 degrees).\n"
            "        #\n"
            "        # In a Brazilian disc the crack starts near the centre and\n"
            "        # runs outward to the platens, so a retreating step is\n"
            "        # inadmissible. That is a statement about the test geometry\n"
            "        # rather than about either rock, which is what makes it safe\n"
            "        # for a lithology this study has not seen.\n"
            "        if np.hypot(nx, ny) >= r_tip - RETREAT_TOL_FRAC * float(ds):\n"
            "            cand_list.append((cheap, psi_new, nx, ny, abs(dth)))\n"
            "        else:\n"
            "            retreating.append(\n"
            "                (np.hypot(nx, ny), (cheap, psi_new, nx, ny, abs(dth))))\n"
            "\n"
            "    # Never let the constraint starve the scan: if every candidate\n"
            "    # retreats, take the least-retreating one and carry on rather\n"
            "    # than ending the path here.\n"
            "    if not cand_list and retreating:\n"
            "        retreating.sort(key=lambda z: -z[0])\n"
            "        cand_list = [retreating[0][1]]\n"
            "\n"
            "    if not cand_list:\n"
            "        return None\n"
            "\n"
            "    # A prescribed direction replaces the scan entirely. This is how a\n"
            "    # physics-guided step is measured: the direction comes from the\n"
            "    # preferred-orientation field rather than from maximising energy\n"
            "    # release, but G and Gc are evaluated by the same machinery below,\n"
            "    # so the two kinds of step are directly comparable.\n"
            "    if force_psi is not None:\n"
            "        fx = tipx + float(ds) * np.cos(float(force_psi))\n"
            "        fy = tipy + float(ds) * np.sin(float(force_psi))\n"
            "        cand_list = [(0.0, float(force_psi), fx, fy,\n"
            "                      abs(wrap_pi_scalar(float(force_psi) - psi_tip)))]\n",
            "the kink scan admitted steps that retreated toward the disc centre, "
            "so the schist path wandered and never reached the boundary",
        ),
        (
            "        score = float(G) - float(Gc)\n"
            "        if score <= 0.0:\n"
            "            continue\n",
            "        score = float(G) - float(Gc)\n"
            "        # G <= Gc means this direction cannot drive the crack, so the\n"
            "        # scan discards it. That is also why a trace built only from\n"
            "        # accepted steps can never show G below Gc: the condition is\n"
            "        # enforced, not observed. A measured physics-guided step keeps\n"
            "        # its value whatever it is, which is what makes the recorded\n"
            "        # G/Gc profile a measurement rather than a restatement of the\n"
            "        # acceptance rule.\n"
            "        if score <= 0.0 and not accept_subcritical:\n"
            "            continue\n",
            "a trace of accepted steps only can never show G below Gc, so the "
            "sub-critical portion of the path was invisible",
        ),
        (
            "                xs_u.append(nxp)\n"
            "                ys_u.append(nyp)\n"
            "                psi_cur = float(ps)\n",

            "                # Measure the physics-guided step with the same SIF and\n"
            "                # energy machinery the scan uses, so it enters the trace\n"
            "                # on equal terms. Without this the record covers only the\n"
            "                # energy-selected fraction of the path, and profile\n"
            "                # statistics measure instrumentation coverage as much as\n"
            "                # crack behaviour.\n"
            "                meas = energy_step_guided(\n"
            "                    xs_u, ys_u, R=R, alpha_const=alpha_const, airy_fit=fit,\n"
            "                    E1=E1, E2=E2, nu12=nu12, G12=G12, ds=ds2,\n"
            "                    kink_scan_deg=cap_deg, kink_n=kink_n, K_keep=K_keep,\n"
            "                    Gc0=float(Gc0), weak_reduction_base=float(wrb),\n"
            "                    eta_deg=float(eta_deg), alpha_wp_line=float(alpha_wp_line),\n"
            "                    ddm_sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),\n"
            "                    X=X, Y=Y, psi_pref_guided_img=ppg, tensile_w_img=twi,\n"
            "                    conf_img=ci, wp_weight_img=wpi, w_load_img=wli,\n"
            "                    corr_cache=corr_cache, N_outer=int(N_outer), debug=debug,\n"
            "                    force_psi=ps, accept_subcritical=True,\n"
            "                )\n"
            "                xs_u.append(nxp)\n"
            "                ys_u.append(nyp)\n"
            "                psi_cur = float(ps)\n"
            "                if meas is not None:\n"
            "                    _, _, _, _, G_p, Gc_p, kII_p, adt_p = meas\n"
            "                    trace_rows.append(dict(\n"
            "                        step=step, x=nxp, y=nyp, psi=ps,\n"
            "                        KI=np.nan, KII=np.nan, G=G_p, Gc=Gc_p,\n"
            "                        score=G_p - Gc_p, kIIratio=kII_p, abs_dth=adt_p,\n"
            "                        cap_deg=cap_deg, kink_n=kink_n, ds=ds2, mode='PHYS'))\n"
            "                    step += 1\n",
            "physics-guided steps advanced the crack without being recorded, so "
            "the G/Gc profile covered only the energy-selected fraction of the path",
        ),
        (
            "                    kIIratio=kIIr, abs_dth=adt, cap_deg=cap_deg,\n"
            "                    kink_n=kink_n, ds=ds_step\n"
            "                ))\n",
            "                    kIIratio=kIIr, abs_dth=adt, cap_deg=cap_deg,\n"
            "                    kink_n=kink_n, ds=ds_step, mode='ENERGY'\n"
            "                ))\n",
            "label the energy-selected steps so the two kinds are distinguishable "
            "in the trace",
        ),
        (
            '    _TC = ["step", "x", "y", "psi", "KI", "KII", "G", "Gc", "score", "kIIratio", "abs_dth",\n'
            '           "cap_deg", "kink_n", "ds"]\n',
            '    _TC = ["step", "x", "y", "psi", "KI", "KII", "G", "Gc", "score", "kIIratio", "abs_dth",\n'
            '           "cap_deg", "kink_n", "ds", "mode"]\n',
            "carry the step kind through to the written trace",
        ),
    ],
}


def apply_patches(text, module):
    """Apply the declared corrections for one module, verifying each matches."""
    for old, new, reason in PATCHES.get(module, []):
        if old not in text:
            raise SystemExit(
                f"{module}: patch target not found, so the correction for "
                f"'{reason}' would be silently skipped:\n{old!r}")
        text = text.replace(old, new)
    return text


def delithologise(text):
    for pattern, repl in LITHOLOGY_LITERALS:
        text = pattern.sub(repl, text)
    return text


#: Where each cell used to write, and which output directory that is now.
#:
#: The notebooks gave every section its own top-level folder named after the
#: section -- physics_force, foliation_deviation_output, ddm_fields,
#: Output_Figures, revision_outputs -- so figures were spread across five
#: directories and none of the names said what was inside. Output is sorted by
#: kind instead, and this is where that rename is applied to the extracted
#: code, so it survives regeneration.
#:
#: A cell writing both kinds under one ``out_dir`` is routed by its dominant
#: kind here and corrected for the other kind in PATCHES.
OUTPUT_DIRS = [
    # Longest first: "ddm_fields/fields_npz" is a single literal in some cells
    # and must be matched before the bare "ddm_fields" rule sees it.
    ('"ddm_fields/fields_npz"', "output_dirs.FIELDS_NPZ"),
    ("'ddm_fields/fields_npz'", "outputs/fields/fields_npz"),
    ("ddm_fields/fields_npz/", "outputs/fields/fields_npz/"),
    ('"Output_Figures"', "output_dirs.FIGURE_DIR"),
    ('"figures"', "output_dirs.FIGURE_DIR"),
    ('"foliation_deviation_output"', "output_dirs.FIGURE_DIR"),
    ('"results"', "output_dirs.TABLE_DIR"),
    ('"revision_outputs"', "output_dirs.TABLE_DIR"),
    ('"physics_force"', "output_dirs.FIELD_DIR"),
    ('"ddm_fields"', "output_dirs.FIELD_DIR"),
]


def reroute_outputs(text):
    """Point the extracted code at the current output tree.

    Runs after the patches, so a patch can go on naming the directory the cell
    named and still land in the right place.
    """
    for literal, const in OUTPUT_DIRS:
        text = text.replace(literal, const)
    return text


#: The spacing values either notebook writes, in any of their spellings.
#: ``0.01`` and ``0.010`` are the same gneiss number written two ways, and
#: missing the short form is how ``weak_spacing=0.01`` survived a first pass.
LITERAL = re.compile(r"^\s*0\.0(?:10|004|02|1)\s*$")


def rewrite_signature_defaults(src, deferred=()):
    """Defer signature defaults that cannot be evaluated at import time.

    Two kinds have to move into the function body:

    * lithology-dependent values -- ``def f(spacing_m=0.010)`` would freeze the
      gneiss number at import, so it becomes ``spacing_m = _rock().spacing_m``
      and follows whichever rock is bound;
    * defaults reading a name the cell bound at top level -- ``path=DATA_PATH``
      -- because that name is now assigned inside ``main`` and does not exist
      when the module is imported.

    Both become ``None`` in the signature and are resolved on entry, so an
    explicit argument still wins and call-time behaviour is unchanged.
    """
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def pos(lineno, col):
        return offsets[lineno - 1] + col

    edits = []          # (start, end, replacement)
    inserts = []        # (position, text)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        pairs = list(zip(a.posonlyargs + a.args, [None] * (
            len(a.posonlyargs) + len(a.args) - len(a.defaults)) + list(a.defaults)))
        pairs += list(zip(a.kwonlyargs, a.kw_defaults))
        resolved = []
        for arg, default in pairs:
            if default is None:
                continue
            seg = ast.get_source_segment(src, default) or ""
            reads = {n.id for n in ast.walk(default) if isinstance(n, ast.Name)}
            if arg.arg in DEFAULT_RESOLUTIONS and LITERAL.match(seg):
                replacement = DEFAULT_RESOLUTIONS[arg.arg]
            elif reads & set(deferred):
                replacement = seg          # same expression, evaluated on entry
            else:
                continue
            edits.append((pos(default.lineno, default.col_offset),
                          pos(default.end_lineno, default.end_col_offset), "None"))
            resolved.append((arg.arg, replacement))
        if not resolved:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str) and len(node.body) > 1):
            first = node.body[1]          # keep the docstring first
        indent = " " * first.col_offset
        block = "".join(
            f"{indent}if {n} is None:\n"
            f"{indent}    {n} = {expr}\n" for n, expr in resolved)
        inserts.append((pos(first.lineno, 0), block))

    # Apply replacements and insertions in one bottom-up pass, so earlier
    # offsets stay valid as later ones are rewritten.
    actions = edits + [(p, p, t) for p, t in inserts]
    for start, end, repl in sorted(actions, key=lambda t: t[0], reverse=True):
        src = src[:start] + repl + src[end:]
    return src


def build_module(sec, src, nb_name, cell_index):
    """Render one section module.

    ``scope`` decides the shape of the result:

    ``rock``     one lithology at a time. Lithology literals are rewritten to
                 read the bound rock, and ``main(rock)`` binds it.
    ``both``     the section already handles both rocks in one pass -- it splits
                 the specimen ids itself, or writes a single un-prefixed figure
                 covering all fourteen. Nothing is rewritten (doing so would
                 collapse its two groups into one) and ``main()`` takes no rock.
    ``library``  definitions only, no driver: helpers other sections call.
    """
    scope = sec.get("scope", "rock")
    imports, defs, driver = split_cell(src)

    if defines_main(defs):
        defs = [rename_cell_main(d) for d in defs]
        driver = [rename_cell_main(d) for d in driver]

        # These cells were written as command-line scripts, so their entry
        # point falls back to sys.argv when handed nothing. That worked in a
        # notebook only because the cell recognised Jupyter's own argv; driven
        # as a library it parses whatever the host process was started with and
        # exits. Passing an explicit empty argv asks for the defaults and takes
        # the guessing out.
        if _takes_argv(defs):
            driver = [re.sub(rf"(?<![\w.]){CELL_ENTRY_POINT}\(\s*\)",
                             f"{CELL_ENTRY_POINT}([])", d) for d in driver]
    note = ""
    if sec.get("note"):
        note = "\nParity note\n-----------\n" + sec["note"] + "\n"
    if scope == "both":
        note += ("\nScope\n-----\nThis section covers both lithologies in one "
                 "pass and is driven from Tensile_general_plots.ipynb, not from "
                 "either lithology notebook.\n")

    parts = [HEADER.format(doc=sec["doc"], src=nb_name, cell=cell_index, note=note)]
    seen = set()
    for imp in imports:
        if imp not in seen:
            parts.append(imp)
            seen.add(imp)
    if scope == "rock":
        parts.append("\nfrom tools.analysis._context import "
                     "bind as _bind, current as _rock\n")
    parts.append("# <<PRELUDE_IMPORTS>>")

    # Only a rock-scoped section is rewritten. A "both" section already
    # distinguishes the lithologies itself, so substituting the bound rock
    # would merge its two groups into one.
    subst = delithologise if scope == "rock" else (lambda t: t)

    # Names the cell bound at top level now live in main(), so a signature
    # default that reads one of them has to be deferred to call time.
    bound = sorted(_toplevel_bindings("\n".join(driver)))
    if scope == "rock":
        # `rock` is main's parameter; Python rejects a name that is both.
        bound = [n for n in bound if n != "rock"]

    if defs:
        parts.append("\n\n# --- implementation ---------------------------------------------------\n")
        parts.extend("\n" + subst(rewrite_signature_defaults(d, bound)) + "\n"
                     for d in defs)

    if scope == "library":
        return "\n".join(parts)

    body = "\n".join(subst(s) for s in driver) or "pass"
    body = "\n".join("    " + line if line.strip() else line
                     for line in body.splitlines())

    # Every name the cell bound at top level was a module-level global in the
    # notebook, and the functions above close over them by that name. main()
    # must therefore rebind them at module scope, not shadow them with locals.
    global_decl = _wrap_global(bound)

    if scope == "rock":
        signature = (
            "\n\ndef main(rock):\n"
            '    """Run this section for one lithology.\n\n'
            "    Parameters\n"
            "    ----------\n"
            "    rock : tools.lithology.Lithology\n"
            "        Supplies the specimen ids, weak-plane spacing and output stem.\n"
            '    """\n'
            + global_decl + "    _bind(rock)\n"
            + "    # <<PRELUDE_BODY>>\n")
    else:
        signature = (
            "\n\ndef main():\n"
            '    """Run this section for both lithologies in one pass."""\n'
            + global_decl
            + "    # <<PRELUDE_BODY>>\n")
    parts.append(signature + body + "\n")
    text = reroute_outputs(apply_patches("\n".join(parts), sec["module"]))
    if "output_dirs." in text:
        text = text.replace("# <<PRELUDE_IMPORTS>>",
                            "from tools import output_dirs\n# <<PRELUDE_IMPORTS>>", 1)
    return _apply_prelude(text, sec["module"])


def _apply_prelude(text, module):
    """Resolve the names the cell used to inherit from earlier cells."""
    header, body = build_prelude(text.replace("# <<PRELUDE_IMPORTS>>", "")
                                     .replace("# <<PRELUDE_BODY>>", "pass"),
                                 module)
    text = text.replace(
        "# <<PRELUDE_IMPORTS>>",
        ("\n# --- inherited from earlier notebook cells ---------------------------\n"
         + "\n".join(header) + "\n") if header else "")
    if body:
        indented = "\n".join("    " + l for b in body for l in b.splitlines())
        text = text.replace("    # <<PRELUDE_BODY>>", indented)
    else:
        text = text.replace("    # <<PRELUDE_BODY>>\n", "")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would be written, write nothing")
    args = ap.parse_args(argv)

    ag_cells, ps_cells = cells(AG), cells(PS)
    source_cells = {"ag": (AG, ag_cells), "ps": (PS, ps_cells)}
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for sec in SECTIONS:
        which = sec.get("src", "ag")
        nb_name, nb_cells = source_cells[which]
        index = sec[which]
        src = nb_cells[index]
        text = build_module(sec, src, nb_name, index)
        GENERATED[sec["module"]] = text
        try:
            ast.parse(text)
        except SyntaxError as exc:
            print(f"  !! {sec['module']}: generated module does not parse: {exc}")
            continue
        if sec.get("scope", "rock") == "rock":
            for pattern, why in FORBIDDEN_LITERALS:
                hit = pattern.search(text)
                if hit:
                    raise SystemExit(
                        f"{sec['module']}: {why} -- {hit.group(0)!r}. A shared "
                        "module must read it from the bound lithology, or both "
                        "rocks will silently run the same number.")

        path = OUT / f"{sec['module']}.py"
        print(f"  {sec['module']:26s} <- {which} cell {index:2d} "
              f"({len(src.splitlines()):5d} lines, scope={sec.get('scope','rock')})")
        if not args.check:
            path.write_text(text, encoding="utf-8")
        written.append(path)

    if not args.check:
        init = OUT / "__init__.py"
        names = [s["module"] for s in SECTIONS]
        init.write_text(
            '"""Per-section analysis modules shared by both lithology notebooks.\n\n'
            "One module per section of the analysis, in notebook order. Each exposes\n"
            "``main(rock)`` and is driven by a :class:`tools.lithology.Lithology`, so\n"
            "the gneiss and schist notebooks call identical code.\n"
            '"""\n\n'
            + "".join(f"from . import {n}  # noqa: F401\n" for n in names)
            + "\n__all__ = [\n"
            + "".join(f'    "{n}",\n' for n in names)
            + "]\n", encoding="utf-8")
    print(f"\n{len(written)} section modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
