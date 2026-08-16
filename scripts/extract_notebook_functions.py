#!/usr/bin/env python3
"""Move the functions duplicated in both notebooks into ``tools/ddm/``.

Both lithology notebooks carried the same displacement-discontinuity toolkit
inline — the same functions, defined twice, byte for byte. That is the reason
the notebooks were hard to read and the reason the two lithologies could drift
apart silently. This script lifts the shared machinery into a package and
leaves the notebooks holding configuration, calls and interpretation.

What moves
----------
A function moves only if it is

  1. defined in *both* notebooks with identical source,
  2. defined with *one* body throughout — see below, and
  3. free of notebook state — every global it reads is either an import, a
     literal constant that is also identical in both notebooks, or another
     function that is itself moving.

Condition 2 is the subtle one. Many of these names are defined several times as
the notebook proceeds, and 44 of them are redefined with a *different* body, so
which version runs depends on how far down the notebook you are. A call in cell
40 does not mean what the same call means in cell 8. Collapsing those to a
single module-level definition would quietly change results, so any name whose
body is not constant across every definition in both notebooks stays where it
is. Only names that are genuinely one function move.

Condition 3 is a fixed point: a function that reads a computed global such as
the stress grid ``X`` stays behind, and so does anything that calls it. That is
deliberate. Those functions are closures over the notebook's live state, and
moving them would change their meaning.

Constants stay in the notebooks as well as moving into the package. They are
the parameter block a reader wants in front of them, and duplicating a literal
costs nothing.

Phases
------
``--generate`` writes ``tools/ddm/`` and is still how that package is
maintained. ``--rewrite`` is retired: the notebooks are generated now by
``scripts/build_notebooks.py`` and hold no definitions to remove.

The scan reads ``legacy_notebooks/pre_refactor/``, which is where the inline
definitions still are. The working notebooks contain none.

    python scripts/extract_notebook_functions.py --check    # report only
    python scripts/extract_notebook_functions.py
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import re
import uuid
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: This reads the *pre-refactor* notebooks, which are where the inline
#: definitions were. The working notebooks are generated now and contain no
#: definitions at all, so scanning them would find nothing to move.
SOURCE_DIR = REPO / "legacy_notebooks" / "pre_refactor"
NOTEBOOKS = ["Tensile_augen_gneiss.ipynb", "Tensile_psammitic_schist.ipynb"]
PKG = REPO / "tools" / "ddm"

#: Names that are always available and never count as a notebook dependency.
AMBIENT = set(dir(builtins)) | {"__name__", "__file__", "display", "get_ipython"}

THEMES = [
    ("angles", r"^(wrap_|angle_|axis_angle|axial_stats|ensure_radians|nematic|"
               r"bilinear_angle|bilinear_sample_line|blend_line|deviation_arrays|"
               r"trough_angle|xi_from_u|map_angle)"),
    ("grids", r"(grid|mesh|mask|downsample|reconstruct|clip_to_disk|curve_on_grid|"
              r"_img_from|sample_line|thin_points|extend_to_circle|make_segments|"
              r"boundary_segments|build_segments|polyline|_pca_axis|snap_path)"),
    ("smoothing", r"(smooth|blur|kde|bandwidth|gaussian_random|_collapse|robust_|"
                  r"nearest_mean|mean_profile|bootstrap_band|compute_poly_ci)"),
    ("elasticity", r"(orthotropic|q_matrix|qbar|transform_Q|modulus|to_MPa|strain|"
                   r"stress|principal|sigma|splitting|tensile|Q_to)"),
    ("failure", r"(failure|anisotrop|weak_band|wp_weight|band_open|asym_activation|"
                r"contact_arc|platen|hertz|kn_ks|resolve_local|surrogate|psi_pref|"
                r"harmonic_mix|min_fractions)"),
    ("io", r"(npz|save_|load_|_search_dirs|get_first_col|parse_sample|sanitize|"
           r"output_prefix|import_or_load|bind_sif|validate_solver|replot|read)"),
    ("figures", r"(plot_|save_all|make_figure|_label_|_get_marker|compact_axis|"
                r"add_band_lines|set_pub_style|apply_plot_style|_make_all_combined|"
                r"colorbar|_mode_to_code)"),
    ("fitting", r"(fit_|model_|compute_metric|compute_metrics|rmse|choose_smallest|"
                r"m7_report|pooled|compare_with|summarize|get_anisotropy|"
                r"get_sigma_weights|get_row_material|generate_random|print_material)"),
]

THEME_DOC = {
    "angles": "Angle wrapping, axial statistics and orientation blending.",
    "grids": "Disc grids, masks, polylines and segment geometry.",
    "smoothing": "Smoothing, kernel density, bootstrap bands and curve collapse.",
    "elasticity": "Orthotropic constitutive relations, stresses and strains.",
    "failure": "Failure criteria, weak-band weighting and contact/platen models.",
    "io": "Field archive read/write, path discovery and solver binding.",
    "figures": "Publication figure helpers.",
    "fitting": "Model fitting, metrics and measured-data comparison.",
    "solver": "Finite-difference operators, tractions and the energy-guided stepper.",
}


def _h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _theme(name: str) -> str:
    for t, rx in THEMES:
        if re.search(rx, name):
            return t
    return "solver"


# --------------------------------------------------------------------------
# notebook inspection
# --------------------------------------------------------------------------

def scan(path: Path):
    """Top-level functions, literal constants and imports, per notebook.

    ``funcs`` maps a name to *every* definition of it, in notebook order. A name
    with more than one distinct body is a redefinition and must not move.
    """
    funcs, consts, imports = {}, {}, {}
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue          # IPython magics etc. — nothing to move from here
        lines = src.splitlines()
        for n in tree.body:
            seg = "\n".join(lines[n.lineno - 1:n.end_lineno])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(n.name, []).append((seg, i, n))
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    imports[a.asname or a.name.split(".")[0]] = seg
            elif isinstance(n, ast.Assign):
                try:
                    ast.literal_eval(n.value)
                    literal = True
                except Exception:
                    literal = False
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = (seg, i, literal)
    return funcs, consts, imports


def free_globals(src: str) -> set:
    """Global names a function reads, ignoring anything it binds itself."""
    fn = ast.parse(src).body[0]
    bound, used = set(), set()
    for a in ast.walk(fn):
        if isinstance(a, ast.arg):
            bound.add(a.arg)
        elif isinstance(a, ast.Name):
            (bound if isinstance(a.ctx, ast.Store) else used).add(a.id)
        elif isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(a.name)
        elif isinstance(a, (ast.Import, ast.ImportFrom)):
            for al in a.names:
                bound.add(al.asname or al.name.split(".")[0])
        elif isinstance(a, ast.ExceptHandler) and a.name:
            bound.add(a.name)
        elif isinstance(a, (ast.Global, ast.Nonlocal)):
            bound.update(a.names)
    return used - bound - AMBIENT


def _single_body(defs):
    """The one source of a name, or None if it is redefined with a different body."""
    bodies = {_h(seg) for seg, _, _ in defs}
    return defs[0][0] if len(bodies) == 1 else None


def plan():
    """Decide what moves. Returns (movable, constants_needed, deps, scan_a, scan_b)."""
    a = scan(SOURCE_DIR / NOTEBOOKS[0])
    b = scan(SOURCE_DIR / NOTEBOOKS[1])
    af, ac, ai = a
    bf, bc, _ = b

    # one constant body per notebook, and the same body in both
    stable = {}
    for k, defs in af.items():
        if k not in bf:
            continue
        sa, sb = _single_body(defs), _single_body(bf[k])
        if sa is not None and sa == sb:
            stable[k] = sa

    identical = set(stable)
    deps = {k: free_globals(stable[k]) for k in identical}
    shared_consts = {c for c, (seg, _, lit) in ac.items()
                     if lit and c in bc and _h(seg) == _h(bc[c][0])}

    # fixed point: drop anything reaching outside the moving set
    movable = set(identical)
    while True:
        escaped = {k for k in movable
                   if deps[k] - movable - shared_consts - set(ai)}
        if not escaped:
            break
        movable -= escaped

    needed = set().union(*(deps[k] for k in movable)) & shared_consts if movable else set()
    return movable, needed, deps, stable, a, b


# --------------------------------------------------------------------------
# phase 1 — generate the package
# --------------------------------------------------------------------------

def generate(movable, consts_needed, stable, scan_a):
    _, ac, ai = scan_a
    af = stable                      # name -> its single, unambiguous source
    assign = {f: _theme(f) for f in movable}
    themes = sorted(set(assign.values()))

    const_src, seen = [], set()
    for c in sorted(consts_needed, key=lambda k: (ac[k][1], ac[k][0])):
        seg = ac[c][0]
        if seg not in seen:
            seen.add(seg)
            const_src.append(seg)

    body = "\n".join(af[f] for f in sorted(movable)) + "\n" + "\n".join(const_src)
    tree = ast.parse(body)
    needed = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    needed |= {n.value.id for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    imports = sorted({stmt for name, stmt in ai.items() if name in needed})

    PKG.mkdir(parents=True, exist_ok=True)
    rule = "# " + "-" * 74

    txt = ['"""Shared implementation of the displacement-discontinuity toolkit.',
           "",
           "Every function here was defined identically in both lithology notebooks.",
           "It lives in one place so the two cannot drift apart, and so the notebooks",
           "themselves carry analysis and interpretation rather than machinery.",
           "",
           "Import from the themed views (:mod:`tools.ddm.angles`, :mod:`tools.ddm.grids`,",
           "...) or from the :mod:`tools.ddm` package root, not from this module.",
           "",
           "Generated by ``scripts/extract_notebook_functions.py``. Edit the functions",
           "here — never paste a copy back into a notebook.",
           '"""',
           "from __future__ import annotations",
           ""]
    txt += imports + ["", rule, "# Module-level constants (identical in both notebooks)",
                      rule, ""]
    txt.append("\n\n".join(const_src))
    for t in themes:
        fns = sorted(f for f in movable if assign[f] == t)
        txt += ["", rule, f"# {t}: {THEME_DOC[t]}", rule, ""]
        txt.append("\n\n\n".join(af[f] for f in fns))
    (PKG / "_toolkit.py").write_text("\n".join(txt) + "\n", encoding="utf-8")

    for t in themes:
        fns = sorted(f for f in movable if assign[f] == t)
        m = [f'"""{THEME_DOC[t]}', "", "Re-exported from :mod:`tools.ddm._toolkit`.",
             '"""', "", "from ._toolkit import (  # noqa: F401"]
        m += [f"    {f}," for f in fns]
        m += [")", "", "__all__ = ["] + [f'    "{f}",' for f in fns] + ["]"]
        (PKG / f"{t}.py").write_text("\n".join(m) + "\n", encoding="utf-8")

    init = ['"""Displacement-discontinuity toolkit shared by both lithology notebooks.',
            "",
            f"{len(movable)} functions and {len(const_src)} constants that were",
            "previously duplicated verbatim in each notebook, grouped into themed views:",
            ""]
    init += [f"  {t:<11} {THEME_DOC[t]}" for t in themes]
    init += ["", "Every name is also importable from this package root.", '"""', "",
             "from . import " + ", ".join(themes) + "  # noqa: F401",
             "from ._toolkit import (  # noqa: F401"]
    for t in themes:
        init.append(f"    # {t}")
        init += [f"    {f}," for f in sorted(f for f in movable if assign[f] == t)]
    init += ["    # constants"] + [f"    {c}," for c in sorted(consts_needed)]
    init += [")", "", "__all__ = ["]
    init += [f'    "{n}",' for n in sorted(movable) + sorted(consts_needed)]
    init += ["]"]
    (PKG / "__init__.py").write_text("\n".join(init) + "\n", encoding="utf-8")

    print(f"  tools/ddm/: {len(movable)} functions, {len(const_src)} constant "
          f"statements, {len(themes)} themed views")
    return assign


# --------------------------------------------------------------------------
# phase 2 — rewrite the notebooks
# --------------------------------------------------------------------------

def _strip_cell(src: str, drop: set):
    """Remove the given top-level defs, plus the comments attached above them."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, []
    lines = src.splitlines()
    kill, removed = set(), []
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if n.name not in drop:
            continue
        start = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
        # absorb a contiguous comment block sitting directly on top of the def
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        kill.update(range(start, n.end_lineno))
        removed.append(n.name)
    if not removed:
        return src, []
    kept = [l for i, l in enumerate(lines) if i not in kill]
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    out, blanks = [], 0
    for l in kept:                      # collapse runs of blank lines
        blanks = blanks + 1 if not l.strip() else 0
        if blanks < 3:
            out.append(l)
    return "\n".join(out), removed


def import_cell(movable, assign):
    themes = sorted(set(assign.values()))
    lines = ["# The displacement-discontinuity toolkit below was defined inline in both",
             "# notebooks. It now lives in tools/ddm/ so the two lithologies provably",
             "# share one implementation. Names are unchanged, so every call site below",
             "# reads exactly as before.",
             "from tools.ddm import (  # noqa: F401"]
    for t in themes:
        fns = sorted(f for f in movable if assign[f] == t)
        lines.append(f"    # {t} — {THEME_DOC[t]}")
        row = "   "
        for f in fns:
            if len(row) + len(f) + 2 > 84:
                lines.append(row)
                row = "   "
            row += f" {f},"
        if row.strip():
            lines.append(row)
    lines += [")", "",
              "print(f'tools.ddm: {len(__import__(\"tools.ddm\", fromlist=[\"x\"]).__all__)}"
              " shared names imported')"]
    return dict(cell_type="code", id=uuid.uuid4().hex[:8], metadata={},
                execution_count=None, outputs=[],
                source=[l + "\n" for l in lines])


def rewrite(movable, assign):
    """Remove the moved definitions from the notebooks and insert the import.

    Retired. The notebooks are generated by ``scripts/build_notebooks.py`` and
    hold no definitions to remove, so this phase has nothing to do and any edit
    it made would be discarded at the next rebuild. ``--generate`` still works
    and is what maintains ``tools/ddm``.
    """
    print("  --rewrite is retired: the notebooks are generated and contain no\n"
          "  definitions. Run scripts/build_notebooks.py instead.")
    return

    for nb in NOTEBOOKS:                                  # pragma: no cover
        path = SOURCE_DIR / nb
        doc = json.loads(path.read_text(encoding="utf-8"))
        cells = doc["cells"]
        n_before = len(cells)

        removed_all, emptied = [], []
        for c in cells:
            if c["cell_type"] != "code":
                continue
            new, removed = _strip_cell("".join(c["source"]), movable)
            if not removed:
                continue
            removed_all += removed
            c["source"] = [l + "\n" for l in new.splitlines()]
            c["outputs"] = []
            c["execution_count"] = None
            if not new.strip():
                emptied.append(id(c))

        # drop cells this pass emptied, and any that were already blank
        n_blank = sum(1 for c in cells
                      if id(c) not in emptied and not "".join(c["source"]).strip())
        cells[:] = [c for c in cells
                    if id(c) not in emptied and "".join(c["source"]).strip()]

        # insert the import right after the notebook's first import cell
        at = next((i for i, c in enumerate(cells)
                   if c["cell_type"] == "code"
                   and re.search(r"^\s*(import|from)\s", "".join(c["source"]), re.M)), 0)
        cells.insert(at + 1, import_cell(movable, assign))

        for c in cells:
            c.setdefault("id", uuid.uuid4().hex[:8])
        doc["nbformat_minor"] = max(doc.get("nbformat_minor", 5), 5)
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"  {nb}: {n_before} -> {len(cells)} cells, "
              f"{len(removed_all)} definitions removed, "
              f"{len(emptied)} cells emptied and dropped, "
              f"{n_blank} already-blank cells dropped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report the plan, change nothing")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--rewrite", action="store_true")
    a = ap.parse_args()
    do_gen = a.generate or not (a.generate or a.rewrite or a.check)
    do_rw = a.rewrite or not (a.generate or a.rewrite or a.check)

    movable, consts, deps, stable, sa, sb = plan()
    identical = set(stable)
    print(f"one unambiguous body in both notebooks: {len(identical)}")
    print(f"movable (also free of notebook state):  {len(movable)}")
    print(f"staying behind:                         {len(identical - movable)}")
    if a.check:
        assign = {f: _theme(f) for f in movable}
        for t, n in sorted(Counter(assign.values()).items()):
            print(f"    {t:<11} {n}")
        print("  stays:", ", ".join(sorted(identical - movable)))
        return 0
    if not movable:
        print("nothing to move — the notebooks are already migrated")
        return 0

    assign = {f: _theme(f) for f in movable}
    if do_gen:
        assign = generate(movable, consts, stable, sa)
    if do_rw:
        rewrite(movable, assign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
