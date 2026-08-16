#!/usr/bin/env python3
"""Generate the three publication notebooks from one description.

The two lithology notebooks are generated from a single template, so the gneiss
and schist workflows cannot drift apart: there is no second copy of the analysis
to fall behind. They differ only in the ``Lithology`` they bind.

None of the three notebooks defines a function. Every call goes into ``tools/``,
where ``tools/analysis/`` holds one module per section of the analysis --
extracted from the old inline cells during the notebook-to-package migration,
which is where the record of what moved and what was dropped lives.

Layout
------
``Tensile_augen_gneiss.ipynb`` and ``Tensile_psammitic_schist.ipynb`` carry the
per-specimen analysis for one rock. ``Tensile_general_plots.ipynb`` carries
everything that compares the two -- the strength models fitted across both
rocks, mesh convergence over all fourteen specimens, the k_max sweep, and the
metric export.

    python scripts/build_notebooks.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CELLS = REPO / "scripts" / "notebook_cells"

KERNEL = dict(
    kernelspec=dict(display_name="Python (viscoquake)", language="python",
                    name="viscoquake"),
    language_info=dict(name="python", version="3.13.5"))

LITHOLOGIES = [
    dict(key="augen_gneiss", const="AUGEN_GNEISS", display="Augen gneiss",
         samples="1-7", notebook="Tensile_augen_gneiss.ipynb",
         note="an inclusion-bearing gneiss with discontinuous foliation, "
              "weak-plane spacing 10 mm"),
    dict(key="psammitic_schist", const="PSAMMITIC_SCHIST",
         display="Psammitic schist", samples="8-14",
         notebook="Tensile_psammitic_schist.ipynb",
         note="a continuous-foliation schist, weak-plane spacing 2 mm"),
]

#: The per-lithology analysis, in the order the original notebooks ran it.
#: Each entry is one ``tools.analysis`` module driven for the bound rock.
ROCK_SECTIONS = [
    ("Displacement-discontinuity solve", "ddm_disk", """
The orthotropic moduli fields, the Hertzian contact load and the cut-cell
boundary treatment, solved by preconditioned conjugate gradient with a
BiCGSTAB fallback. Everything downstream reads the fields this produces."""),
    ("Stress traverses across the weak planes", "stress_graph", """
Radial traverses on the loading diameter, with the weak-band crossings
marked, so the stress concentration at each fabric crossing is visible
against the smooth background."""),
    ("Stress distribution over the disc", "stress_field", """
The full-disc stress field at each fabric angle."""),
    ("Tensile-strain proxy", "strain_proxy", """
The tensile-strain proxy field and the principal direction associated with
it, smoothed nematically so the 180-degree periodicity of an orientation is
respected."""),
    ("Principal tensile strain", "principal_strain", """
Principal tensile strain panels taken from the same solved fields."""),
    ("Mid-section displacement profiles", "displacement_profiles", """
Mean horizontal displacement across the mid-section, which is the measurable
signature of the fabric opening under load."""),
    ("Stress direction circles", "direction_circles", """
Principal-direction circles around the disc boundary."""),
    ("Crack growth, energy and failure fields", "crack_energy_suite", """
Energy-guided displacement-discontinuity crack growth, the four-class failure
maps, the strain-energy field and the stress-tensor glyphs."""),
    ("Crack-path studies", "crack_path_suite", """
Crack paths under energy and field guidance, including the four-class
failure-map guidance block."""),
    ("Fracture deviation from the foliation", "foliation_deviation", """
How far the observed fracture departs from the fabric plane, by angle."""),
]

#: The cross-lithology analysis. These fit or compare both rocks in one pass.
BOTH_SECTIONS = [
    ("Mixed-mode fraction against k_max", "kmax_sweep", """
The mixed-mode area fraction as the local damage cap is swept, computed
separately for the gneiss and schist specimen groups."""),
    ("Local Mohr-Coulomb classification", "mohr_coulomb_local", """
The local Mohr-Coulomb failure-mode classification over all fourteen
specimens, across the spacing and heterogeneity grid."""),
    ("Metric export", "metric_export", """
The formatted spreadsheet of manuscript values."""),
]


def md(text):
    lines = text.strip("\n").split("\n")
    return dict(cell_type="markdown", metadata={},
                source=[l + "\n" for l in lines[:-1]] + [lines[-1]])


def code(text):
    lines = text.strip("\n").split("\n")
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=[l + "\n" for l in lines[:-1]] + [lines[-1]])


def carry_over_outputs(nb, path):
    """Keep the stored outputs of any cell whose source is unchanged.

    Rebuilding is cheap; executing these notebooks is not -- roughly forty
    minutes each. Without this, editing one line of prose discards every
    result, and the temptation is then to hand-edit the notebook instead of
    the template, which is exactly what the generator exists to prevent.

    Matching is on exact source text, so a cell whose code changed correctly
    loses its outputs and shows as unexecuted. Execution counts come along for
    the ride; they will be inconsistent until the next full run, which is the
    honest signal that the notebook is part-executed.
    """
    if not path.exists():
        return nb
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return nb

    stored = {}
    for cell in old.get("cells", []):
        if cell.get("cell_type") != "code" or not cell.get("outputs"):
            continue
        stored.setdefault("".join(cell["source"]), cell)

    kept = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        match = stored.get("".join(cell["source"]))
        if match is not None:
            cell["outputs"] = match["outputs"]
            cell["execution_count"] = match.get("execution_count")
            kept += 1
    nb["_kept_outputs"] = kept
    return nb


def preserved(name):
    """Cells kept verbatim from the previous notebooks."""
    data = json.loads((CELLS / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for c in data:
        maker = md if c["cell_type"] == "markdown" else code
        out.append(maker(c["source"]))
    return out


def build_lithology(cfg):
    L, D = cfg["const"], cfg["display"]
    cells = [
        md(f"""
# Brazilian tensile fracture in {D.lower()}

Stress, strain and fracture analysis of {cfg['note']}, specimens
{cfg['samples']} at fabric angles 0-90 degrees in 15 degree steps.

**This notebook contains no function definitions.** Every call goes into
`tools/`; the analysis sections live in `tools/analysis/`, one module per
section, and both lithology notebooks run the identical code with only the
bound lithology differing. That is what keeps the two rocks comparable.

Analysis that spans both lithologies -- the fitted strength models, mesh
convergence, the k_max sweep and the metric export -- is in
`Tensile_general_plots.ipynb`, not here.
"""),
        md("## 1. Environment and reproducibility"),
        code("""
%matplotlib inline
# Inline is the default under ipykernel, but stating it makes the stored
# outputs of a batch `nbconvert --execute` run independent of that default.

import numpy as np, pandas as pd, matplotlib

from tools import conventions, lithology, traces, failure_classification
from tools import plotting, export, strain_partitioning
from tools import analysis
from tools.plot_style import apply_plot_style

TICKS = apply_plot_style()          # mandated publication style
SEED = 20260807                     # orientation-fit bootstrap

print('numpy', np.__version__, '| pandas', pd.__version__,
      '| matplotlib', matplotlib.__version__)
print('seed', SEED)
"""),
        md("""
## 2. Lithology configuration

Specimen identifiers are fixed by the experimental programme and are never
shared between lithologies. The weak-plane spacing and phase-warp amplitude
are properties of the rock, so they travel with it rather than being repeated
in each section. The assertion fails loudly if this notebook is ever pointed
at the other rock's specimens.
"""),
        code(f"""
ROCK = lithology.{L}
SAMPLES = ROCK.sample_ids

lithology.assert_notebook_lithology(ROCK.key, SAMPLES)

print(f'{{ROCK.display_name}}: samples {{SAMPLES}}')
print(f'fabric angles      : {{ROCK.angles_deg}} deg')
print(f'weak-plane spacing : {{ROCK.spacing_m * 1e3:.1f}} mm')
print(f'phase-warp amplitude: {{ROCK.spacing_warp_amp_m * 1e3:.1f}} mm')
"""),
        md("""
## 3. Convention

Stated once, in `tools.conventions`, and used everywhere: right-handed
(x, y) in metres, disc centre at the origin, **+y the loading axis**, angles
counterclockwise from +x, orientations axial (180-degree periodic), and
tension-positive at the classifier interface.
"""),
        code("""
for a in (0, 90):
    print(conventions.describe_end_member(a))
"""),
    ]

    for n, (title, module, blurb) in enumerate(ROCK_SECTIONS, start=4):
        cells.append(md(f"## {n}. {title}\n{blurb}"))
        cells.append(code(f"analysis.{module}.main(ROCK)"))

    cells.append(md(f"""
---

# Classification, orientation validation and energy partitioning

Diagnostics for {D.lower()}, computed from the fields the sections above
exported, by the shared modules in `tools/`. Comparisons between the two rocks
live in `Tensile_general_plots.ipynb`.

This block supersedes the earlier tensile/shear/mixed classification, the
stress-profile panel built from a uniform placeholder field, and the mode counts
that were carried as literal lists. Each cell stands on its own and does not
depend on variables left behind by the sections above.
"""))
    cells.extend(preserved("lithology"))
    return dict(cells=cells, metadata=KERNEL, nbformat=4, nbformat_minor=5)


def build_general():
    cells = [
        md("""
# Cross-lithology comparison

Everything in this study that compares augen gneiss with psammitic schist, or
fits a model to both at once. The per-specimen fields for each rock are in
`Tensile_augen_gneiss.ipynb` and `Tensile_psammitic_schist.ipynb`.

**This notebook contains no function definitions.** The sections below call
`tools/analysis/`, the same package the lithology notebooks use, so a model
fitted here and a field plotted there cannot disagree about the rock.
"""),
        md("## 1. Environment"),
        code("""
%matplotlib inline
# Inline is the default under ipykernel, but stating it makes the stored
# outputs of a batch `nbconvert --execute` run independent of that default.

import numpy as np, pandas as pd, matplotlib

from tools import lithology, analysis
from tools.plot_style import apply_plot_style

TICKS = apply_plot_style()

GNEISS, SCHIST = lithology.AUGEN_GNEISS, lithology.PSAMMITIC_SCHIST
print('numpy', np.__version__, '| pandas', pd.__version__,
      '| matplotlib', matplotlib.__version__)
for r in (GNEISS, SCHIST):
    print(f'{r.display_name:18s} samples {r.sample_ids}  '
          f'spacing {r.spacing_m * 1e3:.1f} mm')
"""),
    ]

    for n, (title, module, blurb) in enumerate(BOTH_SECTIONS, start=2):
        cells.append(md(f"## {n}. {title}\n{blurb}"))
        cells.append(code(f"analysis.{module}.main()"))

    cells.append(md("""
---

# Consolidated cross-lithology diagnostics

Strength anisotropy in both loading modes, the asymmetric strength envelope and
its fit, the fabric-resolved tractions, failure statistics counted from the
fields rather than carried as literals, sensitivity to spacing and disorder, the
softening cap, fracture-trace validation against a loading-direction null, the
supplementary bounds, and a check that every graphic the manuscript cites
exists.

Per-lithology diagnostics stay in the two lithology notebooks. Every cell here
is self-contained.
"""))
    cells.extend(preserved("general"))
    return dict(cells=cells, metadata=KERNEL, nbformat=4, nbformat_minor=5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would be written, write nothing")
    args = ap.parse_args(argv)

    targets = [(cfg["notebook"], build_lithology(cfg)) for cfg in LITHOLOGIES]
    targets.append(("Tensile_general_plots.ipynb", build_general()))

    for name, nb in targets:
        path = REPO / name
        nb = carry_over_outputs(nb, path)
        kept = nb.pop("_kept_outputs", 0)
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        note = f", kept outputs of {kept}" if kept else ""
        print(f"  {name:34s} {len(nb['cells']):3d} cells ({n_code} code{note})")
        if not args.check:
            path.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
