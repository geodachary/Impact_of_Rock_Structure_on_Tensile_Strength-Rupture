# Impact of rock anisotropy on tensile strength and modes of rupture

Code and data supporting the study of anisotropic tensile fracture in foliated
metamorphic rocks.

**D. Acharya and D. Elsworth**
Submitted to the *Journal of the Mechanics and Physics of Solids*

> This repository provides research code and data supporting the reproducibility
> of the results reported in the associated publication. It reflects a research
> workflow and is not distributed as a general-purpose software package.

**Archived release:** [10.5281/zenodo.22644302](https://doi.org/10.5281/zenodo.22644302) (concept DOI [10.5281/zenodo.22591641](https://doi.org/10.5281/zenodo.22591641) always resolves to the latest version)

---

## 1. Overview

The study combines angle-resolved Brazilian disc experiments with an
anisotropy-aware numerical framework to examine how a planar rock fabric
controls both the tensile strength of a rock and the mode in which it ruptures.

Two foliated Himalayan lithologies are analyzed, augen gneiss and psammitic
schist, at seven fabric orientations each.

The numerical framework combines:

- orthotropic elasticity, through a Lekhnitskii stress solution on the disc;
- a directional tensile strength, applied as a tensile cap;
- mesoscale heterogeneity of the elastic and strength fields;
- foliation spacing, which differs by an order of magnitude between the two
  rocks and is carried as a property of the rock rather than as a fitted number;
- a failure classifier that resolves the **mechanism** of failure (tensile or
  shear) and its **structural locus** (weak plane or intact matrix) together,
  rather than collapsing the two into a single label;
- strain-energy partitioning across those mechanisms.

The framework is used to reproduce tensile strength as a function of loading
angle, to compute the stress, displacement and strain-energy fields, to identify
where and in what mode fracture initiates, and to generate the reported figures
and tables.

Both lithologies are included here. They run the same code and differ only in
the material parameters and fabric spacing bound to each, which is a property
the test suite enforces rather than a claim made in prose.

## 2. Lithologies and specimen numbering

| Lithology | Specimens | Fabric angles `α_exp` | Weak-plane spacing |
|---|---|---|---|
| Augen gneiss | 1 to 7 | 0°, 15°, 30°, 45°, 60°, 75°, 90° | 10 mm |
| Psammitic schist | 8 to 14 | 0°, 15°, 30°, 45°, 60°, 75°, 90° | 2 mm |

Specimen identifiers are never shared between lithologies, and
`tools.lithology` enforces this with an assertion the test suite exercises.
The spacing and the phase-warp amplitude live on the `Lithology` object, so no
analysis module carries either as a literal. They were previously repeated
inline in every long notebook cell, which is how the two rocks drifted apart.

`α_exp` is measured from the horizontal diameter: at 0° the foliation trace is
perpendicular to the loading direction, and at 90° it is parallel to it.

## 3. The four failure mechanisms

| Class | Meaning |
|---|---|
| **WT** | tensile opening along the weak plane (foliation) |
| **WS** | shear sliding along the weak plane |
| **MT** | tensile cracking through the intact matrix |
| **MS** | shear cracking through the intact matrix |

Each point is assigned to the mechanism with the largest admissible utility
ratio. Weak-plane utilities are admissible only where the modeled foliation
activation weight exceeds a stated floor, so a matrix point cannot be labeled a
weak-plane failure.

**Mixed mode is a secondary descriptor, not a fifth class.** It flags points
where the second-largest admissible utility lies within `η_mix` of the largest,
and it never replaces the primary WT/WS/MT/MS assignment.

## 4. Fracture traces

Predicted traces are **heuristic diagnostic trajectories**, not solutions of a
progressive crack-growth problem. The framework identifies where and in what
mode fracture initiates; it does not simulate propagation, arrest or
re-initiation.

Observed and predicted traces are paired one to one by specimen, never by
filename sorting:

```
crack_digitized_data/<lithology>_<angle>_sample_<N>.csv
    ↔  outputs/fields/ddm_crack_sample_<N>.csv
```

for `N = 1` to `14`. Both are in meters in the same specimen frame: disc center
at the origin, `+x` the horizontal diameter, `+y` the loading axis.

A crack-path step is recorded as one of two kinds. An **energy-driven** step is
chosen by maximizing `G − G_c` over admissible directions. Where no admissible
direction clears `G_c`, the stepper follows the preferred-orientation field
instead, and that step is **sub-critical by construction**. Roughly 62% of the
path is of the second kind, so any statistic computed over all steps measures
the mix of the two as much as it measures the crack. `tools.fracture_energy`
reports both, and judges arrest on the energy-driven steps alone.

---

## 5. Repository structure

```
tools/                     the analysis package; every reusable function lives here
  conventions.py             coordinate, angle and stress-sign conventions
  lithology.py               per-lithology configuration and the 14-specimen pairing
  output_dirs.py             the single authority for where output goes
  traces.py                  trace loading, primary-segment rule, orientation fitting
  failure_classification.py  WT/WS/MT/MS utilities and the secondary mixed flag
  fracture_energy.py         along-path G/G_c, energy-driven vs sub-critical steps
  ati_model.py               the two-parameter anisotropic tensile strength fit
  plotting.py                shared publication style, seven-panel figure builders
  export.py                  provenance-carrying table export
  strain_partitioning.py     classifier-driven strain-energy partition
  ddm/                       displacement-discontinuity toolkit shared by both rocks
  analysis/                  one module per section of the analysis, shared by both
  <mechanics modules>        stress, geometry, rotation, Airy solution, crack helpers

scripts/reproduce_all.py               one-command reproduction of tables and figures
scripts/build_notebooks.py             regenerates the three notebooks
scripts/smoke_sections.py              runs every section once, both lithologies
scripts/make_*.py                      the individual figure and table generators
tests/                                 unit, integration and regression tests

Tensile_augen_gneiss.ipynb        specimens 1 to 7
Tensile_psammitic_schist.ipynb    specimens 8 to 14
Tensile_general_plots.ipynb       cross-lithology models and comparisons

crack_digitized_data/      raw: digitized laboratory fracture traces
tensile_samples_data.csv   raw: Brazilian-test strengths, cohesion, friction angle
selected_all_samples.*     raw: the replicate measurements, spreadsheet and CSV
data_mean.ipynb            averages the replicates into tensile_samples_data.xlsx,
                           from which the .csv above is exported

outputs/                   everything generated, sorted by what it is
  figures/                 every figure, in every format it is saved in
  tables/                  every machine-readable result
  fields/                  cached solver state that later steps read back
```

There are no loose Python modules at the repository root: all project code lives
in `tools/`, `scripts/` or `tests/`.

**The notebooks define no functions.** Each is a sequence of markdown and
one-line calls into `tools/`. The analysis itself lives in `tools/analysis/`,
one module per section, and the two lithology notebooks are generated from a
single template by `scripts/build_notebooks.py`, which is what guarantees that
gneiss and schist run identical code and differ only in the `Lithology` they
bind.

Editing a notebook cell by hand will be overwritten on the next build. Edit
instead:

| To change | Edit |
|---|---|
| what an analysis does | the module in `tools/analysis/` |
| notebook structure or prose | `scripts/build_notebooks.py` |
| a cell carried over verbatim | `scripts/notebook_cells/*.json` |

then run `python scripts/build_notebooks.py`. The builder preserves the stored
outputs of any cell whose source is unchanged, so editing prose does not discard
results. A cell whose *code* changed correctly loses its outputs and shows as
unexecuted.

### On `tools/analysis/`

Those modules were mechanically extracted from an earlier form of the notebooks,
which is why each carries a header naming the cell it came from. **They ship as
ordinary source and are the released form of the analysis**: edit them directly.

The extraction tooling and the pre-refactor notebooks it reads are development
material and are not distributed. Nothing here depends on them. The cached field
archives in `outputs/fields/` are what make the results reproducible, and they
are included.

## 6. Where generated output goes

Everything generated lands under one directory, split by what the file is
rather than by which section produced it:

| Directory | Holds |
|---|---|
| `outputs/figures/` | every figure, as PDF and PNG |
| `outputs/tables/` | every machine-readable result a reader would open |
| `outputs/fields/` | solver state later steps read back: the cached field archives, the per-specimen crack traces, the plot caches |

`outputs/fields/` is the expensive part. It is what allows the tables and
figures to be rebuilt without re-running the multi-hour
displacement-discontinuity solve.

`tools/output_dirs.py` is the single authority for these three paths, and the
extractor applies them to the generated analysis modules, so no section carries
an output directory of its own.

---

## 7. Installation

Python 3.10 or later (developed and tested on 3.13.5).

```bash
python -m venv .venv && source .venv/bin/activate      # or conda create ...
python -m pip install -e .
python -m pip install -r requirements.txt              # pinned tested versions
```

## 8. Running the tests

```bash
python -m pytest tests/ -q
```

The suite covers the angle and stress conventions, the four-mechanism
classifier, trace loading and orientation fitting, the 14-specimen pairing, the
publication plot style, and non-regression of the reported numbers. It also
pins several structural properties that are easy to break silently:

- the two lithology notebooks run identical code and differ only in the
  `Lithology` they bind;
- no notebook defines a function;
- no figure filename is written by more than one place;
- every generator script is reachable from a notebook;
- axis labels carry no LaTeX escapes, which would print literally.

## 9. Reproducing the analysis outputs

```bash
python scripts/reproduce_all.py --mode full
```

This validates the raw inputs, records the environment and seed, regenerates the
validation tables and the four seven-panel composite figures from the cached
field archives, and exits non-zero on any failure. A fast structural check is
available with `--mode smoke`.

On a fresh clone, run the notebooks first (Section 10). The fourteen field
archives under `outputs/fields/fields_npz/` are tracked and carry most of what
this script needs, but the mid-plane displacement cache it uses for the
corridor statistics is not: it is keyed on a hash of the specimen geometry and
material, so committing it would only pin a stale solve. The script names that
dependency if the cache is absent.

| Output | Path |
|---|---|
| Specimen pairing manifest | `outputs/tables/sample_pairing_manifest.csv` |
| Orientation comparison | `outputs/tables/trace_comparison_metrics.csv` |
| Orientation validation | `outputs/tables/fracture_orientation_validation.csv` |
| Trace overlays (2 × 7 panels) | `outputs/figures/fracture_trace_overlay_*_7panel.{pdf,png}` |
| Classification maps (2 × 7 panels) | `outputs/figures/failure_mechanism_classification_*_7panel.{pdf,png}` |
| Strain-energy partitioning | `outputs/figures/strain_partitioning_two_rocks.{pdf,png}` |

## 10. Running the notebooks

Run them **from the repository root**; they use relative paths.

Run the two lithology notebooks first, in either order, then the general one:
it compares fields the lithology notebooks export and asserts up front that all
fourteen are present.

```bash
for nb in Tensile_augen_gneiss Tensile_psammitic_schist Tensile_general_plots; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=viscoquake \
        --ExecutePreprocessor.timeout=14400 "$nb.ipynb"
done
```

Expect roughly 40 minutes for a lithology notebook, most of it the
displacement-discontinuity solve and the crack-growth sections.

To check that every section still runs, without executing notebooks:

```bash
python scripts/smoke_sections.py            # every section, both lithologies
python scripts/smoke_sections.py --only stress_field strain_proxy
```

The cached field archives in `outputs/fields/fields_npz/` are included, so
neither the solve nor the notebooks are needed to rebuild the tables and
composite figures; `scripts/reproduce_all.py` does that in seconds.

## 11. Random seeds

Stochastic steps take an explicit seed. The orientation-fit bootstrap uses
`20260807`, recorded in `scripts/reproduce_all.py` and written to
`outputs/tables/environment.json` on every run. Heterogeneity fields use
`numpy.random.default_rng` with explicit integer seeds.

Two consecutive runs with the same seed produce bitwise-identical tables and PNG
figures. PDF outputs differ only in embedded creation timestamps.

---

## 12. Known limitations

- The framework is two-dimensional and pre-failure elastic. It does not model
  progressive damage, post-peak softening, crack-surface friction, or stress
  redistribution during growth.
- Weak-plane strengths are proxies scaled from the matrix values
  (`T_wp = 0.35 T₀`, `c_wp = 0.60 c`). No foliation-plane strength tests exist in
  this dataset, so the absolute balance between weak-plane and matrix utilities
  inherits that uncertainty.
- One specimen per fabric angle, so between-specimen orientation scatter is not
  measurable. Reported per-specimen uncertainty is digitization and fit scatter.
- The fracture-orientation comparison is a consistency check, not a validation
  of predicted orientation: observed and predicted fractures are alike
  loading-subparallel, and a predictor that simply assigns the loading direction
  matches the observations at least as well.
- `G/G_c` uses a nominal unit toughness and expresses relative along-path
  modulation of the crack-driving force, not an absolute margin against failure.
- The stepper advances the trajectory at every increment, so it represents
  neither arrest nor re-initiation.

## 13. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'tools'` | Install the package with `python -m pip install -e .`, or run from the repository root. |
| `FileNotFoundError: tensile_samples_data.csv` | Run with the repository root as the working directory. |
| `ModuleNotFoundError` for a bare module name such as `stress_helpers` | These modules moved into `tools/`. Import `tools.stress_helpers`. |
| `RuntimeError: no lithology bound` | A section's internals were called directly. Call its `main(rock)`, or wrap the call in `tools.analysis._context.using(rock)`. |
| An edit to a notebook cell disappeared | The notebooks are generated; see section 5. |
| A figure was written but is not where you expect | Every figure goes to `outputs/figures/`, whichever section produced it. |
| A long notebook run appears to hang with an empty log | Python buffers output. Run with `python -u`, or check that the process is alive before restarting it. |
| Figures open windows during a batch run | Set a headless backend: `MPLBACKEND=Agg`. |

## 14. Data included in this repository

All data needed to reproduce the reported results is tracked here:

| Data | Location | Files |
|---|---|---|
| Brazilian-test strengths, cohesion, friction angle | `tensile_samples_data.csv`, `selected_all_samples*.csv` | 3 |
| Digitized laboratory fracture traces | `crack_digitized_data/` | 14 |
| Model-derived (DDM) fracture traces | `outputs/fields/ddm_crack_sample_*.csv` | 14 |
| Cached orthotropic field archives | `outputs/fields/fields_npz/` | 14 (36 MB) |
| Step-wise crack diagnostics | `outputs/fields/smoke_*.csv` | 29 |

The cached field archives are included on purpose: they are what allow
`scripts/reproduce_all.py` to regenerate the validation tables and all four
composite figures **without** the multi-hour notebook run.

`selected_all_samples.xlsx` is retained as the spreadsheet alongside its CSV
export; the older `.xls` it superseded has been removed. No data file is
excluded.

## 15. Methods implemented, and the work they come from

The code is an implementation of published methods, not of new ones. Anyone
citing this software should cite the sources of the methods it uses as well.
They are listed here by what the code actually does with them; the manuscript's
bibliography carries the full list, including work that is discussed but not
implemented.

**Stress field**

| Used for | Reference |
| --- | --- |
| Orthotropic Airy-series stress function; the complex parameters and the series the solver fits | Lekhnitskii, S.G., Fern, P., Brandstatter, J.J., & Dill, E.H. (1964). *Theory of elasticity of an anisotropic elastic body.* American Institute of Physics. |
| Brazilian stress field and tensile strength for anisotropic discs | Claesson, J., & Bohloli, B. (2002). *Brazilian test: stress field and tensile strength of anisotropic rocks using an analytical solution.* International Journal of Rock Mechanics and Mining Sciences, 39(8), 991-1004. |
| Isotropic Brazilian solution, used as the limiting check on the solver | Hondros, G. (1959). *The evaluation of Poisson's ratio and the modulus of materials of a low tensile resistance by the Brazilian (indirect tensile) test with particular reference to concrete.* Aust. J. Appl. Sci., 243-264. |
| Saint-Venant decay, which justifies reporting on the 0.85R interior | Toupin, R.A. (1965). *Saint-Venant's principle.* Archive for Rational Mechanics and Analysis, 18(2), 83-96. |
| End-zone lengths in anisotropic solids, which is why interior convergence is measured rather than inferred | Horgan, C.O., & Simmonds, J.G. (1994). *Saint-Venant end effects in composite structures.* Composites Engineering, 4(3), 279-286. |

**Elastic constants and homogenization**

| Used for | Reference |
| --- | --- |
| Rotation of the plane-stress stiffness into the fabric frame (the `qbar_from_Es` transformation) | *Laminated composite plates.* (2000). Massachusetts Institute of Technology Cambridge. |
| Single-inclusion stress concentration, used to bound the neglected grain-scale fluctuation | Eshelby, J.D. (1957). *The determination of the elastic field of an ellipsoidal inclusion, and related problems.* Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 241(1226), 376-396. |
| Effective-medium stiffening bound for the two-phase aggregate | Mori, T., & Tanaka, K. (1973). *Average stress in matrix and average elastic energy of materials with misfitting inclusions.* Acta Metallurgica, 21(5), 571-574. |
| Modulus-anisotropy ratio as a descriptor for transversely isotropic geomaterials | Ip, S.C.Y., Choo, J., & Borja, R.I. (2021). *Impacts of saturation-dependent anisotropy on the shrinkage behavior of clay rocks.* Acta Geotechnica, 16(11), 3381-3400. |

**Fracture and crack path**

| Used for | Reference |
| --- | --- |
| Anisotropic elasticity formalism underlying the crack-tip fields | Stroh, A.N. (1958). *Dislocations and Cracks in Anisotropic Elasticity.* Philosophical Magazine, 3(30), 625-646. |
| Barnett-Lothe energy matrix **H**, which converts the stress-intensity pair into an energy release rate | Barnett, D.M., Lothe, J., Nishioka, K., & Asaro, R.J. (1973). *Elastic surface waves in anisotropic crystals: a simplified method for calculating Rayleigh velocities using dislocation theory.* Journal of Physics F: Metal Physics, 3(6), 1083. |
| Cracks in rectilinearly anisotropic bodies | Sih, G.C., Paris, P.C., & Irwin, G.R. (1965). *On cracks in rectilinearly anisotropic bodies.* International Journal of Fracture Mechanics, 1(3), 189-203. |
| Mixed-mode extension direction | Erdogan, F., & Sih, S.C. (1963). *On the crack extension in paltes under plane loading and transverse shear.* J Basic Eng ASME, 85. |
| Energy criterion the stepper applies | Griffith, A.A. (1921). *VI. The phenomena of rupture and flow in solids.* Philosophical Transactions of the Royal Society of London, Series A: Containing Papers of a Mathematical or Physical Character, 221(582-593), 163-198. |
| Mixed-mode cracking where a compliant layer is present | Hutchinson, J.W., & Suo, Z. (1991). *Mixed Mode Cracking in Layered Materials.* Advances in Applied Mechanics, 29(C), 63-191. |
| Fracture analysis of cracked anisotropic discs | Chen, C., Pan, E., & Amadei, B. (1998). *Fracture mechanics analysis of cracked discs of anisotropic rock using the boundary element method.* International Journal of Rock Mechanics and Mining Sciences, 35(2), 195-218. |

**Failure criteria and regimes**

| Used for | Reference |
| --- | --- |
| Shear failure on a weak plane, the form the weak-plane criterion takes | Jaeger, J.C. (1960). *Shear Failure of Anistropic Rocks.* Geological Magazine, 97(1), 65-72. |
| Naming convention for the three principal-stress permutations | *The dynamics of faulting and dyke formation with applications to Britain.* (1951). |

**Measurement standards and tools**

| Used for | Reference |
| --- | --- |
| Brazilian tensile test procedure the input data follow | ISRM (1977). *Suggested methods for determining tensile strength of rock materials.* International Journal of Rock Mechanics and Mining Sciences and Geomechanics Abstracts, 15, 99-103. |
| Uniaxial compressive strength and deformability | *Suggested methods for determining the uniaxial compressive strength and deformability of rock materials: Part 1. Suggested method for determining deformability of rock materials in uniaxial compression.* (1979). International Journal of Rock Mechanics and Mining Sciences & Geomechanics Abstracts, 16(2), 138-140. |
| Triaxial testing, source of the cohesion and friction angle | *Suggested methods for determining the strength of rock materials in triaxial compression.* (1978). International Journal of Rock Mechanics and Mining Sciences & Geomechanics Abstracts, 15(2), 47-51. |
| Digitizing fracture traces and foliation scanlines | Schneider, C.A., Rasband, W.S., & Eliceiri, K.W. (2012). *NIH Image to ImageJ: 25 years of image analysis.* Nature Methods 2012 9:7, 9(7), 671-675. |

BibTeX entries for all of these, and for this software, are in
`REFERENCES.bib` in the repository root.

## 16. License and citation

See `LICENSE`. Machine-readable citation metadata is in `CITATION.cff`, which
GitHub and Zenodo both read; "Cite this repository" on the GitHub sidebar
renders it as BibTeX or APA.

To cite the software, use the archived release:

> Acharya, D., & Elsworth, D. (2026). *Impact of Rock Anisotropy on Tensile
> Strength and Modes of Rupture* (Version v1.0.1) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.22644302

```bibtex
@misc{Acharya2026ImpactSoftware,
    title     = {{Impact of Rock Anisotropy on Tensile Strength and Modes of Rupture}},
    author    = {Acharya, Durga and Elsworth, Derek},
    year      = {2026},
    version   = {v1.0.1},
    howpublished = {[Software]},
    publisher = {Zenodo},
    doi       = {10.5281/zenodo.22644302}
}
```

One field remains deliberately absent, because an invented one is worse than a
missing one: `orcid:` under each author. Add it when you want the authors
disambiguated; it needs the real identifiers, not a placeholder.

The `preferred-citation` block carries the manuscript as `status: submitted`.
On acceptance, change that to the journal, volume, pages and year, and add the
paper DOI.
