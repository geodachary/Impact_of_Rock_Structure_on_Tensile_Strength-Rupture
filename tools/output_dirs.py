"""Where generated output goes, and making sure it exists.

Everything this code produces lands under one directory, ``outputs/``, split
by what the file *is* rather than by which section wrote it:

``outputs/figures``   every figure, in every format it is saved in
``outputs/tables``    every machine-readable result a reader would open
``outputs/fields``    solver state later steps read back: the cached field
                      archives, the per-specimen crack traces, the plot caches

The alternative, which this replaced, was a folder per producing section:
``physics_force``, ``foliation_deviation_output``, ``ddm_fields``,
``strain_partitioning``, ``stress_tensors``, ``Output_Figures``, ``figures``,
``results``. Eight top-level directories, and none of the names told you what
was inside -- ``stress_tensors`` held a single spreadsheet of strength metrics,
``Output_Figures`` held three CSVs, and figures were spread across five of
them. Sorting by kind means a reader looking for a figure has one place to look
and a new section adds no new directory.

The split also draws the line that matters for reuse: ``fields`` is the
expensive part. It is what lets the tables and figures be rebuilt without
re-running the multi-hour displacement-discontinuity solve.

Document output
---------------
The generators also write figures and tables into a document directory that is
not part of a code-only distribution. On a fresh checkout that directory is
absent, and every one of those scripts used to fail with ``FileNotFoundError``
before producing anything.

That dependency is deliberately one-directional. The code writes into the
document directory so that changing an analysis updates what the document
embeds, but nothing here reads from it, and no analysis result depends on it
existing beforehand. Creating it on demand is what keeps the two independent:
the code runs the same whether or not a document is present alongside it.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The one directory everything generated lives under.
OUT_ROOT = "outputs"

#: Figures, in every format they are saved in.
FIGURE_DIR = "outputs/figures"

#: Machine-readable results: the tables, the summaries, the run manifests.
TABLE_DIR = "outputs/tables"

#: Solver state that later steps read back, rather than results to be read.
FIELD_DIR = "outputs/fields"

#: The cached orthotropic field archives, one per specimen.
FIELDS_NPZ = "outputs/fields/fields_npz"

#: Figures and tables that a companion document would embed. Not part of a
#: code-only distribution; see the module docstring.
DOC_DIR = "manuscript"
DOC_TABLE_DIR = "manuscript/tables"

#: Every directory the pipeline writes into, for a one-call setup.
ALL = (FIGURE_DIR, TABLE_DIR, FIELD_DIR)


def ensure(*subdirs, root: Path | None = None) -> Path:
    """Create the output directories if absent; return the first one.

    Called at the top of each generator, so a script can be run from a fresh
    checkout that carries neither the output tree nor a document directory.
    """
    root = Path(root) if root is not None else REPO_ROOT
    made = [root / s for s in (subdirs or (DOC_DIR,))]
    for d in made:
        d.mkdir(parents=True, exist_ok=True)
    return made[0]


def figures(root: Path | None = None) -> Path:
    """The figure directory, created if absent."""
    return ensure(FIGURE_DIR, root=root)


def tables(root: Path | None = None) -> Path:
    """The results directory, created if absent."""
    return ensure(TABLE_DIR, root=root)


def fields(root: Path | None = None) -> Path:
    """The cached-solver-state directory, created if absent."""
    return ensure(FIELD_DIR, root=root)


def document_source(name: str = "manscript_revision_001.tex",
                    root: Path | None = None) -> Path | None:
    """The document source if it is present, otherwise ``None``.

    Steps that edit the document, rather than write output it embeds, have
    nothing to do without it. Returning ``None`` lets them say so and exit
    cleanly instead of raising, which is what makes them safe to leave in a
    pipeline that ships without the document.
    """
    root = Path(root) if root is not None else REPO_ROOT
    path = root / DOC_DIR / name
    return path if path.is_file() else None
