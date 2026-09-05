"""Shared analysis package for the foliated-rock Brazilian-test study.

One authoritative implementation of every function used by both lithology
notebooks (``Tensile_augen_gneiss.ipynb`` and
``Tensile_psammitic_schist.ipynb``), so the gneiss and schist workflows cannot
drift apart.

Modules
-------
conventions          coordinate, angle and stress-sign conventions
lithology            per-lithology configuration and the specimen pairing table
traces               digitized and DDM fracture-trace loading, fitting, comparison
failure_classification  WT / WS / MT / MS utilities and the secondary mixed flag
plotting             shared publication style and the seven-panel figure builders
export               machine-readable export of manuscript values

The conventions are stated once, in :mod:`tools.conventions`, and every other
module defers to them.
"""

__version__ = "1.0.0"

# Reproducible figures. The PDF backend stamps a CreationDate, so re-running a
# generator produced a byte-different file from an identical plot: the sync
# stage then recopied every figure, and a checksum could not distinguish a
# refreshed figure from a changed one. matplotlib honours SOURCE_DATE_EPOCH for
# that stamp, so fixing it here makes figure output depend on the data alone.
# setdefault, so a caller building a dated release can still override it.
#
# Set at package import because nine modules under tools/analysis/ save figures
# without going through plot_style.apply_plot_style.
import os as _os

_os.environ.setdefault("SOURCE_DATE_EPOCH", "1735689600")   # 2025-01-01 UTC

from . import conventions, lithology, traces, failure_classification, plotting, export  # noqa: F401

__all__ = [
    "conventions",
    "lithology",
    "traces",
    "failure_classification",
    "plotting",
    "export",
]
