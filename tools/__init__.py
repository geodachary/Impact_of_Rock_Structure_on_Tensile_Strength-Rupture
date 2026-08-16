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

from . import conventions, lithology, traces, failure_classification, plotting, export  # noqa: F401

__all__ = [
    "conventions",
    "lithology",
    "traces",
    "failure_classification",
    "plotting",
    "export",
]
