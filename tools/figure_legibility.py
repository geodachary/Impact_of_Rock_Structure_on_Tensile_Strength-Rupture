"""Measure the on-page text size of figures as the reader actually sees them.

A figure can carry perfectly reasonable font sizes internally and still be
illegible in print. What matters is the size *after* LaTeX scales the graphic to
its included width:

.. math::

    \\text{on-page pt} = \\text{internal pt} \\times
        \\frac{\\text{included width}}{\\text{natural width}}

A panel drawn 711 pt wide with 12 pt labels, included at ``0.48\\textwidth``
(≈225 pt of a 469 pt text block), reaches the page at 3.8 pt — well below
readable. That is a layout problem, not a styling problem, and it is invisible
unless the scaling is accounted for.

:data:`MIN_ON_PAGE_PT` is the floor this project holds figures to. Eight points
is a common journal minimum for the smallest text in a figure.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd

#: Width of the text block in points for the manuscript's page geometry.
TEXTWIDTH_PT = 469.0

#: Smallest acceptable on-page text size, in points.
MIN_ON_PAGE_PT = 8.0


def natural_text_sizes(pdf_path):
    """Internal font sizes of every non-empty text span on page 1."""
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        sizes = [s["size"]
                 for b in page.get_text("dict")["blocks"] if b.get("lines")
                 for line in b["lines"] for s in line["spans"]
                 if s["text"].strip()]
        # Sub- and superscripts render at roughly 70% of the surrounding text
        # and would otherwise set the floor for the whole figure. Judge
        # legibility on the body text: drop anything below 80% of the median.
        if sizes:
            import statistics
            cut = 0.8 * statistics.median(sizes)
            body = [z for z in sizes if z >= cut]
            sizes = body or sizes
        return sizes, float(page.rect.width)
    finally:
        doc.close()


def measure(pdf_path, included_frac, textwidth_pt=TEXTWIDTH_PT) -> dict:
    """On-page text sizes for one figure included at ``included_frac`` of the text block."""
    p = Path(pdf_path)
    if not p.exists():
        return dict(figure_file=p.name, status="blocked", reason="file not found")
    sizes, natural_w = natural_text_sizes(p)
    if not sizes:
        return dict(figure_file=p.name, status="blocked", reason="no text spans")
    scale = (float(included_frac) * float(textwidth_pt)) / natural_w
    on_page = [s * scale for s in sizes]
    return dict(
        figure_file=p.name,
        natural_width_pt=round(natural_w, 1),
        included_frac=float(included_frac),
        scale=round(scale, 4),
        min_internal_pt=round(min(sizes), 2),
        median_internal_pt=round(statistics.median(sizes), 2),
        min_on_page_pt=round(min(on_page), 2),
        median_on_page_pt=round(statistics.median(on_page), 2),
        legible=bool(min(on_page) >= MIN_ON_PAGE_PT),
        required_frac=round(
            min(MIN_ON_PAGE_PT / (min(sizes) / natural_w) / textwidth_pt, 99.0), 3),
        status="computed", reason="")


def audit(spec, root=None) -> pd.DataFrame:
    """Measure a mapping of ``{label: (path, included_frac)}``."""
    root = Path(root or ".")
    rows = []
    for label, (path, frac) in spec.items():
        r = measure(root / path, frac)
        rows.append({"figure": label, **r})
    return pd.DataFrame(rows)
