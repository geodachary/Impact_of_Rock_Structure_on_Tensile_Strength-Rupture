"""Operational definitions for energy corridors and isolated lobes.

The strain-energy panels were described as showing continuous corridors in the
schist against localized hotspots in the gneiss, which is a claim about the
*shape and connectivity* of the elevated-energy region and cannot be settled by
looking at two colour maps side by side. This module states what the two words
mean as measurements and returns them.

Take the upper quartile of strain-energy density inside the disc interior and
label its connected components with eight-fold connectivity. Two numbers then
separate the cases:

``n_components``
    how many separate pieces the elevated region falls into. Diametral loading
    produces two energy lobes, one at each platen, so two components is the
    unconnected baseline rather than evidence of fragmentation.
``elongation``
    the square root of the ratio of the second moments of the largest component,
    which is 1 for an equant patch and grows as it stretches.

A **corridor** is an elevated region that has merged into a single connected
component and is strongly elongated. **Isolated lobes** are the same region left
as two separate, comparatively equant pieces. Both are properties of the field
rather than impressions of it, and both are reported per specimen so the reader
can see at which orientations the distinction actually holds.

The threshold matters and is stated deliberately. At the top decile only the two
platen lobes survive in either lithology and nothing distinguishes them; the
corridor structure appears at the quartile, where the schist lobes join and the
gneiss lobes do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import fabric_tractions as ft

#: Percentile defining the elevated-energy set.
ENERGY_PERCENTILE = 75.0

#: Elongation above which the largest component counts as corridor-like.
CORRIDOR_ELONGATION = 2.0


def elevated_set(f, percentile=ENERGY_PERCENTILE, frac=ft.CORE_FRAC):
    """Boolean mask of the elevated strain-energy region in the disc interior."""
    mask = ft.core_mask(f, frac)
    U = np.where(mask, f["U_MPa"], np.nan)
    return np.nan_to_num(U, nan=-np.inf) >= np.nanpercentile(U, float(percentile))


def component_shape(f, percentile=ENERGY_PERCENTILE, frac=ft.CORE_FRAC) -> dict:
    """Connectivity and elongation of the elevated-energy region."""
    hot = elevated_set(f, percentile, frac)
    lab, n = ndimage.label(hot, structure=np.ones((3, 3)))
    if n == 0:
        return dict(n_components=0, largest_fraction=np.nan, elongation=np.nan)

    sizes = ndimage.sum(hot, lab, range(1, n + 1))
    ys, xs = np.nonzero(lab == int(np.argmax(sizes)) + 1)
    px, py = f["X"][ys, xs], f["Y"][ys, xs]
    cov = np.cov(np.vstack([px - px.mean(), py - py.mean()]))
    w = np.linalg.eigvalsh(cov)
    return dict(n_components=int(n),
                largest_fraction=float(sizes.max() / sizes.sum()),
                elongation=float(np.sqrt(max(w) / max(min(w), 1e-30))))


def is_corridor(shape) -> bool:
    """A single connected component that is strongly elongated."""
    return bool(shape["n_components"] == 1
                and shape["elongation"] >= CORRIDOR_ELONGATION)


def localization_table(root=None, percentile=ENERGY_PERCENTILE) -> pd.DataFrame:
    """Corridor diagnostics for every specimen."""
    rows = []
    for p in ft.field_files(root):
        f = ft.load_field(p)
        sh = component_shape(f, percentile)
        rows.append(dict(sample_id=f["sample_id"], rock=f["rock"],
                         angle_deg=f["angle_deg"], **sh, corridor=is_corridor(sh)))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)


def summary(root=None, percentile=ENERGY_PERCENTILE) -> pd.DataFrame:
    """Per-lithology counts of corridor-forming orientations."""
    t = localization_table(root, percentile)
    return (t.groupby("rock")
             .agg(n_corridor=("corridor", "sum"), n_angles=("corridor", "size"),
                  mean_elongation=("elongation", "mean"),
                  max_elongation=("elongation", "max"))
             .reset_index())
