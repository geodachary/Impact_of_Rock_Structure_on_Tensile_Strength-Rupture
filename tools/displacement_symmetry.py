"""Mirror-symmetry index for the Brazilian-test displacement field.

Describing a displacement field as "asymmetric" is only meaningful against a
stated criterion, so this module defines one and applies it identically to every
specimen.

The criterion
-------------
Brazilian loading along ``+y`` on a disc whose fabric is inclined at
:math:`\\alpha` is symmetric about the loading axis ``x = 0`` only in the
end-member cases. Under an ideal mirror about that axis,

* the horizontal component ``u`` is **antisymmetric**: ``u(-x, y) = -u(x, y)``;
* the vertical component ``v`` is **symmetric**:      ``v(-x, y) = +v(x, y)``.

The departure from each expectation is measured as a normalised residual over
the in-disc mask:

.. math::

    A_u = \\frac{\\lVert u(x,y) + u(-x,y) \\rVert}{2\\lVert u \\rVert}, \\qquad
    A_v = \\frac{\\lVert v(x,y) - v(-x,y) \\rVert}{2\\lVert v \\rVert}

Both are dimensionless, zero for a perfectly mirror-symmetric field, and bounded
by one. The factor of two makes the index the mean of the two mirror halves
rather than double-counting them. Values are computed only where a point and its
mirror image are both inside the disc.

Interpretation is deliberately conservative: an index of a few per cent is a
field a reader would reasonably call symmetric, and the label "asymmetric"
requires a departure large enough to see. The threshold used for that wording in
the manuscript is :data:`VISIBLE_ASYMMETRY`, stated rather than implied.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import lithology as lith
from . import output_dirs

#: Cache written by the displacement-profile cells of the lithology notebooks.
CACHE_DIR = output_dirs.FIELD_DIR + "/_cache_uv_profiles_v2"

#: Index above which a field is described as visibly asymmetric rather than
#: approximately symmetric. Chosen so that a departure has to be perceptible in
#: the plotted profile before the word is used.
VISIBLE_ASYMMETRY = 0.10


def _cache_files(root=None) -> dict:
    """``{sample_id: path}`` for every cached displacement solution."""
    root = Path(root or lith.REPO_ROOT)
    out = {}
    for p in sorted((root / CACHE_DIR).glob("idx*_*.npz")):
        m = re.match(r"idx(\d+)_", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _mirror_index(field, X, mask):
    """Normalised residual of ``field`` against its mirror about ``x = 0``.

    Returns ``(antisym_index, sym_index)``: the first treats the field as
    expected-antisymmetric, the second as expected-symmetric.
    """
    f = np.asarray(field, float)
    m = np.asarray(mask, bool)
    # columns are mirrored by reversing the x axis; X must be centred on 0
    f_m = f[:, ::-1]
    m_m = m[:, ::-1]
    both = m & m_m
    if not both.any():
        return np.nan, np.nan
    a, b = f[both], f_m[both]
    norm = np.linalg.norm(a)
    if norm <= 0:
        return np.nan, np.nan
    anti = np.linalg.norm(a + b) / (2.0 * norm)
    sym = np.linalg.norm(a - b) / (2.0 * norm)
    return float(anti), float(sym)


def specimen_symmetry(sample_id, root=None) -> dict:
    """Mirror indices for one specimen, or a blocked row if it is not cached."""
    files = _cache_files(root)
    lit = lith.lithology_of_sample(sample_id)
    base = dict(sample=int(sample_id), lithology=lit.display_name,
                angle_deg=lit.angle_for_sample(sample_id))
    if int(sample_id) not in files:
        return {**base, "status": "blocked",
                "reason": f"no cached displacement solution for idx{int(sample_id)}"}

    z = np.load(files[int(sample_id)], allow_pickle=True)
    X, mask = z["X"], z["mask"].astype(bool)
    u_anti, _ = _mirror_index(z["u"], X, mask)
    _, v_sym = _mirror_index(z["v"], X, mask)
    worst = float(np.nanmax([u_anti, v_sym]))
    return {**base,
            "u_antisymmetry_index": u_anti,
            "v_symmetry_index": v_sym,
            "max_index": worst,
            "visibly_asymmetric": bool(worst > VISIBLE_ASYMMETRY),
            "n_grid": int(mask.sum()),
            "status": "computed", "reason": ""}


def all_specimens(root=None) -> pd.DataFrame:
    """Mirror indices for all 14 specimens; uncached ones are reported blocked."""
    return pd.DataFrame([specimen_symmetry(s, root) for s in range(1, 15)])


def summary(df: pd.DataFrame) -> dict:
    """Whether the displacement fields support the word 'asymmetric'."""
    g = df[df.status == "computed"]
    return dict(
        n_computed=int(len(g)),
        n_blocked=int((df.status != "computed").sum()),
        median_max_index=float(g.max_index.median()),
        min_max_index=float(g.max_index.min()),
        max_max_index=float(g.max_index.max()),
        n_visibly_asymmetric=int(g.visibly_asymmetric.sum()),
        threshold=VISIBLE_ASYMMETRY,
        supports_asymmetric_description=bool(g.visibly_asymmetric.any()),
    )
