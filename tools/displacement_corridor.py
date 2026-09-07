"""Orientation and width of the high-displacement corridor.

The displacement-field figure was read as showing a corridor that "rotates
progressively toward foliation" as the loading angle increases. That is a claim
about the axis of the high-magnitude region, so this module measures it: take
the points carrying the largest displacement magnitude, and fit the principal
axis of that set by total least squares.

What the measurement shows is that the corridor stays within a few degrees of
the loading axis at every orientation, and that its small departure peaks near
45 degrees and returns to vertical at 90 degrees -- the signature of a shear
induced tilt, not of rotation toward the fabric. If the corridor tracked
foliation it would lie horizontal in the 90 degree specimen.

The orientation dependence of these fields is carried by displacement
*magnitude* rather than by corridor direction, which is why the panels look
alike once each is normalised to its own maximum.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import lithology as lith
from . import output_dirs
from .data_io import load_specimen_table

CACHE_DIR = output_dirs.FIELD_DIR + "/_cache_direction_circles_uv_v2"

#: Percentile of displacement magnitude defining the corridor.
CORRIDOR_PERCENTILE = 85.0

#: Radius fraction kept, to stay clear of the platen contacts.
DISK_FRAC = 0.95


def cache_files(root=None):
    """Cached displacement solutions carrying a material fingerprint."""
    root = Path(root or lith.REPO_ROOT)
    return sorted(glob.glob(str(root / CACHE_DIR / "*_v3.npz")),
                  key=lambda p: int(re.search(r"idx(\d+)", p).group(1)))


def corridor_axis(u, v, q=CORRIDOR_PERCENTILE, frac=DISK_FRAC):
    """``(axis_deg, half_width, peak_mm)`` for one specimen.

    ``axis_deg`` is measured from the +x axis, so 90 degrees is the loading
    direction. The axis is the leading eigenvector of the covariance of the
    corridor points, which is the total-least-squares fit and therefore does not
    privilege either coordinate.
    """
    n = u.shape[0]
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, x)
    disk = np.hypot(X, Y) <= float(frac)

    g = np.where(disk, np.hypot(u, v), np.nan)
    thr = np.nanpercentile(g, float(q))
    m = disk & (g >= thr)

    xs, ys = X[m] - X[m].mean(), Y[m] - Y[m].mean()
    w, V = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    axis = V[:, int(np.argmax(w))]
    return (float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0),
            float(2.0 * np.sqrt(max(float(np.min(w)), 0.0))),
            float(np.nanmax(g) * 1e3))


def corridor_table(root=None) -> pd.DataFrame:
    """Corridor axis, width and peak displacement for every specimen.

    Raises if the displacement cache is absent. That cache is written by the
    two lithology notebooks and is deliberately not tracked, because it is
    keyed on a hash of the specimen geometry and material and would only pin a
    stale solve. Without the guard the loop below produces no rows and the
    empty frame fails much later, inside ``sort_values``, with ``KeyError:
    'rock'`` and nothing to indicate what is actually missing.
    """
    paths = cache_files(root)
    if not paths:
        raise FileNotFoundError(
            f"no cached displacement solutions in {CACHE_DIR}. They are written "
            "by the mid-plane displacement section of the two lithology "
            "notebooks, so run those before scripts/reproduce_all.py on a fresh "
            "clone: python scripts/execute_notebooks.py")
    df = load_specimen_table(root)
    rows = []
    for path in paths:
        sid = int(re.search(r"idx(\d+)", path).group(1))
        z = np.load(path)
        ax, width, peak = corridor_axis(z["u"], z["v"])
        rows.append(dict(
            sample_id=sid,
            rock=df.loc[sid, "Rock_type"],
            angle_deg=float(df.loc[sid, "Angle"]),
            corridor_axis_deg=ax,
            deviation_from_load_axis_deg=float(min(abs(ax - 90.0), 180.0 - abs(ax - 90.0))),
            corridor_half_width=width,
            peak_displacement_mm=peak,
        ))
    return pd.DataFrame(rows).sort_values(["rock", "angle_deg"]).reset_index(drop=True)
