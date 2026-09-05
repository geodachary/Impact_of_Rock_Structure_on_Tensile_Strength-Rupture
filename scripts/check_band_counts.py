#!/usr/bin/env python3
"""Count weak-plane bands in the exported fields and compare against 2R/s.

The weak-plane spacing reaches the four-mechanism classifier only through the
``wp_weight`` and ``Rt_eff`` fields stored in the archives, never as a visible
argument, so a wrong spacing is silent everywhere except in the geometry of
those fields. Counting the bands is the direct check: a disc of radius R cut by
planes spaced s apart shows about 2R/s maxima along a diameter.

The spacing is carried by ``wp_weight``, not by ``Rt_eff``. This script read
``Rt_eff`` and reported a MISMATCH on both rocks, because at 0 degrees the
foliation is clamped shut and the tensile utility is the smooth matrix term
sigma_1 / T_m. That is the correct field, not a defect, and
``test_spacing_modulates_resistance`` had already been corrected to read
``wp_weight`` while this script and the profile figure were left behind.

This reports rather than asserts, and it prints both rocks side by side, so the
question "are these two actually different?" is answered by looking. The same
comparison is pinned as a test in
``tests/regression/test_spacing_modulates_resistance.py``.

    python scripts/check_band_counts.py
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.signal import find_peaks

from tools import fabric_tractions as ft
from tools import lithology as lith


def profile(f, key):
    """Values of ``key`` down the vertical diameter, inside the disc."""
    X, M = f["X"], f["M"].astype(bool)
    i = int(np.argmin(np.abs(X[0, :])))
    col = M[:, i]
    return np.asarray(f[key][:, i][col], float)


def n_maxima(v):
    span = float(np.nanmax(v) - np.nanmin(v))
    peaks, _ = find_peaks(v, prominence=0.02 * (span + 1e-12))
    return len(peaks)


def main():
    files = ft.field_files()
    if not files:
        print("no exported fields; run the lithology notebooks first")
        return 1

    by_rock = {}
    for path in files:
        f = ft.load_field(path)
        if abs(f["angle_deg"]) < 1e-9:            # the 0 degree specimen
            by_rock[f["rock"]] = f

    spacing = {l.display_name: l.spacing_m for l in lith.LITHOLOGIES.values()}

    print(f"{'lithology':20s} {'s (mm)':>7s} {'R (mm)':>7s} "
          f"{'2R/s':>7s} {'w bands':>9s} {'s1 bands':>9s} {'Rt bands':>9s}")
    rows = []
    for rock, f in sorted(by_rock.items()):
        s = spacing.get(rock)
        if s is None:
            print(f"  {rock}: not a configured lithology")
            continue
        R = float(f["R_m"])
        expected = 2.0 * R / s
        n_rt = n_maxima(profile(f, "wp_weight"))
        n_s1 = n_maxima(profile(f, "s1"))
        n_ut = n_maxima(profile(f, "Rt_eff"))
        print(f"{rock:20s} {s * 1e3:7.1f} {R * 1e3:7.2f} "
              f"{expected:7.1f} {n_rt:9d} {n_s1:9d} {n_ut:9d}")
        rows.append((rock, expected, n_rt, n_s1, n_ut))

    print()
    ok = True
    for rock, expected, n_rt, n_s1, n_ut in rows:
        # The proximity weight must carry the spacing ...
        if abs(n_rt - expected) > max(2.0, 0.1 * expected):
            print(f"  MISMATCH {rock}: {n_rt} bands against {expected:.0f} "
                  f"expected from 2R/s")
            ok = False
        # ... and the elastic field must not. The Lekhnitskii solution is
        # homogeneous orthotropic; periodic content in sigma_1 would mean the
        # strength modulation had leaked into the stress field.
        if n_s1 > 2:
            print(f"  LEAK {rock}: sigma_1 shows {n_s1} maxima; the elastic "
                  f"field should carry no spacing signature")
            ok = False
        # ... and at 0 degrees the tensile utility must not: the planes are
        # clamped, so R_t is the smooth matrix term. Periodic content here
        # would mean weak-plane terms were admitted against compression.
        if n_ut > 2:
            print(f"  CLAMP {rock}: R_t shows {n_ut} maxima at 0 degrees; the "
                  f"foliation is shut there and R_t should be smooth")
            ok = False

    if ok and len(rows) == 2:
        a, b = rows
        print(f"  both rocks match 2R/s, and they differ from each other "
              f"({a[2]} against {b[2]} bands) as their spacings require")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
