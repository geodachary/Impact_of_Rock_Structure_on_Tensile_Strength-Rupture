"""Specimen table, triangular mesh and the local Mohr-Coulomb failure test used by the early diagnostic cells.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 3 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import os
import copy
import sys
import warnings
import argparse
import inspect
import json
import time
import math
import scipy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.optimize import curve_fit
from collections import deque
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, t
from scipy import stats
# <<PRELUDE_IMPORTS>>


# --- implementation ---------------------------------------------------


def load_samples(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = DATA_PATH
    return pd.read_csv(path, delimiter=",", header=0, index_col=0)


def wrap_pi(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def angle_diff_periodic(a, b):
    return wrap_pi(a - b)


def anisotropic_tensile_strength(base_tensile_strength_mpa, anisotropic_angle, orientation, spacing, radius):
    # Delta measured from the foliation normal; see tools/analysis/
    # mohr_coulomb_local.py for the same convention.
    normal = float(anisotropic_angle) + np.pi / 2.0
    dphi = angle_diff_periodic(normal, orientation)
    reduction = np.cos(dphi) ** 2
    distance_to_plane = np.abs(radius * np.cos(dphi)) % spacing
    spacing_effect = np.sin(np.pi * distance_to_plane / spacing) ** 2
    return base_tensile_strength_mpa * reduction * spacing_effect


def create_triangular_mesh(diameter, num_points=40000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    radius = diameter / 2.0
    angles = rng.uniform(0.0, 2.0 * np.pi, num_points)
    radii = np.sqrt(rng.uniform(0.0, 1.0, num_points)) * radius
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    triangulation = tri.Triangulation(x, y)
    radial_distance = np.sqrt(x ** 2 + y ** 2)
    return triangulation, radial_distance, x, y


def evaluate_failure_mode_mohr_coulomb_local(sigma_1, sigma_3,
                                             tensile_strength_arr,
                                             cohesion_arr, friction_angle_arr,
                                             k_max=0.70, cohesion_floor=0.05,
                                             report_counts=False):
    sigma_1 = np.asarray(sigma_1)
    sigma_3 = np.asarray(sigma_3)
    tensile_strength_arr = np.asarray(tensile_strength_arr)
    cohesion_arr = np.asarray(cohesion_arr)
    friction_angle_arr = np.asarray(friction_angle_arr)

    sigma_m = 0.5 * (sigma_1 + sigma_3)
    tau_max = 0.5 * (sigma_1 - sigma_3)

    sigma_ref = float(np.mean(tensile_strength_arr))
    k_field = k_max * (1.0 - np.exp(-np.abs(sigma_m) / (sigma_ref + 1e-9)))
    effective_cohesion = np.maximum(cohesion_arr * (1.0 - k_field), cohesion_floor)

    sigma_m_comp = np.maximum(0.0, -sigma_m)
    tension_crit = sigma_1 >= tensile_strength_arr
    shear_strength = effective_cohesion + sigma_m_comp * np.tan(friction_angle_arr)
    shear_crit = tau_max >= shear_strength

    mode = np.full(sigma_1.shape, 'no_failure', dtype=object)
    mode[tension_crit & shear_crit] = 'mixed'
    mode[tension_crit & ~shear_crit] = 'tensile'
    mode[shear_crit & ~tension_crit] = 'shear'

    if report_counts:
        print(f"  Tensile pts: {np.sum(tension_crit)}")
        print(f"  Shear pts:   {np.sum(shear_crit)}")
        print(f"  Mixed pts:   {np.sum(tension_crit & shear_crit)}")
        print(f"  No-failure pts: {np.sum(mode == 'no_failure')}")

    return mode
