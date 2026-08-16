"""Smoothing, kernel density, bootstrap bands and curve collapse.

Re-exported from :mod:`tools.ddm._toolkit`.
"""

from ._toolkit import (  # noqa: F401
    compute_poly_ci,
    gaussian_kde_manual,
    gaussian_smooth_1d,
    mean_profile_midband,
    nan_box_blur,
    nearest_mean,
    robust_bandwidth,
)

__all__ = [
    "compute_poly_ci",
    "gaussian_kde_manual",
    "gaussian_smooth_1d",
    "mean_profile_midband",
    "nan_box_blur",
    "nearest_mean",
    "robust_bandwidth",
]
