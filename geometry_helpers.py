# geomertry_helpers.py
# -*- coding: utf-8 -*-
"""
Geometry + angle + rotation helpers (self-contained).

Goal:
- avoid circular imports between crack_helpers.py and cracked_disk_ddm.py
- keep only pure geometry/rotation utilities here
"""

import numpy as np


# =============================================================================
# Angle helpers
# =============================================================================

def _wrap_pi(a):
    """
    Wrap angle(s) to (-pi, pi].

    Works for scalars or numpy arrays.
    """
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff_periodic(a, b):
    """
    Smallest signed difference (a - b) wrapped to (-pi, pi].

    Works for scalars or numpy arrays.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return _wrap_pi(a - b)


def reduce_angle_0_90(angle_deg: float) -> float:
    """
    Map any angle in degrees to [0, 90].
    Example: 100 -> 80, -20 -> 20, 170 -> 10
    """
    a = float(angle_deg) % 180.0
    if a > 90.0:
        a = 180.0 - a
    return abs(a)


def map_angle_to_alpha(angle_deg: float, angle_map: str = "direct") -> float:
    """
    Convert your CSV foliation angle (deg) to alpha (rad) used by material rotation.

    Your CSV meaning (as you stated):
      angle_deg = angle from foliation direction to the horizontal diameter (+X).

    For this convention, the correct default is:
      angle_map="direct"  -> alpha = theta

    Supported:
      - "direct"    : alpha =  theta
      - "neg"       : alpha = -theta
      - "pi2_minus" : alpha = pi/2 - theta   (use ONLY if your theta is measured from vertical)
      - "pi2_plus"  : alpha = pi/2 + theta   (use ONLY if your theta is measured from vertical)
    """
    # Treat foliation as a LINE: theta and theta+180 are identical
    theta_deg = float(angle_deg) % 180.0
    theta = np.deg2rad(theta_deg)

    if angle_map == "direct":
        alpha = theta
    elif angle_map == "neg":
        alpha = -theta
    elif angle_map == "pi2_minus":
        alpha = np.pi / 2.0 - theta
    elif angle_map == "pi2_plus":
        alpha = np.pi / 2.0 + theta
    else:
        raise ValueError(
            f"Unknown angle_map='{angle_map}'. Choose: direct, neg, pi2_minus, pi2_plus"
        )

    # wrap to [-pi, pi)
    return float(_wrap_pi(alpha))

# =============================================================================
# Disk grid helpers
# =============================================================================

def create_disk_grid(D: float, n: int = 101):
    """
    Create a square meshgrid (X,Y) covering the disk and mask M for points inside disk.
    Returns: X, Y, M
    """
    D = float(D)
    R = D / 2.0
    xs = np.linspace(-R, R, int(n))
    ys = np.linspace(-R, R, int(n))
    X, Y = np.meshgrid(xs, ys)
    M = (X * X + Y * Y) <= (R * R)
    return X, Y, M


def points_in_disk(D: float, n: int = 101):
    """
    Return (x_points, y_points, X, Y, M)
    where x_points, y_points are flattened arrays of points inside the disk.
    """
    X, Y, M = create_disk_grid(D, n=n)
    return X[M], Y[M], X, Y, M


# =============================================================================
# Segment tangent/normal
# =============================================================================

def unit_tangent_normal(p1, p2, eps: float = 1e-18):
    """
    Return (t_hat, n_hat) for segment p1->p2.

    t_hat: unit tangent from p1 to p2
    n_hat: left normal (-ty, tx)
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L < float(eps):
        return (1.0, 0.0), (0.0, 1.0)
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    return (tx, ty), (nx, ny)
