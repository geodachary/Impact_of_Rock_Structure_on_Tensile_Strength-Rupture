import numpy as np

# =============================================================================
# Rotation helpers (global <-> material)
# Convention matches what you used in your snippets:
#   xm =  ca*xg + sa*yg
#   ym = -sa*xg + ca*yg
# =============================================================================

def rot_to_material(xg, yg, alpha: float):
    """Rotate global -> material by +alpha."""
    alpha = float(alpha)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    xm = ca * xg + sa * yg
    ym = -sa * xg + ca * yg
    return xm, ym


def rot_to_global(xm, ym, alpha: float):
    """Rotate material -> global (inverse rotation)."""
    alpha = float(alpha)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    xm = np.asarray(xm, dtype=float)
    ym = np.asarray(ym, dtype=float)
    xg = ca * xm - sa * ym
    yg = sa * xm + ca * ym
    return xg, yg


def vec_rot_to_material(vx, vy, alpha):
    """
    Rotate vector components global -> material by +alpha.
    Works for scalars OR numpy arrays.
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    return (ca * vx + sa * vy, -sa * vx + ca * vy)

def vec_rot_to_global(vx, vy, alpha):
    """
    Rotate vector components material -> global by +alpha (inverse of above).
    Works for scalars OR numpy arrays.
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    return (ca * vx - sa * vy, sa * vx + ca * vy)
