import numpy as np

# ==========================================================
# Airy evaluation: normalized z/R basis
# ==========================================================
def eval_boundary_tractions(theta_m, R, p1, p2, a1, a2):
    x = R*np.cos(theta_m)
    y = R*np.sin(theta_m)
    z1 = x + p1*y
    z2 = x + p2*y
    zh1 = z1 / R
    zh2 = z2 / R
    f1pp = np.zeros_like(z1, dtype=complex)
    f2pp = np.zeros_like(z2, dtype=complex)
    Mdeg = len(a1)-1
    invR2 = 1.0/(R*R)
    for m in range(2, Mdeg+1):
        f1pp += a1[m] * m*(m-1) * (zh1**(m-2)) * invR2
        f2pp += a2[m] * m*(m-1) * (zh2**(m-2)) * invR2
    sxx = np.real(p1**2*f1pp + p2**2*f2pp)
    syy = np.real(f1pp + f2pp)
    txy = -np.real(p1*f1pp + p2*f2pp)
    nx = np.cos(theta_m); ny = np.sin(theta_m)
    tx = sxx*nx + txy*ny
    ty = txy*nx + syy*ny
    tr = tx*nx + ty*ny
    tt = -tx*ny + ty*nx
    return tr, tt