# cracked_disk_ddm.py
# -*- coding: utf-8 -*-
"""
Anisotropic (orthotropic) DDM/BEM correction solver for cracked Brazilian disk.

Computes displacement discontinuity densities to cancel background tractions
on crack faces and enforce traction-free outer boundary.

Main entry points:
- solve_cracked_disk_correction_ddm(...)     → correction dict
- KI_KII_from_COD_near_tip(...)             → SIF from COD extrapolation
- sif_two_tips_from_crack(...)              → (KI0,KII0,KI1,KII1) convenience wrapper

Also exports:
- _try_call_sif_two_tips(...)               → robust call adapter for varying APIs

IMPORTANT:
This module MUST NOT import crack_helpers.py (to avoid circular imports).
"""

from __future__ import annotations

import numpy as np
import inspect
from geometry_helpers import _wrap_pi, angle_diff_periodic, unit_tangent_normal
from rotation_helpers import rot_to_material, rot_to_global, vec_rot_to_material, vec_rot_to_global


# =============================================================================
# Orthotropic plane-stress stiffness & compliance
# =============================================================================

def orthotropic_Q_plane_stress(E1, E2, nu12, G12):
    """Plane-stress reduced stiffness matrix Q in material coordinates."""
    nu21 = nu12 * E2 / E1
    den = 1.0 - nu12 * nu21
    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12
    return np.array([
        [Q11, Q12, 0.0],
        [Q12, Q22, 0.0],
        [0.0,  0.0, Q66]
    ], dtype=float)

def rotate_Q(Q, theta):
    """Rotate plane-stress stiffness matrix to new coordinate system (Voigt)."""
    m, n = np.cos(theta), np.sin(theta)
    m2, n2, mn = m*m, n*n, m*n

    T_sigma = np.array([
        [m2, n2,  2*mn],
        [n2, m2, -2*mn],
        [-mn, mn, m2-n2]
    ], dtype=float)

    T_eps = np.array([
        [m2, n2,  mn],
        [n2, m2, -mn],
        [-2*mn, 2*mn, m2-n2]
    ], dtype=float)

    # Qbar = T_sigma^{-1} Q (T_eps^{-T})
    Qbar = np.linalg.solve(T_sigma, Q @ np.linalg.inv(T_eps).T)
    return Qbar

def compliance_from_Q(Q):
    return np.linalg.inv(Q)

def reduced_compliance_components(Sbar):
    """S11, S22, S12, S16, S26, S66 from 3×3 barred compliance matrix."""
    return (
        Sbar[0,0], Sbar[1,1], Sbar[0,1],
        Sbar[0,2], Sbar[1,2], Sbar[2,2]
    )


# =============================================================================
# Complex characteristic roots pα (Im>0)
# =============================================================================

def p_roots_quartic(S11, S22, S12, S16, S26, S66):
    """Solve characteristic equation for orthotropic/monoclinic in-plane problem."""
    coef = [S11, -2*S16, 2*S12 + S66, -2*S26, S22]
    roots = np.roots(coef).astype(complex)

    pos_im = [r for r in roots if np.imag(r) > 1e-12]
    if len(pos_im) >= 2:
        pos_im.sort(key=lambda z: -np.imag(z))
        return pos_im[0], pos_im[1]

    roots = sorted(roots, key=lambda z: -np.imag(z))
    return roots[0], roots[1]


# =============================================================================
# Stroh formalism — Lα and Aα (2×2 complex matrices)
# =============================================================================

def stroh_LA_inplane(S11, S22, S12, S16, S26, S66, p1, p2):
    """Compute 2×2 complex L and A matrices (simple normalization)."""
    ps = [p1, p2]
    L = np.zeros((2, 2), dtype=complex)
    A = np.zeros((2, 2), dtype=complex)

    for a, p in enumerate(ps):
        # choose L such that L2=1
        L[:, a] = np.array([-p, 1.0 + 0j])

        L1, L2 = L[0,a], L[1,a]
        A[0,a] = ((S16 - S11*p)/p) * L1 + S12 * L2
        A[1,a] = ((S26 - S12*p)/p) * L1 + (S22/p) * L2

        dot = np.dot(L[:,a], A[:,a])
        if abs(dot) > 1e-30:
            A[:,a] /= (2.0 * dot)

    return L, A


# =============================================================================
# Constant-strength DDM element
# =============================================================================

class DDMElement:
    __slots__ = ('x1','y1','x2','y2','mid','n','t','kind','xi1','xi2')

    def __init__(self, x1, y1, x2, y2, kind, n_vec, t_vec, p1, p2):
        self.x1, self.y1 = float(x1), float(y1)
        self.x2, self.y2 = float(x2), float(y2)
        self.mid = (0.5*(x1+x2), 0.5*(y1+y2))
        self.n   = (float(n_vec[0]), float(n_vec[1]))
        self.t   = (float(t_vec[0]), float(t_vec[1]))
        self.kind = str(kind)

        self.xi1 = np.array([x1 + p1*y1, x1 + p2*y1], dtype=complex)
        self.xi2 = np.array([x2 + p1*y2, x2 + p2*y2], dtype=complex)


def build_outer_circle_elements(R, N, p1, p2, colloc_offset):
    th = np.linspace(-np.pi, np.pi, int(N)+1, endpoint=True)
    x, y = R*np.cos(th), R*np.sin(th)

    elems = []
    for i in range(int(N)):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        t_vec, n_vec = unit_tangent_normal((x1,y1), (x2,y2))

        # outward normal check at midpoint
        mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
        if n_vec[0]*mx + n_vec[1]*my < 0:
            n_vec = (-n_vec[0], -n_vec[1])

        elems.append(DDMElement(x1,y1,x2,y2,"outer", n_vec, t_vec, p1, p2))

    colloc = []
    normals = []
    tangents = []
    kinds = []
    for e in elems:
        cx = e.mid[0] - colloc_offset*e.n[0]  # inside
        cy = e.mid[1] - colloc_offset*e.n[1]
        colloc.append([cx, cy])
        normals.append(e.n)
        tangents.append(e.t)
        kinds.append(e.kind)

    return elems, np.array(colloc), np.array(normals), np.array(tangents), np.array(kinds, dtype=object)


def build_crack_polyline_elements(xc, yc, p1, p2, colloc_offset):
    xc = np.asarray(xc, float)
    yc = np.asarray(yc, float)

    elems = []
    for i in range(len(xc)-1):
        x1,y1 = xc[i], yc[i]
        x2,y2 = xc[i+1], yc[i+1]
        t_vec, n_vec = unit_tangent_normal((x1,y1), (x2,y2))
        elems.append(DDMElement(x1,y1,x2,y2,"crack", n_vec, t_vec, p1, p2))

    colloc = []
    normals = []
    tangents = []
    kinds = []
    for e in elems:
        # pick one side for collocation; COD sampling later uses both sides explicitly
        cx = e.mid[0] + colloc_offset*e.n[0]
        cy = e.mid[1] + colloc_offset*e.n[1]
        colloc.append([cx, cy])
        normals.append(e.n)
        tangents.append(e.t)
        kinds.append(e.kind)

    return elems, np.array(colloc), np.array(normals), np.array(tangents), np.array(kinds, dtype=object)


# =============================================================================
# Kernel: traction at field point due to one constant DDM element
# =============================================================================

def traction_from_element(zx, zy, nx, ny, elem, p1, p2, L, u1, u2):
    ps = (p1, p2)
    u = np.array([u1, u2], dtype=complex)

    sig1 = np.zeros(2, dtype=float)  # σ11, σ12 (Voigt-like)
    sig2 = np.zeros(2, dtype=float)  # σ21, σ22

    for a, p in enumerate(ps):
        z = zx + p*zy
        I = (1.0/(z - elem.xi2[a]) - 1.0/(z - elem.xi1[a])) / (2j*np.pi)

        dot = L[0,a]*u[0] + L[1,a]*u[1]
        coeff = dot * I

        sig2[0] += 2*np.real(L[0,a]*coeff)
        sig2[1] += 2*np.real(L[1,a]*coeff)
        sig1[0] += -2*np.real(p*L[0,a]*coeff)
        sig1[1] += -2*np.real(p*L[1,a]*coeff)

    t1 = sig1[0]*nx + sig1[1]*ny
    t2 = sig2[0]*nx + sig2[1]*ny
    return np.array([t1, t2], dtype=float)


def total_traction_at_point_material(
    xm: float,
    ym: float,
    n_m,
    correction: dict,
    R: float,
    airy_fit: dict,
    eval_stress_field_material,
) -> np.ndarray:
    """
    Total traction vector (tx,ty) at a point in MATERIAL coordinates on a plane
    with outward unit normal n_m (also in material coords).

    Total traction = background (Airy) traction + DDM correction traction.

    Parameters
    ----------
    xm, ym : float
        Field point in MATERIAL coordinates.
    n_m : array-like (2,)
        Unit normal in MATERIAL coordinates.
    correction : dict
        Output from solve_cracked_disk_correction_ddm().
        Must contain: p1,p2,L,A,elems,U
    R : float
        Disk radius (m).
    airy_fit : dict
        Airy fit dict containing p1,p2,a1,a2.
    eval_stress_field_material : callable
        Your eval_stress_field_material(xm,ym,R,p1,p2,a1,a2) function.

    Returns
    -------
    t_total : np.ndarray shape (2,)
        Traction vector in MATERIAL coords: [tx, ty]
    """
    xm = float(xm)
    ym = float(ym)
    n_m = np.asarray(n_m, dtype=float).reshape(2,)
    nx, ny = float(n_m[0]), float(n_m[1])

    # --- background traction from Airy (material coords) ---
    t_bg = traction_from_background_airy(
        xm, ym, (nx, ny), float(R), airy_fit, eval_stress_field_material
    )

    # --- correction traction from all DDM elements ---
    elems = correction["elems"]
    U = correction["U"]
    p1m = correction["p1"]
    p2m = correction["p2"]
    Lm = correction["L"]

    t_corr = np.zeros(2, dtype=float)

    # Sum traction contribution from each element using its actual DD vector u_j
    for j, e in enumerate(elems):
        uj = np.array([U[2*j], U[2*j + 1]], dtype=complex)
        t_corr += traction_from_element_u(xm, ym, nx, ny, e, p1m, p2m, Lm, uj)

    return t_bg + t_corr



def assemble_traction_matrix(colloc, normals, elems, p1, p2, L):
    nc = colloc.shape[0]
    ne = len(elems)
    A = np.zeros((2*nc, 2*ne), dtype=float)

    for i in range(nc):
        zx, zy = colloc[i]
        nx, ny = normals[i]

        for j, e in enumerate(elems):
            off = 2*j
            # basis u1
            t = traction_from_element(zx, zy, nx, ny, e, p1, p2, L, 1.0, 0.0)
            A[2*i:2*i+2, off+0] = t
            # basis u2
            t = traction_from_element(zx, zy, nx, ny, e, p1, p2, L, 0.0, 1.0)
            A[2*i:2*i+2, off+1] = t

    return A


# =============================================================================
# Displacement correction field at arbitrary point
# =============================================================================

def displacement_at_point(zx, zy, elems, U, p1, p2, L, A_mat):
    ps = (p1, p2)
    uout = np.zeros(2, dtype=float)

    for j, e in enumerate(elems):
        u = np.array(U[2*j:2*j+2], dtype=complex)

        for a, p in enumerate(ps):
            z = zx + p*zy
            log_diff = np.log(z - e.xi2[a]) - np.log(z - e.xi1[a])
            dot = L[0,a]*u[0] + L[1,a]*u[1]
            coeff = dot * log_diff / (2j*np.pi)

            uout += 2*np.real(A_mat[:,a] * coeff)

    return uout


# =============================================================================
# Background traction from Airy solution (material coords)
# =============================================================================

def traction_from_background_airy(xm, ym, n_vec, R, airy_fit, eval_stress_field_material):
    sxx, syy, txy = eval_stress_field_material(
        np.array([xm], float), np.array([ym], float), float(R),
        airy_fit["p1"], airy_fit["p2"], airy_fit["a1"], airy_fit["a2"]
    )
    sxx = float(sxx[0]); syy = float(syy[0]); txy = float(txy[0])

    nx, ny = float(n_vec[0]), float(n_vec[1])
    tx = sxx*nx + txy*ny
    ty = txy*nx + syy*ny
    return np.array([tx, ty], dtype=float)


# =============================================================================
# Main solver: traction-free correction problem
# =============================================================================

def solve_cracked_disk_correction_ddm(
    R: float,
    alpha_const: float,
    airy_fit: dict,
    xs_crack_global: np.ndarray,
    ys_crack_global: np.ndarray,
    E1: float, E2: float, nu12: float, G12: float,
    eval_stress_field_material,
    N_outer: int = 220,
    colloc_eps_frac: float = 2e-5,
    reg_lam: float = 1e-10,
) -> dict:
    """
    Solve DDM correction:
      - traction = 0 on outer boundary
      - traction = -background traction on crack faces
    """
    Qm = orthotropic_Q_plane_stress(E1, E2, nu12, G12)
    Sm = compliance_from_Q(Qm)
    S11m, S22m, S12m, S16m, S26m, S66m = reduced_compliance_components(Sm)

    p1m, p2m = p_roots_quartic(S11m, S22m, S12m, S16m, S26m, S66m)
    Lm, Am = stroh_LA_inplane(S11m, S22m, S12m, S16m, S26m, S66m, p1m, p2m)

    offset = float(colloc_eps_frac) * float(R)

    # outer boundary in material coords
    outer_elems, outer_colloc, outer_normals, _, _ = build_outer_circle_elements(
        float(R), int(N_outer), p1m, p2m, offset
    )

    # crack polyline in material coords
    xm, ym = rot_to_material(np.asarray(xs_crack_global), np.asarray(ys_crack_global), float(alpha_const))
    crack_elems, crack_colloc, crack_normals, _, _ = build_crack_polyline_elements(
        xm, ym, p1m, p2m, offset
    )

    all_elems = outer_elems + crack_elems
    colloc_pts = np.vstack([outer_colloc, crack_colloc])
    all_normals = np.vstack([outer_normals, crack_normals])

    A = assemble_traction_matrix(colloc_pts, all_normals, all_elems, p1m, p2m, Lm)

    b = np.zeros(2*len(colloc_pts), dtype=float)
    n_outer = len(outer_elems)

    # crack RHS = -background traction
    for i in range(n_outer, len(colloc_pts)):
        xm_i, ym_i = colloc_pts[i]
        n_i = all_normals[i]
        t_bg = traction_from_background_airy(xm_i, ym_i, n_i, R, airy_fit, eval_stress_field_material)
        b[2*i:2*i+2] = -t_bg

    ne = len(all_elems)
    if float(reg_lam) > 0:
        reg = np.sqrt(float(reg_lam)) * np.eye(2*ne)
        A_aug = np.vstack([A, reg])
        b_aug = np.concatenate([b, np.zeros(2*ne)])
    else:
        A_aug, b_aug = A, b

    # remove rigid modes (mean U on outer ~ 0)
    c1 = np.zeros(2*ne, dtype=float)
    c2 = np.zeros(2*ne, dtype=float)
    for j in range(n_outer):
        c1[2*j]   += 1.0
        c2[2*j+1] += 1.0
    c1 /= max(n_outer, 1)
    c2 /= max(n_outer, 1)

    A_aug = np.vstack([A_aug, c1[None,:], c2[None,:]])
    b_aug = np.concatenate([b_aug, [0.0, 0.0]])

    U, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=1e-12)

    return {
        "p1": p1m, "p2": p2m,
        "L": Lm, "A": Am,
        "elems": all_elems,
        "outer_count": n_outer,
        "crack_count": len(crack_elems),
        "crack_poly_m": np.column_stack([xm, ym]),
        "U": U,
    }


# =============================================================================
# Stress intensity factors from near-tip COD (Banks–Sills style)
# =============================================================================

def KI_KII_from_COD_near_tip(
    tip_global: tuple[float, float],
    prev_global: tuple[float, float],
    alpha_const: float,
    E1: float, E2: float, nu12: float, G12: float,
    correction: dict,
    sample_rs,
    cod_offset: float = 1e-5,
) -> tuple[float, float, dict]:
    """
    COD-based KI/KII estimate near a tip.
    Returns (KI, KII, diagnostic dict).
    """
    # tangent/normal (global)
    t_vec_g, n_vec_g = unit_tangent_normal(prev_global, tip_global)

    # into material
    t_vec_m = vec_rot_to_material(*t_vec_g, alpha_const)
    n_vec_m = vec_rot_to_material(*n_vec_g, alpha_const)

    # rotate compliance to local crack direction (material)
    theta_tip_m = np.arctan2(t_vec_m[1], t_vec_m[0])
    Q_tip = rotate_Q(orthotropic_Q_plane_stress(E1, E2, nu12, G12), theta_tip_m)
    S_tip = compliance_from_Q(Q_tip)
    S11t, S22t, S12t, S16t, S26t, S66t = reduced_compliance_components(S_tip)

    p1t, p2t = p_roots_quartic(S11t, S22t, S12t, S16t, S26t, S66t)

    Im = np.imag
    Dm = Im(p1t*p2t)*Im(1.0/(p1t*p2t)) - Im(p1t+p2t)*Im((p1t+p2t)/(p1t*p2t))
    Dm = max(float(Dm), 1e-30)

    p1m, p2m = correction["p1"], correction["p2"]
    Lm, Am = correction["L"], correction["A"]
    elems, U = correction["elems"], correction["U"]

    KI_vals = []
    KII_vals = []

    for r in np.asarray(sample_rs, dtype=float):
        if r <= 0:
            continue

        # point behind tip along crack line (global)
        xg = tip_global[0] - r*t_vec_g[0]
        yg = tip_global[1] - r*t_vec_g[1]

        # opposite faces (global)
        xp = xg + cod_offset*n_vec_g[0]
        yp = yg + cod_offset*n_vec_g[1]
        xm_ = xg - cod_offset*n_vec_g[0]
        ym_ = yg - cod_offset*n_vec_g[1]

        # to material
        xmp, ymp = rot_to_material(xp, yp, alpha_const)
        xmm, ymm = rot_to_material(xm_, ym_, alpha_const)

        up = displacement_at_point(float(xmp), float(ymp), elems, U, p1m, p2m, Lm, Am)
        um = displacement_at_point(float(xmm), float(ymm), elems, U, p1m, p2m, Lm, Am)

        dU_m = up - um  # COD in material basis

        # project to local crack frame (material)
        Du_t = dU_m[0]*t_vec_m[0] + dU_m[1]*t_vec_m[1]
        Du_n = dU_m[0]*n_vec_m[0] + dU_m[1]*n_vec_m[1]

        fac = np.sqrt(2*np.pi/r) / (4*S11t*S22t*Dm)

        KI  = fac * (Du_t*S22t*Im(1.0/(p1t*p2t)) + Du_n*S11t*Im((p1t+p2t)/(p1t*p2t)))
        KII = -fac * (Du_t*S22t*Im((p1t+p2t)/(p1t*p2t)) + Du_n*S11t*Im(p1t*p2t))

        KI_vals.append(float(KI))
        KII_vals.append(float(KII))

    if not KI_vals:
        return np.nan, np.nan, {}

    k = min(5, len(KI_vals))
    KI0  = float(np.median(KI_vals[:k]))
    KII0 = float(np.median(KII_vals[:k]))

    return KI0, KII0, {
        "KI_samples": np.array(KI_vals, float),
        "KII_samples": np.array(KII_vals, float),
        "p_tip": (p1t, p2t),
        "Dm": Dm,
    }


# =============================================================================
# Convenience wrapper – compute SIFs at both tips
# =============================================================================

def sif_two_tips_from_crack(
    crack_x,
    crack_y=None,
    R=None,
    airy_fit=None,
    eval_stress_field_material=None,
    E1=None, E2=None, nu12=None, G12=None,
    alpha_const=None,
    correction=None,
    sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    **solver_kwargs
):
    """
    Compute KI/KII at both crack tips using COD method.

    Supports call patterns:
      sif_two_tips_from_crack(xs, ys, R=..., airy_fit=..., eval_stress_field_material=..., ...)
      sif_two_tips_from_crack(xyNx2, R=..., fit=..., eval_stress_field_material=..., ...)

    Returns: (KI0, KII0, KI1, KII1)
    """
    # allow aliases
    if airy_fit is None and "fit" in solver_kwargs:
        airy_fit = solver_kwargs.pop("fit")
    if airy_fit is None:
        raise TypeError("sif_two_tips_from_crack: airy_fit (or fit=...) is required")

    if R is None:
        R = solver_kwargs.pop("R", None)
    if R is None:
        raise TypeError("sif_two_tips_from_crack: R is required (pass R=...)")

    if eval_stress_field_material is None:
        eval_stress_field_material = solver_kwargs.pop("eval_stress_field_material", None)
    if eval_stress_field_material is None:
        # last-resort fallback (avoid if possible)
        from eval_stress_field_material import eval_stress_field_material as _ev
        eval_stress_field_material = _ev

    # parse crack arrays
    if crack_y is None:
        xy = np.asarray(crack_x, float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise TypeError("If crack_y is None, crack_x must be an (N,2) array")
        xs = xy[:,0]
        ys = xy[:,1]
    else:
        xs = np.asarray(crack_x, float)
        ys = np.asarray(crack_y, float)

    if alpha_const is None:
        alpha_const = solver_kwargs.pop("alpha_const", None)
    if alpha_const is None:
        raise TypeError("sif_two_tips_from_crack: alpha_const is required")

    for name, val in [("E1",E1),("E2",E2),("nu12",nu12),("G12",G12)]:
        if val is None:
            if name in solver_kwargs:
                continue
            raise TypeError(f"sif_two_tips_from_crack: {name} is required")

    E1 = float(E1 if E1 is not None else solver_kwargs.pop("E1"))
    E2 = float(E2 if E2 is not None else solver_kwargs.pop("E2"))
    nu12 = float(nu12 if nu12 is not None else solver_kwargs.pop("nu12"))
    G12 = float(G12 if G12 is not None else solver_kwargs.pop("G12"))
    alpha_const = float(alpha_const)

    if correction is None:
        correction = solve_cracked_disk_correction_ddm(
            float(R), alpha_const, airy_fit,
            xs, ys,
            E1, E2, nu12, G12,
            eval_stress_field_material,
            **solver_kwargs
        )

    pts = np.column_stack([xs, ys])

    # Tip 0
    KI0, KII0, _ = KI_KII_from_COD_near_tip(
        tuple(pts[0]), tuple(pts[1]), alpha_const,
        E1, E2, nu12, G12, correction,
        sample_rs, float(cod_offset)
    )
    # Tip 1
    KI1, KII1, _ = KI_KII_from_COD_near_tip(
        tuple(pts[-1]), tuple(pts[-2]), alpha_const,
        E1, E2, nu12, G12, correction,
        sample_rs, float(cod_offset)
    )

    return float(KI0), float(KII0), float(KI1), float(KII1)


# =============================================================================
# Robust call adapter (needed by crack_helpers / older code)
# =============================================================================

def _filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(func)
    except Exception:
        return kwargs
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_varkw:
        return kwargs
    allowed = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _try_call_sif_two_tips(func, crack_x, crack_y, **kwargs):
    """
    Try calling a SIF function with several likely signatures.
    Maps fit->airy_fit automatically, and passes eval_stress_field_material if present.
    """
    xs = np.asarray(crack_x, float)
    ys = np.asarray(crack_y, float)
    xy = np.column_stack([xs, ys])

    base_kw = dict(kwargs)

    # normalize alias
    if "airy_fit" not in base_kw and "fit" in base_kw:
        base_kw["airy_fit"] = base_kw["fit"]

    ev_fn = base_kw.get("eval_stress_field_material", None)

    # build attempts (positional and keyword variations)
    attempts = []

    # preferred: xs, ys with explicit keywords
    attempts.append(lambda: func(xs, ys, **_filter_kwargs_for_callable(func, base_kw)))

    # Nx2 form
    attempts.append(lambda: func(xy, **_filter_kwargs_for_callable(func, base_kw)))

    # force-keyword names if provided
    if "airy_fit" in base_kw and ev_fn is not None:
        kw2 = dict(base_kw)
        attempts.append(lambda: func(xs, ys,
                                    airy_fit=kw2["airy_fit"],
                                    eval_stress_field_material=ev_fn,
                                    **_filter_kwargs_for_callable(func, kw2)))

    last_err = None
    for caller in attempts:
        try:
            return caller()
        except TypeError as e:
            last_err = e

    raise TypeError(f"Could not call SIF function with available patterns. Last error: {last_err}")



def traction_from_element_u(zx, zy, nx, ny, elem, p1, p2, L, u):
    """
    Traction at (zx,zy) on plane with normal (nx,ny) caused by ONE constant DDM element,
    using the actual complex DD vector u=[u1,u2] (not basis calls).
    All coordinates are in MATERIAL frame.
    """
    zx = float(zx); zy = float(zy)
    nx = float(nx); ny = float(ny)

    u = np.asarray(u, dtype=complex).reshape(2,)

    ps = (p1, p2)

    sig1 = np.zeros(2, dtype=float)  # [σ11, σ12]
    sig2 = np.zeros(2, dtype=float)  # [σ21, σ22]

    for a, p in enumerate(ps):
        z = zx + p * zy
        I = (1.0/(z - elem.xi2[a]) - 1.0/(z - elem.xi1[a])) / (2j*np.pi)

        dot = L[0, a]*u[0] + L[1, a]*u[1]
        coeff = dot * I

        sig2[0] += 2.0*np.real(L[0, a]*coeff)
        sig2[1] += 2.0*np.real(L[1, a]*coeff)
        sig1[0] += -2.0*np.real(p*L[0, a]*coeff)
        sig1[1] += -2.0*np.real(p*L[1, a]*coeff)

    # correct scalar tractions
    t1 = sig1[0]*nx + sig1[1]*ny
    t2 = sig2[0]*nx + sig2[1]*ny

    return np.array([float(t1), float(t2)], dtype=float)



def kink_angle_pls_anisotropic_from_traction(
    tip_global, psi_tip_global,
    alpha_const,
    correction,
    R, airy_fit, eval_stress_field_material,
    vec_rot_to_material,
    rot_to_material,
    theta_max_rad=np.deg2rad(8.0),
    r_ahead_frac=0.015,
    n_scan=31,
    bisection_iters=30,
):
    """
    Find kink angle theta in [-theta_max,+theta_max] such that shear traction ahead is ~0.
    Choose the root that maximizes tensile normal traction sigma_n.

    Returns theta (rad) in GLOBAL direction convention: psi_new = psi_tip_global + theta.
    """
    theta_max = float(abs(theta_max_rad))
    r_ahead = float(r_ahead_frac) * float(R)

    # candidate thetas
    thetas = np.linspace(-theta_max, theta_max, int(n_scan))

    tau = np.zeros_like(thetas, dtype=float)
    sigN = np.zeros_like(thetas, dtype=float)

    for i, th in enumerate(thetas):
        psi = float(psi_tip_global + th)
        t_g = np.array([np.cos(psi), np.sin(psi)], float)
        n_g = np.array([-np.sin(psi), np.cos(psi)], float)

        # point ahead in global
        xg = float(tip_global[0] + r_ahead * t_g[0])
        yg = float(tip_global[1] + r_ahead * t_g[1])

        # convert to material
        xm, ym = rot_to_material(xg, yg, alpha_const)
        t_m = np.array(vec_rot_to_material(t_g[0], t_g[1], alpha_const), float)
        n_m = np.array(vec_rot_to_material(n_g[0], n_g[1], alpha_const), float)

        # total traction on candidate plane
        tt = total_traction_at_point_material(
            xm, ym, n_m,
            correction,
            float(R), airy_fit, eval_stress_field_material
        )

        tau[i] = float(tt @ t_m)   # shear traction on plane
        sigN[i] = float(tt @ n_m)  # normal traction (positive=tension if your convention is +tension)

    # find sign changes of tau
    s = np.sign(tau)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        # fallback: choose min |tau|, but prefer tensile sigma_n
        j = int(np.argmin(np.abs(tau) - 1e-6*sigN))
        return float(np.clip(thetas[j], -theta_max, theta_max))

    # bisection refine each bracket, then choose best by sigma_n (tensile max)
    candidates = []
    for k in idx:
        a = float(thetas[k]); b = float(thetas[k+1])
        fa = float(tau[k]);   fb = float(tau[k+1])

        for _ in range(int(bisection_iters)):
            m = 0.5*(a+b)
            psi = float(psi_tip_global + m)
            t_g = np.array([np.cos(psi), np.sin(psi)], float)
            n_g = np.array([-np.sin(psi), np.cos(psi)], float)
            xg = float(tip_global[0] + r_ahead * t_g[0])
            yg = float(tip_global[1] + r_ahead * t_g[1])

            xm, ym = rot_to_material(xg, yg, alpha_const)
            t_m = np.array(vec_rot_to_material(t_g[0], t_g[1], alpha_const), float)
            n_m = np.array(vec_rot_to_material(n_g[0], n_g[1], alpha_const), float)

            tt = total_traction_at_point_material(
                xm, ym, n_m,
                correction, float(R), airy_fit, eval_stress_field_material
            )

            fm = float(tt @ t_m)
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        th_star = 0.5*(a+b)

        # evaluate sigma_n at root
        psi = float(psi_tip_global + th_star)
        t_g = np.array([np.cos(psi), np.sin(psi)], float)
        n_g = np.array([-np.sin(psi), np.cos(psi)], float)
        xg = float(tip_global[0] + r_ahead * t_g[0])
        yg = float(tip_global[1] + r_ahead * t_g[1])

        xm, ym = rot_to_material(xg, yg, alpha_const)
        n_m = np.array(vec_rot_to_material(n_g[0], n_g[1], alpha_const), float)

        tt = total_traction_at_point_material(
            xm, ym, n_m,
            correction, float(R), airy_fit, eval_stress_field_material
        )

        sigma_n = float(tt @ n_m)
        candidates.append((th_star, sigma_n))

    # choose the root with maximum tensile normal traction
    candidates.sort(key=lambda z: z[1], reverse=True)
    return float(np.clip(candidates[0][0], -theta_max, theta_max))