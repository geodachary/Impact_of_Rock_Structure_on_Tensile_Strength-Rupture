"""Displacement-discontinuity solve on the cut-cell disk: moduli fields, contact loading, PCG/BiCGSTAB solvers, stresses.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 8 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.

Parity note
-----------
schist cell 8 is a stale duplicate of this cell and is dropped.
"""
from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from tools.analysis._context import bind as _bind, current as _rock


# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    band_open_factor_from_sigma_n, hertz_contact_halfwidth,
)
from tools.data_io import load_specimen_table



# --- implementation ---------------------------------------------------


def to_MPa_modulus(val):
    val = float(val)
    return val * 1e3 if val < 1e3 else val


def orthotropic_from_UCS_anisotropy(E_ref_mpa, nu12, G_ref_mpa, ucs_ratio, p=1.0):
    E_ref = float(E_ref_mpa)
    nu12 = float(nu12)
    r = max(float(ucs_ratio), 1.0) ** float(p)
    sr = np.sqrt(r)
    E1 = E_ref * sr
    E2 = E_ref / sr
    if np.isfinite(G_ref_mpa) and G_ref_mpa > 0:
        G12 = float(G_ref_mpa)
    else:
        G12 = np.sqrt(E1 * E2) / (2.0 * (1.0 + nu12))
    return E1, E2, nu12, G12


def principal_stresses_2d(sxx, syy, txy):
    s_avg = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5*(sxx - syy))**2 + txy**2)
    return s_avg + rad, s_avg - rad


def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def harmonic_mix(a, b, w):
    return 1.0 / ((1.0 - w)/(a + 1e-30) + w/(b + 1e-30) + 1e-30)


def kn_ks_from_min_fractions(E2_base, G12_base, fracE2, fracG, t_band):
    fracE2 = max(float(fracE2), 1e-6)
    fracG  = max(float(fracG), 1e-6)
    kn = (fracE2 * float(E2_base)) / (t_band + 1e-30)
    ks = (fracG  * float(G12_base)) / (t_band + 1e-30)
    return kn, ks


def sigma_normal_to_plane(sxx, syy, txy, plane_angle):
    a = np.asarray(plane_angle, float)
    nx = -np.sin(a); ny = np.cos(a)
    return sxx*(nx*nx) + 2.0*txy*(nx*ny) + syy*(ny*ny)


def gaussian_random_field(shape, dx, corr_len_m, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)

    ky = np.fft.fftfreq(shape[0], d=dx) * 2*np.pi
    kx = np.fft.fftfreq(shape[1], d=dx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX*KX + KY*KY

    L = max(float(corr_len_m), 1e-12)
    H = np.exp(-0.5 * K2 * (L**2))

    W = np.fft.fft2(w)
    f = np.fft.ifft2(W * H).real
    f -= np.mean(f)
    s = np.std(f)
    if s > 1e-12:
        f /= s
    return f


def weak_band_weight_phasewarp(X, Y, angle_local, spacing, warp_m, band_halfwidth_m, phase, sharp_power):
    bw = float(band_halfwidth_m)
    s = float(spacing)
    a = np.asarray(angle_local, float)
    warp = np.asarray(warp_m, float)

    nx = -np.sin(a)
    ny =  np.cos(a)

    d = nx*X + ny*Y + float(phase)
    d_eff = d + warp
    dist_eff = (s / np.pi) * np.abs(np.sin(np.pi * d_eff / (s + 1e-30)))
    w = np.exp(-sharp_power * (dist_eff / (bw + 1e-30))**2)
    return np.clip(w, 0.0, 1.0)


def qbar_from_Es(E1, E2, nu12, G12, theta):
    E1 = np.asarray(E1, float)
    E2 = np.asarray(E2, float)
    G12 = np.asarray(G12, float)
    nu12 = float(nu12)

    nu21 = nu12 * (E2 / (E1 + 1e-30))
    den = 1.0 - nu12*nu21

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    m = np.cos(theta); n = np.sin(theta)
    m2 = m*m; n2 = n*n
    m3 = m2*m; n3 = n2*n
    m4 = m2*m2; n4 = n2*n2

    Qbar11 = Q11*m4 + 2.0*(Q12 + 2.0*Q66)*m2*n2 + Q22*n4
    Qbar22 = Q11*n4 + 2.0*(Q12 + 2.0*Q66)*m2*n2 + Q22*m4
    Qbar12 = (Q11 + Q22 - 4.0*Q66)*m2*n2 + Q12*(m4 + n4)
    Qbar16 = (Q11 - Q12 - 2.0*Q66)*m3*n - (Q22 - Q12 - 2.0*Q66)*m*n3
    Qbar26 = (Q11 - Q12 - 2.0*Q66)*m*n3 - (Q22 - Q12 - 2.0*Q66)*m3*n
    Qbar66 = (Q11 + Q22 - 2.0*Q12 - 2.0*Q66)*m2*n2 + Q66*(m4 + n4)
    return Qbar11, Qbar22, Qbar12, Qbar16, Qbar26, Qbar66


def ddx(f, mask, h):
    out = np.zeros_like(f)
    central = mask[:, 1:-1] & mask[:, 0:-2] & mask[:, 2:]
    forward = mask[:, 1:-1] & (~mask[:, 0:-2]) & mask[:, 2:]
    backward = mask[:, 1:-1] & mask[:, 0:-2] & (~mask[:, 2:])
    core = out[:, 1:-1]
    core[:] = 0.0
    core += central * ((f[:, 2:] - f[:, 0:-2]) / (2*h))
    core += forward * ((f[:, 2:] - f[:, 1:-1]) / h)
    core += backward * ((f[:, 1:-1] - f[:, 0:-2]) / h)
    eL = mask[:, 0] & mask[:, 1]
    eR = mask[:, -1] & mask[:, -2]
    out[eL, 0] = (f[eL, 1] - f[eL, 0]) / h
    out[eR, -1] = (f[eR, -1] - f[eR, -2]) / h
    out[~mask] = 0.0
    return out


def ddy(f, mask, h):
    out = np.zeros_like(f)
    central = mask[1:-1, :] & mask[0:-2, :] & mask[2:, :]
    forward = mask[1:-1, :] & (~mask[0:-2, :]) & mask[2:, :]
    backward = mask[1:-1, :] & mask[0:-2, :] & (~mask[2:, :])
    core = out[1:-1, :]
    core[:] = 0.0
    core += central * ((f[2:, :] - f[0:-2, :]) / (2*h))
    core += forward * ((f[2:, :] - f[1:-1, :]) / h)
    core += backward * ((f[1:-1, :] - f[0:-2, :]) / h)
    eB = mask[0, :] & mask[1, :]
    eT = mask[-1, :] & mask[-2, :]
    out[0, eB] = (f[1, eB] - f[0, eB]) / h
    out[-1, eT] = (f[-1, eT] - f[-2, eT]) / h
    out[~mask] = 0.0
    return out


def boundary_segments_cut_cells(X, Y, mask, R, h):
    mR = np.zeros_like(mask); mR[:, :-1] = mask[:, 1:]
    mL = np.zeros_like(mask); mL[:, 1:]  = mask[:, :-1]
    mU = np.zeros_like(mask); mU[:-1, :] = mask[1:, :]
    mD = np.zeros_like(mask); mD[1:, :]  = mask[:-1, :]

    boundary = mask & (~(mR & mL & mU & mD))
    I0, J0 = np.where(boundary)

    Iseg, Jseg = [], []
    xm_list, ym_list, nx_list, ny_list, L_list = [], [], [], [], []

    def phi_circle(x, y):
        return x*x + y*y - R*R

    edges = [(0,1),(1,2),(2,3),(3,0)]

    for i, j in zip(I0, J0):
        xc = float(X[i, j]); yc = float(Y[i, j])
        c = np.array([
            [xc - 0.5*h, yc - 0.5*h],
            [xc + 0.5*h, yc - 0.5*h],
            [xc + 0.5*h, yc + 0.5*h],
            [xc - 0.5*h, yc + 0.5*h],
        ], float)

        phi = phi_circle(c[:, 0], c[:, 1])

        pts = []
        for a, b in edges:
            pa = phi[a]; pb = phi[b]
            if (pa <= 0 and pb >= 0) or (pa >= 0 and pb <= 0):
                denom = (pa - pb)
                if abs(denom) < 1e-30:
                    continue
                t = pa / (pa - pb)
                pts.append(c[a] + t*(c[b] - c[a]))

        if len(pts) < 2:
            continue

        pts = np.array(pts, float)
        if len(pts) > 2:
            dmax = -1.0
            p1 = pts[0]; p2 = pts[1]
            for a in range(len(pts)):
                for b in range(a+1, len(pts)):
                    d = float(np.sum((pts[a] - pts[b])**2))
                    if d > dmax:
                        dmax = d
                        p1 = pts[a]
                        p2 = pts[b]
        else:
            p1, p2 = pts[0], pts[1]

        th1 = float(np.arctan2(p1[1], p1[0]))
        th2 = float(np.arctan2(p2[1], p2[0]))
        dth = (th2 - th1) % (2*np.pi)
        dth = min(dth, 2*np.pi - dth)
        L = R * dth
        if (not np.isfinite(L)) or (L <= 0):
            continue

        u1 = np.array([np.cos(th1), np.sin(th1)])
        u2 = np.array([np.cos(th2), np.sin(th2)])
        um = u1 + u2
        if np.hypot(um[0], um[1]) < 1e-12:
            pm = 0.5*(p1 + p2)
            thm = float(np.arctan2(pm[1], pm[0]))
        else:
            thm = float(np.arctan2(um[1], um[0]))

        xb = R*np.cos(thm); yb = R*np.sin(thm)
        nx = xb / (R + 1e-30); ny = yb / (R + 1e-30)

        Iseg.append(int(i)); Jseg.append(int(j))
        xm_list.append(xb); ym_list.append(yb)
        nx_list.append(nx); ny_list.append(ny)
        L_list.append(L)

    Iseg = np.asarray(Iseg, dtype=int)
    Jseg = np.asarray(Jseg, dtype=int)
    xb = np.asarray(xm_list, dtype=float)
    yb = np.asarray(ym_list, dtype=float)
    nbx = np.asarray(nx_list, dtype=float)
    nby = np.asarray(ny_list, dtype=float)
    Lseg = np.asarray(L_list, dtype=float)

    if Iseg.size == 0:
        raise RuntimeError("No cut-cell boundary segments found. Increase grid_N.")
    return Iseg, Jseg, xb, yb, nbx, nby, Lseg


def build_moduli_fields(w_active):
    w = np.clip(w_active, 0.0, 1.0)

    E1_intact = E1_base * stiff_factor
    E2_intact = E2_base * stiff_factor
    G12_intact = G12_base * stiff_factor

    E1_weak = E1_band * np.ones_like(E1_intact)
    E2_weak = E2_band * np.ones_like(E2_intact)
    G12_weak = G_band  * np.ones_like(G12_intact)

    E1 = harmonic_mix(E1_intact, E1_weak, w)
    E2 = harmonic_mix(E2_intact, E2_weak, w)
    G12 = harmonic_mix(G12_intact, G12_weak, w)

    return np.maximum(E1, 1e-6), np.maximum(E2, 1e-6), np.maximum(G12, 1e-6)


def update_Q_from_moduli(E1, E2, G12):
    global Q11, Q22, Q12, Q16, Q26, Q66, _precond_du, _precond_dv
    Q11, Q22, Q12, Q16, Q26, Q66 = qbar_from_Es(E1, E2, nu12, G12, theta=float(alpha))
    if use_precond:
        mag = (np.abs(Q11) + np.abs(Q22) + 2*np.abs(Q12) + np.abs(Q66) +
               np.abs(Q16) + np.abs(Q26))
        du = (mag / (h*h + 1e-30)) + float(precond_floor)
        dv = du.copy()
        du[~mask] = 1.0
        dv[~mask] = 1.0
        _precond_du = du
        _precond_dv = dv
    else:
        _precond_du = None
        _precond_dv = None


def stresses(u, v):
    exx = ddx(u, mask, h)
    eyy = ddy(v, mask, h)
    gxy = ddy(u, mask, h) + ddx(v, mask, h)
    sxx = Q11*exx + Q12*eyy + Q16*gxy
    syy = Q12*exx + Q22*eyy + Q26*gxy
    txy = Q16*exx + Q26*eyy + Q66*gxy
    sxx[~mask] = 0.0
    syy[~mask] = 0.0
    txy[~mask] = 0.0
    return sxx, syy, txy


def apply_K_inplace(u, v, fx_out, fy_out):
    sxx, syy, txy = stresses(u, v)

    sxx_fx = _work["sxx_fx"]; txy_fx = _work["txy_fx"]
    syy_fy = _work["syy_fy"]; txy_fy = _work["txy_fy"]
    tmpL   = _work["tmpL"];   tmpD   = _work["tmpD"]
    divx   = _work["divx"];   divy   = _work["divy"]

    sxx_fx.fill(0.0); txy_fx.fill(0.0); syy_fy.fill(0.0); txy_fy.fill(0.0)

    sxx_fx[:, :-1] = 0.5*(sxx[:, :-1] + sxx[:, 1:]) * mask_x[:, :-1]
    txy_fx[:, :-1] = 0.5*(txy[:, :-1] + txy[:, 1:]) * mask_x[:, :-1]

    syy_fy[:-1, :] = 0.5*(syy[:-1, :] + syy[1:, :]) * mask_y[:-1, :]
    txy_fy[:-1, :] = 0.5*(txy[:-1, :] + txy[1:, :]) * mask_y[:-1, :]

    tmpL.fill(0.0); tmpL[:, 1:] = sxx_fx[:, :-1]
    tmpD.fill(0.0); tmpD[1:, :] = txy_fy[:-1, :]
    divx[:] = (sxx_fx - tmpL + txy_fy - tmpD) / h

    tmpL.fill(0.0); tmpL[:, 1:] = txy_fx[:, :-1]
    tmpD.fill(0.0); tmpD[1:, :] = syy_fy[:-1, :]
    divy[:] = (txy_fx - tmpL + syy_fy - tmpD) / h

    fx_out[:] = -divx
    fy_out[:] = -divy
    fx_out[~mask] = 0.0
    fy_out[~mask] = 0.0


def dot_mask(ax, ay, bx, by):
    axf = ax.ravel(); ayf = ay.ravel()
    bxf = bx.ravel(); byf = by.ravel()
    return float(np.dot(axf[mask_idx], bxf[mask_idx]) + np.dot(ayf[mask_idx], byf[mask_idx]))


def norm_mask(ax, ay):
    return float(np.sqrt(dot_mask(ax, ay, ax, ay))) + 1e-30


def project_rigid(u, v):
    uf = u.ravel(); vf = v.ravel()
    n = mask_idx.size
    if n <= 0:
        return u, v
    um = float(np.mean(uf[mask_idx]))
    vm = float(np.mean(vf[mask_idx]))
    uf[mask_idx] -= um
    vf[mask_idx] -= vm
    denom = float(np.sum((Xf[mask_idx]**2 + Yf[mask_idx]**2))) + 1e-30
    omega = float(np.sum(Xf[mask_idx]*vf[mask_idx] - Yf[mask_idx]*uf[mask_idx]) / denom)
    uf[mask_idx] += omega * Yf[mask_idx]
    vf[mask_idx] -= omega * Xf[mask_idx]
    return u, v


def project_force_moment(bx, by):
    bxf = bx.ravel(); byf = by.ravel()
    A = mask_idx.size
    if A <= 0:
        return bx, by
    bx0 = float(np.mean(bxf[mask_idx]))
    by0 = float(np.mean(byf[mask_idx]))
    bxf[mask_idx] -= bx0
    byf[mask_idx] -= by0
    denom = float(np.sum((Xf[mask_idx]**2 + Yf[mask_idx]**2))) + 1e-30
    moment = float(np.sum(Xf[mask_idx]*byf[mask_idx] - Yf[mask_idx]*bxf[mask_idx]))
    c = moment / denom
    bxf[mask_idx] += c * Yf[mask_idx]
    byf[mask_idx] -= c * Xf[mask_idx]
    return bx, by


def Minv(ax, ay, zx, zy):
    if _precond_du is None:
        zx[:] = ax
        zy[:] = ay
    else:
        zx[:] = ax / (_precond_du + 1e-30)
        zy[:] = ay / (_precond_dv + 1e-30)
    zx[~mask] = 0.0
    zy[~mask] = 0.0


def solve_pcg(bx, by, u0=None, v0=None, maxiter=1000, tol_rel=1e-7):
    # If (p,Ap) becomes non-positive, PCG is not valid -> return failure
    if u0 is None:
        u = np.zeros_like(bx)
        v = np.zeros_like(by)
    else:
        u = u0.copy()
        v = v0.copy()

    u, v = project_rigid(u, v)

    Ax = np.zeros_like(bx); Ay = np.zeros_like(by)
    apply_K_inplace(u, v, Ax, Ay)

    rx = bx - Ax
    ry = by - Ay
    rx[~mask] = 0.0
    ry[~mask] = 0.0

    r0 = norm_mask(rx, ry)
    if r0 < 1e-30:
        return u, v, True

    zx = np.zeros_like(rx); zy = np.zeros_like(ry)
    Minv(rx, ry, zx, zy)

    px = zx.copy(); py = zy.copy()
    rz_old = dot_mask(rx, ry, zx, zy)

    Ap_x = np.zeros_like(rx); Ap_y = np.zeros_like(ry)

    for _k in range(maxiter):
        apply_K_inplace(px, py, Ap_x, Ap_y)
        pAp = dot_mask(px, py, Ap_x, Ap_y)
        if not np.isfinite(pAp) or pAp <= 0.0:
            return u, v, False  # not SPD / breakdown

        alpha_c = rz_old / (pAp + 1e-30)

        u[mask] += alpha_c * px[mask]
        v[mask] += alpha_c * py[mask]
        u, v = project_rigid(u, v)

        rx[mask] -= alpha_c * Ap_x[mask]
        ry[mask] -= alpha_c * Ap_y[mask]
        rx[~mask] = 0.0
        ry[~mask] = 0.0

        if norm_mask(rx, ry) / r0 < tol_rel:
            return u, v, True

        Minv(rx, ry, zx, zy)
        rz_new = dot_mask(rx, ry, zx, zy)

        beta = rz_new / (rz_old + 1e-30)
        px[mask] = zx[mask] + beta * px[mask]
        py[mask] = zy[mask] + beta * py[mask]
        px[~mask] = 0.0
        py[~mask] = 0.0
        rz_old = rz_new

    return u, v, True


def solve_bicgstab(bx, by, u0=None, v0=None, maxiter=2500, tol_rel=1e-7):
    if u0 is None:
        u = np.zeros_like(bx)
        v = np.zeros_like(by)
    else:
        u = u0.copy()
        v = v0.copy()

    u, v = project_rigid(u, v)

    Ku = np.zeros_like(bx)
    Kv = np.zeros_like(by)
    apply_K_inplace(u, v, Ku, Kv)

    rx = bx - Ku
    ry = by - Kv
    rx[~mask] = 0.0
    ry[~mask] = 0.0

    rhatx = rx.copy()
    rhaty = ry.copy()

    rho_old = 1.0
    alpha_c = 1.0
    omega = 1.0

    px = np.zeros_like(rx); py = np.zeros_like(ry)
    vx = np.zeros_like(rx); vy = np.zeros_like(ry)

    sx = np.zeros_like(rx); sy = np.zeros_like(ry)
    tx = np.zeros_like(rx); ty = np.zeros_like(ry)

    yx = np.zeros_like(rx); yy = np.zeros_like(ry)
    zx = np.zeros_like(rx); zy = np.zeros_like(ry)

    r0 = norm_mask(rx, ry)

    for _k in range(maxiter):
        rho_new = dot_mask(rhatx, rhaty, rx, ry)
        if abs(rho_new) < 1e-30:
            break

        beta_c = (rho_new/rho_old) * (alpha_c/omega)

        px[mask] = rx[mask] + beta_c*(px[mask] - omega*vx[mask])
        py[mask] = ry[mask] + beta_c*(py[mask] - omega*vy[mask])
        px[~mask] = 0.0
        py[~mask] = 0.0

        Minv(px, py, yx, yy)
        apply_K_inplace(yx, yy, vx, vy)

        denom = dot_mask(rhatx, rhaty, vx, vy) + 1e-30
        alpha_c = rho_new / denom

        sx[:] = rx - alpha_c*vx
        sy[:] = ry - alpha_c*vy
        sx[~mask] = 0.0
        sy[~mask] = 0.0

        if norm_mask(sx, sy) / r0 < tol_rel:
            u[mask] += alpha_c*yx[mask]
            v[mask] += alpha_c*yy[mask]
            u, v = project_rigid(u, v)
            return u, v

        Minv(sx, sy, zx, zy)
        apply_K_inplace(zx, zy, tx, ty)

        tt = dot_mask(tx, ty, tx, ty) + 1e-30
        omega = dot_mask(tx, ty, sx, sy) / tt

        u[mask] += alpha_c*yx[mask] + omega*zx[mask]
        v[mask] += alpha_c*yy[mask] + omega*zy[mask]
        u, v = project_rigid(u, v)

        rx[:] = sx - omega*tx
        ry[:] = sy - omega*ty
        rx[~mask] = 0.0
        ry[~mask] = 0.0

        if norm_mask(rx, ry) / r0 < tol_rel:
            return u, v

        rho_old = rho_new

    return u, v


def solve_system(bx, by, u0, v0, tol_rel, maxiter):
    if solver_mode == "pcg_then_bicgstab":
        u, v, ok = solve_pcg(bx, by, u0=u0, v0=v0, maxiter=maxiter, tol_rel=tol_rel)
        if ok:
            return u, v
        # fallback
        return solve_bicgstab(bx, by, u0=u, v0=v, maxiter=maxiter, tol_rel=tol_rel)
    else:
        return solve_bicgstab(bx, by, u0=u0, v0=v0, maxiter=maxiter, tol_rel=tol_rel)


def build_rhs_contact_unit_load(b_contact_m):
    bx_rhs = np.zeros_like(X)
    by_rhs = np.zeros_like(Y)

    th = np.arctan2(yb, xb)
    s_top = R * np.abs(wrap_pi(th - np.pi/2))
    s_bot = R * np.abs(wrap_pi(th + np.pi/2))

    b = float(max(b_contact_m, 1e-6))
    wtop = np.sqrt(np.maximum(1.0 - (s_top/b)**2, 0.0))
    wbot = np.sqrt(np.maximum(1.0 - (s_bot/b)**2, 0.0))
    wload = wtop + wbot

    denom = 1e6 * thickness * float(np.sum(wload * Lseg))
    if denom <= 0:
        raise RuntimeError("Contact width too small -> no boundary segments loaded. Increase b_contact_m or grid_N.")
    p0_mpa = 1.0 / denom

    p = p0_mpa * wload
    tx = -p * nbx
    ty = -p * nby

    if platen_mu != 0.0:
        tx_t = -nby
        ty_t =  nbx
        tau = float(platen_mu) * p
        tx += -tau * tx_t
        ty += -tau * ty_t

    wfac = (Lseg / (h*h))
    np.add.at(bx_rhs, (Iseg, Jseg), tx * wfac)
    np.add.at(by_rhs, (Iseg, Jseg), ty * wfac)

    bx_rhs[~mask] = 0.0
    by_rhs[~mask] = 0.0
    bx_rhs, by_rhs = project_force_moment(bx_rhs, by_rhs)
    return bx_rhs, by_rhs, p0_mpa


def add_band_lines(ax, R, band_angle, spacing, phase, halfwidth=None):
    a = float(band_angle)
    s = float(spacing)
    ph = float(phase)
    n = np.array([-np.sin(a), np.cos(a)], float)
    t = np.array([ np.cos(a), np.sin(a)], float)

    def draw(offset, lw, ls, alpha=0.25):
        kmin = int(np.ceil((ph - R - offset) / s))
        kmax = int(np.floor((ph + R - offset) / s))
        for k in range(kmin, kmax + 1):
            d0 = (k*s - ph) + offset
            if abs(d0) > R:
                continue
            x0 = d0 * n
            L = np.sqrt(max(R*R - d0*d0, 0.0))
            p1 = x0 - L*t
            p2 = x0 + L*t
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k", lw=lw, ls=ls, alpha=alpha)

    draw(0.0, 1.1, "-")
    if halfwidth is not None and halfwidth > 0:
        draw(+halfwidth, 0.8, "--", alpha=0.18)
        draw(-halfwidth, 0.8, "--", alpha=0.18)


def gaussian_smooth_nan_fft(field, mask, sigma_pix):
    sigma_pix = float(sigma_pix)
    if sigma_pix <= 0:
        out = np.array(field, float)
        out[~mask] = np.nan
        return out

    H, W = field.shape
    key = (H, W, sigma_pix)
    if key in _fft_cache:
        Gk = _fft_cache[key]
    else:
        ky = np.fft.fftfreq(H, d=1.0) * 2*np.pi
        kx = np.fft.fftfreq(W, d=1.0) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX*KX + KY*KY
        Gk = np.exp(-0.5 * (sigma_pix**2) * K2)
        _fft_cache[key] = Gk

    f0 = np.where(mask, field, 0.0)
    m0 = mask.astype(float)

    num = np.fft.ifft2(np.fft.fft2(f0) * Gk).real
    den = np.fft.ifft2(np.fft.fft2(m0) * Gk).real
    out = num / (den + 1e-30)
    out[~mask] = np.nan
    return out


def plot_combined_disk(field, title, cbar_label):
    extent = [xlin[0], xlin[-1], ylin[0], ylin[-1]]
    f = np.ma.array(field, mask=~mask)

    vals = np.asarray(field[mask], float)
    abs_lim = float(np.nanpercentile(np.abs(vals), disk_percentile_abs)) if vals.size else 1.0
    abs_lim = max(abs_lim, 1e-6)
    vmin, vmax = (-abs_lim, abs_lim) if colorbar_symmetric else (np.nanmin(vals), np.nanmax(vals))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(f, origin="lower", extent=extent, cmap="coolwarm",
                   norm=norm, interpolation=plot_interpolation)

    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.88)
    cb.set_label(cbar_label)
    cb.locator = MaxNLocator(nbins=int(colorbar_nbins))
    cb.formatter = FormatStrFormatter(colorbar_fmt)
    cb.update_ticks()

    fs = gaussian_smooth_nan_fft(field, mask, sigma_pix=contour_sigma_pix)
    ax.contour(X, Y, fs,
               levels=np.linspace(vmin, vmax, contour_levels),
               colors="k", linewidths=contour_linewidth, alpha=contour_alpha,
               corner_mask=contour_corner_mask)

    if show_band_lines:
        add_band_lines(ax, R, band_angle0, spacing_m, phase_plot,
                       halfwidth=(band_halfwidth_m if show_band_edges else None))

    th = np.linspace(0, 2*np.pi, 500)
    ax.plot(R*np.cos(th), R*np.sin(th), "k-", lw=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_one_radial(X, Y, sigma_tens, sigma_comp):
    r = np.sqrt(X[mask]**2 + Y[mask]**2).ravel()
    ten = sigma_tens[mask].ravel()
    comp = sigma_comp[mask].ravel()

    m = np.isfinite(r) & np.isfinite(ten) & np.isfinite(comp)
    r = r[m]; ten = ten[m]; comp = comp[m]

    rmax = R * (1.0 - float(radial_exclude_boundary_fraction))
    keep = r <= rmax
    r = r[keep]; ten = ten[keep]; comp = comp[keep]

    if r.size > radial_scatter_max_points:
        rng = np.random.default_rng(radial_seed)
        idx = rng.choice(r.size, size=radial_scatter_max_points, replace=False)
        r = r[idx]; ten = ten[idx]; comp = comp[idx]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    mten = ten > 0
    mcomp = comp < 0

    ax.scatter(r[mten], ten[mten], s=6, alpha=0.25, label="Tensile (σ1+)")
    ax.scatter(r[mcomp], comp[mcomp], s=6, alpha=0.25, label="Compressive (σyy−)")

    bins = np.linspace(0.0, rmax, int(radial_bins) + 1)
    centers = 0.5*(bins[:-1] + bins[1:])

    def binned_median(x, y):
        out = np.full(centers.shape, np.nan)
        for i in range(len(centers)):
            sel = (x >= bins[i]) & (x < bins[i+1])
            if np.any(sel):
                out[i] = np.nanmedian(y[sel])
        return out

    ax.plot(centers, binned_median(r, ten), lw=2.0, label="Median σ1+ (binned)")
    ax.plot(centers, binned_median(r, comp), lw=2.0, label="Median σyy− (binned)")

    ax.set_xlabel("Radial Distance r (m)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Stress vs Radial Distance (boundary ring excluded)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    plt.show()



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global E1, E1_band, E1_base, E2, E2_band, E2_base, E_eff_MPa, E_in, \
        E_ref_mpa, G12, G12_base, G_band, G_in, G_ref_mpa, Iseg, Jseg, \
        Lseg, P_calib_N, Q11, Q12, Q16, Q22, Q26, Q66, R, X, Xf, Y, Yf, \
        _fft_cache, _precond_du, _precond_dv, _work, alpha, \
        angle_hetero_deg, angle_local, b_contact, b_new, b_old, \
        band_angle0, band_angle_deg, band_halfwidth_m, \
        band_min_E1_fraction, band_min_E2_fraction, band_min_G12_fraction, \
        band_sharp_power, bx_rhs, by_rhs, closed_compliance_fraction, \
        closure_fixed_point_iters_max, closure_rel_change_tol, \
        closure_sigma0_mpa, colorbar_fmt, colorbar_nbins, \
        colorbar_symmetric, combined, contour_alpha, contour_corner_mask, \
        contour_levels, contour_linewidth, contour_sigma_pix, den, \
        diameter, disk_percentile_abs, enable_heterogeneity, \
        enable_joint_closure, fA, fE, fW, final_b_contact, \
        final_p0_unit_mpa, final_w_active, grid_N, h, \
        hertz_outer_iters_max, hertz_rel_b_tol, hetero_corr_len_m, \
        hetero_seed, ic, index_number, it, jc, joint_kn_MPa_per_m, \
        joint_ks_MPa_per_m, kn0, ks0, mask, mask_idx, mask_x, mask_y, \
        maxiter_final, maxiter_inner, n_fp, n_outer, nbx, nby, nu12, \
        nu_contact, num, open_fac, outer, p0_unit_mpa, p_link, p_peak_mpa, \
        phase_mode, phase_plot, phase_shift_m, platen_mu, \
        plot_interpolation, precond_floor, radial_bins, \
        radial_exclude_boundary_fraction, radial_scatter_max_points, \
        radial_seed, rel, s1, s3, scale, scale_tmp, show_band_edges, \
        show_band_lines, sigma1_plus, sigma_ln, sigma_n, sign, sign_tmp, \
        solver_mode, spacing_m, spacing_warp_amp_m, st, stiff_factor, \
        stiffness_cv, sxx, sxx0, sxx_center0, sxx_sc, syy, syy0, syy_cap, \
        syy_minus, syy_sc, t_band, target_sigma_xx_center_mpa, thickness, \
        tol_rel_final, tol_rel_inner, txy, txy0, txy_sc, u, u_prev, \
        ucs_anisotropy_ratio, use_hertz_contact_width, use_precond, v, \
        v_prev, w_active, w_band, w_comp, w_new, w_prev, warp_m, xb, xlin, \
        yb, ylin
    _bind(rock)
    df = load_specimen_table()
    index_number = 4
    target_sigma_xx_center_mpa = 0.63
    spacing_m = _rock().spacing_m
    ucs_anisotropy_ratio = 2.5
    p_link = 1.0
    band_halfwidth_m = 0.00045
    band_sharp_power = 12
    phase_mode = "center_safe"
    phase_shift_m = 0.0
    band_angle_deg = None
    joint_kn_MPa_per_m = None
    joint_ks_MPa_per_m = None
    band_min_E2_fraction = 0.20
    band_min_G12_fraction = 0.15
    band_min_E1_fraction = 0.95
    enable_joint_closure = True
    closure_sigma0_mpa = 0.20
    closed_compliance_fraction = 0.08
    closure_fixed_point_iters_max = 3
    closure_rel_change_tol = 2e-3
    enable_heterogeneity = True
    hetero_seed = 123
    hetero_corr_len_m = 0.008
    stiffness_cv = 0.10
    angle_hetero_deg = 3.5
    spacing_warp_amp_m= _rock().spacing_warp_amp_m
    use_hertz_contact_width = True
    nu_contact = None
    hertz_outer_iters_max = 6
    hertz_rel_b_tol = 2e-3
    platen_mu = 0.0
    grid_N = 401
    tol_rel_inner = 2e-6
    maxiter_inner = 800
    tol_rel_final = 1e-7
    maxiter_final = 2500
    use_precond = True
    precond_floor = 1e-6
    solver_mode = "pcg_then_bicgstab"
    plot_interpolation = "bicubic"
    disk_percentile_abs = 97.0
    colorbar_symmetric = True
    colorbar_nbins = 9
    colorbar_fmt = "%.2f"
    contour_levels = 16
    contour_sigma_pix = 1.35
    contour_corner_mask = True
    contour_linewidth = 0.55
    contour_alpha = 0.60
    show_band_lines = True
    show_band_edges = True
    radial_exclude_boundary_fraction = 0.05
    radial_scatter_max_points = 60000
    radial_bins = 70
    radial_seed = 0
    diameter = float(df.loc[index_number, "Diameter_mm"]) * 1e-3
    thickness = float(df.loc[index_number, "Thickness_mm"]) * 1e-3
    E_in = float(df.loc[index_number, "Modulus_of_Elasticity"])
    nu12 = float(df.loc[index_number, "Poisson_Ratio"])
    alpha = float(df.loc[index_number, "Radians"])
    G_in = float(df.loc[index_number, "Shear_Modulus"]) if "Shear_Modulus" in df.columns else np.nan
    R = diameter / 2.0
    if nu_contact is None:
        nu_contact = nu12
    E_ref_mpa = to_MPa_modulus(E_in)
    G_ref_mpa = to_MPa_modulus(G_in) if np.isfinite(G_in) else np.nan
    E1_base, E2_base, nu12, G12_base = orthotropic_from_UCS_anisotropy(
        E_ref_mpa, nu12, G_ref_mpa, ucs_anisotropy_ratio, p=p_link
    )
    xlin = np.linspace(-R, R, grid_N)
    ylin = np.linspace(-R, R, grid_N)
    h = float(xlin[1] - xlin[0])
    X, Y = np.meshgrid(xlin, ylin)
    mask = (X*X + Y*Y) <= R*R
    mask_idx = np.flatnonzero(mask.ravel())
    Xf = X.ravel()
    Yf = Y.ravel()
    mask_x = np.zeros_like(mask, dtype=bool)
    mask_x[:, :-1] = mask[:, :-1] & mask[:, 1:]
    mask_y = np.zeros_like(mask, dtype=bool)
    mask_y[:-1, :] = mask[:-1, :] & mask[1:, :]
    phase_plot = 0.5*spacing_m if phase_mode == "center_safe" else float(phase_shift_m)
    band_angle0 = alpha if (band_angle_deg is None) else np.deg2rad(float(band_angle_deg))
    if enable_heterogeneity:
        fE = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=hetero_seed + 1)
        fA = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=hetero_seed + 2)
        fW = gaussian_random_field(X.shape, dx=h, corr_len_m=hetero_corr_len_m, seed=hetero_seed + 3)

        sigma_ln = np.sqrt(np.log(1.0 + float(stiffness_cv)**2)) if stiffness_cv > 0 else 0.0
        stiff_factor = np.exp(sigma_ln * fE)

        angle_local = band_angle0 + np.deg2rad(float(angle_hetero_deg)) * fA
        warp_m = float(spacing_warp_amp_m) * fW
    else:
        stiff_factor = np.ones_like(X)
        angle_local = band_angle0 * np.ones_like(X)
        warp_m = np.zeros_like(X)
    stiff_factor[~mask] = 1.0
    angle_local[~mask] = band_angle0
    warp_m[~mask] = 0.0
    w_band = weak_band_weight_phasewarp(
        X, Y, angle_local=angle_local, spacing=spacing_m, warp_m=warp_m,
        band_halfwidth_m=band_halfwidth_m, phase=phase_plot, sharp_power=band_sharp_power
    )
    w_band[~mask] = 0.0
    t_band = 2.0 * float(band_halfwidth_m)
    if joint_kn_MPa_per_m is None or joint_ks_MPa_per_m is None:
        kn0, ks0 = kn_ks_from_min_fractions(E2_base, G12_base,
                                            band_min_E2_fraction, band_min_G12_fraction, t_band)
        joint_kn_MPa_per_m = kn0 if joint_kn_MPa_per_m is None else joint_kn_MPa_per_m
        joint_ks_MPa_per_m = ks0 if joint_ks_MPa_per_m is None else joint_ks_MPa_per_m
    E2_band = joint_kn_MPa_per_m * t_band
    G_band  = joint_ks_MPa_per_m * t_band
    E1_band = band_min_E1_fraction * E1_base
    Q11 = Q22 = Q12 = Q16 = Q26 = Q66 = None
    _precond_du = None
    _precond_dv = None
    _work = {
        "sxx_fx": np.zeros_like(X),
        "txy_fx": np.zeros_like(X),
        "syy_fy": np.zeros_like(X),
        "txy_fy": np.zeros_like(X),
        "tmpL":   np.zeros_like(X),
        "tmpD":   np.zeros_like(X),
        "divx":   np.zeros_like(X),
        "divy":   np.zeros_like(X),
    }
    Iseg, Jseg, xb, yb, nbx, nby, Lseg = boundary_segments_cut_cells(X, Y, mask, R, h)
    E_eff_MPa = float(np.sqrt(E1_base * E2_base))
    b_contact = 0.0010
    u_prev = None
    v_prev = None
    final_w_active = w_band.copy()
    final_b_contact = b_contact
    final_p0_unit_mpa = 1.0
    n_outer = int(hertz_outer_iters_max) if use_hertz_contact_width else 1
    for outer in range(n_outer):
        bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(b_contact)

        w_active = w_band.copy()
        w_prev = None

        n_fp = int(closure_fixed_point_iters_max) if enable_joint_closure else 1
        n_fp = max(n_fp, 1)

        # closure loop (fast)
        for it in range(n_fp):
            E1, E2, G12 = build_moduli_fields(w_active)
            update_Q_from_moduli(E1, E2, G12)

            u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev, tol_rel=tol_rel_inner, maxiter=maxiter_inner)
            u_prev, v_prev = u, v

            sxx0, syy0, txy0 = stresses(u, v)
            sxx0 = np.where(mask, sxx0, np.nan)
            syy0 = np.where(mask, syy0, np.nan)
            txy0 = np.where(mask, txy0, np.nan)

            ic = (grid_N - 1)//2
            jc = (grid_N - 1)//2
            st = 2
            sxx_center0 = float(np.nanmean(sxx0[ic-st:ic+st+1, jc-st:jc+st+1]))
            if (not np.isfinite(sxx_center0)) or abs(sxx_center0) < 1e-12:
                raise RuntimeError("Center σxx invalid/too small; contact too narrow or solver unstable.")

            sign_tmp = -1.0 if sxx_center0 < 0 else 1.0
            scale_tmp = float(target_sigma_xx_center_mpa) / abs(sxx_center0)

            if not enable_joint_closure:
                break

            sxx_sc = sign_tmp * scale_tmp * sxx0
            syy_sc = sign_tmp * scale_tmp * syy0
            txy_sc = sign_tmp * scale_tmp * txy0

            sigma_n = sigma_normal_to_plane(sxx_sc, syy_sc, txy_sc, plane_angle=angle_local)
            open_fac = band_open_factor_from_sigma_n(sigma_n, sigma0_mpa=closure_sigma0_mpa)

            w_new = w_band * (closed_compliance_fraction + (1.0 - closed_compliance_fraction) * open_fac)
            w_new[~mask] = 0.0

            if w_prev is not None:
                num = np.nanmax(np.abs(w_new[mask] - w_active[mask]))
                den = np.nanmax(np.abs(w_active[mask])) + 1e-30
                if (num / den) < closure_rel_change_tol:
                    w_active = w_new
                    break

            w_prev = w_active
            w_active = w_new

        # store current "best" state
        final_w_active = w_active.copy()
        final_b_contact = b_contact
        final_p0_unit_mpa = float(p0_unit_mpa)

        # Hertz update
        P_calib_N = abs(scale_tmp) * 1.0  # RHS normalized to 1 N
        if not use_hertz_contact_width:
            break

        b_old = b_contact
        b_new = hertz_contact_halfwidth(P_calib_N, thickness, R, E_eff_MPa, nu_contact)
        b_new = float(np.clip(b_new, 0.00015, 0.0060))
        b_contact = 0.55 * b_contact + 0.45 * b_new

        rel = abs(b_contact - b_old) / (abs(b_old) + 1e-30)
        if rel < hertz_rel_b_tol:
            break
    bx_rhs, by_rhs, p0_unit_mpa = build_rhs_contact_unit_load(final_b_contact)
    E1, E2, G12 = build_moduli_fields(final_w_active)
    update_Q_from_moduli(E1, E2, G12)
    u, v = solve_system(bx_rhs, by_rhs, u_prev, v_prev, tol_rel=tol_rel_final, maxiter=maxiter_final)
    sxx0, syy0, txy0 = stresses(u, v)
    sxx0 = np.where(mask, sxx0, np.nan)
    syy0 = np.where(mask, syy0, np.nan)
    txy0 = np.where(mask, txy0, np.nan)
    ic = (grid_N - 1)//2
    jc = (grid_N - 1)//2
    st = 2
    sxx_center0 = float(np.nanmean(sxx0[ic-st:ic+st+1, jc-st:jc+st+1]))
    sign = -1.0 if sxx_center0 < 0 else 1.0
    scale = float(target_sigma_xx_center_mpa) / (abs(sxx_center0) + 1e-30)
    sxx = sign * scale * sxx0
    syy = sign * scale * syy0
    txy = sign * scale * txy0
    s1, s3 = principal_stresses_2d(sxx, syy, txy)
    sigma1_plus = np.maximum(s1, 0.0)
    syy_minus = np.minimum(syy, 0.0)
    p_peak_mpa = abs(scale) * float(p0_unit_mpa)
    w_comp = float(target_sigma_xx_center_mpa) / (p_peak_mpa + 1e-30)
    w_comp = float(np.clip(w_comp, 0.0, 1.0))
    syy_cap = np.maximum(syy_minus, -p_peak_mpa)
    combined = sigma1_plus + w_comp * syy_cap
    print("\n=== COMBINED PLOT (IMPROVED) SUMMARY ===")
    print("Sign convention: + tension, - compression")
    print(f"D={diameter*1e3:.1f} mm, t={thickness*1e3:.1f} mm, spacing={spacing_m*1e3:.2f} mm")
    print(f"Hertz contact half-width b ≈ {final_b_contact*1e3:.3f} mm")
    print(f"Peak boundary pressure p_peak ≈ {p_peak_mpa:.3f} MPa")
    print(f"Compression weight w_comp = {w_comp:.6f}")
    print("Combined = σ1+ + w_comp * clamp(σyy−, -p_peak, 0)\n")
    _fft_cache = {}
    plot_combined_disk(
        combined,
        "Improved combined field: σ1+ + w_comp·clamp(σyy−, -p_peak, 0)",
        "Combined (MPa)"
    )
    plot_one_radial(X, Y, sigma1_plus, syy_minus)
