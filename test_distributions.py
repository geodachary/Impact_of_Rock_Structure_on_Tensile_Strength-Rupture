# crack_helpers.py
# -*- coding: utf-8 -*-
"""
Helpers for Brazilian disk crack modeling (Airy-fit driven).

✅ BASE: your WORKING graph-platen code (kept verbatim for the overlapping parts)
✅ PLUS: extra long-code features added around it:
   - local platen-to-platen stepper (physics/utility)
   - legacy center-start + mirror
   - optional DDM/LEFM path utilities (kept)

Key fixes (from working code, preserved):
- angle_wrap_pi is VECTORIZED (works on arrays) -> fixes TypeError.
- ratios_and_dirs_at_point accepts **_unused so extra kwargs never crash.
- crack_path_single_mirror_physics supports start_mode="graph_platen".

Requirements:
- geometry_helpers: angle_diff_periodic, _wrap_pi
- rotation_helpers: rot_to_material
- stress_helpers: stress_material_to_global, principal_from_components, eval_stress_field_material
- failure_mapping_helpers: failure_mode_map
- cracked_disk_ddm: _try_call_sif_two_tips, sif_two_tips_from_crack (DDM path optional)

IMPORTANT:
- Use the REVISED cracked_disk_ddm.py that does NOT import crack_helpers.py (avoid circular import).
"""

from __future__ import annotations

import numpy as np
import heapq

from geometry_helpers import angle_diff_periodic, _wrap_pi
from rotation_helpers import rot_to_material
from stress_helpers import stress_material_to_global, principal_from_components, eval_stress_field_material
from failure_mapping_helpers import failure_mode_map

# DDM/BEM solver (optional but kept)
from cracked_disk_ddm import _try_call_sif_two_tips, sif_two_tips_from_crack


# =============================================================================
# Vector-safe angle helpers (WORKING CODE - KEPT)
# =============================================================================

def angle_wrap_pi(a):
    """
    Wrap angle(s) to (-pi, pi]. Works for scalars AND numpy arrays.
    """
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi_half(a):
    """
    Wrap line-angle(s) to (-pi/2, pi/2]. Works for scalars/arrays.
    """
    a = np.asarray(a, dtype=float)
    return (a + np.pi/2.0) % np.pi - np.pi/2.0


def _align_line_direction(psi_target, psi_ref):
    """
    Align a LINE direction so psi and psi+pi are equivalent, picking the closest to psi_ref.
    Scalars only (used in local stepping; graph solver does not need it).
    """
    psi_target = _wrap_pi(float(psi_target))
    psi_ref = _wrap_pi(float(psi_ref))
    d1 = abs(angle_diff_periodic(psi_target, psi_ref))
    d2 = abs(angle_diff_periodic(psi_target + np.pi, psi_ref))
    return _wrap_pi(psi_target if d1 <= d2 else (psi_target + np.pi))


def _angle_mean_line(a, b, w):
    """
    Mean of LINE angles using doubled-angle averaging. Scalars only.
    """
    a = float(a); b = float(b); w = float(w)
    za = np.exp(2j * a)
    zb = np.exp(2j * b)
    z = (1.0 - w) * za + w * zb
    if abs(z) < 1e-20:
        return float(wrap_pi_half(a))
    return float(wrap_pi_half(0.5 * np.angle(z)))


# =============================================================================
# Sampler (WORKING CODE - KEPT)
# =============================================================================

class UniformGridSampler:
    def __init__(self, X, Y):
        self.xmin = float(X[0, 0]); self.xmax = float(X[0, -1])
        self.ymin = float(Y[0, 0]); self.ymax = float(Y[-1, 0])
        self.nx = int(X.shape[1]); self.ny = int(X.shape[0])
        self.dx = (self.xmax - self.xmin) / max(self.nx - 1, 1)
        self.dy = (self.ymax - self.ymin) / max(self.ny - 1, 1)

    def sample(self, F, x, y, fill=np.nan):
        x = float(x); y = float(y)
        if (x < self.xmin) or (x > self.xmax) or (y < self.ymin) or (y > self.ymax):
            return float(fill)

        fx = (x - self.xmin) / self.dx
        fy = (y - self.ymin) / self.dy
        j0 = int(np.floor(fx)); i0 = int(np.floor(fy))
        j1 = min(j0 + 1, self.nx - 1)
        i1 = min(i0 + 1, self.ny - 1)
        tx = fx - j0; ty = fy - i0

        f00 = F[i0, j0]; f10 = F[i0, j1]
        f01 = F[i1, j0]; f11 = F[i1, j1]

        if not all(np.isfinite(v) for v in [f00, f10, f01, f11]):
            vals = np.array([f00, f10, f01, f11], dtype=float)
            ok = np.isfinite(vals)
            return float(vals[ok][0]) if np.any(ok) else float(fill)

        return float((1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 +
                     (1-tx)*ty*f01 + tx*ty*f11)


# =============================================================================
# Stress evaluation (vectorized) (WORKING CODE - KEPT)
# =============================================================================

def stress_global_from_fit(xg, yg, R, alpha_const, fit):
    """
    Vectorized stress evaluation in global coordinates.
    xg, yg can be scalars or arrays.
    """
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    xm, ym = rot_to_material(xg, yg, float(alpha_const))
    sxx_m, syy_m, txy_m = eval_stress_field_material(
        xm, ym, float(R), fit["p1"], fit["p2"], fit["a1"], fit["a2"]
    )
    return stress_material_to_global(sxx_m, syy_m, txy_m, float(alpha_const))


# =============================================================================
# Pointwise ratios + utility (WORKING CODE - KEPT)
# =============================================================================

def ratios_and_dirs_at_point(
    x, y,
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12,
    **_unused
):
    Teff = sampler.sample(Teff_img, x, y, fill=np.nan)
    Coh = sampler.sample(Coh_img, x, y, fill=np.nan)
    Phi = sampler.sample(Phi_img, x, y, fill=np.nan)
    if (not np.isfinite(Teff)) or (not np.isfinite(Coh)) or (not np.isfinite(Phi)):
        return np.nan, np.nan, np.nan, np.nan

    sxx, syy, txy = stress_global_from_fit(x, y, R, alpha_const, fit)
    sxx = float(np.asarray(sxx)); syy = float(np.asarray(syy)); txy = float(np.asarray(txy))

    s1, _, th = principal_from_components(np.array([sxx]), np.array([syy]), np.array([txy]))
    s1 = float(s1[0]); th = float(th[0])

    weak_phi_arr = None
    if weak_phi is not None:
        weak_phi_arr = np.array([float(weak_phi)], dtype=float)

    _, Rt_eff, Rs_eff, beta_crit = failure_mode_map(
        np.array([sxx]), np.array([syy]), np.array([txy]), np.array([s1]),
        Tm=np.array([Teff]), Coh=np.array([Coh]), Phi=np.array([Phi]),
        alpha_const=float(alpha_const),
        basis="first",
        util_min=0.0,
        mixed_band=0.45,
        margin=0.0,
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=str(strength_model),
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi_arr,
        eps=float(eps)
    )

    Rt = float(Rt_eff[0])
    Rs = float(Rs_eff[0])
    bc = float(beta_crit[0])

    psi_t = _wrap_pi(th + np.pi/2.0)
    psi_s = _wrap_pi(bc + np.pi/2.0)
    return Rt, Rs, psi_t, psi_s


def util_at_point(
    x, y,
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    **failure_kwargs
):
    Rt, Rs, _, _ = ratios_and_dirs_at_point(
        x, y, R, alpha_const, fit,
        sampler, Teff_img, Coh_img, Phi_img,
        **failure_kwargs
    )
    if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
        return np.nan
    return float(max(Rt, Rs))


# =============================================================================
# Endpoint enforcement (WORKING CODE - KEPT)
# =============================================================================

def straighten_endcaps_parallel_to_load(xs, ys, R, beta_deg, extra_deg=6.0, ncap=14, power=2.0):
    xs2 = np.asarray(xs, float).copy()
    ys2 = np.asarray(ys, float).copy()
    N = len(xs2)
    if N < 6:
        return xs2, ys2

    beta = np.deg2rad(float(beta_deg) + float(extra_deg))
    ncap = int(max(3, min(int(ncap), N-1)))
    power = float(max(power, 0.5))

    def _do_end(end_idx):
        x_end = float(xs2[end_idx])
        y_end = float(ys2[end_idx])
        th = float(np.arctan2(y_end, x_end))
        center = (np.pi/2.0) if (y_end >= 0) else (-np.pi/2.0)
        if abs(float(angle_wrap_pi(th - center))) > beta:
            return

        if end_idx == 0:
            idxs = np.arange(0, ncap, dtype=int)
        else:
            idxs = np.arange(N-1, N-1-ncap, -1, dtype=int)

        s = np.linspace(1.0, 0.0, len(idxs))
        w = s**power
        xs2[idxs] = w * x_end + (1.0 - w) * xs2[idxs]

    _do_end(0)
    _do_end(-1)
    return xs2, ys2


def _extend_point_to_circle(p0, p1, R):
    x0,y0 = float(p0[0]), float(p0[1])
    x1,y1 = float(p1[0]), float(p1[1])
    vx,vy = x1-x0, y1-y0
    if vx*vx + vy*vy < 1e-20:
        rr = np.hypot(x1,y1) + 1e-30
        return (R*x1/rr, R*y1/rr)

    A = vx*vx + vy*vy
    B = 2.0*(x1*vx + y1*vy)
    C = x1*x1 + y1*y1 - R*R
    disc = B*B - 4*A*C
    if disc < 0:
        rr = np.hypot(x1,y1) + 1e-30
        return (R*x1/rr, R*y1/rr)

    t1 = (-B + np.sqrt(disc)) / (2*A)
    t2 = (-B - np.sqrt(disc)) / (2*A)
    t = max(t1, t2)
    return (x1 + t*vx, y1 + t*vy)


def enforce_platen_endpoints(xs, ys, R, beta_deg, snap_extra_deg=8.0, snap_to_platen_center=False):
    xs = np.asarray(xs, float).copy()
    ys = np.asarray(ys, float).copy()
    if len(xs) < 3:
        return xs, ys

    if np.hypot(xs[0], ys[0]) < 0.98*R:
        xs[0], ys[0] = _extend_point_to_circle((xs[1], ys[1]), (xs[0], ys[0]), R)
    if np.hypot(xs[-1], ys[-1]) < 0.98*R:
        xs[-1], ys[-1] = _extend_point_to_circle((xs[-2], ys[-2]), (xs[-1], ys[-1]), R)

    beta = np.deg2rad(float(beta_deg) + float(snap_extra_deg))
    for idx in [0, -1]:
        x, y = float(xs[idx]), float(ys[idx])
        th = float(np.arctan2(y, x))
        center = (np.pi/2.0) if (y >= 0) else (-np.pi/2.0)

        if snap_to_platen_center:
            xs[idx] = 0.0
            ys[idx] = float(R*np.sin(center))
        else:
            dth = float(angle_wrap_pi(th - center))
            dth = float(np.clip(dth, -beta, beta))
            th2 = center + dth
            xs[idx] = float(R*np.cos(th2))
            ys[idx] = float(R*np.sin(th2))

    return xs, ys


# =============================================================================
# Graph (Dijkstra) platen-to-platen solver (WORKING CODE - KEPT)
# =============================================================================

def _make_platen_masks(X, Y, R, beta_deg, extra_deg=8.0, r_frac=0.94):
    r = np.hypot(X, Y)
    inside = r <= (0.999*float(R))
    beta = np.deg2rad(float(beta_deg) + float(extra_deg))
    th = np.arctan2(Y, X)

    # IMPORTANT: angle_wrap_pi is vectorized -> no crash here
    top_arc = inside & (Y > 0) & (np.abs(angle_wrap_pi(th - np.pi/2.0)) <= beta) & (r >= r_frac*R)
    bot_arc = inside & (Y < 0) & (np.abs(angle_wrap_pi(th + np.pi/2.0)) <= beta) & (r >= r_frac*R)
    return inside, top_arc, bot_arc


def _compute_utility_grid(
    R, alpha_const, fit,
    X, Y,
    Teff_img, Coh_img, Phi_img,
    mixed_band=0.45,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,
    eps=1e-12
):
    ny, nx = X.shape
    U = np.full((ny, nx), np.nan, dtype=float)

    r = np.hypot(X, Y)
    inside = (r <= 0.999*float(R)) & np.isfinite(Teff_img) & np.isfinite(Coh_img) & np.isfinite(Phi_img)
    ii, jj = np.where(inside)
    if ii.size < 10:
        return U

    x = X[ii, jj].ravel()
    y = Y[ii, jj].ravel()

    sxx, syy, txy = stress_global_from_fit(x, y, float(R), float(alpha_const), fit)
    sxx = np.asarray(sxx, float).ravel()
    syy = np.asarray(syy, float).ravel()
    txy = np.asarray(txy, float).ravel()

    s1, _, _ = principal_from_components(sxx, syy, txy)

    Tm = np.asarray(Teff_img[ii, jj], float).ravel()
    Coh = np.asarray(Coh_img[ii, jj], float).ravel()
    Phi = np.asarray(Phi_img[ii, jj], float).ravel()

    weak_phi_arr = None
    if weak_phi is not None:
        weak_phi_arr = np.full_like(Tm, float(weak_phi), dtype=float)

    _, Rt_eff, Rs_eff, _ = failure_mode_map(
        sxx, syy, txy, s1,
        Tm=Tm, Coh=Coh, Phi=Phi,
        alpha_const=float(alpha_const),
        basis="first",
        util_min=0.0,
        mixed_band=float(mixed_band),
        margin=0.0,
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=str(strength_model),
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi_arr,
        eps=float(eps)
    )

    Rt_eff = np.asarray(Rt_eff, float)
    Rs_eff = np.asarray(Rs_eff, float)

    # Slightly downweight shear when tensile is present -> encourages Brazilian split
    rtmax = np.nanmax(Rt_eff) if np.any(np.isfinite(Rt_eff)) else 0.0
    rsmax = np.nanmax(Rs_eff) if np.any(np.isfinite(Rs_eff)) else 0.0
    shear_weight = 0.85 if (rtmax >= 0.55*rsmax) else 1.0

    U[ii, jj] = np.maximum(Rt_eff, shear_weight * Rs_eff)
    return U


def crack_path_platen_to_platen_graph(
    R, alpha_const, fit,
    X, Y, Teff_img, Coh_img, Phi_img,
    platen_beta_deg=10.0,
    platen_snap_extra_deg=10.0,
    snap_to_platen_center=False,

    mixed_band=0.45,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,

    axis_weight=0.22,
    axis_sigma_frac=0.18,
    rim_weight=0.10,
    rim_power=6.0,

    smooth_passes=10,
    smooth_lambda=0.30,

    u_floor=1e-3,
    eps=1e-12,
    **_unused
):
    R = float(R)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    inside, startM, goalM = _make_platen_masks(
        X, Y, R,
        beta_deg=float(platen_beta_deg),
        extra_deg=float(platen_snap_extra_deg),
        r_frac=0.94
    )

    U = _compute_utility_grid(
        R=float(R), alpha_const=float(alpha_const), fit=fit,
        X=X, Y=Y,
        Teff_img=Teff_img, Coh_img=Coh_img, Phi_img=Phi_img,
        mixed_band=float(mixed_band),
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=strength_model,
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi,
        eps=float(eps)
    )

    valid = inside & np.isfinite(U)
    start = valid & startM
    goal = valid & goalM

    if (np.count_nonzero(start) < 3) or (np.count_nonzero(goal) < 3):
        xs = np.array([0.0, 0.0], float)
        ys = np.array([0.97*R, -0.97*R], float)
        xs, ys = enforce_platen_endpoints(xs, ys, R, beta_deg=platen_beta_deg,
                                          snap_extra_deg=platen_snap_extra_deg,
                                          snap_to_platen_center=snap_to_platen_center)
        return {"xs": xs, "ys": ys, "psi0": -np.pi/2, "model": "graph_fallback"}

    ny, nx = U.shape
    dx = float(X[0,1] - X[0,0]) if nx > 1 else (2*R/max(nx,1))
    dy = float(Y[1,0] - Y[0,0]) if ny > 1 else (2*R/max(ny,1))

    sigma = float(axis_sigma_frac) * R + 1e-30
    rnorm = np.hypot(X, Y) / (R + 1e-30)

    axis_pen = 1.0 + float(axis_weight) * (X / sigma)**2
    rim_pen = 1.0 + float(rim_weight) * (rnorm**float(rim_power))
    pen = axis_pen * rim_pen

    Uc = np.maximum(U, float(u_floor))
    res = (1.0 / (Uc + 1e-30)) * pen
    res[~valid] = np.inf

    N = ny * nx
    dist = np.full(N, np.inf, dtype=float)
    prev = np.full(N, -1, dtype=int)
    visited = np.zeros(N, dtype=bool)

    start_idx = np.flatnonzero(start.ravel())
    goal_idx = np.flatnonzero(goal.ravel())
    goal_set = set(goal_idx.tolist())

    heap = []
    for si in start_idx:
        dist[int(si)] = 0.0
        heapq.heappush(heap, (0.0, int(si)))

    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1,-1), (-1, 1), (1,-1), (1, 1)]

    def lin(i,j): return i*nx + j

    goal_hit = -1
    while heap:
        dcur, uidx = heapq.heappop(heap)
        if visited[uidx]:
            continue
        visited[uidx] = True

        if uidx in goal_set:
            goal_hit = uidx
            break

        if dcur > dist[uidx]:
            continue

        i = uidx // nx
        j = uidx - i*nx

        for di,dj in neigh:
            ii = i + di
            jj = j + dj
            if (ii < 0) or (ii >= ny) or (jj < 0) or (jj >= nx):
                continue
            if not valid[ii, jj]:
                continue
            vidx = lin(ii, jj)

            step = np.hypot(dj*dx, di*dy)
            w = 0.5*(res[i,j] + res[ii,jj]) * step
            nd = dcur + w
            if nd < dist[vidx]:
                dist[vidx] = nd
                prev[vidx] = uidx
                heapq.heappush(heap, (nd, vidx))

    if goal_hit < 0:
        cand = np.flatnonzero((valid & (Y < 0)).ravel())
        goal_hit = int(cand[np.nanargmin(dist[cand])]) if cand.size else int(start_idx[0])

    # backtrack
    path = []
    cur = int(goal_hit)
    for _ in range(N):
        path.append(cur)
        cur = int(prev[cur])
        if cur < 0:
            break
    path = path[::-1]

    xs = np.array([X.flat[k] for k in path], dtype=float)
    ys = np.array([Y.flat[k] for k in path], dtype=float)

    # smooth (keep endpoints fixed)
    if len(xs) >= 5 and smooth_passes > 0:
        for _ in range(int(smooth_passes)):
            xnew = xs.copy()
            ynew = ys.copy()
            xnew[1:-1] = xs[1:-1] + float(smooth_lambda) * (0.5*(xs[0:-2] + xs[2:]) - xs[1:-1])
            ynew[1:-1] = ys[1:-1] + float(smooth_lambda) * (0.5*(ys[0:-2] + ys[2:]) - ys[1:-1])

            rr = np.hypot(xnew, ynew)
            mask = rr > 0.999*R
            xnew[mask] *= (0.999*R) / (rr[mask] + 1e-30)
            ynew[mask] *= (0.999*R) / (rr[mask] + 1e-30)
            xs, ys = xnew, ynew

    xs, ys = enforce_platen_endpoints(
        xs, ys, R,
        beta_deg=float(platen_beta_deg),
        snap_extra_deg=float(platen_snap_extra_deg),
        snap_to_platen_center=bool(snap_to_platen_center),
    )

    xs, ys = straighten_endcaps_parallel_to_load(
        xs, ys, R,
        beta_deg=float(platen_beta_deg),
        extra_deg=6.0, ncap=14, power=2.0
    )

    psi0 = float(np.arctan2(ys[min(1,len(ys)-1)] - ys[0], xs[min(1,len(xs)-1)] - xs[0]))
    return {"xs": xs, "ys": ys, "psi0": psi0, "model": "physics_graph"}


# =============================================================================
# EXTRA (from long code): helpers for local stepping + legacy + DDM
# =============================================================================

def _clamp_outward(tip_x, tip_y, psi, eps_ang=1e-4):
    """Limit direction so it does not point too far inward (legacy center-start)."""
    r2 = float(tip_x)*float(tip_x) + float(tip_y)*float(tip_y)
    if r2 < 1e-16:
        return _wrap_pi(psi)
    theta_r = np.arctan2(tip_y, tip_x)
    d = angle_diff_periodic(psi, theta_r)
    lim = (np.pi/2.0) - float(eps_ang)
    if abs(d) > lim:
        psi = _wrap_pi(theta_r + np.sign(d)*lim)
    return _wrap_pi(psi)


def _seg_intersect(p1, p2, q1, q2, eps=1e-12):
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def onseg(a,b,c):
        return (min(a[0],b[0])-eps <= c[0] <= max(a[0],b[0])+eps) and \
               (min(a[1],b[1])-eps <= c[1] <= max(a[1],b[1])+eps)

    o1, o2 = orient(p1,p2,q1), orient(p1,p2,q2)
    o3, o4 = orient(q1,q2,p1), orient(q1,q2,p2)

    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if abs(o1) <= eps and onseg(p1,p2,q1): return True
    if abs(o2) <= eps and onseg(p1,p2,q2): return True
    if abs(o3) <= eps and onseg(q1,q2,p1): return True
    if abs(o4) <= eps and onseg(q1,q2,p2): return True
    return False


def _on_platen_arc(x, y, beta, top=True):
    th = np.arctan2(y, x)
    center = (np.pi/2.0) if top else (-np.pi/2.0)
    return abs(float(angle_wrap_pi(th - center))) <= beta


def _pick_start_on_top_platen(
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    beta_deg,
    r_frac=0.97,
    n=61,
    extra_deg=6.0,
    **failure_kwargs
):
    beta = np.deg2rad(float(beta_deg) + float(extra_deg))
    ths = np.linspace(np.pi/2.0 - beta, np.pi/2.0 + beta, int(n))
    rr = float(r_frac) * float(R)

    best = (-np.inf, 0.0, rr)
    for th in ths:
        x = rr*np.cos(th); y = rr*np.sin(th)
        u = util_at_point(x, y, R, alpha_const, fit, sampler, Teff_img, Coh_img, Phi_img, **failure_kwargs)
        if np.isfinite(u) and (u > best[0]):
            best = (float(u), float(x), float(y))
    return best[1], best[2]


def crack_path_platen_to_platen_physics(
    R, alpha_const, fit,
    X, Y, Teff_img, Coh_img, Phi_img,
    ds_frac=0.005,
    max_steps=5000,
    mixed_band=0.45,
    relax=0.75,
    n_theta_mc=361,
    fail_margin=0.0,
    max_turn_deg=14.0,
    hysteresis=1.30,
    target_smooth=0.55,
    crack_path_util_min=0.98,
    seg_check_n=7,
    ds_retry_factors=(1.0, 0.80, 0.65, 0.50, 0.35, 0.25, 0.18),
    cand_n=51,
    target_penalty=0.22,
    turn_penalty=0.22,

    inward_r0=0.85,
    inward_cos_max=-0.05,

    goal_weight=0.55,
    goal_power=1.4,

    axis_weight=0.08,
    axis_sigma_frac=0.22,

    platen_beta_deg=10.0,
    platen_snap_extra_deg=8.0,
    snap_to_platen_center=False,

    platen_endcap_n=14,
    platen_endcap_power=2.0,
    platen_endcap_extra_deg=6.0,

    weak_plane_prefer_path=True,
    weak_plane_tfrac_min=0.70,

    self_intersection_stop=True,
    eps=1e-12,
    **failure_kwargs
):
    sampler = UniformGridSampler(X, Y)
    failure_kwargs = dict(failure_kwargs)
    failure_kwargs.setdefault("n_theta_mc", n_theta_mc)

    # start point on top platen
    sx, sy = _pick_start_on_top_platen(
        R, alpha_const, fit, sampler, Teff_img, Coh_img, Phi_img,
        beta_deg=float(platen_beta_deg),
        r_frac=0.97,
        n=61,
        extra_deg=float(platen_snap_extra_deg),
        **failure_kwargs
    )

    goal_pt = np.array([0.0, -0.97*float(R)], dtype=float)

    ds0 = float(ds_frac) * float(R)
    m = float(fail_margin)

    psi = float(np.arctan2(goal_pt[1] - sy, goal_pt[0] - sx))
    psi0 = float(psi)

    xs = [float(sx)]
    ys = [float(sy)]

    cap = np.deg2rad(float(max_turn_deg))
    hys = float(max(hysteresis, 1.0))
    seg_check_n = int(max(seg_check_n, 3))
    seg_t = np.linspace(0.15, 1.0, seg_check_n)
    cand_n = int(max(cand_n, 9))

    psi_target_prev = psi
    drive_mode_prev = None

    sigx = float(axis_sigma_frac) * float(R) + 1e-12
    beta_arc = np.deg2rad(float(platen_beta_deg) + float(platen_snap_extra_deg))

    for step_i in range(int(max_steps)):
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
        r = float(np.hypot(tip_x, tip_y))
        rn = r / (float(R) + eps)

        if (rn >= 0.92) and (tip_y < 0.0) and _on_platen_arc(tip_x, tip_y, beta_arc, top=False):
            break

        Rt, Rs, psi_t, psi_s = ratios_and_dirs_at_point(
            tip_x, tip_y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            **failure_kwargs
        )
        if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
            break

        if (Rt < 1.0 + m) and (Rs < 1.0 + m):
            if rn >= 0.25:
                break

        psi_t = _align_line_direction(psi_t, psi)
        psi_s = _align_line_direction(psi_s, psi)

        psi_wp = _align_line_direction(_wrap_pi(alpha_const), psi)
        tfrac = float(Rt / (Rt + Rs + eps))

        rel = abs(Rt - Rs) / max(max(Rt, Rs), eps)
        if rel <= float(mixed_band):
            drive_mode = "mixed"
        else:
            drive_mode = "tensile" if (Rt >= Rs) else "shear"

        if drive_mode_prev is not None and drive_mode != drive_mode_prev and drive_mode != "mixed":
            if drive_mode_prev == "tensile" and not (Rs > Rt * hys):
                drive_mode = drive_mode_prev
            if drive_mode_prev == "shear" and not (Rt > Rs * hys):
                drive_mode = drive_mode_prev
        drive_mode_prev = drive_mode

        if drive_mode == "tensile":
            psi_target = psi_t
        elif drive_mode == "shear":
            psi_target = _angle_mean_line(psi_s, psi_t, 0.25 + 0.25*tfrac)
        else:
            w = float(np.clip(0.40 + 0.55*tfrac, 0.40, 0.95))
            psi_target = _angle_mean_line(psi_s, psi_t, w)

        if weak_plane_prefer_path and str(failure_kwargs.get("strength_model","")).lower() == "weak_plane":
            if tfrac >= float(weak_plane_tfrac_min):
                if abs(angle_diff_periodic(psi_wp, psi_target)) < np.deg2rad(12.0):
                    psi_target = psi_wp

        psi_goal = float(np.arctan2(goal_pt[1] - tip_y, goal_pt[0] - tip_x))
        dgoal = float(np.hypot(goal_pt[0] - tip_x, goal_pt[1] - tip_y))
        gsoft = float(np.clip(1.0 - dgoal/(2.0*float(R) + eps), 0.0, 1.0))
        wg = float(goal_weight) * (gsoft ** float(goal_power))
        psi_target = _angle_mean_line(psi_target, psi_goal, wg)

        psi_target = _angle_mean_line(psi_target_prev, psi_target, float(target_smooth))
        psi_target_prev = psi_target

        dpsi0 = float(np.clip(angle_diff_periodic(psi_target, psi), -cap, cap))
        psi_center = _wrap_pi(psi + float(relax)*dpsi0)

        cand = psi + np.linspace(-cap, cap, cand_n)
        cand = np.array([_wrap_pi(ci) for ci in cand], dtype=float)

        best = None
        for f in ds_retry_factors:
            ds = ds0 * float(f)
            best = None

            for pc in cand:
                pc = _align_line_direction(pc, psi)

                if rn >= float(inward_r0):
                    er = np.array([tip_x, tip_y], dtype=float) / (r + 1e-30)
                    v = np.array([np.cos(pc), np.sin(pc)], dtype=float)
                    if float(v.dot(er)) > float(inward_cos_max):
                        continue

                nx = tip_x + ds*np.cos(pc)
                ny = tip_y + ds*np.sin(pc)
                if nx*nx + ny*ny >= (0.999*float(R))**2:
                    continue

                utils = []
                ok = True
                for tt in seg_t:
                    xi = tip_x + (tt*ds)*np.cos(pc)
                    yi = tip_y + (tt*ds)*np.sin(pc)
                    if xi*xi + yi*yi >= (0.999*float(R))**2:
                        ok = False
                        break
                    u = util_at_point(
                        xi, yi, R, alpha_const, fit,
                        sampler, Teff_img, Coh_img, Phi_img,
                        **failure_kwargs
                    )
                    if (not np.isfinite(u)) or (u < float(crack_path_util_min)):
                        ok = False
                        break
                    utils.append(u)

                if not ok:
                    continue

                u_avg = float(np.mean(utils))
                dturn = abs(angle_diff_periodic(pc, psi))
                dtarget = abs(angle_diff_periodic(pc, psi_center))
                xpen = float(axis_weight) * (nx/(sigx + 1e-30))**2

                score = u_avg \
                        - float(turn_penalty)*(dturn/max(cap,1e-9))**2 \
                        - float(target_penalty)*(dtarget/max(cap,1e-9))**2 \
                        - xpen

                v = np.array([np.cos(pc), np.sin(pc)], dtype=float)
                toG = goal_pt - np.array([tip_x, tip_y], dtype=float)
                toGn = float(np.hypot(toG[0], toG[1])) + 1e-30
                toG /= toGn
                score += float(wg) * float(np.clip(v.dot(toG), -1.0, 1.0))

                if (best is None) or (score > best[0]):
                    best = (score, nx, ny, pc)

            if best is not None:
                break

        if best is None:
            break

        _, nx, ny, psi_new = best

        if self_intersection_stop and (len(xs) > 25) and (step_i % 2 == 0):
            p1 = (xs[-1], ys[-1])
            p2 = (nx, ny)
            for j in range(1, len(xs)-10):
                q1 = (xs[j-1], ys[j-1])
                q2 = (xs[j], ys[j])
                if _seg_intersect(p1, p2, q1, q2):
                    best = None
                    break
            if best is None:
                break

        xs.append(float(nx)); ys.append(float(ny))
        psi = float(psi_new)

    Xs = np.array(xs, float)
    Ys = np.array(ys, float)

    Xs, Ys = enforce_platen_endpoints(
        Xs, Ys, float(R),
        beta_deg=float(platen_beta_deg),
        snap_extra_deg=float(platen_snap_extra_deg),
        snap_to_platen_center=bool(snap_to_platen_center),
    )

    Xs, Ys = straighten_endcaps_parallel_to_load(
        Xs, Ys, float(R),
        beta_deg=float(platen_beta_deg),
        extra_deg=float(platen_endcap_extra_deg),
        ncap=int(platen_endcap_n),
        power=float(platen_endcap_power),
    )

    return {"xs": Xs, "ys": Ys, "psi0": float(psi0), "model": "physics_platen_to_platen_local"}


def pick_initial_psi0_by_first(
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    n_dir=181,
    eps=1e-12,
    **failure_kwargs
):
    r0 = 0.01 * float(R)
    psi_grid = np.linspace(0.0, np.pi, int(n_dir), endpoint=True)

    best = (-np.inf, np.pi/2.0)
    for psi in psi_grid:
        x = r0*np.cos(psi)
        y = r0*np.sin(psi)
        u = util_at_point(
            x, y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            **failure_kwargs
        )
        if not np.isfinite(u):
            continue
        if u > best[0]:
            best = (float(u), float(psi))
    return float(best[1])


# =============================================================================
# Dispatcher (WORKING SIGNATURE, extended modes supported via start_mode)
# =============================================================================

def crack_path_single_mirror_physics(
    R, alpha_const, fit,
    X, Y, Teff_img, Coh_img, Phi_img,
    start_mode="graph_platen",
    platen_beta_deg=10.0,
    platen_snap_extra_deg=10.0,
    snap_to_platen_center=False,

    mixed_band=0.45,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.4,
    weak_C_ratio=0.6,
    weak_phi=None,

    axis_weight=0.22,
    axis_sigma_frac=0.18,
    rim_weight=0.10,
    rim_power=6.0,

    smooth_passes=10,
    smooth_lambda=0.30,

    **kwargs
):
    """
    start_mode options:
      - "graph_platen" (default): robust global Dijkstra platen-to-platen (your working code)
      - "platen_to_platen": local stepper (extra; uses kwargs for its knobs)
      - "center_mirror": legacy center-start + mirror (extra; uses kwargs for its knobs)
    """
    mode = str(start_mode).lower().strip()

    if mode in ("graph", "graph_platen", "platen_graph", "platen_to_platen_graph"):
        return crack_path_platen_to_platen_graph(
            R=R, alpha_const=alpha_const, fit=fit,
            X=X, Y=Y, Teff_img=Teff_img, Coh_img=Coh_img, Phi_img=Phi_img,
            platen_beta_deg=platen_beta_deg,
            platen_snap_extra_deg=platen_snap_extra_deg,
            snap_to_platen_center=snap_to_platen_center,
            mixed_band=mixed_band,
            n_theta_mc=n_theta_mc,
            mc_compression_only=mc_compression_only,
            mc_sigma_comp_min=mc_sigma_comp_min,
            strength_model=strength_model,
            weak_T_ratio=weak_T_ratio,
            weak_C_ratio=weak_C_ratio,
            weak_phi=weak_phi,
            axis_weight=axis_weight,
            axis_sigma_frac=axis_sigma_frac,
            rim_weight=rim_weight,
            rim_power=rim_power,
            smooth_passes=smooth_passes,
            smooth_lambda=smooth_lambda,
        )

    if mode in ("platen", "platen_to_platen", "platens", "local_platen"):
        return crack_path_platen_to_platen_physics(
            R=float(R), alpha_const=float(alpha_const), fit=fit,
            X=X, Y=Y, Teff_img=Teff_img, Coh_img=Coh_img, Phi_img=Phi_img,
            ds_frac=float(kwargs.get("ds_frac", 0.005)),
            max_steps=int(kwargs.get("max_steps", 5000)),
            mixed_band=float(mixed_band),
            relax=float(kwargs.get("relax", 0.75)),
            n_theta_mc=int(n_theta_mc),
            fail_margin=float(kwargs.get("fail_margin", 0.0)),
            max_turn_deg=float(kwargs.get("max_turn_deg", 14.0)),
            hysteresis=float(kwargs.get("hysteresis", 1.30)),
            target_smooth=float(kwargs.get("target_smooth", 0.55)),
            crack_path_util_min=float(kwargs.get("crack_path_util_min", 0.98)),
            platen_beta_deg=float(platen_beta_deg),
            platen_snap_extra_deg=float(platen_snap_extra_deg),
            snap_to_platen_center=bool(snap_to_platen_center),
            strength_model=strength_model,
            weak_T_ratio=weak_T_ratio,
            weak_C_ratio=weak_C_ratio,
            weak_phi=weak_phi,
            mc_compression_only=mc_compression_only,
            mc_sigma_comp_min=mc_sigma_comp_min,
        )

    # ---------------- center-start + mirror (legacy) ----------------
    sampler = UniformGridSampler(X, Y)
    failure_kwargs = dict(kwargs)
    failure_kwargs.setdefault("n_theta_mc", n_theta_mc)
    failure_kwargs.setdefault("mc_compression_only", mc_compression_only)
    failure_kwargs.setdefault("mc_sigma_comp_min", mc_sigma_comp_min)
    failure_kwargs.setdefault("strength_model", strength_model)
    failure_kwargs.setdefault("weak_T_ratio", weak_T_ratio)
    failure_kwargs.setdefault("weak_C_ratio", weak_C_ratio)
    failure_kwargs.setdefault("weak_phi", weak_phi)

    psi0 = pick_initial_psi0_by_first(
        R, alpha_const, fit, sampler,
        Teff_img, Coh_img, Phi_img,
        n_dir=int(kwargs.get("n_dir", 181)),
        eps=float(kwargs.get("eps", 1e-12)),
        **failure_kwargs
    )

    ds0 = float(kwargs.get("ds_frac", 0.005)) * float(R)
    a0 = float(kwargs.get("a0_frac", 0.008)) * float(R)
    m = float(kwargs.get("fail_margin", 0.0))

    xs = [0.0, a0*np.cos(psi0)]
    ys = [0.0, a0*np.sin(psi0)]
    psi = float(psi0)

    cap = np.deg2rad(float(kwargs.get("max_turn_deg", 12.0)))
    relax = float(kwargs.get("relax", 0.75))
    hysteresis = float(kwargs.get("hysteresis", 1.35))
    hys = float(max(hysteresis, 1.0))
    target_smooth = float(kwargs.get("target_smooth", 0.55))
    crack_path_util_min = float(kwargs.get("crack_path_util_min", 0.98))
    self_intersection_stop = bool(kwargs.get("self_intersection_stop", True))
    enforce_outward = bool(kwargs.get("enforce_outward", True))
    ds_retry_factors = tuple(kwargs.get("ds_retry_factors", (1.0, 0.75, 0.55, 0.40, 0.30, 0.22, 0.15, 0.10)))
    cand_n = int(kwargs.get("cand_n", 41))
    seg_check_n = int(kwargs.get("seg_check_n", 7))
    target_penalty = float(kwargs.get("target_penalty", 0.25))
    turn_penalty = float(kwargs.get("turn_penalty", 0.25))

    cand_n = int(max(cand_n, 7))
    seg_check_n = int(max(seg_check_n, 3))
    seg_t = np.linspace(0.15, 1.0, seg_check_n)

    psi_target_prev = psi

    for step_i in range(int(kwargs.get("max_steps", 2500))):
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
        if tip_x*tip_x + tip_y*tip_y >= (0.999*float(R))**2:
            break

        Rt, Rs, psi_t, psi_s = ratios_and_dirs_at_point(
            tip_x, tip_y, R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            **failure_kwargs
        )
        if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
            break

        if (Rt < 1.0 + m) and (Rs < 1.0 + m):
            break

        psi_t = _align_line_direction(psi_t, psi)
        psi_s = _align_line_direction(psi_s, psi)

        rel = abs(Rt - Rs) / max(max(Rt, Rs), 1e-12)
        if rel <= float(mixed_band):
            w = Rt / (Rt + Rs + 1e-12)
            psi_target = _angle_mean_line(psi_s, psi_t, w)
        else:
            psi_target = psi_t if (Rt >= Rs) else psi_s

        psi_target = _angle_mean_line(psi_target_prev, psi_target, float(target_smooth))
        psi_target_prev = psi_target

        if enforce_outward:
            psi_target = _clamp_outward(tip_x, tip_y, psi_target)

        dpsi0 = float(np.clip(angle_diff_periodic(psi_target, psi), -cap, cap))
        psi_center = _wrap_pi(psi + float(relax)*dpsi0)
        if enforce_outward:
            psi_center = _clamp_outward(tip_x, tip_y, psi_center)

        cand = psi + np.linspace(-cap, cap, cand_n)
        cand = np.array([_wrap_pi(ci) for ci in cand], dtype=float)

        best = None
        for f in ds_retry_factors:
            ds = ds0 * float(f)
            best = None

            for pc in cand:
                pc = _align_line_direction(pc, psi)
                if enforce_outward:
                    pc = _clamp_outward(tip_x, tip_y, pc)

                nx = tip_x + ds*np.cos(pc)
                ny = tip_y + ds*np.sin(pc)
                if nx*nx + ny*ny >= (0.999*float(R))**2:
                    continue

                utils = []
                ok = True
                for tt in seg_t:
                    xi = tip_x + (tt*ds)*np.cos(pc)
                    yi = tip_y + (tt*ds)*np.sin(pc)
                    if xi*xi + yi*yi >= (0.999*float(R))**2:
                        ok = False
                        break
                    u = util_at_point(
                        xi, yi, R, alpha_const, fit,
                        sampler, Teff_img, Coh_img, Phi_img,
                        **failure_kwargs
                    )
                    if (not np.isfinite(u)) or (u < float(crack_path_util_min)):
                        ok = False
                        break
                    utils.append(u)

                if not ok:
                    continue

                u_avg = float(np.mean(utils))
                dturn = abs(angle_diff_periodic(pc, psi))
                dtarget = abs(angle_diff_periodic(pc, psi_center))
                score = u_avg \
                        - float(turn_penalty)*(dturn/max(cap,1e-9))**2 \
                        - float(target_penalty)*(dtarget/max(cap,1e-9))**2

                if (best is None) or (score > best[0]):
                    best = (score, nx, ny, pc)

            if best is not None:
                break

        if best is None:
            break

        _, nx, ny, psi_new = best

        if self_intersection_stop and (len(xs) > 25) and (step_i % 2 == 0):
            p1 = (xs[-1], ys[-1])
            p2 = (nx, ny)
            for j in range(1, len(xs)-10):
                q1 = (xs[j-1], ys[j-1])
                q2 = (xs[j], ys[j])
                if _seg_intersect(p1, p2, q1, q2):
                    best = None
                    break
            if best is None:
                break

        xs.append(float(nx)); ys.append(float(ny))
        psi = float(psi_new)

    xu = np.array(xs, float); yu = np.array(ys, float)
    Xs = np.concatenate([-xu[::-1], xu[1:]])
    Ys = np.concatenate([-yu[::-1], yu[1:]])

    Xs, Ys = enforce_platen_endpoints(
        Xs, Ys, float(R),
        beta_deg=float(platen_beta_deg),
        snap_extra_deg=float(platen_snap_extra_deg),
        snap_to_platen_center=bool(snap_to_platen_center),
    )

    Xs, Ys = straighten_endcaps_parallel_to_load(
        Xs, Ys, float(R),
        beta_deg=float(platen_beta_deg),
        extra_deg=float(kwargs.get("platen_endcap_extra_deg", 6.0)),
        ncap=int(kwargs.get("platen_endcap_n", 14)),
        power=float(kwargs.get("platen_endcap_power", 2.0)),
    )

    return {"xs": Xs, "ys": Ys, "psi0": float(psi0), "model": "physics_center_mirror"}


# =============================================================================
# DDM utilities kept (optional) (WORKING CODE - KEPT)
# =============================================================================

def _extract_tip_sifs(sif_res):
    if sif_res is None:
        return np.nan, np.nan, np.nan, np.nan
    if isinstance(sif_res, (tuple, list, np.ndarray)) and len(sif_res) >= 4:
        KI0, KII0, KI1, KII1 = sif_res[0], sif_res[1], sif_res[2], sif_res[3]
        return float(KI0), float(KII0), float(KI1), float(KII1)
    if isinstance(sif_res, dict):
        for keys in [
            ("KI0","KII0","KI1","KII1"),
            ("KI_tip0","KII_tip0","KI_tip1","KII_tip1"),
            ("KI_left","KII_left","KI_right","KII_right"),
        ]:
            if all(k in sif_res for k in keys):
                KI0, KII0, KI1, KII1 = (sif_res[keys[0]], sif_res[keys[1]], sif_res[keys[2]], sif_res[keys[3]])
                return float(KI0), float(KII0), float(KI1), float(KII1)
    raise TypeError(f"Unsupported sif result format: {type(sif_res)}")


# --- Optional DDM path (extra; unchanged logic) ---

def max_hoop_kink_angle(KI, KII, eps=1e-18):
    KI = float(KI); KII = float(KII)
    if abs(KII) < eps:
        return 0.0
    root = np.sqrt(KI*KI + 8.0*KII*KII)
    return 2.0 * np.arctan2(2.0*KII, KI + root)


def pick_initial_psi0_by_ddm(
    R, alpha_const, fit,
    E1, E2, nu12, G12,
    a0_frac=0.008,
    n_dir=181,
    enforce_outward=True,
    ddm_sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    **ddm_kwargs
):
    a0 = float(a0_frac) * float(R)
    psi_grid = np.linspace(0.0, np.pi, int(n_dir), endpoint=True)

    best = (-np.inf, np.pi/2.0)
    for psi in psi_grid:
        xs = np.array([0.0, a0*np.cos(psi)], dtype=float)
        ys = np.array([0.0, a0*np.sin(psi)], dtype=float)

        psi_use = float(psi)
        if enforce_outward:
            psi_use = _clamp_outward(xs[-1], ys[-1], psi_use)

        sif_res = _try_call_sif_two_tips(
            sif_two_tips_from_crack,
            xs, ys,
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=fit,
            eval_stress_field_material=eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
            **ddm_kwargs
        )
        KI0, KII0, KI1, KII1 = _extract_tip_sifs(sif_res)
        Keq = float(np.sqrt(KI1*KI1 + KII1*KII1))
        if np.isfinite(Keq) and (Keq > best[0]):
            best = (Keq, psi_use)

    return float(best[1])


def crack_path_single_mirror_ddm(
    R, alpha_const, fit,
    E1, E2, nu12, G12,
    ds_frac=0.005,
    a0_frac=0.008,
    max_steps=2500,
    max_turn_deg=12.0,
    relax=0.85,
    enforce_outward=True,
    self_intersection_stop=True,
    kmin_frac=0.02,
    ddm_sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    platen_beta_deg=10.0,
    platen_endcap_n=14,
    platen_endcap_power=2.0,
    platen_endcap_extra_deg=6.0,
    **ddm_kwargs
):
    psi0 = pick_initial_psi0_by_ddm(
        R, alpha_const, fit,
        E1, E2, nu12, G12,
        a0_frac=a0_frac,
        n_dir=181,
        enforce_outward=enforce_outward,
        ddm_sample_rs=ddm_sample_rs,
        cod_offset=cod_offset,
        **ddm_kwargs
    )

    ds0 = float(ds_frac) * float(R)
    a0 = float(a0_frac) * float(R)
    cap = np.deg2rad(float(max_turn_deg))

    xs = [0.0, a0*np.cos(psi0)]
    ys = [0.0, a0*np.sin(psi0)]
    psi = float(psi0)

    KI_hist = []
    KII_hist = []
    Keq_hist = []
    Kref = None

    for _ in range(int(max_steps)):
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
        if tip_x*tip_x + tip_y*tip_y >= (0.999*float(R))**2:
            break

        sif_res = _try_call_sif_two_tips(
            sif_two_tips_from_crack,
            np.array(xs, float), np.array(ys, float),
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=fit,
            eval_stress_field_material=eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
            **ddm_kwargs
        )
        KI0, KII0, KI, KII = _extract_tip_sifs(sif_res)
        Keq = float(np.sqrt(KI*KI + KII*KII))

        KI_hist.append(float(KI))
        KII_hist.append(float(KII))
        Keq_hist.append(float(Keq))

        if not np.isfinite(Keq):
            break
        if Kref is None:
            Kref = max(Keq, 1e-30)
        if Keq < float(kmin_frac) * float(Kref):
            break

        dtheta = float(max_hoop_kink_angle(KI, KII))
        dtheta = float(np.clip(dtheta, -cap, cap))

        psi_target = _wrap_pi(psi + dtheta)
        if enforce_outward:
            psi_target = _clamp_outward(tip_x, tip_y, psi_target)

        dpsi = angle_diff_periodic(psi_target, psi)
        dpsi = float(np.clip(dpsi, -cap, cap))

        psi_new = _wrap_pi(psi + float(relax)*dpsi)
        if enforce_outward:
            psi_new = _clamp_outward(tip_x, tip_y, psi_new)

        nx = tip_x + ds0*np.cos(psi_new)
        ny = tip_y + ds0*np.sin(psi_new)

        if nx*nx + ny*ny >= (0.999*float(R))**2:
            break

        if self_intersection_stop and (len(xs) > 25):
            p1, p2 = (xs[-1], ys[-1]), (nx, ny)
            for j in range(1, len(xs)-10):
                if _seg_intersect(p1, p2, (xs[j-1], ys[j-1]), (xs[j], ys[j])):
                    nx = xs[-1]; ny = ys[-1]
                    break

        xs.append(float(nx)); ys.append(float(ny))
        psi = float(psi_new)

    xu = np.array(xs, float); yu = np.array(ys, float)
    Xs = np.concatenate([-xu[::-1], xu[1:]])
    Ys = np.concatenate([-yu[::-1], yu[1:]])

    Xs, Ys = straighten_endcaps_parallel_to_load(
        Xs, Ys, R=float(R),
        beta_deg=float(platen_beta_deg),
        extra_deg=float(platen_endcap_extra_deg),
        ncap=int(platen_endcap_n),
        power=float(platen_endcap_power),
    )

    return {
        "xs": Xs,
        "ys": Ys,
        "psi0": float(psi0),
        "KI_tip": np.array(KI_hist, float),
        "KII_tip": np.array(KII_hist, float),
        "Keq_tip": np.array(Keq_hist, float),
        "model": "ddm",
    }
