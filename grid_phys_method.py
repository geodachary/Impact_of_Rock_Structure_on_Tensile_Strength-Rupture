#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
grid_phys_method.py

Improved grid-phys crack surrogate:
- Dijkstra on an "allowed" failure-utilization band
- direction guidance uses LINE angle diff (psi == psi+pi)
- center preference uses clearance distance to band edge (medial-axis style)
- NEW: symmetry-axis preference (penalize |x-axis_x0|)
- NEW: optional spatial smoothing of psi_pref field (line-aware)
- NEW: optional confidence scaling for direction penalty (psi_conf_img)
- smoothing + snap-back after smoothing

Dependencies:
- angle_diff_periodic, _wrap_pi imported from bound_all_helpers (or geometry_helpers)
"""

from __future__ import annotations

import heapq
import numpy as np

# Use your aggregator (recommended)
from geometry_helpers import angle_diff_periodic, _wrap_pi
from stress_helpers import stress_material_to_global, principal_from_components, eval_stress_field_material


# ==========================================================
# Helpers (GRID_PHYS only)
# ==========================================================
def _wrap_pi_half(a: float) -> float:
    """Wrap a *line* orientation to [-pi/2, pi/2)."""
    return ((a + np.pi / 2) % np.pi) - np.pi / 2


def angle_diff_line(a: float, b: float) -> float:
    """Smallest angular difference when direction is a LINE: b ≡ b + π."""
    da = abs(angle_diff_periodic(a, b))
    db = abs(angle_diff_periodic(a, b + np.pi))
    return float(min(da, db))


def _line_angle_mean(a: float, b: float, w: float) -> float:
    """Weighted mean of two *line* orientations using doubled angles."""
    a = float(a)
    b = float(b)
    w = float(w)
    z = (1.0 - w) * np.exp(2j * a) + w * np.exp(2j * b)
    if abs(z) < 1e-20:
        return _wrap_pi_half(a)
    return _wrap_pi_half(0.5 * np.angle(z))


def _smooth_polyline(x, y, window=21, iters=3):
    """Simple strong moving-average smoothing (endpoints fixed)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 5:
        return x, y

    w = int(window)
    if w < 3:
        return x, y
    if w % 2 == 0:
        w += 1

    half = w // 2
    ker = np.ones(w, float) / w

    xs = x.copy()
    ys = y.copy()

    for _ in range(int(iters)):
        x0, y0 = xs[0], ys[0]
        xN, yN = xs[-1], ys[-1]

        xpad = np.r_[xs[half:0:-1], xs, xs[-2:-half-2:-1]]
        ypad = np.r_[ys[half:0:-1], ys, ys[-2:-half-2:-1]]

        xs2 = np.convolve(xpad, ker, mode="valid")
        ys2 = np.convolve(ypad, ker, mode="valid")

        xs2[0], ys2[0] = x0, y0
        xs2[-1], ys2[-1] = xN, yN
        xs, ys = xs2, ys2

    return xs, ys


def _pick_start_centered(X, Y, M, util_img, start_r_frac=0.03):
    R = float(np.nanmax(np.hypot(X[M], Y[M])))
    r = np.hypot(X, Y)
    inside = (M.astype(bool)) & np.isfinite(util_img) & (util_img > 0.0)

    cand = inside & (r <= float(start_r_frac) * R)
    if np.any(cand):
        score = util_img - 0.35 * (r / (float(start_r_frac) * R + 1e-12)) ** 2
        score[~cand] = -np.inf
        imax = int(np.nanargmax(score))
        return np.unravel_index(imax, util_img.shape)

    score = (r / (R + 1e-12)) + 0.05 * (1.0 / np.maximum(util_img, 1e-6))
    score[~inside] = np.inf
    imin = int(np.nanargmin(score))
    return np.unravel_index(imin, util_img.shape)


def _grid_connects_to_boundary(start_ij, allowed, r, r_end):
    from collections import deque

    ny, nx = allowed.shape
    si, sj = start_ij
    if not allowed[si, sj]:
        return False

    q = deque()
    q.append((si, sj))
    seen = np.zeros_like(allowed, dtype=bool)
    seen[si, sj] = True

    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        i, j = q.popleft()
        if r[i, j] >= r_end:
            return True
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                continue
            if not allowed[ni, nj] or seen[ni, nj]:
                continue
            seen[ni, nj] = True
            q.append((ni, nj))
    return False


def _widest_threshold(start_ij, base_mask, util_img, r, r_end, iters=26):
    uvals = util_img[base_mask]
    if uvals.size == 0:
        return 0.0
    lo = 0.0
    hi = float(np.nanmax(uvals))
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        allowed = base_mask & (util_img >= mid)
        if _grid_connects_to_boundary(start_ij, allowed, r, r_end):
            lo = mid
        else:
            hi = mid
    return float(lo)


def _clearance_distance(allowed, dx, dy):
    """Distance-to-edge field inside allowed region (True). Multi-source Dijkstra from band boundary."""
    allowed = allowed.astype(bool)
    ny, nx = allowed.shape
    dist = np.full((ny, nx), np.inf, float)

    neigh = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    steps = [float(np.hypot(di * dy, dj * dx)) for di, dj in neigh]

    def is_edge(i, j):
        if not allowed[i, j]:
            return False
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                return True
            if not allowed[ni, nj]:
                return True
        return False

    heap = []
    for i in range(ny):
        for j in range(nx):
            if is_edge(i, j):
                dist[i, j] = 0.0
                heap.append((0.0, i, j))
    heapq.heapify(heap)

    while heap:
        d, i, j = heapq.heappop(heap)
        if d != dist[i, j]:
            continue
        for step, (di, dj) in zip(steps, neigh):
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                continue
            if not allowed[ni, nj]:
                continue
            nd = d + step
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                heapq.heappush(heap, (nd, ni, nj))

    dist[~allowed] = np.nan
    return dist


def _snap_polyline_to_grid(x, y, X, Y, allowed, util_img, clearance=None, radius=3):
    """Snap interior points to best nearby allowed grid node (keeps path on band)."""
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    if len(x) < 3:
        return x, y

    xmin = float(X[0, 0])
    ymin = float(Y[0, 0])
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    ny, nx = X.shape

    rad = int(max(radius, 1))
    for k in range(1, len(x) - 1):
        j0 = int(np.round((x[k] - xmin) / dx))
        i0 = int(np.round((y[k] - ymin) / dy))
        best = None
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                i = i0 + di
                j = j0 + dj
                if i < 0 or i >= ny or j < 0 or j >= nx:
                    continue
                if not allowed[i, j]:
                    continue
                u = float(util_img[i, j]) if np.isfinite(util_img[i, j]) else -np.inf
                if clearance is not None and np.isfinite(clearance[i, j]):
                    u = u + 0.03 * float(clearance[i, j] / (min(dx, dy) + 1e-12))
                if (best is None) or (u > best[0]):
                    best = (u, i, j)
        if best is not None:
            _, bi, bj = best
            x[k] = float(X[bi, bj])
            y[k] = float(Y[bi, bj])
    return x, y


def _smooth_line_field(psi_pref_img, mask, window=7, iters=1):
    """
    Spatial smoothing of psi_pref as a LINE field using doubled-angle complex averaging.
    Only smooth inside mask. Keeps NaNs outside.
    """
    psi = np.asarray(psi_pref_img, float)
    m = np.asarray(mask, bool)

    if window is None or int(window) <= 1 or iters is None or int(iters) <= 0:
        return psi

    w = int(window)
    if w % 2 == 0:
        w += 1

    # exp(2i psi) represents a line orientation
    z = np.zeros_like(psi, dtype=np.complex128)
    ok = m & np.isfinite(psi)
    z[ok] = np.exp(2j * psi[ok])

    ker = np.ones((w, w), float)

    def conv2(a):
        # separable-ish via two 1D convs (fast enough for modest grids)
        tmp = np.apply_along_axis(lambda v: np.convolve(v, np.ones(w), mode="same"), 1, a)
        out = np.apply_along_axis(lambda v: np.convolve(v, np.ones(w), mode="same"), 0, tmp)
        return out

    zr = z.real.copy()
    zi = z.imag.copy()
    ww = ok.astype(float)

    for _ in range(int(iters)):
        zr2 = conv2(zr) / (w * w)
        zi2 = conv2(zi) / (w * w)
        ww2 = conv2(ww) / (w * w)

        # normalize by local weight
        denom = np.maximum(ww2, 1e-12)
        zsm = (zr2 + 1j * zi2) / denom

        # only update where we have support
        upd = m & (ww2 > 1e-6)
        psi_new = np.full_like(psi, np.nan, float)
        psi_new[upd] = 0.5 * np.angle(zsm[upd])
        psi_new[upd] = np.array([_wrap_pi_half(v) for v in psi_new[upd]], float)

        psi = np.where(upd, psi_new, psi)

        # refresh z fields
        z[:] = 0.0
        ok = m & np.isfinite(psi)
        z[ok] = np.exp(2j * psi[ok])
        zr = z.real.copy()
        zi = z.imag.copy()
        ww = ok.astype(float)

    return psi


# ==========================================================
# Main API
# ==========================================================
def crack_path_grid_phys_mirror(
    R, X, Y, M,
    util_img,
    psi_pref_img=None,

    # NEW: optional confidence for direction guidance (0..1)
    psi_conf_img=None,

    start_r_frac=0.03,
    r_end_frac=0.99,

    util_soft_min=0.98,
    util_hard_min=0.30,          # bumped default (helps reduce wandering)
    widest_relax=0.995,          # bumped default (narrower band)
    util_power=10.0,             # bumped default (strongly favors high util)
    dip_penalty=60.0,            # bumped default

    center_weight=6.0,           # bumped default

    # NEW: symmetry-axis preference (big improvement for your plots)
    axis_weight=2.0,
    axis_x0=0.0,

    # direction / curvature penalties
    dir_penalty=3.0,
    turn_penalty=2.5,
    backtrack_penalty=2.0,

    # smoothing + snap
    smooth_window=21,
    smooth_iters=3,
    snap_radius=3,

    # NEW: smooth psi_pref field before Dijkstra
    psi_pref_smooth_window=7,
    psi_pref_smooth_iters=1,

    hard_min_tries=(1.0, 0.85, 0.7, 0.55, 0.4, 0.3),
):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    util_img = np.asarray(util_img, float)
    ny, nx = X.shape

    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    r = np.hypot(X, Y)
    r_end = float(r_end_frac) * float(R)

    neigh = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    steps = [float(np.hypot(di * dy, dj * dx)) for di, dj in neigh]
    move_angle = [float(np.arctan2(di * dy, dj * dx)) for di, dj in neigh]

    base_inside = (M.astype(bool)) & np.isfinite(util_img) & (util_img > 0.0)
    start_ij = _pick_start_centered(X, Y, M, util_img, start_r_frac=start_r_frac)

    best_path = None

    for hm_fac in hard_min_tries:
        hard_min = float(util_hard_min) * float(hm_fac)
        base_mask = base_inside & (util_img >= hard_min)

        if not base_mask[start_ij]:
            rr = r.copy()
            rr[~base_mask] = np.inf
            if not np.any(np.isfinite(rr)):
                continue
            imin = int(np.nanargmin(rr))
            start_ij = np.unravel_index(imin, rr.shape)

        t_star = _widest_threshold(start_ij, base_mask, util_img, r, r_end)
        thr = max(float(hard_min), float(widest_relax) * float(t_star))

        allowed = base_inside & (util_img >= thr)
        if not _grid_connects_to_boundary(start_ij, allowed, r, r_end):
            continue

        # Smooth psi_pref within allowed region (optional)
        psi_used = None
        if psi_pref_img is not None:
            psi_used = _smooth_line_field(
                psi_pref_img,
                mask=allowed,
                window=int(psi_pref_smooth_window),
                iters=int(psi_pref_smooth_iters),
            )

        clearance = _clearance_distance(allowed, dx, dy)
        c_eps = 0.5 * min(dx, dy) + 1e-12

        u_soft = max(float(util_soft_min), 1e-12)

        cost_node = np.full((ny, nx), np.inf, float)
        ok = allowed
        u = util_img[ok]

        base = (1.0 / np.maximum(u, u_soft)) ** float(util_power)
        dip = np.clip(u_soft - u, 0.0, None)
        pen = float(dip_penalty) * (dip / u_soft) ** 2

        cl = clearance[ok]
        center_pen = float(center_weight) * (c_eps / np.maximum(cl + c_eps, 1e-12))

        # NEW: symmetry-axis penalty
        axpen = float(axis_weight) * ((X[ok] - float(axis_x0)) / (float(R) + 1e-12)) ** 2

        cost_node[ok] = base + pen + center_pen + axpen

        nstate = 9  # 8 dirs + start
        dist = np.full((ny, nx, nstate), np.inf, float)
        prev = np.full((ny, nx, nstate, 3), -1, int)

        si, sj = start_ij
        if not np.isfinite(cost_node[si, sj]):
            continue

        dist[si, sj, 8] = 0.0
        heap = [(0.0, si, sj, 8)]
        found = None

        while heap:
            du, i, j, pdir = heapq.heappop(heap)
            if du != dist[i, j, pdir]:
                continue

            if r[i, j] >= r_end:
                found = (i, j, pdir)
                break

            # psi_pref at node
            psi_pref = None
            if psi_used is not None:
                pp = psi_used[i, j]
                if np.isfinite(pp):
                    psi_pref = float(pp)

            # confidence scale (optional)
            conf = 1.0
            if psi_conf_img is not None:
                cc = psi_conf_img[i, j]
                if np.isfinite(cc):
                    conf = float(np.clip(cc, 0.0, 1.0))

            ri = r[i, j]

            for ndir, (di, dj) in enumerate(neigh):
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= ny or nj < 0 or nj >= nx:
                    continue
                if not np.isfinite(cost_node[ni, nj]):
                    continue

                step = steps[ndir]
                w = step * 0.5 * (cost_node[i, j] + cost_node[ni, nj])

                # direction guidance (LINE angle difference)
                if psi_pref is not None and dir_penalty > 0:
                    ang = move_angle[ndir]
                    dpsi = angle_diff_line(ang, psi_pref)
                    w += float(dir_penalty) * conf * (dpsi / (0.5 * np.pi)) ** 2 * step

                # turn penalty
                if pdir != 8 and turn_penalty > 0:
                    dth = abs(angle_diff_periodic(move_angle[ndir], move_angle[pdir]))
                    w += float(turn_penalty) * (dth / np.pi) ** 2 * step

                # backtrack penalty (discourage decreasing radius)
                rj = r[ni, nj]
                if backtrack_penalty > 0 and (rj < ri - 1e-12):
                    w += float(backtrack_penalty) * ((ri - rj) / max(dx, dy)) * step

                nd = du + w
                if nd < dist[ni, nj, ndir]:
                    dist[ni, nj, ndir] = nd
                    prev[ni, nj, ndir, :] = (i, j, pdir)
                    heapq.heappush(heap, (nd, ni, nj, ndir))

        if found is None:
            continue

        # backtrack
        i, j, ddir = found
        path = [(i, j)]
        while True:
            pi, pj, pd = prev[i, j, ddir, :]
            if pi < 0 or pj < 0:
                break
            i, j, ddir = int(pi), int(pj), int(pd)
            path.append((i, j))
            if i == si and j == sj and ddir == 8:
                break
        path = path[::-1]

        ii = np.array([p[0] for p in path], int)
        jj = np.array([p[1] for p in path], int)
        xu = X[ii, jj].astype(float)
        yu = Y[ii, jj].astype(float)

        # anchor start at origin
        xu[0] = 0.0
        yu[0] = 0.0

        # smooth + snap back
        xu_s, yu_s = _smooth_polyline(xu, yu, window=int(smooth_window), iters=int(smooth_iters))
        if int(snap_radius) > 0:
            xu_s, yu_s = _snap_polyline_to_grid(
                xu_s, yu_s, X, Y,
                allowed=allowed, util_img=util_img,
                clearance=clearance, radius=int(snap_radius)
            )

        # mirror
        xd = -xu_s
        yd = -yu_s
        Xs = np.concatenate([xd[::-1], xu_s[1:]])
        Ys = np.concatenate([yd[::-1], yu_s[1:]])

        psi0 = float(np.arctan2(yu_s[1] - yu_s[0], xu_s[1] - xu_s[0])) if len(xu_s) >= 2 else np.pi / 2.0
        best_path = (Xs, Ys, psi0)
        break

    if best_path is None:
        # fallback straight vertical
        xs = np.array([0.0, 0.0], float)
        ys = np.array([0.0, 0.9 * float(R)], float)
        xd = -xs
        yd = -ys
        Xs = np.concatenate([xd[::-1], xs[1:]])
        Ys = np.concatenate([yd[::-1], ys[1:]])
        return Xs, Ys, np.pi / 2.0

    return best_path



def compute_psi_pref_field(
    X, Y, M,
    Rt_eff, Rs_eff,
    sxx, syy, txy,
    beta_crit,
    drive_basis="first",
    mixed_band=0.45,
    eps=1e-12
):
    """
    Build psi_pref as a *LINE* orientation (θ ≡ θ+π), no outward forcing.
    """
    _, _, th = principal_from_components(sxx, syy, txy)

    psi_t = _wrap_pi_half(th + np.pi/2.0)
    psi_s = _wrap_pi_half(beta_crit + np.pi/2.0)

    Rt = Rt_eff
    Rs = Rs_eff

    psi_pref = np.full_like(Rt, np.nan, dtype=float)

    if str(drive_basis).lower() == "first":
        kt = np.full_like(Rt, np.inf, dtype=float)
        ks = np.full_like(Rs, np.inf, dtype=float)
        np.divide(1.0, Rt, out=kt, where=(Rt > eps))
        np.divide(1.0, Rs, out=ks, where=(Rs > eps))

        kmin = np.minimum(kt, ks)
        rel = np.abs(kt - ks) / np.maximum(kmin, eps)
        is_mixed = rel <= float(mixed_band)

        w = Rt / (Rt + Rs + eps)
        pm = np.array([_line_angle_mean(a, b, ww) for a, b, ww in zip(psi_s, psi_t, w)], dtype=float)
        psi_pref[is_mixed] = pm[is_mixed]

        nm = ~is_mixed
        psi_pref[nm] = np.where(kt[nm] <= ks[nm], psi_t[nm], psi_s[nm])

    else:
        rel = np.abs(Rt - Rs) / np.maximum(np.maximum(Rt, Rs), eps)
        is_mixed = rel <= float(mixed_band)

        w = Rt / (Rt + Rs + eps)
        pm = np.array([_line_angle_mean(a, b, ww) for a, b, ww in zip(psi_s, psi_t, w)], dtype=float)
        psi_pref[is_mixed] = pm[is_mixed]

        nm = ~is_mixed
        psi_pref[nm] = np.where(Rt[nm] >= Rs[nm], psi_t[nm], psi_s[nm])

    return np.array([_wrap_pi_half(v) if np.isfinite(v) else np.nan for v in psi_pref], dtype=float)
