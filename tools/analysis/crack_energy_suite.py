"""Energy-guided DDM crack growth, four-class failure maps, strain-energy and stress-tensor glyph figures.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 33 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.

Parity note
-----------
gneiss cell is authoritative: it routes through postprocess_best_crack_physics, which the schist cell lacked.
"""
from __future__ import annotations


import os
import sys
import inspect
import argparse
import numpy as np
import pandas as pd
from tools.data_io import canonical_rock_type
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tools.bound_all_helpers import (
    points_in_disk,
    map_angle_to_alpha,
    rot_to_material,
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
    fit_orthotropic_airy_disk,
    fit_orthotropic_airy_disk_auto,
    failure_mode_map,
    compute_psi_pref_field,
    weak_plane_weight_field,
    solve_cracked_disk_correction_ddm,
    _extract_tip_sifs,
    _seg_intersect,
    Gc_theta_weak_plane,
    G_from_K_orthotropic,
)
from tools.cracked_disk_ddm import (
    _try_call_sif_two_tips,
    sif_two_tips_from_crack,
)

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    _grid_axes_from_mesh, _grid_cell_area, _img_from_mask,
    _polyline_length, _thin_points_for_glyphs, bilinear_sample_scalar,
    build_psi_pref_field, compute_wp_weight_img,
    disk_failure_stats_all_points, get_anisotropy_ratio,
    save_full_fields_npz,
    strain_energy_density_ortho_plane_stress_material,
)



# --- implementation ---------------------------------------------------


def modulus_to_MPa(val):
    v = float(val)
    if not np.isfinite(v) or v <= 0.0:
        raise RuntimeError(f"Invalid modulus value encountered: {val}")
    return v * 1e3 if v < 1e3 else v


def sanitize_argv(argv):
    out = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "-f":
            skip_next = True
            continue
        if a.startswith("--f=") or a.startswith("-f="):
            continue
        out.append(a)
    return out


def wrap_pi(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi_scalar(a: float) -> float:
    return float(((float(a) + np.pi) % (2.0 * np.pi)) - np.pi)


def wrap_pi_half(a):
    a = np.asarray(a, dtype=float)
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def wrap_pi_half_scalar(a: float) -> float:
    return float(((float(a) + np.pi / 2.0) % np.pi) - np.pi / 2.0)


def blend_line_orientations(psi_a, psi_b, w):
    psi_a = np.asarray(psi_a, float)
    psi_b = np.asarray(psi_b, float)
    w = np.asarray(w, float)
    out = psi_a.copy()
    ok = np.isfinite(psi_a) & np.isfinite(psi_b) & np.isfinite(w)
    if not np.any(ok):
        return out
    za = np.exp(2j * psi_a[ok])
    zb = np.exp(2j * psi_b[ok])
    z = (1.0 - w[ok]) * za + w[ok] * zb
    ang = 0.5 * np.angle(z)
    out[ok] = wrap_pi_half(ang)
    return out


def bilinear_sample_line_angle(psi_img, X, Y, x, y, fill=np.nan):
    x1d, y1d = _grid_axes_from_mesh(X, Y)
    nx = len(x1d)
    ny = len(y1d)
    if not (x1d[0] <= x <= x1d[-1]) or not (y1d[0] <= y <= y1d[-1]):
        return float(fill)
    ix = int(np.searchsorted(x1d, x) - 1)
    iy = int(np.searchsorted(y1d, y) - 1)
    ix = max(0, min(ix, nx - 2))
    iy = max(0, min(iy, ny - 2))
    x0, x1 = x1d[ix], x1d[ix + 1]
    y0, y1 = y1d[iy], y1d[iy + 1]
    if (x1 - x0) == 0 or (y1 - y0) == 0:
        return float(fill)
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    a00 = psi_img[iy, ix]
    a10 = psi_img[iy, ix + 1]
    a01 = psi_img[iy + 1, ix]
    a11 = psi_img[iy + 1, ix + 1]
    if not (np.isfinite(a00) and np.isfinite(a10) and np.isfinite(a01) and np.isfinite(a11)):
        return float(fill)
    z00 = np.exp(2j * a00)
    z10 = np.exp(2j * a10)
    z01 = np.exp(2j * a01)
    z11 = np.exp(2j * a11)
    z0 = (1 - tx) * z00 + tx * z10
    z1 = (1 - tx) * z01 + tx * z11
    z = (1 - ty) * z0 + ty * z1
    if abs(z) < 1e-20:
        return float(fill)
    return float(wrap_pi_half_scalar(0.5 * np.angle(z)))


def extend_to_circle(p0, p1, R):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    vx, vy = x1 - x0, y1 - y0
    if vx * vx + vy * vy < 1e-20:
        rr = np.hypot(x1, y1) + 1e-30
        return (R * x1 / rr, R * y1 / rr)
    A = vx * vx + vy * vy
    B = 2.0 * (x1 * vx + y1 * vy)
    C = x1 * x1 + y1 * y1 - R * R
    disc = B * B - 4 * A * C
    if disc < 0:
        rr = np.hypot(x1, y1) + 1e-30
        return (R * x1 / rr, R * y1 / rr)
    t1 = (-B + np.sqrt(disc)) / (2 * A)
    t2 = (-B - np.sqrt(disc)) / (2 * A)
    t = max(t1, t2)
    return (float(x1 + t * vx), float(y1 + t * vy))


def contact_arc_weight(X, Y, R, beta_deg, smooth_deg):
    phi = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    rn = r / (R + 1e-12)
    beta = np.deg2rad(float(beta_deg))
    sm = np.deg2rad(float(smooth_deg))
    width = beta + sm + 1e-6
    dtop = np.abs(wrap_pi(phi - np.pi / 2))
    dbot = np.abs(wrap_pi(phi + np.pi / 2))
    dphi = np.minimum(dtop, dbot)
    w_ang = np.exp(-(dphi / width) ** 2)
    w_rad = np.exp(-((1.0 - rn) / 0.10) ** 2)
    return np.clip(w_ang * w_rad, 0.0, 1.0)


def _pca_axis(x, y):
    P = np.column_stack([x, y])
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3:
        return np.array([0.0, 1.0]), 1.0
    C = np.cov(P.T)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    v = V[:, 0]
    v = v / (np.linalg.norm(v) + 1e-30)
    ratio = float(w[0] / (w[1] + 1e-30))
    return v, ratio


def _arc_median_collapse(xs, ys, R, nbins=260, smooth_win=13):
    """
    Collapse a cloud of (possibly curved) points to a single median curve.

    1. Sort by PCA primary axis (initial ordering only).
    2. Build cumulative arc-length along the ordered cloud.
    3. Bin by arc-length; take median x,y in each bin.
    4. Smooth the perpendicular residual only.
    5. Extend endpoints to disk boundary.
    """
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return x, y

    u, _ratio = _pca_axis(x, y)
    v = np.array([-u[1], u[0]], float)

    P = np.column_stack([x, y])
    order = np.argsort(P @ u)
    xs_ord = x[order]
    ys_ord = y[order]

    diffs = np.hypot(np.diff(xs_ord), np.diff(ys_ord))
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    arc_total = arc[-1]
    if arc_total < 1e-12:
        return x, y

    t_ord = np.column_stack([xs_ord, ys_ord]) @ v

    nbins = int(max(60, nbins))
    edges = np.linspace(0.0, arc_total, nbins + 1)
    s_bin = 0.5 * (edges[:-1] + edges[1:])
    t_bin = np.full(nbins, np.nan, float)
    x_bin = np.full(nbins, np.nan, float)
    y_bin = np.full(nbins, np.nan, float)

    for i in range(nbins):
        m = (arc >= edges[i]) & (arc < edges[i + 1])
        if np.any(m):
            t_bin[i] = float(np.median(t_ord[m]))
            x_bin[i] = float(np.median(xs_ord[m]))
            y_bin[i] = float(np.median(ys_ord[m]))

    good = np.isfinite(t_bin) & np.isfinite(x_bin)
    if np.sum(good) < 6:
        return x, y

    t_bin = np.interp(s_bin, s_bin[good], t_bin[good])
    x_bin = np.interp(s_bin, s_bin[good], x_bin[good])
    y_bin = np.interp(s_bin, s_bin[good], y_bin[good])

    win = int(max(5, smooth_win))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ker = np.ones(win, float) / float(win)
    t_sm = np.convolve(np.pad(t_bin, pad, mode="reflect"), ker, mode="valid")

    t_raw_bin = np.column_stack([x_bin, y_bin]) @ v
    dt = t_sm - t_raw_bin
    x2 = x_bin + dt * v[0]
    y2 = y_bin + dt * v[1]

    if len(x2) >= 2 and np.hypot(x2[0], y2[0]) < 0.98 * R:
        x2[0], y2[0] = extend_to_circle((x2[1], y2[1]), (x2[0], y2[0]), R)
    if len(x2) >= 2 and np.hypot(x2[-1], y2[-1]) < 0.98 * R:
        x2[-1], y2[-1] = extend_to_circle((x2[-2], y2[-2]), (x2[-1], y2[-1]), R)

    return x2, y2


def _path_perp_width_frac(xs, ys, R):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return 0.0
    u, _ = _pca_axis(x, y)
    v = np.array([-u[1], u[0]], float)
    t = np.column_stack([x, y]) @ v
    width = float(np.percentile(t, 90) - np.percentile(t, 10))
    return width / (R + 1e-30)


def postprocess_best_crack(xs, ys, R,
                           pca_ratio_thresh=35.0,
                           width_frac_thresh=0.12,
                           nbins=260, smooth_win=13):
    """
    Collapse a mirrored crack path cloud to a single curve.

    Triggers arc-median collapse when EITHER:
      (a) PCA aspect ratio >= pca_ratio_thresh  (classic straight-path check)
      (b) perpendicular width >= width_frac_thresh * R
          (catches curved-but-double-branch paths, e.g. non-zero angles)
    """
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    _u, ratio = _pca_axis(x, y)
    width_frac = _path_perp_width_frac(x, y, R)
    if ratio >= float(pca_ratio_thresh) or width_frac >= float(width_frac_thresh):
        return _arc_median_collapse(x, y, R, nbins=nbins, smooth_win=smooth_win)
    return x, y


def _principal_axis_metrics(xs, ys):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    u, ratio = _pca_axis(x, y)
    ang_deg = float(np.rad2deg(np.arctan2(u[1], u[0])))
    ang_deg = abs(((ang_deg + 90.0) % 180.0) - 90.0)
    ang_from_vertical_deg = abs(90.0 - ang_deg)
    return ang_deg, ang_from_vertical_deg, float(ratio)


def _sample_path_mean(img, X, Y, xs, ys, fill=np.nan):
    vals = [bilinear_sample_scalar(img, X, Y, float(x), float(y), fill=fill)
            for x, y in zip(xs, ys)]
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fill)
    return float(np.nanmean(vals))


def _collapse_to_single_x_of_y(xs, ys, R, nbins=300, smooth_win=13,
                                centerline=False, center_frac=0.25):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return x, y
    idx = np.argsort(y)
    y, x = y[idx], x[idx]
    ymin, ymax = float(y[0]), float(y[-1])
    edges = np.linspace(ymin, ymax, int(max(80, nbins)) + 1)
    yb = 0.5 * (edges[:-1] + edges[1:])
    xb = np.full_like(yb, np.nan, float)
    for i in range(len(yb)):
        m = (y >= edges[i]) & (y < edges[i + 1])
        if np.any(m):
            xb[i] = float(np.median(x[m]))
    good = np.isfinite(xb)
    if np.sum(good) < 6:
        return x, y
    xb = np.interp(yb, yb[good], xb[good])
    win = int(max(5, smooth_win))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(xb, pad, mode='reflect')
    ker = np.ones(win, float) / float(win)
    xb_s = np.convolve(xp, ker, mode='valid')
    if centerline:
        core = np.abs(yb) <= float(center_frac) * float(R)
        x0 = float(np.median(xb_s[core])) if np.any(core) else float(np.median(xb_s))
        xb_s = xb_s - x0
    xlim = np.sqrt(np.maximum(0.0, float(R) ** 2 - yb ** 2)) - 1e-12
    xb_s = np.clip(xb_s, -xlim, xlim)
    if len(xb_s) >= 2 and np.hypot(xb_s[0], yb[0]) < 0.98 * R:
        xb_s[0], yb[0] = extend_to_circle((xb_s[1], yb[1]), (xb_s[0], yb[0]), R)
    if len(xb_s) >= 2 and np.hypot(xb_s[-1], yb[-1]) < 0.98 * R:
        xb_s[-1], yb[-1] = extend_to_circle((xb_s[-2], yb[-2]), (xb_s[-1], yb[-1]), R)
    return xb_s, yb


def postprocess_best_crack_physics(xs, ys, R, X, Y,
                                   tensile_w_img,
                                   wp_weight_img,
                                   w_load_img,
                                   pca_ratio_thresh=35.0,
                                   vertical_tol_deg=18.0,
                                   tensile_split_min=0.60,
                                   weak_plane_max=0.35,
                                   center_frac=0.25,
                                   width_frac_thresh=0.12):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    if len(x) < 10:
        return x, y

    _ang_deg, ang_from_vertical_deg, ratio = _principal_axis_metrics(x, y)
    tensile_mean = _sample_path_mean(tensile_w_img, X, Y, x, y, fill=0.5)
    weak_mean = _sample_path_mean(wp_weight_img, X, Y, x, y, fill=0.0)

    axial_ratio_trigger = max(10.0, 0.40 * float(pca_ratio_thresh))
    is_axial_split = (
        (ratio >= axial_ratio_trigger) and
        (ang_from_vertical_deg <= float(vertical_tol_deg)) and
        (tensile_mean >= float(tensile_split_min)) and
        (weak_mean <= float(weak_plane_max))
    )
    if is_axial_split:
        return _collapse_to_single_x_of_y(
            x, y, R, nbins=300, smooth_win=13,
            centerline=True, center_frac=center_frac,
        )

    # Use the improved arc-collapse-aware postprocessor
    return postprocess_best_crack(
        x, y, R,
        pca_ratio_thresh=float(pca_ratio_thresh),
        width_frac_thresh=float(width_frac_thresh),
    )


def tip_controls(tipx, tipy, R, ds0, X, Y, tensile_w_img, w_guid_img, w_load_img):
    wt = bilinear_sample_scalar(tensile_w_img, X, Y, tipx, tipy, fill=0.5)
    wg = bilinear_sample_scalar(w_guid_img, X, Y, tipx, tipy, fill=0.0)
    wl = bilinear_sample_scalar(w_load_img, X, Y, tipx, tipy, fill=0.0)
    wt = float(np.clip(wt if np.isfinite(wt) else 0.5, 0.0, 1.0))
    wg = float(np.clip(wg if np.isfinite(wg) else 0.0, 0.0, 1.0))
    wl = float(np.clip(wl if np.isfinite(wl) else 0.0, 0.0, 1.0))
    rr = float(np.hypot(tipx, tipy))
    rn = rr / (R + 1e-12)
    ds = ds0 * (0.70 + 0.90 * (1.0 - rn)) * (1.0 - 0.35 * wl)
    ds = float(np.clip(ds, 0.45 * ds0, 1.60 * ds0))
    cap_deg = 20.0 + 70.0 * (1.0 - wt) + 20.0 * wl
    cap_deg = float(np.clip(cap_deg, 20.0, 90.0))
    kink_n = int(61 + 90 * (1.0 - wt) + 40 * wl)
    kink_n = int(np.clip(kink_n, 61, 181))
    if kink_n % 2 == 0:
        kink_n += 1
    K_keep = int(7 + 12 * (1.0 - wg))
    K_keep = int(np.clip(K_keep, 7, 19))
    return ds, cap_deg, kink_n, K_keep


def _energy_step_candidate_scan(
        xs_u, ys_u,
        R, alpha_const, airy_fit,
        E1, E2, nu12, G12,
        ds, kink_scan_deg, kink_n, K_keep,
        objective, Gc0, weak_reduction_base, eta_deg, alpha_wp_line,
        ddm_sample_rs, cod_offset,
        X, Y,
        psi_pref_guided_img, tensile_w_img, conf_img, wp_weight_img, w_load_img,
        corr_cache=None,
        N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10,
):
    R = float(R)
    cap = np.deg2rad(float(kink_scan_deg))
    kink_n = int(max(9, kink_n))
    K_keep = int(max(0, K_keep))
    obj = str(objective).lower().strip()

    tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
    psi_tip = float(np.arctan2(ys_u[-1] - ys_u[-2], xs_u[-1] - xs_u[-2]))
    dscan = np.linspace(-cap, cap, kink_n)

    stamp = (int(len(xs_u)), float(xs_u[-2]), float(ys_u[-2]), float(xs_u[-1]), float(ys_u[-1]))
    corr_base = None
    if isinstance(corr_cache, dict) and (corr_cache.get("stamp", None) == stamp):
        corr_base = corr_cache.get("corr", None)

    if corr_base is None:
        corr_base = solve_cracked_disk_correction_ddm(
            float(R), float(alpha_const), airy_fit,
            np.asarray(xs_u, float), np.asarray(ys_u, float),
            float(E1), float(E2), float(nu12), float(G12),
            eval_stress_field_material,
            N_outer=int(N_outer),
            colloc_eps_frac=float(colloc_eps_frac),
            reg_lam=float(reg_lam),
        )
        if isinstance(corr_cache, dict):
            corr_cache["stamp"] = stamp
            corr_cache["corr"] = corr_base

    wt_tip = float(np.clip(bilinear_sample_scalar(tensile_w_img, X, Y, tipx, tipy, fill=0.5), 0.0, 1.0))
    wc_tip = float(np.clip(bilinear_sample_scalar(conf_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))
    wwp_tip = float(np.clip(bilinear_sample_scalar(wp_weight_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))
    wload_tip = float(np.clip(bilinear_sample_scalar(w_load_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))

    shear_dom = (1.0 - wt_tip)
    weak_reduction_eff = float(weak_reduction_base) * wwp_tip * (shear_dom ** 2.0) * \
                         (0.6 + 0.4 * wc_tip) * (0.5 + 0.5 * wload_tip)
    weak_reduction_eff = float(np.clip(weak_reduction_eff, 0.0, float(weak_reduction_base)))

    cand_list = []
    psi_line0 = bilinear_sample_line_angle(psi_pref_guided_img, X, Y, tipx, tipy, fill=np.pi / 2)

    for dth in dscan:
        psi_new = float(psi_tip + dth)
        nx = tipx + float(ds) * np.cos(psi_new)
        ny = tipy + float(ds) * np.sin(psi_new)
        if (nx * nx + ny * ny) >= (0.999 * R) ** 2:
            continue
        cheap = 0.0
        if np.isfinite(psi_line0):
            dpsi = abs(wrap_pi_half_scalar(psi_new - psi_line0)) / (np.pi / 2)
            cheap = -dpsi * dpsi
        cand_list.append((cheap, psi_new, nx, ny, abs(dth)))

    if not cand_list:
        return (False, None, None, None, np.nan, np.nan)

    if K_keep > 0 and len(cand_list) > K_keep:
        cand_list.sort(key=lambda z: z[0], reverse=True)
        cand_list = cand_list[:K_keep]

    best = None
    for _cheap, psi_new, nx, ny, abs_dth in cand_list:
        if len(xs_u) > 25:
            p1 = (xs_u[-1], ys_u[-1])
            p2 = (nx, ny)
            bad = False
            for j in range(2, len(xs_u) - 10):
                q1 = (xs_u[j - 1], ys_u[j - 1])
                q2 = (xs_u[j], ys_u[j])
                if _seg_intersect(p1, p2, q1, q2):
                    bad = True
                    break
            if bad:
                continue

        xtrial = np.asarray(list(xs_u) + [nx], float)
        ytrial = np.asarray(list(ys_u) + [ny], float)

        sif_res = _try_call_sif_two_tips(
            sif_two_tips_from_crack,
            xtrial, ytrial,
            R=float(R), alpha_const=float(alpha_const), airy_fit=airy_fit,
            eval_stress_field_material=eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            correction=corr_base,
            sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
        )
        _, _, KI, KII = _extract_tip_sifs(sif_res)
        if not (np.isfinite(KI) and np.isfinite(KII)):
            continue

        G, _H = G_from_K_orthotropic(
            KI, KII,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            alpha_const=float(alpha_const),
            psi_tip_global=float(psi_new),
        )
        if not np.isfinite(G):
            continue

        theta_line = wrap_pi_half_scalar(psi_new)
        Gc = Gc_theta_weak_plane(
            theta_line=float(theta_line), alpha_wp=float(alpha_wp_line),
            Gc_matrix=float(Gc0), weak_reduction=float(weak_reduction_eff),
            eta_deg=float(eta_deg),
        )

        score = (G / (Gc + 1e-30)) if (obj == "ratio") else (G - Gc)
        if obj == "ratio":
            if score < 1.0:
                continue
        else:
            if score <= 0.0:
                continue

        kIIratio = float(abs(KII) / (abs(KI) + 1e-30))

        if best is None:
            best = (score, float(nx), float(ny), float(psi_new), float(G), float(Gc), kIIratio, float(abs_dth))
            continue

        score_best = best[0]
        tol = 0.01 * max(1e-12, abs(score_best))
        if score > score_best + tol:
            best = (score, float(nx), float(ny), float(psi_new), float(G), float(Gc), kIIratio, float(abs_dth))
        elif abs(score - score_best) <= tol:
            if kIIratio < best[6] - 1e-12:
                best = (score, float(nx), float(ny), float(psi_new), float(G), float(Gc), kIIratio, float(abs_dth))
            elif abs(kIIratio - best[6]) <= 1e-12:
                if abs_dth < best[7]:
                    best = (score, float(nx), float(ny), float(psi_new), float(G), float(Gc), kIIratio, float(abs_dth))

    if best is None:
        return (False, None, None, None, np.nan, np.nan)

    score, nx, ny, psi_new, Gbest, Gcbest, _, _ = best
    return (True, nx, ny, psi_new, Gbest, Gcbest)


def crack_path_ddm_hybrid_controller_mirror(
        R, alpha_const, airy_fit,
        X, Y, M,
        psi_pref_guided_img, tensile_w_img, conf_img,
        wp_weight_img, w_load_img, w_guid_img,
        E1, E2, nu12, G12,
        psi0, a0_frac, ds_frac, max_steps, r_end_frac,
        L_switch_frac=0.10, phys_burst=4, energy_retry_every=2, max_energy_fail=8,
        Gc0=1.0, weak_reduction_base=0.35, eta_deg=10.0, alpha_wp_line=0.0,
        ddm_sample_rs=np.logspace(-4, -2, 6), cod_offset=1e-5,
        postprocess_single_curve=True,
        pca_ratio_thresh=35.0,
        width_frac_thresh=0.12,      # NEW: perpendicular-width trigger
        stall_patience=40,
):
    R = float(R)
    ds0 = float(ds_frac) * R
    a0 = float(a0_frac) * R

    xs_u = [0.0, a0 * np.cos(float(psi0))]
    ys_u = [0.0, a0 * np.sin(float(psi0))]
    psi = float(wrap_pi_scalar(psi0))

    mode = "ENERGY"
    phys_left = int(max(1, phys_burst))
    energy_fail_count = 0
    corr_cache = {}

    # Growth stops when the tip reaches the boundary, but the tip does not
    # always get there: the energy criterion can steer the path back inward,
    # and it then orbits without ever satisfying r >= r_end_frac * R. Left to
    # itself the loop spends the whole step budget and, worse, keeps every
    # point of the excursion, so the polyline handed to postprocessing contains
    # an out-and-back path the crack never took. Track the furthest the tip has
    # reached, stop once it has stopped making progress, and keep only the
    # advancing part.
    best_r = -1.0
    best_n = len(xs_u)
    stall = 0

    step = 0
    while step < int(max_steps):
        tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
        rr = float(np.hypot(tipx, tipy))
        if rr >= float(r_end_frac) * R:
            break

        ds_step, cap_deg, kink_n, K_keep = tip_controls(
            tipx, tipy, R, ds0, X, Y, tensile_w_img, w_guid_img, w_load_img
        )

        if mode == "ENERGY":
            ok, nx, ny, psi_new, Gbest, _Gcbest = _energy_step_candidate_scan(
                xs_u, ys_u,
                R=R, alpha_const=alpha_const, airy_fit=airy_fit,
                E1=E1, E2=E2, nu12=nu12, G12=G12,
                ds=ds_step, kink_scan_deg=cap_deg, kink_n=kink_n, K_keep=K_keep,
                objective="diff",
                Gc0=float(Gc0), weak_reduction_base=float(weak_reduction_base),
                eta_deg=float(eta_deg), alpha_wp_line=float(alpha_wp_line),
                ddm_sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
                X=X, Y=Y,
                psi_pref_guided_img=psi_pref_guided_img,
                tensile_w_img=tensile_w_img, conf_img=conf_img,
                wp_weight_img=wp_weight_img, w_load_img=w_load_img,
                corr_cache=corr_cache,
                N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10,
            )

            if ok and np.isfinite(Gbest):
                xs_u.append(float(nx))
                ys_u.append(float(ny))
                psi = float(wrap_pi_scalar(psi_new))
                energy_fail_count = 0
            else:
                energy_fail_count += 1
                mode = "PHYS"
                phys_left = int(max(1, phys_burst))

        if mode == "PHYS":
            ksteps = int(max(1, phys_left))
            phys_left = 0

            for _ in range(ksteps):
                tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
                rr = float(np.hypot(tipx, tipy))
                if rr >= float(r_end_frac) * R:
                    break
                ds_step, *_ = tip_controls(
                    tipx, tipy, R, ds0, X, Y, tensile_w_img, w_guid_img, w_load_img
                )
                psi_line = bilinear_sample_line_angle(
                    psi_pref_guided_img, X, Y, tipx, tipy, fill=np.pi / 2
                )
                if not np.isfinite(psi_line):
                    psi_line = np.pi / 2
                cand1 = float(wrap_pi_scalar(psi_line))
                cand2 = float(wrap_pi_scalar(psi_line + np.pi))
                psi_step = cand2 if abs(wrap_pi_scalar(cand2 - psi)) < abs(
                    wrap_pi_scalar(cand1 - psi)) else cand1
                nx = float(tipx + ds_step * np.cos(psi_step))
                ny = float(tipy + ds_step * np.sin(psi_step))
                if (nx * nx + ny * ny) >= (0.999 * R) ** 2:
                    break
                xs_u.append(nx)
                ys_u.append(ny)
                psi = float(wrap_pi_scalar(psi_step))

            mode = "ENERGY"

        rr_now = float(np.hypot(xs_u[-1], ys_u[-1]))
        if rr_now > best_r + 1e-12:
            best_r = rr_now
            best_n = len(xs_u)
            stall = 0
        else:
            stall += 1
            if stall >= int(stall_patience):
                break

        step += 1

    # discard any trailing excursion that made no outward progress
    xs_u = xs_u[:max(2, best_n)]
    ys_u = ys_u[:max(2, best_n)]

    xu = np.asarray(xs_u, float)
    yu = np.asarray(ys_u, float)
    xs = np.concatenate([(-xu)[::-1], xu[1:]])
    ys = np.concatenate([(-yu)[::-1], yu[1:]])

    if len(xs) >= 4:
        if np.hypot(xs[0], ys[0]) < 0.98 * R:
            xs[0], ys[0] = extend_to_circle((xs[1], ys[1]), (xs[0], ys[0]), R)
        if np.hypot(xs[-1], ys[-1]) < 0.98 * R:
            xs[-1], ys[-1] = extend_to_circle((xs[-2], ys[-2]), (xs[-1], ys[-1]), R)

    # -------------------------------------------------------
    # REVISED postprocessing: physics check first, then
    # arc-collapse-aware width check for non-zero angle cases.
    # -------------------------------------------------------
    if postprocess_single_curve:
        xs, ys = postprocess_best_crack_physics(
            xs, ys, R,
            X=X, Y=Y,
            tensile_w_img=tensile_w_img,
            wp_weight_img=wp_weight_img,
            w_load_img=w_load_img,
            pca_ratio_thresh=float(pca_ratio_thresh),
            vertical_tol_deg=18.0,
            tensile_split_min=0.60,
            weak_plane_max=0.35,
            center_frac=0.25,
            width_frac_thresh=float(width_frac_thresh),
        )

    return xs, ys


def parse_sample_ids(spec, valid_index):
    wanted = set()
    if spec and isinstance(spec, str):
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    wanted.update(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
                except Exception:
                    pass
            else:
                try:
                    wanted.add(int(tok))
                except Exception:
                    pass
    return [i for i in valid_index if i in wanted]


#: Stress-tensor glyph colours. Red for tension, blue for compression --
#: the convention a reader brings to a tensor plot. These are the
#: diverging ends of ColorBrewer RdYlBu, so they stay distinguishable in
#: greyscale and to the common forms of colour blindness. Defined once
#: because the glyphs and the legend must agree.
TENSOR_TENSION_COLOR = "#d7191c"
TENSOR_COMPRESSION_COLOR = "#2c7bb6"


def plot_stress_tensors_glyphs(ax, x, y, s1, s3, th, R,
                               glyph_len_frac=0.10, glyph_alpha=0.85,
                               glyph_cells=18, glyph_per_cell=1, glyph_max=650,
                               title="", r_keep_frac=0.90, len_cap_frac=0.16,
                               bg_vm=None, robust_percentile=95.0,
                               sign_style="linestyle", sigma_floor_MPa=1e-6):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s1 = np.asarray(s1, float)
    s3 = np.asarray(s3, float)
    th = np.asarray(th, float)
    r = np.hypot(x, y)
    keep = (r <= float(r_keep_frac) * float(R))
    keep &= np.isfinite(x) & np.isfinite(y) & np.isfinite(s1) & np.isfinite(s3) & np.isfinite(th)
    xk, yk = x[keep], y[keep]
    s1k, s3k, thk = s1[keep], s3[keep], th[keep]
    bgk = bg_vm[keep] if bg_vm is not None else None
    ax.set_title(title, pad=6)
    ax.add_artist(plt.Circle((0, 0), R, fill=False, color="k", lw=1.2, zorder=3))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.05 * R, 1.05 * R)
    ax.set_ylim(-1.05 * R, 1.05 * R)
    if xk.size == 0:
        return
    idx = _thin_points_for_glyphs(xk, yk, R, cells=int(glyph_cells),
                                   per_cell=int(glyph_per_cell), max_keep=int(glyph_max))
    xk, yk = xk[idx], yk[idx]
    s1k, s3k, thk = s1k[idx], s3k[idx], thk[idx]
    if bgk is not None:
        bgk = bgk[idx]
    if bgk is not None and np.any(np.isfinite(bgk)):
        v = bgk[np.isfinite(bgk)]
        v95 = max(float(np.percentile(v, 95)) if v.size else 1.0, 1e-12)
        c = np.clip(bgk / v95, 0.0, 1.0)
        ax.scatter(xk, yk, c=c, cmap="Greys", s=10, alpha=0.18, zorder=0, linewidths=0)
    abs_all = np.concatenate([np.abs(s1k), np.abs(s3k)])
    abs_all = abs_all[np.isfinite(abs_all)]
    ref = max(float(np.percentile(abs_all, float(robust_percentile))) if abs_all.size else 1.0,
              float(sigma_floor_MPa))
    scale = float(glyph_len_frac) * float(R)
    capL = float(len_cap_frac) * float(R)
    L1 = np.clip(scale * (np.abs(s1k) / ref), 0.0, capL)
    L3 = np.clip(scale * (np.abs(s3k) / ref), 0.0, capL)
    cth, sth_arr = np.cos(thk), np.sin(thk)
    ctp, stp = np.cos(thk + np.pi / 2), np.sin(thk + np.pi / 2)
    seg1 = np.stack([
        np.stack([xk + L1 * cth, yk + L1 * sth_arr], axis=1),
        np.stack([xk - L1 * cth, yk - L1 * sth_arr], axis=1)], axis=1)
    seg3 = np.stack([
        np.stack([xk + L3 * ctp, yk + L3 * stp], axis=1),
        np.stack([xk - L3 * ctp, yk - L3 * stp], axis=1)], axis=1)
    m1_t, m1_c = s1k >= 0, s1k < 0
    m3_t, m3_c = s3k >= 0, s3k < 0
    alpha = float(glyph_alpha)
    # Colour carries the SIGN of the stress, which is what a reader of a
    # tensor-glyph plot expects: red in tension, blue in compression. The
    # previous scheme spent colour on which principal axis a tick was
    # (sigma_1 red, sigma_3 blue) and left the sign to a dashed linestyle,
    # so the physically important distinction was the harder one to see.
    #
    # The axis is still legible: sigma_1 is drawn heavier than sigma_3.
    # Line weight is the right carrier for it because the major and minor
    # axes are perpendicular at every point, so the pair reads as a cross
    # whichever way it is coloured.
    TENSION, COMPRESSION = TENSOR_TENSION_COLOR, TENSOR_COMPRESSION_COLOR
    pairs = [(seg1, m1_t, TENSION, 1.45), (seg1, m1_c, COMPRESSION, 1.45),
             (seg3, m3_t, TENSION, 0.75), (seg3, m3_c, COMPRESSION, 0.75)]
    for seg, mask, col, lw in pairs:
        if np.any(mask):
            ax.add_collection(LineCollection(
                seg[mask], colors=col, linewidths=lw, alpha=alpha,
                zorder=2))


def compact_axis_labels(ax, row, col, nrows, ncols,
                        xlabel="X (m)", ylabel="Y (m)",
                        tick_pad=2, label_pad=2):
    ax.tick_params(axis="both", which="both", pad=tick_pad)
    if row == nrows - 1:
        ax.set_xlabel(xlabel, labelpad=label_pad)
        ax.tick_params(labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if col == 0:
        ax.set_ylabel(ylabel, labelpad=label_pad)
        ax.tick_params(labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global Coh, Coh0, Coh_in, D, Dmm, E1, E1_in, E2, G12, G12_in, HFIG, \
        HROW, M, P, Phi, Phi0, Phi_in, PkN, R, ROCK_ANISO_RATIO, Rs_eff, \
        Rt_eff, Teff, Tm, Tm_in, U, U_img, U_keep, U_max, U_mean, \
        U_total_J, U_vals, WFIG, X, Y, _fc4_csv, _fc4_df, _fc4_err, \
        _fc4_pdf, _fc4_png, _unknown, allU, all_U_vals, alpha_const, \
        alpha_wp_line, ang_deg, anis_ratio, apply_compact, args, argv, axE, \
        axF, axL, axLS, axS, axs_energy, axs_fail, axs_stress, beta_crit, \
        c, cax_energy, cb, cc, cell_area, cmap_fail, col, conf, conf_img, \
        crack_len, denom, dfmeta, disc_counts, disc_pct_all, disc_pct_fail, \
        disc_total, do_auto, energy_mappables, eps, fail_total, \
        failure_mode_map_4class, field_dir, fig_energy, fig_fail, \
        fig_stress, fit, fourclass_records_38, gsE, handles_fail, \
        handles_stress, info4, k, mask_valid, mode4, mode_numeric, modes, \
        n, ncols, norm_fail, nrows, nu12, nu_in, out_csv, out_npz, \
        out_pdf_energy, out_pdf_fail, out_pdf_stress, p, \
        plot_fourclass_maps, psi_mid, psi_pref, psi_pref_guided_img, \
        psi_pref_img, psi_vertical, psi_weak, r, r_pts, required_cols, \
        rmax_frac_stats, row, rr, s1, s11_m, s12_m, s2, s22_m, s3, samples, \
        sc, scE, shear_dom_img, sid, sigma_ref_MPa, sigma_vm, slots, \
        stats_path, stats_rows, sxx, syy, t, tensile_w_img, th, tmm, txy, \
        vmax_global, w_guid, w_load_img, w_vert, weak_T_ratio, \
        weak_reduction_base, wp_weight_img, write_classifier_mapping_csv, \
        wt, xg, xm, xs, yg, ym, ys
    _bind(rock)
    """
    Brazilian disk — DDM crack growth (ENERGY + HYBRID stabilization)
    + Stress tensor glyph plots + Orthotropic strain-energy density plots + field export

    CRACK PATH FIX (this revision):
      - Two-crack-path problem at non-zero angles eliminated.
      - Root cause: mirror assembly [-xu[::-1], xu[1:]] produces two diverging
        branches when the half-crack curves. Old PCA collapse only triggered
        above a high straightness threshold.
      - Fix strategy:
          1. `_arc_median_collapse` — arc-length-parameterised median collapse
             that works on curved paths (not just straight ones).
          2. `postprocess_best_crack` — ALWAYS applies arc collapse when EITHER
             PCA ratio is high OR perpendicular width exceeds width_frac_thresh*R.
          3. `crack_path_ddm_hybrid_controller_mirror` — passes width_frac_thresh
             through to postprocessing.
      - All other physics, field export, and plotting are unchanged.

    LAYOUT FIXES:
      - BIG white space removed; figure width consistent with square subplots.
      - Small wspace brings the two columns close.
      - Energy: thin dedicated colorbar column.

    FIELD EXPORT:
      - Saves one compressed .npz per sample in <out_dir>/fields_npz/

    MATERIAL UPDATE:
      - E1 from CSV column: Modulus_of_Elasticity
      - G12 from CSV column: Shear_Modulus
      - E2 = E1 / anisotropy_ratio  (rock-type table)
      - nu12 read per specimen from CSV column Poisson_Ratio
        (--nu12 is only the fallback when the column is absent)
    """
    ROCK_ANISO_RATIO = {
        "augen gneiss": 2.037,
        "psammitic schist": 3.763,
        "psammatic schist": 3.763,   # historical spelling, still accepted
    }
    p = argparse.ArgumentParser("Brazilian disk — energy-based DDM (hybrid)", allow_abbrev=False)

    p.add_argument("--meta_csv", default="tensile_samples_data.csv")
    p.add_argument("--out_dir", default=output_dirs.FIELD_DIR)
    p.add_argument("--sample_ids", type=str, default=_rock().sample_id_spec())
    p.add_argument("--angle_map", type=str, default="direct",
                   choices=["direct", "neg", "pi2_minus", "pi2_plus"])
    p.add_argument("--points_per_row", type=int, default=161)

    p.add_argument("--hspace", type=float, default=0.14)
    p.add_argument("--wspace", type=float, default=0.02)
    p.add_argument("--row_height_in", type=float, default=3.20)
    p.add_argument("--width_pad_in", type=float, default=0.95)
    p.add_argument("--energy_cbar_ratio", type=float, default=0.045)

    p.add_argument("--nu12", type=float, default=0.25)

    p.add_argument("--platen_half_angle_deg", type=float, default=10.0)
    p.add_argument("--platen_smooth_deg", type=float, default=4.0)
    p.add_argument("--platen_mu", type=float, default=0.0)
    p.add_argument("--airy_auto_tune", action="store_true", default=True)
    p.add_argument("--airy_Nbd_base", type=int, default=420)
    p.add_argument("--airy_Nbd_arc_each", type=int, default=900)
    p.add_argument("--load_factor", type=float, default=1.0)

    p.add_argument("--fail_mixed_band", type=float, default=0.45)
    p.add_argument("--fail_mc_nplanes", type=int, default=361)
    p.add_argument("--fail_margin", type=float, default=0.0)
    p.add_argument("--fail_mode_basis", type=str, default="first", choices=["first", "now"])
    p.add_argument("--fail_util_min", type=float, default=0.98)

    p.add_argument("--psi0_deg", type=float, default=90.0)
    p.add_argument("--a0_frac", type=float, default=0.016)
    p.add_argument("--ds_frac", type=float, default=0.010)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--r_end_frac", type=float, default=0.995)

    p.add_argument("--weak_spacing_m", type=float, default=_rock().spacing_m)
    p.add_argument("--wp_bandwidth_frac", type=float, default=0.12)
    p.add_argument("--eta_deg", type=float, default=10.0)
    p.add_argument("--weak_T_ratio", type=float, default=0.35)
    p.add_argument("--weak_C_ratio", type=float, default=0.60)
    p.add_argument("--Gc0", type=float, default=1.0)
    p.add_argument("--weak_reduction_base", type=float, default=np.nan)

    p.add_argument("--single_curve", action="store_true", default=True)
    p.add_argument("--pca_ratio_thresh", type=float, default=35.0)
    p.add_argument("--width_frac_thresh", type=float, default=0.12,
                   help="Perpendicular width / R above which arc-median collapse is forced. "
                        "Fixes double-path at non-zero angles. Default: 0.12")

    p.add_argument("--save_plots", action="store_true", default=True)
    p.add_argument("--save_fields_npz", action="store_true", default=True)
    p.add_argument("--stats_csv", default="ddm_summary_stats.csv")

    p.add_argument("--glyph_max", type=int, default=650)
    p.add_argument("--glyph_alpha", type=float, default=0.55)
    p.add_argument("--glyph_len_frac", type=float, default=0.10)
    p.add_argument("--glyph_cells", type=int, default=18)
    p.add_argument("--glyph_per_cell", type=int, default=1)
    p.add_argument("--stress_r_keep_frac", type=float, default=0.90)
    p.add_argument("--stress_len_cap_frac", type=float, default=0.16)
    p.add_argument("--stress_bg_vm", action="store_true", default=True)

    argv = sanitize_argv(sys.argv[1:])
    args, _unknown = p.parse_known_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    field_dir = os.path.join(args.out_dir, "fields_npz")
    os.makedirs(field_dir, exist_ok=True)

    dfmeta = pd.read_csv(args.meta_csv, index_col=0)
    dfmeta.columns = [str(c).strip() for c in dfmeta.columns]

    samples = parse_sample_ids(args.sample_ids, list(dfmeta.index))
    if not samples:
        raise RuntimeError("No matching sample IDs found.")

    nu12 = float(args.nu12)

    required_cols = [
        "Rock_type", "Angle", "Diameter_mm", "Thickness_mm", "Load_(KN)",
        "Tensile_strength_Mpa", "Cohesion", "Friction_Angle",
        "Modulus_of_Elasticity", "Shear_Modulus", "Poisson_Ratio"
    ]
    for c in required_cols:
        if c not in dfmeta.columns:
            raise RuntimeError(f"Missing required column in CSV: {c}")

    n = len(samples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    slots = nrows * ncols

    HROW = float(args.row_height_in)
    WFIG = float(ncols * HROW + args.width_pad_in)
    HFIG = float(nrows * HROW)

    fig_fail, axs_fail = plt.subplots(nrows, ncols, figsize=(WFIG, HFIG), constrained_layout=False)
    fig_stress, axs_stress = plt.subplots(nrows, ncols, figsize=(WFIG, HFIG), constrained_layout=False)

    fig_energy = plt.figure(figsize=(WFIG, HFIG), constrained_layout=False)
    gsE = fig_energy.add_gridspec(
        nrows, 3,
        width_ratios=[1.0, 1.0, float(args.energy_cbar_ratio)],
        left=0.07, right=0.985, bottom=0.06, top=0.94,
        wspace=float(args.wspace), hspace=float(args.hspace)
    )
    axs_energy = np.empty((nrows, 2), dtype=object)
    for r in range(nrows):
        for c in range(2):
            axs_energy[r, c] = fig_energy.add_subplot(gsE[r, c])
    cax_energy = fig_energy.add_subplot(gsE[:, 2])

    def apply_compact(fig):
        fig.subplots_adjust(
            left=0.07, right=0.985, bottom=0.06, top=0.94,
            wspace=float(args.wspace), hspace=float(args.hspace)
        )

    apply_compact(fig_fail)
    apply_compact(fig_stress)

    if nrows == 1:
        axs_fail = np.array([axs_fail])
        axs_stress = np.array([axs_stress])

    cmap_fail = mcolors.ListedColormap(["purple", "gold", "darkgreen", "red"])
    norm_fail = mcolors.BoundaryNorm([0, 1, 2, 3, 4], 4)

    weak_T_ratio = float(args.weak_T_ratio)
    if not np.isfinite(float(args.weak_reduction_base)):
        weak_reduction_base = float(np.clip(1.0 - weak_T_ratio, 0.0, 0.90))
    else:
        weak_reduction_base = float(np.clip(float(args.weak_reduction_base), 0.0, 0.90))

    all_U_vals = []
    energy_mappables = []
    stats_rows = []
    rmax_frac_stats = 0.985

    for k, sid in enumerate(samples):
        (rock, ang_deg, Dmm, tmm, PkN, Tm_in, Coh_in, Phi_in,
         E1_in, G12_in, nu_in) = dfmeta.loc[sid, required_cols]
        rock = canonical_rock_type(rock)
        # Poisson's ratio is a measured property of each specimen, so it varies
        # from one to the next; the command-line value is only a fallback.
        nu12 = float(nu_in) if np.isfinite(float(nu_in)) else float(args.nu12)

        anis_ratio = get_anisotropy_ratio(rock)
        E1 = modulus_to_MPa(E1_in)
        G12 = modulus_to_MPa(G12_in)
        E2 = E1 / anis_ratio

        alpha_const = float(map_angle_to_alpha(float(ang_deg), angle_map=str(args.angle_map)))
        alpha_wp_line = wrap_pi_half_scalar(alpha_const)

        D = float(Dmm) * 1e-3
        t = float(tmm) * 1e-3
        P = float(PkN) * 1e3 * float(args.load_factor)
        R = D / 2.0
        sigma_ref_MPa = float((P / (np.pi * R * t + 1e-30)) / 1e6)

        xg, yg, X, Y, M = points_in_disk(D, n=int(args.points_per_row))
        r_pts = np.hypot(xg, yg)
        cell_area = _grid_cell_area(X, Y)

        do_auto = bool(args.airy_auto_tune) and (fit_orthotropic_airy_disk_auto is not None)
        if do_auto:
            fit = fit_orthotropic_airy_disk_auto(
                E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
                beta_deg=float(args.platen_half_angle_deg),
                smooth_deg=float(args.platen_smooth_deg),
                mu=float(args.platen_mu),
                Nbd=int(args.airy_Nbd_base), Nbd_arc_each=int(args.airy_Nbd_arc_each),
            )
        else:
            fit = fit_orthotropic_airy_disk(
                E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
                M=24, Nbd=int(args.airy_Nbd_base),
                beta_deg=float(args.platen_half_angle_deg),
                smooth_deg=float(args.platen_smooth_deg),
                mu=float(args.platen_mu),
                lam=1e-8, w_arc=12.0, Nbd_arc_each=int(args.airy_Nbd_arc_each),
            )

        xm, ym = rot_to_material(xg, yg, alpha_const)
        s11_m, s22_m, s12_m = eval_stress_field_material(
            xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"]
        )
        sxx, syy, txy = stress_material_to_global(s11_m, s22_m, s12_m, alpha_const)
        s1, s3, th = principal_from_components(sxx, syy, txy)
        s2 = s3

        U = strain_energy_density_ortho_plane_stress_material(s11_m, s22_m, s12_m, E1, E2, nu12, G12)
        sigma_vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * (txy ** 2))

        Tm = float(Tm_in)
        Coh0 = float(Coh_in)
        Phi0 = np.deg2rad(float(Phi_in))
        Phi = np.full_like(s1, Phi0, float)
        Coh = np.full_like(s1, Coh0, float)
        Teff = np.full_like(s1, Tm, float)

        modes, Rt_eff, Rs_eff, beta_crit = failure_mode_map(
            sxx, syy, txy, s1,
            Tm=Teff, Coh=Coh, Phi=Phi,
            alpha_const=alpha_const,
            basis=str(args.fail_mode_basis),
            util_min=float(args.fail_util_min),
            mixed_band=float(args.fail_mixed_band),
            margin=float(args.fail_margin),
            n_theta_mc=int(args.fail_mc_nplanes),
            mc_compression_only=True, mc_sigma_comp_min=0.0,
            strength_model="weak_plane",
            weak_T_ratio=float(args.weak_T_ratio),
            weak_C_ratio=float(args.weak_C_ratio),
            weak_phi=None, eta_deg=float(args.eta_deg),
            alpha_wp_line=float(alpha_wp_line),
            x=xg, y=yg,
            weak_spacing=float(args.weak_spacing_m),
            weak_bandwidth_frac=0.25, weak_floor=0.10,
            stress_sign_mode="auto",
            conf_k=0.65, wing_k=1.00, wing_p=2.0,
        )
        # --- 4CLASS ADJACENT CALL (additive; manuscript revision 2026-07) ----
        # Locus-aware WT/WS/MT/MS classifier run ALONGSIDE the legacy
        # failure_mode_map above, on the SAME already-computed stress state.
        # Nothing in the legacy pipeline (modes/Rt_eff/Rs_eff/beta_crit and
        # everything downstream) reads mode4/info4; only the new export block
        # at the end of this cell consumes fourclass_records_38.
        try:
            from tools.failure_mapping_helpers import failure_mode_map_4class
            mode4, info4 = failure_mode_map_4class(
                sxx, syy, txy, s1,
                Tm=Teff, Coh=Coh, Phi=Phi,
                alpha_wp_line=float(alpha_wp_line),
                x=xg, y=yg,
                weak_spacing=float(args.weak_spacing_m),
                weak_bandwidth_frac=0.25, weak_floor=0.10,
                weak_T_ratio=float(args.weak_T_ratio),
                weak_C_ratio=float(args.weak_C_ratio),
                weak_phi=None,
                util_min=float(args.fail_util_min),
                n_theta_mc=int(args.fail_mc_nplanes),
                mc_compression_only=True, mc_sigma_comp_min=0.0,
                stress_sign_mode="auto",
            )
            if k == 0:
                fourclass_records_38 = []
            fourclass_records_38.append(dict(
                sid=int(sid), rock=str(rock), ang_deg=float(ang_deg), R=float(R),
                xg=np.asarray(xg, float), yg=np.asarray(yg, float),
                r_pts=np.asarray(r_pts, float),
                modes_old=np.asarray(modes, dtype=object),
                mode4=np.asarray(mode4, dtype=object),
                locus=info4["locus"], mechanism=info4["mechanism"],
                mixed_flag=info4["mixed_flag"],
                secondary_descriptor=info4["secondary_descriptor"],
            ))
        except Exception as _fc4_err:
            print(f"[4class] WARNING sample {sid}: adjacent call failed: {_fc4_err}")
        # --- END 4CLASS ADJACENT CALL -----------------------------------------

        psi_pref = build_psi_pref_field(
            X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
            drive_basis="first", mixed_band=float(args.fail_mixed_band), eps=1e-12,
        )
        psi_pref_img = np.full_like(X, np.nan, float)
        psi_pref_img[M] = psi_pref

        eps = 1e-12
        wt = Rt_eff / (Rt_eff + Rs_eff + eps)
        tensile_w_img = np.full_like(X, 0.0, float)
        tensile_w_img[M] = np.clip(wt, 0.0, 1.0)

        denom = (np.abs(s1) + np.abs(s2) + 1e-12)
        conf = np.clip((-np.minimum(s2, 0.0)) / denom, 0.0, 1.0)
        conf_img = np.full_like(X, 0.0, float)
        conf_img[M] = conf

        wp_weight_img = compute_wp_weight_img(
            X, Y, alpha_wp_line=float(alpha_wp_line),
            spacing=float(args.weak_spacing_m),
            bandwidth_frac=float(args.wp_bandwidth_frac),
        )
        w_load_img = contact_arc_weight(
            X, Y, R,
            beta_deg=float(args.platen_half_angle_deg),
            smooth_deg=float(args.platen_smooth_deg),
        )

        psi_vertical = np.full_like(X, np.pi / 2.0, float)
        w_vert = np.clip(tensile_w_img, 0.0, 1.0) ** 2.0
        w_vert *= (1.0 - 0.85 * w_load_img)
        w_vert = np.clip(w_vert, 0.0, 1.0)
        psi_mid = blend_line_orientations(psi_pref_img, psi_vertical, w_vert)

        shear_dom_img = (1.0 - tensile_w_img)
        w_guid = wp_weight_img * (shear_dom_img ** 2.0) * (0.6 + 0.4 * conf_img) * (
                0.40 + 0.60 * w_load_img)
        w_guid = np.clip(w_guid, 0.0, 1.0)
        psi_weak = np.full_like(X, float(alpha_wp_line), float)
        psi_pref_guided_img = blend_line_orientations(psi_mid, psi_weak, w_guid)

        xs, ys = crack_path_ddm_hybrid_controller_mirror(
            R=R, alpha_const=alpha_const, airy_fit=fit,
            X=X, Y=Y, M=M,
            psi_pref_guided_img=psi_pref_guided_img,
            tensile_w_img=tensile_w_img,
            conf_img=conf_img,
            wp_weight_img=wp_weight_img,
            w_load_img=w_load_img,
            w_guid_img=w_guid,
            E1=E1, E2=E2, nu12=nu12, G12=G12,
            psi0=np.deg2rad(float(args.psi0_deg)),
            a0_frac=max(0.006, float(args.a0_frac) * 0.5),
            ds_frac=float(args.ds_frac),
            max_steps=int(args.max_steps),
            r_end_frac=float(args.r_end_frac),
            Gc0=float(args.Gc0),
            weak_reduction_base=float(weak_reduction_base),
            eta_deg=float(args.eta_deg),
            alpha_wp_line=float(alpha_wp_line),
            ddm_sample_rs=np.logspace(-4, -2, 6),
            cod_offset=1e-5,
            postprocess_single_curve=bool(args.single_curve),
            pca_ratio_thresh=float(args.pca_ratio_thresh),
            width_frac_thresh=float(args.width_frac_thresh),   # NEW
        )

        out_csv = os.path.join(args.out_dir, f"ddm_crack_sample_{sid}.csv")
        pd.DataFrame({"order": np.arange(len(xs)), "x_m": xs, "y_m": ys}).to_csv(out_csv, index=False)

        U_img = _img_from_mask(M, U, fill=np.nan)
        U_vals = U_img[M]
        all_U_vals.append(U_vals)

        U_mean = float(np.nanmean(U_vals))
        U_max = float(np.nanmax(U_vals))    
        U_total_J = float(np.nansum(U_vals)) * 1e6 * cell_area * t

        disc_total, disc_counts, disc_pct_all, fail_total, disc_pct_fail, mask_valid = \
            disk_failure_stats_all_points(modes=modes, r_pts=r_pts, R=R, rmax_frac=rmax_frac_stats)

        crack_len = float(_polyline_length(xs, ys))
        stats_rows.append(dict(
            Sample=int(sid), Rock=str(rock), Angle_deg=float(ang_deg),
            CrackPts=int(len(xs)), CrackLength_m=float(crack_len),
            E1_MPa=float(E1), E2_MPa=float(E2), G12_MPa=float(G12),
            Anisotropy_Ratio=float(anis_ratio),
            U_mean_MPa=float(U_mean), U_max_MPa=float(U_max), U_total_J=float(U_total_J),
        ))

        if args.save_fields_npz:
            out_npz = os.path.join(field_dir, f"sample_{int(sid):04d}_full_fields.npz")
            save_full_fields_npz(
                out_npz, sid=sid, rock=rock, ang_deg=ang_deg,
                D=D, t=t, R=R, P=P,
                alpha_const=alpha_const, alpha_wp_line=alpha_wp_line,
                sigma_ref_MPa=sigma_ref_MPa,
                E1=E1, E2=E2, nu12=nu12, G12=G12,
                X=X, Y=Y, M=M, xg=xg, yg=yg,
                s11_m=s11_m, s22_m=s22_m, s12_m=s12_m,
                sxx=sxx, syy=syy, txy=txy,
                s1=s1, s3=s3, th=th,
                sigma_vm=sigma_vm, U=U,
                Rt_eff=Rt_eff, Rs_eff=Rs_eff, beta_crit=beta_crit,
                psi_pref_img=psi_pref_img,
                psi_pref_guided_img=psi_pref_guided_img,
                tensile_w_img=tensile_w_img, conf_img=conf_img,
                wp_weight_img=wp_weight_img, w_load_img=w_load_img, w_guid_img=w_guid,
                xs=xs, ys=ys,
                cell_area=cell_area, U_mean=U_mean, U_max=U_max, U_total_J=U_total_J,
                modes=modes,
            )

        row, col = divmod(k, 2)

        # --- Failure plot ---
        axF = axs_fail[row, col]
        mode_numeric = np.array(
            [{"no_failure": 0, "tensile": 1, "shear": 2, "mixed": 3}.get(str(m), 0) for m in modes],
            dtype=int)
        axF.scatter(
            xg[mask_valid], yg[mask_valid],
            c=mode_numeric[mask_valid], s=16,
            cmap=cmap_fail, norm=norm_fail,
            edgecolors="k", linewidths=0.12, zorder=1
        )
        axF.add_artist(plt.Circle((0, 0), R, fill=False, color="k", lw=1.2, zorder=2))
        axF.plot(xs, ys, "k-", lw=2.4, zorder=10)
        axF.set_aspect("equal", "box")
        axF.set_xlim(-1.05 * R, 1.05 * R)
        axF.set_ylim(-1.05 * R, 1.05 * R)
        axF.set_title(f"{rock} | Angle={float(ang_deg):.0f}°", pad=6)
        compact_axis_labels(axF, row, col, nrows, ncols)

        # --- Stress plot ---
        axS = axs_stress[row, col]
        plot_stress_tensors_glyphs(
            axS, xg, yg, s1, s3, th, R,
            glyph_len_frac=float(args.glyph_len_frac),
            glyph_alpha=float(args.glyph_alpha),
            glyph_cells=int(args.glyph_cells),
            glyph_per_cell=int(args.glyph_per_cell),
            glyph_max=int(args.glyph_max),
            title=f"{rock} | Angle={float(ang_deg):.0f}°",
            r_keep_frac=float(args.stress_r_keep_frac),
            len_cap_frac=float(args.stress_len_cap_frac),
            bg_vm=(sigma_vm if args.stress_bg_vm else None),
            robust_percentile=95.0, sign_style="color"
        )
        compact_axis_labels(axS, row, col, nrows, ncols)

        # --- Energy plot ---
        axE = axs_energy[row, col]
        U_keep = mask_valid & np.isfinite(U)
        scE = axE.scatter(xg[U_keep], yg[U_keep], c=U[U_keep], s=16, cmap="inferno", zorder=1)
        energy_mappables.append(scE)
        axE.add_artist(plt.Circle((0, 0), R, fill=False, color="k", lw=1.2, zorder=2))
        axE.plot(xs, ys, "w-", lw=2.2, zorder=10)
        axE.set_aspect("equal", "box")
        axE.set_xlim(-1.05 * R, 1.05 * R)
        axE.set_ylim(-1.05 * R, 1.05 * R)
        axE.set_title(f"{rock} | Angle={float(ang_deg):.0f}°", pad=6)
        compact_axis_labels(axE, row, col, nrows, ncols)

    # --- Legend slot ---
    if n < slots:
        rr, cc = divmod(n, ncols)

        axL = axs_fail[rr, cc]
        axL.axis("off")
        handles_fail = [
            plt.Line2D([0], [0], marker="o", color="w", label="no_failure",
                       markerfacecolor="purple", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", label="tensile",
                       markerfacecolor="gold", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", label="shear",
                       markerfacecolor="darkgreen", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", label="mixed",
                       markerfacecolor="red", markersize=8),
            plt.Line2D([0], [0], color="k", lw=2.4, label="Best crack path"),
        ]
        axL.legend(handles=handles_fail, loc="center", frameon=True, edgecolor="black", ncol=2)

        axLS = axs_stress[rr, cc]
        axLS.axis("off")
        handles_stress = [
            Line2D([0], [0], color=TENSOR_TENSION_COLOR, lw=2.0,
                   label="tension"),
            Line2D([0], [0], color=TENSOR_COMPRESSION_COLOR, lw=2.0,
                   label="compression"),
            Line2D([0], [0], color="0.35", lw=2.4, label=r"$\sigma_1$"),
            Line2D([0], [0], color="0.35", lw=1.0, label=r"$\sigma_3$"),
        ]
        axLS.legend(handles=handles_stress, loc="center", frameon=True, edgecolor="black", ncol=2)
        axs_energy[rr, cc].axis("off")

    # --- Energy colorbar ---
    if len(all_U_vals):
        allU = np.hstack([u[np.isfinite(u)] for u in all_U_vals if u.size])
        if allU.size and energy_mappables:
            vmax_global = float(np.percentile(allU, 95))
            for sc in energy_mappables:
                sc.set_clim(0.0, vmax_global)
            cb = fig_energy.colorbar(energy_mappables[-1], cax=cax_energy)
            cb.set_label("U (MPa)")

    # --- Save stats ---
    stats_path = os.path.join(args.out_dir, str(args.stats_csv))
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    if args.save_plots:
        # The three-class failure-field panel is superseded by the
        # WT/WS/MT/MS map written to the figure directory below,
        # resolves mechanism and structural locus together instead of
        # collapsing weak-plane opening into a matrix category. The
        # current manuscript cites that figure and no longer includes
        # *_failure_fields.pdf, so the panel is not written.
        # args.out_dir is where the field archives and the per-specimen
        # crack traces go. These two are figures, so they follow the
        # figures instead of being filed under the section that drew them.
        _fig_dir = output_dirs.figures()
        out_pdf_stress = os.path.join(_fig_dir, f"{_rock().key}_stress_tensors.pdf")
        out_pdf_energy = os.path.join(_fig_dir, f"{_rock().key}_strain_energy.pdf")

        plt.close(fig_fail)
        fig_stress.savefig(out_pdf_stress, dpi=300, bbox_inches="tight", pad_inches=0.08, transparent=True)
        fig_energy.savefig(out_pdf_energy, dpi=300, bbox_inches="tight", pad_inches=0.08, transparent=True)

        print(f"\n✓ Saved: {out_pdf_stress}")
        print(f"✓ Saved: {out_pdf_energy}")
        print(f"✓ Stats: {stats_path}")

    if args.save_fields_npz:
        print(f"✓ Full fields (.npz) saved in: {field_dir}")

    # --- 4CLASS EXPORT (additive; manuscript revision 2026-07) ---------------
    # Writes the old->new classifier mapping table to the results
    # directory and the locus-resolved WT/WS/MT/MS figure to the figure
    # directory. All existing outputs above are untouched.
    try:
        if "fourclass_records_38" in globals() and len(fourclass_records_38) > 0:
            from tools.fourclass_export import (
                write_classifier_mapping_csv, plot_fourclass_maps,
            )
            # Per lithology. Both rocks wrote this same path, so whichever
            # notebook ran second replaced the first rock's table and the
            # file silently held one lithology while reading as though it
            # held both.
            _fc4_csv = os.path.join(
                output_dirs.TABLE_DIR, f"classifier_mapping_{_rock().key}.csv")
            _fc4_df = write_classifier_mapping_csv(
                fourclass_records_38, _fc4_csv,
                source="cell38_production_grid", rmax_frac=rmax_frac_stats,
            )
            _fc4_pdf, _fc4_png = plot_fourclass_maps(
                fourclass_records_38,
                os.path.join(output_dirs.FIGURE_DIR, f"{_rock().key}_fourclass_map.pdf"),
                os.path.join(output_dirs.FIGURE_DIR, f"{_rock().key}_fourclass_map.png"),
                lith_title="Augen gneiss", rmax_frac=rmax_frac_stats,
            )
            print(f"[4class] mapping table: {_fc4_csv} ({len(_fc4_df)} rows)")
            print(f"[4class] figure: {_fc4_pdf} / {_fc4_png}")
    except Exception as _fc4_err:
        print(f"[4class] WARNING: export failed: {_fc4_err}")
    # --- END 4CLASS EXPORT ----------------------------------------------------
    plt.show()
