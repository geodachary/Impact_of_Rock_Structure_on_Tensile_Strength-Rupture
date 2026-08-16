"""Crack-path studies with energy and field guidance.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 37 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.

Parity note
-----------
gneiss cell is authoritative: build_guidance_fields carries the four-class failure-map block absent from the schist cell.
"""
from __future__ import annotations


from __future__ import annotations
import os
import sys
import json
import argparse
import importlib
import importlib.util
import inspect
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tools.bound_all_helpers import (
    points_in_disk, map_angle_to_alpha,
    rot_to_material, stress_material_to_global,
    principal_from_components, eval_stress_field_material,
    fit_orthotropic_airy_disk, fit_orthotropic_airy_disk_auto,
    failure_mode_map, compute_psi_pref_field, weak_plane_weight_field,
    solve_cracked_disk_correction_ddm,
    _extract_tip_sifs, _seg_intersect,
    Gc_theta_weak_plane, G_from_K_orthotropic,
)
from tools.cracked_disk_ddm import _try_call_sif_two_tips, sif_two_tips_from_crack

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    _adaptive_N_outer, _adaptive_kink_n, _grid_axes,
    _make_all_combined_plots, _safe_compute_psi_pref, _safe_wp_weight,
    bilinear_scalar, bind_sif_hooks, get_anisotropy_ratio,
    import_or_load_solver, load_module_from_path, replot_from_csvs,
    sif_hooks, validate_solver_api,
)



# --- implementation ---------------------------------------------------


def sanitize_argv(argv):
    out = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "-f":
            skip = True
            continue
        if a.startswith("--f=") or a.startswith("-f="):
            continue
        out.append(a)
    return out


def parse_sample_ids(spec, valid_index=None):
    wanted = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            parts = tok.split("-", 1)
            try:
                a, b = int(parts[0]), int(parts[1])
                wanted.update(range(min(a, b), max(a, b) + 1))
            except ValueError:
                pass
        else:
            try:
                wanted.add(int(tok))
            except ValueError:
                pass
    result = sorted(wanted)
    if valid_index is not None:
        vi = set(valid_index)
        result = [i for i in result if i in vi]
    return result


def modulus_to_MPa(val):
    v = float(val)
    if not np.isfinite(v) or v <= 0.0:
        raise RuntimeError(f"Invalid modulus value encountered: {val}")
    return v * 1e3 if v < 1e3 else v


def wrap_pi(a):
    a = np.asarray(a, float)
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_pi_scalar(a):
    return float(((float(a) + np.pi) % (2 * np.pi)) - np.pi)


def wrap_pi_half(a):
    a = np.asarray(a, float)
    return (a + np.pi / 2) % np.pi - np.pi / 2


def wrap_pi_half_scalar(a):
    return float(((float(a) + np.pi / 2) % np.pi) - np.pi / 2)


def blend_line_orientations(psi_a, psi_b, w):
    psi_a = np.asarray(psi_a, float)
    psi_b = np.asarray(psi_b, float)
    w = np.asarray(w, float)
    out = psi_a.copy()
    ok = np.isfinite(psi_a) & np.isfinite(psi_b) & np.isfinite(w)
    if not np.any(ok):
        return out
    z = (1.0 - w[ok]) * np.exp(2j * psi_a[ok]) + w[ok] * np.exp(2j * psi_b[ok])
    out[ok] = wrap_pi_half(0.5 * np.angle(z))
    return out


def bilinear_angle(psi_img, X, Y, x, y, fill=np.nan):
    x1d, y1d = _grid_axes(X, Y)
    if not (x1d[0] <= x <= x1d[-1]) or not (y1d[0] <= y <= y1d[-1]):
        return float(fill)
    ix = int(np.clip(np.searchsorted(x1d, x) - 1, 0, len(x1d) - 2))
    iy = int(np.clip(np.searchsorted(y1d, y) - 1, 0, len(y1d) - 2))
    x0, x1e, y0, y1e = x1d[ix], x1d[ix + 1], y1d[iy], y1d[iy + 1]
    if (x1e - x0) == 0 or (y1e - y0) == 0:
        return float(fill)
    tx, ty = (x - x0) / (x1e - x0), (y - y0) / (y1e - y0)
    a = [psi_img[iy, ix], psi_img[iy, ix + 1], psi_img[iy + 1, ix], psi_img[iy + 1, ix + 1]]
    if not all(np.isfinite(v) for v in a):
        return float(fill)
    z = (
        (1 - ty) * ((1 - tx) * np.exp(2j * a[0]) + tx * np.exp(2j * a[1]))
        + ty * ((1 - tx) * np.exp(2j * a[2]) + tx * np.exp(2j * a[3]))
    )
    if abs(z) < 1e-20:
        return float(fill)
    return float(wrap_pi_half_scalar(0.5 * np.angle(z)))


def contact_arc_weight(X, Y, R, beta_deg=10.0, smooth_deg=4.0):
    phi = np.arctan2(Y, X)
    rn = np.hypot(X, Y) / (R + 1e-12)
    width = np.deg2rad(float(beta_deg) + float(smooth_deg)) + 1e-6
    dphi = np.minimum(np.abs(wrap_pi(phi - np.pi / 2)), np.abs(wrap_pi(phi + np.pi / 2)))
    return np.clip(np.exp(-(dphi / width) ** 2) * np.exp(-((1.0 - rn) / 0.10) ** 2), 0.0, 1.0)


def extend_to_circle(p0, p1, R):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    vx, vy = x1 - x0, y1 - y0
    if vx * vx + vy * vy < 1e-20:
        rr = np.hypot(x1, y1) + 1e-30
        return R * x1 / rr, R * y1 / rr
    A = vx * vx + vy * vy
    B = 2 * (x1 * vx + y1 * vy)
    C = x1 * x1 + y1 * y1 - R * R
    disc = B * B - 4 * A * C
    if disc < 0:
        rr = np.hypot(x1, y1) + 1e-30
        return R * x1 / rr, R * y1 / rr
    t = max((-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A))
    return float(x1 + t * vx), float(y1 + t * vy)


def _pca_axis(x, y):
    P = np.column_stack([x, y])
    P = P[np.all(np.isfinite(P), 1)]
    if len(P) < 3:
        return np.array([0.0, 1.0]), 1.0
    w, V = np.linalg.eigh(np.cov(P.T))
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    return V[:, 0] / (np.linalg.norm(V[:, 0]) + 1e-30), float(w[0] / (w[1] + 1e-30))


def _collapse_curve(xs, ys, R, nbins=260, sw=11):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return x, y
    u, _ = _pca_axis(x, y)
    v = np.array([-u[1], u[0]])
    P = np.column_stack([x, y])
    s = P @ u
    t = P @ v
    idx = np.argsort(s)
    s, t = s[idx], t[idx]
    edges = np.linspace(s[0], s[-1], max(60, nbins) + 1)
    sb = 0.5 * (edges[:-1] + edges[1:])
    tb = np.full_like(sb, np.nan)
    for i in range(len(sb)):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if np.any(m):
            tb[i] = float(np.median(t[m]))
    good = np.isfinite(tb)
    if good.sum() < 6:
        return x, y
    tb = np.interp(sb, sb[good], tb[good])
    w2 = max(5, sw)
    w2 = w2 + 1 if w2 % 2 == 0 else w2
    pad = w2 // 2
    tb_s = np.convolve(np.pad(tb, pad, mode="reflect"), np.ones(w2) / w2, mode="valid")
    P2 = np.outer(sb, u) + np.outer(tb_s, v)
    x2, y2 = P2[:, 0], P2[:, 1]
    if np.hypot(x2[0], y2[0]) < 0.98 * R and len(x2) >= 2:
        x2[0], y2[0] = extend_to_circle((x2[1], y2[1]), (x2[0], y2[0]), R)
    if np.hypot(x2[-1], y2[-1]) < 0.98 * R and len(x2) >= 2:
        x2[-1], y2[-1] = extend_to_circle((x2[-2], y2[-2]), (x2[-1], y2[-1]), R)
    return x2, y2


def postprocess_best_crack(xs, ys, R, thresh=35.0):
    _, ratio = _pca_axis(np.asarray(xs, float), np.asarray(ys, float))
    if ratio >= thresh:
        return _collapse_curve(xs, ys, R)
    return np.asarray(xs, float), np.asarray(ys, float)


def build_guidance_fields(E1, E2, nu12, G12, R, t, P, alpha_const, alpha_wp_line, fit,
                          Tm, Coh, Phi_deg, points_per_row=61,
                          weak_T_ratio=0.35, weak_C_ratio=0.60, eta_deg=10.0,
                          weak_spacing=None, wp_bandwidth_frac=0.12,
                          platen_half_angle_deg=10.0, platen_smooth_deg=4.0,
                          fail_mixed_band=0.45, fail_mc_nplanes=61, debug=False):
    if weak_spacing is None:
        weak_spacing = _rock().spacing_m
    D = 2.0 * float(R)
    xg, yg, X, Y, M = bd.points_in_disk(D, n=int(points_per_row))
    xm, ym = bd.rot_to_material(xg, yg, alpha_const)
    s11_m, s22_m, s12_m = bd.eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
    sxx, syy, txy = bd.stress_material_to_global(s11_m, s22_m, s12_m, alpha_const)
    s1, s3, th = bd.principal_from_components(sxx, syy, txy)
    s2 = s3
    Phi0 = np.deg2rad(float(Phi_deg))
    modes, Rt_eff, Rs_eff, beta_crit = bd.failure_mode_map(
        sxx, syy, txy, s1,
        Tm=np.full_like(s1, float(Tm)),
        Coh=np.full_like(s1, float(Coh)),
        Phi=np.full_like(s1, Phi0),
        alpha_const=float(alpha_const),
        basis="first",
        util_min=0.98,
        mixed_band=float(fail_mixed_band),
        margin=0.0,
        n_theta_mc=int(fail_mc_nplanes),
        mc_compression_only=True,
        mc_sigma_comp_min=0.0,
        strength_model="weak_plane",
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=None,
        eta_deg=float(eta_deg),
        alpha_wp_line=float(alpha_wp_line),
        x=xg,
        y=yg,
        weak_spacing=float(weak_spacing),
        weak_bandwidth_frac=0.25,
        weak_floor=0.10,
        stress_sign_mode="auto",
        conf_k=0.65,
        wing_k=1.0,
        wing_p=2.0,
    )
    # --- 4CLASS ADJACENT CALL (additive; manuscript revision 2026-07) --------
    # Same locus-aware classifier wired alongside the legacy call above, on the
    # SAME stress state and weak-plane geometry. The legacy guidance-field
    # pipeline below does not read mode4_gf/info4_gf; they are returned as
    # extra dict keys only.
    mode4_gf, info4_gf = None, None
    try:
        from tools.failure_mapping_helpers import failure_mode_map_4class
        mode4_gf, info4_gf = failure_mode_map_4class(
            sxx, syy, txy, s1,
            Tm=np.full_like(s1, float(Tm)),
            Coh=np.full_like(s1, float(Coh)),
            Phi=np.full_like(s1, Phi0),
            alpha_wp_line=float(alpha_wp_line),
            x=xg, y=yg,
            weak_spacing=float(weak_spacing),
            weak_bandwidth_frac=0.25, weak_floor=0.10,
            weak_T_ratio=float(weak_T_ratio),
            weak_C_ratio=float(weak_C_ratio),
            weak_phi=None,
            util_min=0.98,
            n_theta_mc=int(fail_mc_nplanes),
            mc_compression_only=True, mc_sigma_comp_min=0.0,
            stress_sign_mode="auto",
        )
    except Exception as _fc4_err:
        print(f"[4class] WARNING: adjacent call failed in build_guidance_fields: {_fc4_err}")
    # --- END 4CLASS ADJACENT CALL ---------------------------------------------
    psi_pref = _safe_compute_psi_pref(
        X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
        drive_basis="first", mixed_band=float(fail_mixed_band)
    )
    psi_pref_img = np.full_like(X, np.nan, float)
    psi_pref_img[M] = psi_pref
    wt = Rt_eff / (Rt_eff + Rs_eff + 1e-12)
    tensile_w_img = np.full_like(X, 0.0, float)
    tensile_w_img[M] = np.clip(wt, 0.0, 1.0)
    conf = np.clip((-np.minimum(s2, 0.0)) / (np.abs(s1) + np.abs(s2) + 1e-12), 0.0, 1.0)
    conf_img = np.full_like(X, 0.0, float)
    conf_img[M] = conf
    if hasattr(bd, "weak_plane_weight_field"):
        wp_weight_img = _safe_wp_weight(X, Y, float(alpha_wp_line), float(weak_spacing), float(wp_bandwidth_frac))
    else:
        wp_weight_img = np.zeros_like(X)
    w_load_img = contact_arc_weight(X, Y, R, beta_deg=float(platen_half_angle_deg), smooth_deg=float(platen_smooth_deg))
    psi_vertical = np.full_like(X, np.pi / 2.0, float)
    w_vert = np.clip(np.clip(tensile_w_img, 0.0, 1.0) ** 2.0 * (1.0 - 0.85 * w_load_img), 0.0, 1.0)
    psi_mid = blend_line_orientations(psi_pref_img, psi_vertical, w_vert)
    shear_dom = 1.0 - tensile_w_img
    w_guid = np.clip(wp_weight_img * (shear_dom ** 2.0) * (0.6 + 0.4 * conf_img) * (0.40 + 0.60 * w_load_img), 0.0, 1.0)
    psi_pref_guided_img = blend_line_orientations(psi_mid, np.full_like(X, float(alpha_wp_line)), w_guid)
    return dict(
        X=X, Y=Y, M=M,
        psi_pref_guided_img=psi_pref_guided_img,
        tensile_w_img=tensile_w_img,
        conf_img=conf_img,
        wp_weight_img=wp_weight_img,
        w_load_img=w_load_img,
        w_guid_img=w_guid,
        # 4CLASS (additive): outputs of the adjacent locus-aware classifier
        mode4_gf=mode4_gf,
        info4_gf=info4_gf,
    )


def tip_controls(tipx, tipy, R, ds0, X, Y, tensile_w_img, w_guid_img, w_load_img):
    wt = float(np.clip(bilinear_scalar(tensile_w_img, X, Y, tipx, tipy, fill=0.5), 0.0, 1.0))
    wg = float(np.clip(bilinear_scalar(w_guid_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))
    wl = float(np.clip(bilinear_scalar(w_load_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))
    rn = float(np.hypot(tipx, tipy)) / (R + 1e-12)
    ds = float(np.clip(ds0 * (0.70 + 0.90 * (1.0 - rn)) * (1.0 - 0.35 * wl), 0.45 * ds0, 1.60 * ds0))
    cap_deg = float(np.clip(20.0 + 70.0 * (1.0 - wt) + 20.0 * wl, 20.0, 90.0))
    kink_n = int(np.clip(61 + 90 * (1.0 - wt) + 40 * wl, 61, 181))
    kink_n += kink_n % 2 == 0
    K_keep = int(np.clip(7 + 12 * (1.0 - wg), 7, 19))
    return ds, cap_deg, kink_n, K_keep


#: How far a step may retreat toward the disc centre, as a fraction
#: of the step length. Zero would forbid any inward motion and make
#: the path brittle to rounding; a quarter step absorbs that without
#: letting the tip wander back into the interior.
RETREAT_TOL_FRAC = 0.25


def energy_step_guided(xs_u, ys_u, R, alpha_const, airy_fit, E1, E2, nu12, G12, ds,
                       kink_scan_deg, kink_n, K_keep, Gc0, weak_reduction_base,
                       eta_deg, alpha_wp_line, ddm_sample_rs, cod_offset,
                       X, Y, psi_pref_guided_img, tensile_w_img, conf_img,
                       wp_weight_img, w_load_img, corr_cache=None,
                       N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10, debug=False,
                       force_psi=None, accept_subcritical=False):
    R = float(R)
    tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
    rn = float(np.hypot(tipx, tipy)) / (R + 1e-12)
    psi_tip = float(np.arctan2(ys_u[-1] - ys_u[-2], xs_u[-1] - xs_u[-2]))
    cap = np.deg2rad(float(kink_scan_deg))

    kink_n = _adaptive_kink_n(int(max(9, kink_n)), rn)
    n_segs = len(xs_u) - 1
    N_outer_use = _adaptive_N_outer(n_segs, int(N_outer))

    if corr_cache is None:
        corr_cache = {}

    tip_key = (round(tipx, 10), round(tipy, 10))
    corr = corr_cache.get(tip_key)

    if corr is None:
        corr = bd.solve_cracked_disk_correction_ddm(
            float(R),
            float(alpha_const),
            airy_fit,
            np.asarray(xs_u, dtype=float),
            np.asarray(ys_u, dtype=float),
            float(E1),
            float(E2),
            float(nu12),
            float(G12),
            bd.eval_stress_field_material,
            N_outer=int(N_outer_use),
            colloc_eps_frac=float(colloc_eps_frac),
            reg_lam=float(reg_lam),
        )
        corr_cache[tip_key] = corr
        if len(corr_cache) > 3:
            oldest_key = next(iter(corr_cache))
            del corr_cache[oldest_key]

    wt_tip = float(np.clip(bilinear_scalar(tensile_w_img, X, Y, tipx, tipy, fill=0.5), 0.0, 1.0))
    wc_tip = float(np.clip(bilinear_scalar(conf_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))
    wwp_tip = float(np.clip(bilinear_scalar(wp_weight_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))
    wl_tip = float(np.clip(bilinear_scalar(w_load_img, X, Y, tipx, tipy, fill=0.0), 0.0, 1.0))

    shear_dom = 1.0 - wt_tip
    wrb_eff = float(np.clip(
        float(weak_reduction_base) * wwp_tip * (shear_dom ** 2.0) *
        (0.6 + 0.4 * wc_tip) * (0.5 + 0.5 * wl_tip),
        0.0,
        float(weak_reduction_base)
    ))

    psi_line0 = bilinear_angle(psi_pref_guided_img, X, Y, tipx, tipy, fill=np.pi / 2.0)

    dscan = np.linspace(-cap, cap, kink_n)
    r_tip = float(np.hypot(tipx, tipy))
    cand_list = []
    retreating = []
    for dth in dscan:
        psi_new = float(psi_tip + dth)
        nx = tipx + float(ds) * np.cos(psi_new)
        ny = tipy + float(ds) * np.sin(psi_new)

        if (nx * nx + ny * ny) >= (0.999 * R) ** 2:
            continue

        cheap = 0.0
        if np.isfinite(psi_line0):
            dpsi = abs(wrap_pi_half_scalar(psi_new - psi_line0)) / (np.pi / 2.0)
            cheap = -dpsi * dpsi

        # Radial progress. The scan is symmetric about the current
        # direction and cap_deg reaches 90 degrees where the tip is
        # shear-dominated, so a step may point back toward the disc
        # centre. Nothing else forbade it, and over successive steps
        # the tip then wandered instead of propagating: the schist
        # retreated on 98 of 149 steps and never reached the
        # boundary, while the gneiss happened to drift outward and
        # terminate. That difference was not physics -- it was the
        # schist's finer fabric giving a systematically wider scan
        # (median cap 47 against 33 degrees).
        #
        # In a Brazilian disc the crack starts near the centre and
        # runs outward to the platens, so a retreating step is
        # inadmissible. That is a statement about the test geometry
        # rather than about either rock, which is what makes it safe
        # for a lithology this study has not seen.
        if np.hypot(nx, ny) >= r_tip - RETREAT_TOL_FRAC * float(ds):
            cand_list.append((cheap, psi_new, nx, ny, abs(dth)))
        else:
            retreating.append(
                (np.hypot(nx, ny), (cheap, psi_new, nx, ny, abs(dth))))

    # Never let the constraint starve the scan: if every candidate
    # retreats, take the least-retreating one and carry on rather
    # than ending the path here.
    if not cand_list and retreating:
        retreating.sort(key=lambda z: -z[0])
        cand_list = [retreating[0][1]]

    if not cand_list:
        return None

    # A prescribed direction replaces the scan entirely. This is how a
    # physics-guided step is measured: the direction comes from the
    # preferred-orientation field rather than from maximising energy
    # release, but G and Gc are evaluated by the same machinery below,
    # so the two kinds of step are directly comparable.
    if force_psi is not None:
        fx = tipx + float(ds) * np.cos(float(force_psi))
        fy = tipy + float(ds) * np.sin(float(force_psi))
        cand_list = [(0.0, float(force_psi), fx, fy,
                      abs(wrap_pi_scalar(float(force_psi) - psi_tip)))]

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
                if bd._seg_intersect(p1, p2, (xs_u[j - 1], ys_u[j - 1]), (xs_u[j], ys_u[j])):
                    bad = True
                    break
            if bad:
                continue

        xt = np.asarray(list(xs_u) + [nx], dtype=float)
        yt = np.asarray(list(ys_u) + [ny], dtype=float)

        sr = TRY_SIF(
            SIF_FUN,
            xt,
            yt,
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=airy_fit,
            eval_stress_field_material=bd.eval_stress_field_material,
            E1=float(E1),
            E2=float(E2),
            nu12=float(nu12),
            G12=float(G12),
            correction=corr,
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
        )

        _, _, KI, KII = bd._extract_tip_sifs(sr)
        if not (np.isfinite(KI) and np.isfinite(KII)):
            continue

        G, _ = bd.G_from_K_orthotropic(
            float(KI),
            float(KII),
            E1=float(E1),
            E2=float(E2),
            nu12=float(nu12),
            G12=float(G12),
            alpha_const=float(alpha_const),
            psi_tip_global=float(psi_new),
        )
        if not np.isfinite(G):
            continue

        Gc = bd.Gc_theta_weak_plane(
            theta_line=float(wrap_pi_half_scalar(psi_new)),
            alpha_wp=float(alpha_wp_line),
            Gc_matrix=float(Gc0),
            weak_reduction=float(wrb_eff),
            eta_deg=float(eta_deg),
        )

        score = float(G) - float(Gc)
        # G <= Gc means this direction cannot drive the crack, so the
        # scan discards it. That is also why a trace built only from
        # accepted steps can never show G below Gc: the condition is
        # enforced, not observed. A measured physics-guided step keeps
        # its value whatever it is, which is what makes the recorded
        # G/Gc profile a measurement rather than a restatement of the
        # acceptance rule.
        if score <= 0.0 and not accept_subcritical:
            continue

        kIIr = float(abs(KII) / (abs(KI) + 1e-30))
        adt = float(abs_dth)

        if best is None:
            best = (score, nx, ny, float(psi_new), float(G), float(Gc), kIIr, adt)
            continue

        s_best = best[0]
        tol = 0.01 * max(1e-12, abs(s_best))

        if score > s_best + tol:
            best = (score, nx, ny, float(psi_new), float(G), float(Gc), kIIr, adt)
        elif abs(score - s_best) <= tol:
            if kIIr < best[6] - 1e-12:
                best = (score, nx, ny, float(psi_new), float(G), float(Gc), kIIr, adt)
            elif abs(kIIr - best[6]) <= 1e-12 and adt < best[7]:
                best = (score, nx, ny, float(psi_new), float(G), float(Gc), kIIr, adt)

    return best


def build_fields_and_run(dfmeta_row, ds_frac=0.010, max_steps=1400, N_outer=80,
                         psi0_deg=90.0, a0_frac=0.016,
                         nu12=0.25,
                         platen_half_angle_deg=10.0, platen_smooth_deg=4.0, platen_mu=0.0,
                         airy_auto_tune=True, airy_Nbd_base=260, airy_Nbd_arc_each=400,
                         load_factor=1.0, Gc0=1.0, eta_deg=10.0, weak_T_ratio=0.35, weak_C_ratio=0.60,
                         weak_reduction_base=np.nan, weak_spacing=None, wp_bandwidth_frac=0.12,
                         fail_mixed_band=0.45, fail_mc_nplanes=61, points_per_row=61,
                         r_end_frac=0.995, ddm_sample_rs=None, cod_offset=1e-5,
                         postprocess_single_curve=True, pca_ratio_thresh=35.0, debug=False):

    if weak_spacing is None:
        weak_spacing = _rock().spacing_m
    rock, ang_deg, Dmm, tmm, PkN, Tm_in, Coh_in, Phi_in, E1_in, G12_in = dfmeta_row
    anis_ratio = get_anisotropy_ratio(rock)
    E1 = modulus_to_MPa(E1_in)
    G12 = modulus_to_MPa(G12_in)
    E2 = E1 / anis_ratio
    nu12 = float(nu12)
    alpha_const = float(bd.map_angle_to_alpha(float(ang_deg), angle_map="direct"))
    alpha_wp_line = float(wrap_pi_half_scalar(alpha_const))
    D = float(Dmm) * 1e-3
    t = float(tmm) * 1e-3
    R = D / 2.0
    P = float(PkN) * 1e3 * float(load_factor)

    if bool(airy_auto_tune) and hasattr(bd, "fit_orthotropic_airy_disk_auto"):
        fit = bd.fit_orthotropic_airy_disk_auto(
            E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
            beta_deg=float(platen_half_angle_deg), smooth_deg=float(platen_smooth_deg),
            mu=float(platen_mu), Nbd=int(airy_Nbd_base), Nbd_arc_each=int(airy_Nbd_arc_each)
        )
    else:
        fit = bd.fit_orthotropic_airy_disk(
            E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
            M=24, Nbd=int(airy_Nbd_base), beta_deg=float(platen_half_angle_deg),
            smooth_deg=float(platen_smooth_deg), mu=float(platen_mu),
            lam=1e-8, w_arc=12.0, Nbd_arc_each=int(airy_Nbd_arc_each)
        )

    wrb = float(weak_reduction_base)
    if not np.isfinite(wrb):
        wrb = float(np.clip(1.0 - float(weak_T_ratio), 0.0, 0.9))
    else:
        wrb = float(np.clip(wrb, 0.0, 0.9))

    if ddm_sample_rs is None:
        ddm_sample_rs = np.logspace(-4, -2, 4)

    gf = build_guidance_fields(
        E1, E2, nu12, G12, R, t, P, alpha_const, alpha_wp_line, fit,
        Tm=float(Tm_in), Coh=float(Coh_in), Phi_deg=float(Phi_in),
        points_per_row=int(points_per_row), weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio), eta_deg=float(eta_deg),
        weak_spacing=float(weak_spacing), wp_bandwidth_frac=float(wp_bandwidth_frac),
        platen_half_angle_deg=float(platen_half_angle_deg),
        platen_smooth_deg=float(platen_smooth_deg), fail_mixed_band=float(fail_mixed_band),
        fail_mc_nplanes=int(fail_mc_nplanes), debug=debug
    )

    ds0 = float(ds_frac) * R
    a0 = float(a0_frac) * R
    psi0 = np.deg2rad(float(psi0_deg))
    xs_u = [0.0, a0 * np.cos(psi0)]
    ys_u = [0.0, a0 * np.sin(psi0)]
    corr_cache = {}
    trace_rows = []
    stop_reason = "max_steps"
    mode = "ENERGY"
    phys_burst = 4
    phys_left = 0
    psi_cur = float(psi0)
    X, Y = gf["X"], gf["Y"]
    ppg = gf["psi_pref_guided_img"]
    twi = gf["tensile_w_img"]
    ci = gf["conf_img"]
    wpi = gf["wp_weight_img"]
    wli = gf["w_load_img"]
    wgi = gf["w_guid_img"]

    step = 0
    while step < int(max_steps):
        tx, ty = float(xs_u[-1]), float(ys_u[-1])
        if np.hypot(tx, ty) >= float(r_end_frac) * R:
            stop_reason = "reached_boundary"
            break

        ds_step, cap_deg, kink_n, K_keep = tip_controls(tx, ty, R, ds0, X, Y, twi, wgi, wli)

        if mode == "ENERGY":
            best = energy_step_guided(
                xs_u, ys_u, R=R, alpha_const=alpha_const, airy_fit=fit,
                E1=E1, E2=E2, nu12=nu12, G12=G12, ds=ds_step,
                kink_scan_deg=cap_deg, kink_n=kink_n, K_keep=K_keep,
                Gc0=float(Gc0), weak_reduction_base=float(wrb),
                eta_deg=float(eta_deg), alpha_wp_line=float(alpha_wp_line),
                ddm_sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
                X=X, Y=Y, psi_pref_guided_img=ppg, tensile_w_img=twi,
                conf_img=ci, wp_weight_img=wpi, w_load_img=wli,
                corr_cache=corr_cache, N_outer=int(N_outer), debug=debug
            )
            if best is not None:
                _, nx, ny, psi_new, G, Gc, kIIr, adt = best
                xs_u.append(nx)
                ys_u.append(ny)
                psi_cur = float(psi_new)
                trace_rows.append(dict(
                    step=step, x=nx, y=ny, psi=psi_new,
                    KI=np.nan, KII=np.nan, G=G, Gc=Gc, score=G - Gc,
                    kIIratio=kIIr, abs_dth=adt, cap_deg=cap_deg,
                    kink_n=kink_n, ds=ds_step, mode='ENERGY'
                ))
                step += 1
            else:
                mode = "PHYS"
                phys_left = int(max(1, phys_burst))

        if mode == "PHYS":
            for _ in range(phys_left):
                tx2, ty2 = float(xs_u[-1]), float(ys_u[-1])
                if np.hypot(tx2, ty2) >= float(r_end_frac) * R:
                    break
                ds2, *_ = tip_controls(tx2, ty2, R, ds0, X, Y, twi, wgi, wli)
                psi_line = bilinear_angle(ppg, X, Y, tx2, ty2, fill=np.pi / 2.0)
                if not np.isfinite(psi_line):
                    psi_line = np.pi / 2.0
                c1 = float(wrap_pi_scalar(psi_line))
                c2 = float(wrap_pi_scalar(psi_line + np.pi))
                ps = c2 if abs(wrap_pi_scalar(c2 - psi_cur)) < abs(wrap_pi_scalar(c1 - psi_cur)) else c1
                nxp = float(tx2 + ds2 * np.cos(ps))
                nyp = float(ty2 + ds2 * np.sin(ps))
                if nxp * nxp + nyp * nyp >= (0.999 * R) ** 2:
                    break
                # Measure the physics-guided step with the same SIF and
                # energy machinery the scan uses, so it enters the trace
                # on equal terms. Without this the record covers only the
                # energy-selected fraction of the path, and profile
                # statistics measure instrumentation coverage as much as
                # crack behaviour.
                meas = energy_step_guided(
                    xs_u, ys_u, R=R, alpha_const=alpha_const, airy_fit=fit,
                    E1=E1, E2=E2, nu12=nu12, G12=G12, ds=ds2,
                    kink_scan_deg=cap_deg, kink_n=kink_n, K_keep=K_keep,
                    Gc0=float(Gc0), weak_reduction_base=float(wrb),
                    eta_deg=float(eta_deg), alpha_wp_line=float(alpha_wp_line),
                    ddm_sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
                    X=X, Y=Y, psi_pref_guided_img=ppg, tensile_w_img=twi,
                    conf_img=ci, wp_weight_img=wpi, w_load_img=wli,
                    corr_cache=corr_cache, N_outer=int(N_outer), debug=debug,
                    force_psi=ps, accept_subcritical=True,
                )
                xs_u.append(nxp)
                ys_u.append(nyp)
                psi_cur = float(ps)
                if meas is not None:
                    _, _, _, _, G_p, Gc_p, kII_p, adt_p = meas
                    trace_rows.append(dict(
                        step=step, x=nxp, y=nyp, psi=ps,
                        KI=np.nan, KII=np.nan, G=G_p, Gc=Gc_p,
                        score=G_p - Gc_p, kIIratio=kII_p, abs_dth=adt_p,
                        cap_deg=cap_deg, kink_n=kink_n, ds=ds2, mode='PHYS'))
                    step += 1
            phys_left = 0
            mode = "ENERGY"

    xu = np.asarray(xs_u, float)
    yu = np.asarray(ys_u, float)
    xs = np.concatenate([(-xu)[::-1], xu[1:]])
    ys = np.concatenate([(-yu)[::-1], yu[1:]])

    if len(xs) >= 4:
        if np.hypot(xs[0], ys[0]) < 0.98 * R:
            xs[0], ys[0] = extend_to_circle((xs[1], ys[1]), (xs[0], ys[0]), R)
        if np.hypot(xs[-1], ys[-1]) < 0.98 * R:
            xs[-1], ys[-1] = extend_to_circle((xs[-2], ys[-2]), (xs[-1], ys[-1]), R)

    if bool(postprocess_single_curve):
        xs, ys = postprocess_best_crack(xs, ys, R, float(pca_ratio_thresh))

    trace = pd.DataFrame(trace_rows)
    for c in _TC:
        if c not in trace.columns:
            trace[c] = np.nan

    return dict(
        rock=str(rock),
        angle_deg=float(ang_deg),
        alpha_const=float(alpha_const),
        alpha_wp_line=float(alpha_wp_line),
        R=float(R),
        xs=xs,
        ys=ys,
        trace=trace[_TC],
        stop_reason=str(stop_reason),
        Gc0_internal_used=float(Gc0),
    )


def run_smoke(dfmeta, sample_id, out_dir, load_factor=1.0, Gc0=1.0, a0_frac=0.016,
              cod_offset=1e-5, target_steps=1400, debug=False, N_outer=80,
              ds_frac=0.010, points_per_row=61,
              postprocess_single_curve=True, pca_ratio_thresh=35.0):
    os.makedirs(out_dir, exist_ok=True)
    row = dfmeta.loc[sample_id, META_COLS]
    res = build_fields_and_run(
        row, ds_frac=float(ds_frac), max_steps=int(target_steps),
        N_outer=int(N_outer), a0_frac=float(a0_frac), cod_offset=float(cod_offset),
        load_factor=float(load_factor), Gc0=float(Gc0), points_per_row=int(points_per_row),
        postprocess_single_curve=bool(postprocess_single_curve),
        pca_ratio_thresh=float(pca_ratio_thresh), debug=bool(debug)
    )
    n = int(len(res["trace"].dropna(subset=["step"])))
    print(f"  sid={sample_id}  stop={res['stop_reason']}  nsteps={n}  Gc0={res['Gc0_internal_used']:.3e}")
    pd.DataFrame({"x_m": res["xs"], "y_m": res["ys"]}).to_csv(
        os.path.join(out_dir, f"smoke_path_sid{sample_id}.csv"), index=False
    )
    res["trace"].to_csv(os.path.join(out_dir, f"smoke_trace_sid{sample_id}.csv"), index=False)
    with open(os.path.join(out_dir, f"used_params_sid{sample_id}.json"), "w") as f:
        json.dump(
            dict(load_factor=float(load_factor), Gc0=float(Gc0),
                 a0_frac=float(a0_frac), cod_offset=float(cod_offset)),
            f, indent=2
        )
    res["sample_id"] = sample_id
    return res


def _run_smoke_worker(args_tuple):
    meta_csv, solver_mod, solver_py, sample_id, out_dir, kw = args_tuple
    global bd, SIF_FUN, TRY_SIF
    if solver_py:
        bd = load_module_from_path(solver_py, "bd")
    else:
        bd = import_or_load_solver(solver_mod, meta_csv=meta_csv)
    validate_solver_api(bd)
    bind_sif_hooks(bd)
    # bind_sif_hooks binds the hooks inside tools.ddm. This section
    # keeps its own module-level SIF_FUN/TRY_SIF and the stepper
    # reads those, so mirror the binding across; otherwise
    # energy_step_guided calls None.
    SIF_FUN, TRY_SIF = sif_hooks()
    dfmeta = pd.read_csv(meta_csv, index_col=0)
    dfmeta.columns = [str(c).strip() for c in dfmeta.columns]
    return run_smoke(dfmeta, sample_id, out_dir, **kw)


def run_smoke_all(dfmeta, out_dir=output_dirs.FIELD_DIR, sample_ids=None, load_factor=1.0,
                  Gc0=1.0, a0_frac=0.016, cod_offset=1e-5, target_steps=1400,
                  debug=False, N_outer=40, ds_frac=0.010, points_per_row=61,
                  postprocess_single_curve=True, pca_ratio_thresh=35.0,
                  n_workers=1, meta_csv="tensile_samples_data.csv",
                  solver_mod="tools.bound_all_helpers", solver_py=""):
    os.makedirs(out_dir, exist_ok=True)
    if sample_ids is None:
        sample_ids = list(range(8, 16))
    kw = dict(
        load_factor=float(load_factor), Gc0=float(Gc0), a0_frac=float(a0_frac),
        cod_offset=float(cod_offset), target_steps=int(target_steps), debug=bool(debug),
        N_outer=int(N_outer), ds_frac=float(ds_frac), points_per_row=int(points_per_row),
        postprocess_single_curve=bool(postprocess_single_curve),
        pca_ratio_thresh=float(pca_ratio_thresh)
    )
    valid_ids = [sid for sid in sample_ids if sid in dfmeta.index]
    missing = [sid for sid in sample_ids if sid not in dfmeta.index]
    for sid in missing:
        print(f"[smoke_all] sid={sid} not in CSV — skip")

    results = []
    if int(n_workers) <= 1:
        for sid in valid_ids:
            print(f"\n[smoke_all] === sid={sid} ===")
            try:
                res = run_smoke(dfmeta, sid, out_dir, **kw)
                results.append(res)
            except Exception as e:
                print(f"[smoke_all] sid={sid} FAILED: {e}")
    else:
        print(f"[smoke_all] Parallel mode: {min(n_workers, len(valid_ids))} workers")
        worker_args = [(meta_csv, solver_mod, solver_py, sid, out_dir, kw) for sid in valid_ids]
        with ProcessPoolExecutor(max_workers=int(n_workers)) as pool:
            futures = {pool.submit(_run_smoke_worker, a): a[3] for a in worker_args}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    n = int(len(res["trace"].dropna(subset=["step"])))
                    print(f"  [parallel] sid={sid} done  nsteps={n}")
                except Exception as e:
                    print(f"  [parallel] sid={sid} FAILED: {e}")
        results.sort(key=lambda r: r.get("sample_id", 0))

    if not results:
        print("[smoke_all] No successful runs.")
        return

    # out_dir holds the step traces and the run summary. The three
    # combined plots are figures, so they are left to default to the
    # figure directory rather than landing beside the CSVs.
    _make_all_combined_plots(results)
    pd.DataFrame([
        dict(
            sample_id=r["sample_id"], rock=r["rock"], angle_deg=r["angle_deg"],
            R_m=r["R"], stop_reason=r["stop_reason"],
            nsteps=int(len(r["trace"].dropna(subset=["step"]))),
            Gc0=r["Gc0_internal_used"]
        ) for r in results
    ]).to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)
    print(f"\n=== Done. Folder: {os.path.abspath(out_dir)} ===")


def run_section(argv=None):
    global bd, SIF_FUN, TRY_SIF
    if argv is None:
        raw = sanitize_argv(sys.argv[1:])
        _jupyter_indicators = any(
            a.endswith(".json") or a.startswith("--ip") or a.startswith("--stdin")
            or a.startswith("--control") or a.startswith("--hb") or a.startswith("--shell")
            or a.startswith("--transport") or a.startswith("--iopub")
            for a in raw
        )
        argv = [] if (_jupyter_indicators or len(raw) == 0) else raw

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--solver_py", default=os.environ.get("SOLVER_PY", ""))
    ap.add_argument("--solver_mod", default=os.environ.get("SOLVER_MOD", "tools.bound_all_helpers"))
    ap.add_argument("--meta_csv", default="tensile_samples_data.csv")
    ap.add_argument("--out_dir", default=output_dirs.FIELD_DIR)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--load_factor", type=float, default=1.0)
    ap.add_argument("--Gc0", type=float, default=1.0)
    ap.add_argument("--a0_frac", type=float, default=0.016)
    ap.add_argument("--cod_offset", type=float, default=1e-5)
    ap.add_argument("--target_steps", type=int, default=1400)
    ap.add_argument("--ds_frac", type=float, default=0.010)
    ap.add_argument("--N_outer", type=int, default=80)
    ap.add_argument("--points_per_row", type=int, default=61)
    ap.add_argument("--no_postprocess", action="store_true")
    ap.add_argument("--pca_ratio_thresh", type=float, default=35.0)

    ap.add_argument(
        "--sample_ids", type=str, default=_rock().sample_id_spec(),
        help='Sample IDs for smoke_all. Accepts _rock().sample_id_spec(), "1,2,3", "1-3,5". Default: 1-7.'
    )

    ap.add_argument(
        "--sample_id", type=int, default=1,
        help="Single sample ID for the smoke sub-command. Default: 1."
    )
    ap.add_argument(
        "--n_workers", type=int, default=1,
        help="Number of parallel worker processes (default 1 = serial)."
    )

    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("smoke")
    sub.add_parser("smoke_all")
    sp_rep = sub.add_parser("replot")
    sp_rep.add_argument("--out_dir", default=output_dirs.FIELD_DIR)
    sp_rep.add_argument("--meta_csv", default="tensile_samples_data.csv")

    if len(argv) == 0:
        argv = ["smoke_all"]

    args = ap.parse_args(argv)

    if args.cmd is None:
        args.cmd = "smoke_all"

    if args.cmd == "replot":
        # Name the specimens explicitly. The combined-plot writer names
        # its output after the rocks it was given, so replotting all
        # fourteen produces one cross-rock figure rather than the two
        # per-lithology figures the manuscript cites.
        replot_from_csvs(out_dir=args.out_dir, meta_csv=args.meta_csv,
                         sample_ids=list(_rock().sample_ids))
        return

    if str(args.solver_py).strip():
        bd = load_module_from_path(str(args.solver_py).strip(), "bd")
    else:
        bd = import_or_load_solver(args.solver_mod, meta_csv=args.meta_csv, debug=args.debug)

    validate_solver_api(bd)
    bind_sif_hooks(bd)
    # bind_sif_hooks binds the hooks inside tools.ddm. This section
    # keeps its own module-level SIF_FUN/TRY_SIF and the stepper
    # reads those, so mirror the binding across; otherwise
    # energy_step_guided calls None.
    SIF_FUN, TRY_SIF = sif_hooks()

    dfmeta = pd.read_csv(args.meta_csv, index_col=0)
    dfmeta.columns = [str(c).strip() for c in dfmeta.columns]

    upsc = not bool(args.no_postprocess)
    kw = dict(
        load_factor=float(args.load_factor),
        Gc0=float(args.Gc0),
        a0_frac=float(args.a0_frac),
        cod_offset=float(args.cod_offset),
        target_steps=int(args.target_steps),
        debug=bool(args.debug),
        N_outer=int(args.N_outer),
        ds_frac=float(args.ds_frac),
        points_per_row=int(args.points_per_row),
        postprocess_single_curve=upsc,
        pca_ratio_thresh=float(args.pca_ratio_thresh),
    )

    if args.cmd == "smoke_all":
        ids = parse_sample_ids(args.sample_ids, valid_index=list(dfmeta.index))
        if not ids:
            raise RuntimeError(
                f"No valid sample IDs found for spec '{args.sample_ids}'. "
                f"Available: {sorted(dfmeta.index.tolist())}"
            )
        print(f"[smoke_all] Running sample IDs: {ids}")
        run_smoke_all(
            dfmeta, out_dir=args.out_dir, sample_ids=ids,
            n_workers=int(args.n_workers),
            meta_csv=args.meta_csv,
            solver_mod=args.solver_mod,
            solver_py=args.solver_py,
            **kw
        )
        return

    if args.sample_id not in dfmeta.index:
        raise RuntimeError(f"sample_id {args.sample_id} not in CSV.")

    os.makedirs(args.out_dir, exist_ok=True)
    run_smoke(dfmeta, args.sample_id, args.out_dir, **kw)



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global META_COLS, ROCK_ANISO_RATIO, SIF_FUN, TRY_SIF, _ANGLE_MARKERS, \
        _COLORS, _TC, bd
    _bind(rock)
    """
    Brazilian Disk crack-path studies suite — ENERGY + field guidance.
    Publication-quality combined plots only (no individual subplots).

    SPEED FIXES (this revision) — ZERO physics or plot changes:
      1. corr_cache keyed only on crack-tip pair (not full path length),
         so the DDM correction is reused across kink-scan candidates at
         the same tip position.  This alone cuts inner-loop cost by ×kink_n.
      2. Adaptive N_outer: starts small (20) and ramps to args.N_outer only
         once the crack is long enough that the correction matters.
      3. Default airy_Nbd_base / airy_Nbd_arc_each halved for the propagation
         loop (the Airy fit is only needed for the stress boundary condition,
         not for final field maps).
      4. kink_n hard-capped at 61 during early steps (r/R < 0.25); full scan
         only when tip is deep.
      5. guidance fields built once per sample and cached — not rebuilt per step.
      6. ddm_sample_rs shortened to 4 points (was 6) — negligible accuracy loss.
      7. K_keep floor raised to 5 (was 7 but tip_controls can return 7–19);
         early filtering removes obviously-bad directions before SIF calls.
      8. Parallel sample processing via concurrent.futures (optional, enabled
         by --n_workers N, default 1 = serial for reproducibility).

    JUPYTER:
        run_section([])                    # run all 7 samples
        run_section(["replot"])          # re-plot from existing CSVs instantly
        run_section(["smoke","--sample_id","1"])
    """
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.6",
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
    _COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
               "#CC79A7", "#56B4E9", "#F0E442"]
    _ANGLE_MARKERS = {0: "o", 15: "s", 30: "^", 45: "D", 60: "v", 75: "P", 90: "X"}
    ROCK_ANISO_RATIO = {
        "augen gneiss": 2.037,
        "psammitic schist": 3.763,
        "psammatic schist": 3.763,   # historical spelling, still accepted
    }
    bd = None
    SIF_FUN = None
    TRY_SIF = None
    _TC = ["step", "x", "y", "psi", "KI", "KII", "G", "Gc", "score", "kIIratio", "abs_dth",
           "cap_deg", "kink_n", "ds", "mode"]
    META_COLS = ["Rock_type", "Angle", "Diameter_mm", "Thickness_mm", "Load_(KN)",
                 "Tensile_strength_Mpa", "Cohesion", "Friction_Angle",
                 "Modulus_of_Elasticity", "Shear_Modulus"]
    run_section([])
