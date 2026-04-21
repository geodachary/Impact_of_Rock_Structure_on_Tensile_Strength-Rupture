#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brazilian Disk crack-path studies suite (physics-based ENERGY, Local Symmetry tie-break).
Solver unit system: MPa + mm  (lengths in mm, stresses in MPa, G/Gc in MPa·mm)

Commands:
  smoke      : one run + paper-ready plots (path + G/Gc + log(G) + log(G/Gc))
  smoke_all  : run samples 1-7 and plot ALL crack paths in ONE combined figure (augen_geniss)
  verify     : convergence sweep (ds_frac, N_outer) + summary CSV + plot
  validate   : overlay vs experiment polyline CSV + metrics CSV + plots

USAGE:
  python augen_geniss.py smoke --sample_id 1 --debug
  python augen_geniss.py smoke_all --auto_fix
  python augen_geniss.py verify --sample_id 1
  python augen_geniss.py validate --sample_id 1 --exp_path_csv exp.csv --exp_scale 1.0

Jupyter:
  main(["smoke", "--sample_id", "1", "--debug"])
  main(["smoke_all", "--auto_fix"])
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import importlib
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Jupyter argv sanitizer (removes -f and --f=...)
# ==========================================================
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


# ==========================================================
# Dynamic loader for solver .py file
# ==========================================================
def load_module_from_path(py_path: str, module_name: str = "bd"):
    py_path = os.path.abspath(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"Solver file not found: {py_path}")
    solver_dir = os.path.dirname(py_path)
    if solver_dir and solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate_search_dirs(meta_csv: str | None = None):
    dirs = []
    try:
        dirs.append(os.getcwd())
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        dirs.append(here)
    except Exception:
        pass
    if meta_csv:
        try:
            mdir = os.path.dirname(os.path.abspath(meta_csv))
            if mdir:
                dirs.append(mdir)
        except Exception:
            pass
    out = []
    seen = set()
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            out.append(d)
            seen.add(d)
    return out


def _find_module_py(module_name: str, search_dirs):
    if module_name.lower().endswith(".py") and os.path.isfile(module_name):
        return os.path.abspath(module_name)
    fname = module_name + ".py"
    for d in search_dirs:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def import_or_load_solver(solver_mod: str, meta_csv: str | None = None, debug: bool = False):
    solver_mod = str(solver_mod).strip()
    if not solver_mod:
        raise RuntimeError("Empty solver module name.")
    search_dirs = _candidate_search_dirs(meta_csv=meta_csv)
    for d in search_dirs:
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        if debug:
            print(f"[solver] trying import_module('{solver_mod}')")
        return importlib.import_module(solver_mod)
    except Exception as e1:
        if debug:
            print(f"[solver] import failed: {type(e1).__name__}: {e1}")
    py_path = _find_module_py(solver_mod, search_dirs)
    if py_path is not None:
        if debug:
            print(f"[solver] loading from path: {py_path}")
        return load_module_from_path(py_path, module_name="bd")
    msg = (
        "Could not import/load solver module.\n\n"
        f"Requested solver_mod: {solver_mod}\n"
        "Searched for solver_mod.py in:\n"
        + "\n".join([f"  - {d}" for d in search_dirs]) +
        "\n\nFix options:\n"
        "  (A) Put the solver file (e.g. bound_all_helpers.py) in the working folder, OR\n"
        "  (B) Pass --solver_py <path/to/solver.py>\n"
        "  (C) Pass --solver_mod the_module_name (and make sure it is importable).\n"
    )
    raise RuntimeError(msg)


# ==========================================================
# Angle wrappers
# ==========================================================
def wrap_pi(a):
    a = np.asarray(a, float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def wrap_pi_scalar(a: float) -> float:
    a = float(a)
    return ((a + np.pi) % (2.0 * np.pi)) - np.pi

def wrap_pi_half(a):
    a = np.asarray(a, float)
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0

def wrap_pi_half_scalar(a: float) -> float:
    a = float(a)
    return ((a + np.pi / 2.0) % np.pi) - np.pi / 2.0


# ==========================================================
# Geometry helpers  (all lengths in mm)
# ==========================================================
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


# ==========================================================
# PCA single-curve postprocess (to remove chatter / loops)
# ==========================================================
def _pca_axis(x, y):
    P = np.column_stack([x, y])
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3:
        return np.array([0.0, 1.0]), 1.0
    C = np.cov(P.T)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    u = V[:, 0]
    u = u / (np.linalg.norm(u) + 1e-30)
    ratio = float(w[0] / (w[1] + 1e-30))
    return u, ratio

def _collapse_to_single_curve(xs, ys, R, nbins=260, smooth_win=11):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return x, y
    u, _ratio = _pca_axis(x, y)
    v = np.array([-u[1], u[0]], float)
    P = np.column_stack([x, y])
    s = P @ u; t = P @ v
    idx = np.argsort(s); s, t = s[idx], t[idx]
    smin, smax = float(s[0]), float(s[-1])
    edges = np.linspace(smin, smax, int(max(60, nbins)) + 1)
    sb = 0.5 * (edges[:-1] + edges[1:])
    tb = np.full_like(sb, np.nan, float)
    for i in range(len(sb)):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if np.any(m):
            tb[i] = float(np.median(t[m]))
    good = np.isfinite(tb)
    if np.sum(good) < 6:
        return x, y
    tb = np.interp(sb, sb[good], tb[good])
    win = int(max(5, smooth_win))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    tp = np.pad(tb, pad, mode="reflect")
    ker = np.ones(win, float) / float(win)
    tb_s = np.convolve(tp, ker, mode="valid")
    P2 = np.outer(sb, u) + np.outer(tb_s, v)
    x2, y2 = P2[:, 0], P2[:, 1]
    if np.hypot(x2[0], y2[0]) < 0.98 * R and len(x2) >= 2:
        x2[0], y2[0] = extend_to_circle((x2[1], y2[1]), (x2[0], y2[0]), R)
    if np.hypot(x2[-1], y2[-1]) < 0.98 * R and len(x2) >= 2:
        x2[-1], y2[-1] = extend_to_circle((x2[-2], y2[-2]), (x2[-1], y2[-1]), R)
    return x2, y2

def postprocess_best_crack(xs, ys, R, pca_ratio_thresh=35.0):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    _u, ratio = _pca_axis(x, y)
    if ratio >= float(pca_ratio_thresh):
        return _collapse_to_single_curve(x, y, R, nbins=260, smooth_win=11)
    return x, y


# ==========================================================
# Small-kink isotropic transform (prefilter only)
# ==========================================================
def kink_sif_transform_isotropic(KI: float, KII: float, h: np.ndarray):
    h = np.asarray(h, float)
    ch2 = np.cos(0.5 * h); sh2 = np.sin(0.5 * h)
    c3  = np.cos(1.5 * h); s3  = np.sin(1.5 * h)
    kI  = 0.25 * (3.0 * ch2 + c3) * KI  - 0.75 * (sh2 + s3) * KII
    kII = 0.25 * (sh2 + s3)       * KI  + 0.25 * (ch2 + 3.0 * c3) * KII
    return kI, kII


# ==========================================================
# Validation geometry: polyline metrics
# ==========================================================
def resample_polyline_by_s(x, y, n=400):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return x.copy(), y.copy()
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    L = float(s[-1])
    if L < 1e-30:
        return np.full(n, x[0]), np.full(n, y[0])
    st = np.linspace(0.0, L, n)
    return np.interp(st, s, x), np.interp(st, s, y)

def path_rms_max_distance(x_ref, y_ref, x, y, n=500):
    xr, yr = resample_polyline_by_s(x_ref, y_ref, n=n)
    xt, yt = resample_polyline_by_s(x, y, n=n)
    d = np.hypot(xt - xr, yt - yr)
    return float(np.sqrt(np.mean(d * d))), float(np.max(d))

def symmetric_mean_min_distance(xa, ya, xb, yb, n=500):
    xa, ya = resample_polyline_by_s(xa, ya, n=n)
    xb, yb = resample_polyline_by_s(xb, yb, n=n)
    A = np.stack([xa, ya], axis=1); B = np.stack([xb, yb], axis=1)
    dAB = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    return 0.5 * (float(np.mean(np.min(dAB, axis=1))) + float(np.mean(np.min(dAB, axis=0))))

def initiation_angle_deg(x, y, rmax_frac, R):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = np.hypot(x, y)
    keep = r <= float(rmax_frac) * float(R)
    if np.sum(keep) < 3:
        return float(np.degrees(np.arctan2(y[1] - y[0], x[1] - x[0])))
    xx, yy = x[keep], y[keep]
    C = np.cov(np.stack([xx, yy], axis=0))
    w, V = np.linalg.eigh(C)
    v = V[:, np.argmax(w)]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    if ang >= 180:
        ang -= 360
    return float(ang)


# ==========================================================
# Plotting utilities  (axis labels say mm)
# ==========================================================
def save_path_plot(xs, ys, R, out_path, title, exp_xy=None):
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111)
    if exp_xy is not None:
        ax.plot(exp_xy[0], exp_xy[1], lw=2.0, label="experiment")
    ax.plot(xs, ys, lw=2.0, label="model")
    circ = plt.Circle((0, 0), float(R), fill=False)
    ax.add_artist(circ)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.05 * R, 1.05 * R); ax.set_ylim(-1.05 * R, 1.05 * R)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title(title); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


def save_all_paths_plot(results_list, out_path, title="augen_geniss — crack paths samples 1–7"):
    """Plot all crack paths (samples 1–7) in a single figure, no subplots."""
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results_list)))
    R_max = max(float(r["R"]) for r in results_list)
    circ = plt.Circle((0, 0), R_max, fill=False, color="black", lw=1.4, zorder=5)
    ax.add_artist(circ)
    for i, res in enumerate(results_list):
        sid   = res["sample_id"]
        rock  = str(res.get("rock", ""))
        ang   = res.get("angle_deg", "")
        nstep = int(len(res["trace"].dropna(subset=["step"])))
        label = f"sid {sid}  {rock}  {ang}°  (n={nstep})"
        ax.plot(res["xs"], res["ys"], lw=1.8, color=colors[i], label=label, zorder=4)
    ax.set_aspect("equal", "box")
    pad = 1.08 * R_max
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_xlabel("X (mm)", fontsize=11); ax.set_ylabel("Y (mm)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, lw=0.5, alpha=0.6)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[smoke_all] Combined path plot saved: {out_path}")


def save_energy_plots(trace: pd.DataFrame, out_prefix: str, title: str):
    t = trace.dropna(subset=["step"]).copy()
    if len(t) == 0:
        for suf in ["_G_Gc_log.png", "_G_over_Gc_log.png", "_G_over_Gc_clip.png"]:
            fig = plt.figure(figsize=(7.5, 4.8)); ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Empty trace (no growth)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title)
            fig.tight_layout(); fig.savefig(out_prefix + suf, dpi=300); plt.close(fig)
        return
    G = t["G"].to_numpy(float); Gc = t["Gc"].to_numpy(float)
    ratio = G / (Gc + 1e-30)

    # 1) G and Gc log
    fig = plt.figure(figsize=(7.5, 4.8)); ax = fig.add_subplot(111)
    ax.plot(t["step"], G,  marker="o", label="G")
    ax.plot(t["step"], Gc, marker="o", label="Gc")
    ax.set_yscale("log"); ax.set_xlabel("Step")
    ax.set_ylabel("Energy (MPa·mm, log)"); ax.grid(True, which="both"); ax.legend()
    ax.set_title(title + " (log G, Gc)")
    fig.tight_layout(); fig.savefig(out_prefix + "_G_Gc_log.png", dpi=300); plt.close(fig)

    # 2) ratio log
    fig = plt.figure(figsize=(7.5, 4.8)); ax = fig.add_subplot(111)
    ax.plot(t["step"], ratio, marker="o", label="G/Gc")
    ax.axhline(1.0, linestyle="--"); ax.set_yscale("log"); ax.set_xlabel("Step")
    ax.set_ylabel("G/Gc (log)"); ax.grid(True, which="both"); ax.legend()
    ax.set_title(title + " (log G/Gc)")
    fig.tight_layout(); fig.savefig(out_prefix + "_G_over_Gc_log.png", dpi=300); plt.close(fig)

    # 3) ratio linear clipped (so spikes don't destroy the plot)
    cap = max(float(np.nanpercentile(ratio, 98)), 2.0)
    ratio_clip = np.clip(ratio, 0.0, cap)
    fig = plt.figure(figsize=(7.5, 4.8)); ax = fig.add_subplot(111)
    ax.plot(t["step"], ratio_clip, marker="o", label="G/Gc (clipped p98)")
    ax.axhline(1.0, linestyle="--"); ax.set_xlabel("Step")
    ax.set_ylabel("G/Gc (clipped)"); ax.grid(True); ax.legend()
    ax.set_title(title + " (linear, clipped)")
    fig.tight_layout(); fig.savefig(out_prefix + "_G_over_Gc_clip.png", dpi=300); plt.close(fig)


# ==========================================================
# Globals: solver module handle + SIF hooks
# ==========================================================
bd = None
SIF_FUN = None
TRY_SIF = None


def validate_solver_api(mod):
    required = [
        "map_angle_to_alpha", "eval_stress_field_material",
        "fit_orthotropic_airy_disk", "solve_cracked_disk_correction_ddm",
        "_extract_tip_sifs", "_seg_intersect",
        "G_from_K_orthotropic", "Gc_theta_weak_plane",
    ]
    missing = [n for n in required if not hasattr(mod, n)]
    if missing:
        raise RuntimeError("Solver missing required API: " + ", ".join(missing))


def bind_sif_hooks(mod):
    global SIF_FUN, TRY_SIF
    SIF_FUN = getattr(mod, "sif_two_tips_from_crack", None)
    TRY_SIF = getattr(mod, "_try_call_sif_two_tips", None)
    if (SIF_FUN is None) or (TRY_SIF is None):
        try:
            from cracked_disk_ddm import sif_two_tips_from_crack as _sif
            from cracked_disk_ddm import _try_call_sif_two_tips as _try
            if SIF_FUN is None: SIF_FUN = _sif
            if TRY_SIF is None: TRY_SIF = _try
        except Exception:
            pass
    if (SIF_FUN is None) or (TRY_SIF is None):
        raise RuntimeError(
            "Could not bind SIF hooks.\n"
            "Need sif_two_tips_from_crack and _try_call_sif_two_tips either in solver module\n"
            "or importable from cracked_disk_ddm."
        )


# ==========================================================
# Unit conversion helper for Gc
# MPa + mm unit system: G and Gc have units MPa·mm = N/mm
# Conversion: Gc [MPa·mm] = Gc [J/m²] / 1000
#   because 1 J/m² = 1 N/m = 0.001 N/mm = 0.001 MPa·mm
# ==========================================================
def Gc_internal_from_Jm2(Gc0_Jm2: float, unit_mode: str = "MPA_MM") -> float:
    um = str(unit_mode).strip().upper()
    if um in ("MPA_MM", "MPA"):
        # MPa·mm = N/mm;  1 J/m² = 1e-3 N/mm
        return float(Gc0_Jm2) * 1e-3
    if um == "PA":
        return float(Gc0_Jm2)
    if um == "AUTO":
        return float(Gc0_Jm2) * 1e-3
    raise ValueError(f"Unknown unit_mode: {unit_mode}  (use MPA_MM, PA, AUTO)")


# ==========================================================
# ENERGY step (physics) with Local Symmetry tie-break
# All lengths in mm, stresses in MPa
# ==========================================================
def energy_step_physics_local_symmetry(
    xs_u, ys_u,
    R, alpha_const, airy_fit,
    E1, E2, nu12, G12,
    ds,
    cap_deg=60.0, kink_n=121,

    # prefilter
    use_kink_prefilter=True,
    K_refine=17,
    prefilter_min_segs=8,
    fallback_full_scan=True,

    # toughness
    Gc0_internal=1.0,
    weak_reduction_base=0.0,
    eta_deg=10.0,
    alpha_wp_line=0.0,

    # ddm
    ddm_sample_rs=None,
    cod_offset=0.1,
    corr_cache=None,
    N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10,

    # selection
    criterion="delta",
    score_degeneracy_frac=0.005,
    debug=False,
):
    if corr_cache is None:
        corr_cache = {}

    tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
    psi_tip = float(np.arctan2(ys_u[-1] - ys_u[-2], xs_u[-1] - xs_u[-2]))

    # ---- correction cache
    stamp = (int(len(xs_u)), float(xs_u[-2]), float(ys_u[-2]), float(xs_u[-1]), float(ys_u[-1]))
    corr = corr_cache.get("corr", None) if corr_cache.get("stamp", None) == stamp else None
    if corr is None:
        corr = bd.solve_cracked_disk_correction_ddm(
            float(R), float(alpha_const), airy_fit,
            np.asarray(xs_u, float), np.asarray(ys_u, float),
            float(E1), float(E2), float(nu12), float(G12),
            bd.eval_stress_field_material,
            N_outer=int(N_outer),
            colloc_eps_frac=float(colloc_eps_frac),
            reg_lam=float(reg_lam),
        )
        corr_cache["stamp"] = stamp
        corr_cache["corr"] = corr

    # ---- scan angles
    cap = np.deg2rad(float(cap_deg))
    kink_n = int(max(9, kink_n))
    if kink_n % 2 == 0:
        kink_n += 1

    dth = np.linspace(-cap, cap, kink_n)
    psi_new = psi_tip + dth
    nx = tipx + float(ds) * np.cos(psi_new)
    ny = tipy + float(ds) * np.sin(psi_new)

    inside  = (nx * nx + ny * ny) < (0.999 * float(R)) ** 2
    forward = np.cos(dth) > 1e-12
    ok = inside & forward
    if not np.any(ok):
        if debug:
            print("[debug] no candidate points inside disk (ds too large or tip near boundary)")
        return None

    cand_idx = np.where(ok)[0]

    # ---- optional kink prefilter (only when crack is long enough)
    do_prefilter = bool(use_kink_prefilter) and (len(xs_u) - 1 >= int(prefilter_min_segs))

    if do_prefilter:
        sif0 = TRY_SIF(
            SIF_FUN,
            np.asarray(xs_u, float), np.asarray(ys_u, float),
            R=float(R), alpha_const=float(alpha_const), airy_fit=airy_fit,
            eval_stress_field_material=bd.eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            correction=corr, sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
        )
        _, _, KI0, KII0 = bd._extract_tip_sifs(sif0)
        if not (np.isfinite(KI0) and np.isfinite(KII0)):
            do_prefilter = False

    if do_prefilter:
        kI_app, kII_app = kink_sif_transform_isotropic(float(KI0), float(KII0), dth[cand_idx])
        G_app = np.full_like(kI_app, np.nan, dtype=float)
        for i in range(len(cand_idx)):
            Gtmp, _ = bd.G_from_K_orthotropic(
                float(kI_app[i]), float(kII_app[i]),
                E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
                alpha_const=float(alpha_const),
                psi_tip_global=float(psi_new[cand_idx[i]]),
            )
            G_app[i] = float(Gtmp) if np.isfinite(Gtmp) else np.nan

        theta_line = wrap_pi_half(psi_new[cand_idx])
        Gc_app = np.array([
            bd.Gc_theta_weak_plane(
                theta_line=float(theta_line[i]), alpha_wp=float(alpha_wp_line),
                Gc_matrix=float(Gc0_internal), weak_reduction=float(weak_reduction_base),
                eta_deg=float(eta_deg),
            )
            for i in range(len(cand_idx))
        ], dtype=float)

        score_app = G_app - Gc_app if str(criterion).lower() != "ratio" else G_app / (Gc_app + 1e-30)
        good = np.isfinite(score_app)
        if np.any(good):
            order = np.argsort(score_app[good])[::-1]
            sel = cand_idx[good][order]
            if K_refine is not None and int(K_refine) > 0:
                sel = sel[: int(min(int(K_refine), len(sel)))]
            cand_idx_pref = sel
        else:
            cand_idx_pref = np.array([], dtype=int)

        if (len(cand_idx_pref) == 0) and bool(fallback_full_scan):
            cand_idx_use = cand_idx
        else:
            cand_idx_use = cand_idx_pref if len(cand_idx_pref) > 0 else cand_idx
    else:
        cand_idx_use = cand_idx

    # ---- full energetic evaluation (DDM) with Local Symmetry tie-break
    best = None
    tried = 0
    finite = 0

    for ii in cand_idx_use:
        tried += 1
        xtrial = np.asarray(list(xs_u) + [float(nx[ii])], float)
        ytrial = np.asarray(list(ys_u) + [float(ny[ii])], float)

        # self-intersection guard
        if len(xs_u) > 25:
            p1 = (xs_u[-1], ys_u[-1]); p2 = (float(nx[ii]), float(ny[ii]))
            bad = False
            for j in range(2, len(xs_u) - 10):
                q1 = (xs_u[j - 1], ys_u[j - 1]); q2 = (xs_u[j], ys_u[j])
                if bd._seg_intersect(p1, p2, q1, q2):
                    bad = True; break
            if bad:
                continue

        sif_res = TRY_SIF(
            SIF_FUN, xtrial, ytrial,
            R=float(R), alpha_const=float(alpha_const), airy_fit=airy_fit,
            eval_stress_field_material=bd.eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            correction=corr, sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
        )
        _, _, KI, KII = bd._extract_tip_sifs(sif_res)
        if not (np.isfinite(KI) and np.isfinite(KII)):
            continue
        finite += 1

        G, _H = bd.G_from_K_orthotropic(
            float(KI), float(KII),
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            alpha_const=float(alpha_const), psi_tip_global=float(psi_new[ii]),
        )
        if not np.isfinite(G):
            continue

        theta_line_i = float(wrap_pi_half_scalar(float(psi_new[ii])))
        Gc = bd.Gc_theta_weak_plane(
            theta_line=float(theta_line_i), alpha_wp=float(alpha_wp_line),
            Gc_matrix=float(Gc0_internal), weak_reduction=float(weak_reduction_base),
            eta_deg=float(eta_deg),
        )

        if str(criterion).lower() == "ratio":
            score = float(G) / (float(Gc) + 1e-30)
            ok_growth = score > 1.0
        else:
            score = float(G) - float(Gc)
            ok_growth = score > 0.0

        if not ok_growth:
            continue

        kIIratio = float(abs(KII) / (abs(KI) + 1e-30))
        abs_dth  = float(abs(dth[ii]))

        if best is None:
            best = dict(
                nx=float(nx[ii]), ny=float(ny[ii]), psi_new=float(psi_new[ii]),
                KI=float(KI), KII=float(KII), G=float(G), Gc=float(Gc),
                score=float(score), kIIratio=float(kIIratio), abs_dth=float(abs_dth),
            )
            continue

        # primary: maximize score
        score_best = float(best["score"])
        tol = float(score_degeneracy_frac) * max(1e-12, abs(score_best))

        if score > score_best + tol:
            best.update(nx=float(nx[ii]), ny=float(ny[ii]), psi_new=float(psi_new[ii]),
                        KI=float(KI), KII=float(KII), G=float(G), Gc=float(Gc),
                        score=float(score), kIIratio=float(kIIratio), abs_dth=float(abs_dth))
        elif abs(score - score_best) <= tol:
            # Local Symmetry tie-break: minimize |KII|/|KI|
            if kIIratio < float(best["kIIratio"]) - 1e-12:
                best.update(nx=float(nx[ii]), ny=float(ny[ii]), psi_new=float(psi_new[ii]),
                            KI=float(KI), KII=float(KII), G=float(G), Gc=float(Gc),
                            score=float(score), kIIratio=float(kIIratio), abs_dth=float(abs_dth))
            elif abs(kIIratio - float(best["kIIratio"])) <= 1e-12:
                # final tie-break: smaller kink for smoothness
                if abs_dth < float(best["abs_dth"]):
                    best.update(nx=float(nx[ii]), ny=float(ny[ii]), psi_new=float(psi_new[ii]),
                                KI=float(KI), KII=float(KII), G=float(G), Gc=float(Gc),
                                score=float(score), kIIratio=float(kIIratio), abs_dth=float(abs_dth))

    if debug:
        print(f"[debug] tried={tried}, finite(KI,KII)={finite}, best={'yes' if best else 'no'}")

    return best


_TRACE_COLUMNS = [
    "step", "x", "y", "psi", "KI", "KII", "G", "Gc", "score", "kIIratio", "abs_dth",
    "cap_deg", "kink_n", "ds", "criterion"
]


# ==========================================================
# Build + run propagation  — lengths stay in mm throughout
# ==========================================================
def build_fields_and_run(
    dfmeta_row,
    ds_frac=0.010,
    max_steps=1200,
    N_outer=80,
    psi0_deg=90.0,
    psi0_delta_deg=0.0,
    a0_frac=0.016,

    # material (input in GPa, converted to MPa inside)
    E1_GPa=50.0, E2_GPa=30.0, nu12=0.25, G12_GPa=12.0,

    # platens / airy
    platen_half_angle_deg=10.0, platen_smooth_deg=4.0, platen_mu=0.0,
    airy_auto_tune=True, airy_Nbd_base=420, airy_Nbd_arc_each=900,
    load_factor=1.0,

    # weak plane
    eta_deg=10.0,
    weak_T_ratio=0.35,

    # toughness  (Gc0_Jm2 converted to MPa·mm = N/mm via /1000)
    Gc0_Jm2=50.0,
    unit_mode="MPA_MM",
    Gc0_internal_override=None,
    weak_reduction_base=np.nan,

    # end condition
    r_end_frac=0.995,
    ddm_sample_rs=None,
    cod_offset=0.1,

    # propagation controls
    cap_deg=60.0,
    kink_n=121,
    use_kink_prefilter=True,
    K_refine=17,
    prefilter_min_segs=8,
    fallback_full_scan=True,
    criterion="delta",

    # postprocess
    postprocess_single_curve=True,
    pca_ratio_thresh=35.0,

    debug=False,
):
    rock, ang_deg, Dmm, tmm, PkN, Tm_in, Coh_in, Phi_in = dfmeta_row

    # Elastic constants: GPa -> MPa
    E1  = float(E1_GPa)  * 1e3
    E2  = float(E2_GPa)  * 1e3
    G12 = float(G12_GPa) * 1e3
    nu12 = float(nu12)

    alpha_const  = float(bd.map_angle_to_alpha(float(ang_deg), angle_map="direct"))
    alpha_wp_line = float(wrap_pi_half_scalar(alpha_const))

    # Geometry — keep in mm (solver expects mm)
    D  = float(Dmm)          # mm
    t  = float(tmm)          # mm
    R  = D / 2.0             # mm

    # Load: kN -> N  (solver expects N if lengths in mm and stresses in MPa)
    P = float(PkN) * 1e3 * float(load_factor)   # N

    if debug:
        print(f"[debug] D={D:.1f} mm  R={R:.1f} mm  t={t:.1f} mm  P={P:.1f} N")

    # Airy fit
    if bool(airy_auto_tune) and (getattr(bd, "fit_orthotropic_airy_disk_auto", None) is not None):
        fit = bd.fit_orthotropic_airy_disk_auto(
            E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
            beta_deg=float(platen_half_angle_deg), smooth_deg=float(platen_smooth_deg),
            mu=float(platen_mu), Nbd=int(airy_Nbd_base), Nbd_arc_each=int(airy_Nbd_arc_each),
        )
    else:
        fit = bd.fit_orthotropic_airy_disk(
            E1, E2, nu12, G12, R=R, t=t, P=P, alpha=alpha_const,
            M=24, Nbd=int(airy_Nbd_base),
            beta_deg=float(platen_half_angle_deg), smooth_deg=float(platen_smooth_deg),
            mu=float(platen_mu), lam=1e-8, w_arc=12.0, Nbd_arc_each=int(airy_Nbd_arc_each),
        )

    # Toughness conversion: J/m² -> MPa·mm  (divide by 1000)
    if Gc0_internal_override is None:
        Gc0_use = Gc_internal_from_Jm2(float(Gc0_Jm2), unit_mode=str(unit_mode))
    else:
        Gc0_use = float(Gc0_internal_override)

    if debug:
        print(f"[debug] unit_mode={unit_mode}  Gc0_Jm2={Gc0_Jm2:g} J/m²  =>  Gc0_internal={Gc0_use:.6e} MPa·mm")

    # weak reduction base (material constant)
    if not np.isfinite(float(weak_reduction_base)):
        weak_reduction_base = float(np.clip(1.0 - float(weak_T_ratio), 0.0, 0.90))
    else:
        weak_reduction_base = float(np.clip(float(weak_reduction_base), 0.0, 0.90))

    if ddm_sample_rs is None:
        # sample_rs in mm — fractions of R
        ddm_sample_rs = np.logspace(np.log10(0.01 * R), np.log10(0.5 * R), 6)

    ds0 = float(ds_frac) * R                   # mm
    a0  = float(a0_frac) * R                   # mm
    psi0 = np.deg2rad(float(psi0_deg) + float(psi0_delta_deg))

    xs_u = [0.0, a0 * np.cos(float(psi0))]
    ys_u = [0.0, a0 * np.sin(float(psi0))]

    corr_cache = {}
    trace_rows = []
    stop_reason = "max_steps"

    for step in range(int(max_steps)):
        tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
        if np.hypot(tipx, tipy) >= float(r_end_frac) * R:
            stop_reason = "reached_boundary"
            break

        best = energy_step_physics_local_symmetry(
            xs_u, ys_u,
            R=R, alpha_const=alpha_const, airy_fit=fit,
            E1=E1, E2=E2, nu12=nu12, G12=G12,
            ds=ds0,
            cap_deg=float(cap_deg), kink_n=int(kink_n),
            use_kink_prefilter=bool(use_kink_prefilter),
            K_refine=int(K_refine), prefilter_min_segs=int(prefilter_min_segs),
            fallback_full_scan=bool(fallback_full_scan),
            Gc0_internal=float(Gc0_use), weak_reduction_base=float(weak_reduction_base),
            eta_deg=float(eta_deg), alpha_wp_line=float(alpha_wp_line),
            ddm_sample_rs=ddm_sample_rs, cod_offset=float(cod_offset),
            corr_cache=corr_cache, N_outer=int(N_outer),
            criterion=str(criterion), debug=debug,
        )

        if best is None:
            stop_reason = "no_admissible_growth_or_nan_sif"
            break

        xs_u.append(best["nx"]); ys_u.append(best["ny"])
        trace_rows.append(dict(
            step=step, x=best["nx"], y=best["ny"], psi=best["psi_new"],
            KI=best["KI"], KII=best["KII"], G=best["G"], Gc=best["Gc"],
            score=best["score"], kIIratio=best["kIIratio"], abs_dth=best["abs_dth"],
            cap_deg=float(cap_deg), kink_n=int(kink_n), ds=float(ds0), criterion=str(criterion),
        ))

    # mirror to full crack
    xu = np.asarray(xs_u, float); yu = np.asarray(ys_u, float)
    xs = np.concatenate([(-xu)[::-1], xu[1:]])
    ys = np.concatenate([(-yu)[::-1], yu[1:]])

    # extend ends if needed
    if len(xs) >= 4:
        if np.hypot(xs[0],  ys[0])  < 0.98 * R:
            xs[0],  ys[0]  = extend_to_circle((xs[1],  ys[1]),  (xs[0],  ys[0]),  R)
        if np.hypot(xs[-1], ys[-1]) < 0.98 * R:
            xs[-1], ys[-1] = extend_to_circle((xs[-2], ys[-2]), (xs[-1], ys[-1]), R)

    # postprocess (optional) to remove chatter/loops when nearly straight
    if bool(postprocess_single_curve):
        xs, ys = postprocess_best_crack(xs, ys, R, pca_ratio_thresh=float(pca_ratio_thresh))

    trace = pd.DataFrame(trace_rows)
    for c in _TRACE_COLUMNS:
        if c not in trace.columns:
            trace[c] = np.nan
    trace = trace[_TRACE_COLUMNS]

    return dict(
        rock=str(rock), angle_deg=float(ang_deg),
        alpha_const=float(alpha_const), alpha_wp_line=float(alpha_wp_line),
        R=float(R),
        xs=xs, ys=ys,
        trace=trace,
        stop_reason=str(stop_reason),
        Gc0_internal_used=float(Gc0_use),
        unit_mode=str(unit_mode).upper(),
    )


# ==========================================================
# Growth probe + auto-fix
# ==========================================================
def _probe_growth(row, debug=False, **run_kwargs):
    # IMPORTANT: do NOT hardcode N_outer/ds_frac here — run_kwargs controls it
    res = build_fields_and_run(row, max_steps=10, debug=bool(debug), **run_kwargs)
    return int(len(res["trace"].dropna(subset=["step"]))), res

def auto_find_growth_params(row,
                            base_load_factor, base_Gc0_Jm2, base_a0_frac, base_cod_offset,
                            min_steps_accept=6, debug=False, **run_kwargs):
    a0_list   = [base_a0_frac, 0.03, 0.05, 0.08, 0.12]
    load_list = [base_load_factor, 1.5*base_load_factor, 2*base_load_factor,
                 3*base_load_factor, 5*base_load_factor, 10*base_load_factor]
    gc_list   = [base_Gc0_Jm2, base_Gc0_Jm2/2, base_Gc0_Jm2/5,
                 base_Gc0_Jm2/10, base_Gc0_Jm2/50, 1e-3]
    cod_list  = [base_cod_offset, 0.05, 0.1, 0.2, 0.5]

    best_any = None
    for a0 in a0_list:
        for lf in load_list:
            for gc in gc_list:
                for co in cod_list:
                    nsteps, res = _probe_growth(
                        row, load_factor=float(lf), Gc0_Jm2=float(gc),
                        a0_frac=float(a0), cod_offset=float(co), debug=False, **run_kwargs
                    )
                    if best_any is None or nsteps > best_any[0]:
                        best_any = (nsteps, res, lf, gc, a0, co)
                    if nsteps >= int(min_steps_accept):
                        if debug:
                            print(f"[auto] ok: steps={nsteps} load={lf:g} Gc0={gc:g} a0={a0:g} cod={co:g}")
                        return dict(load_factor=float(lf), Gc0_Jm2=float(gc),
                                    a0_frac=float(a0), cod_offset=float(co))

    if best_any is not None:
        nsteps, res, lf, gc, a0, co = best_any
        if nsteps > 0:
            if debug:
                print(f"[auto] best: steps={nsteps} load={lf:g} Gc0={gc:g} a0={a0:g} cod={co:g}")
            return dict(load_factor=float(lf), Gc0_Jm2=float(gc),
                        a0_frac=float(a0), cod_offset=float(co))

    reason = best_any[1].get("stop_reason", "unknown") if best_any else "unknown"
    raise RuntimeError(
        f"Auto-search failed: zero growth for all tested parameters.\nLast stop_reason={reason}"
    )

def ensure_growth_params(row, *, load_factor, Gc0_Jm2, a0_frac, cod_offset,
                         auto_fix=False, debug=False, min_steps_accept=6, **run_kwargs):
    nsteps, probe = _probe_growth(
        row, load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
        a0_frac=float(a0_frac), cod_offset=float(cod_offset), debug=False, **run_kwargs
    )
    if debug:
        print(f"[probe] nsteps={nsteps} stop_reason={probe.get('stop_reason')}")

    if nsteps > 0 and nsteps < int(min_steps_accept) and not bool(auto_fix):
        print(f"[warn] Growth is short (nsteps={nsteps} < {min_steps_accept}) but continuing.\n"
              "       Pass --auto_fix for auto-tuning.")
        return dict(load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
                    a0_frac=float(a0_frac), cod_offset=float(cod_offset))

    if nsteps >= int(min_steps_accept):
        return dict(load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
                    a0_frac=float(a0_frac), cod_offset=float(cod_offset))

    if not bool(auto_fix):
        raise RuntimeError(
            "No crack growth with provided parameters (auto-fix disabled).\n"
            "Try: --auto_fix OR increase --load_factor OR decrease --Gc0_Jm2 OR increase --cod_offset."
        )
    return auto_find_growth_params(
        row, base_load_factor=float(load_factor), base_Gc0_Jm2=float(Gc0_Jm2),
        base_a0_frac=float(a0_frac), base_cod_offset=float(cod_offset),
        min_steps_accept=int(min_steps_accept), debug=bool(debug), **run_kwargs
    )


# ==========================================================
# SMOKE / SMOKE_ALL / VERIFY / VALIDATE
# ==========================================================
META_COLS = [
    "Rock_type", "Angle", "Diameter_mm", "Thickness_mm", "Load_(KN)",
    "Tensile_strength_Mpa", "Cohesion", "Friction_Angle"
]

def run_smoke(dfmeta, sample_id, out_dir,
              load_factor=1.0, Gc0_Jm2=50.0, a0_frac=0.016, cod_offset=0.1,
              auto_fix=False, target_steps=200, debug=False,
              criterion="delta", cap_deg=60.0, kink_n=121,
              use_kink_prefilter=True, K_refine=17,
              prefilter_min_segs=8, fallback_full_scan=True,
              N_outer=80, ds_frac=0.010, unit_mode="MPA_MM",
              postprocess_single_curve=True, pca_ratio_thresh=35.0):
    os.makedirs(out_dir, exist_ok=True)
    row = dfmeta.loc[sample_id, META_COLS]

    used = ensure_growth_params(
        row, load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
        a0_frac=float(a0_frac), cod_offset=float(cod_offset),
        auto_fix=bool(auto_fix), debug=bool(debug), min_steps_accept=6,
        criterion=str(criterion), cap_deg=float(cap_deg), kink_n=int(kink_n),
        use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
        prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
        N_outer=int(N_outer), ds_frac=float(ds_frac), unit_mode=str(unit_mode),
        postprocess_single_curve=bool(postprocess_single_curve),
        pca_ratio_thresh=float(pca_ratio_thresh),
    )
    with open(os.path.join(out_dir, f"used_params_smoke_sid{sample_id}.json"), "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2)

    res = build_fields_and_run(
        row, ds_frac=float(ds_frac), max_steps=int(target_steps), N_outer=int(N_outer),
        load_factor=float(used["load_factor"]), Gc0_Jm2=float(used["Gc0_Jm2"]),
        a0_frac=float(used["a0_frac"]), cod_offset=float(used["cod_offset"]),
        cap_deg=float(cap_deg), kink_n=int(kink_n),
        use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
        prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
        criterion=str(criterion), unit_mode=str(unit_mode),
        postprocess_single_curve=bool(postprocess_single_curve),
        pca_ratio_thresh=float(pca_ratio_thresh), debug=bool(debug),
    )

    print(f"=== SMOKE sid={sample_id} (physics + local symmetry) ===")
    print("stop_reason :", res["stop_reason"])
    print("nsteps      :", int(len(res["trace"].dropna(subset=["step"]))))
    print("Gc0_internal:", res["Gc0_internal_used"], "MPa·mm")
    print("unit_mode   :", res["unit_mode"])
    print("Saved to    :", out_dir)

    pd.DataFrame({"x_mm": res["xs"], "y_mm": res["ys"]}).to_csv(
        os.path.join(out_dir, f"smoke_path_sid{sample_id}.csv"), index=False)
    res["trace"].to_csv(os.path.join(out_dir, f"smoke_trace_sid{sample_id}.csv"), index=False)
    save_path_plot(res["xs"], res["ys"], res["R"],
                   os.path.join(out_dir, f"smoke_path_sid{sample_id}.png"),
                   title=f"Smoke path (sid {sample_id})")
    save_energy_plots(res["trace"], os.path.join(out_dir, f"smoke_energy_sid{sample_id}"),
                      title=f"Driving force vs toughness (criterion={criterion})")

    res["sample_id"] = sample_id
    return res


def run_smoke_all(dfmeta, out_dir,
                  sample_ids=None,
                  load_factor=1.0, Gc0_Jm2=50.0, a0_frac=0.016, cod_offset=0.1,
                  auto_fix=True, target_steps=200, debug=False,
                  criterion="delta", cap_deg=60.0, kink_n=121,
                  use_kink_prefilter=True, K_refine=17,
                  prefilter_min_segs=8, fallback_full_scan=True,
                  N_outer=80, ds_frac=0.010, unit_mode="MPA_MM",
                  postprocess_single_curve=True, pca_ratio_thresh=35.0):
    """Run smoke for samples 1–7 and plot ALL crack paths in ONE combined figure (augen_geniss)."""
    os.makedirs(out_dir, exist_ok=True)
    if sample_ids is None:
        sample_ids = list(range(1, 8))

    results_list = []
    for sid in sample_ids:
        if sid not in dfmeta.index:
            print(f"[smoke_all] sample_id={sid} not in meta CSV — skipping.")
            continue
        print(f"\n[smoke_all] ===== sample_id={sid} =====")
        sid_out = os.path.join(out_dir, f"sid{sid}")
        os.makedirs(sid_out, exist_ok=True)
        try:
            res = run_smoke(
                dfmeta, sid, sid_out,
                load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
                a0_frac=float(a0_frac), cod_offset=float(cod_offset),
                auto_fix=bool(auto_fix), target_steps=int(target_steps), debug=bool(debug),
                criterion=str(criterion), cap_deg=float(cap_deg), kink_n=int(kink_n),
                use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
                prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
                N_outer=int(N_outer), ds_frac=float(ds_frac), unit_mode=str(unit_mode),
                postprocess_single_curve=bool(postprocess_single_curve),
                pca_ratio_thresh=float(pca_ratio_thresh),
            )
            results_list.append(res)
        except Exception as exc:
            print(f"[smoke_all] sample_id={sid} FAILED: {exc}")

    if len(results_list) == 0:
        print("[smoke_all] No successful runs — combined plot skipped.")
        return

    combined_png = os.path.join(out_dir, "augen_geniss_all_paths.png")
    save_all_paths_plot(results_list, out_path=combined_png,
                        title="augen_geniss — crack paths samples 1–7")

    summary_rows = [dict(
        sample_id=r["sample_id"], rock=r["rock"], angle_deg=r["angle_deg"], R_mm=r["R"],
        stop_reason=r["stop_reason"],
        nsteps=int(len(r["trace"].dropna(subset=["step"]))),
        Gc0_internal_MPa_mm=r["Gc0_internal_used"], unit_mode=r["unit_mode"],
    ) for r in results_list]
    summary_csv = os.path.join(out_dir, "augen_geniss_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n[smoke_all] Summary CSV : {summary_csv}")
    print(f"[smoke_all] Combined PNG: {combined_png}")


def run_verification(dfmeta, sample_id, out_dir,
                     load_factor=1.0, Gc0_Jm2=50.0, a0_frac=0.016, cod_offset=0.1,
                     auto_fix=False, debug=False, target_steps=1200,
                     criterion="delta", cap_deg=60.0, kink_n=101,
                     use_kink_prefilter=True, K_refine=17,
                     prefilter_min_segs=8, fallback_full_scan=True,
                     unit_mode="MPA_MM", postprocess_single_curve=True, pca_ratio_thresh=35.0):
    os.makedirs(out_dir, exist_ok=True)
    row = dfmeta.loc[sample_id, META_COLS]
    ds_list = [0.014, 0.010, 0.007, 0.005]
    Nouter_list = [50, 80, 120, 160]

    used = ensure_growth_params(
        row, load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
        a0_frac=float(a0_frac), cod_offset=float(cod_offset),
        auto_fix=bool(auto_fix), debug=bool(debug), min_steps_accept=6,
        criterion=str(criterion), cap_deg=float(cap_deg), kink_n=int(kink_n),
        use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
        prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
        N_outer=max(Nouter_list), ds_frac=min(ds_list), unit_mode=str(unit_mode),
        postprocess_single_curve=bool(postprocess_single_curve), pca_ratio_thresh=float(pca_ratio_thresh),
    )
    with open(os.path.join(out_dir, f"used_params_verify_sid{sample_id}.json"), "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2)

    ref = build_fields_and_run(
        row, ds_frac=min(ds_list), N_outer=max(Nouter_list), max_steps=int(target_steps),
        load_factor=float(used["load_factor"]), Gc0_Jm2=float(used["Gc0_Jm2"]),
        a0_frac=float(used["a0_frac"]), cod_offset=float(used["cod_offset"]),
        cap_deg=float(cap_deg), kink_n=int(kink_n),
        use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
        prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
        criterion=str(criterion), unit_mode=str(unit_mode),
        postprocess_single_curve=bool(postprocess_single_curve),
        pca_ratio_thresh=float(pca_ratio_thresh), debug=bool(debug),
    )
    x_ref, y_ref = ref["xs"], ref["ys"]
    save_path_plot(ref["xs"], ref["ys"], ref["R"],
                   os.path.join(out_dir, f"verify_ref_path_sid{sample_id}.png"),
                   title=f"Reference path (sid {sample_id})")
    save_energy_plots(ref["trace"], os.path.join(out_dir, f"verify_ref_energy_sid{sample_id}"),
                      title=f"Reference driving force vs toughness (criterion={criterion})")

    rows = []
    for ds in ds_list:
        for No in Nouter_list:
            res = build_fields_and_run(
                row, ds_frac=float(ds), N_outer=int(No), max_steps=int(target_steps),
                load_factor=float(used["load_factor"]), Gc0_Jm2=float(used["Gc0_Jm2"]),
                a0_frac=float(used["a0_frac"]), cod_offset=float(used["cod_offset"]),
                cap_deg=float(cap_deg), kink_n=int(kink_n),
                use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
                prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
                criterion=str(criterion), unit_mode=str(unit_mode),
                postprocess_single_curve=bool(postprocess_single_curve),
                pca_ratio_thresh=float(pca_ratio_thresh), debug=False,
            )
            x, y = res["xs"], res["ys"]
            tr = res["trace"]
            nsteps = int(len(tr.dropna(subset=["step"])))
            rms, dmax = path_rms_max_distance(x_ref, y_ref, x, y, n=600)
            Gmean = float(tr["G"].tail(10).mean()) if nsteps else np.nan
            rows.append(dict(ds_frac=float(ds), N_outer=int(No), nsteps=nsteps,
                             path_rms_mm=float(rms), path_max_mm=float(dmax),
                             G_last10=float(Gmean), stop_reason=str(res["stop_reason"])))
            tag = f"sid{sample_id}_ds{ds:.3f}_No{No}"
            pd.DataFrame({"x_mm": x, "y_mm": y}).to_csv(os.path.join(out_dir, f"pred_path_{tag}.csv"), index=False)
            tr.to_csv(os.path.join(out_dir, f"trace_{tag}.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"verification_summary_sid{sample_id}.csv"), index=False)
    fig = plt.figure(figsize=(8.0, 5.0)); ax = fig.add_subplot(111)
    for No in Nouter_list:
        sub = df[df["N_outer"] == No].sort_values("ds_frac")
        ax.plot(sub["ds_frac"], sub["path_rms_mm"], marker="o", label=f"N_outer={No}")
    ax.set_xlabel("ds_frac"); ax.set_ylabel("Path RMS distance to reference (mm)")
    ax.set_title(f"Convergence (sid {sample_id}, criterion={criterion})")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"verification_convergence_sid{sample_id}.png"), dpi=300)
    plt.close(fig)
    print("Saved verification outputs to:", out_dir)


def run_validation(dfmeta, sample_id, exp_path_csv, out_dir,
                   exp_scale=1.0,
                   load_factor=1.0, Gc0_Jm2=50.0, a0_frac=0.016, cod_offset=0.1,
                   auto_fix=False, debug=False, target_steps=1200,
                   criterion="delta", cap_deg=60.0, kink_n=101,
                   use_kink_prefilter=True, K_refine=17,
                   prefilter_min_segs=8, fallback_full_scan=True,
                   unit_mode="MPA_MM", postprocess_single_curve=True, pca_ratio_thresh=35.0):
    os.makedirs(out_dir, exist_ok=True)
    row = dfmeta.loc[sample_id, META_COLS]

    used = ensure_growth_params(
        row, load_factor=float(load_factor), Gc0_Jm2=float(Gc0_Jm2),
        a0_frac=float(a0_frac), cod_offset=float(cod_offset),
        auto_fix=bool(auto_fix), debug=bool(debug), min_steps_accept=6,
        criterion=str(criterion), cap_deg=float(cap_deg), kink_n=int(kink_n),
        use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
        prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
        N_outer=80, ds_frac=0.010, unit_mode=str(unit_mode),
        postprocess_single_curve=bool(postprocess_single_curve), pca_ratio_thresh=float(pca_ratio_thresh),
    )
    with open(os.path.join(out_dir, f"used_params_validate_sid{sample_id}.json"), "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2)

    pred = build_fields_and_run(
        row, ds_frac=0.010, max_steps=int(target_steps), N_outer=80,
        load_factor=float(used["load_factor"]), Gc0_Jm2=float(used["Gc0_Jm2"]),
        a0_frac=float(used["a0_frac"]), cod_offset=float(used["cod_offset"]),
        cap_deg=float(cap_deg), kink_n=int(kink_n),
        use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(K_refine),
        prefilter_min_segs=int(prefilter_min_segs), fallback_full_scan=bool(fallback_full_scan),
        criterion=str(criterion), unit_mode=str(unit_mode),
        postprocess_single_curve=bool(postprocess_single_curve),
        pca_ratio_thresh=float(pca_ratio_thresh), debug=bool(debug),
    )
    xp, yp = pred["xs"], pred["ys"]; R = pred["R"]

    exp = pd.read_csv(exp_path_csv)
    xe = exp.iloc[:, 0].to_numpy(float) * float(exp_scale)
    ye = exp.iloc[:, 1].to_numpy(float) * float(exp_scale)

    ang_p = initiation_angle_deg(xp, yp, rmax_frac=0.12, R=R)
    ang_e = initiation_angle_deg(xe, ye, rmax_frac=0.12, R=R)
    ang_err = float(wrap_pi_scalar(np.deg2rad(ang_p - ang_e)) * 180.0 / np.pi)
    rms, dmax = path_rms_max_distance(xe, ye, xp, yp, n=600)
    smmd = symmetric_mean_min_distance(xe, ye, xp, yp, n=500)

    metrics = dict(
        sample_id=int(sample_id), exp_path_csv=str(exp_path_csv), exp_scale=float(exp_scale),
        init_angle_pred_deg=float(ang_p), init_angle_exp_deg=float(ang_e),
        init_angle_error_deg=float(ang_err),
        path_rms_mm=float(rms), path_max_mm=float(dmax), sym_mean_min_dist_mm=float(smmd),
        stop_reason=str(pred["stop_reason"]),
        nsteps=int(len(pred["trace"].dropna(subset=["step"]))),
        used_load_factor=float(used["load_factor"]), used_Gc0_Jm2=float(used["Gc0_Jm2"]),
        used_a0_frac=float(used["a0_frac"]), used_cod_offset=float(used["cod_offset"]),
        criterion=str(criterion), unit_mode=str(unit_mode).upper(),
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"validation_metrics_sid{sample_id}.csv"), index=False)
    save_path_plot(xp, yp, R, os.path.join(out_dir, f"validation_overlay_sid{sample_id}.png"),
                   title=f"Validation overlay (sid {sample_id})\nΔinit={ang_err:.2f}°, RMS={rms:.3e} mm",
                   exp_xy=(xe, ye))
    save_energy_plots(pred["trace"], os.path.join(out_dir, f"validation_energy_sid{sample_id}"),
                      title=f"Validation driving force vs toughness (criterion={criterion})")
    pd.DataFrame({"x_mm": xp, "y_mm": yp}).to_csv(
        os.path.join(out_dir, f"pred_path_sid{sample_id}.csv"), index=False)
    pred["trace"].to_csv(os.path.join(out_dir, f"pred_trace_sid{sample_id}.csv"), index=False)
    print("Saved validation outputs to:", out_dir)


# ==========================================================
# CLI
# ==========================================================
def main(argv=None):
    global bd
    if argv is None:
        argv = sanitize_argv(sys.argv[1:])

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug",    action="store_true")
    common.add_argument("--auto_fix", action="store_true")

    common.add_argument("--load_factor",  type=float, default=1.0)
    common.add_argument("--Gc0_Jm2",      type=float, default=50.0)
    common.add_argument("--a0_frac",      type=float, default=0.016)
    common.add_argument("--cod_offset",   type=float, default=0.1)    # mm units default
    common.add_argument("--target_steps", type=int,   default=1200)

    common.add_argument("--criterion", type=str,   default="delta", choices=["delta", "ratio"])
    common.add_argument("--cap_deg",   type=float, default=60.0)
    common.add_argument("--kink_n",    type=int,   default=121)
    common.add_argument("--ds_frac",   type=float, default=0.010)
    common.add_argument("--N_outer",   type=int,   default=80)

    common.add_argument("--no_kink_prefilter",     action="store_true")
    common.add_argument("--K_refine",              type=int, default=17)
    common.add_argument("--prefilter_min_segs",    type=int, default=8)
    common.add_argument("--no_fallback_full_scan", action="store_true")

    common.add_argument("--unit_mode", type=str, default="MPA_MM",
                        choices=["MPA_MM", "MPA", "PA", "AUTO"])

    common.add_argument("--no_postprocess_single_curve", action="store_true")
    common.add_argument("--pca_ratio_thresh", type=float, default=35.0)

    ap = argparse.ArgumentParser(add_help=True, parents=[common])
    ap.add_argument("--solver_py",  default=os.environ.get("SOLVER_PY", ""))
    ap.add_argument("--solver_mod", default=os.environ.get("SOLVER_MOD", "bound_all_helpers"))
    ap.add_argument("--meta_csv",   default="tensile_samples_data.csv")
    ap.add_argument("--out_dir",    default="jmps_out_physics_ls")
    ap.add_argument("--sample_id",  type=int, default=1)

    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("smoke",     parents=[common])
    sub.add_parser("verify",    parents=[common])
    sub.add_parser("smoke_all", parents=[common])

    spv = sub.add_parser("validate", parents=[common])
    spv.add_argument("--exp_path_csv", required=True)
    spv.add_argument("--exp_scale", type=float, default=1.0)   # mm default (no conversion needed)

    if len(argv) == 0:
        argv = ["smoke", "--debug", "--target_steps", "200"]

    args = ap.parse_args(argv)
    if args.cmd is None:
        args.cmd = "smoke"

    # load solver
    if str(args.solver_py).strip():
        bd = load_module_from_path(str(args.solver_py).strip(), module_name="bd")
    else:
        bd = import_or_load_solver(args.solver_mod, meta_csv=args.meta_csv, debug=args.debug)

    validate_solver_api(bd)
    bind_sif_hooks(bd)

    dfmeta = pd.read_csv(args.meta_csv, index_col=0)
    dfmeta.columns = [str(c).strip() for c in dfmeta.columns]

    use_kink_prefilter       = not bool(args.no_kink_prefilter)
    fallback_full_scan       = not bool(args.no_fallback_full_scan)
    postprocess_single_curve = not bool(args.no_postprocess_single_curve)

    out_dir_base = args.out_dir

    # ---- smoke_all: loops over all sample IDs, no single sample_id check
    if args.cmd == "smoke_all":
        os.makedirs(out_dir_base, exist_ok=True)
        run_smoke_all(
            dfmeta, out_dir_base, sample_ids=list(range(1, 8)),
            load_factor=float(args.load_factor), Gc0_Jm2=float(args.Gc0_Jm2),
            a0_frac=float(args.a0_frac), cod_offset=float(args.cod_offset),
            auto_fix=bool(args.auto_fix), target_steps=int(args.target_steps),
            debug=bool(args.debug), criterion=str(args.criterion),
            cap_deg=float(args.cap_deg), kink_n=int(args.kink_n),
            use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(args.K_refine),
            prefilter_min_segs=int(args.prefilter_min_segs),
            fallback_full_scan=bool(fallback_full_scan),
            N_outer=int(args.N_outer), ds_frac=float(args.ds_frac),
            unit_mode=str(args.unit_mode),
            postprocess_single_curve=bool(postprocess_single_curve),
            pca_ratio_thresh=float(args.pca_ratio_thresh),
        )
        return

    # ---- single-sample commands
    if args.sample_id not in dfmeta.index:
        raise RuntimeError(f"sample_id {args.sample_id} not found in meta_csv index.")
    out_dir = os.path.join(out_dir_base, f"sid{args.sample_id}")
    os.makedirs(out_dir, exist_ok=True)

    if args.cmd == "smoke":
        run_smoke(
            dfmeta, args.sample_id, out_dir,
            load_factor=float(args.load_factor), Gc0_Jm2=float(args.Gc0_Jm2),
            a0_frac=float(args.a0_frac), cod_offset=float(args.cod_offset),
            auto_fix=bool(args.auto_fix), target_steps=int(args.target_steps),
            debug=bool(args.debug), criterion=str(args.criterion),
            cap_deg=float(args.cap_deg), kink_n=int(args.kink_n),
            use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(args.K_refine),
            prefilter_min_segs=int(args.prefilter_min_segs),
            fallback_full_scan=bool(fallback_full_scan),
            N_outer=int(args.N_outer), ds_frac=float(args.ds_frac),
            unit_mode=str(args.unit_mode),
            postprocess_single_curve=bool(postprocess_single_curve),
            pca_ratio_thresh=float(args.pca_ratio_thresh),
        )

    elif args.cmd == "verify":
        run_verification(
            dfmeta, args.sample_id, out_dir,
            load_factor=float(args.load_factor), Gc0_Jm2=float(args.Gc0_Jm2),
            a0_frac=float(args.a0_frac), cod_offset=float(args.cod_offset),
            auto_fix=bool(args.auto_fix), debug=bool(args.debug),
            target_steps=int(args.target_steps), criterion=str(args.criterion),
            cap_deg=float(args.cap_deg), kink_n=int(args.kink_n),
            use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(args.K_refine),
            prefilter_min_segs=int(args.prefilter_min_segs),
            fallback_full_scan=bool(fallback_full_scan),
            unit_mode=str(args.unit_mode),
            postprocess_single_curve=bool(postprocess_single_curve),
            pca_ratio_thresh=float(args.pca_ratio_thresh),
        )

    elif args.cmd == "validate":
        run_validation(
            dfmeta, args.sample_id, args.exp_path_csv, out_dir,
            exp_scale=float(args.exp_scale),
            load_factor=float(args.load_factor), Gc0_Jm2=float(args.Gc0_Jm2),
            a0_frac=float(args.a0_frac), cod_offset=float(args.cod_offset),
            auto_fix=bool(args.auto_fix), debug=bool(args.debug),
            target_steps=int(args.target_steps), criterion=str(args.criterion),
            cap_deg=float(args.cap_deg), kink_n=int(args.kink_n),
            use_kink_prefilter=bool(use_kink_prefilter), K_refine=int(args.K_refine),
            prefilter_min_segs=int(args.prefilter_min_segs),
            fallback_full_scan=bool(fallback_full_scan),
            unit_mode=str(args.unit_mode),
            postprocess_single_curve=bool(postprocess_single_curve),
            pca_ratio_thresh=float(args.pca_ratio_thresh),
        )

if __name__ == "__main__":
    main()