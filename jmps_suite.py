#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JMPS Brazilian-disk crack-path study suite (single combined script)

Key changes vs your earlier versions:
  ✅ Units are EXPLICIT (no heuristic inference). You MUST set --stress_unit {MPa,Pa}.
     - stress_unit=MPa => solver stresses in MPa, moduli in MPa, strengths in MPa
     - stress_unit=Pa  => solver stresses in Pa,  moduli in Pa,  strengths in Pa
     - Fracture energy G, Gc:
         internal_unit = (stress_unit) * m
         J/m^2 = internal * stress_scale, where stress_scale = 1e6 (MPa) or 1 (Pa)

  ✅ Auto-fix (parameter search) is a separate "calibrate_growth" command,
     NOT used in verification/validation unless you explicitly pass --param_json.

  ✅ Verification (convergence) generates multiple QoIs + plots automatically.

  ✅ Validation supports single-sample and batch (table + plots).

What this script expects:
  - Your solver module is importable (default: bound_all_helpers) OR provide --solver_py path.
  - Your meta CSV has columns:
      Rock_type, Angle, Diameter_mm, Thickness_mm, Load_(KN),
      Tensile_strength_Mpa, Cohesion, Friction_Angle
    (names can vary by spacing; we strip/standardize)

Run in Jupyter:
  main(["--stress_unit","MPa","smoke","--sample_id","1","--debug"])
  main(["--stress_unit","MPa","verify","--sample_id","1"])
  main(["--stress_unit","MPa","validate","--sample_id","1","--exp_path_csv","exp_sid1.csv"])
  main(["--stress_unit","MPa","batch_validate","--pairs_csv","pairs.csv"])

pairs.csv format:
  sample_id,exp_path_csv
  1,/path/to/exp1.csv
  2,/path/to/exp2.csv
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import importlib
import importlib.util
import inspect
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Jupyter argv sanitizer (removes -f and --f=...)
# ==========================================================
def sanitize_argv(argv: List[str]) -> List[str]:
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
# Solver import/load utilities
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


def _candidate_search_dirs(meta_csv: Optional[str] = None) -> List[str]:
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

    out, seen = [], set()
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen and os.path.isdir(d):
            out.append(d)
            seen.add(d)
    return out


def _find_module_py(module_name: str, search_dirs: List[str]) -> Optional[str]:
    if module_name.lower().endswith(".py") and os.path.isfile(module_name):
        return os.path.abspath(module_name)
    fname = module_name + ".py"
    for d in search_dirs:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def import_or_load_solver(solver_mod: str, meta_csv: Optional[str] = None, debug: bool = False):
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
        "  (A) Put your solver file (e.g. bound_all_helpers.py) in your working folder, OR\n"
        "  (B) Pass --solver_py /full/path/to/your_solver.py, OR\n"
        "  (C) Pass --solver_mod your_module_name (and make sure it is importable).\n"
    )
    raise RuntimeError(msg)


def validate_solver_api(mod):
    required = [
        # geometry/grid
        "points_in_disk",
        "map_angle_to_alpha",
        # rotations/stress
        "rot_to_material",
        "stress_material_to_global",
        "principal_from_components",
        "eval_stress_field_material",
        # airy
        "fit_orthotropic_airy_disk",
        # failure fields
        "failure_mode_map",
        "compute_psi_pref_field",
        "weak_plane_weight_field",
        # ddm/sif + fracture
        "solve_cracked_disk_correction_ddm",
        "sif_two_tips_from_crack",
        "_try_call_sif_two_tips",
        "_extract_tip_sifs",
        "_seg_intersect",
        "G_from_K_orthotropic",
        "Gc_theta_weak_plane",
    ]
    missing, noneish, notcall = [], [], []
    for name in required:
        if not hasattr(mod, name):
            missing.append(name)
            continue
        val = getattr(mod, name)
        if val is None:
            noneish.append(name)
            continue
        if not callable(val):
            notcall.append(name)

    if missing or noneish or notcall:
        lines = ["Solver module is not providing the required API:"]
        if missing:
            lines.append("  Missing attributes: " + ", ".join(missing))
        if noneish:
            lines.append("  Attributes are None: " + ", ".join(noneish))
        if notcall:
            lines.append("  Attributes not callable: " + ", ".join(notcall))
        lines.append("")
        lines.append("Fix: ensure your solver module exports these functions.")
        raise RuntimeError("\n".join(lines))


# ==========================================================
# Units (explicit, no inference)
# ==========================================================
@dataclass
class UnitSystem:
    stress_unit: str  # "MPa" or "Pa"

    @property
    def stress_scale_to_Pa(self) -> float:
        # multiply internal stress by this to get Pa
        return 1e6 if self.stress_unit.upper() == "MPA" else 1.0

    @property
    def stress_scale_from_MPa(self) -> float:
        # multiply MPa values by this to get internal stress
        return 1.0 if self.stress_unit.upper() == "MPA" else 1e6

    @property
    def E_scale_from_GPa(self) -> float:
        # multiply GPa values by this to get internal stress units
        return 1e3 if self.stress_unit.upper() == "MPA" else 1e9

    def Gc_internal_from_Jm2(self, Gc_Jm2: float) -> float:
        # internal G has units (stress_unit)*m
        # 1 MPa*m = 1e6 J/m^2
        return float(Gc_Jm2) / self.stress_scale_to_Pa

    def G_Jm2_from_internal(self, G_internal: float) -> float:
        return float(G_internal) * self.stress_scale_to_Pa


# ==========================================================
# Global solver module handle
# ==========================================================
bd = None


# ==========================================================
# Numerically safe angle wrappers
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
# Line-orientation blending (pi-periodic)
# ==========================================================
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


# ==========================================================
# Grid sampling helpers
# ==========================================================
def _grid_axes_from_mesh(X, Y):
    x1d = np.asarray(X[0, :], float)
    y1d = np.asarray(Y[:, 0], float)
    return x1d, y1d

def bilinear_sample_scalar(img, X, Y, x, y, fill=np.nan):
    x1d, y1d = _grid_axes_from_mesh(X, Y)
    nx, ny = len(x1d), len(y1d)
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

    f00 = img[iy, ix]
    f10 = img[iy, ix + 1]
    f01 = img[iy + 1, ix]
    f11 = img[iy + 1, ix + 1]
    if not (np.isfinite(f00) and np.isfinite(f10) and np.isfinite(f01) and np.isfinite(f11)):
        return float(fill)

    f0 = (1 - tx) * f00 + tx * f10
    f1 = (1 - tx) * f01 + tx * f11
    return float((1 - ty) * f0 + ty * f1)

def bilinear_sample_line_angle(psi_img, X, Y, x, y, fill=np.nan):
    x1d, y1d = _grid_axes_from_mesh(X, Y)
    nx, ny = len(x1d), len(y1d)
    if not (x1d[0] <= x <= x1d[-1]) or not (y1d[0] <= y <= y1d[-1]):
        return float(fill)

    ix = int(np.searchsorted(x1d, x) - 1)
    iy = int(np.searchsorted(y1d, y) - 1)
    ix = max(0, min(ix, nx - 2))
    iy = max(0, min(iy, ny - 2))

    a00 = psi_img[iy, ix]
    a10 = psi_img[iy, ix + 1]
    a01 = psi_img[iy + 1, ix]
    a11 = psi_img[iy + 1, ix + 1]
    if not (np.isfinite(a00) and np.isfinite(a10) and np.isfinite(a01) and np.isfinite(a11)):
        return float(fill)

    x0, x1 = x1d[ix], x1d[ix + 1]
    y0, y1 = y1d[iy], y1d[iy + 1]
    tx = 0.0 if (x1 - x0) == 0 else (x - x0) / (x1 - x0)
    ty = 0.0 if (y1 - y0) == 0 else (y - y0) / (y1 - y0)

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


# ==========================================================
# Polyline resampling + distances (QoIs)
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
    xr = np.interp(st, s, x)
    yr = np.interp(st, s, y)
    return xr, yr

def path_rms_max_distance(x_ref, y_ref, x, y, n=500):
    xr, yr = resample_polyline_by_s(x_ref, y_ref, n=n)
    xt, yt = resample_polyline_by_s(x, y, n=n)
    d = np.hypot(xt - xr, yt - yr)
    return float(np.sqrt(np.mean(d * d))), float(np.max(d))

def symmetric_mean_min_distance(xa, ya, xb, yb, n=500):
    xa, ya = resample_polyline_by_s(xa, ya, n=n)
    xb, yb = resample_polyline_by_s(xb, yb, n=n)
    A = np.stack([xa, ya], axis=1)
    B = np.stack([xb, yb], axis=1)
    dAB = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    mAB = float(np.mean(np.min(dAB, axis=1)))
    mBA = float(np.mean(np.min(dAB, axis=0)))
    return 0.5 * (mAB + mBA)

def polyline_length(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))

def initiation_angle_deg(x, y, rmax_frac, R):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = np.hypot(x, y)
    keep = r <= float(rmax_frac) * float(R)
    if np.sum(keep) < 3 and len(x) >= 2:
        dx, dy = (x[1] - x[0]), (y[1] - y[0])
        return float(np.degrees(np.arctan2(dy, dx)))
    if np.sum(keep) < 3:
        return np.nan
    xx, yy = x[keep], y[keep]
    C = np.cov(np.stack([xx, yy], axis=0))
    w, V = np.linalg.eigh(C)
    v = V[:, np.argmax(w)]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    if ang >= 180:
        ang -= 360
    return float(ang)


# ==========================================================
# Signature-safe solver wrappers
# ==========================================================
def call_compute_psi_pref(X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit, mixed_band=0.45):
    fn = bd.compute_psi_pref_field
    sig = inspect.signature(fn)
    names = list(sig.parameters.keys())
    if len(names) >= 3 and names[0].lower() == "x" and names[1].lower() == "y" and names[2].lower() == "m":
        return fn(X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
                  drive_basis="first", mixed_band=float(mixed_band), eps=1e-12)
    return fn(Rt_eff, Rs_eff, sxx, syy, txy, beta_crit,
              drive_basis="first", mixed_band=float(mixed_band), eps=1e-12)

def compute_wp_weight_img(X, Y, alpha_wp_line, spacing, bandwidth_frac):
    fn = bd.weak_plane_weight_field
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    if ("phase" in params) and ("bandwidth_frac" in params):
        w = fn(X, Y, alpha_wp_line=float(alpha_wp_line), spacing=float(spacing), phase=0.0, bandwidth_frac=float(bandwidth_frac))
    elif ("bandwidth_frac" in params):
        w = fn(X, Y, alpha_wp_line=float(alpha_wp_line), spacing=float(spacing), bandwidth_frac=float(bandwidth_frac))
    else:
        w = fn(X, Y, float(alpha_wp_line), float(spacing))
    w = np.asarray(w, float)
    return np.clip(w, 0.0, 1.0)

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


# ==========================================================
# Energy stepping (single-step selection)
# ==========================================================
_TRACE_COLUMNS = [
    "step", "x", "y", "psi", "KI", "KII", "G_internal", "Gc_internal", "score_internal",
    "G_Jm2", "Gc_Jm2", "score_Jm2",
    "weak_reduction_eff", "cap_deg", "kink_n", "K_keep", "ds"
]

def energy_step_best_candidate(
    xs_u, ys_u,
    R, alpha_const, airy_fit,
    E1, E2, nu12, G12,
    ds,
    kink_scan_deg, kink_n, K_keep,
    psi_pref_guided_img, X, Y,
    tensile_w_img, conf_img, wp_weight_img, w_load_img,
    Gc0_internal, weak_reduction_base, eta_deg, alpha_wp_line,
    ddm_sample_rs, cod_offset,
    corr_cache,
    units: UnitSystem,
    N_outer=80, colloc_eps_frac=2e-5, reg_lam=1e-10,
    debug=False
):
    tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
    psi_tip = float(np.arctan2(ys_u[-1] - ys_u[-2], xs_u[-1] - xs_u[-2]))

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

    cap = np.deg2rad(float(kink_scan_deg))
    dscan = np.linspace(-cap, cap, int(kink_n))

    wt_tip = bilinear_sample_scalar(tensile_w_img, X, Y, tipx, tipy, fill=0.5)
    wc_tip = bilinear_sample_scalar(conf_img, X, Y, tipx, tipy, fill=0.0)
    wwp_tip = bilinear_sample_scalar(wp_weight_img, X, Y, tipx, tipy, fill=0.0)
    wload_tip = bilinear_sample_scalar(w_load_img, X, Y, tipx, tipy, fill=0.0)
    wt_tip = float(np.clip(wt_tip if np.isfinite(wt_tip) else 0.5, 0.0, 1.0))
    wc_tip = float(np.clip(wc_tip if np.isfinite(wc_tip) else 0.0, 0.0, 1.0))
    wwp_tip = float(np.clip(wwp_tip if np.isfinite(wwp_tip) else 0.0, 0.0, 1.0))
    wload_tip = float(np.clip(wload_tip if np.isfinite(wload_tip) else 0.0, 0.0, 1.0))

    shear_dom = (1.0 - wt_tip)
    weak_red_eff = float(weak_reduction_base) * wwp_tip * (shear_dom ** 2.0) * (0.6 + 0.4 * wc_tip) * (0.5 + 0.5 * wload_tip)
    weak_red_eff = float(np.clip(weak_red_eff, 0.0, float(weak_reduction_base)))

    psi_line = bilinear_sample_line_angle(psi_pref_guided_img, X, Y, tipx, tipy, fill=np.pi/2)

    cand = []
    for dth in dscan:
        psi_new = float(psi_tip + dth)
        nx = tipx + float(ds) * np.cos(psi_new)
        ny = tipy + float(ds) * np.sin(psi_new)
        if (nx * nx + ny * ny) >= (0.999 * float(R)) ** 2:
            continue
        cheap = 0.0
        if np.isfinite(psi_line):
            dpsi = abs(wrap_pi_half_scalar(psi_new - psi_line)) / (np.pi / 2)
            cheap = -dpsi * dpsi
        cand.append((cheap, psi_new, nx, ny))

    if not cand:
        if debug:
            print("[debug] no candidate points inside disk (ds too large?)")
        return None

    cand.sort(key=lambda z: z[0], reverse=True)
    if int(K_keep) > 0:
        cand = cand[: int(K_keep)]

    best = None
    n_try = 0
    n_finite = 0

    for _cheap, psi_new, nx, ny in cand:
        n_try += 1

        # self-intersection check (optional but useful)
        if len(xs_u) > 25:
            p1 = (xs_u[-1], ys_u[-1]); p2 = (nx, ny)
            bad = False
            for j in range(2, len(xs_u) - 10):
                q1 = (xs_u[j - 1], ys_u[j - 1])
                q2 = (xs_u[j], ys_u[j])
                if bd._seg_intersect(p1, p2, q1, q2):
                    bad = True
                    break
            if bad:
                continue

        xtrial = np.asarray(list(xs_u) + [nx], float)
        ytrial = np.asarray(list(ys_u) + [ny], float)

        sif_res = bd._try_call_sif_two_tips(
            bd.sif_two_tips_from_crack,
            xtrial, ytrial,
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=airy_fit,
            eval_stress_field_material=bd.eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            correction=corr,
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
        )
        _, _, KI, KII = bd._extract_tip_sifs(sif_res)
        if not (np.isfinite(KI) and np.isfinite(KII)):
            continue
        n_finite += 1

        G_internal, _H = bd.G_from_K_orthotropic(
            KI, KII,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            alpha_const=float(alpha_const),
            psi_tip_global=float(psi_new),
        )
        if not np.isfinite(G_internal):
            continue

        theta_line = float(wrap_pi_half_scalar(psi_new))
        Gc_internal = bd.Gc_theta_weak_plane(
            theta_line=float(theta_line),
            alpha_wp=float(alpha_wp_line),
            Gc_matrix=float(Gc0_internal),
            weak_reduction=float(weak_red_eff),
            eta_deg=float(eta_deg),
        )
        score_internal = float(G_internal - Gc_internal)

        if (best is None) or (score_internal > best["score_internal"]):
            best = dict(
                nx=float(nx), ny=float(ny),
                psi_new=float(psi_new),
                KI=float(KI), KII=float(KII),
                G_internal=float(G_internal),
                Gc_internal=float(Gc_internal),
                score_internal=float(score_internal),
                G_Jm2=units.G_Jm2_from_internal(G_internal),
                Gc_Jm2=units.G_Jm2_from_internal(Gc_internal),
                score_Jm2=units.G_Jm2_from_internal(score_internal),
                weak_reduction_eff=float(weak_red_eff),
            )

    if debug:
        print(f"[debug] candidates tried={n_try}, finite(KI,KII)={n_finite}, best={'yes' if best else 'no'}")

    return best


# ==========================================================
# Run parameters (paper-ready, explicit)
# ==========================================================
@dataclass
class RunParams:
    points_per_row: int = 161
    ds_frac: float = 0.010
    max_steps: int = 1200
    N_outer: int = 80

    psi0_deg: float = 90.0
    psi0_delta_deg: float = 0.0
    a0_frac: float = 0.016
    cod_offset: float = 1e-5

    # material
    E1_GPa: float = 50.0
    E2_GPa: float = 30.0
    nu12: float = 0.25
    G12_GPa: float = 12.0

    # platens / airy
    platen_half_angle_deg: float = 10.0
    platen_smooth_deg: float = 4.0
    platen_mu: float = 0.0
    airy_auto_tune: bool = True
    airy_Nbd_base: int = 420
    airy_Nbd_arc_each: int = 900
    load_factor: float = 1.0

    # failure model
    fail_mixed_band: float = 0.45
    fail_mc_nplanes: int = 361
    fail_margin: float = 0.0
    fail_mode_basis: str = "first"
    fail_util_min: float = 0.98

    # weak plane
    weak_spacing_m: float = 0.02
    wp_bandwidth_frac: float = 0.12
    eta_deg: float = 10.0
    weak_T_ratio: float = 0.35
    weak_C_ratio: float = 0.60

    # toughness (EXTERNAL physical input)
    Gc0_Jm2: float = 50.0

    # end
    r_end_frac: float = 0.995


def build_fields_and_run(
    dfmeta_row,
    params: RunParams,
    units: UnitSystem,
    debug: bool = False,
) -> Dict[str, Any]:
    rock, ang_deg, Dmm, tmm, PkN, Tm_in_MPa, Coh_in_MPa, Phi_in_deg = dfmeta_row

    # Convert material constants to internal stress units
    E1 = float(params.E1_GPa) * units.E_scale_from_GPa
    E2 = float(params.E2_GPa) * units.E_scale_from_GPa
    G12 = float(params.G12_GPa) * units.E_scale_from_GPa
    nu12 = float(params.nu12)

    # Convert strengths from MPa (meta) to internal stress units
    Tm = float(Tm_in_MPa) * units.stress_scale_from_MPa
    Coh0 = float(Coh_in_MPa) * units.stress_scale_from_MPa
    Phi0 = np.deg2rad(float(Phi_in_deg))

    alpha_const = float(bd.map_angle_to_alpha(float(ang_deg), angle_map="direct"))
    alpha_wp_line = float(wrap_pi_half_scalar(alpha_const))

    # Geometry + load
    D = float(Dmm) * 1e-3
    t = float(tmm) * 1e-3
    P = float(PkN) * 1e3 * float(params.load_factor)  # N
    R = D / 2.0

    xg, yg, X, Y, M = bd.points_in_disk(D, n=int(params.points_per_row))

    # Fit Airy stress function (optional auto)
    if bool(params.airy_auto_tune) and (getattr(bd, "fit_orthotropic_airy_disk_auto", None) is not None):
        fit = bd.fit_orthotropic_airy_disk_auto(
            E1, E2, nu12, G12,
            R=R, t=t, P=P,
            alpha=alpha_const,
            beta_deg=float(params.platen_half_angle_deg),
            smooth_deg=float(params.platen_smooth_deg),
            mu=float(params.platen_mu),
            Nbd=int(params.airy_Nbd_base),
            Nbd_arc_each=int(params.airy_Nbd_arc_each),
        )
    else:
        fit = bd.fit_orthotropic_airy_disk(
            E1, E2, nu12, G12,
            R=R, t=t, P=P,
            alpha=alpha_const,
            M=24,
            Nbd=int(params.airy_Nbd_base),
            beta_deg=float(params.platen_half_angle_deg),
            smooth_deg=float(params.platen_smooth_deg),
            mu=float(params.platen_mu),
            lam=1e-8,
            w_arc=12.0,
            Nbd_arc_each=int(params.airy_Nbd_arc_each),
        )

    xm, ym = bd.rot_to_material(xg, yg, alpha_const)
    sxx_m, syy_m, txy_m = bd.eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
    sxx, syy, txy = bd.stress_material_to_global(sxx_m, syy_m, txy_m, alpha_const)
    s1, s2, _ = bd.principal_from_components(sxx, syy, txy)

    # toughness in internal units (EXPLICIT)
    Gc0_internal = units.Gc_internal_from_Jm2(params.Gc0_Jm2)
    if debug:
        print(f"[debug] units: stress_unit={units.stress_unit}, stress_scale_to_Pa={units.stress_scale_to_Pa:g}")
        print(f"[debug] using load_factor={params.load_factor:g}, Gc0_Jm2={params.Gc0_Jm2:g} => Gc0_internal={Gc0_internal:.6e}")

    Phi = np.full_like(s1, Phi0, float)
    Coh = np.full_like(s1, Coh0, float)
    Teff = np.full_like(s1, Tm, float)

    modes, Rt_eff, Rs_eff, beta_crit = bd.failure_mode_map(
        sxx, syy, txy, s1,
        Tm=Teff, Coh=Coh, Phi=Phi,
        alpha_const=alpha_const,
        basis=str(params.fail_mode_basis),
        util_min=float(params.fail_util_min),
        mixed_band=float(params.fail_mixed_band),
        margin=float(params.fail_margin),
        n_theta_mc=int(params.fail_mc_nplanes),

        mc_compression_only=True,
        mc_sigma_comp_min=0.0,
        strength_model="weak_plane",
        weak_T_ratio=float(params.weak_T_ratio),
        weak_C_ratio=float(params.weak_C_ratio),
        weak_phi=None,
        eta_deg=float(params.eta_deg),

        alpha_wp_line=float(alpha_wp_line),
        x=xg, y=yg,
        weak_spacing=float(params.weak_spacing_m),
        weak_bandwidth_frac=0.25,
        weak_floor=0.10,

        stress_sign_mode="auto",
        conf_k=0.65,
        wing_k=1.00,
        wing_p=2.0,
    )

    psi_pref = call_compute_psi_pref(X, Y, M, Rt_eff, Rs_eff, sxx, syy, txy, beta_crit, mixed_band=float(params.fail_mixed_band))
    psi_pref_img = np.full_like(X, np.nan, float); psi_pref_img[M] = psi_pref

    eps = 1e-12
    wt = Rt_eff / (Rt_eff + Rs_eff + eps)
    tensile_w_img = np.full_like(X, 0.0, float); tensile_w_img[M] = np.clip(wt, 0.0, 1.0)

    denom = (np.abs(s1) + np.abs(s2) + 1e-12)
    conf = np.clip((-np.minimum(s2, 0.0)) / denom, 0.0, 1.0)
    conf_img = np.full_like(X, 0.0, float); conf_img[M] = conf

    wp_weight_img = compute_wp_weight_img(X, Y, alpha_wp_line=float(alpha_wp_line),
                                          spacing=float(params.weak_spacing_m),
                                          bandwidth_frac=float(params.wp_bandwidth_frac))
    w_load_img = contact_arc_weight(X, Y, R, beta_deg=float(params.platen_half_angle_deg), smooth_deg=float(params.platen_smooth_deg))

    # guided psi field
    psi_vertical = np.full_like(X, np.pi / 2.0, float)
    w_vert = (np.clip(tensile_w_img, 0.0, 1.0) ** 2.0) * (1.0 - 0.85 * w_load_img)
    w_vert = np.clip(w_vert, 0.0, 1.0)
    psi_mid = blend_line_orientations(psi_pref_img, psi_vertical, w_vert)

    shear_dom_img = (1.0 - tensile_w_img)
    w_guid = wp_weight_img * (shear_dom_img ** 2.0) * (0.6 + 0.4 * conf_img) * (0.40 + 0.60 * w_load_img)
    w_guid = np.clip(w_guid, 0.0, 1.0)
    psi_weak = np.full_like(X, float(alpha_wp_line), float)
    psi_pref_guided_img = blend_line_orientations(psi_mid, psi_weak, w_guid)

    weak_reduction_base = float(np.clip(1.0 - float(params.weak_T_ratio), 0.0, 0.90))

    ddm_sample_rs = np.logspace(-4, -2, 6)

    ds0 = float(params.ds_frac) * R
    a0 = float(params.a0_frac) * R
    psi0 = np.deg2rad(float(params.psi0_deg) + float(params.psi0_delta_deg))

    xs_u = [0.0, a0 * np.cos(float(psi0))]
    ys_u = [0.0, a0 * np.sin(float(psi0))]

    corr_cache: Dict[str, Any] = {}
    trace_rows: List[Dict[str, Any]] = []
    stop_reason = "max_steps"

    for step in range(int(params.max_steps)):
        tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
        if np.hypot(tipx, tipy) >= float(params.r_end_frac) * R:
            stop_reason = "reached_boundary"
            if debug:
                print(f"[STOP] reached boundary at step={step}")
            break

        wt_tip = bilinear_sample_scalar(tensile_w_img, X, Y, tipx, tipy, fill=0.5)
        wl_tip = bilinear_sample_scalar(w_load_img, X, Y, tipx, tipy, fill=0.0)
        wt_tip = float(np.clip(wt_tip if np.isfinite(wt_tip) else 0.5, 0.0, 1.0))
        wl_tip = float(np.clip(wl_tip if np.isfinite(wl_tip) else 0.0, 0.0, 1.0))

        cap_deg = float(np.clip(20.0 + 70.0 * (1.0 - wt_tip) + 20.0 * wl_tip, 20.0, 90.0))
        kink_n = int(np.clip(61 + 90 * (1.0 - wt_tip) + 40 * wl_tip, 61, 181))
        if kink_n % 2 == 0:
            kink_n += 1
        K_keep = int(np.clip(9 + 10 * (1.0 - wt_tip), 9, 19))

        best = energy_step_best_candidate(
            xs_u, ys_u,
            R=R, alpha_const=alpha_const, airy_fit=fit,
            E1=E1, E2=E2, nu12=nu12, G12=G12,
            ds=ds0,
            kink_scan_deg=cap_deg, kink_n=kink_n, K_keep=K_keep,
            psi_pref_guided_img=psi_pref_guided_img, X=X, Y=Y,
            tensile_w_img=tensile_w_img, conf_img=conf_img,
            wp_weight_img=wp_weight_img, w_load_img=w_load_img,
            Gc0_internal=Gc0_internal,
            weak_reduction_base=weak_reduction_base,
            eta_deg=float(params.eta_deg),
            alpha_wp_line=float(alpha_wp_line),
            ddm_sample_rs=ddm_sample_rs,
            cod_offset=float(params.cod_offset),
            corr_cache=corr_cache,
            units=units,
            N_outer=int(params.N_outer),
            debug=debug
        )

        if best is None:
            stop_reason = "best_none"
            if debug:
                print(f"[STOP] step={step}: best=None (no finite candidates)")
            break

        if step == 0 and debug:
            print(f"[debug] step0: G={best['G_internal']:.6e}  Gc={best['Gc_internal']:.6e}  G-Gc={best['score_internal']:.6e}")

        if best["score_internal"] <= 0.0:
            stop_reason = "score_le_0"
            if debug:
                print(f"[STOP] step={step}: score<=0 (G-Gc={best['score_internal']:.3e})")
            break

        xs_u.append(best["nx"]); ys_u.append(best["ny"])
        trace_rows.append(dict(
            step=step,
            x=best["nx"], y=best["ny"],
            psi=best["psi_new"],
            KI=best["KI"], KII=best["KII"],
            G_internal=best["G_internal"], Gc_internal=best["Gc_internal"], score_internal=best["score_internal"],
            G_Jm2=best["G_Jm2"], Gc_Jm2=best["Gc_Jm2"], score_Jm2=best["score_Jm2"],
            weak_reduction_eff=best["weak_reduction_eff"],
            cap_deg=cap_deg, kink_n=kink_n, K_keep=K_keep,
            ds=ds0
        ))

    xu = np.asarray(xs_u, float); yu = np.asarray(ys_u, float)
    xs = np.concatenate([(-xu)[::-1], xu[1:]])
    ys = np.concatenate([(-yu)[::-1], yu[1:]])

    trace = pd.DataFrame(trace_rows)
    for c in _TRACE_COLUMNS:
        if c not in trace.columns:
            trace[c] = np.nan
    trace = trace[_TRACE_COLUMNS]

    return dict(
        rock=str(rock),
        angle_deg=float(ang_deg),
        alpha_const=float(alpha_const),
        alpha_wp_line=float(alpha_wp_line),
        R=float(R),
        xs=xs, ys=ys,
        trace=trace,
        stop_reason=str(stop_reason),
        run_params=asdict(params),
        units=dict(stress_unit=units.stress_unit, stress_scale_to_Pa=units.stress_scale_to_Pa),
        Gc0_internal_used=float(Gc0_internal),
    )


# ==========================================================
# Plot helpers (paper-ish)
# ==========================================================
def save_energy_plot(trace: pd.DataFrame, out_png: str, title: str):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    if trace is None or len(trace) == 0:
        ax.text(0.5, 0.5, "Empty trace (no growth)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy (J/m$^2$)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    ax.plot(trace["step"].to_numpy(), trace["G_Jm2"].to_numpy(), marker="o", label="G (J/m$^2$)")
    ax.plot(trace["step"].to_numpy(), trace["Gc_Jm2"].to_numpy(), marker="o", label="Gc (J/m$^2$)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (J/m$^2$)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def save_path_plot(xs, ys, R, out_png: str, title: str, exp_xy: Optional[Tuple[np.ndarray,np.ndarray]] = None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    if exp_xy is not None:
        xe, ye = exp_xy
        ax.plot(xe, ye, "-", lw=2.0, label="experiment")

    ax.plot(xs, ys, "-", lw=2.0, label="model")
    ax.add_artist(plt.Circle((0, 0), R, fill=False, lw=1.2))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ==========================================================
# QoI computation
# ==========================================================
def energy_margin_integral(trace: pd.DataFrame) -> float:
    if trace is None or len(trace) == 0:
        return float("-inf")
    ds = np.asarray(trace["ds"], float)
    s = np.maximum(np.asarray(trace["score_Jm2"], float), 0.0)
    return float(np.sum(s * ds))

def summarize_run(res: Dict[str, Any]) -> Dict[str, Any]:
    xs, ys = res["xs"], res["ys"]
    R = float(res["R"])
    trace = res["trace"]
    out = {}
    out["stop_reason"] = res.get("stop_reason", "")
    out["nsteps"] = int(len(trace)) if trace is not None else 0
    out["path_len_m"] = polyline_length(xs, ys)
    out["init_angle_deg"] = initiation_angle_deg(xs, ys, rmax_frac=0.12, R=R)
    out["J_margin"] = energy_margin_integral(trace)

    if trace is not None and len(trace) >= 1:
        out["G0_Jm2"] = float(trace["G_Jm2"].iloc[0])
        out["Gc0_Jm2"] = float(trace["Gc_Jm2"].iloc[0])
        out["score0_Jm2"] = float(trace["score_Jm2"].iloc[0])
        out["G_last_Jm2"] = float(trace["G_Jm2"].iloc[-1])
        out["Gc_last_Jm2"] = float(trace["Gc_Jm2"].iloc[-1])
        out["KI_last"] = float(trace["KI"].iloc[-1])
        out["KII_last"] = float(trace["KII"].iloc[-1])
    else:
        out["G0_Jm2"] = np.nan
        out["Gc0_Jm2"] = np.nan
        out["score0_Jm2"] = np.nan
        out["G_last_Jm2"] = np.nan
        out["Gc_last_Jm2"] = np.nan
        out["KI_last"] = np.nan
        out["KII_last"] = np.nan

    return out


# ==========================================================
# Commands
# ==========================================================
def _load_meta(meta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(meta_csv, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _row_from_meta(dfmeta: pd.DataFrame, sample_id: int) -> pd.Series:
    cols = ["Rock_type","Angle","Diameter_mm","Thickness_mm","Load_(KN)",
            "Tensile_strength_Mpa","Cohesion","Friction_Angle"]
    missing = [c for c in cols if c not in dfmeta.columns]
    if missing:
        raise RuntimeError(f"meta_csv missing columns: {missing}")
    if sample_id not in dfmeta.index:
        raise RuntimeError(f"sample_id {sample_id} not found in meta_csv index.")
    return dfmeta.loc[sample_id, cols]

def cmd_smoke(dfmeta, sample_id: int, out_dir: str, params: RunParams, units: UnitSystem, debug: bool):
    os.makedirs(out_dir, exist_ok=True)
    row = _row_from_meta(dfmeta, sample_id)

    res = build_fields_and_run(row, params=params, units=units, debug=debug)
    summ = summarize_run(res)

    print("=== SMOKE RESULT ===")
    print("stop_reason:", res["stop_reason"])
    print("nsteps:", summ["nsteps"])
    print("Gc0_internal_used:", res["Gc0_internal_used"])
    print("units:", res["units"])
    print("run_params:", res["run_params"])
    if summ["nsteps"] > 0:
        print(f"step0: G0={summ['G0_Jm2']:.3e} J/m2  Gc0={summ['Gc0_Jm2']:.3e} J/m2  score0={summ['score0_Jm2']:.3e} J/m2")

    # save artifacts
    pd.DataFrame({"x_m": res["xs"], "y_m": res["ys"]}).to_csv(os.path.join(out_dir, f"smoke_path_sid{sample_id}.csv"), index=False)
    res["trace"].to_csv(os.path.join(out_dir, f"smoke_trace_sid{sample_id}.csv"), index=False)

    save_energy_plot(res["trace"], os.path.join(out_dir, f"smoke_energy_sid{sample_id}.png"), title="Driving force vs toughness")
    save_path_plot(res["xs"], res["ys"], res["R"], os.path.join(out_dir, f"smoke_path_sid{sample_id}.png"), title=f"Smoke path (sid {sample_id})")

    print("Saved smoke outputs to:", out_dir)


def cmd_calibrate_growth(dfmeta, sample_id: int, out_dir: str, base_params: RunParams, units: UnitSystem, debug: bool):
    """
    Debug/calibration only: searches for params that yield >0 steps.
    Writes param_json you can later pass to verify/validate with --param_json.
    """
    os.makedirs(out_dir, exist_ok=True)
    row = _row_from_meta(dfmeta, sample_id)

    # Search grids (keep limited so it doesn't explode)
    a0_list = [base_params.a0_frac, 0.03, 0.05, 0.08, 0.12]
    load_list = [base_params.load_factor, 1.5, 2.0, 3.0, 5.0]
    cod_list = [base_params.cod_offset, 5e-5, 1e-4, 2e-4]
    Gc_list = [base_params.Gc0_Jm2, base_params.Gc0_Jm2/2, base_params.Gc0_Jm2/5, base_params.Gc0_Jm2/10]

    best = None
    for a0 in a0_list:
        for lf in load_list:
            for cod in cod_list:
                for gc in Gc_list:
                    p = RunParams(**asdict(base_params))
                    p.a0_frac = float(a0)
                    p.load_factor = float(lf)
                    p.cod_offset = float(cod)
                    p.Gc0_Jm2 = float(gc)
                    p.max_steps = min(int(p.max_steps), 25)

                    res = build_fields_and_run(row, params=p, units=units, debug=False)
                    nsteps = int(len(res["trace"]))
                    if debug:
                        print(f"[cal] a0={a0:g} lf={lf:g} cod={cod:g} Gc={gc:g} => steps={nsteps} stop={res['stop_reason']}")
                    if nsteps > 0:
                        best = p
                        break
                if best: break
            if best: break
        if best: break

    if best is None:
        raise RuntimeError(
            "calibrate_growth failed: no growth for tested settings.\n"
            "This usually means: either your load is too low, or toughness too high, or the SIF solver is failing.\n"
            "Try increasing load_factor / cod_offset, or lower Gc0_Jm2, or inspect solver."
        )

    out_json = os.path.join(out_dir, f"calibrated_params_sid{sample_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(best), f, indent=2)

    print("[calibrate_growth] Found growth params:")
    print(json.dumps(asdict(best), indent=2))
    print("Saved:", out_json)


def cmd_verify(dfmeta, sample_id: int, out_dir: str, base_params: RunParams, units: UnitSystem, debug: bool, allow_no_growth: bool):
    """
    Convergence sweep with multiple QoIs. No auto-fix here (paper pipeline).
    """
    os.makedirs(out_dir, exist_ok=True)
    row = _row_from_meta(dfmeta, sample_id)

    ds_list = [0.014, 0.010, 0.007, 0.005]
    Nouter_list = [50, 80, 120, 160]
    grid_list = [121, 161, 201]

    # reference (finest)
    pref = RunParams(**asdict(base_params))
    pref.points_per_row = max(grid_list)
    pref.ds_frac = min(ds_list)
    pref.N_outer = max(Nouter_list)

    ref = build_fields_and_run(row, params=pref, units=units, debug=debug)
    if len(ref["trace"]) == 0 and not allow_no_growth:
        raise RuntimeError(
            "Reference run produced NO growth (empty trace).\n"
            "This makes convergence meaningless.\n"
            "Fix by calibrating parameters (debug step): run calibrate_growth and then pass --param_json.\n"
        )

    x_ref, y_ref = ref["xs"], ref["ys"]
    ref_summ = summarize_run(ref)

    rows = []
    for g in grid_list:
        for ds in ds_list:
            for No in Nouter_list:
                p = RunParams(**asdict(base_params))
                p.points_per_row = int(g)
                p.ds_frac = float(ds)
                p.N_outer = int(No)

                res = build_fields_and_run(row, params=p, units=units, debug=False)
                summ = summarize_run(res)

                if len(ref["trace"]) > 0 and len(res["trace"]) > 0:
                    rms, dmax = path_rms_max_distance(x_ref, y_ref, res["xs"], res["ys"], n=600)
                else:
                    rms, dmax = (np.nan, np.nan)

                rows.append(dict(
                    sample_id=int(sample_id),
                    grid=int(g), ds_frac=float(ds), N_outer=int(No),
                    stop_reason=str(res["stop_reason"]),
                    nsteps=int(summ["nsteps"]),
                    path_rms_m=float(rms) if np.isfinite(rms) else np.nan,
                    path_max_m=float(dmax) if np.isfinite(dmax) else np.nan,
                    init_angle_deg=float(summ["init_angle_deg"]) if np.isfinite(summ["init_angle_deg"]) else np.nan,
                    path_len_m=float(summ["path_len_m"]),
                    J_margin=float(summ["J_margin"]) if np.isfinite(summ["J_margin"]) else np.nan,
                    G0_Jm2=float(summ["G0_Jm2"]) if np.isfinite(summ["G0_Jm2"]) else np.nan,
                    Gc0_Jm2=float(summ["Gc0_Jm2"]) if np.isfinite(summ["Gc0_Jm2"]) else np.nan,
                    score0_Jm2=float(summ["score0_Jm2"]) if np.isfinite(summ["score0_Jm2"]) else np.nan,
                ))

                tag = f"sid{sample_id}_g{g}_ds{ds:.3f}_No{No}"
                pd.DataFrame({"x_m": res["xs"], "y_m": res["ys"]}).to_csv(os.path.join(out_dir, f"pred_path_{tag}.csv"), index=False)
                res["trace"].to_csv(os.path.join(out_dir, f"trace_{tag}.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"verification_summary_sid{sample_id}.csv"), index=False)

    # plots (QoI vs ds_frac for each N_outer) on the finest grid
    finest_g = max(grid_list)
    sub = df[df["grid"] == finest_g].copy()

    def plot_qoi(ycol: str, ylabel: str, fname: str):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for No in Nouter_list:
            ss = sub[sub["N_outer"] == No].sort_values("ds_frac")
            ax.plot(ss["ds_frac"], ss[ycol], marker="o", label=f"N_outer={No}")
        ax.set_xlabel("ds_frac")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Convergence (sid {sample_id}) — grid={finest_g}\nref steps={ref_summ['nsteps']}  ref stop={ref['stop_reason']}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)

    plot_qoi("path_rms_m", "Path RMS to reference (m)", f"conv_path_rms_sid{sample_id}.png")
    plot_qoi("init_angle_deg", "Initiation angle (deg)", f"conv_init_angle_sid{sample_id}.png")
    plot_qoi("path_len_m", "Total crack path length (m)", f"conv_path_length_sid{sample_id}.png")
    plot_qoi("J_margin", "∑ max(G−Gc,0) ds  (J/m)", f"conv_energy_margin_sid{sample_id}.png")

    print("Saved verification outputs to:", out_dir)


def cmd_validate(dfmeta, sample_id: int, exp_path_csv: str, out_dir: str, params: RunParams, units: UnitSystem, debug: bool, allow_no_growth: bool):
    os.makedirs(out_dir, exist_ok=True)
    row = _row_from_meta(dfmeta, sample_id)

    pred = build_fields_and_run(row, params=params, units=units, debug=debug)
    xp, yp = pred["xs"], pred["ys"]
    R = pred["R"]

    exp = pd.read_csv(exp_path_csv)
    xe = exp.iloc[:, 0].to_numpy(float)
    ye = exp.iloc[:, 1].to_numpy(float)

    ang_p = initiation_angle_deg(xp, yp, rmax_frac=0.12, R=R)
    ang_e = initiation_angle_deg(xe, ye, rmax_frac=0.12, R=R)
    ang_err = float(wrap_pi_scalar(np.deg2rad(ang_p - ang_e)) * 180.0 / np.pi) if np.isfinite(ang_p) and np.isfinite(ang_e) else np.nan

    rms, dmax = path_rms_max_distance(xe, ye, xp, yp, n=600)
    smmd = symmetric_mean_min_distance(xe, ye, xp, yp, n=500)

    metrics = dict(
        sample_id=int(sample_id),
        exp_csv=str(exp_path_csv),
        stop_reason=str(pred["stop_reason"]),
        nsteps=int(len(pred["trace"])),
        init_angle_pred_deg=float(ang_p) if np.isfinite(ang_p) else np.nan,
        init_angle_exp_deg=float(ang_e) if np.isfinite(ang_e) else np.nan,
        init_angle_error_deg=float(ang_err) if np.isfinite(ang_err) else np.nan,
        path_rms_m=float(rms),
        path_max_m=float(dmax),
        sym_mean_min_dist_m=float(smmd),
        path_len_pred_m=polyline_length(xp, yp),
        path_len_exp_m=polyline_length(xe, ye),
        units_stress=str(units.stress_unit),
        Gc0_Jm2=float(params.Gc0_Jm2),
        load_factor=float(params.load_factor),
        cod_offset=float(params.cod_offset),
    )

    if len(pred["trace"]) == 0 and not allow_no_growth:
        raise RuntimeError(
            "Validation run produced NO growth (empty trace).\n"
            "For paper-ready validation you must first calibrate a physically justified parameter set "
            "(or correct units) and re-run.\n"
        )

    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"validation_metrics_sid{sample_id}.csv"), index=False)

    save_path_plot(xp, yp, R, os.path.join(out_dir, f"validation_overlay_sid{sample_id}.png"),
                   title=f"Validation (sid {sample_id})\nΔinit={ang_err:.2f}°, RMS={rms:.3e} m",
                   exp_xy=(xe, ye))
    save_energy_plot(pred["trace"], os.path.join(out_dir, f"validation_energy_sid{sample_id}.png"),
                     title="Driving force vs toughness")

    pd.DataFrame({"x_m": xp, "y_m": yp}).to_csv(os.path.join(out_dir, f"pred_path_sid{sample_id}.csv"), index=False)
    pred["trace"].to_csv(os.path.join(out_dir, f"pred_trace_sid{sample_id}.csv"), index=False)

    print("Saved validation outputs to:", out_dir)


def cmd_batch_validate(dfmeta, pairs_csv: str, out_dir: str, params: RunParams, units: UnitSystem, debug: bool, allow_no_growth: bool):
    os.makedirs(out_dir, exist_ok=True)
    pairs = pd.read_csv(pairs_csv)
    needed = {"sample_id", "exp_path_csv"}
    if not needed.issubset(set(pairs.columns)):
        raise RuntimeError(f"pairs_csv must have columns {sorted(list(needed))}")

    all_metrics = []
    for _, r in pairs.iterrows():
        sid = int(r["sample_id"])
        exp_csv = str(r["exp_path_csv"])
        od = os.path.join(out_dir, f"sid{sid}")
        os.makedirs(od, exist_ok=True)

        try:
            cmd_validate(dfmeta, sid, exp_csv, od, params, units, debug=debug, allow_no_growth=allow_no_growth)
            m = pd.read_csv(os.path.join(od, f"validation_metrics_sid{sid}.csv")).iloc[0].to_dict()
            all_metrics.append(m)
        except Exception as e:
            all_metrics.append(dict(sample_id=sid, exp_csv=exp_csv, error=str(e)))

    df = pd.DataFrame(all_metrics)
    out_table = os.path.join(out_dir, "batch_validation_table.csv")
    df.to_csv(out_table, index=False)

    # quick summary plot (RMS)
    if "path_rms_m" in df.columns and df["path_rms_m"].notna().any():
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ok = df[df["path_rms_m"].notna()]
        ax.bar(ok["sample_id"].astype(int).to_numpy(), ok["path_rms_m"].to_numpy())
        ax.set_xlabel("sample_id")
        ax.set_ylabel("Path RMS (m)")
        ax.set_title("Batch validation RMS")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "batch_validation_rms.png"), dpi=200)
        plt.close(fig)

    print("Saved batch validation table to:", out_table)


# ==========================================================
# Param JSON override (paper pipeline: explicit)
# ==========================================================
def load_params_json(path: str, base: RunParams) -> RunParams:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    p = RunParams(**asdict(base))
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


# ==========================================================
# CLI
# ==========================================================
def main(argv=None):
    global bd
    if argv is None:
        argv = sanitize_argv(sys.argv[1:])

    # common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Print debug info (step0 G/Gc, etc.)")
    common.add_argument("--stress_unit", required=True, choices=["MPa", "Pa"], help="EXPLICIT solver stress unit (no inference).")
    common.add_argument("--solver_py", default=os.environ.get("SOLVER_PY", ""), help="Optional path to solver .py")
    common.add_argument("--solver_mod", default=os.environ.get("SOLVER_MOD", "bound_all_helpers"), help="Solver module name")
    common.add_argument("--meta_csv", default="tensile_samples_data.csv")
    common.add_argument("--out_dir", default="jmps_out")
    common.add_argument("--param_json", default="", help="Optional JSON overriding RunParams (recommended for paper runs).")
    common.add_argument("--allow_no_growth", action="store_true", help="Allow empty trace (not paper-ready).")

    # run params exposed
    common.add_argument("--points_per_row", type=int, default=161)
    common.add_argument("--ds_frac", type=float, default=0.010)
    common.add_argument("--max_steps", type=int, default=1200)
    common.add_argument("--N_outer", type=int, default=80)
    common.add_argument("--load_factor", type=float, default=1.0)
    common.add_argument("--Gc0_Jm2", type=float, default=50.0)
    common.add_argument("--a0_frac", type=float, default=0.016)
    common.add_argument("--cod_offset", type=float, default=1e-5)
    common.add_argument("--sample_id", type=int, default=1)

    ap = argparse.ArgumentParser(add_help=True, parents=[common])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("smoke", parents=[common])
    sub.add_parser("verify", parents=[common])

    sp_val = sub.add_parser("validate", parents=[common])
    sp_val.add_argument("--exp_path_csv", required=True)

    sp_cal = sub.add_parser("calibrate_growth", parents=[common])

    sp_bat = sub.add_parser("batch_validate", parents=[common])
    sp_bat.add_argument("--pairs_csv", required=True)

    if len(argv) == 0:
        # default: show how to call (but still need stress_unit; argparse will prompt)
        argv = ["smoke"]

    args = ap.parse_args(argv)
    if args.cmd is None:
        args.cmd = "smoke"

    units = UnitSystem(stress_unit=str(args.stress_unit))

    # load solver
    if str(args.solver_py).strip():
        bd = load_module_from_path(str(args.solver_py).strip(), module_name="bd")
    else:
        bd = import_or_load_solver(args.solver_mod, meta_csv=args.meta_csv, debug=args.debug)

    validate_solver_api(bd)

    dfmeta = _load_meta(args.meta_csv)

    # base params from CLI
    base = RunParams(
        points_per_row=int(args.points_per_row),
        ds_frac=float(args.ds_frac),
        max_steps=int(args.max_steps),
        N_outer=int(args.N_outer),
        load_factor=float(args.load_factor),
        Gc0_Jm2=float(args.Gc0_Jm2),
        a0_frac=float(args.a0_frac),
        cod_offset=float(args.cod_offset),
    )

    # optional paper param override
    params = load_params_json(args.param_json, base) if str(args.param_json).strip() else base

    out_dir = os.path.join(str(args.out_dir), f"sid{int(args.sample_id)}") if args.cmd in ("smoke","verify","validate","calibrate_growth") else str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.cmd == "smoke":
        cmd_smoke(dfmeta, int(args.sample_id), out_dir, params, units, debug=args.debug)

    elif args.cmd == "calibrate_growth":
        cmd_calibrate_growth(dfmeta, int(args.sample_id), out_dir, params, units, debug=args.debug)

    elif args.cmd == "verify":
        cmd_verify(dfmeta, int(args.sample_id), out_dir, params, units, debug=args.debug, allow_no_growth=bool(args.allow_no_growth))

    elif args.cmd == "validate":
        cmd_validate(dfmeta, int(args.sample_id), str(args.exp_path_csv), out_dir, params, units, debug=args.debug, allow_no_growth=bool(args.allow_no_growth))

    elif args.cmd == "batch_validate":
        cmd_batch_validate(dfmeta, str(args.pairs_csv), out_dir=str(args.out_dir), params=params, units=units, debug=args.debug, allow_no_growth=bool(args.allow_no_growth))


if __name__ == "__main__":
    main()
