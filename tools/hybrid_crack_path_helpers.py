#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid crack path:
  1) coarse path from grid_phys Dijkstra (on util field)
  2) local physics walker refinement constrained to a corridor around coarse path
  3) mirror the refined branch to get full path

Designed to be stable: the coarse path gives topology; the local walker adds physics.
"""

from __future__ import annotations

import numpy as np

from.geometry_helpers import angle_diff_periodic, _wrap_pi
from.rotation_helpers import rot_to_material
from.stress_helpers import (
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
)
from.failure_mapping_helpers import failure_ratios_pointwise

# coarse path solver
from.grid_phys_method import crack_path_grid_phys_mirror


# ==========================================================
# small utilities
# ==========================================================
def _wrap_pi_half(a: float) -> float:
    return ((float(a) + np.pi / 2.0) % np.pi) - np.pi / 2.0

def angle_diff_line(a: float, b: float) -> float:
    """Smallest difference when angles represent a LINE (θ ≡ θ+π)."""
    da = abs(angle_diff_periodic(a, b))
    db = abs(angle_diff_periodic(a, b + np.pi))
    return min(da, db)

def _angle_mean(a: float, b: float, w: float) -> float:
    """Mean of directed angles (arrow)."""
    a = float(a); b = float(b); w = float(w)
    z = (1.0 - w) * np.exp(1j * a) + w * np.exp(1j * b)
    if abs(z) < 1e-20:
        return _wrap_pi(a)
    return _wrap_pi(np.angle(z))

def _align_line_direction(psi_target: float, psi_ref: float) -> float:
    """Choose psi_target or psi_target+pi to be closest to psi_ref."""
    psi_target = _wrap_pi(float(psi_target))
    psi_ref = _wrap_pi(float(psi_ref))
    d1 = abs(angle_diff_periodic(psi_target, psi_ref))
    d2 = abs(angle_diff_periodic(psi_target + np.pi, psi_ref))
    return _wrap_pi(psi_target if d1 <= d2 else (psi_target + np.pi))

def _seg_intersect(p1, p2, q1, q2, eps=1e-12) -> bool:
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def onseg(a,b,c):
        return (min(a[0],b[0])-eps <= c[0] <= max(a[0],b[0])+eps) and (min(a[1],b[1])-eps <= c[1] <= max(a[1],b[1])+eps)
    o1 = orient(p1,p2,q1)
    o2 = orient(p1,p2,q2)
    o3 = orient(q1,q2,p1)
    o4 = orient(q1,q2,p2)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if abs(o1) <= eps and onseg(p1,p2,q1): return True
    if abs(o2) <= eps and onseg(p1,p2,q2): return True
    if abs(o3) <= eps and onseg(q1,q2,p1): return True
    if abs(o4) <= eps and onseg(q1,q2,p2): return True
    return False


# ==========================================================
# Sampling helper (bilinear)
# ==========================================================
class UniformGridSampler:
    def __init__(self, X, Y):
        self.xmin = float(X[0, 0]); self.xmax = float(X[0, -1])
        self.ymin = float(Y[0, 0]); self.ymax = float(Y[-1, 0])
        self.nx = X.shape[1]; self.ny = X.shape[0]
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)

    def sample(self, F, x, y, fill=np.nan):
        x = float(x); y = float(y)
        if (x < self.xmin) or (x > self.xmax) or (y < self.ymin) or (y > self.ymax):
            return float(fill)
        fx = (x - self.xmin) / self.dx
        fy = (y - self.ymin) / self.dy
        j0 = int(np.floor(fx)); i0 = int(np.floor(fy))
        j1 = min(j0 + 1, self.nx - 1); i1 = min(i0 + 1, self.ny - 1)
        tx = fx - j0; ty = fy - i0
        f00 = F[i0, j0]; f10 = F[i0, j1]; f01 = F[i1, j0]; f11 = F[i1, j1]
        if (not np.isfinite(f00)) or (not np.isfinite(f10)) or (not np.isfinite(f01)) or (not np.isfinite(f11)):
            vals = np.array([f00, f10, f01, f11], dtype=float)
            ok = np.isfinite(vals)
            return float(vals[ok][0]) if np.any(ok) else float(fill)
        return float((1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11)


# ==========================================================
# Polyline utilities (nearest point + tangent)
# ==========================================================
def _nearest_polyline_info(x, y, xs, ys):
    """
    Return:
      dmin, xproj, yproj, ang_tan
    where ang_tan is the segment direction (arrow angle).
    """
    x = float(x); y = float(y)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    best_d2 = np.inf
    best = (0.0, 0.0, 0.0)

    for k in range(len(xs) - 1):
        x1, y1 = xs[k], ys[k]
        x2, y2 = xs[k+1], ys[k+1]
        vx = x2 - x1; vy = y2 - y1
        L2 = vx*vx + vy*vy
        if L2 < 1e-20:
            continue
        t = ((x - x1)*vx + (y - y1)*vy) / L2
        t = float(np.clip(t, 0.0, 1.0))
        xp = x1 + t*vx
        yp = y1 + t*vy
        dx = x - xp; dy = y - yp
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best = (xp, yp, np.arctan2(vy, vx))

    return float(np.sqrt(best_d2)), float(best[0]), float(best[1]), float(best[2])

def _pick_one_branch_from_mirrored(xs, ys):
    """
    coarse grid_phys returns a mirrored full path.
    split it at closest-to-origin and pick the branch whose end is 'upper' (y larger).
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    i0 = int(np.argmin(xs*xs + ys*ys))

    b1x = xs[i0:];     b1y = ys[i0:]
    b2x = xs[:i0+1][::-1]; b2y = ys[:i0+1][::-1]

    # pick branch that ends with larger y (usually the "upper" one)
    if len(b1y) == 0:
        return b2x, b2y
    if len(b2y) == 0:
        return b1x, b1y
    return (b1x, b1y) if (b1y[-1] >= b2y[-1]) else (b2x, b2y)


# ==========================================================
# Physics ratios + directions at a point
# ==========================================================
def _ratios_and_dirs_at_point(
    x, y,
    R, alpha_const, fit,
    sampler, Teff_img, Coh_img, Phi_img,
    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.35,
    weak_C_ratio=0.60,
    weak_phi=None,
    eps=1e-12
):
    Teff = sampler.sample(Teff_img, x, y, fill=np.nan)
    Coh  = sampler.sample(Coh_img,  x, y, fill=np.nan)
    Phi  = sampler.sample(Phi_img,  x, y, fill=np.nan)
    if (not np.isfinite(Teff)) or (not np.isfinite(Coh)) or (not np.isfinite(Phi)):
        return np.nan, np.nan, np.nan, np.nan

    xm, ym = rot_to_material(np.array([x], float), np.array([y], float), float(alpha_const))
    sxx_m, syy_m, txy_m = eval_stress_field_material(
        xm, ym, float(R),
        fit["p1"], fit["p2"], fit["a1"], fit["a2"]
    )
    sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, float(alpha_const))

    s1, _, th = principal_from_components(sxx, syy, txy)
    s1 = float(s1[0]); th = float(th[0])
    sxx = float(sxx[0]); syy = float(syy[0]); txy = float(txy[0])

    weak_phi_in = None
    if weak_phi is not None:
        weak_phi_in = np.array([float(weak_phi)], float)

    Rt_eff, Rs_eff, beta_crit = failure_ratios_pointwise(
        np.array([sxx], float), np.array([syy], float), np.array([txy], float), np.array([s1], float),
        np.array([Teff], float), np.array([Coh], float), np.array([Phi], float),
        float(alpha_const),
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),
        strength_model=str(strength_model),
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi_in,
        eps=float(eps),
    )

    Rt = float(Rt_eff[0]); Rs = float(Rs_eff[0]); bc = float(beta_crit[0])

    psi_t = _wrap_pi(th + np.pi/2.0)
    psi_s = _wrap_pi(bc + np.pi/2.0)
    return Rt, Rs, psi_t, psi_s

def _util_at_point(*args, **kwargs):
    Rt, Rs, _, _ = _ratios_and_dirs_at_point(*args, **kwargs)
    if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
        return np.nan
    return float(max(Rt, Rs))


# ==========================================================
# The hybrid crack path
# ==========================================================
def crack_path_hybrid_mirror_physics(
    R, alpha_const, fit,
    X, Y, M,
    Teff_img, Coh_img, Phi_img,
    util_img,
    psi_pref_img=None,

    # ----- coarse grid_phys knobs (forwarded) -----
    grid_start_r_frac=0.03,
    grid_r_end_frac=0.99,
    grid_util_soft_min=0.98,
    grid_util_hard_min=0.20,
    grid_widest_relax=0.98,
    grid_util_power=6.0,
    grid_dip_penalty=25.0,
    grid_center_weight=4.0,
    grid_dir_penalty=1.5,
    grid_turn_penalty=1.0,
    grid_backtrack_penalty=1.0,
    grid_smooth_window=11,
    grid_smooth_iters=2,
    grid_snap_radius=2,

    # ----- refine corridor knobs -----
    corridor_width_frac=0.06,     # corridor half-width around coarse (fraction of R)
    corridor_weight=2.0,          # penalize distance to coarse
    tangent_weight=1.5,           # prefer aligning with coarse tangent (line)
    corridor_hard_clip=2.5,       # discard candidates farther than clip*width

    # ----- local physics walker knobs -----
    drive_basis="first",
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.35,
    weak_C_ratio=0.60,
    weak_phi=None,

    ds_frac=0.004,               # smaller than normal local (refinement)
    a0_frac=0.004,
    max_steps=4000,
    mixed_band=0.45,
    relax=0.75,
    n_theta_mc=361,
    fail_margin=0.0,

    max_turn_deg=12.0,
    hysteresis=1.25,
    target_smooth=0.45,

    crack_path_util_min=0.98,
    seg_check_n=5,
    ds_retry_factors=(1.0, 0.75, 0.55, 0.40, 0.30),
    cand_n=31,
    target_penalty=0.30,
    turn_penalty=0.30,
    self_intersection_stop=True,
    eps=1e-12
):
    R = float(R)
    alpha_const = float(alpha_const)
    sampler = UniformGridSampler(X, Y)

    # --------------------------
    # 1) COARSE PATH (grid_phys)
    # --------------------------
    xs0, ys0, _ = crack_path_grid_phys_mirror(
        R=R, X=X, Y=Y, M=M,
        util_img=util_img,
        psi_pref_img=psi_pref_img,
        start_r_frac=float(grid_start_r_frac),
        r_end_frac=float(grid_r_end_frac),
        util_soft_min=float(grid_util_soft_min),
        util_hard_min=float(grid_util_hard_min),
        widest_relax=float(grid_widest_relax),
        util_power=float(grid_util_power),
        dip_penalty=float(grid_dip_penalty),
        center_weight=float(grid_center_weight),
        dir_penalty=float(grid_dir_penalty),
        turn_penalty=float(grid_turn_penalty),
        backtrack_penalty=float(grid_backtrack_penalty),
        smooth_window=int(grid_smooth_window),
        smooth_iters=int(grid_smooth_iters),
        snap_radius=int(grid_snap_radius),
    )

    # choose one branch to refine (center -> boundary)
    xu_coarse, yu_coarse = _pick_one_branch_from_mirrored(xs0, ys0)
    if len(xu_coarse) < 2:
        # fallback: return coarse
        return np.asarray(xs0, float), np.asarray(ys0, float), np.pi/2.0

    # enforce exact start at origin
    xu_coarse = xu_coarse.copy()
    yu_coarse = yu_coarse.copy()
    xu_coarse[0] = 0.0
    yu_coarse[0] = 0.0

    psi0 = float(np.arctan2(yu_coarse[1] - yu_coarse[0], xu_coarse[1] - xu_coarse[0]))

    # corridor width in meters
    corridor_w = float(corridor_width_frac) * R
    corridor_clip = float(corridor_hard_clip) * corridor_w

    # --------------------------
    # 2) REFINEMENT WALKER
    # --------------------------
    ds0 = float(ds_frac) * R
    a0  = float(a0_frac) * R
    m   = float(fail_margin)

    xs = [0.0, a0*np.cos(psi0)]
    ys = [0.0, a0*np.sin(psi0)]
    psi = float(psi0)

    cap = np.deg2rad(float(max_turn_deg))
    hys = float(max(hysteresis, 1.0))
    cand_n = int(max(cand_n, 7))
    seg_check_n = int(max(seg_check_n, 3))
    seg_t = np.linspace(0.15, 1.0, seg_check_n)

    drive_mode_prev = None
    psi_target_prev = psi

    for step_i in range(int(max_steps)):
        tip_x, tip_y = xs[-1], ys[-1]
        if tip_x*tip_x + tip_y*tip_y >= (0.999*R)**2:
            break

        # nearest coarse info at current tip
        d_tip, _, _, psi_coarse_tan = _nearest_polyline_info(tip_x, tip_y, xu_coarse, yu_coarse)

        # if we drift too far, snap by stopping refinement (coarse already ok)
        if d_tip > corridor_clip:
            break

        Rt, Rs, psi_t, psi_s = _ratios_and_dirs_at_point(
            tip_x, tip_y,
            R, alpha_const, fit,
            sampler, Teff_img, Coh_img, Phi_img,
            n_theta_mc=n_theta_mc,
            mc_compression_only=mc_compression_only,
            mc_sigma_comp_min=mc_sigma_comp_min,
            strength_model=strength_model,
            weak_T_ratio=weak_T_ratio,
            weak_C_ratio=weak_C_ratio,
            weak_phi=weak_phi,
            eps=eps
        )
        if (not np.isfinite(Rt)) or (not np.isfinite(Rs)):
            break

        if str(drive_basis).lower() == "now":
            if (Rt < 1.0 + m) and (Rs < 1.0 + m):
                break

        # align line-direction consistency
        psi_t = _align_line_direction(psi_t, psi)
        psi_s = _align_line_direction(psi_s, psi)

        # choose physics target
        if str(drive_basis).lower() == "first":
            kt = (1.0/Rt) if Rt > eps else np.inf
            ks = (1.0/Rs) if Rs > eps else np.inf
            kmin = min(kt, ks)
            rel = abs(kt - ks) / max(kmin, eps)
            if rel <= float(mixed_band):
                drive_mode = "mixed"
            else:
                drive_mode = "tensile" if kt <= ks else "shear"

            if drive_mode_prev is not None and drive_mode != drive_mode_prev and drive_mode != "mixed":
                if drive_mode_prev == "tensile" and not (ks < kt / hys):
                    drive_mode = drive_mode_prev
                if drive_mode_prev == "shear" and not (kt < ks / hys):
                    drive_mode = drive_mode_prev

            if drive_mode == "mixed":
                w = Rt / (Rt + Rs + eps)
                psi_target = _angle_mean(psi_s, psi_t, w)
            elif drive_mode == "tensile":
                psi_target = psi_t
            else:
                psi_target = psi_s
        else:
            rel = abs(Rt - Rs) / max(max(Rt, Rs), eps)
            if rel <= float(mixed_band):
                drive_mode = "mixed"
            else:
                drive_mode = "tensile" if Rt >= Rs else "shear"

            if drive_mode_prev is not None and drive_mode != drive_mode_prev and drive_mode != "mixed":
                if drive_mode_prev == "tensile" and not (Rs > Rt * hys):
                    drive_mode = drive_mode_prev
                if drive_mode_prev == "shear" and not (Rt > Rs * hys):
                    drive_mode = drive_mode_prev

            if drive_mode == "mixed":
                w = Rt / (Rt + Rs + eps)
                psi_target = _angle_mean(psi_s, psi_t, w)
            elif drive_mode == "tensile":
                psi_target = psi_t
            else:
                psi_target = psi_s

        drive_mode_prev = drive_mode

        # smooth physics target
        psi_target = _angle_mean(psi_target_prev, psi_target, float(target_smooth))
        psi_target_prev = psi_target

        # build a "guided" center direction = physics + coarse tangent (stronger when close)
        psi_coarse_al = _align_line_direction(psi_coarse_tan, psi_target)
        wguide = float(np.exp(-(d_tip / (0.75*corridor_w + 1e-12))**2))
        psi_center = _angle_mean(psi_target, psi_coarse_al, wguide)

        # apply relax + cap
        dpsi0 = angle_diff_periodic(psi_center, psi)
        dpsi0 = np.clip(dpsi0, -cap, cap)
        psi_center = _wrap_pi(psi + float(relax)*dpsi0)

        # candidate angles around current heading
        cand = psi + np.linspace(-cap, cap, cand_n)
        cand = np.array([_wrap_pi(ci) for ci in cand], dtype=float)

        best = None

        for f in ds_retry_factors:
            ds = ds0 * float(f)
            best = None

            for pc in cand:
                pc = _align_line_direction(pc, psi)

                nx = tip_x + ds*np.cos(pc)
                ny = tip_y + ds*np.sin(pc)

                if nx*nx + ny*ny >= (0.999*R)**2:
                    continue

                # corridor hard clip at the NEW point
                d_new, _, _, psi_tan_new = _nearest_polyline_info(nx, ny, xu_coarse, yu_coarse)
                if d_new > corridor_clip:
                    continue

                # util check along segment
                utils = []
                ok = True
                for tt in seg_t:
                    xi = tip_x + (tt*ds)*np.cos(pc)
                    yi = tip_y + (tt*ds)*np.sin(pc)
                    if xi*xi + yi*yi >= (0.999*R)**2:
                        ok = False
                        break
                    u = _util_at_point(
                        xi, yi,
                        R, alpha_const, fit,
                        sampler, Teff_img, Coh_img, Phi_img,
                        n_theta_mc=n_theta_mc,
                        mc_compression_only=mc_compression_only,
                        mc_sigma_comp_min=mc_sigma_comp_min,
                        strength_model=strength_model,
                        weak_T_ratio=weak_T_ratio,
                        weak_C_ratio=weak_C_ratio,
                        weak_phi=weak_phi,
                        eps=eps
                    )
                    if (not np.isfinite(u)) or (u < float(crack_path_util_min)):
                        ok = False
                        break
                    utils.append(u)

                if not ok:
                    continue

                u_avg = float(np.mean(utils))

                # penalties
                dturn = abs(angle_diff_periodic(pc, psi))
                dtarget = abs(angle_diff_periodic(pc, psi_center))

                # tangent alignment (line) at the candidate point
                d_tan = angle_diff_line(pc, psi_tan_new)

                # corridor distance penalty
                corr_pen = float(corridor_weight) * (d_new / (corridor_w + 1e-12))**2

                score = (
                    u_avg
                    - float(turn_penalty)   * (dturn/max(cap, 1e-9))**2
                    - float(target_penalty) * (dtarget/max(cap, 1e-9))**2
                    - float(tangent_weight) * (d_tan/(0.5*np.pi))**2
                    - corr_pen
                )

                if (best is None) or (score > best[0]):
                    best = (score, nx, ny, pc)

            if best is not None:
                break

        if best is None:
            break

        _, nx, ny, psi_new = best

        # optional self-intersection stop
        if self_intersection_stop and len(xs) > 25 and (step_i % 2 == 0):
            p1s = (xs[-1], ys[-1])
            p2s = (nx, ny)
            hit = False
            for j in range(1, len(xs) - 10):
                q1 = (xs[j-1], ys[j-1])
                q2 = (xs[j], ys[j])
                if _seg_intersect(p1s, p2s, q1, q2):
                    hit = True
                    break
            if hit:
                break

        psi = float(psi_new)
        xs.append(float(nx))
        ys.append(float(ny))

    # --------------------------
    # 3) Mirror refined branch
    # --------------------------
    xu = np.array(xs, float)
    yu = np.array(ys, float)

    xd = -xu
    yd = -yu
    Xs = np.concatenate([xd[::-1], xu[1:]])
    Ys = np.concatenate([yd[::-1], yu[1:]])

    psi0_out = float(np.arctan2(yu[1] - yu[0], xu[1] - xu[0])) if len(xu) >= 2 else float(np.pi/2)
    return Xs, Ys, psi0_out
