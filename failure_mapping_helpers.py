# failure_mapping_helpers.py
import numpy as np

from rotation_helpers import (
    rot_to_material, rot_to_global,
    vec_rot_to_material, vec_rot_to_global,
)
from stress_helpers import stress_material_to_global, principal_from_components, eval_stress_field_material


# ============================================================
# Small angle utilities
# ============================================================

def _wrap_pi(a: float) -> float:
    return ((float(a) + np.pi) % (2.0 * np.pi)) - np.pi

def _wrap_pi_half(a: float) -> float:
    # line orientation in [-pi/2, pi/2)
    return ((float(a) + np.pi/2.0) % np.pi) - np.pi/2.0


# ============================================================
# Stress sign convention handling
#   All criteria below assume: tension positive, compression negative.
# ============================================================

def _detect_stress_sign_mode(s1, sxx=None, syy=None):
    """
    Heuristic:
      - If most of field is compressive, median(s1) < 0 (tension-positive convention)
      - If most of field is compressive but positive values dominate, then it's compression-positive.
    """
    s1 = np.asarray(s1, float)
    s1f = s1[np.isfinite(s1)]
    if s1f.size < 10:
        return "tension_positive"

    med = float(np.median(s1f))
    mx = float(np.max(s1f))
    mn = float(np.min(s1f))

    # Brazilian disk typically has mixed signs; compressive regions large.
    # If median is negative -> likely tension-positive (compression negative).
    if med < 0:
        return "tension_positive"

    # If median positive but we still see negatives -> ambiguous; assume compression-positive.
    if mn < 0 and mx > 0 and med > 0:
        return "compression_positive"

    # fallback
    return "tension_positive"


def _to_tension_positive(sxx, syy, txy, s1, stress_sign_mode="auto"):
    """
    Convert input stresses into tension-positive sign convention if needed.
    If compression-positive input is detected/selected, flip sign on all components.
    """
    mode = str(stress_sign_mode).lower().strip()
    if mode == "auto":
        mode = _detect_stress_sign_mode(s1, sxx, syy)

    if mode in ("compression_positive", "comp_positive", "compression"):
        return -sxx, -syy, -txy, -s1

    # already tension-positive
    return sxx, syy, txy, s1


# ============================================================
# Mohr-Coulomb scan + traction on plane
# ============================================================

def plane_sigma_tau(sxx, syy, txy, beta):
    """
    Plane traction using global stresses.
    beta = plane NORMAL angle in radians (arrow angle).
    """
    s_avg = 0.5 * (sxx + syy)
    s_diff = 0.5 * (sxx - syy)
    cb = np.cos(2 * beta)
    sb = np.sin(2 * beta)
    sigma_n = s_avg + s_diff * cb + txy * sb
    tau = -s_diff * sb + txy * cb
    return sigma_n, tau


def mc_scan_ratio(
    sxx, syy, txy, c, phi,
    n_theta=361, eps=1e-12,
    compression_only=True, sigma_comp_min=0.0
):
    """
    Returns:
      Is = rmax - 1
      beta_crit = critical plane-normal angle (arrow) for max ratio
    """
    beta = np.linspace(0, np.pi, int(n_theta), endpoint=False)[:, None]
    cb = np.cos(2 * beta)
    sb = np.sin(2 * beta)

    s_avg = 0.5 * (sxx + syy)[None, :]
    s_diff = 0.5 * (sxx - syy)[None, :]
    txyN = txy[None, :]

    sigma_n = s_avg + s_diff * cb + txyN * sb
    tau = -s_diff * sb + txyN * cb

    scmin = max(float(sigma_comp_min), 0.0)

    if compression_only:
        mask = sigma_n <= -scmin
        sigma_n_comp = np.where(mask, -sigma_n, 0.0)
        tau_allow = c[None, :] + sigma_n_comp * np.tan(phi[None, :])
        ratio = np.where(mask, np.abs(tau) / np.maximum(tau_allow, eps), 0.0)
    else:
        sigma_n_comp = np.where(sigma_n <= -scmin, -sigma_n, 0.0) if scmin > 0 else np.maximum(0.0, -sigma_n)
        tau_allow = c[None, :] + sigma_n_comp * np.tan(phi[None, :])
        ratio = np.abs(tau) / np.maximum(tau_allow, eps)

    i_max = np.argmax(ratio, axis=0)
    rmax = ratio[i_max, np.arange(ratio.shape[1])]
    beta_crit = beta[i_max, 0]
    Is = rmax - 1.0
    return Is, beta_crit


# ============================================================
# Weak-plane family with spacing (physics-inspired)
# ============================================================

def weak_plane_weight_xy(
    x, y,
    alpha_wp_line,
    spacing,
    phase=0.0,
    bandwidth_frac=0.12,
    floor=0.0,
    eps=1e-12,
):
    """
    Returns weight w in [0,1] indicating "how close you are to a weak plane band".

    alpha_wp_line : plane line angle (NOT normal), wrapped to [-pi/2,pi/2)
    spacing       : distance between parallel planes along the normal direction
    phase         : offset along normal so you don't force a plane through center
    bandwidth_frac: Gaussian width as fraction of spacing
    floor         : minimum baseline weakening (set 0.0 to avoid smearing everywhere)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    spacing = float(spacing)

    if (not np.isfinite(spacing)) or spacing <= 0:
        return np.zeros_like(x, dtype=float)

    a = _wrap_pi_half(float(alpha_wp_line))
    beta = a + np.pi / 2.0  # plane normal
    nx = np.cos(beta)
    ny = np.sin(beta)

    # signed distance to the plane family measured along normal direction
    d = nx * x + ny * y - float(phase)

    # wrap into [-s/2, s/2]
    s = spacing
    dmod = ((d + 0.5 * s) % s) - 0.5 * s

    bw = max(float(bandwidth_frac), 1e-6) * s
    w = np.exp(-(dmod / (bw + eps)) ** 2)

    floor = float(np.clip(floor, 0.0, 0.999999))
    w = floor + (1.0 - floor) * w
    return np.clip(w, 0.0, 1.0)


# ============================================================
# Failure mapping
#   RETURNS 4 values ALWAYS:
#     modes, Rt_eff, Rs_eff, beta_crit
#   Optional: return_util=True -> adds util as 5th output
# ============================================================

def failure_mode_map(
    sxx, syy, txy, s1,
    Tm, Coh, Phi,
    alpha_const: float,
    basis: str = "first",
    util_min: float = 0.98,
    mixed_band: float = 0.45,
    margin: float = 0.0,
    n_theta_mc: int = 361,
    mc_compression_only: bool = True,
    mc_sigma_comp_min: float = 0.0,

    strength_model: str = "weak_plane",
    weak_T_ratio: float = 0.35,
    weak_C_ratio: float = 0.60,
    weak_phi=None,

    # ---------- NEW: weak-plane spacing family ----------
    alpha_wp_line=None,        # line orientation of weak planes (global)
    x=None, y=None,            # coordinates for spacing weight
    weak_spacing: float = 0.0, # meters
    weak_phase: float = 0.0,   # meters offset along normal
    weak_bandwidth_frac: float = 0.12,
    weak_floor: float = 0.0,

    # ---------- NEW: stress sign ----------
    stress_sign_mode: str = "auto",

    # ---------- NEW: mixed gating + shear bias ----------
    conf_k: float = 0.65,   # require min(Rt,Rs) >= conf_k * util to call "mixed"
    wing_k: float = 0.0,    # optional amplification of shear on weak planes (0 disables)
    wing_p: float = 2.0,    # exponent for wing_k

    # ---------- return options ----------
    return_util: bool = False,

    eps: float = 1e-12,
    **_ignored_kwargs
):
    """
    Physics upgrades vs your older version:
      1) Robust sign handling (auto detects if your stresses are compression-positive).
      2) Weak-plane FAMILY (periodic planes) via distance-to-plane weight w(x,y).
      3) Weak-plane criteria only "activate" near planes (scaled by w), preventing smearing.
      4) Mixed classification requires BOTH ratios to be significant (conf_k gate),
         which removes the big, unrealistic "all mixed" cores you were seeing.

    Returns:
      modes (object array: "no_failure","tensile","shear","mixed")
      Rt_eff (float)
      Rs_eff (float)
      beta_crit (float)   # MC critical plane normal angle (arrow)
      (optional) util if return_util=True
    """
    # --- arrays ---
    sxx = np.asarray(sxx, float)
    syy = np.asarray(syy, float)
    txy = np.asarray(txy, float)
    s1  = np.asarray(s1,  float)
    Tm  = np.asarray(Tm,  float)
    Coh = np.asarray(Coh, float)
    Phi = np.asarray(Phi, float)

    # --- enforce tension-positive convention ---
    sxx, syy, txy, s1 = _to_tension_positive(sxx, syy, txy, s1, stress_sign_mode=stress_sign_mode)

    # --- matrix tensile ratio (principal tension) ---
    Rt_mat = np.maximum(s1, 0.0) / np.maximum(Tm, eps)

    # --- matrix Mohr-Coulomb shear ratio via scan ---
    Is_mat, beta_crit = mc_scan_ratio(
        sxx, syy, txy, Coh, Phi,
        n_theta=int(n_theta_mc), eps=eps,
        compression_only=bool(mc_compression_only),
        sigma_comp_min=float(mc_sigma_comp_min),
    )
    Rs_mat = Is_mat + 1.0

    Rt_wp = np.zeros_like(Rt_mat, dtype=float)
    Rs_wp = np.zeros_like(Rs_mat, dtype=float)

    # --- weak plane model ---
    if str(strength_model).lower() == "weak_plane":
        # plane line angle (global)
        if alpha_wp_line is None:
            alpha_wp_line = _wrap_pi_half(float(alpha_const))
        alpha_wp_line = _wrap_pi_half(float(alpha_wp_line))

        # plane normal angle
        beta_plane = float(alpha_wp_line + np.pi / 2.0)

        # spacing weight w(x,y)
        if (x is not None) and (y is not None) and (float(weak_spacing) > 0.0):
            w = weak_plane_weight_xy(
                x=np.asarray(x, float),
                y=np.asarray(y, float),
                alpha_wp_line=alpha_wp_line,
                spacing=float(weak_spacing),
                phase=float(weak_phase),
                bandwidth_frac=float(weak_bandwidth_frac),
                floor=float(weak_floor),
                eps=eps
            )
        else:
            # no spacing -> always active (legacy behavior)
            w = np.ones_like(Rt_mat, dtype=float)

        # "activation" weight (so far-from-plane does not trigger)
        # If weak_floor=0 -> w already goes to 0 far away.
        w_act = np.clip(w, 0.0, 1.0)

        sigma_n, tau = plane_sigma_tau(sxx, syy, txy, beta_plane)

        # reduced strengths on weak planes
        T_plane = float(weak_T_ratio) * Tm
        C_plane = float(weak_C_ratio) * Coh
        phi_plane = Phi if (weak_phi is None) else np.asarray(weak_phi, float)

        # tensile on plane
        Rt_wp = w_act * (np.maximum(sigma_n, 0.0) / np.maximum(T_plane, eps))

        # shear on plane (MC)
        scmin = max(float(mc_sigma_comp_min), 0.0)
        if bool(mc_compression_only):
            mask = sigma_n <= -scmin
            sigma_comp = np.where(mask, -sigma_n, 0.0)
            tau_allow = C_plane + sigma_comp * np.tan(phi_plane)
            Rs_wp = np.where(mask, w_act * (np.abs(tau) / np.maximum(tau_allow, eps)), 0.0)
        else:
            sigma_comp = np.where(sigma_n <= -scmin, -sigma_n, 0.0) if scmin > 0 else np.maximum(0.0, -sigma_n)
            tau_allow = C_plane + sigma_comp * np.tan(phi_plane)
            Rs_wp = w_act * (np.abs(tau) / np.maximum(tau_allow, eps))

        # optional: amplify shear very near weak planes (helps shear-dominated low angles)
        wing_k = float(wing_k)
        if wing_k != 0.0:
            Rs_wp = Rs_wp * (1.0 + wing_k * (w_act ** float(wing_p)))

    # --- effective ratios ---
    Rt_eff = np.maximum(Rt_mat, Rt_wp)
    Rs_eff = np.maximum(Rs_mat, Rs_wp)
    util = np.maximum(Rt_eff, Rs_eff)

    # --- classify ---
    modes = np.full(util.shape, "no_failure", dtype=object)
    hit = util >= float(util_min - margin)

    # mixed only if BOTH are significant relative to util
    conf_k = float(np.clip(conf_k, 0.0, 1.0))
    both_significant = (np.minimum(Rt_eff, Rs_eff) >= conf_k * util)

    # closeness in log-space is symmetric
    log_close = np.abs(np.log((Rt_eff + eps) / (Rs_eff + eps))) <= np.log(1.0 + float(mixed_band))

    mixed = hit & both_significant & log_close
    modes[mixed] = "mixed"

    tens = hit & (~mixed) & (Rt_eff > Rs_eff)
    modes[tens] = "tensile"

    shear = hit & (~mixed) & (~tens)
    modes[shear] = "shear"

    if return_util:
        return modes, Rt_eff, Rs_eff, beta_crit, util
    return modes, Rt_eff, Rs_eff, beta_crit


def disc_failure_point_stats(modes, r_pts, R, rmax_frac=0.985):
    """
    Count modes within r <= rmax_frac*R.
    Returns:
      disc_total, counts_dict, pct_dict
    """
    modes = np.asarray(modes, dtype=object)
    r_pts = np.asarray(r_pts, float)
    R = float(R)
    keep = r_pts <= float(rmax_frac) * R

    mk = modes[keep]
    disc_total = int(mk.size)

    counts = {
        "tensile": int(np.sum(mk == "tensile")),
        "mixed":   int(np.sum(mk == "mixed")),
        "shear":   int(np.sum(mk == "shear")),
        "no_failure": int(np.sum(mk == "no_failure")),
    }

    denom = max(disc_total, 1)
    pct = {
        "tensile": 100.0 * counts["tensile"] / denom,
        "mixed":   100.0 * counts["mixed"] / denom,
        "shear":   100.0 * counts["shear"] / denom,
        "no_failure": 100.0 * counts["no_failure"] / denom,
    }
    return disc_total, counts, pct


# ============================================================
# Crack-path sampling helper (kept compatible)
# ============================================================

class _UniformGridSampler:
    def __init__(self, X, Y):
        self.xmin = float(X[0, 0]); self.xmax = float(X[0, -1])
        self.ymin = float(Y[0, 0]); self.ymax = float(Y[-1, 0])
        self.nx = X.shape[1]; self.ny = X.shape[0]
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)

    def sample(self, F, x, y, fill=np.nan):
        x = float(x); y = float(y)
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return float(fill)

        fx = (x - self.xmin) / self.dx
        fy = (y - self.ymin) / self.dy
        j0 = int(np.floor(fx)); i0 = int(np.floor(fy))
        j1 = min(j0 + 1, self.nx - 1)
        i1 = min(i0 + 1, self.ny - 1)
        tx = fx - j0; ty = fy - i0

        f00 = F[i0, j0]; f10 = F[i0, j1]
        f01 = F[i1, j0]; f11 = F[i1, j1]

        vals = np.array([f00, f10, f01, f11], dtype=float)
        ok = np.isfinite(vals)
        if not np.all(ok):
            return float(vals[ok][0]) if np.any(ok) else float(fill)

        return float((1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 +
                     (1 - tx) * ty * f01 + tx * ty * f11)


def specimen_mode_from_crack_path(
    xs, ys,
    R, alpha_const, fit,
    X, Y,
    Teff_img, Coh_img, Phi_img,
    specimen_mode_basis="first",
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,
    strength_model="weak_plane",
    weak_T_ratio=0.35,
    weak_C_ratio=0.60,
    weak_phi=None,
    n_theta_mc=361,
    mixed_band=0.45,
    util_min=0.98,

    # spacing params (optional)
    alpha_wp_line=None,
    weak_spacing=0.0,
    weak_phase=0.0,
    weak_bandwidth_frac=0.12,
    weak_floor=0.0,
    stress_sign_mode="auto",
):
    """
    Returns:
      spec_mode (str),
      fr_path = dict(tensile=..., mixed=..., shear=...),
      L total path length
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.size < 2:
        return "no_failure", {"tensile": 0.0, "mixed": 0.0, "shear": 0.0}, np.nan

    pts = np.column_stack([xs, ys])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    L = float(np.sum(seg))

    sampler = _UniformGridSampler(X, Y)

    modes_path = []
    for i in range(1, xs.size - 1):
        x = float(xs[i]); y = float(ys[i])
        if x * x + y * y > (0.999 * float(R)) ** 2:
            continue

        Teff = sampler.sample(Teff_img, x, y, fill=np.nan)
        Coh  = sampler.sample(Coh_img,  x, y, fill=np.nan)
        Phi  = sampler.sample(Phi_img,  x, y, fill=np.nan)
        if not (np.isfinite(Teff) and np.isfinite(Coh) and np.isfinite(Phi)):
            continue

        xm, ym = rot_to_material(x, y, alpha_const)
        sxx_m, syy_m, txy_m = eval_stress_field_material(
            np.array([xm]), np.array([ym]),
            float(R), fit["p1"], fit["p2"], fit["a1"], fit["a2"]
        )
        sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, alpha_const)
        s1, _, _ = principal_from_components(sxx, syy, txy)

        modes_i, Rt_eff, Rs_eff, _beta, _util = failure_mode_map(
            np.asarray(sxx, float), np.asarray(syy, float), np.asarray(txy, float), np.asarray(s1, float),
            Tm=np.asarray([Teff], float),
            Coh=np.asarray([Coh], float),
            Phi=np.asarray([Phi], float),
            alpha_const=float(alpha_const),
            util_min=float(util_min),
            mixed_band=float(mixed_band),
            n_theta_mc=int(n_theta_mc),
            mc_compression_only=bool(mc_compression_only),
            mc_sigma_comp_min=float(mc_sigma_comp_min),
            strength_model=str(strength_model),
            weak_T_ratio=float(weak_T_ratio),
            weak_C_ratio=float(weak_C_ratio),
            weak_phi=weak_phi,

            alpha_wp_line=alpha_wp_line,
            x=np.asarray([x]), y=np.asarray([y]),
            weak_spacing=float(weak_spacing),
            weak_phase=float(weak_phase),
            weak_bandwidth_frac=float(weak_bandwidth_frac),
            weak_floor=float(weak_floor),

            stress_sign_mode=str(stress_sign_mode),
            return_util=True,
        )

        m = str(modes_i[0])
        if m in ("tensile", "mixed", "shear"):
            modes_path.append(m)

    if not modes_path:
        return "no_failure", {"tensile": 0.0, "mixed": 0.0, "shear": 0.0}, L

    modes_path = np.asarray(modes_path, dtype=object)
    nt = float(np.sum(modes_path == "tensile"))
    nm = float(np.sum(modes_path == "mixed"))
    ns = float(np.sum(modes_path == "shear"))
    denom = max(float(modes_path.size), 1.0)

    fr = {"tensile": 100.0 * nt / denom, "mixed": 100.0 * nm / denom, "shear": 100.0 * ns / denom}
    spec_mode = max(fr.keys(), key=lambda k: fr[k])
    return spec_mode, fr, L


def failure_ratios_pointwise(
    sxx, syy, txy, s1,
    Tm, Coh, Phi,
    alpha_const,

    n_theta_mc=361,
    mc_compression_only=True,
    mc_sigma_comp_min=0.0,

    strength_model="weak_plane",
    weak_T_ratio=0.35,
    weak_C_ratio=0.60,
    weak_phi=None,

    alpha_wp_line=None,
    x=None, y=None,
    weak_spacing=0.0,
    weak_phase=0.0,
    weak_bandwidth_frac=0.12,
    weak_floor=0.0,

    stress_sign_mode="auto",
    eps=1e-12,
):
    """
    Returns:
      Rt_eff, Rs_eff, beta_crit
    """
    modes, Rt_eff, Rs_eff, beta_crit = failure_mode_map(
        sxx, syy, txy, s1,
        Tm=Tm, Coh=Coh, Phi=Phi,
        alpha_const=alpha_const,
        n_theta_mc=int(n_theta_mc),
        mc_compression_only=bool(mc_compression_only),
        mc_sigma_comp_min=float(mc_sigma_comp_min),

        strength_model=str(strength_model),
        weak_T_ratio=float(weak_T_ratio),
        weak_C_ratio=float(weak_C_ratio),
        weak_phi=weak_phi,

        alpha_wp_line=alpha_wp_line,
        x=x, y=y,
        weak_spacing=float(weak_spacing),
        weak_phase=float(weak_phase),
        weak_bandwidth_frac=float(weak_bandwidth_frac),
        weak_floor=float(weak_floor),

        stress_sign_mode=str(stress_sign_mode),
        eps=eps
    )
    return Rt_eff, Rs_eff, beta_crit

def weak_plane_weight_field(X, Y, alpha_wp_line, spacing, phase=0.0, bandwidth_frac=0.12):
    """
    Returns weight ~1 near the weak planes (periodic family), ~0 between planes.
    Planes have LINE direction alpha_wp_line; normal is alpha_wp_line + 90deg.
    """
    spacing = float(spacing)
    if spacing <= 0:
        return np.zeros_like(X, float)

    n_ang = float(alpha_wp_line) + np.pi/2.0  # normal direction
    d = X*np.cos(n_ang) + Y*np.sin(n_ang) + float(phase)

    # map to nearest plane distance in [-spacing/2, spacing/2]
    dd = ((d + 0.5*spacing) % spacing) - 0.5*spacing
    dist = np.abs(dd)

    sigma = float(bandwidth_frac) * spacing + 1e-12
    return np.exp(-(dist / sigma) ** 2)
