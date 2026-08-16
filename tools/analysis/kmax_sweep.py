"""Mixed-mode fraction against the local damage cap k_max.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 63 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.

Scope
-----
This section covers both lithologies in one pass and is driven from Tensile_general_plots.ipynb, not from either lithology notebook.
"""
from __future__ import annotations


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.data_io import load_specimen_table
from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    choose_smallest_near_peak, load_sample_fields_physical,
    npz_path_for_sample, summarize_modes,
)



# --- implementation ---------------------------------------------------


def ensure_radians(phi):
    phi = np.asarray(phi, dtype=float)
    vals = phi[np.isfinite(phi)]
    if vals.size == 0:
        return phi
    if np.nanmax(np.abs(vals)) > (np.pi + 1e-6):
        return np.deg2rad(phi)
    return phi


def robust_unit_interval(x, qlo=20.0, qhi=95.0, eps=1e-12):
    """
    Robust percentile normalization to [0, 1].
    """
    x = np.asarray(x, dtype=float)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=float)

    lo = float(np.percentile(vals, qlo))
    hi = float(np.percentile(vals, qhi))
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=float)

    z = (x - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)


def evaluate_failure_mode_local_damage(
    s1, s3, U, Tens0, Coh0, phi0,
    tensile_w, wp_weight, conf_w,
    *,
    kC_max,
    kT_max=None,
    pU=1.25,
    pWP=1.20,
    pS=1.10,
    phi_drop_deg=0.0,
    u_qlo=20.0,
    u_qhi=95.0,
    mc_compression_only=True,
):
    """
    Local anisotropic damage-softening classifier.

    Parameters
    ----------
    kC_max : float
        Maximum local cohesion-loss cap. Must be in [0, 0.95].
    kT_max : float or None
        Maximum local tensile-strength loss cap. Must be in [0, 0.95].
        If None, uses 0.35 * kC_max.
    """
    s1 = np.asarray(s1, dtype=float)
    s3 = np.asarray(s3, dtype=float)
    U = np.asarray(U, dtype=float)
    Tens0 = np.asarray(Tens0, dtype=float)
    Coh0 = np.asarray(Coh0, dtype=float)
    phi0 = ensure_radians(phi0)

    tensile_w = np.clip(np.asarray(tensile_w, dtype=float), 0.0, 1.0)
    wp_weight = np.clip(np.asarray(wp_weight, dtype=float), 0.0, 1.0)
    conf_w = np.clip(np.asarray(conf_w, dtype=float), 0.0, 1.0)

    # Enforce principal ordering
    smax = np.maximum(s1, s3)
    smin = np.minimum(s1, s3)
    s1 = smax
    s3 = smin

    # Hard validation: do not silently clip beyond physical range
    if not (0.0 <= float(kC_max) <= 0.95):
        raise ValueError(
            f"kC_max={kC_max} is outside the valid range [0, 0.95] for this model."
        )
    kC_max = float(kC_max)

    if kT_max is None:
        kT_max = 0.35 * kC_max

    if not (0.0 <= float(kT_max) <= 0.95):
        raise ValueError(
            f"kT_max={kT_max} is outside the valid range [0, 0.95] for this model."
        )
    kT_max = float(kT_max)

    # Robust local energy activator
    Uhat = robust_unit_interval(U, qlo=u_qlo, qhi=u_qhi)

    wt = tensile_w
    ws = 1.0 - wt
    wwp = wp_weight
    wc = conf_w

    # Tensile-damage channel
    Dt = kT_max * (Uhat ** pU) * (0.25 + 0.75 * wt)

    # Shear/cohesion-damage channel
    Ds = (
        kC_max *
        (Uhat ** pU) *
        (0.20 + 0.80 * (wwp ** pWP)) *
        (0.20 + 0.80 * (ws ** pS)) *
        (0.50 + 0.50 * wc)
    )

    Dt = np.clip(Dt, 0.0, 0.95)
    Ds = np.clip(Ds, 0.0, 0.95)

    # Local softened strengths
    T_eff = np.maximum(0.05 * Tens0, Tens0 * (1.0 - Dt))
    C_eff = np.maximum(0.05 * Coh0, Coh0 * (1.0 - Ds))

    # Optional friction-angle softening
    if abs(float(phi_drop_deg)) > 0.0:
        phi_eff = phi0 - np.deg2rad(float(phi_drop_deg)) * Ds
        phi_eff = np.clip(phi_eff, np.deg2rad(5.0), np.deg2rad(85.0))
    else:
        phi_eff = phi0.copy()

    # Failure checks
    tensile_fail = (s1 >= T_eff)

    f_mc = (s1 - s3) + (s1 + s3) * np.sin(phi_eff) - 2.0 * C_eff * np.cos(phi_eff)
    shear_fail = (f_mc >= 0.0)

    if mc_compression_only:
        sigma_n_crit = 0.5 * (s1 + s3) - 0.5 * (s1 - s3) * np.sin(phi_eff)
        shear_fail &= (sigma_n_crit <= 0.0)

    modes = np.full(s1.shape, "no_failure", dtype=object)
    modes[tensile_fail & ~shear_fail] = "tensile"
    modes[shear_fail & ~tensile_fail] = "shear"
    modes[tensile_fail & shear_fail] = "mixed"

    extra = {
        "Uhat": Uhat,
        "Dt": Dt,
        "Ds": Ds,
        "T_eff": T_eff,
        "C_eff": C_eff,
        "phi_eff": phi_eff,
        "f_mc": f_mc,
        "tensile_fail": tensile_fail,
        "shear_fail": shear_fail,
    }
    return modes, extra


def compute_metric_curve_for_samples(
    sample_ids,
    kmax_list,
    dump_counts=False,
    mc_compression_only=True,
    kT_ratio=0.35,
    pU=1.25,
    pWP=1.20,
    pS=1.10,
    phi_drop_deg=0.0,
    u_qlo=20.0,
    u_qhi=95.0,
):
    """
    Returns mean/std curves across samples.

    Main metric:
      mixed fraction among failed points
      = mixed / (tensile + shear + mixed)

    Also returns:
      mixed fraction among all valid points
      = mixed / all_valid
      failure fraction
      = failed / all_valid

    Here kmax_list is interpreted as kC_max.
    """
    kvals = np.asarray(kmax_list, dtype=float)
    if np.any(kvals < 0.0) or np.any(kvals > 0.95):
        raise ValueError(
            "All kmax_list values must lie within [0, 0.95] for this model."
        )

    sub = df.loc[list(sample_ids)].sort_values("Angle")
    n_samples = len(sub)

    per_sample_mixed_failed = []
    per_sample_mixed_all = []
    per_sample_fail_frac = []

    for sid, row in sub.iterrows():
        angle_deg = float(row["Angle"])

        npz_path = npz_path_for_sample(DATA_DIR, sid)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Missing {npz_path} for sample {sid}. "
                f"Set DATA_DIR to the exporter folder, e.g. outputs/fields/fields_npz."
            )

        s1v, s3v, Uv, wtv, wwpv, wcv, Tv, Cv, phiv = load_sample_fields_physical(npz_path, row)

        mixed_failed_curve = np.full(len(kvals), np.nan, dtype=float)
        mixed_all_curve = np.full(len(kvals), np.nan, dtype=float)
        fail_frac_curve = np.full(len(kvals), np.nan, dtype=float)

        if dump_counts:
            print(f"\n=== Angle = {angle_deg:6.2f}° (sample {sid}) ===")
            print(f"Valid points: {s1v.size}")

        for i_k, kC_max in enumerate(kvals):
            kT_max = float(kT_ratio) * float(kC_max)

            modes, _extra = evaluate_failure_mode_local_damage(
                s1v, s3v, Uv, Tv, Cv, phiv,
                tensile_w=wtv,
                wp_weight=wwpv,
                conf_w=wcv,
                kC_max=float(kC_max),
                kT_max=float(kT_max),
                pU=float(pU),
                pWP=float(pWP),
                pS=float(pS),
                phi_drop_deg=float(phi_drop_deg),
                u_qlo=float(u_qlo),
                u_qhi=float(u_qhi),
                mc_compression_only=mc_compression_only,
            )

            counts = summarize_modes(modes)

            n_mix = counts["mixed"]
            n_tens = counts["tensile"]
            n_shear = counts["shear"]
            n_fail = counts["failed_total"]
            n_all = counts["all_total"]

            mixed_failed_curve[i_k] = (n_mix / float(n_fail)) if n_fail > 0 else np.nan
            mixed_all_curve[i_k] = n_mix / float(n_all)
            fail_frac_curve[i_k] = n_fail / float(n_all)

            if dump_counts:
                print(
                    f"kC_max = {kC_max:5.3f}:  "
                    f"kT_max = {kT_max:5.3f}:  "
                    f"Tensile = {n_tens:7d},  "
                    f"Shear   = {n_shear:7d},  "
                    f"Mixed   = {n_mix:7d},  "
                    f"Failed  = {n_fail:7d},  "
                    f"Mixed/Failed = {mixed_failed_curve[i_k]:.4f}"
                )

        per_sample_mixed_failed.append(mixed_failed_curve)
        per_sample_mixed_all.append(mixed_all_curve)
        per_sample_fail_frac.append(fail_frac_curve)

    per_sample_mixed_failed = np.asarray(per_sample_mixed_failed, dtype=float)
    per_sample_mixed_all = np.asarray(per_sample_mixed_all, dtype=float)
    per_sample_fail_frac = np.asarray(per_sample_fail_frac, dtype=float)

    out = {
        "mixed_failed_mean": np.nanmean(per_sample_mixed_failed, axis=0),
        "mixed_failed_std": np.nanstd(per_sample_mixed_failed, axis=0, ddof=1) if n_samples > 1 else np.zeros(len(kvals)),
        "mixed_all_mean": np.nanmean(per_sample_mixed_all, axis=0),
        "mixed_all_std": np.nanstd(per_sample_mixed_all, axis=0, ddof=1) if n_samples > 1 else np.zeros(len(kvals)),
        "fail_frac_mean": np.nanmean(per_sample_fail_frac, axis=0),
        "fail_frac_std": np.nanstd(per_sample_fail_frac, axis=0, ddof=1) if n_samples > 1 else np.zeros(len(kvals)),
        "n_samples": n_samples,
    }
    return out



def main():
    """Run this section for both lithologies in one pass."""
    global CSV_PATH, DATA_DIR, KT_RATIO, MC_COMPRESSION_ONLY, OUTPUT_FIG, \
        PHI_DROP_DEG, P_S, P_U, P_WP, REL_TOL, U_QHIGH, U_QLOW, ax, \
        best_idx_g, best_idx_s, best_kC_g, best_kC_s, best_kT_g, best_kT_s, \
        df, fig, kmax_list, leg, missing, peak_g, peak_s, required_cols, \
        res_g, res_s, sample_ids_gneiss, sample_ids_schist, upper_g, \
        upper_s
    """
    k_max sensitivity analysis using LOCAL ANISOTROPIC damage-softening
    on exported full-field .npz files.

    Physics in this revision
    ------------------------
    Instead of uniform cohesion softening everywhere:

        C_soft = C0 * (1 - k_max)

    this script uses local, anisotropic, energy-driven softening:

        T_eff(x,y) = T0 * [1 - D_t(x,y)]
        C_eff(x,y) = C0 * [1 - D_s(x,y)]

    with:
        D_t ~ kT_max * f(Uhat, tensile_w)
        D_s ~ kC_max * f(Uhat, wp_weight, shear_tendency, confinement)

    The scanned parameter in kmax_list is interpreted as:
        kC_max = maximum local cohesion-loss cap

    and tensile softening is tied as:
        kT_max = KT_RATIO * kC_max

    Selection rule
    --------------
    To avoid unphysical "best value = upper scan bound" behavior,
    the selected value is NOT the raw argmax.

    Instead, the script chooses:
        the smallest kC_max whose metric is within 98% of the peak.

    Reads:
      - tensile_samples_data.csv
      - exported files:
            outputs/fields/fields_npz/sample_0001_full_fields.npz
            ...

    Required NPZ keys:
      - s1, s3, U_MPa, M, tensile_w, wp_weight, conf_w
    """
    CSV_PATH = "tensile_samples_data.csv"
    DATA_DIR = output_dirs.FIELDS_NPZ
    OUTPUT_FIG = "kmax_local_anisotropic_mixed_fraction.pdf"
    sample_ids_gneiss = range(1, 8)
    sample_ids_schist = range(8, 15)
    kmax_list = np.linspace(0.05, 0.90, 18)
    MC_COMPRESSION_ONLY = True
    KT_RATIO = 0.35
    P_U = 1.25
    P_WP = 1.20
    P_S = 1.10
    PHI_DROP_DEG = 0.0
    U_QLOW = 20.0
    U_QHIGH = 95.0
    REL_TOL = 0.02
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    df = load_specimen_table()
    df.columns = [str(c).strip() for c in df.columns]
    required_cols = [
        "Rock_type", "Angle",
        "Diameter_mm", "Thickness_mm", "Load_(KN)",
        "Tensile_strength_Mpa",
        "Cohesion", "Friction_Angle",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    res_g = compute_metric_curve_for_samples(
        sample_ids_gneiss,
        kmax_list=kmax_list,
        dump_counts=False,
        mc_compression_only=MC_COMPRESSION_ONLY,
        kT_ratio=KT_RATIO,
        pU=P_U,
        pWP=P_WP,
        pS=P_S,
        phi_drop_deg=PHI_DROP_DEG,
        u_qlo=U_QLOW,
        u_qhi=U_QHIGH,
    )
    res_s = compute_metric_curve_for_samples(
        sample_ids_schist,
        kmax_list=kmax_list,
        dump_counts=False,
        mc_compression_only=MC_COMPRESSION_ONLY,
        kT_ratio=KT_RATIO,
        pU=P_U,
        pWP=P_WP,
        pS=P_S,
        phi_drop_deg=PHI_DROP_DEG,
        u_qlo=U_QLOW,
        u_qhi=U_QHIGH,
    )
    best_idx_g, best_kC_g, peak_g, upper_g = choose_smallest_near_peak(
        kmax_list, res_g["mixed_failed_mean"], rel_tol=REL_TOL
    )
    best_idx_s, best_kC_s, peak_s, upper_s = choose_smallest_near_peak(
        kmax_list, res_s["mixed_failed_mean"], rel_tol=REL_TOL
    )
    best_kT_g = KT_RATIO * best_kC_g
    best_kT_s = KT_RATIO * best_kC_s
    print(f"Best kC_max for Augen gneiss      : {best_kC_g:.3f}")
    print(f"Best kT_max for Augen gneiss      : {best_kT_g:.3f}")
    print(f"Best kC_max for Psammitic schist  : {best_kC_s:.3f}")
    print(f"Best kT_max for Psammitic schist  : {best_kT_s:.3f}")
    print("\nMetric used for selection: mixed / failed_total")
    print(f"Selection rule: smallest k within {(1.0 - REL_TOL) * 100:.1f}% of peak metric")
    print("Model: local anisotropic energy-driven damage softening")
    if upper_g:
        print("[WARN] Augen gneiss curve is still near peak at the upper scan bound.")
    if upper_s:
        print("[WARN] Psammitic schist curve is still near peak at the upper scan bound.")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(
        kmax_list, res_g["mixed_failed_mean"],
        marker="o", linestyle="-", linewidth=1.8, markersize=6,
        label="Augen gneiss"
    )
    ax.plot(
        kmax_list, res_s["mixed_failed_mean"],
        marker="s", linestyle="-", linewidth=1.8, markersize=6,
        label="Psammitic schist"
    )
    ax.fill_between(
        kmax_list,
        res_g["mixed_failed_mean"] - res_g["mixed_failed_std"],
        res_g["mixed_failed_mean"] + res_g["mixed_failed_std"],
        alpha=0.15
    )
    ax.fill_between(
        kmax_list,
        res_s["mixed_failed_mean"] - res_s["mixed_failed_std"],
        res_s["mixed_failed_mean"] + res_s["mixed_failed_std"],
        alpha=0.15
    )
    ax.plot(best_kC_g, res_g["mixed_failed_mean"][best_idx_g], "o", ms=8)
    ax.plot(best_kC_s, res_s["mixed_failed_mean"][best_idx_s], "s", ms=8)
    ax.set_xlabel(r"$k_{C,\mathrm{max}}$")
    ax.set_ylabel("Mixed fraction among failed points")
    ax.set_xlim(float(np.min(kmax_list)), float(np.max(kmax_list)))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.35)
    leg = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        edgecolor="black"
    )
    leg.get_frame().set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Figure saved to {OUTPUT_FIG}")
