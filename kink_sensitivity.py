# ==========================================================
# C) ORTHOTROPIC ENERGY SCALING: (1) sensitivity plot (easy + JMPS-safe)
# ==========================================================
# This is the “surrogate G” defense:
# show that the selected kink angle is insensitive to E' scaling,
# because for a fixed (KI,KII) computed by your DDM kernel,
# multiplying E' just scales G but does not change the argmax across θ
# UNLESS you compare across θ with different KI,KII responses that interact with thresholds.
#
# We do this properly by:
#   - running the kink scan at one step (or several steps)
#   - recomputing G(θ) for multiple E' choices
#   - showing θ* is unchanged (or nearly unchanged).
#
# Hook: call this at, say, step 10, 30, 60 for a few samples.

import numpy as np
import matplotlib.pyplot as plt

from geometry_helpers import angle_diff_periodic, _wrap_pi
from rotation_helpers import rot_to_material
from stress_helpers import (
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
)
from failure_mapping_helpers import failure_mode_map

# Optional (kept)
from cracked_disk_ddm import (_try_call_sif_two_tips, sif_two_tips_from_crack, traction_from_element_u,
                              total_traction_at_point_material, kink_angle_pls_anisotropic_from_traction, solve_cracked_disk_correction_ddm, traction_from_element)


from crack_helpers import Gc_theta_weak_plane, Eprime_equiv, wrap_pi_half

def kink_angle_sensitivity_Eprime(
    xs_u, ys_u,
    R, alpha_const, airy_fit,
    E1, E2, nu12, G12,
    ds,
    kink_scan_deg=25.0,
    kink_n=61,
    use_weak_plane=True,
    Gc0=1.0,
    weak_reduction=0.35,
    eta_deg=10.0,
    alpha_wp_line=None,
    ddm_sample_rs=np.logspace(-4, -2, 8),
    cod_offset=1e-5,
    Eprime_multipliers=(0.5, 0.8, 1.0, 1.25, 1.6, 2.0),
    objective="diff",
    **ddm_kwargs
):
    """
    Returns:
      thetas (rad), best_theta_by_mult dict, curves dict(mult -> score array)
    """
    if not _HAVE_SIF:
        raise RuntimeError("SIF solver not available.")

    if alpha_wp_line is None:
        alpha_wp_line = wrap_pi_half(float(alpha_const))

    tipx, tipy = float(xs_u[-1]), float(ys_u[-1])
    psi_tip = float(np.arctan2(ys_u[-1] - ys_u[-2], xs_u[-1] - xs_u[-2]))

    cap = np.deg2rad(float(kink_scan_deg))
    thetas = np.linspace(-cap, cap, int(max(9, kink_n)))
    obj = str(objective).lower().strip()

    # Base E' choice (your current)
    Eprime0 = float(Eprime_equiv(E1, E2))

    # Precompute KI,KII per candidate theta (expensive part)
    KIs = np.full_like(thetas, np.nan, float)
    KIIs = np.full_like(thetas, np.nan, float)
    Gcs = np.full_like(thetas, np.nan, float)

    for i, dth in enumerate(thetas):
        psi_new = float(psi_tip + dth)
        nx = tipx + float(ds) * np.cos(psi_new)
        ny = tipy + float(ds) * np.sin(psi_new)
        if (nx*nx + ny*ny) >= (0.999*float(R))**2:
            continue

        xtrial = np.asarray(list(xs_u) + [nx], float)
        ytrial = np.asarray(list(ys_u) + [ny], float)

        sif_res = _try_call_sif_two_tips(
            sif_two_tips_from_crack,
            xtrial, ytrial,
            R=float(R),
            alpha_const=float(alpha_const),
            airy_fit=airy_fit,
            eval_stress_field_material=eval_stress_field_material,
            E1=float(E1), E2=float(E2), nu12=float(nu12), G12=float(G12),
            sample_rs=ddm_sample_rs,
            cod_offset=float(cod_offset),
            **ddm_kwargs
        )
        KI0, KII0, KI, KII = _extract_tip_sifs(sif_res)
        if not (np.isfinite(KI) and np.isfinite(KII)):
            continue

        KIs[i] = float(KI)
        KIIs[i] = float(KII)

        theta_line = _wrap_pi_half(psi_new)
        if use_weak_plane:
            Gcs[i] = float(Gc_theta_weak_plane(
                theta_line=float(theta_line),
                alpha_wp=float(alpha_wp_line),
                Gc_matrix=float(Gc0),
                weak_reduction=float(weak_reduction),
                eta_deg=float(eta_deg),
            ))
        else:
            Gcs[i] = float(Gc0)

    curves = {}
    best_theta_by_mult = {}

    for mult in Eprime_multipliers:
        Eprime = float(Eprime0) * float(mult)
        G = (KIs*KIs + KIIs*KIIs) / (Eprime + 1e-30)

        if obj == "ratio":
            score = G / (Gcs + 1e-30)
        else:
            score = G - Gcs

        curves[mult] = score
        if np.any(np.isfinite(score)):
            j = int(np.nanargmax(score))
            best_theta_by_mult[mult] = float(thetas[j])
        else:
            best_theta_by_mult[mult] = np.nan

    return thetas, best_theta_by_mult, curves


def plot_kink_sensitivity(thetas, best_theta_by_mult, curves, out_png):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    for mult, sc in curves.items():
        ax.plot(np.rad2deg(thetas), sc, label=f"E' x{mult:g}")

    ax.set_xlabel("kink angle δ (deg)")
    ax.set_ylabel("score (G-Gc or G/Gc)")
    ax.set_title("Kink selection sensitivity to E′ scaling")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)

    # print summary (optional)
    print("Best δ by E' multiplier:")
    for mult, th in best_theta_by_mult.items():
        print(f"  x{mult:g}: {np.rad2deg(th):.2f} deg")
