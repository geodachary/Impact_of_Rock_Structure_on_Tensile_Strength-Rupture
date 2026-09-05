"""Advanced stress evaluation and the local Mohr-Coulomb failure-mode classifier.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 67 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.

Scope
-----
This section covers both lithologies in one pass and is driven from Tensile_general_plots.ipynb, not from either lithology notebook.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from tools.data_io import load_specimen_table

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    create_grid_mesh, generate_random_props, principal_from_components,
    splitting_stress_pa, surrogate_conf_weight, surrogate_energy_proxy,
    surrogate_tensile_weight, surrogate_wp_weight,
)



# --- implementation ---------------------------------------------------


def wrap_pi(a):
    return ((np.asarray(a, float) + np.pi) % (2*np.pi)) - np.pi


def angle_diff_periodic(a, b):
    return wrap_pi(np.asarray(a, float) - np.asarray(b, float))


def ensure_radians(phi):
    phi = np.asarray(phi, dtype=float)
    vals = phi[np.isfinite(phi)]
    if vals.size and np.nanmax(np.abs(vals)) > (np.pi + 1e-6):
        return np.deg2rad(phi)
    return phi


def robust_unit_interval(x, qlo=20.0, qhi=95.0, eps=1e-12):
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


def resolve_local_damage_params(rock_type):
    """Softening caps for this lithology, from the one module that owns them.

    This was a second copy of the resolver in ``local_damage``, reading four
    module-level names that are only bound when ``main`` runs. Calling it
    before then raised ``NameError``, and after then it could return values
    that module did not agree with. It now delegates, so there is one source.
    """
    from .local_damage import resolve_local_damage_params as _resolve
    return _resolve(rock_type)


def anisotropic_tensile_strength(base_T, anis_angle, crack_plane_angle,
                                 spacing=0.0, x=None, y=None, min_factor=0.30):
    """
    Effective tensile strength field (same units as base_T).

    base_T            : scalar tensile strength (MPa)
    anis_angle        : anisotropy normal angle (rad)
    crack_plane_angle : assumed crack plane angle (rad)
    spacing           : foliation spacing (m)
    x, y              : coordinates (m)
    """
    # Delta is measured from the foliation NORMAL: Delta = 0 is a crack
    # running across the planes, Delta = pi/2 one running along them. The
    # caller supplies the foliation line direction, so the normal is that
    # plus pi/2.
    normal = float(anis_angle) + np.pi / 2.0
    dphi = angle_diff_periodic(normal, crack_plane_angle)
    ang_fac = np.cos(dphi)**2

    if spacing is not None and spacing > 0.0 and (x is not None) and (y is not None):
        ca = np.cos(normal)
        sa = np.sin(normal)
        xp = ca * x + sa * y            # offset along the foliation normal
        # Minimum on an interface, intact value midway between interfaces.
        # cos^2 here would put the strongest material on the weak planes.
        space_fac = np.sin(np.pi * xp / spacing)**2
    else:
        space_fac = 1.0

    fac = np.clip(ang_fac * space_fac, min_factor, 1.0)
    return base_T * fac


def calculate_advanced_stress(x, y, r, D, P, t, base_T, anis_angle_x, spacing,
                              shear_amp=1.2):
    """
    Simplified surrogate stress field.

    Returns:
        s1, s3, sxx, syy, txy    (all in MPa)

    Notes:
      - This is a toy stress field for sensitivity testing
      - Not a full Brazilian-disk orthotropic solution
    """
    applied_MPa = splitting_stress_pa(P, D, t) / 1e6  # Pa -> MPa

    crack_plane_angle = np.pi / 2.0  # assumed vertical

    T_eff = anisotropic_tensile_strength(
        base_T=base_T,
        anis_angle=anis_angle_x,
        crack_plane_angle=crack_plane_angle,
        spacing=spacing,
        x=x, y=y,
        min_factor=0.30
    )
    strength_factor = T_eff / max(float(base_T), 1e-12)

    sxx = applied_MPa * strength_factor
    syy = applied_MPa * strength_factor

    r_safe = np.where(r == 0.0, 1e-12, r)
    txy = shear_amp * applied_MPa * (x / r_safe) * strength_factor

    s1, s3, _ = principal_from_components(sxx, syy, txy)
    return s1, s3, sxx, syy, txy


def evaluate_failure_mode_mohr_coulomb_local(
    sigma_1, sigma_3,
    tensile_strength_arr,
    cohesion_arr,
    friction_angle_arr,
    *,
    rock_type=None,
    kC_max=None,
    kT_max=None,
    U_field=None,
    tensile_w=None,
    wp_weight=None,
    conf_w=None,
    sxx=None,
    syy=None,
    txy=None,
    x=None,
    y=None,
    anis_angle=None,
    spacing=0.0,
    mc_compression_only=True,
    verbose=False,
    return_counts=False,
    return_details=False,
):
    """
    Revised local tensile + Mohr-Coulomb classification using LOCAL ANISOTROPIC
    damage-softening, consistent with the calibrated model.

    Required physics inputs
    -----------------------
    We can either pass the local damage drivers explicitly:
      - U_field
      - tensile_w
      - wp_weight
      - conf_w

    or let the function build SURROGATE drivers from:
      - sxx, syy, txy
      - x, y
      - anis_angle
      - spacing

    Local softening model
    ---------------------
      T_eff = T0 * (1 - D_t)
      C_eff = C0 * (1 - D_s)

      D_t ~ kT_max * f(Uhat, tensile_w)
      D_s ~ kC_max * f(Uhat, wp_weight, shear_tendency, confinement)

    Adopted caps
    ------------
    Both lithologies use kC_max = 0.05 with kT_max = 0.0175, the value the
    sweep in ``kmax_sweep`` selects under its stated rule, the smallest cap
    whose mean mixed-mode fraction reaches within 2% of the peak attained.

    These were previously 0.850/0.297 for the augen gneiss and 0.650/0.227 for
    the psammitic schist, labelled "calibrated defaults". The sweep had already
    superseded them and the value was never propagated, so the module ran at a
    cap seventeen times the adopted one while the manuscript reported 0.05.
    Nothing published consumed these numbers, since this path writes no figure
    or table, but anyone running it got a model the paper does not describe.
    The sweep finds the same cap for both rocks, so they are no longer
    lithology-specific.

    Failure criteria
    ----------------
      tensile: s1 >= T_eff
      shear:   f_MC >= 0

      f_MC = (s1 - s3) + (s1 + s3) sin(phi_eff) - 2 C_eff cos(phi_eff)
    """
    s1 = np.asarray(sigma_1, dtype=float)
    s3 = np.asarray(sigma_3, dtype=float)
    Tens0 = np.asarray(tensile_strength_arr, dtype=float)
    Coh0 = np.asarray(cohesion_arr, dtype=float)
    phi0 = ensure_radians(friction_angle_arr)

    _ = s1 + s3 + Tens0 + Coh0 + phi0

    # Ensure principal ordering
    smax = np.maximum(s1, s3)
    smin = np.minimum(s1, s3)
    s1, s3 = smax, smin

    # Resolve calibrated softening caps
    if (kC_max is None) or (kT_max is None):
        if rock_type is None:
            raise ValueError(
                "Provide either rock_type or both kC_max and kT_max explicitly."
            )
        kc_def, kt_def = resolve_local_damage_params(rock_type)
        if kC_max is None:
            kC_max = kc_def
        if kT_max is None:
            kT_max = kt_def

    kC_max = float(kC_max)
    kT_max = float(kT_max)

    if not (0.0 <= kC_max <= 0.95):
        raise ValueError(f"kC_max={kC_max} must lie in [0, 0.95].")
    if not (0.0 <= kT_max <= 0.95):
        raise ValueError(f"kT_max={kT_max} must lie in [0, 0.95].")

    # Build or accept local damage drivers
    if U_field is None:
        if sxx is None or syy is None or txy is None:
            raise ValueError(
                "Provide either U_field explicitly or provide sxx, syy, txy "
                "to build a surrogate energy field."
            )
        U = surrogate_energy_proxy(sxx, syy, txy)
    else:
        U = np.asarray(U_field, dtype=float)

    if tensile_w is None:
        wt = surrogate_tensile_weight(s1, s3)
    else:
        wt = np.clip(np.asarray(tensile_w, dtype=float), 0.0, 1.0)

    if conf_w is None:
        wc = surrogate_conf_weight(s1, s3)
    else:
        wc = np.clip(np.asarray(conf_w, dtype=float), 0.0, 1.0)

    if wp_weight is None:
        if x is None or y is None or anis_angle is None:
            wwp = np.zeros_like(s1, dtype=float)
        else:
            wwp = surrogate_wp_weight(x, y, anis_angle, spacing)
    else:
        wwp = np.clip(np.asarray(wp_weight, dtype=float), 0.0, 1.0)

    valid = (
        np.isfinite(s1) &
        np.isfinite(s3) &
        np.isfinite(Tens0) &
        np.isfinite(Coh0) &
        np.isfinite(phi0) &
        np.isfinite(U) &
        np.isfinite(wt) &
        np.isfinite(wwp) &
        np.isfinite(wc)
    )

    mode = np.full(s1.shape, "no_failure", dtype=object)

    # Default detail arrays
    Uhat = np.full(s1.shape, np.nan, dtype=float)
    Dt = np.full(s1.shape, np.nan, dtype=float)
    Ds = np.full(s1.shape, np.nan, dtype=float)
    T_eff = np.full(s1.shape, np.nan, dtype=float)
    C_eff = np.full(s1.shape, np.nan, dtype=float)
    phi_eff = np.full(s1.shape, np.nan, dtype=float)
    f_mc = np.full(s1.shape, np.nan, dtype=float)
    sigma_n_crit = np.full(s1.shape, np.nan, dtype=float)

    if np.any(valid):
        Uhat[valid] = robust_unit_interval(U[valid], qlo=U_QLOW, qhi=U_QHIGH)

        ws = 1.0 - wt

        # Tensile damage
        Dt[valid] = kT_max * (Uhat[valid] ** P_U) * (0.25 + 0.75 * wt[valid])

        # Shear / cohesion damage
        Ds[valid] = (
            kC_max *
            (Uhat[valid] ** P_U) *
            (0.20 + 0.80 * (wwp[valid] ** P_WP)) *
            (0.20 + 0.80 * (ws[valid] ** P_S)) *
            (0.50 + 0.50 * wc[valid])
        )

        Dt[valid] = np.clip(Dt[valid], 0.0, 0.95)
        Ds[valid] = np.clip(Ds[valid], 0.0, 0.95)

        T_eff[valid] = np.maximum(0.05 * Tens0[valid], Tens0[valid] * (1.0 - Dt[valid]))
        C_eff[valid] = np.maximum(0.05 * Coh0[valid], Coh0[valid] * (1.0 - Ds[valid]))

        if abs(float(PHI_DROP_DEG)) > 0.0:
            phi_eff[valid] = phi0[valid] - np.deg2rad(float(PHI_DROP_DEG)) * Ds[valid]
            phi_eff[valid] = np.clip(phi_eff[valid], np.deg2rad(5.0), np.deg2rad(85.0))
        else:
            phi_eff[valid] = phi0[valid]

        tensile_fail = np.zeros(s1.shape, dtype=bool)
        tensile_fail[valid] = (s1[valid] >= T_eff[valid])

        f_mc[valid] = (
            (s1[valid] - s3[valid]) +
            (s1[valid] + s3[valid]) * np.sin(phi_eff[valid]) -
            2.0 * C_eff[valid] * np.cos(phi_eff[valid])
        )

        shear_fail = np.zeros(s1.shape, dtype=bool)
        shear_fail[valid] = (f_mc[valid] >= 0.0)

        if mc_compression_only:
            sigma_n_crit[valid] = (
                0.5 * (s1[valid] + s3[valid]) -
                0.5 * (s1[valid] - s3[valid]) * np.sin(phi_eff[valid])
            )
            shear_fail[valid] &= (sigma_n_crit[valid] <= 0.0)

        mode[valid & tensile_fail & ~shear_fail] = "tensile"
        mode[valid & shear_fail & ~tensile_fail] = "shear"
        mode[valid & tensile_fail & shear_fail] = "mixed"

    n_t = int(np.count_nonzero(mode == "tensile"))
    n_s = int(np.count_nonzero(mode == "shear"))
    n_m = int(np.count_nonzero(mode == "mixed"))
    n_nf = int(np.count_nonzero(mode == "no_failure"))
    n_fail = n_t + n_s + n_m
    mixed_fraction_failed = (n_m / float(n_fail)) if n_fail > 0 else np.nan

    if verbose:
        print(
            f"kC_max={kC_max:.3f} | kT_max={kT_max:.3f} | "
            f"tensile={n_t} shear={n_s} mixed={n_m} "
            f"no_failure={n_nf} failed_total={n_fail} "
            f"mixed/failed={mixed_fraction_failed:.4f}"
        )

    info_basic = {
        "kC_max_used": kC_max,
        "kT_max_used": kT_max,
        "tensile": n_t,
        "shear": n_s,
        "mixed": n_m,
        "no_failure": n_nf,
        "failed_total": n_fail,
        "mixed_fraction_failed": mixed_fraction_failed,
    }

    info_detail = {
        "Uhat": Uhat,
        "Dt": Dt,
        "Ds": Ds,
        "T_eff": T_eff,
        "C_eff": C_eff,
        "phi_eff": phi_eff,
        "f_mc": f_mc,
        "sigma_n_crit": sigma_n_crit,
        "valid_mask": valid,
        "tensile_w": wt,
        "wp_weight": wwp,
        "conf_w": wc,
    }

    if return_counts and return_details:
        out = {}
        out.update(info_basic)
        out.update(info_detail)
        return mode, out
    elif return_counts:
        return mode, info_basic
    elif return_details:
        out = {
            "kC_max_used": kC_max,
            "kT_max_used": kT_max,
        }
        out.update(info_detail)
        return mode, out

    return mode



def main():
    """Run this section for both lithologies in one pass."""
    global Coh, Coh_fix, Coh_ref, D, KCMAX_AUGEN_GNEISS, \
        KCMAX_PSAMMATIC_SCHIST, KTMAX_AUGEN_GNEISS, KTMAX_PSAMMATIC_SCHIST, \
        MC_COMPRESSION_ONLY, P, PHI_DROP_DEG, POINTS_PER_ROW, P_S, P_U, \
        P_WP, R, Tens, Tens_fix, Tens_ref, U_QHIGH, U_QLOW, _, \
        anis_angle_x, anis_rad_load, base_T, base_draws, cohesion, df, \
        diff_ct, diff_pct, het, het_fixed, heterogeneity_levels, kC_used, \
        kT_used, modes, modes_ref, modes_ref_het, n, phi, phi_deg, phi_fix, \
        phi_ref, r, rng, rock_type, row, s1, s1_base, s1_ref, s3, s3_base, \
        s3_ref, sample_id, sample_ids, sp, spacing_fixed, spacing_values, \
        sxx, sxx_base, sxx_ref, syy, syy_base, syy_ref, t, txy, txy_base, \
        txy_ref, x, y
    spacing_values = [0.000, 0.0010, 0.0020, 0.0030, 0.0040]
    heterogeneity_levels = [0.00, 0.10, 0.20, 0.30, 0.40]
    POINTS_PER_ROW = 50
    sample_ids = range(1, 15)
    MC_COMPRESSION_ONLY = True
    P_U = 1.25
    P_WP = 1.20
    P_S = 1.10
    PHI_DROP_DEG = 0.0
    U_QLOW = 20.0
    U_QHIGH = 95.0
    df = load_specimen_table()
    df.columns = [str(c).strip() for c in df.columns]
    for sample_id in sample_ids:
        print(f"\n\n==== SAMPLE {sample_id} ====")
        row = df.loc[sample_id]

        rock_type = str(row["Rock_type"])
        kC_used, kT_used = resolve_local_damage_params(rock_type)

        # Geometry / loads
        D = float(row["Diameter_mm"]) * 1e-3   # m
        t = float(row["Thickness_mm"]) * 1e-3  # m
        P = float(row["Load_(KN)"]) * 1e3      # N

        # Strength inputs
        base_T = float(row["Tensile_strength_Mpa"])   # MPa
        cohesion = float(row["Cohesion"])             # MPa
        phi_deg = float(row["Friction_Angle"])        # degrees

        # Anisotropy angle (load-axis -> x-axis)
        anis_rad_load = float(row["Radians"])
        anis_angle_x = wrap_pi((np.pi / 2.0) - anis_rad_load)

        print(f"Rock type : {rock_type}")
        print(f"kC_max used: {kC_used:.3f}")
        print(f"kT_max used: {kT_used:.3f}")

        # Grid
        x, y, r, R = create_grid_mesh(D, points_per_row=POINTS_PER_ROW)
        n = x.size
        rng = np.random.default_rng(10_000 + sample_id)

        # ======================================================
        # A) SPACING EFFECT (heterogeneity fixed)
        # ======================================================
        het_fixed = heterogeneity_levels[0]
        Tens_fix, phi_fix, Coh_fix, _ = generate_random_props(
            n, base_T, phi_deg, cohesion, het_fixed, rng=rng
        )

        s1_ref, s3_ref, sxx_ref, syy_ref, txy_ref = calculate_advanced_stress(
            x, y, r, D, P, t, base_T, anis_angle_x, spacing=0.0
        )

        modes_ref = evaluate_failure_mode_mohr_coulomb_local(
            s1_ref, s3_ref,
            Tens_fix, Coh_fix, phi_fix,
            rock_type=rock_type,
            sxx=sxx_ref, syy=syy_ref, txy=txy_ref,
            x=x, y=y, anis_angle=anis_angle_x, spacing=0.0,
            mc_compression_only=MC_COMPRESSION_ONLY,
            verbose=False
        )

        print("\n--- SPACING EFFECT (heterogeneity fixed) ---")
        for sp in spacing_values:
            s1, s3, sxx, syy, txy = calculate_advanced_stress(
                x, y, r, D, P, t, base_T, anis_angle_x, spacing=sp
            )

            modes = evaluate_failure_mode_mohr_coulomb_local(
                s1, s3,
                Tens_fix, Coh_fix, phi_fix,
                rock_type=rock_type,
                sxx=sxx, syy=syy, txy=txy,
                x=x, y=y, anis_angle=anis_angle_x, spacing=sp,
                mc_compression_only=MC_COMPRESSION_ONLY,
                verbose=False
            )

            diff_ct = int(np.count_nonzero(modes != modes_ref))
            diff_pct = 100.0 * diff_ct / float(n)
            print(f"Spacing = {sp:6.4f} m -> ΔFailure: {diff_ct:6d}  ({diff_pct:6.2f} %)")

        # ======================================================
        # B) HETEROGENEITY EFFECT (spacing fixed)
        # ======================================================
        spacing_fixed = spacing_values[0]  # usually 0.0
        s1_base, s3_base, sxx_base, syy_base, txy_base = calculate_advanced_stress(
            x, y, r, D, P, t, base_T, anis_angle_x, spacing=spacing_fixed
        )

        # Freeze random draws so heterogeneity is a controlled perturbation
        _, _, _, base_draws = generate_random_props(
            n, base_T, phi_deg, cohesion, hetero_scale=1.0, rng=rng
        )

        Tens_ref, phi_ref, Coh_ref, _ = generate_random_props(
            n, base_T, phi_deg, cohesion, hetero_scale=0.0, base_draws=base_draws
        )

        modes_ref_het = evaluate_failure_mode_mohr_coulomb_local(
            s1_base, s3_base,
            Tens_ref, Coh_ref, phi_ref,
            rock_type=rock_type,
            sxx=sxx_base, syy=syy_base, txy=txy_base,
            x=x, y=y, anis_angle=anis_angle_x, spacing=spacing_fixed,
            mc_compression_only=MC_COMPRESSION_ONLY,
            verbose=False
        )

        print("\n--- MATERIAL HETEROGENEITY EFFECT (spacing fixed) ---")
        for het in heterogeneity_levels:
            Tens, phi, Coh, _ = generate_random_props(
                n, base_T, phi_deg, cohesion, hetero_scale=het, base_draws=base_draws
            )

            modes = evaluate_failure_mode_mohr_coulomb_local(
                s1_base, s3_base,
                Tens, Coh, phi,
                rock_type=rock_type,
                sxx=sxx_base, syy=syy_base, txy=txy_base,
                x=x, y=y, anis_angle=anis_angle_x, spacing=spacing_fixed,
                mc_compression_only=MC_COMPRESSION_ONLY,
                verbose=False
            )

            diff_ct = int(np.count_nonzero(modes != modes_ref_het))
            diff_pct = 100.0 * diff_ct / float(n)
            print(f"Heterogeneity = {het:4.2f} -> ΔFailure: {diff_ct:6d}  ({diff_pct:6.2f} %)")
