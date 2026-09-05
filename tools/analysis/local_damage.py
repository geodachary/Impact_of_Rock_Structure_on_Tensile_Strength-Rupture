"""Local cohesion-softening damage model.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 66 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import numpy as np
# <<PRELUDE_IMPORTS>>


# --- implementation ---------------------------------------------------


def ensure_radians(phi):
    phi = np.asarray(phi, dtype=float)
    vals = phi[np.isfinite(phi)]
    if vals.size and np.nanmax(np.abs(vals)) > (np.pi + 1e-6):
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


def resolve_local_damage_params(
    rock_type,
    *,
    kC_gneiss=0.05,
    kT_gneiss=0.0175,
    kC_schist=0.05,
    kT_schist=0.0175,
):
    """
    Resolve calibrated local softening caps by lithology.
    Accepts both 'Psammatic' and 'Psammitic' spellings.
    """
    rock = str(rock_type).strip().lower()
    rock = rock.replace("_", " ").replace("-", " ")
    rock = " ".join(rock.split())

    if "augen" in rock and "gneiss" in rock:
        return float(kC_gneiss), float(kT_gneiss)

    if "schist" in rock and (
        "psammatic" in rock or
        "psammitic" in rock or
        "psammat" in rock or
        "psammit" in rock
    ):
        return float(kC_schist), float(kT_schist)

    raise ValueError(
        f"Unrecognized rock_type={rock_type!r}. "
        "Expected Augen gneiss or Psammatic or Psammitic schist."
    )


def evaluate_failure_mode_local_damage(
    sigma_1, sigma_3,
    tensile_strength_arr,
    cohesion_arr,
    friction_angle_arr,
    U_field,
    tensile_w,
    wp_weight,
    conf_w,
    *,
    rock_type=None,
    kC_max=None,
    kT_max=None,
    kC_gneiss=0.05,
    kT_gneiss=0.0175,
    kC_schist=0.05,
    kT_schist=0.0175,
    pU=1.25,
    pWP=1.20,
    pS=1.10,
    phi_drop_deg=0.0,
    u_qlo=20.0,
    u_qhi=95.0,
    mc_compression_only=True,
    verbose=False,
    return_counts=False,
    return_details=False,
):
    """
    Local anisotropic tensile + Mohr-Coulomb failure classification.

    Physics in this revision
    ------------------------
    Uses local energy-driven softening:

        T_eff(x,y) = T0 * [1 - D_t(x,y)]
        C_eff(x,y) = C0 * [1 - D_s(x,y)]

    where:
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

    Failure checks:
      - tensile: s1 >= T_eff
      - shear:   f_MC >= 0

        f_MC = (s1 - s3) + (s1 + s3) sin(phi_eff) - 2 C_eff cos(phi_eff)

    Optional compression-only gating:
      sigma_n_crit = 0.5*(s1+s3) - 0.5*(s1-s3)*sin(phi_eff)
      require sigma_n_crit <= 0 for frictional shear activation

    Notes
    -----
    - tension-positive convention: sigma > 0 in tension, sigma < 0 in compression
    - principal stresses are reordered internally so s1 >= s3
    - phi may be supplied in degrees or radians
    - U_field is assumed to be compatible with the exported U_MPa field
    """

    # ------------------------------------------------------
    # Arrays
    # ------------------------------------------------------
    s1 = np.asarray(sigma_1, dtype=float)
    s3 = np.asarray(sigma_3, dtype=float)
    Tens0 = np.asarray(tensile_strength_arr, dtype=float)
    Coh0 = np.asarray(cohesion_arr, dtype=float)
    phi0 = ensure_radians(friction_angle_arr)

    U = np.asarray(U_field, dtype=float)
    wt = np.clip(np.asarray(tensile_w, dtype=float), 0.0, 1.0)
    wwp = np.clip(np.asarray(wp_weight, dtype=float), 0.0, 1.0)
    wc = np.clip(np.asarray(conf_w, dtype=float), 0.0, 1.0)

    # Broadcast sanity
    _ = s1 + s3 + Tens0 + Coh0 + phi0 + U + wt + wwp + wc

    # ------------------------------------------------------
    # Ensure principal ordering
    # ------------------------------------------------------
    smax = np.maximum(s1, s3)
    smin = np.minimum(s1, s3)
    s1, s3 = smax, smin

    # ------------------------------------------------------
    # Resolve calibrated local softening caps
    # ------------------------------------------------------
    if (kC_max is None) or (kT_max is None):
        if rock_type is None:
            raise ValueError(
                "Provide either both kC_max and kT_max explicitly, "
                "or provide rock_type for calibrated defaults."
            )
        kc_def, kt_def = resolve_local_damage_params(
            rock_type,
            kC_gneiss=kC_gneiss,
            kT_gneiss=kT_gneiss,
            kC_schist=kC_schist,
            kT_schist=kT_schist,
        )
        if kC_max is None:
            kC_max = kc_def
        if kT_max is None:
            kT_max = kt_def

    kC_max = float(kC_max)
    kT_max = float(kT_max)

    if not (0.0 <= kC_max <= 0.95):
        raise ValueError(f"kC_max={kC_max} must be within [0, 0.95].")
    if not (0.0 <= kT_max <= 0.95):
        raise ValueError(f"kT_max={kT_max} must be within [0, 0.95].")

    # ------------------------------------------------------
    # Valid mask
    # ------------------------------------------------------
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

    # Prepare default detail arrays
    Uhat = np.full(s1.shape, np.nan, dtype=float)
    Dt = np.full(s1.shape, np.nan, dtype=float)
    Ds = np.full(s1.shape, np.nan, dtype=float)
    T_eff = np.full(s1.shape, np.nan, dtype=float)
    C_eff = np.full(s1.shape, np.nan, dtype=float)
    phi_eff = np.full(s1.shape, np.nan, dtype=float)
    f_mc = np.full(s1.shape, np.nan, dtype=float)
    sigma_n_crit = np.full(s1.shape, np.nan, dtype=float)

    if not np.any(valid):
        info_basic = {
            "kC_max_used": kC_max,
            "kT_max_used": kT_max,
            "tensile": 0,
            "shear": 0,
            "mixed": 0,
            "no_failure": int(mode.size),
            "failed_total": 0,
            "mixed_fraction_failed": np.nan,
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

    # ------------------------------------------------------
    # Local damage drivers
    # ------------------------------------------------------
    Uhat[valid] = robust_unit_interval(U[valid], qlo=u_qlo, qhi=u_qhi)

    ws = 1.0 - wt

    # Tensile-damage channel
    Dt[valid] = kT_max * (Uhat[valid] ** pU) * (0.25 + 0.75 * wt[valid])

    # Shear/cohesion-damage channel
    Ds[valid] = (
        kC_max *
        (Uhat[valid] ** pU) *
        (0.20 + 0.80 * (wwp[valid] ** pWP)) *
        (0.20 + 0.80 * (ws[valid] ** pS)) *
        (0.50 + 0.50 * wc[valid])
    )

    Dt[valid] = np.clip(Dt[valid], 0.0, 0.95)
    Ds[valid] = np.clip(Ds[valid], 0.0, 0.95)

    # ------------------------------------------------------
    # Local softened strengths
    # ------------------------------------------------------
    T_eff[valid] = np.maximum(0.05 * Tens0[valid], Tens0[valid] * (1.0 - Dt[valid]))
    C_eff[valid] = np.maximum(0.05 * Coh0[valid], Coh0[valid] * (1.0 - Ds[valid]))

    if abs(float(phi_drop_deg)) > 0.0:
        phi_eff[valid] = phi0[valid] - np.deg2rad(float(phi_drop_deg)) * Ds[valid]
        phi_eff[valid] = np.clip(phi_eff[valid], np.deg2rad(5.0), np.deg2rad(85.0))
    else:
        phi_eff[valid] = phi0[valid]

    # ------------------------------------------------------
    # Failure checks
    # ------------------------------------------------------
    tensile_fail = np.zeros(s1.shape, dtype=bool)
    tensile_fail[valid] = (s1[valid] >= T_eff[valid])

    f_mc[valid] = (
        (s1[valid] - s3[valid]) +
        (s1[valid] + s3[valid]) * np.sin(phi_eff[valid]) -
        2.0 * C_eff[valid] * np.cos(phi_eff[valid])
    )

    shear_fail = np.zeros(s1.shape, dtype=bool)
    shear_fail[valid] = (f_mc[valid] >= 0.0)

    # ------------------------------------------------------
    # Optional compression-only gating
    # ------------------------------------------------------
    if mc_compression_only:
        sigma_n_crit[valid] = (
            0.5 * (s1[valid] + s3[valid]) -
            0.5 * (s1[valid] - s3[valid]) * np.sin(phi_eff[valid])
        )
        shear_fail[valid] &= (sigma_n_crit[valid] <= 0.0)

    # ------------------------------------------------------
    # Classify
    # ------------------------------------------------------
    mode[valid & tensile_fail & ~shear_fail] = "tensile"
    mode[valid & shear_fail & ~tensile_fail] = "shear"
    mode[valid & tensile_fail & shear_fail] = "mixed"

    # ------------------------------------------------------
    # Counts / diagnostics
    # ------------------------------------------------------
    n_tens = int(np.count_nonzero(mode == "tensile"))
    n_shear = int(np.count_nonzero(mode == "shear"))
    n_mixed = int(np.count_nonzero(mode == "mixed"))
    n_no = int(np.count_nonzero(mode == "no_failure"))
    n_fail = n_tens + n_shear + n_mixed
    mixed_fraction_failed = (n_mixed / float(n_fail)) if n_fail > 0 else np.nan

    if verbose:
        print(
            f"kC_max={kC_max:.3f} | kT_max={kT_max:.3f} | "
            f"tensile={n_tens} | shear={n_shear} | mixed={n_mixed} | "
            f"no_failure={n_no} | failed_total={n_fail} | "
            f"mixed/failed={mixed_fraction_failed:.4f}"
        )

    info_basic = {
        "kC_max_used": kC_max,
        "kT_max_used": kT_max,
        "tensile": n_tens,
        "shear": n_shear,
        "mixed": n_mixed,
        "no_failure": n_no,
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
