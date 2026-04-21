#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py  (FAST + ALWAYS SHOWS FIGURE EARLY)

Key improvements:
- PASS 1: compute + plot + save FAILURE maps first (fast) and show figure immediately.
- PASS 2: optional crack prediction; by default runs DDM only for FIRST sample unless you request more.
- DDM "fast preset" defaults reduce max_steps and increase ds to prevent hour-long runs.
- Optional --crack_psi0_deg (if crack_path_single_mirror_ddm accepts psi0) to skip costly initial search.

Examples:
  # Fast, figure first, no cracks:
  python run.py --no_crack --mpl_backend MacOSX

  # Fast crack on only sample 8 using PHYSICS model:
  python run.py --crack_model physics --crack_sample_ids 8 --mpl_backend MacOSX

  # DDM but only one sample and fast preset:
  python run.py --crack_model ddm --crack_sample_ids 8 --mpl_backend MacOSX

  # DDM for several samples:
  python run.py --crack_model ddm --crack_sample_ids 8,9,10 --mpl_backend MacOSX
"""

import os
import sys
import time
import argparse
import inspect
import numpy as np
import pandas as pd

IN_IPY = any(m in sys.modules for m in ("ipykernel", "IPython"))

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def parse_ids(spec, valid_index):
    """Parse '8-15' or '8,9,10' into a list of ints that exist in valid_index."""
    wanted = set()
    if spec and isinstance(spec, str):
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    a = int(a); b = int(b)
                    wanted.update(range(min(a, b), max(a, b) + 1))
                except Exception:
                    pass
            else:
                try:
                    wanted.add(int(tok))
                except Exception:
                    pass
    return [i for i in valid_index if i in wanted]

def log(msg, quiet=False):
    if not quiet:
        print(msg, flush=True)

def call_filtered(func, **kwargs):
    """Call func with only kwargs that exist in its signature."""
    sig = inspect.signature(func)
    fkw = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**fkw)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Brazilian disk (Lekhnitskii Airy) + failure + crack path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # IO
    p.add_argument("--meta_csv", default="tensile_samples_data.csv")
    p.add_argument("--out_dir", default="stress_tensors_orthotropic_airy_results")
    p.add_argument("--csv_out", default="specimen_stats.csv")
    p.add_argument("--save_basename", default="failure_modes_and_crack_results")

    # plotting
    p.add_argument("--mpl_backend", default=None,
                   help="Terminal runs: MacOSX / TkAgg / QtAgg. Ignored in notebooks.")
    p.add_argument("--no_show", action="store_true", help="Do not call plt.show()")
    p.add_argument("--no_save", action="store_true", help="Do not save figures")
    p.add_argument("--save_png", action="store_true", default=True)
    p.add_argument("--save_pdf", action="store_true", default=True)

    # selection
    p.add_argument("--sample_ids", type=str, default="8-15",
                   help="Samples for FAILURE map pass (fast).")
    p.add_argument("--crack_sample_ids", type=str, default=None,
                   help="Samples for CRACK pass (slow). Default: first sample only.")
    p.add_argument("--angle_map", type=str, default="pi2_minus",
                   choices=["direct", "neg", "pi2_minus", "pi2_plus"])
    p.add_argument("--points_per_row", type=int, default=121)

    # orthotropic elastic props
    p.add_argument("--E1_GPa", type=float, default=50.0)
    p.add_argument("--E2_GPa", type=float, default=30.0)
    p.add_argument("--nu12", type=float, default=0.25)
    p.add_argument("--G12_GPa", type=float, default=12.0)

    # platen traction parameters
    p.add_argument("--platen_half_angle_deg", type=float, default=10.0)
    p.add_argument("--platen_smooth_deg", type=float, default=4.0)
    p.add_argument("--platen_mu", type=float, default=0.0)

    # Airy fit controls
    p.add_argument("--airy_auto_tune", action="store_true", default=True)
    p.add_argument("--airy_M", type=int, default=24)
    p.add_argument("--airy_lam", type=float, default=1e-10)
    p.add_argument("--airy_Nbd_base", type=int, default=480)
    p.add_argument("--airy_Nbd_arc_each", type=int, default=1200)

    # failure params
    p.add_argument("--fail_mixed_band", type=float, default=0.45)
    p.add_argument("--fail_mc_nplanes", type=int, default=361)
    p.add_argument("--fail_margin", type=float, default=0.0)
    p.add_argument("--fail_mode_basis", type=str, default="first", choices=["first", "now"])
    p.add_argument("--fail_util_min", type=float, default=0.98)

    p.add_argument("--mc_compression_only", action="store_true", default=True)
    p.add_argument("--mc_sigma_comp_min_MPa", type=float, default=None)
    p.add_argument("--mc_sigma_comp_min_frac_p0", type=float, default=0.20)

    p.add_argument("--tensile_factor", type=float, default=1.0)
    p.add_argument("--coh_factor", type=float, default=1.0)
    p.add_argument("--phi_offset_deg", type=float, default=0.0)

    p.add_argument("--strength_model", type=str, default="weak_plane",
                   choices=["matrix", "weak_plane"])
    p.add_argument("--weak_plane_T_ratio", type=float, default=0.35)
    p.add_argument("--weak_plane_C_ratio", type=float, default=0.60)
    p.add_argument("--weak_plane_phi_deg", type=float, default=None)

    p.add_argument("--plot_rmax_frac", type=float, default=0.985)
    p.add_argument("--stats_rmax_frac", type=float, default=0.985)

    # crack controls
    p.add_argument("--no_crack", action="store_true", help="Disable crack prediction pass.")
    p.add_argument("--crack_model", type=str, default="ddm", choices=["physics", "ddm"])
    p.add_argument("--crack_psi0_deg", type=float, default=None,
                   help="If supported by your DDM crack function, sets initial crack angle to skip slow search.")
    p.add_argument("--ddm_fast", action="store_true", default=True,
                   help="Apply a fast preset for DDM (recommended).")
    p.add_argument("--crack_ds_frac", type=float, default=0.010, help="(fast default) step size fraction of R")
    p.add_argument("--crack_a0_frac", type=float, default=0.015, help="(fast default) initial crack length fraction of R")
    p.add_argument("--crack_max_steps", type=int, default=700, help="(fast default) reduce from 2500")
    p.add_argument("--crack_max_turn_deg", type=float, default=12.0)
    p.add_argument("--crack_relax", type=float, default=0.80)
    p.add_argument("--crack_kmin_frac", type=float, default=0.05, help="(fast default) earlier stop")

    p.add_argument("--load_factor", type=float, default=1.0)
    p.add_argument("--quiet", action="store_true")

    args, _ = p.parse_known_args([] if IN_IPY else None)

    # Backend BEFORE importing pyplot
    if (not IN_IPY) and args.mpl_backend:
        import matplotlib
        matplotlib.use(args.mpl_backend, force=True)

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Style
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.linewidth"] = 1.2

    minorTick = {'which': 'minor', 'direction': 'out', 'length': 3, 'width': 1}
    majorTick = {'which': 'major', 'direction': 'out', 'length': 5, 'width': 1.5}
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}

    def close_box(ax, lw=1.2):
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(lw)

    # Project imports
    from bound_all_helpers import (
        points_in_disk,
        map_angle_to_alpha,
        reduce_angle_0_90,
        rot_to_material,
        stress_material_to_global,
        principal_from_components,
        eval_stress_field_material,
        fit_orthotropic_airy_disk,
        fit_orthotropic_airy_disk_auto,
        failure_mode_map,
        disc_failure_point_stats,
        specimen_mode_from_crack_path,
        crack_path_single_mirror_physics,
        crack_path_single_mirror_ddm,
    )

    # Load meta
    dfmeta = pd.read_csv(args.meta_csv, index_col=0)
    valid_ids = list(dfmeta.index)

    samples = parse_ids(args.sample_ids, valid_ids)
    if not samples:
        raise RuntimeError("No matching sample IDs for --sample_ids")

    # crack ids (default: only first sample)
    if args.crack_sample_ids is None:
        crack_ids = [samples[0]]
    else:
        crack_ids = parse_ids(args.crack_sample_ids, samples)

    log(f"Loaded meta_csv: {args.meta_csv} | nrows={len(dfmeta)}", args.quiet)
    log(f"Samples (failure maps): {samples}", args.quiet)
    log(f"Samples (cracks): {crack_ids}" if (not args.no_crack) else "Cracks disabled.", args.quiet)
    log(f"Output dir: {args.out_dir}", args.quiet)

    os.makedirs(args.out_dir, exist_ok=True)
    crack_dir_root = os.path.join(args.out_dir, "crack_paths")
    os.makedirs(crack_dir_root, exist_ok=True)

    required_cols = [
        'Rock_type', 'Angle', 'Diameter_mm', 'Thickness_mm', 'Load_(KN)',
        'Tensile_strength_Mpa', 'Cohesion', 'Friction_Angle'
    ]
    missing_cols = [c for c in required_cols if c not in dfmeta.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in CSV: {missing_cols}")

    # material (MPa)
    E1 = float(args.E1_GPa) * 1e3
    E2 = float(args.E2_GPa) * 1e3
    G12 = float(args.G12_GPa) * 1e3
    nu12 = float(args.nu12)

    raw = {}
    summary_rows = []

    # -----------------------------------------------------------------
    # PASS 1: compute + failure map (fast)
    # -----------------------------------------------------------------
    for sid in samples:
        t0 = time.perf_counter()
        rock, ang_deg, Dmm, tmm, PkN, Tm_in, coh_in, phi_deg_in = dfmeta.loc[sid, required_cols]
        alpha_const = map_angle_to_alpha(float(ang_deg), angle_map=str(args.angle_map))

        D = float(Dmm) * 1e-3
        t = float(tmm) * 1e-3
        P = float(PkN) * 1e3 * float(args.load_factor)
        R = D / 2.0

        log(f"\n--- Sample {sid} ---", args.quiet)
        log(f"Rock={rock} | theta={float(ang_deg):.2f} deg | alpha={alpha_const:.4f} rad | R={R:.4e} m", args.quiet)

        xg, yg, X, Y, M = points_in_disk(D, n=int(args.points_per_row))
        r_pts = np.hypot(xg, yg)

        use_auto = bool(args.airy_auto_tune) and (fit_orthotropic_airy_disk_auto is not None)
        log(f"Fitting Airy solution... (auto={use_auto})", args.quiet)

        tfit0 = time.perf_counter()
        if use_auto:
            fit = fit_orthotropic_airy_disk_auto(
                E1, E2, nu12, G12,
                R=R, t=t, P=P,
                alpha=alpha_const,
                beta_deg=args.platen_half_angle_deg,
                smooth_deg=args.platen_smooth_deg,
                mu=args.platen_mu,
                Nbd=args.airy_Nbd_base,
                Nbd_arc_each=args.airy_Nbd_arc_each,
            )
        else:
            fit = fit_orthotropic_airy_disk(
                E1, E2, nu12, G12,
                R=R, t=t, P=P,
                alpha=alpha_const,
                M=int(args.airy_M),
                Nbd=int(args.airy_Nbd_base),
                beta_deg=args.platen_half_angle_deg,
                smooth_deg=args.platen_smooth_deg,
                mu=args.platen_mu,
                lam=float(args.airy_lam),
                Nbd_arc_each=int(args.airy_Nbd_arc_each),
            )
        tfit1 = time.perf_counter()
        log(f"Airy fit: {(tfit1-tfit0):.2f}s | p0={fit['p0']:.2f} MPa | RMS={fit['res_rms_all']:.3e} MPa", args.quiet)

        if args.mc_sigma_comp_min_MPa is not None:
            mc_scmin = float(args.mc_sigma_comp_min_MPa)
        else:
            mc_scmin = float(max(args.mc_sigma_comp_min_frac_p0, 0.0)) * float(fit["p0"])

        # stresses
        xm, ym = rot_to_material(xg, yg, alpha_const)
        sxx_m, syy_m, txy_m = eval_stress_field_material(xm, ym, R, fit["p1"], fit["p2"], fit["a1"], fit["a2"])
        sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, alpha_const)
        s1, _, _ = principal_from_components(sxx, syy, txy)

        # strengths
        Tm = float(Tm_in) * float(args.tensile_factor)
        Coh0 = float(coh_in) * float(args.coh_factor)
        phi0 = np.deg2rad(float(phi_deg_in) + float(args.phi_offset_deg))

        Phi = np.full_like(s1, phi0, dtype=float)
        Coh = np.full_like(s1, Coh0, dtype=float)
        Teff = np.full_like(s1, Tm, dtype=float)

        # failure modes now (so we can plot fast)
        modes, Rt_eff, Rs_eff, util = failure_mode_map(
            sxx, syy, txy, s1,
            Tm=Teff, Coh=Coh, Phi=Phi,
            alpha_const=alpha_const,
            basis=args.fail_mode_basis,
            util_min=args.fail_util_min,
            mixed_band=args.fail_mixed_band,
            margin=args.fail_margin,
            n_theta_mc=args.fail_mc_nplanes,
            mc_compression_only=args.mc_compression_only,
            mc_sigma_comp_min=mc_scmin,
            strength_model=args.strength_model,
            weak_T_ratio=args.weak_plane_T_ratio,
            weak_C_ratio=args.weak_plane_C_ratio,
            weak_phi=None,
        )
        disc_total, disc_counts, disc_pct = disc_failure_point_stats(
            modes=modes, r_pts=r_pts, R=R, rmax_frac=args.stats_rmax_frac
        )

        # grids for crack physics (if needed)
        Teff_img = np.full_like(X, np.nan, dtype=float)
        Coh_img  = np.full_like(X, np.nan, dtype=float)
        Phi_img  = np.full_like(X, np.nan, dtype=float)
        Teff_img[M] = float(Tm)
        Coh_img[M]  = float(Coh0)
        Phi_img[M]  = float(phi0)

        weak_phi_scalar = None
        if args.weak_plane_phi_deg is not None:
            weak_phi_scalar = float(np.deg2rad(args.weak_plane_phi_deg))

        raw[sid] = dict(
            rock=str(rock), ang=float(ang_deg), alpha=float(alpha_const),
            R=R, t=t, P=P,
            x=xg, y=yg, r=r_pts, X=X, Y=Y, M=M,
            sxx=sxx, syy=syy, txy=txy, s1=s1,
            Coh=Coh, Phi=Phi, Teff=Teff,
            Teff_img=Teff_img, Coh_img=Coh_img, Phi_img=Phi_img,
            fit=fit, mc_scmin=mc_scmin, weak_phi_scalar=weak_phi_scalar,
            modes=modes, disc_total=disc_total, disc_counts=disc_counts, disc_pct=disc_pct
        )

        summary_rows.append(dict(
            Sample=int(sid),
            Rock=str(rock),
            Angle_deg=float(ang_deg),
            Angle_0_90_deg=float(reduce_angle_0_90(float(ang_deg))),
            angle_map=str(args.angle_map),
            alpha_rad=float(alpha_const),
            StrengthModel=str(args.strength_model),
            mc_sigma_comp_min_MPa=float(mc_scmin),
            p0_MPa=float(fit["p0"]),
            AiryTractionRMS_MPa=float(fit["res_rms_all"]),
            DiscFailPts_Total=int(disc_total),
            DiscFailPts_Tensile=int(disc_counts["tensile"]),
            DiscFailPts_Mixed=int(disc_counts["mixed"]),
            DiscFailPts_Shear=int(disc_counts["shear"]),
            DiscFailPct_Tensile=float(disc_pct["tensile"]),
            DiscFailPct_Mixed=float(disc_pct["mixed"]),
            DiscFailPct_Shear=float(disc_pct["shear"]),
            CrackModel=str(args.crack_model),
            SpecimenMode="(not_run)" if args.no_crack else "(pending)",
            CrackPath_Length_m=np.nan,
            Kmax=np.nan,
        ))

        t1 = time.perf_counter()
        log(f"Sample {sid} done in {(t1-t0):.2f}s", args.quiet)

    # -----------------------------------------------------------------
    # Plot FAILURE maps NOW (fast) + SHOW immediately
    # -----------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n = len(samples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    figF, axsF = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows))
    if nrows == 1:
        axsF = np.array([axsF])

    mode_map = {'no_failure': 0, 'tensile': 1, 'shear': 2, 'mixed': 3}
    cmap = mcolors.ListedColormap(['purple', 'gold', 'darkgreen', 'red'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], 4)

    for k, sid in enumerate(samples):
        d = raw[sid]
        row, col = divmod(k, ncols)
        ax = axsF[row, col]

        modes = d["modes"]
        disc_total = d["disc_total"]
        disc_pct = d["disc_pct"]

        mode_numeric = np.array([mode_map[m] for m in modes])
        keep = d["r"] <= (args.plot_rmax_frac * d["R"])

        ax.scatter(
            d["x"][keep], d["y"][keep],
            c=mode_numeric[keep], s=18, cmap=cmap, norm=norm,
            edgecolors='k', linewidths=0.15, zorder=1
        )
        ax.add_artist(plt.Circle((0, 0), d["R"], fill=False, color='k', lw=1.2, zorder=2))

        ax.set_aspect('equal', 'box')
        ax.set_xlim(-d["R"] * 1.05, d["R"] * 1.05)
        ax.set_ylim(-d["R"] * 1.05, d["R"] * 1.05)

        ax.set_title(
            f"{d['rock']} (θ={d['ang']:.0f}°) AiryRMS={d['fit']['res_rms_all']:.2e} MPa\n"
            f"DISC pts={disc_total} | DISC% T={disc_pct['tensile']:.0f} M={disc_pct['mixed']:.0f} S={disc_pct['shear']:.0f}"
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.tick_params(axis='x', **majorTick); ax.tick_params(axis='x', **minorTick)
        ax.tick_params(axis='y', **majorTick); ax.tick_params(axis='y', **minorTick)
        close_box(ax)

    for k in range(n, nrows * ncols):
        row, col = divmod(k, ncols)
        axsF[row, col].axis("off")

    handlesF = [
        plt.Line2D([0], [0], marker='o', color='w', label='no_failure', markerfacecolor='purple', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='tensile', markerfacecolor='gold', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='shear', markerfacecolor='darkgreen', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='mixed', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], color='k', lw=2, label='predicted crack path'),
    ]
    figF.legend(handles=handlesF, loc="lower center", ncol=3, frameon=True, edgecolor="black")
    figF.tight_layout(rect=[0, 0.06, 1, 1])

    # Save early (failure-only)
    if not args.no_save:
        base = os.path.join(args.out_dir, args.save_basename + "_FAILURE_ONLY")
        if args.save_pdf:
            figF.savefig(base + ".pdf", dpi=300, bbox_inches="tight", transparent=True)
        if args.save_png:
            figF.savefig(base + ".png", dpi=300, bbox_inches="tight", transparent=True)
        log(f"\n✓ Saved failure-only figure: {base}.(pdf/png)", args.quiet)

    # Show figure immediately (so you SEE something even while cracks run)
    if not args.no_show:
        try:
            plt.show(block=False)
            plt.pause(0.2)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # PASS 2: crack prediction (slow) — only for crack_ids
    # -----------------------------------------------------------------
    if (not args.no_crack) and crack_ids:
        log("\nCrack pass starting...", args.quiet)
        log(f"Crack model: {args.crack_model}", args.quiet)

        # Apply fast preset (already defaulted via args, but keep explicit)
        if args.crack_model == "ddm" and args.ddm_fast:
            log(f"DDM fast preset active: ds_frac={args.crack_ds_frac}, a0_frac={args.crack_a0_frac}, "
                f"max_steps={args.crack_max_steps}, kmin_frac={args.crack_kmin_frac}", args.quiet)

        for sid in crack_ids:
            d = raw[sid]
            k = samples.index(sid)
            row, col = divmod(k, ncols)
            ax = axsF[row, col]

            try:
                if args.crack_model == "physics":
                    cres = crack_path_single_mirror_physics(
                        R=d["R"], alpha_const=d["alpha"], fit=d["fit"],
                        X=d["X"], Y=d["Y"], M=d["M"],
                        Teff_img=d["Teff_img"], Coh_img=d["Coh_img"], Phi_img=d["Phi_img"],
                        drive_basis=args.fail_mode_basis,
                        mc_compression_only=args.mc_compression_only,
                        mc_sigma_comp_min=d["mc_scmin"],
                        strength_model=args.strength_model,
                        weak_T_ratio=args.weak_plane_T_ratio,
                        weak_C_ratio=args.weak_plane_C_ratio,
                        weak_phi=d["weak_phi_scalar"],
                        weak_plane_prefer_path=True,
                        ds_frac=args.crack_ds_frac,
                        a0_frac=args.crack_a0_frac,
                        max_steps=args.crack_max_steps,
                        mixed_band=args.fail_mixed_band,
                        relax=args.crack_relax,
                        n_theta_mc=args.fail_mc_nplanes,
                        fail_margin=args.fail_margin,
                        max_turn_deg=args.crack_max_turn_deg,
                        hysteresis=1.35,
                        target_smooth=0.55,
                        crack_path_util_min=args.fail_util_min,
                        seg_check_n=7,
                        cand_n=41,
                        target_penalty=0.25,
                        turn_penalty=0.25,
                        enforce_outward=True,
                        self_intersection_stop=True
                    )
                else:
                    # DDM can be slow; optionally pass psi0 (if supported) to skip initial scan
                    ddm_kw = dict(
                        R=d["R"], alpha_const=d["alpha"], fit=d["fit"],
                        E1=E1, E2=E2, nu12=nu12, G12=G12,
                        ds_frac=args.crack_ds_frac,
                        a0_frac=args.crack_a0_frac,
                        max_steps=args.crack_max_steps,
                        max_turn_deg=args.crack_max_turn_deg,
                        relax=args.crack_relax,
                        enforce_outward=True,
                        self_intersection_stop=True,
                        kmin_frac=args.crack_kmin_frac,
                    )
                    if args.crack_psi0_deg is not None:
                        psi0_rad = np.deg2rad(float(args.crack_psi0_deg))
                        # only pass if signature includes psi0
                        sig = inspect.signature(crack_path_single_mirror_ddm)
                        if "psi0" in sig.parameters:
                            ddm_kw["psi0"] = psi0_rad
                            log(f"Using user psi0={args.crack_psi0_deg:.2f} deg (skips DDM initial search).", args.quiet)

                    cres = call_filtered(crack_path_single_mirror_ddm, **ddm_kw)

                xs = np.asarray(cres.get("xs", []), float)
                ys = np.asarray(cres.get("ys", []), float)

                if xs.size >= 2:
                    ax.plot(xs, ys, 'k-', lw=2.2, zorder=10)
                    plt.pause(0.05)  # update window during long runs

                # specimen mode along path
                spec_mode, fr_path, L = specimen_mode_from_crack_path(
                    xs, ys,
                    R=d["R"], alpha_const=d["alpha"], fit=d["fit"],
                    X=d["X"], Y=d["Y"],
                    Teff_img=d["Teff_img"], Coh_img=d["Coh_img"], Phi_img=d["Phi_img"],
                    specimen_mode_basis=args.fail_mode_basis,
                    mc_compression_only=args.mc_compression_only,
                    mc_sigma_comp_min=d["mc_scmin"],
                    strength_model=args.strength_model,
                    weak_T_ratio=args.weak_plane_T_ratio,
                    weak_C_ratio=args.weak_plane_C_ratio,
                    weak_phi=d["weak_phi_scalar"],
                    n_theta_mc=args.fail_mc_nplanes,
                    mixed_band=args.fail_mixed_band
                )

                # save path csv
                rock_dir = os.path.join(crack_dir_root, str(d["rock"]).replace(" ", "_"))
                os.makedirs(rock_dir, exist_ok=True)
                pd.DataFrame({"order": np.arange(xs.size), "x_m": xs, "y_m": ys}).to_csv(
                    os.path.join(rock_dir, f"crack_path_sample_{sid}.csv"), index=False
                )

                # update summary row
                for rr in summary_rows:
                    if rr["Sample"] == int(sid):
                        rr["SpecimenMode"] = str(spec_mode)
                        rr["CrackPath_Length_m"] = float(L) if np.isfinite(L) else np.nan
                        break

                log(f"[crack] sample {sid}: npts={xs.size} spec_mode={spec_mode} L={L:.4e} m", args.quiet)

            except KeyboardInterrupt:
                log("\n[STOP] Ctrl+C caught. Keeping plotted figure + saving what we have.", args.quiet)
                break
            except Exception as e:
                log(f"[WARN] Crack failed for sample {sid}: {e}", args.quiet)
                continue

        # save final figure with cracks overlay
        if not args.no_save:
            base = os.path.join(args.out_dir, args.save_basename + "_WITH_CRACKS")
            if args.save_pdf:
                figF.savefig(base + ".pdf", dpi=300, bbox_inches="tight", transparent=True)
            if args.save_png:
                figF.savefig(base + ".png", dpi=300, bbox_inches="tight", transparent=True)
            log(f"\n✓ Saved crack-overlay figure: {base}.(pdf/png)", args.quiet)

    # save summary csv
    dfR = pd.DataFrame(summary_rows)
    dfR.to_csv(os.path.join(args.out_dir, args.csv_out), index=False)
    log(f"\n✓ Saved summary CSV: {os.path.join(args.out_dir, args.csv_out)}", args.quiet)

    # final show (blocking)
    if not args.no_show:
        try:
            import matplotlib.pyplot as plt
            plt.show(block=(not IN_IPY))
        except TypeError:
            plt.show()

if __name__ == "__main__":
    main()
