# ==========================================================
# B) ABLATION RUNNER + FIGURES (paths + success rate)
# ==========================================================
# This assumes you already have per-sample:
#   - R, alpha_const, fit, X,Y,M, util_path_img, psi_pref_guided_img
#   - and "physics-only" method available:
#       crack_path_single_mirror_physics(...)  (graph/local from crack_helpers)
#
# It generates:
#   - One multi-panel figure per method
#   - A bar chart of “success rate” by method
#
# You can integrate this into your main loop or run as a separate script.

import numpy as np
import matplotlib.pyplot as plt

def _is_success(xs, ys, R, r_end_frac=0.98, min_pts=15):
    if xs is None or ys is None:
        return False
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < int(min_pts):
        return False
    rmax = float(np.max(np.hypot(xs, ys)))
    return (rmax >= float(r_end_frac) * float(R))

def run_ablation_for_samples(sample_records, out_dir, r_end_frac=0.98):
    """
    sample_records: list of dicts, each dict has keys:
      sid, rock, ang_deg, R, alpha_const, fit, X, Y, M, util_path_img, psi_pref_guided_img,
      E1,E2,nu12,G12,
      Teff_img,Coh_img,Phi_img  (for physics-only crack_helpers path)

    Methods:
      1) energy_only
      2) physics_only (graph/local)
      3) hybrid (controller)
      4) hybrid_no_align  (set w_align=0 in guided or ignore psi_pref in phys burst)
      5) hybrid_no_Gc     (use_weak_plane=False)
    """
    os.makedirs(out_dir, exist_ok=True)

    methods = [
        ("energy_only", dict()),
        ("physics_only", dict()),
        ("hybrid", dict()),
        ("hybrid_no_align", dict()),
        ("hybrid_no_Gc", dict()),
    ]

    results = {m: [] for m, _ in methods}

    for rec in sample_records:
        sid = rec["sid"]
        R = rec["R"]

        for m, kw in methods:
            xs = ys = None
            ok = False
            err = None

            try:
                if m == "energy_only":
                    xs, ys = crack_path_ddm_energy_mirror(
                        R=rec["R"], alpha_const=rec["alpha_const"], airy_fit=rec["fit"],
                        E1=rec["E1"], E2=rec["E2"], nu12=rec["nu12"], G12=rec["G12"],
                        psi0=np.pi/2, a0_frac=0.008, ds_frac=0.006,
                        max_steps=2200, r_end_frac=0.995,
                        kink_scan_deg=25.0, kink_n=61, objective="diff",
                        use_weak_plane=True, Gc0=1.0, weak_reduction=0.35, eta_deg=10.0,
                    )

                elif m == "physics_only":
                    # from crack_helpers.py
                    out = crack_path_single_mirror_physics(
                        R=rec["R"], alpha_const=rec["alpha_const"], fit=rec["fit"],
                        X=rec["X"], Y=rec["Y"],
                        Teff_img=rec["Teff_img"], Coh_img=rec["Coh_img"], Phi_img=rec["Phi_img"],
                        start_mode="platen_to_platen",   # or "graph_platen"
                        platen_beta_deg=10.0,
                        platen_snap_extra_deg=10.0,
                    )
                    xs, ys = out["xs"], out["ys"]

                elif m == "hybrid":
                    xs, ys = crack_path_ddm_hybrid_controller_mirror(
                        R=rec["R"], alpha_const=rec["alpha_const"], airy_fit=rec["fit"],
                        X=rec["X"], Y=rec["Y"], M=rec["M"],
                        util_path_img=rec["util_path_img"],
                        psi_pref_guided_img=rec["psi_pref_guided_img"],
                        E1=rec["E1"], E2=rec["E2"], nu12=rec["nu12"], G12=rec["G12"],
                        psi0=np.pi/2,
                        a0_frac=0.010,
                        ds_frac=0.006,
                        max_steps=2500,
                        use_weak_plane=True,
                    )

                elif m == "hybrid_no_align":
                    # Disable any use of psi_pref during phys bursts by forcing vertical-only pref
                    psi_vertical = np.full_like(rec["psi_pref_guided_img"], np.pi/2.0, float)
                    xs, ys = crack_path_ddm_hybrid_controller_mirror(
                        R=rec["R"], alpha_const=rec["alpha_const"], airy_fit=rec["fit"],
                        X=rec["X"], Y=rec["Y"], M=rec["M"],
                        util_path_img=rec["util_path_img"],
                        psi_pref_guided_img=psi_vertical,
                        E1=rec["E1"], E2=rec["E2"], nu12=rec["nu12"], G12=rec["G12"],
                        psi0=np.pi/2,
                        a0_frac=0.010,
                        ds_frac=0.006,
                        max_steps=2500,
                        use_weak_plane=True,
                    )

                elif m == "hybrid_no_Gc":
                    xs, ys = crack_path_ddm_hybrid_controller_mirror(
                        R=rec["R"], alpha_const=rec["alpha_const"], airy_fit=rec["fit"],
                        X=rec["X"], Y=rec["Y"], M=rec["M"],
                        util_path_img=rec["util_path_img"],
                        psi_pref_guided_img=rec["psi_pref_guided_img"],
                        E1=rec["E1"], E2=rec["E2"], nu12=rec["nu12"], G12=rec["G12"],
                        psi0=np.pi/2,
                        a0_frac=0.010,
                        ds_frac=0.006,
                        max_steps=2500,
                        use_weak_plane=False,   # <-- key ablation
                        Gc0=1.0,
                    )

                ok = _is_success(xs, ys, R, r_end_frac=r_end_frac, min_pts=15)

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                ok = False

            results[m].append({
                "sid": sid,
                "rock": rec.get("rock"),
                "ang_deg": rec.get("ang_deg"),
                "xs": xs, "ys": ys,
                "ok": ok,
                "err": err,
            })

    # ---------- FIGURES: paths ----------
    for m, _ in methods:
        recs = results[m]
        n = len(recs)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(14, 5.0*nrows))
        if nrows == 1:
            axs = np.array([axs])

        for k, rr in enumerate(recs):
            row, col = divmod(k, ncols)
            ax = axs[row, col]
            sid = rr["sid"]
            xs = rr["xs"]; ys = rr["ys"]

            # disk circle (need R from sample_records)
            R = sample_records[k]["R"] if k < len(sample_records) else sample_records[0]["R"]
            ax.add_artist(plt.Circle((0, 0), R, fill=False, color="k", lw=1.2))
            if xs is not None and ys is not None:
                ax.plot(xs, ys, "k-", lw=2.0)
            ax.set_aspect("equal", "box")
            ax.set_xlim(-1.05*R, 1.05*R)
            ax.set_ylim(-1.05*R, 1.05*R)
            ax.set_title(f"{m} | sample {sid} | ok={rr['ok']}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

        for kk in range(n, nrows*ncols):
            row, col = divmod(kk, ncols)
            axs[row, col].axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"ablation_paths_{m}.png"), dpi=250)
        plt.close(fig)

    # ---------- FIGURE: success rate ----------
    labels = [m for m, _ in methods]
    succ = [sum(1 for r in results[m] if r["ok"]) for m, _ in methods]
    total = [len(results[m]) for m, _ in methods]
    frac = [s / max(t, 1) for s, t in zip(succ, total)]

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(np.arange(len(labels)), frac)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success fraction")
    ax.set_title("Ablation: crack path success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_success_rate.png"), dpi=250)
    plt.close(fig)

    return results
