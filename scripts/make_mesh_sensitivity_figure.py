#!/usr/bin/env python3
"""Convergence of the computed field with respect to its two discretisations.

The panel this replaces sampled a placeholder field, ``applied_stress`` times a
periodic strength factor, on progressively finer meshes. That expression is
closed form, so evaluating it at more points cannot converge to anything: the
curve it produced measured the smoothness of an analytic function, not the
accuracy of the solution the study uses.

Two discretisations in the actual computation can be refined, and both are
tested here against the quantities the study reports.

Boundary collocation
    The orthotropic Airy coefficients are fitted by collocating tractions on the
    disc boundary. Refining the number of collocation points is the convergence
    that matters, since it controls how well the traction boundary condition is
    satisfied. Error is measured against the most refined solution for the same
    specimen.

Field sampling
    The stress field is then evaluated on a grid. Refining it does not change
    the solution, but it does change area-integrated quantities, so the grid is
    tested on the failed-area fraction, which is the integral the results
    actually quote.

    python scripts/make_mesh_sensitivity_figure.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools.ddm._toolkit as tk                       # noqa: E402
from tools import output_dirs  # noqa: E402
from tools.data_io import load_specimen_table         # noqa: E402
from tools.plot_style import (apply_plot_style, ANNOT_FS,
                              LEGEND_FS, PANEL_LABEL_FS)         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = (REPO / output_dirs.DOC_DIR, REPO / output_dirs.FIGURE_DIR)
COLOR = {"Augen gneiss": "#4C72B0", "Psammitic schist": "#DD4B39"}

#: Collocation refinements; the production setting is the third.
NBD_LEVELS = [(105, 225), (210, 450), (420, 900), (840, 1800)]
NBD_REFERENCE = (1680, 3600)

#: Grid sizes; the production setting is 161, giving 20081 interior points.
GRID_LEVELS = [41, 61, 81, 121, 161, 201]


def _material(row):
    E1 = float(row["Modulus_of_Elasticity"]) * 1e3
    return dict(E1=E1, E2=E1 / tk.get_anisotropy_ratio(row["Rock_type"]),
                G=float(row["Shear_Modulus"]) * 1e3, nu=float(row["Poisson_Ratio"]),
                D=float(row["Diameter_mm"]) * 1e-3, t=float(row["Thickness_mm"]) * 1e-3,
                P=float(row["Load_(KN)"]) * 1e3, alpha=float(np.deg2rad(float(row["Angle"]))))


def _fit(m, nbd, arc):
    return tk.fit_orthotropic_airy_disk_auto(
        m["E1"], m["E2"], m["nu"], m["G"], R=m["D"] / 2, t=m["t"], P=m["P"],
        alpha=m["alpha"], beta_deg=10.0, smooth_deg=4.0, mu=0.0,
        Nbd=nbd, Nbd_arc_each=arc)


def _sigma_xx(m, fit, n=161):
    xg, yg, X, Y, M = tk.points_in_disk(m["D"], n=n)
    xm, ym = tk.rot_to_material(xg, yg, m["alpha"])
    a, b, c = tk.eval_stress_field_material(xm, ym, m["D"] / 2, fit["p1"], fit["p2"],
                                            fit["a1"], fit["a2"])
    return tk.stress_material_to_global(a, b, c, m["alpha"])[0]


def collocation_convergence(df) -> pd.DataFrame:
    rows = []
    for sid, row in df.iterrows():
        m = _material(row)
        ref = _sigma_xx(m, _fit(m, *NBD_REFERENCE))
        scale = float(np.max(np.abs(ref)))
        for nbd, arc in NBD_LEVELS:
            s = _sigma_xx(m, _fit(m, nbd, arc))
            rows.append(dict(sample_id=sid, rock=row["Rock_type"],
                             angle_deg=float(row["Angle"]), n_boundary=nbd,
                             rel_error=float(np.max(np.abs(s - ref)) / scale)))
        print(f"    specimen {sid:2d} done", flush=True)
    return pd.DataFrame(rows)


def grid_convergence(df) -> pd.DataFrame:
    """Area fraction above a fixed tensile level, against sampling density."""
    rows = []
    for sid, row in df.iterrows():
        m = _material(row)
        fit = _fit(m, *NBD_LEVELS[2])
        for n in GRID_LEVELS:
            xg, yg, X, Y, M = tk.points_in_disk(m["D"], n=n)
            xm, ym = tk.rot_to_material(xg, yg, m["alpha"])
            a, b, c = tk.eval_stress_field_material(xm, ym, m["D"] / 2, fit["p1"],
                                                    fit["p2"], fit["a1"], fit["a2"])
            sxx, syy, txy = tk.stress_material_to_global(a, b, c, m["alpha"])
            s1, _, _ = tk.principal_from_components(sxx, syy, txy)
            frac = float(np.mean(s1 >= float(row["Tensile_strength_Mpa"])))
            rows.append(dict(sample_id=sid, rock=row["Rock_type"],
                             n_interior=int(M.sum()), area_fraction=frac))
    return pd.DataFrame(rows)


def main():
    apply_plot_style()
    df = load_specimen_table()
    print("  boundary-collocation convergence (14 specimens x 4 levels):")
    col = collocation_convergence(df)
    grid = grid_convergence(df)
    output_dirs.tables()
    col.to_csv(REPO / output_dirs.TABLE_DIR / "convergence_collocation.csv", index=False)
    grid.to_csv(REPO / output_dirs.TABLE_DIR / "convergence_grid.csv", index=False)

    # side by side at full text width, drawn near the size it is reproduced at
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for rock, g in col.groupby("rock"):
        for sid, gg in g.groupby("sample_id"):
            gg = gg.sort_values("n_boundary")
            axes[0].plot(gg.n_boundary, 100 * gg.rel_error, "-o", ms=3, lw=1.0,
                         color=COLOR[rock], alpha=0.55)
        axes[0].plot([], [], "-o", color=COLOR[rock], label=rock, ms=4)
    axes[0].axvline(NBD_LEVELS[2][0], color="0.35", ls=":", lw=1.2)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    # explicit ticks: the default log minor labels collide at this width
    axes[0].set_xticks([n for n, _ in NBD_LEVELS])
    axes[0].set_xticklabels([str(n) for n, _ in NBD_LEVELS])
    axes[0].minorticks_off()
    axes[0].set_xlabel("Boundary collocation points")
    axes[0].set_ylabel(r"max $|\Delta\sigma_{xx}|$ (%)")
    axes[0].legend(frameon=True, fontsize=LEGEND_FS)

    for rock, g in grid.groupby("rock"):
        for sid, gg in g.groupby("sample_id"):
            gg = gg.sort_values("n_interior")
            axes[1].plot(gg.n_interior, 100 * gg.area_fraction, "-o", ms=3, lw=1.0,
                         color=COLOR[rock], alpha=0.55)
    axes[1].axvline(20081, color="0.35", ls=":", lw=1.2)
    axes[1].set_xscale("log")
    ticks = sorted(grid.n_interior.unique())
    axes[1].set_xticks(ticks[::2])
    axes[1].set_xticklabels([f"{t//1000}k" if t >= 1000 else str(t) for t in ticks[::2]])
    axes[1].minorticks_off()
    axes[1].set_xlabel("Interior sampling points")
    axes[1].set_ylabel("Area above $T_m$ (%)")

    for a, lab in zip(axes, ("(a)", "(b)")):
        a.text(0.5, -0.30, lab, transform=a.transAxes, ha="center", va="top",
               fontsize=PANEL_LABEL_FS)
        a.grid(True, alpha=0.3, which="both")
        for s in a.spines.values():
            s.set_linewidth(1.5)
    fig.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "mesh_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    prod = col[col.n_boundary == NBD_LEVELS[2][0]]
    print(f"\n  at the production setting ({NBD_LEVELS[2][0]} collocation points) the "
          f"largest deviation\n  from the refined solution is "
          f"{100 * prod.rel_error.max():.3f}% over all fourteen specimens")
    g = grid.sort_values("n_interior")
    last = grid[grid.n_interior == sorted(grid.n_interior.unique())[-1]]
    at_prod = grid[grid.n_interior == 20081]
    d_ = (at_prod.set_index("sample_id").area_fraction
          - last.set_index("sample_id").area_fraction).abs()
    print(f"  changing the grid from 20081 to {sorted(grid.n_interior.unique())[-1]} points "
          f"moves the area fraction by at most {100 * d_.max():.3f} percentage points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
