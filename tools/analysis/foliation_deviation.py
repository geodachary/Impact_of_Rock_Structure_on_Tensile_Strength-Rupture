"""Fracture deviation from the foliation, by angle.

**Maintained directly.** Originally extracted from Tensile_augen_gneiss.ipynb cell 60 during the
notebook-to-package migration; that migration is complete and this module is now
the source, so edit it here. The extraction tooling is retained only as a record
of the migration and refuses to run without ``--force``.

At extraction the code was unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.
"""
from __future__ import annotations


import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tools.bound_all_helpers import (
    points_in_disk,
    map_angle_to_alpha,
    rot_to_material,
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
    fit_orthotropic_airy_disk,
    fit_orthotropic_airy_disk_auto,
)

from tools.analysis._context import bind as _bind, current as _rock

from tools import output_dirs

# --- inherited from earlier notebook cells ---------------------------
from tools.ddm import (  # noqa: F401
    compute_sample_stresses, deviation_arrays, make_figure1, make_figure2,
    output_prefix_from_records, set_pub_style,
)
from tools.lithology import repo_relative
from tools.ddm._toolkit import ROCK_ANISO_RATIO as _CANONICAL_ANISO_RATIO



# --- implementation ---------------------------------------------------


def sanitize_argv(argv):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "-f":
            skip = True
            continue
        if a.startswith("--f=") or a.startswith("-f="):
            continue
        out.append(a)
    return out


def parse_sample_ids(spec, valid_index):
    wanted = set()
    if spec and isinstance(spec, str):
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    wanted.update(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
                except ValueError:
                    pass
            else:
                try:
                    wanted.add(int(tok))
                except ValueError:
                    pass
    return [i for i in valid_index if i in wanted]



def main(rock):
    """Run this section for one lithology.

    Parameters
    ----------
    rock : tools.lithology.Lithology
        Supplies the specimen ids, weak-plane spacing and output stem.
    """
    global ROCK_ANISO_RATIO, TOL_VIBRANT, _unknown, ang_deg, args, argv, c, \
        d_arr, d_sample, dfmeta, drv, fig1, fig1_path, fig2, fig2_path, \
        out_dir, p, prefix, required_cols, row, row_dict, sample_records, \
        samples, sid
    _bind(rock)
    """
    Publication-quality figures — revised foliation deviation analysis
    ==================================================================

    Physics revision:
      - Uses row-specific elastic properties from the CSV:
          E1   = Modulus_of_Elasticity
          G12  = Shear_Modulus
          nu12 = Poisson_Ratio
      - Derives E2 from rock-type anisotropy:
          E2 = E1 / anisotropy_ratio
      - Default sample range is 1-7 (Augen gneiss)

    Produces exactly TWO figures:

      Figure 1
          Single-panel trimmed violin summary of signed deviation
          Δ = θ_p − α_wp for all specimens / foliation angles.

      Figure 2
          Median signed deviation vs. foliation angle using per-angle IQR
          error bars, with a pooled ECDF inset in the upper-left.
    """
    # One dict, defined in _toolkit and derived from the replicate table.
    ROCK_ANISO_RATIO = dict(_CANONICAL_ANISO_RATIO)
    TOL_VIBRANT = [
        "#0077BB",  # blue
        "#EE7733",  # orange
        "#009988",  # teal
        "#CC3311",  # red
        "#33BBEE",  # cyan
        "#EE3377",  # magenta
        "#BBBBBB",  # grey
    ]
    p = argparse.ArgumentParser(
        "Publication figures — revised foliation deviation",
        allow_abbrev=False,
    )
    p.add_argument("--meta_csv", default="tensile_samples_data.csv")
    p.add_argument("--out_dir", default=output_dirs.FIGURE_DIR)

    # Default kept at 1-7 as requested
    p.add_argument("--sample_ids", type=str, default=_rock().sample_id_spec())

    p.add_argument("--angle_map", type=str, default="direct",
                   choices=["direct", "neg", "pi2_minus", "pi2_plus"])
    p.add_argument("--points_per_row", type=int, default=161)

    # Fallback defaults only; row values are preferred
    p.add_argument("--E1_GPa", type=float, default=50.0)
    p.add_argument("--G12_GPa", type=float, default=12.0)
    p.add_argument("--nu12", type=float, default=0.25)

    p.add_argument("--platen_half_angle_deg", type=float, default=10.0)
    p.add_argument("--platen_smooth_deg", type=float, default=4.0)
    p.add_argument("--platen_mu", type=float, default=0.0)

    p.add_argument("--airy_auto_tune", action="store_true", default=True)
    p.add_argument("--airy_Nbd_base", type=int, default=420)
    p.add_argument("--airy_Nbd_arc_each", type=int, default=900)

    p.add_argument("--load_factor", type=float, default=1.0)
    p.add_argument("--keep_top_percent", type=float, default=20.0)
    p.add_argument("--rmax_frac", type=float, default=0.985)

    p.add_argument("--fig1_width_in", type=float, default=5.0,
                   help="Figure 1 width in inches")
    p.add_argument("--fig1_height_in", type=float, default=3.5,
                   help="Figure 1 height in inches")
    p.add_argument("--fig2_width_in", type=float, default=3.7,
                   help="Figure 2 width in inches")
    p.add_argument("--fig2_height_in", type=float, default=3.80,
                   help="Figure 2 height in inches")

    argv = sanitize_argv(sys.argv[1:])
    args, _unknown = p.parse_known_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir

    set_pub_style()

    dfmeta = pd.read_csv(args.meta_csv, index_col=0)
    dfmeta.columns = [str(c).strip() for c in dfmeta.columns]

    required_cols = [
        "Rock_type", "Angle", "Diameter_mm", "Thickness_mm", "Load_(KN)",
        "Tensile_strength_Mpa", "Cohesion", "Friction_Angle",
        "Modulus_of_Elasticity", "Poisson_Ratio", "Shear_Modulus",
    ]
    for c in required_cols:
        if c not in dfmeta.columns:
            raise RuntimeError(f"Missing column: {c}")

    samples = parse_sample_ids(args.sample_ids, list(dfmeta.index))
    if not samples:
        raise RuntimeError("No matching sample IDs found.")

    sample_records = []
    for sid in samples:
        row = dfmeta.loc[sid, required_cols]
        row_dict = dict(zip(required_cols, row))
        rock = str(row_dict["Rock_type"])
        ang_deg = float(row_dict["Angle"])

        d_sample = compute_sample_stresses(row_dict, args)

        print(
            f"  Processing sample {sid}: {rock}  α={ang_deg:.0f}°"
            f"  |  E1={d_sample['E1_GPa']:.2f} GPa"
            f"  E2={d_sample['E2_GPa']:.2f} GPa"
            f"  G12={d_sample['G12_GPa']:.2f} GPa"
            f"  nu12={d_sample['nu12']:.2f}"
        )

        d_arr, drv = deviation_arrays(
            d_sample,
            rmax_frac=float(args.rmax_frac),
            keep_top_percent=float(args.keep_top_percent),
        )

        sample_records.append(dict(
            sid=sid,
            rock=rock,
            ang_deg=ang_deg,
            alpha_wp_deg=float(np.degrees(d_sample["alpha_wp_line"])),
            delta_deg=d_arr,
            drive=drv,
        ))

    prefix = output_prefix_from_records(sample_records)

    fig1_path = os.path.join(out_dir, f"{prefix}_combined_distribution.pdf")
    fig1 = make_figure1(
        sample_records,
        fig1_path,
        fig_width_in=float(args.fig1_width_in),
        fig_height_in=float(args.fig1_height_in),
    )

    fig2_path = os.path.join(out_dir, f"{prefix}_deviation_vs_angle.pdf")
    fig2 = make_figure2(
        sample_records,
        fig2_path,
        fig_width_in=float(args.fig2_width_in),
        fig_height_in=float(args.fig2_height_in),
    )

    # Export the per-specimen statistics the manuscript quotes. Until now this
    # analysis produced only two figures, so the deviation medians in Section
    # 4.5 had no machine-readable producer and could not be checked against
    # anything. They are the last values in the paper without one.
    import pandas as _pd
    from tools import output_dirs as _od
    rows = []
    for r in sample_records:
        dv = np.asarray(r["delta_deg"], float)
        rows.append(dict(
            sample_id=r["sid"], rock=r["rock"], angle_deg=r["ang_deg"],
            n_points=int(dv.size),
            delta_median_deg=float(np.median(dv)),
            delta_abs_median_deg=float(np.median(np.abs(dv))),
            delta_p25_deg=float(np.percentile(dv, 25)),
            delta_p75_deg=float(np.percentile(dv, 75)),
        ))
    _df = _pd.DataFrame(rows).sort_values("sample_id")
    _path = Path(_od.tables()) / "foliation_deviation_statistics.csv"
    if _path.exists():                       # called once per lithology
        try:
            _prev = _pd.read_csv(_path)
            if list(_prev.columns) == list(_df.columns):
                _df = _pd.concat([_prev[~_prev.sample_id.isin(_df.sample_id)], _df],
                                 ignore_index=True).sort_values("sample_id")
        except Exception:
            pass
    _df.to_csv(_path, index=False)
    print(f"  deviation statistics -> {repo_relative(_path)}")

    plt.show()
    print("\nDone. Both revised figures written to:", out_dir)
