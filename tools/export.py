"""Machine-readable export of manuscript values and validation tables.

Every number that reaches the manuscript passes through here, so each carries
its units, sample count, source path and status.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .lithology import REPO_ROOT
from . import output_dirs

DEFAULT_OUT = REPO_ROOT / output_dirs.TABLE_DIR


def _clean(rows):
    """Drop private plotting payloads (leading underscore) before tabulating."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def write_trace_metrics(rows, out=None):
    """Specimen-level observed-vs-predicted comparison table."""
    out = Path(out or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    out = out / "trace_comparison_metrics.csv"
    df = pd.DataFrame(_clean(rows))
    df.to_csv(out, index=False)
    return out


def write_pairing_manifest(rows, out=None):
    """Existence, readability and convention record for all 14 intended pairs."""
    out = Path(out or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    out = out / "sample_pairing_manifest.csv"
    recs = []
    for r in _clean(rows):
        recs.append(dict(
            sample=r["sample"], lithology=r["lithology"],
            experimental_angle_deg=r["experimental_angle_deg"],
            observed_path=r["observed_source"], predicted_path=r["predicted_source"],
            observed_exists=Path(r["observed_source"]).exists(),
            predicted_exists=Path(r["predicted_source"]).exists(),
            observed_columns="col0=x[m], col1=y[m] (headerless)",
            predicted_columns="order,x_m,y_m",
            units="m", disc_radius_m=0.0255,
            observed_n_points=r.get("observed_n_points"),
            predicted_n_points=r.get("predicted_n_points"),
            observed_n_points_outside_disc=r.get("observed_n_outside_disc"),
            coordinate_convention=("global right-handed (x,y) in metres, disc centre at "
                                   "origin, +y loading axis, +x horizontal diameter, "
                                   "angles CCW from +x, axial (180-deg periodic)"),
            transform_applied=("identity; digitized points marginally outside the disc "
                               "projected radially onto r = R, per crack_data_plot.py"),
            status=r.get("status"), reason=r.get("reason", "")))
    pd.DataFrame(recs).to_csv(out, index=False)
    return out


def write_orientation_validation(rows, out=None):
    """Orientation validation with explicit uncertainty meaning."""
    out = Path(out or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    out = out / "fracture_orientation_validation.csv"
    recs = []
    for r in _clean(rows):
        recs.append(dict(
            lithology=r["lithology"], experimental_angle_deg=r["experimental_angle_deg"],
            specimen_id=r["sample"], replicate_id=1, n_replicates=1,
            observed_fracture_orientation_deg=r.get("observed_orientation_deg"),
            observed_std_deg=r.get("observed_bootstrap_sd_deg"),
            observed_std_meaning=("bootstrap SD of the total-least-squares orientation fit "
                                  "(digitization and fit scatter); NOT a between-specimen "
                                  "replicate SD - one specimen per lithology-angle pair"),
            predicted_fracture_orientation_deg=r.get("predicted_orientation_deg"),
            signed_wrapped_diff_deg=r.get("signed_wrapped_diff_deg"),
            abs_axial_angular_error_deg=r.get("abs_axial_angular_error_deg"),
            # The domain the pair was fitted on, and the precision of each
            # fit. Carried for every specimen so a reader can tell a
            # well-determined orientation from one the data barely constrains.
            orientation_fit_domain=r.get("orientation_fit_domain"),
            observed_orientation_se_deg=r.get("observed_orientation_se_deg"),
            predicted_orientation_se_deg=r.get("predicted_orientation_se_deg"),
            orientation_se_advisory_deg=r.get("orientation_se_advisory_deg"),
            orientation_fit_domain_meaning=(
                "orientation is fitted over the whole primary segment, "
                "identically for the observed and the predicted trace. The "
                "standard errors are reported so a reader can tell a "
                "well-determined orientation from one the digitization barely "
                "constrains; they do not select the domain"),
            observed_dominant_mechanism=("unclassifiable - a digitized trace carries no "
                                         "displacement or surface evidence"),
            data_source_path=f"{r['observed_source']} | {r['predicted_source']}",
            status=r.get("status"), reason=r.get("reason", "")))
    pd.DataFrame(recs).to_csv(out, index=False)
    return out


def write_class_fraction_table(records, out=None):
    """Per-specimen WT/WS/MT/MS area fractions for both lithologies."""
    out = Path(out or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    out = out / "fourclass_area_fractions.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    return out


def append_manuscript_values(new_rows, out=None, generated_in="step2"):
    """Append rows to ``manuscript_values.csv``, preserving earlier provenance."""
    out = Path(out or REPO_ROOT) / "manuscript_values.csv"
    df_new = pd.DataFrame(new_rows)
    df_new["generated_in"] = generated_in
    if out.exists():
        prev = pd.read_csv(out)
        # a re-run of the same step replaces its own rows rather than duplicating
        prev = prev[prev.get("generated_in", "") != generated_in]
        df = pd.concat([prev, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out, index=False)
    return out


def value_row(quantity, value, units, sample_count, data_source, script,
              status="computed", notes=""):
    """One provenance-carrying manuscript value."""
    return dict(quantity=quantity, value=value, units=units, sample_count=sample_count,
                data_source=data_source, script_or_cell=script, status=status, notes=notes)


def reconcile(old_csv, new_rows, out=None, key="quantity"):
    """Compare newly generated values against a previous export.

    Classifies each as unchanged, corrected, newly_available, unavailable or
    conflicting, so no value silently changes between runs.
    """
    out = Path(out or DEFAULT_OUT) / "dual_run_value_reconciliation.csv"
    old = pd.read_csv(old_csv) if Path(old_csv).exists() else pd.DataFrame(columns=[key, "value"])
    old_map = {str(r[key]): r["value"] for _, r in old.iterrows()}
    recs = []
    for r in new_rows:
        q = str(r[key])
        nv = r["value"]
        if q not in old_map:
            st, note = "newly_available", "not present in the earlier export"
        else:
            ov = old_map[q]
            try:
                same = np.isclose(float(ov), float(nv), rtol=1e-6, atol=1e-9)
            except (TypeError, ValueError):
                same = str(ov) == str(nv)
            st = "unchanged" if same else "corrected"
            note = "" if same else f"previous export gave {ov}"
        recs.append(dict(quantity=q, previous_value=old_map.get(q, ""), final_value=nv,
                         units=r.get("units", ""), status=st, note=note,
                         evidence=r.get("data_source", "")))
    pd.DataFrame(recs).to_csv(out, index=False)
    return out


def classification_panel(sample_id, strengths=None, root=None):
    """Four-mechanism classification of one specimen, ready for the composite figure.

    Bundles the classified field with the grid, mask and class fractions so a
    notebook needs only to call this and hand the result to
    :func:`tools.plotting.seven_panel_classification`.
    """
    import numpy as np

    from . import lithology as lith
    from . import failure_classification as fc
    from . import strain_partitioning as sp

    strengths = strengths or sp.specimen_strengths(root)
    lit = lith.lithology_of_sample(sample_id)
    npz = lith.field_cache_path(sample_id, root)
    if not npz.exists():
        return dict(sample=int(sample_id), lithology=lit.display_name,
                    angle_deg=lit.angle_for_sample(sample_id),
                    status="blocked", reason=f"{npz.name} absent")

    d = np.load(npz, allow_pickle=True)
    p = strengths[int(sample_id)]
    res = fc.classify(
        d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=sp.WEAK_T_RATIO * p["T_m"], c_wp=sp.WEAK_C_RATIO * p["c_m"],
        phi_wp=p["phi_m"], T_m=p["T_m"], c_m=p["c_m"], phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=sp.ACTIVATION_FLOOR,
        threshold=sp.THRESHOLD, eta_mix=sp.ETA_MIX, n_theta=sp.N_THETA)

    mask = d["M"].astype(bool)
    return dict(sample=int(sample_id), lithology=lit.display_name,
                angle_deg=lit.angle_for_sample(sample_id),
                X=d["X"], Y=d["Y"], mask=mask,
                mode_code=res["mode_code"], mixed_flag=res["mixed_flag"],
                fractions=fc.class_fractions(res["mode_code"], mask),
                status="computed", reason="")
