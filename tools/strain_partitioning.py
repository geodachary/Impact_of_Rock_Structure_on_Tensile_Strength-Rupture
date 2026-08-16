"""Strain-energy partitioning driven by the four-mechanism classifier.

Why this module exists
----------------------
The published partitioning was computed by a separate parametric
energy-blending routine (``partition_energy`` in the notebooks) that takes only
``s1``, ``s3`` and the energy field and blends them into tensile / shear / mixed
using per-lithology tuned constants. **It never consults the failure
classifier.** Adding the WT/WS/MT/MS scheme therefore could not change its
output, which is why the partitioning appeared not to respond to the new
classification.

This module computes the partition directly from the four-mechanism classifier,
so the energy split and the failure maps are two views of one calculation.
Everything is rock-specific: strengths come from the per-specimen experimental
table and the stress and energy fields come from that specimen's own archive.

Regimes
-------
The three "principal-stress permutations" of the published figure are, in the
generating code, additive mean-stress offsets applied to both in-plane
principals. They are reproduced here with their original labels so the two
analyses can be compared, but named for what they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import lithology as lith
from . import failure_classification as fc

#: Mean-stress offsets (MPa) the published code applies, with its regime labels.
REGIME_OFFSETS_MPA = {"Thrust": +0.50, "Strike-Slip": 0.00, "Extensional": -0.50}

#: Weak-plane strength proxies: no foliation-plane strength tests exist, so the
#: weak-plane strengths are scaled from the measured matrix values.
WEAK_T_RATIO = 0.35
WEAK_C_RATIO = 0.60

#: Classifier settings, identical to those used for the failure-mode figures.
THRESHOLD = 1.0
ETA_MIX = 0.90
N_THETA = 361
ACTIVATION_FLOOR = 0.05


def specimen_strengths(root=None) -> dict:
    """Per-specimen matrix strengths from the experimental table.

    Returns ``{sample_id: {T_m, c_m, phi_m}}`` with the friction angle in
    radians. These differ specimen by specimen, which is the primary reason the
    two lithologies must not share a parameter set.

    Specimens are identified by lithology and foliation angle rather than by a
    serial-number column. The table is regenerated from the per-replicate
    measurements in ``selected_all_samples.csv``, and that regeneration does not
    carry a serial number through; lithology and angle do identify a specimen
    uniquely, so they are the stable key. A ``SN`` column is still honoured when
    present, and the historical ``Psammatic`` spelling still resolves.
    """
    root = root or lith.REPO_ROOT
    df = pd.read_csv(root / "tensile_samples_data.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    missing = {"Rock_type", "Angle", "Tensile_strength_Mpa", "Cohesion",
               "Friction_Angle"} - set(df.columns)
    if missing:
        raise KeyError(f"tensile_samples_data.csv is missing {sorted(missing)}")

    out = {}
    for r in df.itertuples():
        if hasattr(r, "SN") and not pd.isna(getattr(r, "SN")):
            sid = int(getattr(r, "SN"))
        else:
            token = str(r.Rock_type).strip().lower().replace(" ", "_")
            token = lith.LEGACY_TOKENS.get(token, token)
            sid = lith.get(token).sample_for_angle(int(round(float(r.Angle))))
        out[sid] = dict(T_m=float(r.Tensile_strength_Mpa),
                        c_m=float(r.Cohesion),
                        phi_m=np.radians(float(r.Friction_Angle)))

    expected = set(range(1, 15))
    if set(out) != expected:
        raise ValueError(
            f"expected specimens {sorted(expected)}, resolved {sorted(out)} — "
            "check Rock_type spellings and the 0-90 deg angle coverage")
    return out


def apply_mean_stress_offset(s1, s3, theta, offset_mpa):
    """Rebuild the in-plane stress tensor with a uniform mean-stress offset.

    Principal directions and deviatoric magnitudes are unchanged; only the mean
    stress moves. This is what the published regimes actually do.
    """
    a = np.asarray(s1, float) + offset_mpa
    b = np.asarray(s3, float) + offset_mpa
    c, s = np.cos(theta), np.sin(theta)
    sxx = a * c * c + b * s * s
    syy = a * s * s + b * c * c
    txy = (a - b) * c * s
    return sxx, syy, txy


def partition_specimen(sample_id, regime="Strike-Slip", strengths=None, root=None):
    """Four-mechanism energy partition for one specimen under one regime.

    Returns a dict with the energy in each class, the percentage split, the
    locus and mechanism roll-ups, and the parameters used — so the row carries
    its own provenance.
    """
    strengths = strengths or specimen_strengths(root)
    npz = lith.field_cache_path(sample_id, root)
    lit = lith.lithology_of_sample(sample_id)
    if not npz.exists():
        return dict(sample=sample_id, lithology=lit.display_name, regime=regime,
                    status="blocked", reason=f"{npz.name} absent")

    d = np.load(npz, allow_pickle=True)
    p = strengths[int(sample_id)]
    mask = d["M"].astype(bool)
    U = np.where(mask, np.asarray(d["U_MPa"], float), 0.0)

    sxx, syy, txy = apply_mean_stress_offset(
        d["s1"], d["s3"], d["th_rad"], REGIME_OFFSETS_MPA[regime])

    res = fc.classify(
        sxx, syy, txy, alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=WEAK_T_RATIO * p["T_m"], c_wp=WEAK_C_RATIO * p["c_m"], phi_wp=p["phi_m"],
        T_m=p["T_m"], c_m=p["c_m"], phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=ACTIVATION_FLOOR,
        threshold=THRESHOLD, eta_mix=ETA_MIX, n_theta=N_THETA)

    code = res["mode_code"]
    total = float(U.sum())
    row = dict(sample=int(sample_id), lithology=lit.display_name,
               lithology_key=lit.key, angle_deg=lit.angle_for_sample(sample_id),
               regime=regime, mean_stress_offset_MPa=REGIME_OFFSETS_MPA[regime],
               total_U_MJ_m3=total)
    for cls in fc.CLASS_ORDER:
        e = float(U[mask & (code == fc.CLASS_CODES[cls])].sum())
        row[f"{cls}_MJ_m3"] = e
        row[f"{cls}_pct"] = 100.0 * e / total if total else np.nan
    e_nf = float(U[mask & (code == fc.CLASS_CODES[fc.NO_FAILURE])].sum())
    row["below_threshold_MJ_m3"] = e_nf
    row["below_threshold_pct"] = 100.0 * e_nf / total if total else np.nan

    row["weak_plane_pct"] = row["WT_pct"] + row["WS_pct"]
    row["matrix_pct"] = row["MT_pct"] + row["MS_pct"]
    row["tensile_pct"] = row["WT_pct"] + row["MT_pct"]
    row["shear_pct"] = row["WS_pct"] + row["MS_pct"]
    row["mixed_flag_pct"] = (100.0 * float(U[mask & res["mixed_flag"]].sum()) / total
                             if total else np.nan)
    # provenance: the rock-specific inputs that make the two lithologies differ
    row["T_m_MPa"] = p["T_m"]
    row["c_m_MPa"] = p["c_m"]
    row["phi_m_deg"] = float(np.degrees(p["phi_m"]))
    row["status"] = "computed"
    row["reason"] = ""
    return row


def partition_all(root=None) -> pd.DataFrame:
    """Four-mechanism partition for all 14 specimens and all three regimes."""
    strengths = specimen_strengths(root)
    rows = [partition_specimen(sid, reg, strengths, root)
            for sid in range(1, 15) for reg in REGIME_OFFSETS_MPA]
    return pd.DataFrame(rows)


def lithology_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side gneiss vs schist means, with the difference.

    This is the table that demonstrates the two lithologies do not share a
    result. A zero difference row would indicate the rock-specific inputs are
    not reaching the calculation.
    """
    g = df[df.status == "computed"]
    cols = ["WT_pct", "WS_pct", "MT_pct", "MS_pct", "below_threshold_pct",
            "weak_plane_pct", "matrix_pct", "tensile_pct", "shear_pct",
            "total_U_MJ_m3", "T_m_MPa", "c_m_MPa", "phi_m_deg"]
    piv = g.groupby("lithology")[cols].mean().T
    piv.columns.name = None
    piv["difference"] = piv["Augen gneiss"] - piv["Psammitic schist"]
    piv["identical"] = piv["difference"].abs() < 1e-9
    return piv
