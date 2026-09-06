"""Integration tests: data integrity, the 14-specimen pairing, and the
end-to-end trace and classification pipelines for both lithologies.

These exercise real repository inputs, not synthetic fixtures, so a broken path,
a renamed file or a cross-lithology mix-up fails here rather than silently in a
notebook.
"""
from pathlib import Path

import numpy as np
import pytest

from tools import lithology as lith
from tools import traces as tr
from tools import failure_classification as fc
from tools import strain_partitioning as sp
from tools import fabric_tractions as ft
from tools.conventions import RADIUS_M

ANGLES = (0, 15, 30, 45, 60, 75, 90)


# ---------------------------------------------------------------------------
# Data integrity and the authoritative pairing
# ---------------------------------------------------------------------------
def test_pairing_has_exactly_14_specimens(pairing):
    assert len(pairing) == 14
    assert sorted(r["sample"] for r in pairing) == list(range(1, 15))


def test_gneiss_owns_1_to_7_and_schist_8_to_14():
    assert lith.AUGEN_GNEISS.sample_ids == (1, 2, 3, 4, 5, 6, 7)
    assert lith.PSAMMITIC_SCHIST.sample_ids == (8, 9, 10, 11, 12, 13, 14)
    for s in range(1, 8):
        assert lith.lithology_of_sample(s).key == "augen_gneiss"
    for s in range(8, 15):
        assert lith.lithology_of_sample(s).key == "psammitic_schist"


def test_each_lithology_covers_all_seven_angles():
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        assert tuple(lit.angles_deg) == ANGLES
        assert len({lit.sample_for_angle(a) for a in ANGLES}) == 7


@pytest.mark.parametrize("sample", range(1, 15))
def test_observed_and_predicted_trace_files_exist(sample):
    assert lith.observed_trace_path(sample).exists(), f"missing observed trace, sample {sample}"
    assert lith.predicted_trace_path(sample).exists(), f"missing DDM trace, sample {sample}"


def test_explicit_endpoint_pairings_from_the_specification():
    """The two pairings the revision specification names explicitly."""
    assert lith.predicted_trace_path(1).name == "ddm_crack_sample_1.csv"
    assert lith.observed_trace_path(1).name == "augen_gneiss_0_sample_1.csv"
    assert lith.predicted_trace_path(14).name == "ddm_crack_sample_14.csv"
    assert lith.observed_trace_path(14).name == "psammitic_schist_90_sample_14.csv"


@pytest.mark.parametrize("sample", range(1, 15))
def test_filename_encodes_the_expected_lithology_and_angle(sample):
    p = lith.observed_trace_path(sample)
    lit = lith.lithology_of_sample(sample)
    stem = p.stem
    assert stem.startswith(lit.key), f"{p.name} does not carry token {lit.key}"
    assert int(stem.split("_")[-3]) == lit.angle_for_sample(sample)
    assert int(stem.split("_")[-1]) == sample


def test_no_active_path_uses_the_legacy_psammatic_spelling():
    for r in lith.pairing_table():
        assert "psammatic" not in str(r["observed_path"]).lower()
        assert "psammatic" not in str(r["predicted_path"]).lower()


@pytest.mark.parametrize("sample", range(1, 15))
def test_traces_load_in_metres_inside_the_specimen(sample):
    o = tr.load_observed_trace(lith.observed_trace_path(sample))
    p = tr.load_predicted_trace(lith.predicted_trace_path(sample))
    assert o["n_points"] >= 10 and p["n_points"] >= 10
    # after the documented radial clip, nothing may lie outside the disc
    assert np.max(np.hypot(o["x"], o["y"])) <= RADIUS_M * (1 + 1e-9)
    # metres, not millimetres
    assert np.max(np.hypot(p["x"], p["y"])) < 1.0


# ---------------------------------------------------------------------------
# Orientation pipeline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sample", range(1, 15))
def test_specimen_comparison_is_computable(sample):
    r = tr.compare_specimen(sample)
    assert r["status"] == "computed", r.get("reason")
    assert 0.0 <= r["abs_axial_angular_error_deg"] <= 90.0
    assert 0.0 <= r["observed_orientation_deg"] < 180.0
    assert 0.0 <= r["predicted_orientation_deg"] < 180.0


def test_aggregate_statistics_use_only_computed_rows():
    rows = tr.compare_all()
    agg = tr.aggregate_statistics(rows)
    assert agg["n"] == sum(1 for r in rows if r["status"] == "computed")
    assert agg["n_within_5"] <= agg["n_within_10"] <= agg["n"]
    assert agg["median_deg"] <= agg["max_deg"]
    assert agg["mae_deg"] <= agg["rmse_deg"] + 1e-12  # MAE never exceeds RMSE


def test_null_model_is_evaluated_on_the_same_specimens():
    rows = tr.compare_all()
    assert tr.null_model_statistics(rows)["n"] == tr.aggregate_statistics(rows)["n"]


# ---------------------------------------------------------------------------
# Four-mechanism classification, per lithology, on real cached fields
# ---------------------------------------------------------------------------
def _classify(sample):
    """Classify one specimen exactly as the reported pipeline does.

    The strengths must come from ``strain_partitioning.specimen_strengths``,
    not from placeholder constants. An earlier version of this helper passed
    T_wp = c_wp = 1 and T_m = c_m = 2, which is a different material from the
    one the study reports, and the class fractions it produced disagreed with
    outputs/tables/fourclass_area_fractions.csv by two orders of magnitude.
    A test that exercises a different material than the manuscript cannot
    protect the manuscript, so the call below mirrors
    scripts/reproduce_all._classify_panel.
    """
    npz = lith.field_cache_path(sample)
    if not npz.exists():
        pytest.skip(f"field cache absent for sample {sample}")
    d = np.load(npz, allow_pickle=True)
    p = sp.specimen_strengths()[int(sample)]
    res = fc.classify(
        d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=sp.WEAK_T_RATIO * p["T_m"], c_wp=sp.WEAK_C_RATIO * p["c_m"],
        phi_wp=p["phi_m"], T_m=p["T_m"], c_m=p["c_m"], phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=sp.ACTIVATION_FLOOR,
        threshold=sp.THRESHOLD, eta_mix=sp.ETA_MIX, n_theta=sp.N_THETA)
    return res, d


def _core(d):
    """The 0.85 R reporting interior, as used for every field statistic."""
    m = d["M"].astype(bool)
    r = np.hypot(np.asarray(d["X"]), np.asarray(d["Y"]))
    return m & (r <= ft.CORE_FRAC * float(np.nanmax(r[m])))


@pytest.mark.parametrize("sample", [1, 7, 8, 14])
def test_classification_produces_only_the_four_primary_classes(sample):
    res, _d = _classify(sample)
    labels = set(np.unique(res["mode"]))
    assert labels <= set(fc.CLASS_ORDER) | {fc.NO_FAILURE}, labels
    codes = set(np.unique(res["mode_code"]).tolist())
    assert codes <= set(fc.CLASS_CODES.values())


@pytest.mark.parametrize("sample", [1, 7, 8, 14])
def test_weak_plane_classes_never_appear_where_the_plane_is_inactive(sample):
    res, _d = _classify(sample)
    inactive = ~res["weak_plane_active"]
    if inactive.any():
        assert not np.isin(res["mode_code"][inactive], [fc.CLASS_CODES["WT"],
                                                        fc.CLASS_CODES["WS"]]).any()


@pytest.mark.parametrize("sample", [1, 7, 8, 14])
def test_mixed_flag_is_secondary_only(sample):
    """A mixed flag must never replace the primary class."""
    res, _d = _classify(sample)
    flagged = res["mixed_flag"]
    if flagged.any():
        assert np.isin(res["mode_code"][flagged],
                       [fc.CLASS_CODES[c] for c in fc.CLASS_ORDER]).all()


def _wt_fraction(sample):
    res, d = _classify(sample)
    return fc.class_fractions(res["mode_code"], _core(d))["WT"]


@pytest.mark.parametrize("lo,hi,expected_hi", [(1, 7, 0.144), (8, 14, 0.151)])
def test_weak_plane_opening_grows_as_fabric_aligns_with_loading(lo, hi, expected_hi):
    """Physical expectation stated before assertion.

    At alpha_exp = 0 the foliation lies across the loading diameter, the planes
    are clamped in compression, and no weak-plane opening is admissible. At 90
    the planes are loading-parallel and opening is the delamination mechanism.
    Section 4 reports the WT area fraction as zero below 75 degrees and 0.144
    (gneiss) and 0.151 (schist) at 90, so those are the values locked here.
    """
    f0 = _wt_fraction(lo)
    f9 = _wt_fraction(hi)
    assert f0 == 0.0, f"weak planes are clamped at 0 deg, WT must vanish: {f0}"
    assert f9 > f0, f"WT did not grow with fabric alignment: {f0} -> {f9}"
    assert abs(f9 - expected_hi) < 5e-4, (
        f"WT at 90 deg drifted from the reported {expected_hi}: {f9}")


def test_class_fractions_sum_to_one():
    res, d = _classify(1)
    fr = fc.class_fractions(res["mode_code"], d["M"].astype(bool))
    total = sum(fr[c] for c in fc.CLASS_ORDER) + fr[fc.NO_FAILURE]
    assert abs(total - 1.0) < 1e-9


def test_no_cross_lithology_sample_leakage():
    with pytest.raises(AssertionError):
        lith.assert_notebook_lithology("augen_gneiss", [8])
    with pytest.raises(AssertionError):
        lith.assert_notebook_lithology("psammitic_schist", [1])
    lith.assert_notebook_lithology("augen_gneiss", [1, 7])
    lith.assert_notebook_lithology("psammitic_schist", [8, 14])
