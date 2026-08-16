"""Regression guard: the two lithologies must not share a partitioning result.

This exists because the published partitioning was computed by a routine that
never consulted the failure classifier, so adding the four-mechanism scheme
could not change it, and the summary figure was driven by hard-coded literals
that could not track a recomputation. Both failure modes are asserted against
here.
"""
import numpy as np
import pandas as pd
import pytest

from tools import strain_partitioning as sp
from tools import failure_classification as fc


@pytest.fixture(scope="module")
def partition():
    return sp.partition_all()


def test_all_specimens_and_regimes_computed(partition):
    assert len(partition) == 14 * 3
    assert (partition["status"] == "computed").all(), \
        partition.loc[partition.status != "computed", "reason"].tolist()


def test_rock_specific_strengths_actually_differ(partition):
    """The inputs that make the two rocks different must reach the calculation."""
    g = partition[partition.lithology == "Augen gneiss"]
    s = partition[partition.lithology == "Psammitic schist"]
    for col in ("T_m_MPa", "c_m_MPa", "phi_m_deg"):
        assert g[col].mean() != pytest.approx(s[col].mean(), abs=1e-9), \
            f"{col} is identical between lithologies - parameters are not rock-specific"


def test_partitioning_differs_between_lithologies(partition):
    """The headline assertion: gneiss and schist must not give the same split."""
    c = sp.lithology_contrast(partition)
    identical = c.index[c["identical"]].tolist()
    assert not identical, f"identical between lithologies: {identical}"


@pytest.mark.parametrize("col", ["WT_pct", "WS_pct", "MS_pct", "total_U_MJ_m3"])
def test_each_class_share_differs_materially(partition, col):
    c = sp.lithology_contrast(partition)
    assert abs(c.loc[col, "difference"]) > 1e-6, f"{col} is effectively identical"


def test_no_specimen_pair_across_lithologies_coincides(partition):
    """Angle-matched specimens from the two rocks must differ, not just the means."""
    ss = partition[partition.regime == "Strike-Slip"]
    for angle in sorted(ss.angle_deg.unique()):
        a = ss[(ss.angle_deg == angle) & (ss.lithology == "Augen gneiss")]
        b = ss[(ss.angle_deg == angle) & (ss.lithology == "Psammitic schist")]
        assert len(a) == 1 and len(b) == 1
        same = all(np.isclose(a.iloc[0][f"{c}_pct"], b.iloc[0][f"{c}_pct"], atol=1e-9)
                   for c in fc.CLASS_ORDER)
        assert not same, f"identical partitioning at alpha_exp = {angle} deg"


def test_class_shares_sum_to_one_hundred(partition):
    g = partition[partition.status == "computed"]
    tot = sum(g[f"{c}_pct"] for c in fc.CLASS_ORDER) + g["below_threshold_pct"]
    assert np.allclose(tot, 100.0, atol=1e-6)


def test_regime_offsets_are_the_documented_values():
    assert sp.REGIME_OFFSETS_MPA == {"Thrust": +0.50, "Strike-Slip": 0.00,
                                     "Extensional": -0.50}


def test_two_rock_figure_is_data_driven(tmp_path, partition):
    """The figure must follow the data, not a pasted snapshot."""
    import matplotlib
    matplotlib.use("Agg")
    from tools import plotting as tp
    out = tp.strain_partition_two_rocks(partition, tmp_path / "sp", formats=("png",))
    assert out and out[0].exists()
    # perturbing the data must change the rendered figure
    d2 = partition.copy()
    d2.loc[d2.lithology == "Psammitic schist", "MS_pct"] *= 0.5
    out2 = tp.strain_partition_two_rocks(d2, tmp_path / "sp2", formats=("png",))
    assert out[0].read_bytes() != out2[0].read_bytes(), \
        "figure did not change when the data changed - it is not data-driven"
