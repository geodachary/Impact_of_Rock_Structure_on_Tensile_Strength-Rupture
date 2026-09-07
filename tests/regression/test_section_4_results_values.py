"""Section 4's reported values, against the outputs they are drawn from.

Thirteen statements had drifted, and three of them were interpretation errors
rather than stale digits:

* the stress-tensor caption said the orientation spread was "slightly the
  larger in the schist" while the numbers printed beside it, gneiss first,
  showed the gneiss larger on both the spread and the median gradient;
* matrix shear was said to fall monotonically from 0 degrees when it rises to
  a maximum at 30 first, and to be overtaken at 60 degrees "in both
  lithologies" when only the schist is;
* the failed-fraction ordering named the schist as higher at 75 and 90
  degrees, where the gneiss is.

The rest were superseded numbers: the ATI R-squared and RMSE, the OLS slope
and intercept, the energy colour scale (also reversed between the rocks), the
Eshelby upper bound, the per-angle deviation medians, the principal-rotation
medians (which contradicted the figure caption in the same document), and six
values in the principal-stress permutation subsection.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TAB = REPO / "outputs" / "tables"
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
G, P = "Augen gneiss", "Psammitic schist"

pytestmark = pytest.mark.skipif(
    not (TEX.is_file() and (TAB / "ati_envelope_parameters.csv").is_file()),
    reason="manuscript or outputs absent")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def _t(name):
    return pd.read_csv(TAB / name)


# --- 4.2 ATI fit -----------------------------------------------------------

def test_the_ati_fit_metrics_agree_between_text_and_table(tex):
    a = _t("ati_envelope_parameters.csv").set_index("rock")
    assert a.loc[P, "R2"] == pytest.approx(0.915, abs=0.001)
    assert a.loc[G, "R2"] == pytest.approx(0.887, abs=0.001)
    assert a.loc[P, "RMSE"] == pytest.approx(0.589, abs=0.001)
    assert a.loc[G, "RMSE"] == pytest.approx(0.374, abs=0.001)
    # the body sentence quoted 0.912 and 0.622 while the table gave 0.915/0.589
    m = re.search(r"\$R\^\{2\} = ([\d.]+)\$ and \$([\d.]+)\$ and RMSE \$([\d.]+)\$ and \$([\d.]+)\$~MPa", tex)
    assert m, "the ATI accuracy sentence has changed shape"
    assert float(m.group(1)) == pytest.approx(a.loc[P, "R2"], abs=0.001)
    assert float(m.group(3)) == pytest.approx(a.loc[P, "RMSE"], abs=0.001)


def test_the_predicted_versus_actual_regression(tex):
    m = re.search(r"regression through all specimens has slope \$([\d.]+)\$ and intercept \$([\d.]+)\$~MPa", tex)
    assert m, "the OLS sentence has changed shape"
    assert float(m.group(1)) == pytest.approx(0.97, abs=0.005)
    assert float(m.group(2)) == pytest.approx(0.11, abs=0.005)


# --- 4.5 orientation statistics -------------------------------------------

def test_the_principal_rotation_medians_are_consistent_everywhere(tex):
    r = _t("principal_rotation.csv")
    rmax = {k: g.rotation_median_deg.max() for k, g in r.groupby("rock")}
    assert rmax[G] == pytest.approx(0.96, abs=0.01)
    assert rmax[P] == pytest.approx(2.42, abs=0.01)
    # the body once said 1.09 and 1.88 while the caption said 0.96 and 2.42
    assert "$1.09^\\circ$" not in tex and "$1.88^\\circ$" not in tex, (
        "the superseded principal-rotation medians are back, and they "
        "contradict the figure caption in the same document")
    assert tex.count("$0.96^\\circ$") >= 2 and tex.count("$2.42^\\circ$") >= 2


def test_the_orientation_spread_is_attributed_to_the_right_rock(tex):
    het = _t("principal_heterogeneity.csv")
    grd = _t("orientation_gradient.csv")
    hmean = {k: g.circ_sd_deg.mean() for k, g in het.groupby("rock")}
    gmed = {k: g.grad_median_deg_per_mm.mean() for k, g in grd.groupby("rock")}
    assert hmean[G] > hmean[P] and gmed[G] > gmed[P], (
        "the gneiss no longer leads on spread and median gradient; the caption "
        "names the larger lithology explicitly and would need regenerating")
    assert "marginally the larger in the gneiss" in tex, (
        "the caption has reverted to naming the schist, which the numbers "
        "printed beside it contradict")


def test_the_per_angle_deviation_medians(tex):
    f = _t("foliation_deviation_statistics.csv")
    g = f[f.rock == G].sort_values("angle_deg").delta_abs_median_deg.to_numpy()
    p = f[f.rock == P].sort_values("angle_deg").delta_abs_median_deg.to_numpy()
    want = [2.8, 14.0, 28.5, 43.8, 59.4, 74.8, 87.0]
    assert np.allclose(np.round(g, 1), want, atol=0.05)
    assert np.abs(p - g).max() == pytest.approx(2.1, abs=0.1)
    assert np.median(g[:6]) == pytest.approx(36.2, abs=0.1)
    assert np.median(p[:6]) == pytest.approx(38.1, abs=0.1)


def test_the_eshelby_bounds():
    e = _t("eshelby_bound.csv")
    mt = _t("eshelby_mori_tanaka.csv")
    assert e.concentration.min() == pytest.approx(1.13, abs=0.01)
    assert e.concentration.max() == pytest.approx(1.73, abs=0.01)
    assert mt.stiffening.min() == pytest.approx(1.04, abs=0.01)
    assert mt.stiffening.max() == pytest.approx(1.44, abs=0.01)


# --- 4.6 failure statistics ------------------------------------------------

def test_matrix_shear_is_not_monotonic_and_peaks_at_thirty():
    f = _t("fourclass_area_fractions.csv")
    for rock, at0, peak in ((G, 0.319, 0.479), (P, 0.372, 0.494)):
        g = f[f.rock == rock].sort_values("angle_deg")
        ms = g.set_index("angle_deg").MS
        assert ms.loc[0] == pytest.approx(at0, abs=0.002)
        assert ms.max() == pytest.approx(peak, abs=0.002)
        assert int(ms.idxmax()) == 30, "matrix shear no longer peaks at 30 deg"
        assert ms.loc[0] < ms.loc[30], (
            "matrix shear now falls from 0 deg; Section 4.6 describes a rise "
            "to an interior maximum")
        assert ms.loc[75] == 0.0 and ms.loc[90] == 0.0


def test_matrix_shear_is_overtaken_at_sixty_in_the_schist_only():
    f = _t("fourclass_area_fractions.csv").set_index(["rock", "angle_deg"])
    g60 = f.loc[(G, 60.0)]
    p60 = f.loc[(P, 60.0)]
    assert g60.MS > g60.WS, "the gneiss is now overtaken at 60 deg too"
    assert p60.WS > p60.MS, "the schist is no longer overtaken at 60 deg"


def test_the_failed_fraction_ordering_by_angle():
    s = _t("failure_statistics.csv")
    g = s[s.rock == G].set_index("angle_deg").p_fail
    p = s[s.rock == P].set_index("angle_deg").p_fail
    schist_higher = sorted(int(a) for a in g.index if p[a] > g[a])
    assert schist_higher == [0, 30, 45], (
        f"the schist now leads at {schist_higher}; Section 4.6 lists 0, 30 "
        "and 45 and the gneiss at the rest")


def test_the_mixed_mode_interior_area_fractions():
    """Interior-area denominator, distinct from the failed-point one."""
    from tools import export
    from tools import strain_partitioning as sp
    st = sp.specimen_strengths()
    out = {}
    for sid in range(1, 15):
        q = export.classification_panel(sid, st)
        m = np.asarray(q["mixed_flag"])[q["core_mask"]]
        out.setdefault(q["lithology"], {})[int(q["angle_deg"])] = 100.0 * m.mean()
    assert out[G][45] == pytest.approx(6.1, abs=0.1)
    assert out[P][60] == pytest.approx(7.6, abs=0.1)
    assert out[G][75] == pytest.approx(0.9, abs=0.1)
    assert out[P][75] == 0.0
    for rock in (G, P):
        assert out[rock][0] == 0.0 and out[rock][90] == 0.0


def test_the_open_fraction_at_ninety():
    t = _t("fabric_tractions.csv").set_index(["rock", "angle_deg"])
    assert t.loc[(G, 90.0)].open_fraction == pytest.approx(0.71, abs=0.005)
    assert t.loc[(P, 90.0)].open_fraction == pytest.approx(0.79, abs=0.005)
    assert t.loc[(G, 0.0)].open_fraction == 0.0


# --- 4.10 energy -----------------------------------------------------------

def test_the_energy_colour_scale_is_not_reversed(tex):
    c = _t("energy_colour_scale.csv").set_index("rock")
    g = c.loc[G, "u_95th_percentile_MPa"]
    p = c.loc[P, "u_95th_percentile_MPa"]
    assert g == pytest.approx(0.035, abs=0.001)
    assert p == pytest.approx(0.036, abs=0.001)
    assert p > g, "the schist no longer has the higher 95th percentile"
    assert "$0.042$~MPa" not in tex, "the superseded colour-scale value is back"


def test_the_energy_step_fractions():
    m = _t("fig20_GGc_profile_metrics.csv")
    assert m.n_below_Gc_energy.sum() == 0
    assert m.min_G_over_Gc_energy.min() == pytest.approx(1.00, abs=0.01)
    assert m.phys_fraction.mean() == pytest.approx(0.77, abs=0.01)


# --- 4.11 principal-stress permutations ------------------------------------

def _partition():
    d = _t("strain_partition_fourclass.csv")
    return d[d.status == "computed"]


def test_matrix_shear_dominates_the_energy_in_every_regime():
    d = _partition()
    for rock, lo, hi in ((G, 43, 46), (P, 46, 48)):
        m = d[d.lithology == rock].groupby("regime").MS_pct.mean()
        assert lo - 0.6 <= m.min() and m.max() <= hi + 0.6, (
            f"{rock} matrix-shear energy share is now {m.min():.1f}-{m.max():.1f}%")
        for reg, g in d[d.lithology == rock].groupby("regime"):
            shares = {c: g[c].mean() for c in ("WT_pct", "WS_pct", "MT_pct", "MS_pct")}
            assert max(shares, key=shares.get) == "MS_pct"


def test_the_regime_shifts_quoted_in_section_4_11():
    d = _partition()
    ws = {r: g.groupby("regime").WS_pct.mean() for r, g in d.groupby("lithology")}
    for rock in (G, P):
        span = ws[rock].max() - ws[rock].min()
        assert span == pytest.approx(0.19, abs=0.02), (
            f"{rock} sliding shifts by {span:.2f} pp; the text says 0.19 in both")
    def gain(rock, col):
        t = d[(d.lithology == rock) & (d.regime == "Thrust")][col].mean()
        e = d[(d.lithology == rock) & (d.regime == "Extensional")][col].mean()
        return e - t
    assert gain(G, "WT_pct") == pytest.approx(1.4, abs=0.1)
    assert gain(P, "WT_pct") == pytest.approx(2.8, abs=0.1)
    assert gain(P, "MT_pct") == pytest.approx(1.5, abs=0.1)
    assert gain(G, "MT_pct") == 0.0
    assert -gain(G, "below_threshold_pct") == pytest.approx(4.7, abs=0.1)
    assert -gain(P, "below_threshold_pct") == pytest.approx(5.7, abs=0.1)
    # "close to twice as responsive" rests on this ratio
    assert gain(P, "WT_pct") / gain(G, "WT_pct") == pytest.approx(2.1, abs=0.15)


def test_the_schist_matrix_tensile_energy_share():
    d = _partition()
    m = d[d.lithology == P].MT_pct
    assert m.mean() == pytest.approx(0.6, abs=0.1)
    assert m.max() == pytest.approx(7.1, abs=0.2)
    assert d[d.lithology == G].MT_pct.max() == 0.0
