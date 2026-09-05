"""The four-class area fractions must mean the same thing wherever they appear.

Two producers write them and the manuscript quotes both:

* ``scripts/make_failure_statistics_figures.class_fractions`` writes
  ``outputs/tables/fourclass_area_fractions.csv``
* ``tools.export.classification_panel`` supplies the per-lithology tables and
  the seven-panel classification figure that Section 4.6 is written from

They disagreed. The first takes the fractions over ``CORE_FRAC`` = 0.85R, the
disc interior every other field statistic in the paper reports; the second took
them over the mask stored in the field archive, ``r <= 0.985R``, which is an
anti-aliasing guard on the outermost ring of grid pixels rather than a physical
region. The gap is not cosmetic: at 0.985R the matrix-shear share of the augen
gneiss at 0 degrees reads 0.197 and at 0.85R it reads 0.273, and the ring
between 0.97R and the rim is where the boundary-traction fit carries its
largest residual, so including it attributes fit error to the material.

The manuscript quoted the 0.985R numbers while citing the 0.85R interior in the
surrounding sentences. Both producers now use ``CORE_FRAC``, and this test
holds them together, because nothing else would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import export
from tools import fabric_tractions as ft
from tools import lithology as lith
from tools import strain_partitioning as sp

CLASSES = ("WT", "WS", "MT", "MS")

_have_fields = all(lith.field_cache_path(s).exists() for s in range(1, 15))
needs_fields = pytest.mark.skipif(not _have_fields, reason="fields not exported yet")


@pytest.fixture(scope="module")
def strengths():
    return sp.specimen_strengths()


@needs_fields
def test_the_panel_reports_fractions_over_the_analysis_interior(strengths):
    """Not over the stored mask, which is a grid guard and not a region."""
    panel = export.classification_panel(1, strengths)
    assert "core_mask" in panel, "the panel no longer exposes the interior it used"
    assert panel["core_frac"] == pytest.approx(ft.CORE_FRAC)
    assert int(panel["core_mask"].sum()) < int(panel["mask"].sum()), (
        "the reporting interior is not smaller than the stored mask, so one of "
        "them is not what it claims to be")


@needs_fields
def test_both_producers_give_the_same_fractions(strengths):
    """The invariant: one quantity, one value, whichever code path reaches it."""
    ref_path = lith.REPO_ROOT / "outputs/tables/fourclass_area_fractions.csv"
    if not ref_path.exists():
        pytest.skip("fourclass_area_fractions.csv not written yet")
    ref = pd.read_csv(ref_path)

    worst, where = 0.0, None
    for sid in range(1, 15):
        p = export.classification_panel(sid, strengths)
        m = ref[(ref.rock == p["lithology"])
                & (np.isclose(ref.angle_deg, p["angle_deg"]))]
        assert len(m) == 1, f"specimen {sid} not matched in the reference table"
        m = m.iloc[0]
        for c in CLASSES:
            d = abs(float(p["fractions"][c]) - float(m[c]))
            if d > worst:
                worst, where = d, f"specimen {sid}, class {c}"
    assert worst < 1e-9, (
        f"the two four-class producers disagree by {worst:.6f} at {where}. "
        "They are the same quantity over the same interior; if one has moved "
        "to a different mask, Section 4.6 is quoting one of them and citing "
        "the other."
    )


@needs_fields
def test_the_stored_mask_would_have_given_a_different_answer(strengths):
    """Guard the guard: prove the two masks are not accidentally identical.

    If they ever coincide the test above passes for the wrong reason, so the
    difference that motivated this file is asserted to still exist.
    """
    d = np.load(lith.field_cache_path(1), allow_pickle=True)
    from tools import failure_classification as fc
    p = strengths[1]
    res = fc.classify(
        d["sxx"], d["syy"], d["txy"], alpha_f=float(d["alpha_wp_line_rad"]),
        T_wp=sp.WEAK_T_RATIO * p["T_m"], c_wp=sp.WEAK_C_RATIO * p["c_m"],
        phi_wp=p["phi_m"], T_m=p["T_m"], c_m=p["c_m"], phi_m=p["phi_m"],
        weak_plane_weight=d["wp_weight"], activation_floor=sp.ACTIVATION_FLOOR,
        threshold=sp.THRESHOLD, eta_mix=sp.ETA_MIX, n_theta=sp.N_THETA)
    stored = d["M"].astype(bool)
    radius = np.hypot(np.asarray(d["X"], float), np.asarray(d["Y"], float))
    core = stored & (radius <= ft.CORE_FRAC * float(d["R_m"]))

    f_stored = fc.class_fractions(res["mode_code"], stored)
    f_core = fc.class_fractions(res["mode_code"], core)
    spread = max(abs(f_stored[c] - f_core[c]) for c in CLASSES)
    assert spread > 0.01, (
        "the stored mask and the analysis interior now give the same "
        "fractions, so the test above no longer proves anything")
