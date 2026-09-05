"""A classification is only evidence if its criteria are calibrated for the test.

Each disc was loaded until it broke and the model is scaled to that load, so a
criterion calibrated for this stress state should be reached at a load factor of
one. That check is cheap, independent of the criterion, and has to be made
before a pointwise map is read: a criterion reached at a fifth of the observed
load will win the argmax everywhere and the map will describe the
miscalibration rather than the material.

**This file previously pinned the opposite finding.** On the cohesion values
carried before the strength table was corrected, linear Mohr-Coulomb was
reached between 0.22 and 1.37 with a median ``|ln lambda|`` of 0.74, fired
early on eleven of the fourteen specimens, and placed failure of the psammitic
schist at 45 degrees at 22% of the load it actually carried. Matrix tensile
failure was consequently empty everywhere, and the tests here asserted that
state so it could not drift unnoticed.

It did not drift; it was corrected. With the measured cohesion the two matrix
criteria are calibrated comparably, Mohr-Coulomb is the better of the two on
the median, and matrix tensile failure appears. These tests now pin *that*,
including the margin, so a regression back toward the old imbalance fails here
rather than in a figure.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import criterion_consistency as cc

FIELDS_AVAILABLE = bool(cc.ft.field_files())
needs_fields = pytest.mark.skipif(not FIELDS_AVAILABLE,
                                  reason="fields not exported yet")


@pytest.fixture(scope="module")
def table():
    return cc.consistency_table()


@needs_fields
def test_every_specimen_is_scaled_to_its_own_failure_load(table):
    """The premise: the model carries the load the disc actually broke at."""
    from tools import fabric_tractions as ft
    for path in ft.field_files():
        f = ft.load_field(path)
        ref = float(f["sigma_ref_MPa"])
        sid = int(f["sample_id"])
        T_m = cc.sp.specimen_strengths()[sid]["T_m"]
        assert abs(ref - T_m) / T_m < 0.01, (
            f"specimen {sid}: the field is scaled to {ref:.3f} MPa but the "
            f"measured strength is {T_m:.3f}. The load factors below are only "
            "interpretable while these agree."
        )


@needs_fields
def test_mohr_coulomb_is_calibrated_for_this_stress_state(table):
    """The corrected finding, asserted so a regression cannot pass quietly."""
    lam = table["lam_mohr_coulomb"].to_numpy(float)
    early = int((lam < 1.0).sum())
    assert early <= 7, (
        f"Mohr-Coulomb now fires before the observed failure load on {early} "
        "of fourteen. Above half the specimens is the signature of the "
        "miscalibration this file used to record; check the cohesion column "
        "of tensile_samples_data.csv before believing any four-class map."
    )
    assert lam.min() > 0.5, (
        f"the worst case is {lam.min():.2f}. Below 0.5 the envelope is "
        "predicting failure at half the load the disc carried."
    )
    assert cc.is_consistent(table, "lam_mohr_coulomb")


@needs_fields
def test_both_matrix_criteria_are_commensurate(table):
    """Neither criterion may be far better calibrated than the other.

    This is what makes the matrix-shear versus matrix-tensile ordering in the
    four-class maps mechanically meaningful rather than an artefact. The old
    failure mode was a factor of four between them.
    """
    s = cc.summary(table).set_index("criterion")
    tension = s.loc["maximum principal stress", "median_abs_log_lambda"]
    mc = s.loc["Mohr-Coulomb", "median_abs_log_lambda"]
    ratio = max(tension, mc) / max(min(tension, mc), 1e-12)
    assert ratio < 2.0, (
        f"the two matrix criteria differ by {ratio:.1f}x in median "
        f"|ln lambda| (tension {tension:.3f}, Mohr-Coulomb {mc:.3f}). One is "
        "being evaluated far outside the range it was measured in, and the "
        "class that wins the argmax will reflect that rather than the fabric."
    )
    assert cc.is_consistent(table, "lam_tension")


@needs_fields
def test_griffith_agrees_with_the_other_two(table):
    """Griffith accounts for the compressive sigma_3 the tensile criterion drops.

    It is kept as a third opinion. While all three sit in the same band the
    ordering of the matrix classes does not depend on which tensile form is
    chosen, which is the claim Section 4.6 makes.
    """
    v = table["lam_griffith"].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = cc.CONSISTENT_BAND
    assert ((v >= lo) & (v <= hi)).all(), (
        f"Griffith load factors {v.min():.2f} to {v.max():.2f} leave the "
        "consistency band; Section 4.6 states that it does not change the "
        "ordering, which assumes it is calibrated too."
    )


@needs_fields
def test_the_band_is_wide_enough_to_be_a_real_test(table):
    """A consistency band that nothing can fail is not a check."""
    lo, hi = cc.CONSISTENT_BAND
    assert lo <= 0.5 and hi >= 2.0, "the band has been narrowed into a tripwire"
    lam = table["lam_tension"].to_numpy(float)
    assert ((lam >= lo) & (lam <= hi)).all(), (
        "the tensile criterion now falls outside a band chosen to be wide; "
        "something has changed in the loading or the strengths"
    )
