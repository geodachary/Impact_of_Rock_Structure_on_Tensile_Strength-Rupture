"""The stress-sign convention must be declared, never guessed.

``failure_mode_map_4class`` accepts ``stress_sign_mode="auto"``, which calls a
heuristic to decide whether the caller's stresses are tension-positive or
compression-positive. That heuristic used to read a Brazilian disc field as
compression-positive and negate it:

    if mn < 0 and mx > 0 and med > 0:      # the rule that has been replaced
        return "compression_positive"

A Brazilian field satisfies exactly that. ``sigma_1`` is the *maximum*
principal stress, which over most of the disc interior is the tensile
horizontal stress, so the median is positive while the compressive zones under
the platens supply the negatives. Measured on the exported fields the median is
about +6.5 MPa with 98% of points positive, and the detector flips the sign of
every component.

After the flip ``R_MT = max(sigma_1, 0) / T_m`` is identically zero, so matrix
tensile failure can never win the argmax and the class vanishes from the maps.

Every unit test of the classifier passes ``stress_sign_mode="tension_positive"``
explicitly, so the heuristic was never exercised on real data and nothing caught
this. The rule now compares ``s1`` against the mean normal stress, which is a
property of the convention rather than of the loading, and the production call
sites additionally pin the convention explicitly. These tests hold both in
place and record why the old rule failed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

#: Modules that hand real solver fields to the classifier.
CALL_SITES = (
    "tools/analysis/crack_energy_suite.py",
    "tools/analysis/crack_path_suite.py",
    "tools/sensitivity_sweeps.py",
)


@pytest.mark.parametrize("relpath", CALL_SITES)
def test_no_production_call_site_guesses_the_convention(relpath):
    src = (REPO / relpath).read_text(encoding="utf-8")
    assert 'stress_sign_mode="auto"' not in src, (
        f"{relpath} lets the detector guess. It feeds tension-positive "
        "Lekhnitskii fields, which the detector misreads as "
        "compression-positive and negates."
    )
    assert 'stress_sign_mode="tension_positive"' in src, (
        f"{relpath} no longer declares the convention at all"
    )


def test_the_detector_reads_a_brazilian_field_correctly():
    """The repaired rule, on the field shape that defeated the old one.

    The old rule looked at the median of ``s1`` and called anything positive
    with some negatives compression-positive, which is exactly a Brazilian
    disc. The replacement tests ``s1 >= (sxx + syy) / 2``, which holds
    everywhere in a tension-positive field and fails in a compression-positive
    one, and is a property of the convention rather than of the loading.
    """
    from tools.failure_mapping_helpers import _detect_stress_sign_mode

    from tools import fabric_tractions as ft

    for path in ft.field_files():
        f = ft.load_field(path)
        m = ft.core_mask(f)
        s1, sxx, syy = (np.asarray(f[k])[m] for k in ("s1", "sxx", "syy"))
        assert _detect_stress_sign_mode(s1, sxx, syy) == "tension_positive", (
            f"{path.name}: a solver field read as compression-positive"
        )
        # and the same field negated must be recognised as the other convention
        assert _detect_stress_sign_mode(-s1, -sxx, -syy) == "compression_positive"


def test_the_detector_cannot_be_fooled_by_a_positive_median():
    """The specific shape of the old failure, pinned so it cannot return."""
    from tools.failure_mapping_helpers import _detect_stress_sign_mode

    rng = np.random.default_rng(0)
    # Tension-positive: s1 is the algebraically largest principal stress.
    sxx = np.concatenate([rng.uniform(1.0, 12.0, 950), rng.uniform(-12.0, -1.0, 50)])
    syy = sxx - rng.uniform(5.0, 30.0, sxx.size)
    s1 = np.maximum(sxx, syy)
    assert float(np.median(s1)) > 0 and float(np.min(s1)) < 0, "test setup"
    assert _detect_stress_sign_mode(s1, sxx, syy) == "tension_positive"


def test_flipping_the_sign_would_extinguish_matrix_tensile():
    """Why the misread matters: R_MT collapses to zero under the flip."""
    T_m = 10.47
    s1 = np.array([12.68, 8.0, 6.5, 3.0, -4.0])

    r_mt_correct = np.maximum(s1, 0.0) / T_m
    r_mt_flipped = np.maximum(-s1, 0.0) / T_m

    assert (r_mt_correct > 0).sum() == 4
    assert (r_mt_flipped > 0).sum() == 1, (
        "under the flip only the one genuinely compressive point registers any "
        "tensile utility, which is the mechanism by which MT disappears"
    )


def test_the_published_classifier_declares_its_convention():
    """``failure_classification.classify`` must not acquire a detector."""
    import inspect

    from tools import failure_classification as fc

    default = inspect.signature(fc.classify).parameters["stress_convention"].default
    assert default == "tension_positive", (
        "the published four-class path declares tension-positive explicitly; "
        f"it now defaults to {default!r}"
    )
    src = inspect.getsource(fc)
    assert "_detect_stress_sign_mode" not in src, (
        "the published path has acquired the sign-guessing heuristic"
    )
