"""The adopted softening factors must be reproducible and must match the text.

These values previously existed only as printed output from a notebook cell.
Nothing checked them, and the manuscript drifted to quoting $0.80$ and $0.46$
against a code that produced $0.850$ and $0.750$, with a sweep range stated as
$[0.1, 0.7]$ against an actual $[0.05, 0.90]$. Every one of those errors is the
kind a test catches for free.

The last test reads the numbers back out of the manuscript, so the text cannot
drift away from the code again without a failure.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tools import softening_selection as ss

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript/manscript_revision_001.tex"

ADOPTED = {"Augen gneiss": 0.85, "Psammitic schist": 0.75}

has_fields = (REPO / ss.FIELD_DIR).exists() and any(
    (REPO / ss.FIELD_DIR).glob("sample_*_full_fields.npz"))
needs_fields = pytest.mark.skipif(not has_fields, reason="fields not exported yet")


# ----------------------------------------------------------------- selection rule
def test_rule_takes_the_smallest_near_peak_not_the_argmax():
    """A saturating curve must not put the answer wherever the sweep stopped."""
    grid = np.linspace(0.0, 0.9, 10)
    curve = np.minimum(grid / 0.5, 1.0)            # rises then flat at 1
    idx, k, peak, at_upper = ss.choose_smallest_near_peak(grid, curve, rel_tol=0.02)
    assert peak == pytest.approx(1.0)
    assert k < 0.9, "selection drifted to the upper bound of a saturated curve"
    assert not at_upper
    assert curve[idx] >= 0.98 * peak


def test_tensile_cap_defaults_to_the_tied_value():
    """Omitting kT_max must be identical to passing 0.35 * kC_max explicitly."""
    rng = np.random.default_rng(1)
    a = dict(s1=rng.uniform(-4, 4, (10, 10)), U=rng.uniform(0, 1, (10, 10)),
             w=rng.uniform(0, 1, (10, 10)))
    s3 = a["s1"] - rng.uniform(0, 4, (10, 10))
    T0 = np.full_like(a["s1"], 3.0); C0 = np.full_like(a["s1"], 8.0)
    phi = np.full_like(a["s1"], np.deg2rad(30.0))
    k = 0.6
    common = (a["s1"], s3, a["U"], T0, C0, phi, a["w"], a["w"], a["w"])
    tied = ss.classify_local_damage(*common, kC_max=k)
    explicit = ss.classify_local_damage(*common, kC_max=k, kT_max=ss.KT_RATIO * k)
    assert np.array_equal(tied, explicit)
    # and a different tensile cap must actually change something
    other = ss.classify_local_damage(*common, kC_max=k, kT_max=0.9)
    assert not np.array_equal(tied, other)


def test_sweep_section_constants_match_the_module():
    """The sweep's constants must not drift from the module's.

    The sweep used to live in a notebook cell; it is now
    ``tools/analysis/kmax_sweep.py``, run once for both rocks from the general
    notebook. The constants are still written out there, so they are still
    capable of drifting, and this still checks them.
    """
    src = (REPO / "tools" / "analysis" / "kmax_sweep.py").read_text(encoding="utf-8")
    assert "kmax_list = np.linspace" in src, "k_max sweep grid not found"
    for name, value in (("KT_RATIO", ss.KT_RATIO), ("P_U", ss.P_U),
                        ("P_WP", ss.P_WP), ("P_S", ss.P_S), ("REL_TOL", ss.REL_TOL)):
        m = re.search(rf"^\s*{name}\s*=\s*([0-9.]+)", src, re.M)
        assert m, f"{name} not found in the sweep section"
        assert float(m.group(1)) == pytest.approx(value), (
            f"{name}: notebook {m.group(1)} against module {value}")
    m = re.search(r"kmax_list = np\.linspace\(([0-9.]+),\s*([0-9.]+),\s*(\d+)\)", src)
    assert m, "sweep grid not found"
    assert float(m.group(1)) == pytest.approx(ss.DEFAULT_GRID.min())
    assert float(m.group(2)) == pytest.approx(ss.DEFAULT_GRID.max())
    assert int(m.group(3)) == len(ss.DEFAULT_GRID)


@pytest.mark.parametrize("bad", [-0.01, 0.96, 1.5])
def test_out_of_range_caps_are_refused(bad):
    z = np.zeros((4, 4))
    with pytest.raises(ValueError):
        ss.classify_local_damage(z, z, z, z + 1.0, z + 1.0, z + 0.5,
                                 z, z, z, kC_max=bad)


def test_zero_softening_leaves_strength_untouched():
    """With no damage the classifier must reduce to the undamaged criteria."""
    rng = np.random.default_rng(0)
    s1 = rng.uniform(-5, 5, (12, 12))
    s3 = s1 - rng.uniform(0, 5, (12, 12))
    U = rng.uniform(0, 1, (12, 12))
    T0 = np.full_like(s1, 3.0)
    C0 = np.full_like(s1, 8.0)
    phi = np.full_like(s1, np.deg2rad(30.0))
    w = rng.uniform(0, 1, (12, 12))
    modes = ss.classify_local_damage(s1, s3, U, T0, C0, phi, w, w, w, kC_max=0.0)
    # no damage means tensile failure is exactly s1 >= T0
    tens = np.isin(modes, ["tensile", "mixed"])
    assert np.array_equal(tens, np.maximum(s1, s3) >= T0)


# ------------------------------------------------------------------- real values
@needs_fields
def test_adopted_values_reproduce():
    df = ss.select_all()
    for _, r in df.iterrows():
        assert r.kC_max == pytest.approx(ADOPTED[r.lithology], abs=1e-6)
        assert r.kT_max == pytest.approx(ss.KT_RATIO * ADOPTED[r.lithology], abs=1e-6)


@needs_fields
def test_selection_lies_inside_the_swept_range():
    """The complaint that started this: an adopted value outside its own sweep."""
    lo, hi = ss.DEFAULT_GRID.min(), ss.DEFAULT_GRID.max()
    df = ss.select_all()
    for _, r in df.iterrows():
        assert lo <= r.kC_max <= hi, f"{r.lithology}: {r.kC_max} outside [{lo}, {hi}]"
        assert not r.hit_upper_bound, (
            f"{r.lithology}: curve still near peak at the upper bound; "
            "the sweep is too narrow to justify the value")


@needs_fields
def test_manuscript_quotes_the_computed_values():
    """The text must not drift away from the code a second time."""
    tex = TEX.read_text(encoding="utf-8")
    for lit, k in ADOPTED.items():
        assert f"${k:.2f}$" in tex, f"adopted {lit} cap {k:.2f} not quoted in the manuscript"
    lo, hi = ss.DEFAULT_GRID.min(), ss.DEFAULT_GRID.max()
    assert re.search(rf"\[{lo:.2f},\s*{hi:.2f}\]", tex), "swept range not stated as computed"
    assert "0.46" not in tex.split("\\section{Supplementary")[0], "stale cap 0.46 still present"
