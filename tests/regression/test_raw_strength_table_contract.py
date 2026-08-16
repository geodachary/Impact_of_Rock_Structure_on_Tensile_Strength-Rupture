"""The experimental strength table must stay readable however it is regenerated.

``tensile_samples_data.csv`` is not hand-maintained: it is the per-angle mean of
the replicate measurements in ``selected_all_samples.csv``, rebuilt by
``data_mean.ipynb``. A rebuild has already changed the column set once — the
serial-number column disappeared and the lithology spelling changed — which
silently broke every downstream strength lookup.

These tests pin the contract the rest of the code depends on, so the next
rebuild fails here with a clear message instead of somewhere deep in a
classifier run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import lithology as lith
from tools import strain_partitioning as sp

REQUIRED = ["Rock_type", "Angle", "Tensile_strength_Mpa", "Cohesion", "Friction_Angle"]


@pytest.fixture(scope="module")
def raw():
    df = pd.read_csv(lith.REPO_ROOT / "tensile_samples_data.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def test_required_columns_present(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    assert not missing, f"regenerated table lost {missing}; update data_mean.ipynb"


def test_one_row_per_lithology_and_angle(raw):
    assert len(raw) == 14, f"expected 14 specimens, found {len(raw)}"
    pairs = raw.groupby([raw.Rock_type.str.strip().str.lower(),
                         raw.Angle.round().astype(int)]).size()
    assert (pairs == 1).all(), f"duplicate lithology/angle rows: {pairs[pairs > 1]}"


def test_rock_type_spellings_are_resolvable(raw):
    for name in raw.Rock_type.unique():
        token = str(name).strip().lower().replace(" ", "_")
        token = lith.LEGACY_TOKENS.get(token, token)
        lith.get(token)          # raises if the spelling is unknown


def test_angles_cover_the_programme(raw):
    for name, grp in raw.groupby(raw.Rock_type.str.strip().str.lower()):
        angles = sorted(int(round(a)) for a in grp.Angle)
        assert angles == list(lith.ANGLES_DEG), f"{name}: angles {angles}"


def test_all_fourteen_specimens_resolve():
    """The lookup the classifier and partitioning both depend on."""
    s = sp.specimen_strengths()
    assert sorted(s) == list(range(1, 15))
    for sid, v in s.items():
        assert v["T_m"] > 0, f"sample {sid} has non-positive tensile strength"
        assert v["c_m"] > 0, f"sample {sid} has non-positive cohesion"
        assert 0 < v["phi_m"] < np.pi / 2, f"sample {sid} friction angle out of range"


def test_strengths_match_the_replicate_means():
    """The table must remain the mean of selected_all_samples.csv, not drift from it."""
    src = lith.REPO_ROOT / "selected_all_samples.csv"
    if not src.exists():
        pytest.skip("selected_all_samples.csv not present")
    rep = pd.read_csv(src, encoding="utf-8-sig")
    rep.columns = [c.strip() for c in rep.columns]
    means = (rep.groupby([rep.Rock_type.str.strip().str.lower(),
                          rep.Angle.round().astype(int)])
                .Tensile_strength_Mpa.mean())
    s = sp.specimen_strengths()
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        for angle in lit.angles_deg:
            key = (lit.display_name.lower(), angle)
            if key not in means.index:
                continue
            sid = lit.sample_for_angle(angle)
            assert s[sid]["T_m"] == pytest.approx(means[key], rel=1e-6), (
                f"{lit.display_name} {angle} deg: table {s[sid]['T_m']:.4f} "
                f"vs replicate mean {means[key]:.4f}")


def test_specimen_ids_follow_lithology_order():
    """Gneiss is 1-7 and schist 8-14, in ascending angle order."""
    assert lith.AUGEN_GNEISS.sample_ids == (1, 2, 3, 4, 5, 6, 7)
    assert lith.PSAMMITIC_SCHIST.sample_ids == (8, 9, 10, 11, 12, 13, 14)
    for lit in (lith.AUGEN_GNEISS, lith.PSAMMITIC_SCHIST):
        for angle in lit.angles_deg:
            sid = lit.sample_for_angle(angle)
            assert lit.angle_for_sample(sid) == angle
