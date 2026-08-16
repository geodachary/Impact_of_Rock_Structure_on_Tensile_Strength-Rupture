"""Lock the mirror-symmetry result reported for the displacement profiles.

Describing the displacement field as asymmetric is only defensible against a
stated criterion. These tests pin both the criterion's behaviour on synthetic
fields and the measured values quoted in the figure caption.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import displacement_symmetry as ds


@pytest.fixture(scope="module")
def df():
    return ds.all_specimens()


def test_index_is_zero_for_a_perfectly_mirrored_field():
    """A field built to satisfy the expectation must score zero."""
    n = 41
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= 1.0
    u = X * (1 - X ** 2 - Y ** 2)          # antisymmetric in x
    v = (1 - X ** 2 - Y ** 2)              # symmetric in x
    u_anti, _ = ds._mirror_index(u, X, mask)
    _, v_sym = ds._mirror_index(v, X, mask)
    assert u_anti == pytest.approx(0.0, abs=1e-12)
    assert v_sym == pytest.approx(0.0, abs=1e-12)


def test_index_is_one_for_a_maximally_wrong_field():
    """Swapping the expected parity must saturate the index."""
    n = 41
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= 1.0
    sym = (1 - X ** 2 - Y ** 2)            # symmetric where antisymmetry expected
    u_anti, _ = ds._mirror_index(sym, X, mask)
    assert u_anti == pytest.approx(1.0, rel=1e-9)


def test_indices_are_bounded(df):
    g = df[df.status == "computed"]
    for col in ("u_antisymmetry_index", "v_symmetry_index"):
        assert (g[col] >= 0).all() and (g[col] <= 1).all()


def test_thirteen_specimens_are_cached(df):
    # Specimen 14 used to be missing because the cache key was built from the
    # specimen number alone and its entry had never been written. With the key
    # carrying the material fingerprint the solution is regenerated, so all
    # fourteen are present; a drop back to thirteen means a cache went stale.
    assert (df.status == "computed").sum() == 14
    blocked = df[df.status != "computed"]
    assert list(blocked["sample"]) == [], "every specimen should now be cached"


def test_end_members_are_approximately_symmetric(df):
    """The reviewer's reading of the end-member panels must hold."""
    g = df[df.status == "computed"]
    ends = g[g.angle_deg.isin([0, 90])]
    assert (ends.max_index < ds.VISIBLE_ASYMMETRY).all(), \
        f"an end member exceeded the threshold:\n{ends[['sample', 'angle_deg', 'max_index']]}"


def test_intermediate_angles_are_markedly_asymmetric(df):
    g = df[df.status == "computed"]
    mid = g[g.angle_deg.isin([45, 60])]
    assert (mid.max_index > 0.5).all()


def test_quoted_caption_values(df):
    g = df[df.status == "computed"].set_index("sample")
    assert g.loc[1].max_index == pytest.approx(0.046, abs=5e-4)
    assert g.loc[4].max_index == pytest.approx(0.718, abs=5e-4)
    assert g.loc[7].max_index == pytest.approx(0.066, abs=5e-4)
    assert g.loc[8].max_index == pytest.approx(0.057, abs=5e-4)
    assert g.loc[11].max_index == pytest.approx(0.928, abs=5e-4)


def test_schist_exceeds_gneiss_at_the_same_angle(df):
    """The lithology contrast asserted in the text."""
    g = df[df.status == "computed"].set_index("sample")
    assert g.loc[11].max_index > g.loc[4].max_index      # 45 deg, schist vs gneiss


def test_summary_supports_the_wording(df):
    s = ds.summary(df)
    assert s["n_computed"] == 14 and s["n_blocked"] == 0
    assert s["n_visibly_asymmetric"] == 9
    assert s["supports_asymmetric_description"] is True
