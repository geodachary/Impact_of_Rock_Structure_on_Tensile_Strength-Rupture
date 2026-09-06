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
    """The reviewer's reading of the end-member panels must hold.

    Three of the four end members sit below the visibility threshold. The
    schist at 0 degrees is marginally above it, at 0.114 against 0.10, so the
    claim is that end-member fields are close to symmetric rather than
    strictly below the threshold.
    """
    g = df[df.status == "computed"]
    ends = g[g.angle_deg.isin([0, 90])]
    assert (ends.max_index < 0.12).all(), \
        f"an end member is no longer close to symmetric:\n{ends[['sample', 'angle_deg', 'max_index']]}"
    assert int((ends.max_index < ds.VISIBLE_ASYMMETRY).sum()) == 3, (
        "the number of end members below the visibility threshold has moved")


def test_intermediate_angles_are_markedly_asymmetric(df):
    """Departures are largest away from the end members, with one exception.

    The schist at 45 degrees is the exception: it is close to symmetric, at
    0.036, while its own neighbours at 30 and 60 degrees are 0.344 and 0.341.
    It is recorded rather than smoothed over.
    """
    g = df[df.status == "computed"]
    mid = g[g.angle_deg.isin([15, 30, 45, 60, 75])]
    assert (mid.max_index > ds.VISIBLE_ASYMMETRY).sum() >= 8, (
        "the intermediate angles are no longer mostly asymmetric")
    odd = g[(g.lithology == "Psammitic schist") & (g.angle_deg == 45)]
    assert float(odd.max_index.iloc[0]) < 0.05, (
        "the schist 45 degree case is no longer the near-symmetric outlier "
        "the text records")


def test_quoted_caption_values(df):
    g = df[df.status == "computed"].set_index("sample")
    assert g.loc[1].max_index == pytest.approx(0.0623, abs=5e-4)
    assert g.loc[4].max_index == pytest.approx(0.2049, abs=5e-4)
    assert g.loc[7].max_index == pytest.approx(0.0731, abs=5e-4)
    assert g.loc[8].max_index == pytest.approx(0.1143, abs=5e-4)
    assert g.loc[11].max_index == pytest.approx(0.0357, abs=5e-4)


def test_the_schist_departs_further_over_the_series(df):
    """The lithology contrast, stated over the series rather than at 45 degrees.

    At 45 degrees the schist is now the more symmetric of the two, which is
    the one specimen that runs against the pattern; taken over all seven
    orientations the schist still departs further.
    """
    g = df[df.status == "computed"]
    med = g.groupby("lithology").max_index.median()
    assert med["Psammitic schist"] > med["Augen gneiss"], (
        "the schist no longer departs further from mirror symmetry overall")


def test_summary_supports_the_wording(df):
    s = ds.summary(df)
    assert s["n_computed"] == 14 and s["n_blocked"] == 0
    # Re-locked at M = 48: the gneiss at 15 deg crossed the 0.10 visibility
    # threshold, so ten of fourteen now exceed it rather than nine.
    assert s["n_visibly_asymmetric"] == 10
    assert s["supports_asymmetric_description"] is True
