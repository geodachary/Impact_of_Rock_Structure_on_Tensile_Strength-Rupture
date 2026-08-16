"""Corridor and lobe must be measurements, not impressions.

The strain-energy panels were described as showing continuous corridors in the
schist against localized hotspots in the gneiss, and a reviewer could not see the
distinction. Neither could the earlier quantification in the caption, whose
component counts could not be reproduced under any threshold or mask tried.

These tests pin the operational definitions and the values the manuscript quotes,
so the claim stands or falls on a number.
"""
from __future__ import annotations

import pytest

from tools import energy_localization as el
from tools import fabric_tractions as ft

FIELDS = ft.field_files()
needs_fields = pytest.mark.skipif(not FIELDS, reason="fields not exported yet")


@needs_fields
def test_gneiss_never_forms_a_corridor():
    t = el.localization_table()
    g = t[t.rock == "Augen gneiss"]
    assert not g.corridor.any(), "gneiss elevated region should stay in separate lobes"
    assert g.n_components.eq(2).all(), "diametral loading gives two lobes"
    assert g.elongation.max() < el.CORRIDOR_ELONGATION


@needs_fields
def test_schist_forms_a_corridor_only_at_high_fabric_angle():
    t = el.localization_table()
    s = t[t.rock == "Psammitic schist"]
    formed = set(s.loc[s.corridor, "angle_deg"])
    assert formed == {60.0, 75.0, 90.0}, f"corridor angles moved: {sorted(formed)}"
    assert s.loc[s.corridor, "n_components"].eq(1).all()
    assert s.loc[s.corridor, "elongation"].min() > 4.0
    # below 60 degrees the schist is indistinguishable from the gneiss
    assert s.loc[~s.corridor, "n_components"].eq(2).all()


@needs_fields
def test_the_distinction_needs_the_quartile_threshold():
    """At the top decile neither lithology connects; the claim must state its threshold."""
    for p in FIELDS:
        f = ft.load_field(p)
        assert el.component_shape(f, percentile=90.0)["n_components"] >= 2


@needs_fields
def test_summary_matches_the_caption():
    s = el.summary().set_index("rock")
    assert int(s.loc["Augen gneiss", "n_corridor"]) == 0
    assert int(s.loc["Psammitic schist", "n_corridor"]) == 3
    assert s.loc["Psammitic schist", "max_elongation"] == pytest.approx(4.84, abs=0.05)
