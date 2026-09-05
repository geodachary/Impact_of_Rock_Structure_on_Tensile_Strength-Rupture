"""The specimen table the pipeline reads must match the one that generated it.

``data_mean.ipynb`` averages the replicate measurements in
``selected_all_samples.csv`` and writes ``tensile_samples_data.xlsx``. Every
other notebook and script reads ``tensile_samples_data.csv``. Nothing converts
one to the other, so the CSV is maintained by hand from the spreadsheet, and
the two are free to drift.

They are the single most consequential file in the repository. Every stress
field, every classification and every published number descends from the
elastic constants, strengths and geometry in that table, and a change made to
one copy and not the other would move all of it with nothing to show why.

This is the same defect that produced the hardcoded ``M=24``, the two output
roots and the stale notebook cells: a second copy of something with nothing
keeping it in step. Here the check is cheap, so it is asserted rather than
trusted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "tensile_samples_data.csv"
XLSX = REPO / "tensile_samples_data.xlsx"

pytestmark = pytest.mark.skipif(
    not (CSV.is_file() and XLSX.is_file()),
    reason="specimen table or its spreadsheet is absent")


@pytest.fixture(scope="module")
def pair():
    return pd.read_csv(CSV), pd.read_excel(XLSX)


def test_the_two_copies_have_the_same_shape_and_columns(pair):
    csv, xlsx = pair
    assert list(csv.columns) == list(xlsx.columns), (
        "the CSV the pipeline reads and the spreadsheet data_mean.ipynb writes "
        "no longer have the same columns")
    assert csv.shape == xlsx.shape == (14, len(csv.columns))


def test_every_value_agrees(pair):
    csv, xlsx = pair
    mismatched = {}
    for c in csv.columns:
        a, b = csv[c], xlsx[c]
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            x, y = a.to_numpy(float), b.to_numpy(float)
            if not np.allclose(x, y, equal_nan=True, rtol=1e-9, atol=0.0):
                mismatched[c] = float(np.nanmax(np.abs(x - y)))
        elif not (a.astype(str) == b.astype(str)).all():
            mismatched[c] = "text differs"
    assert not mismatched, (
        f"the specimen table and its source spreadsheet disagree: {mismatched}. "
        "Re-run data_mean.ipynb and export the CSV from it; do not edit one "
        "copy alone, because every published number descends from this table."
    )


def test_the_columns_the_pipeline_depends_on_are_present(pair):
    """Named explicitly, so a rename upstream fails here and not mid-solve."""
    csv, _ = pair
    required = ["Rock_type", "Angle", "Diameter_mm", "Thickness_mm",
                "Load_(KN)", "Tensile_strength_Mpa", "Modulus_of_Elasticity",
                "Poisson_Ratio", "Shear_Modulus", "UCS_(Mpa)", "Cohesion",
                "Friction_Angle"]
    missing = [c for c in required if c not in csv.columns]
    assert not missing, f"the specimen table has lost columns the code reads: {missing}"
