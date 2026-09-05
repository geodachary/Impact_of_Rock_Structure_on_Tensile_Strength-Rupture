"""Table 1 is the only place some measurements reach the paper.

Mass, density, bulk density, porosity, area and the two strain columns are
read by no module in ``tools/`` or ``scripts/``. They are reported in Table 1
and nowhere else. That makes them invisible to every other guard in this
suite: the pipeline can be re-run from scratch, all 547 tests can pass, and
Table 1 can still disagree with the spreadsheet, because nothing computes
anything from it.

That is not hypothetical. A revision of the density and porosity columns left
Table 1 quoting 2.79 and 2.68 g/cm3 against measured values of 2.78 and 2.85,
and porosity 1.06 and 1.56 against 1.65 and 1.17. The porosity ordering
reversed with it: the schist had been the less porous of the two and became
the more porous, so the sentence describing the contrast was not merely out of
date but backwards.

This test reads the replicate file and asserts the table against it, which is
the only way these values can be checked at all.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"
RAW = REPO / "selected_all_samples.csv"

pytestmark = pytest.mark.skipif(
    not (TEX.is_file() and RAW.is_file()),
    reason="manuscript or replicate data not present")

#: Table 1 row label -> (column in selected_all_samples.csv, decimals quoted).
#: Order is schist then gneiss, as the table columns run.
ROWS = {
    "Mean density (g/cm$^{3}$)": ("Density", 2),
    "Mean porosity (\\%)": ("Porosity_percent", 2),
    "Mean UCS (MPa)": ("UCS_(Mpa)", 2),
    "Mean Young's modulus (GPa)": ("Modulus_of_Elasticity", 2),
    "Mean Poisson's ratio": ("Poisson_Ratio", 2),
}


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def means():
    d = pd.read_csv(RAW)
    return {rk: g for rk, g in d.groupby("Rock_type")}


def _row_values(tex, label):
    """The two numbers on the Table 1 row carrying this label."""
    m = re.search(re.escape(label) + r"\s*&(.+?)\\\\", tex, re.S)
    assert m, f"Table 1 has no row labelled {label!r}"
    return re.findall(r"(\d+\.\d+)", m.group(1))


@pytest.mark.parametrize("label", sorted(ROWS))
def test_table1_row_matches_the_replicate_means(tex, means, label):
    col, dec = ROWS[label]
    quoted = _row_values(tex, label)
    assert len(quoted) == 2, f"{label}: expected two values, found {quoted}"
    for got, rock in zip(quoted, ("Psammitic schist", "Augen gneiss")):
        want = f"{means[rock][col].mean():.{dec}f}"
        assert got == want, (
            f"Table 1, {label}, {rock}: the paper says {got} and the "
            f"measurements give {want}. Nothing computes from this column, so "
            "only this test can catch it."
        )


def test_the_porosity_sentence_matches_the_table(tex, means):
    """The prose states the contrast; it must not invert it.

    Quoted to one decimal in the text and two in the table, so the check is
    that each rounds to what the text says and that the ordering agrees.
    """
    m = re.search(r"low porosity, \$([\d.]+)\$\\% in the schist against\s*\n?"
                  r"\$([\d.]+)\$\\% in the gneiss", tex)
    assert m, "the porosity sentence is no longer in its expected form"
    said_schist, said_gneiss = float(m.group(1)), float(m.group(2))
    real_schist = means["Psammitic schist"].Porosity_percent.mean()
    real_gneiss = means["Augen gneiss"].Porosity_percent.mean()

    assert round(real_schist, 1) == said_schist, (
        f"text says {said_schist}% for the schist, data gives {real_schist:.2f}%")
    assert round(real_gneiss, 1) == said_gneiss, (
        f"text says {said_gneiss}% for the gneiss, data gives {real_gneiss:.2f}%")
    assert (said_schist > said_gneiss) == (real_schist > real_gneiss), (
        "the sentence has the two rocks the wrong way round: it reads "
        f"{said_schist} against {said_gneiss} while the measurements give "
        f"{real_schist:.2f} and {real_gneiss:.2f}")


def test_the_columns_table1_reports_are_still_unread_by_code():
    """If one of these becomes an input, it needs a stronger guard than this.

    The premise of this file is that these columns reach the paper only
    through Table 1. Should the pipeline start consuming one, a stale value
    would corrupt results rather than just a table, and the check belongs
    somewhere the outputs can see it.
    """
    unread = ["Mass", "Bulk_Density", "Porosity_percent"]
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for pat in ("tools/**/*.py", "scripts/*.py")
        for p in REPO.glob(pat))
    consumed = [c for c in unread if c in blob]
    assert not consumed, (
        f"{consumed} are now read by code. They were report-only when this "
        "test was written; a stale value in them now affects results, so the "
        "input-side guards need extending to cover them."
    )


# ---------------------------------------------------------------------------
# Section 4.1 quotes several statistics derived from the same replicate file.
# They are computed nowhere in ``tools/`` either, so they share Table 1's
# blind spot: the pipeline can be re-run and every one of them can be stale.
# ---------------------------------------------------------------------------

def _per_angle(means, rock, col):
    return means[rock].groupby("Angle")[col].mean()


def test_the_anisotropy_ratios_match_the_measurements(tex, means):
    """End-member contrast and full-range ratio, tensile and compressive."""
    want = {
        "Psammitic schist": dict(ends=1.91, full=2.76, ucs=1.69),
        "Augen gneiss":     dict(ends=1.31, full=1.37, ucs=1.75),
    }
    for rock, w in want.items():
        t = _per_angle(means, rock, "Tensile_strength_Mpa")
        u = _per_angle(means, rock, "UCS_(Mpa)")
        assert round(t.loc[0] / t.loc[90], 2) == w["ends"], (
            f"{rock}: end-member ratio is {t.loc[0] / t.loc[90]:.3f}, "
            f"the paper says {w['ends']}")
        assert round(t.max() / t.min(), 2) == w["full"], (
            f"{rock}: tensile anisotropy ratio is {t.max() / t.min():.3f}, "
            f"the paper says {w['full']}")
        assert round(u.max() / u.min(), 2) == w["ucs"], (
            f"{rock}: compressive ratio is {u.max() / u.min():.3f}, "
            f"the paper says {w['ucs']}")


def test_the_two_modes_are_minimized_at_different_angles(means):
    """Section 4.1 rests on this: tensile at 75 in both, compressive elsewhere."""
    expect_ucs_min = {"Psammitic schist": 45, "Augen gneiss": 30}
    for rock, ucs_angle in expect_ucs_min.items():
        t = _per_angle(means, rock, "Tensile_strength_Mpa")
        u = _per_angle(means, rock, "UCS_(Mpa)")
        assert t.idxmin() == 75, f"{rock}: tensile minimum moved to {t.idxmin()} deg"
        assert u.idxmin() == ucs_angle, (
            f"{rock}: compressive minimum is at {u.idxmin()} deg, the paper "
            f"says {ucs_angle}")


def test_scatter_is_largest_at_the_weakest_orientation(tex, means):
    """Claim (iii). The two quoted coefficients of variation, and the pattern."""
    said = {"Psammitic schist": 19.8, "Augen gneiss": 6.2}
    for rock, cv75 in said.items():
        g = means[rock].groupby("Angle")["Tensile_strength_Mpa"]
        cv = 100 * g.std() / g.mean()
        assert round(cv.loc[75], 1) == cv75, (
            f"{rock}: CV at 75 deg is {cv.loc[75]:.1f}%, the paper says {cv75}%")
        assert cv.idxmax() == 75, (
            f"{rock}: scatter now peaks at {cv.idxmax()} deg, not at the weakest "
            "orientation, which is the claim the sentence makes")


def test_which_mode_carries_the_stronger_fabric_effect(means):
    """The per-lithology reversal, which the paragraph turns on."""
    ratios = {}
    for rock in ("Psammitic schist", "Augen gneiss"):
        t = _per_angle(means, rock, "Tensile_strength_Mpa")
        u = _per_angle(means, rock, "UCS_(Mpa)")
        ratios[rock] = (t.max() / t.min(), u.max() / u.min())
    t_g, u_g = ratios["Augen gneiss"]
    t_s, u_s = ratios["Psammitic schist"]
    assert u_g > t_g, "the gneiss no longer shows the stronger effect in compression"
    assert t_s > u_s, "the schist no longer shows the stronger effect in tension"
    # and the quoted relative separations
    assert round(100 * (u_g - t_g) / t_g) == 28, f"gneiss separation {100*(u_g-t_g)/t_g:.1f}%"
    assert round(100 * (t_s - u_s) / u_s) == 63, f"schist separation {100*(t_s-u_s)/u_s:.1f}%"
