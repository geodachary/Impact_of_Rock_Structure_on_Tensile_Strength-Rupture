"""Constants quoted in Section 3 must match the code the notebooks actually run.

Four separate errors of one kind motivated this file. Section 3.7 described a
crack stepper that is never invoked, quoted a three-mode classifier after the
four-class one had replaced it, claimed a decoupled energy matrix the code does
not use, and stated a weak-plane bandwidth and floor belonging to the
sensitivity sweeps rather than to the field computation. Each looked right in
isolation, because each described a real code path; none of them described the
path that produces the reported numbers.

The repository holds more than one implementation of several steps, kept for
comparison. Reading a constant out of the wrong one is therefore the easiest
mistake to make here, and nothing but a test catches it.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "manuscript" / "manscript_revision_001.tex"

pytestmark = pytest.mark.skipif(not TEX.is_file(), reason="manuscript not present")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def _body(tex):
    """Main text only; the supplement restates values in its own captions."""
    return tex.split("\\appendix")[0]


def test_airy_series_constants_match_the_solver(tex):
    """The boundary discretization must come from the invoked call.

    ``fit_orthotropic_airy_disk`` defaults to Nbd=480 with 1200 per arc, and
    nothing in production uses that pair. ``crack_energy_suite``, which writes
    every archived field, and ``foliation_deviation`` both pass 420 and 900,
    and the notebooks call them with no arguments. The manuscript previously
    quoted the signature defaults and, with them, the convergence of a level
    no run uses: 0.11% where production achieves 0.016%. Refinement does not
    order these two, 480/1200 being the worse of the pair here, so nothing
    downstream would have flagged it.
    """
    from tools.airy_solution import fit_orthotropic_airy_disk as fit
    from tools.analysis import crack_energy_suite as ces
    from tools.analysis import foliation_deviation as fdv
    p = inspect.signature(fit).parameters
    body = _body(tex)
    assert str(p["M"].default) in body, "series truncation is not stated in the text"

    invoked = _argparse_defaults(inspect.getsource(ces))
    other = _argparse_defaults(inspect.getsource(fdv))
    for flag in ("airy_Nbd_base", "airy_Nbd_arc_each"):
        assert invoked[flag] == other[flag], (
            f"{flag} differs between the two field producers: "
            f"{invoked[flag]} against {other[flag]}")
        assert str(invoked[flag]) in body, (
            f"{flag} = {invoked[flag]} is what production runs and the text "
            "does not state it")
    assert str(p["Nbd_arc_each"].default) not in body, (
        "the text quotes the solver's signature default for the platen arc, "
        "which no production run passes")
    assert f"{p['beta_deg'].default:.0f}^\\circ" in body or "10^\\circ" in body
    assert p["mu"].default == 0.0, "text describes frictionless platens"

    # The regularization weight. It moved from 1e-10 to 1e-18 when the design
    # matrix started being column-equilibrated, and the text kept the old one,
    # which described a solve that no longer happens.
    lam = p["lam"].default
    exp = int(round(-np.log10(lam)))
    assert f"10^{{-{exp}}}" in body, (
        f"the text does not state the Tikhonov weight in force, lambda = {lam:g}")
    assert "10^{-10}" not in body, "the retired regularization weight is still quoted"


def test_the_series_order_claim_is_not_the_pre_equilibration_one(tex):
    """Column equilibration removed the conditioning limit the text described.

    The old text stopped the series at M ~ 52 because the fit degraded beyond
    it. That was a property of the unequilibrated design matrix at lambda =
    1e-10. With equilibration the boundary residual falls monotonically to at
    least M = 96, so the stated reason for the cap had to change with it.
    """
    body = _body(tex)
    for stale in ("ill-conditioned", "M \\approx 52"):
        assert stale not in body, (
            f"{stale!r} describes the solver before column equilibration")
    assert "column-equilibrated" in body, (
        "the text no longer says what keeps the fit conditioned")


def test_weak_plane_bandwidth_matches_the_computed_field(tex):
    """Eq. (26) must carry the bandwidth used to build the stored wp_weight."""
    from tools.failure_mapping_helpers import weak_plane_weight_field
    frac = inspect.signature(weak_plane_weight_field).parameters["bandwidth_frac"].default
    m = re.search(r"b_w = ([0-9.]+)\\,s", _body(tex))
    assert m, "Eq. (26) no longer states b_w as a multiple of s"
    assert float(m.group(1)) == pytest.approx(frac), (
        f"text says b_w = {m.group(1)} s, the field is built with {frac} s. "
        "The sensitivity sweeps use a wider band; that value belongs in the "
        "sweep paragraph, not in Eq. (26).")


def test_the_weak_plane_weight_has_no_floor(tex):
    """The stored weight decays to zero, so no floor may be claimed for it."""
    from tools.failure_mapping_helpers import weak_plane_weight_field
    x = np.linspace(-0.02, 0.02, 401)
    X, Y = np.meshgrid(x, x)
    w = weak_plane_weight_field(X, Y, alpha_wp_line=0.0, spacing=0.01)
    assert float(w.min()) < 1e-6, "field now has a floor; Eq. (26) must say so"
    assert "w_{\\mathrm{floor}}" not in _body(tex), (
        "the text reintroduced a floor the computed weight does not have")


def test_classifier_constants_match_strain_partitioning(tex):
    from tools import strain_partitioning as sp
    body = _body(tex)
    assert sp.N_THETA == 361 and "361" in body
    assert sp.ACTIVATION_FLOOR == pytest.approx(0.05) and "0.05" in body
    assert sp.WEAK_T_RATIO == pytest.approx(0.35)
    assert sp.WEAK_C_RATIO == pytest.approx(0.60)
    # and the text has to carry the same two ratios, in Eq. (25) and Table 5
    assert f"{sp.WEAK_T_RATIO:.2f}\\,T_0" in body, "T_wp/T_0 is not stated"
    assert f"{sp.WEAK_C_RATIO:.2f}\\,c" in body, "c_wp/c is not stated"
    assert sp.THRESHOLD == pytest.approx(1.0), (
        "the reported classification fails a point at u >= 1; 0.98 is the "
        "relaxed threshold used only for counting in the sweeps")
    # eta_mix is quoted in the text as a percentage
    assert f"{round((1 - sp.ETA_MIX) * 100)}\\%" in body


def test_crack_stepper_constants_come_from_the_invoked_suite(tex):
    """The published traces come from crack_energy_suite, not crack_path_suite.

    Both modules run in the lithology notebooks and both step a crack, which
    is the trap. ``crack_path_suite`` writes ``smoke_path_sid*.csv``, used for
    nothing that is published; ``crack_energy_suite`` writes
    ``ddm_crack_sample_*.csv``, which is what every trace figure and
    Table B.5 read. Their step budgets differ, 1400 against 1000, and the
    manuscript quoted the one belonging to the module whose output it does
    not show. Assert against the writer of the published file.
    """
    from tools.analysis import crack_energy_suite as ces
    from tools import lithology as lith
    assert "ddm_crack_sample" in inspect.getsource(ces), (
        "crack_energy_suite no longer writes the published trace; find the "
        "module that does and assert against that one")
    assert lith.predicted_trace_path(1).name.startswith("ddm_crack_sample")

    body = _body(tex)
    src = inspect.getsource(ces.main)
    for flag, label in (("max_steps", "step budget"), ("ds_frac", "increment"),
                        ("a0_frac", "seed length")):
        assert f'"--{flag}"' in inspect.getsource(ces), f"{label} flag is gone"

    defaults = _argparse_defaults(inspect.getsource(ces))
    assert str(defaults["max_steps"]) in body, (
        f"the manuscript does not quote the invoked step budget "
        f"{defaults['max_steps']}")
    assert defaults["ds_frac"] == pytest.approx(0.01)
    # the seed is a half-length, mirrored, so the trace is twice a0_frac
    assert defaults["a0_frac"] == pytest.approx(0.016)
    # constants that belong to the unused helper must not reappear
    for stale in ("1.25\\,R_t", "700 steps", "weight 0.45", "1400"):
        assert stale not in body, f"{stale!r} is from the uninvoked stepper"


def _argparse_defaults(src):
    out = {}
    for m in re.finditer(r'add_argument\("--(\w+)",[^)]*default=([-\d.eE]+)', src):
        try:
            out[m.group(1)] = float(m.group(2)) if "." in m.group(2) else int(m.group(2))
        except ValueError:
            pass
    return out


def test_energy_matrix_coupling_is_not_claimed_to_be_zero(tex):
    """rho_H is computed from the crack-local compliance, not set to zero."""
    from tools.crack_helpers import H_matrix_from_stroh
    H = np.asarray(H_matrix_from_stroh(
        42.5e9, 20.8e9, 0.24, 18.1e9,
        alpha_const=np.deg2rad(45.0), psi_tip_global=np.deg2rad(30.0))).reshape(2, 2)
    rho = H[0, 1] / np.sqrt(H[0, 0] * H[1, 1])
    assert rho > 0.0, "coupling is genuinely nonzero at this orientation"
    assert "\\rho_H = 0" not in _body(tex), (
        "the text claims a decoupled production value; the implementation "
        "builds the coupling from S16 and S26 and caps it at 0.15")


def test_mirror_symmetry_csv_agrees_with_the_published_table(tex):
    """The machine-readable copy must not age while the paper stays current.

    This CSV had no writer for a fortnight. It kept showing one specimen as
    having no cached displacement solution long after the cache existed, so
    the manuscript's counts looked wrong to anyone auditing from
    ``outputs/tables/`` when in fact the manuscript was right and the CSV was
    stale. ``make_fig20_supplement.py`` now writes both.
    """
    import pandas as pd
    csv = REPO / "outputs" / "tables" / "displacement_mirror_symmetry.csv"
    if not csv.is_file():
        pytest.skip("mirror-symmetry table not exported yet")
    df = pd.read_csv(csv)
    blocked = df[df["status"] != "computed"]
    assert blocked.empty, (
        "specimens without a mirror index: "
        f"{blocked[['sample', 'reason']].to_dict('records')}. Re-run "
        "scripts/make_fig20_supplement.py; the CSV is probably stale.")
    body = _body(tex)
    n_visible = int((df.max_index > 0.10).sum())
    assert f"{n_visible} of the fourteen exceed" in body or str(n_visible) in body
    assert f"{df.max_index.median():.2f}" in body, (
        f"median index {df.max_index.median():.4f} is not the value quoted")
