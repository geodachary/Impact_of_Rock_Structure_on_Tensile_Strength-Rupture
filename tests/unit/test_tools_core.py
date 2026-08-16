#!/usr/bin/env python3
"""Tests for the shared `tools` package.

Expected values are stated from analytic reasoning BEFORE comparison with the
implementation, so a test cannot simply enshrine current behaviour.

Run:  PYTHONPATH=. python tools/tests/test_tools.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import conventions as cv          # noqa: E402
from tools import lithology as lith          # noqa: E402
from tools import traces as tr               # noqa: E402
from tools import failure_classification as fc  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
# At alpha_exp = 0 the foliation trace is parallel to +x, so its tangent is
# (1, 0) and its normal is (0, 1) -- the normal points along the loading axis.
t0, n0 = cv.foliation_trace_tangent_normal(cv.alpha_f_from_alpha_exp_deg(0))
check("alpha_exp=0: trace tangent along +x", np.allclose(t0, (1, 0), atol=1e-12), f"{t0}")
check("alpha_exp=0: normal along +y (loading axis)", np.allclose(n0, (0, 1), atol=1e-12), f"{n0}")

# At alpha_exp = 90 the trace is parallel to the loading axis: tangent (0, 1).
t9, n9 = cv.foliation_trace_tangent_normal(cv.alpha_f_from_alpha_exp_deg(90))
check("alpha_exp=90: trace tangent along +y (loading axis)",
      np.allclose(np.abs(t9), (0, 1), atol=1e-12), f"{t9}")
check("alpha_exp=90: normal along x", np.allclose(np.abs(n9), (1, 0), atol=1e-12), f"{n9}")

# Axial error is 180-periodic and symmetric; 170 vs 10 differ by 20 degrees.
check("axial error 170 vs 10 == 20", np.isclose(cv.axial_angular_error_deg(170, 10), 20.0))
check("axial error 5 vs 175 == 10", np.isclose(cv.axial_angular_error_deg(5, 175), 10.0))
check("axial error is in [0,90]",
      0 <= float(cv.axial_angular_error_deg(123.4, 7.7)) <= 90)
check("axial error identical lines == 0", np.isclose(cv.axial_angular_error_deg(37, 217), 0.0))

# Tension-positive conversion flips sign exactly once.
a = cv.to_tension_positive(1.0, -2.0, 0.5, "compression_positive")
check("compression->tension flips all components", np.allclose(a, (-1.0, 2.0, -0.5)))

# ---------------------------------------------------------------------------
# Lithology guards
# ---------------------------------------------------------------------------
check("gneiss owns samples 1-7", lith.AUGEN_GNEISS.sample_ids == (1, 2, 3, 4, 5, 6, 7))
check("schist owns samples 8-14", lith.PSAMMITIC_SCHIST.sample_ids == (8, 9, 10, 11, 12, 13, 14))
check("sample 14 is schist at 90 deg",
      lith.lithology_of_sample(14).key == "psammitic_schist"
      and lith.PSAMMITIC_SCHIST.angle_for_sample(14) == 90)
check("sample 1 is gneiss at 0 deg",
      lith.lithology_of_sample(1).key == "augen_gneiss"
      and lith.AUGEN_GNEISS.angle_for_sample(1) == 0)
check("legacy 'psammatic' spelling resolves to canonical",
      lith.get("psammatic_schist").key == "psammitic_schist")
try:
    lith.assert_notebook_lithology("augen_gneiss", [8])
    check("cross-lithology sample guard raises", False)
except AssertionError:
    check("cross-lithology sample guard raises", True)

# The two pairings the brief names explicitly.
check("pairing sample 1 -> ddm_crack_sample_1.csv",
      lith.predicted_trace_path(1).name == "ddm_crack_sample_1.csv")
check("pairing sample 14 -> ddm_crack_sample_14.csv",
      lith.predicted_trace_path(14).name == "ddm_crack_sample_14.csv")
check("14 rows in pairing table", len(lith.pairing_table()) == 14)

# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------
# A perfectly vertical line of points must fit to 90 degrees from +x.
yy = np.linspace(-0.01, 0.01, 21)
ang, col = tr.fit_orientation_deg(np.zeros_like(yy), yy)
check("vertical point set fits to 90 deg", np.isclose(ang, 90.0, atol=1e-9), f"{ang:.6f}")
check("collinear point set has collinearity ~1", np.isclose(col, 1.0, atol=1e-9))

ang45, _ = tr.fit_orientation_deg(yy, yy)
check("45-degree point set fits to 45 deg", np.isclose(ang45, 45.0, atol=1e-9), f"{ang45:.6f}")

# The longest connected run must win the primary-segment rule.
x = np.concatenate([np.linspace(0, 0.001, 3), np.linspace(0.02, 0.03, 30)])
y = np.zeros_like(x)
px, py, nseg, plen, tot = tr.primary_segment(x, y)
check("primary segment picks the longest run", len(px) == 30 and nseg == 2,
      f"nseg={nseg}, n_primary={len(px)}")

# ---------------------------------------------------------------------------
# Four-mechanism classifier — analytic states, expectations stated first
# ---------------------------------------------------------------------------
P = dict(T_wp=1.0, c_wp=1.0, phi_wp=np.radians(30.0),
         T_m=2.0, c_m=2.0, phi_m=np.radians(30.0), n_theta=361)

# (1) Uniaxial tension normal to the foliation. alpha_f = 0 means the foliation
# normal is +y, so syy = +3 acts directly across the weak plane:
# sigma_n_wp = 3 -> R_WT = 3/1 = 3, which must exceed R_MT = 3/2 = 1.5.
r = fc.classify(np.array([0.0]), np.array([3.0]), np.array([0.0]), alpha_f=0.0, **P)
check("tension across foliation -> WT", r["mode"][0] == "WT", f"mode={r['mode'][0]}")
check("  R_WT == sigma_n/T_wp == 3", np.isclose(r["R_WT"][0], 3.0))
check("  R_MT == s1/T_m == 1.5", np.isclose(r["R_MT"][0], 1.5))
check("  locus is weak_plane", r["locus"][0] == "weak_plane")
check("  mechanism is tensile", r["mechanism"][0] == "tensile")

# (2) Same stress, foliation rotated 90 deg: the weak-plane normal is now +x,
# so sigma_n_wp = 0 and the weak plane carries no opening traction.
# R_WT = 0, and matrix tension must govern.
r2 = fc.classify(np.array([0.0]), np.array([3.0]), np.array([0.0]),
                 alpha_f=np.pi / 2, **P)
check("tension along foliation -> MT", r2["mode"][0] == "MT", f"mode={r2['mode'][0]}")
check("  R_WT == 0 (no opening traction on the plane)", np.isclose(r2["R_WT"][0], 0.0))

# (3) Weak-plane utilities must be inactive where no weak plane exists.
r3 = fc.classify(np.array([0.0]), np.array([3.0]), np.array([0.0]), alpha_f=0.0,
                 weak_plane_weight=np.array([0.0]), activation_floor=0.0, **P)
check("inactive weak plane -> R_WT is NaN", np.isnan(r3["R_WT"][0]))
check("inactive weak plane cannot win argmax -> MT", r3["mode"][0] == "MT",
      f"mode={r3['mode'][0]}")
check("  locus is matrix", r3["locus"][0] == "matrix")

# (4) Hydrostatic tension carries no shear on any plane, so every shear utility
# is zero and a tensile class must govern.
r4 = fc.classify(np.array([3.0]), np.array([3.0]), np.array([0.0]), alpha_f=0.0, **P)
check("hydrostatic tension has zero weak-plane shear", np.isclose(r4["R_WS"][0], 0.0))
check("hydrostatic tension -> tensile mechanism", r4["mechanism"][0] == "tensile")

# (5) Pure shear: sxx = -syy = 0, txy = 4 with alpha_f = 0 gives tau_wp = 4 and
# sigma_n_wp = 0, so R_WS = 4/(c_wp + 0) = 4.
r5 = fc.classify(np.array([0.0]), np.array([0.0]), np.array([4.0]), alpha_f=0.0, **P)
check("pure shear on the foliation -> R_WS == 4", np.isclose(r5["R_WS"][0], 4.0))
check("pure shear -> WS", r5["mode"][0] == "WS", f"mode={r5['mode'][0]}")
check("  mechanism is shear", r5["mechanism"][0] == "shear")

# (6) Below threshold -> no failure, and no class is assigned.
r6 = fc.classify(np.array([0.0]), np.array([0.1]), np.array([0.0]), alpha_f=0.0, **P)
check("below threshold -> no_failure", r6["mode"][0] == fc.NO_FAILURE)
check("  mode_code is -1", r6["mode_code"][0] == -1)

# (7) Candidate-plane resolution convergence.
big = fc.classify(np.array([0.0]), np.array([0.0]), np.array([4.0]), alpha_f=0.0,
                  **{**P, "n_theta": 1441})
rel = abs(big["R_MS"][0] - r5["R_MS"][0]) / max(r5["R_MS"][0], 1e-12)
check("R_MS converged: 361 vs 1441 within 2%", rel < 0.02, f"rel={rel:.4%}")

# (8) Parameter validation must reject unphysical input.
for name, kw in (("negative T_m", dict(T_m=-1.0)), ("NaN phi_m", dict(phi_m=np.nan)),
                 ("phi_m >= 90 deg", dict(phi_m=np.radians(100.0))),
                 ("zero T_wp", dict(T_wp=0.0))):
    try:
        fc.classify(np.array([1.0]), np.array([1.0]), np.array([0.0]), alpha_f=0.0,
                    **{**P, **kw})
        check(f"rejects {name}", False)
    except ValueError:
        check(f"rejects {name}", True)

# (9) Zero cohesion must not divide by zero.
try:
    z = fc.classify(np.array([0.0]), np.array([0.0]), np.array([1.0]), alpha_f=0.0,
                    **{**P, "c_wp": 0.0})
    check("zero cohesion is finite (guarded)", np.isfinite(z["R_WS"][0]))
except ZeroDivisionError:
    check("zero cohesion is finite (guarded)", False)

# (10) Mixed flag is secondary and never a fifth class.
rm = fc.classify(np.array([0.0]), np.array([3.0]), np.array([0.0]), alpha_f=0.0,
                 **{**P, "eta_mix": 0.0})
check("mixed flag does not change the primary class", rm["mode"][0] == "WT")
check("every primary label is one of WT/WS/MT/MS/none",
      set(np.unique(rm["mode"])) <= set(fc.CLASS_ORDER) | {fc.NO_FAILURE})

# (11) Traction resolution against hand values.
sn, ta = fc.resolve_traction(5.0, 0.0, 0.0, 0.0)
check("sigma_xx=5 on beta=0 gives sigma_n=5, tau=0",
      np.isclose(sn, 5.0) and np.isclose(ta, 0.0, atol=1e-12))
sn, ta = fc.resolve_traction(5.0, 0.0, 0.0, np.pi / 4)
check("sigma_xx=5 on beta=45 gives sigma_n=2.5, |tau|=2.5",
      np.isclose(sn, 2.5) and np.isclose(abs(ta), 2.5))

# (12) Principal ordering.
rng = np.random.default_rng(42)
for _ in range(20):
    a, b, c = rng.normal(size=3) * 5
    s1, s3, _th = fc.principal_stresses(a, b, c)
    if s1 < s3 - 1e-12:
        check("principal ordering s1 >= s3", False)
        break
else:
    check("principal ordering s1 >= s3 (20 random states)", True)

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")


def test_all_tools_core_checks_passed():
    """Report the module-level checks above as a single pytest result."""
    assert not FAIL, "failing checks: " + ", ".join(FAIL)


def test_expected_number_of_checks_ran():
    """Guard against the checks silently not executing."""
    assert len(PASS) + len(FAIL) >= 45, (
        f"only {len(PASS) + len(FAIL)} checks ran; the module may have exited early")


if __name__ == "__main__":
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)
