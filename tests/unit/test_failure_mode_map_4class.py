# test_failure_mode_map_4class.py
# Tests for failure_mapping_helpers.failure_mode_map_4class.
# pytest is NOT installed in this environment, so this is a plain
# assert-based script with an `if __name__ == "__main__"` runner; the
# test functions are still pytest-compatible (test_* names, bare asserts)
# so they can be collected by pytest later if it gets installed.
#
# All stress states below are passed with stress_sign_mode="tension_positive"
# so the sign-auto-detection heuristic (built for large fields) cannot
# reinterpret tiny synthetic arrays.

import os
import sys
import traceback

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.failure_mapping_helpers import failure_mode_map_4class, _mode_pair_label
from tools.conventions import alpha_f_from_alpha_exp_deg, foliation_trace_tangent_normal

PHI30 = np.deg2rad(30.0)


def _call(sxx, syy, txy, s1, Tm, Coh, Phi, alpha_wp_line, **kw):
    """Single-point convenience wrapper: scalars in, dict of scalars out."""
    a = lambda v: np.asarray([float(v)])
    kw.setdefault("stress_sign_mode", "tension_positive")
    mode, info = failure_mode_map_4class(
        a(sxx), a(syy), a(txy), a(s1),
        a(Tm), a(Coh), a(Phi),
        alpha_wp_line=float(alpha_wp_line), **kw
    )
    flat = {k: v[0] for k, v in info.items()}
    return str(mode[0]), flat


# ----------------------------------------------------------------------
# Test 1
# EXPECTATION (stated before running): pure hydrostatic compression
# (sxx=syy=-10 MPa, txy=0, s1=-10) has no tension anywhere and zero shear
# stress on every plane, so R_MT=0, R_MS~0, R_WT=0, R_WS~0 ->
# primary_mode "no_failure", failed False, util_ratio 0 (guarded).
# ----------------------------------------------------------------------
def test_1_hydrostatic_compression_no_failure():
    mode, f = _call(
        -10.0, -10.0, 0.0, -10.0,
        Tm=10.0, Coh=5.0, Phi=PHI30, alpha_wp_line=0.0,
        x=np.asarray([0.0]), y=np.asarray([0.0]),
        weak_spacing=10.0, weak_phase=0.0,
    )
    assert mode == "no_failure", mode
    assert not f["failed"]
    assert abs(f["R_MT"]) < 1e-12, f["R_MT"]
    assert abs(f["R_MS"]) < 1e-9, f["R_MS"]
    assert abs(f["R_WT"]) < 1e-12, f["R_WT"]
    assert abs(f["R_WS"]) < 1e-9, f["R_WS"]
    assert f["util_ratio"] == 0.0, f["util_ratio"]
    assert f["locus"] == "none" and f["mechanism"] == "none"
    return dict(mode=mode, R_MT=f["R_MT"], R_MS=f["R_MS"],
                R_WT=f["R_WT"], R_WS=f["R_WS"], util_max=f["util_max"])


# ----------------------------------------------------------------------
# Test 2
# EXPECTATION: uniaxial tension S=5 along global +y, foliation trace along
# +x (alpha_wp_line=0 -> plane normal along y), point ON a weak plane
# (x=y=0, phase=0 -> w_act=1). sigma_n_wp = S = 5 (full projection onto
# the normal). With Tm=4: R_WT = 5/(0.35*4) = 3.5714, R_MT = 5/4 = 1.25,
# shear utilities ~0 (no compression, no shear). WT must govern:
# primary "WT", locus "weak_plane", mechanism "tensile".
# ----------------------------------------------------------------------
def test_2_uniaxial_tension_on_band_WT():
    S, Tm = 5.0, 4.0
    mode, f = _call(
        0.0, S, 0.0, S,
        Tm=Tm, Coh=5.0, Phi=PHI30, alpha_wp_line=0.0,
        x=np.asarray([0.0]), y=np.asarray([0.0]),
        weak_spacing=10.0, weak_phase=0.0,
    )
    R_WT_expect = S / (0.35 * Tm)          # 3.571428...
    R_MT_expect = S / Tm                   # 1.25
    assert mode == "WT", mode
    assert f["wp_active"]
    assert abs(f["w_act"] - 1.0) < 1e-12, f["w_act"]
    assert abs(f["sigma_n_wp"] - S) < 1e-9, f["sigma_n_wp"]
    assert abs(f["tau_wp"]) < 1e-9, f["tau_wp"]
    assert abs(f["R_WT"] - R_WT_expect) < 1e-9, (f["R_WT"], R_WT_expect)
    assert abs(f["R_MT"] - R_MT_expect) < 1e-12, (f["R_MT"], R_MT_expect)
    assert f["locus"] == "weak_plane" and f["mechanism"] == "tensile"
    assert f["failed"]
    return dict(mode=mode, R_WT=f["R_WT"], R_MT=f["R_MT"],
                sigma_n_wp=f["sigma_n_wp"], w_act=f["w_act"])


# ----------------------------------------------------------------------
# Test 3
# EXPECTATION: identical stress state to Test 2 but the point sits exactly
# mid-way between weak planes (phase = spacing/2 -> |d_mod| = spacing/2,
# w_act = exp(-(5/1.2)^2) ~ 2.9e-8 << wp_active_frac=0.05). Weak-plane
# utilities must be EXCLUDED (NaN), not just small, and MT governs with
# R_MT = 1.25 >= util_min.
# ----------------------------------------------------------------------
def test_3_uniaxial_tension_off_band_MT():
    S, Tm, spacing = 5.0, 4.0, 10.0
    mode, f = _call(
        0.0, S, 0.0, S,
        Tm=Tm, Coh=5.0, Phi=PHI30, alpha_wp_line=0.0,
        x=np.asarray([0.0]), y=np.asarray([0.0]),
        weak_spacing=spacing, weak_phase=spacing / 2.0,
    )
    assert mode == "MT", mode
    assert not f["wp_active"], f["w_act"]
    assert f["w_act"] < 0.05, f["w_act"]
    assert np.isnan(f["R_WT"]), f["R_WT"]         # excluded, not merely small
    assert np.isnan(f["R_WS"]), f["R_WS"]
    assert abs(abs(f["d_mod"]) - spacing / 2.0) < 1e-9, f["d_mod"]
    assert abs(f["R_MT"] - S / Tm) < 1e-12, f["R_MT"]
    assert f["locus"] == "matrix" and f["mechanism"] == "tensile"
    assert f["failed"]
    return dict(mode=mode, R_MT=f["R_MT"], R_WT=f["R_WT"],
                w_act=f["w_act"], d_mod=f["d_mod"])


# ----------------------------------------------------------------------
# Test 4
# EXPECTATION: pure shear state sxx=+5, syy=-5, txy=0 (sigma_m=0, s1=5),
# Coh=1, Phi=30 deg, Tm=10 (so R_MT=0.5). Hand analysis of the MC ratio
# r(beta) = |tau|/(Coh + max(-sigma_n,0) tan(phi)) on this state:
# sigma_n = 5 cos(2b), tau = -5 sin(2b); on the compressive half the
# stationarity condition gives no interior optimum, so the CONTINUUM max
# sits at sigma_n = 0, i.e. beta = 45 deg (or 135 deg), with R_MS = S/Coh
# = 5.0. NOTE on discretization (original expectation of exactly 5.0 was
# corrected after inspection): the scan grid linspace(0, pi, 361,
# endpoint=False) has step 180/361 ~ 0.4986 deg, so beta = 45 deg is NOT
# sampled; the nearest grid plane on the compressive side is beta =
# 91*180/361 = 45.374 deg where sigma_n = -0.0653, |tau| = 4.99958,
# tau_allow = 1 + 0.0653*tan30 = 1.0377 -> ratio = 4.8180 (hand-computed).
# So we assert: (a) R_MS == independent direct 361-plane scan exactly,
# (b) R_MS == hand-recomputed MC ratio AT the returned beta_crit_matrix,
# (c) R_MS within 4% of the continuum optimum 5.0, (d) beta_crit within
# one grid step of 45 or 135 deg. Point placed off-band so MS governs.
# ----------------------------------------------------------------------
def test_4_pure_shear_matrix_MS():
    S, Coh, Phi, Tm = 5.0, 1.0, PHI30, 10.0
    spacing = 10.0
    mode, f = _call(
        S, -S, 0.0, S,
        Tm=Tm, Coh=Coh, Phi=Phi, alpha_wp_line=0.0,
        x=np.asarray([0.0]), y=np.asarray([0.0]),
        weak_spacing=spacing, weak_phase=spacing / 2.0,  # off-band
    )
    # independent direct scan (same grid definition, independent code path)
    beta = np.linspace(0, np.pi, 361, endpoint=False)
    sig_n = 0.5 * (S + (-S)) + 0.5 * (S - (-S)) * np.cos(2 * beta)
    tau = -0.5 * (S - (-S)) * np.sin(2 * beta)
    m = sig_n <= 0.0
    allow = Coh + np.where(m, -sig_n, 0.0) * np.tan(Phi)
    ratio = np.where(m, np.abs(tau) / np.maximum(allow, 1e-12), 0.0)
    R_MS_direct = float(np.max(ratio))
    assert mode == "MS", mode
    assert abs(f["R_MS"] - R_MS_direct) < 1e-12, (f["R_MS"], R_MS_direct)
    # hand-recompute the MC ratio at the returned critical plane
    b_c = float(f["beta_crit_matrix"])
    sn_c = S * np.cos(2 * b_c)
    tau_c = -S * np.sin(2 * b_c)
    r_hand = abs(tau_c) / (Coh + max(-sn_c, 0.0) * np.tan(Phi))
    assert sn_c <= 0.0, sn_c  # compression_only: critical plane must be compressive
    assert abs(f["R_MS"] - r_hand) < 1e-12, (f["R_MS"], r_hand)
    assert abs(f["R_MS"] - 5.0) < 0.2, f["R_MS"]  # within 4% of continuum optimum
    bc = np.rad2deg(b_c)
    grid_step = 180.0 / 361.0
    assert min(abs(bc - 45.0), abs(bc - 135.0)) <= grid_step + 1e-9, bc
    assert abs(f["R_MT"] - 0.5) < 1e-12, f["R_MT"]
    assert f["locus"] == "matrix" and f["mechanism"] == "shear"
    return dict(mode=mode, R_MS=f["R_MS"], R_MS_direct=R_MS_direct,
                beta_crit_deg=bc, R_MS_hand_at_beta_crit=r_hand, R_MT=f["R_MT"])


# ----------------------------------------------------------------------
# Test 5
# EXPECTATION: end-member fabric angles via angle_convention. Pure tension
# S=3 along global +y (sxx=0, syy=S, txy=0). The analytic projection is
# sigma_n = nx^2 sxx + 2 nx ny txy + ny^2 syy with n_hat from
# foliation_trace_tangent_normal(alpha_f):
#   alpha_exp = 0  -> n_hat=(0,1)  -> sigma_n_wp = S  (opening across foliation)
#   alpha_exp = 90 -> n_hat=(-1,0) -> sigma_n_wp = 0  (foliation || loading)
# |tau_wp| must match |t.sigma.n| (0 at both end-members here). sigma_n is
# invariant to the +/- sign of the normal, so exact equality is required.
# ----------------------------------------------------------------------
def test_5_end_member_angles_projection():
    S = 3.0
    out = {}
    for aexp in (0.0, 90.0):
        alpha_f = alpha_f_from_alpha_exp_deg(aexp)
        t_hat, n_hat = foliation_trace_tangent_normal(alpha_f)
        sxx, syy, txy = 0.0, S, 0.0
        nx, ny = n_hat
        tx, ty = t_hat
        sig_n_expect = nx * nx * sxx + 2 * nx * ny * txy + ny * ny * syy
        tau_expect = tx * nx * sxx + (tx * ny + ty * nx) * txy + ty * ny * syy
        mode, f = _call(
            sxx, syy, txy, S,
            Tm=10.0, Coh=5.0, Phi=PHI30, alpha_wp_line=alpha_f,
            x=np.asarray([0.0]), y=np.asarray([0.0]),
            weak_spacing=10.0, weak_phase=0.0,
        )
        assert abs(f["sigma_n_wp"] - sig_n_expect) < 1e-12, (aexp, f["sigma_n_wp"], sig_n_expect)
        # tau sign depends on the tangent's arrow; magnitude must agree
        assert abs(abs(f["tau_wp"]) - abs(tau_expect)) < 1e-12, (aexp, f["tau_wp"], tau_expect)
        out[aexp] = dict(sigma_n_wp=f["sigma_n_wp"], expect=sig_n_expect,
                         tau_wp=f["tau_wp"], n_hat=n_hat)
    assert abs(out[0.0]["sigma_n_wp"] - S) < 1e-12
    assert abs(out[90.0]["sigma_n_wp"] - 0.0) < 1e-12
    return out


# ----------------------------------------------------------------------
# Test 6
# EXPECTATION (documented behavior choice): Tm=0 is treated as an invalid
# non-positive strength -> clear ValueError. Coh=0 is a physically valid
# cohesionless matrix -> no crash, finite utilities. Two sub-cases:
# (a) pure shear sxx=+5/syy=-5: the critical grid plane carries a small
#     compressive sigma_n (grid never lands exactly on sigma_n=0, see
#     Test 4 note), so tau_allow = sigma_comp*tan(phi) > 0 and R_MS is
#     large-but-moderate (hand value at beta=45.374 deg:
#     4.99958/(0.0653*tan30) ~ 132.7) -- original ">1e6" expectation was
#     corrected: the eps guard is not reached on this state.
# (b) txy-only state (sxx=syy=0, txy=1): beta=0 IS on the grid, there
#     sigma_n=0 exactly -> tau_allow=0 -> the eps=1e-12 guard engages and
#     R_MS = |tau|/eps = 1e12, finite (genuine guard test, no crash).
# ----------------------------------------------------------------------
def test_6_zero_strength_guards():
    got_raise = False
    try:
        _call(0.0, 5.0, 0.0, 5.0, Tm=0.0, Coh=5.0, Phi=PHI30, alpha_wp_line=0.0)
    except ValueError:
        got_raise = True
    assert got_raise, "Tm=0 must raise ValueError"

    mode_a, fa = _call(5.0, -5.0, 0.0, 5.0, Tm=10.0, Coh=0.0, Phi=PHI30,
                       alpha_wp_line=0.0)
    assert np.isfinite(fa["R_MS"]), fa["R_MS"]
    assert fa["R_MS"] > 100.0, fa["R_MS"]

    mode_b, fb = _call(0.0, 0.0, 1.0, 1.0, Tm=10.0, Coh=0.0, Phi=PHI30,
                       alpha_wp_line=0.0)
    assert np.isfinite(fb["R_MS"]), fb["R_MS"]
    assert abs(fb["R_MS"] - 1.0e12) < 1e3, fb["R_MS"]  # |tau|/eps with eps=1e-12
    return dict(Tm0_raised=got_raise, R_MS_coh0_shear=fa["R_MS"],
                R_MS_coh0_txy_only=fb["R_MS"])


# ----------------------------------------------------------------------
# Test 7
# EXPECTATION: invalid parameters must raise ValueError, never silently
# propagate NaN: (a) NaN Phi, (b) negative Tm, (c) NaN Tm, (d) Phi=100
# (auto-detected as degrees -> 1.745 rad >= pi/2 -> out of range),
# (e) NaN weak_phi override.
# ----------------------------------------------------------------------
def test_7_invalid_parameter_rejection():
    cases = {
        "nan_phi": dict(Tm=10.0, Coh=5.0, Phi=np.nan),
        "neg_Tm": dict(Tm=-2.0, Coh=5.0, Phi=PHI30),
        "nan_Tm": dict(Tm=np.nan, Coh=5.0, Phi=PHI30),
        "phi_100deg_out_of_range": dict(Tm=10.0, Coh=5.0, Phi=100.0),
        "nan_weak_phi": dict(Tm=10.0, Coh=5.0, Phi=PHI30, weak_phi=np.nan),
    }
    results = {}
    for name, kw in cases.items():
        try:
            _call(0.0, 5.0, 0.0, 5.0, alpha_wp_line=0.0, **kw)
            results[name] = "NO RAISE"
        except ValueError as e:
            results[name] = f"ValueError: {e}"
        except Exception as e:  # wrong exception type = fail
            results[name] = f"WRONG EXC {type(e).__name__}: {e}"
    for name, r in results.items():
        assert r.startswith("ValueError"), (name, r)
    return results


# ----------------------------------------------------------------------
# Test 8
# EXPECTATION: mixed_flag boundary behavior. Pure-shear state sxx=S,
# syy=-S gives R_MS ~ S/Coh (continuum value; the discrete 361-plane grid
# lands slightly off 45 deg, see Test 4 note, so the exact value is
# measured, not assumed -- original "ratio == 0.85 exactly" expectation
# corrected for the same discretization reason: with S=2, Coh=1, Tm=2/1.7
# the measured util_ratio is R_MT/R_MS_grid = 1.7/1.9701 ~ 0.8629, not
# 0.85). We therefore measure r0 = util_ratio from a first call (sanity:
# 0.80 < r0 < 0.90), then verify the boundary flip against the MEASURED
# r0: mixed_flag True at eta_mix == r0 (>= comparison), True just below,
# False just above. Then a 200-point random sweep reports flagged-mixed
# counts for eta_mix in {0.70..0.95}; counts must be non-increasing with
# eta_mix.
# ----------------------------------------------------------------------
def test_8_mixed_flag_sensitivity():
    S, Coh, Tm = 2.0, 1.0, 2.0 / 1.7
    base = dict(Tm=Tm, Coh=Coh, Phi=PHI30, alpha_wp_line=0.0)
    mode, f = _call(S, -S, 0.0, S, **base)
    r0 = float(f["util_ratio"])
    assert f["failed"]
    assert 0.80 < r0 < 0.90, r0  # measured ratio; see discretization note above
    assert f["secondary_descriptor"] == _mode_pair_label("MS", "MT") == "MS-MT", \
        f["secondary_descriptor"]

    _, f_at = _call(S, -S, 0.0, S, eta_mix=r0, **base)
    _, f_below = _call(S, -S, 0.0, S, eta_mix=max(r0 - 1e-9, 0.0), **base)
    _, f_above = _call(S, -S, 0.0, S, eta_mix=min(r0 + 1e-9, 1.0), **base)
    assert bool(f_at["mixed_flag"]) is True, "flag must be True at threshold (>=)"
    assert bool(f_below["mixed_flag"]) is True
    assert bool(f_above["mixed_flag"]) is False

    # small illustrative sweep on random failed states (not manuscript-scale)
    rng = np.random.default_rng(42)
    n = 200
    sxx = rng.uniform(-8, 8, n)
    syy = rng.uniform(-8, 8, n)
    txy = rng.uniform(-4, 4, n)
    s_avg = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5 * (sxx - syy)) ** 2 + txy ** 2)
    s1 = s_avg + rad
    xs = rng.uniform(-1, 1, n)  # drawn ONCE: same points at every eta
    ys = rng.uniform(-1, 1, n)
    sweep = {}
    for eta in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        _, info = failure_mode_map_4class(
            sxx, syy, txy, s1,
            np.full(n, 3.0), np.full(n, 1.5), np.full(n, PHI30),
            alpha_wp_line=np.deg2rad(30.0),
            x=xs, y=ys,
            weak_spacing=0.5, weak_bandwidth_frac=0.12,
            eta_mix=eta, stress_sign_mode="tension_positive",
        )
        sweep[eta] = (int(np.sum(info["mixed_flag"])), int(np.sum(info["failed"])))
    counts = [sweep[e][0] for e in sorted(sweep)]
    assert all(a >= b for a, b in zip(counts, counts[1:])), \
        f"mixed counts must be non-increasing with eta_mix: {sweep}"
    return dict(util_ratio=r0, at=bool(f_at["mixed_flag"]),
                below=bool(f_below["mixed_flag"]), above=bool(f_above["mixed_flag"]),
                sweep={e: f"{m} mixed / {fl} failed of 200" for e, (m, fl) in sweep.items()})


ALL_TESTS = [
    test_1_hydrostatic_compression_no_failure,
    test_2_uniaxial_tension_on_band_WT,
    test_3_uniaxial_tension_off_band_MT,
    test_4_pure_shear_matrix_MS,
    test_5_end_member_angles_projection,
    test_6_zero_strength_guards,
    test_7_invalid_parameter_rejection,
    test_8_mixed_flag_sensitivity,
]


if __name__ == "__main__":
    n_fail = 0
    for t in ALL_TESTS:
        try:
            detail = t()
            print(f"PASS  {t.__name__}")
            if detail:
                print(f"      values: {detail}")
        except Exception:
            n_fail += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(ALL_TESTS) - n_fail}/{len(ALL_TESTS)} passed")
    sys.exit(1 if n_fail else 0)
