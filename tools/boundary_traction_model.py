# boundary_traction_model.py
# -*- coding: utf-8 -*-

import numpy as np
from.geometry_helpers import angle_diff_periodic, _wrap_pi


def smooth_arc_window(d, half_width, smooth):
    d = np.asarray(d, dtype=float)
    w = np.zeros_like(d, dtype=float)
    a = float(half_width)
    s = max(float(smooth), 0.0)

    if s <= 0.0:
        w[d <= a] = 1.0
        return w

    core = d <= (a - s)
    trans = (d > (a - s)) & (d < (a + s))

    w[core] = 1.0
    xi = (d[trans] - (a - s)) / (2.0 * s)
    w[trans] = 0.5 * (1.0 + np.cos(np.pi * xi))
    return w


def platen_tractions_theta(theta, beta, smooth, p0, mu=0.0):
    theta = np.asarray(theta, dtype=float)

    d_top = np.abs(angle_diff_periodic(theta, np.pi / 2.0))
    d_bot = np.abs(angle_diff_periodic(theta, -np.pi / 2.0))

    w_top = smooth_arc_window(d_top, beta, smooth)
    w_bot = smooth_arc_window(d_bot, beta, smooth)

    p = float(p0) * (w_top + w_bot)  # MPa
    tr = -p

    if mu is None or float(mu) <= 0.0:
        return tr, np.zeros_like(tr)

    sign_top = -np.sign(angle_diff_periodic(theta, np.pi / 2.0))
    sign_bot =  np.sign(angle_diff_periodic(theta, -np.pi / 2.0))

    tt = float(mu) * p * (w_top * sign_top + w_bot * sign_bot)
    return tr, tt


def pressure_amplitude_from_load_tapered(P_N, t_m, R_m, beta, smooth, nint=6000):
    th = np.linspace(np.pi / 2.0 - (beta + smooth), np.pi / 2.0 + (beta + smooth), int(nint))

    d_top = np.abs(angle_diff_periodic(th, np.pi / 2.0))
    w_top = smooth_arc_window(d_top, beta, smooth)

    I = np.trapezoid(w_top * np.sin(th), th)
    denom = 1e6 * float(t_m) * float(R_m) * max(float(I), 1e-12)  # MPa->Pa
    return float(P_N) / denom  # MPa


def _build_theta_m_with_arc_oversample(alpha, beta, smooth, N_base=360, N_arc_each=900):
    th_base = np.linspace(-np.pi, np.pi, int(N_base), endpoint=False)

    span = float(beta + smooth)
    thg_top = np.linspace(np.pi / 2.0 - span, np.pi / 2.0 + span, int(N_arc_each), endpoint=True)
    thg_bot = np.linspace(-np.pi / 2.0 - span, -np.pi / 2.0 + span, int(N_arc_each), endpoint=True)

    thm_top = _wrap_pi(thg_top - float(alpha))
    thm_bot = _wrap_pi(thg_bot - float(alpha))

    th = np.concatenate([th_base, thm_top, thm_bot])
    th = np.unique(np.round(th, 14))
    th.sort()
    return th
