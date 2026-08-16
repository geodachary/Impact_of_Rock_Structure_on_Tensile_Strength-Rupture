import numpy as np
from.geometry_helpers import angle_diff_periodic, _wrap_pi
from.lekhnitskii_root import lekh_roots_p, orthotropic_compliances_plane_stress
from.stress_helpers import stress_material_to_global, principal_from_components, eval_stress_field_material
from.boundary_traction_model import platen_tractions_theta, pressure_amplitude_from_load_tapered, _build_theta_m_with_arc_oversample
from.rotation_helpers import rot_to_material, rot_to_global, vec_rot_to_material, vec_rot_to_global


# =====================================================================
# Airy-fit helpers
# =====================================================================

def _traction_target_material(theta_m, R, alpha, beta, smooth, p0, mu=0.0):
    """
    Build target boundary traction vector in MATERIAL x-y components at
    boundary points defined by theta_m (material polar angle).

    Steps:
    - Convert theta_m -> theta_g (global)
    - Compute (tr, tt) in global normal/tangent basis
    - Convert to global Cartesian traction vector
    - Rotate that vector into material coordinates
    """
    theta_m = np.asarray(theta_m, dtype=float)
    theta_g = _wrap_pi(theta_m + float(alpha))

    tr, tt = platen_tractions_theta(theta_g, beta, smooth, p0, mu=mu)  # MPa

    cg = np.cos(theta_g)
    sg = np.sin(theta_g)

    # global normal & tangent
    nxg, nyg = cg, sg
    txg, tyg = -sg, cg

    # traction vector (global cartesian)
    Txg = tr * nxg + tt * txg
    Tyg = tr * nyg + tt * tyg

    # rotate traction into material coords
    Txm, Tym = vec_rot_to_material(Txg, Tyg, alpha)
    return np.asarray(Txm, float), np.asarray(Tym, float)


def _traction_from_airy_material(theta_m, R, p1, p2, a1, a2):
    """
    Compute traction vector (Tx,Ty) on the OUTER boundary in MATERIAL components.

    Uses eval_stress_field_material at boundary points, then traction = sigma*n.
    """
    theta_m = np.asarray(theta_m, dtype=float)
    xm = float(R) * np.cos(theta_m)
    ym = float(R) * np.sin(theta_m)

    sxx, syy, txy = eval_stress_field_material(xm, ym, float(R), p1, p2, a1, a2)

    nx = np.cos(theta_m)
    ny = np.sin(theta_m)

    Tx = sxx * nx + txy * ny
    Ty = txy * nx + syy * ny
    return np.asarray(Tx, float), np.asarray(Ty, float)


def fit_orthotropic_airy_disk(
    E1, E2, nu12, G12,
    R: float, t: float, P: float,
    alpha: float,
    M: int = 24,
    Nbd: int = 480,
    beta_deg: float = 10.0,
    smooth_deg: float = 4.0,
    mu: float = 0.0,
    lam: float = 1e-10,
    Nbd_arc_each: int = 1200,
):
    """
    Least-squares fit of Airy series coefficients (a1,a2) for a disk boundary
    traction profile from platen_tractions_theta.

    Returns dict with keys:
      p1,p2,a1,a2,p0,res_rms_all,R,t,P,alpha
    """
    R = float(R); t = float(t); P = float(P)
    alpha = float(alpha)
    M = int(M)

    beta = np.deg2rad(float(beta_deg))
    smooth = np.deg2rad(float(smooth_deg))

    # characteristic roots
    p1, p2 = lekh_roots_p(float(E1), float(E2), float(nu12), float(G12))

    # boundary angles in MATERIAL coords (oversample platen arcs)
    theta_m = _build_theta_m_with_arc_oversample(
        alpha=alpha,
        beta=beta,
        smooth=smooth,
        N_base=int(Nbd),
        N_arc_each=int(Nbd_arc_each),
    )

    # compute p0 from load so integrated vertical force matches P
    p0 = pressure_amplitude_from_load_tapered(P, t, R, beta, smooth)

    # target traction in material components
    Tx_tar, Ty_tar = _traction_target_material(theta_m, R, alpha, beta, smooth, p0, mu=float(mu))

    # unknowns: a1[m], a2[m] for m=2..M (complex)
    # solve real system for Re/Im parts.
    ms = np.arange(2, M + 1, dtype=int)
    nterm = len(ms)
    nunk = 4 * nterm  # Re/Im for a1 + Re/Im for a2

    # Build matrix by basis evaluation (robust + simple)
    # Equations: Tx(theta_i) = Tx_tar_i, Ty(theta_i) = Ty_tar_i
    npt = len(theta_m)
    A = np.zeros((2 * npt, nunk), dtype=float)
    b = np.zeros((2 * npt,), dtype=float)
    b[0::2] = Tx_tar
    b[1::2] = Ty_tar

    # Precompute geometry
    th = theta_m
    xm = R * np.cos(th)
    ym = R * np.sin(th)
    nx = np.cos(th)
    ny = np.sin(th)

    # Helper: contribution of a single coefficient to traction
    def basis_traction(which: str, m: int, coef: complex):
        # which: "a1" uses p1, "a2" uses p2
        p = p1 if which == "a1" else p2
        z = xm + p * ym
        zh = z / R
        invR2 = 1.0 / (R * R)
        fpp = coef * (m * (m - 1)) * (zh ** (m - 2)) * invR2

        if which == "a1":
            sxx = np.real((p1 ** 2) * fpp)
            syy = np.real(fpp)
            txy = -np.real(p1 * fpp)
        else:
            sxx = np.real((p2 ** 2) * fpp)
            syy = np.real(fpp)
            txy = -np.real(p2 * fpp)

        Tx = sxx * nx + txy * ny
        Ty = txy * nx + syy * ny
        return Tx, Ty

    col = 0
    for j, m in enumerate(ms):
        # a1 real part basis
        Tx, Ty = basis_traction("a1", int(m), 1.0 + 0j)
        A[0::2, col] = Tx
        A[1::2, col] = Ty
        col += 1

        # a1 imag part basis
        Tx, Ty = basis_traction("a1", int(m), 1.0j)
        A[0::2, col] = Tx
        A[1::2, col] = Ty
        col += 1

    for j, m in enumerate(ms):
        # a2 real part basis
        Tx, Ty = basis_traction("a2", int(m), 1.0 + 0j)
        A[0::2, col] = Tx
        A[1::2, col] = Ty
        col += 1

        # a2 imag part basis
        Tx, Ty = basis_traction("a2", int(m), 1.0j)
        A[0::2, col] = Tx
        A[1::2, col] = Ty
        col += 1

    # regularization
    if lam and lam > 0:
        lam = float(lam)
        reg = np.sqrt(lam) * np.eye(nunk, dtype=float)
        A_aug = np.vstack([A, reg])
        b_aug = np.concatenate([b, np.zeros(nunk)])
    else:
        A_aug, b_aug = A, b

    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=1e-12)

    # unpack to complex a1, a2
    a1 = np.zeros((M + 1,), dtype=complex)
    a2 = np.zeros((M + 1,), dtype=complex)

    idx = 0
    # a1
    for m in ms:
        re = x[idx]; im = x[idx + 1]
        a1[int(m)] = re + 1j * im
        idx += 2
    # a2
    for m in ms:
        re = x[idx]; im = x[idx + 1]
        a2[int(m)] = re + 1j * im
        idx += 2

    # residual
    Tx_fit, Ty_fit = _traction_from_airy_material(theta_m, R, p1, p2, a1, a2)
    err = np.sqrt((Tx_fit - Tx_tar) ** 2 + (Ty_fit - Ty_tar) ** 2)
    res_rms_all = float(np.sqrt(np.mean(err ** 2)))

    return {
        "R": R,
        "t": t,
        "P": P,
        "alpha": float(alpha),
        "beta_deg": float(beta_deg),
        "smooth_deg": float(smooth_deg),
        "mu": float(mu),
        "p0": float(p0),
        "p1": p1,
        "p2": p2,
        "a1": a1,
        "a2": a2,
        "res_rms_all": res_rms_all,
        "theta_m_fit": np.asarray(theta_m, float),
    }


def fit_orthotropic_airy_disk_auto(
    E1, E2, nu12, G12,
    R: float, t: float, P: float,
    alpha: float,
    beta_deg: float = 10.0,
    smooth_deg: float = 4.0,
    mu: float = 0.0,
    Nbd: int = 480,
    Nbd_arc_each: int = 1200,
):
    """
    Light-weight "auto" wrapper. (You can extend it later to sweep M/lam.)
    """
    return fit_orthotropic_airy_disk(
        E1, E2, nu12, G12,
        R=R, t=t, P=P,
        alpha=alpha,
        M=24,
        Nbd=Nbd,
        beta_deg=beta_deg,
        smooth_deg=smooth_deg,
        mu=mu,
        lam=1e-10,
        Nbd_arc_each=Nbd_arc_each,
    )