import numpy as np
from.geometry_helpers import angle_diff_periodic, _wrap_pi
from.lekhnitskii_root import lekh_roots_p, orthotropic_compliances_plane_stress
from.stress_helpers import stress_material_to_global, principal_from_components, eval_stress_field_material
from.boundary_traction_model import platen_tractions_theta, pressure_amplitude_from_load_tapered, _build_theta_m_with_arc_oversample
from.rotation_helpers import rot_to_material, rot_to_global, vec_rot_to_material, vec_rot_to_global


#: Airy series truncation used everywhere. Raised from 24 after the convergence
#: study in ``scripts/make_mesh_sensitivity_figure.py``, which writes
#: ``outputs/tables/convergence_series_order.csv``: at 24 the relative
#: boundary-traction residual is 6.6e-2 and the interior field is still 22%
#: from converged over the analysis core; by 48 the residual is 6.3e-3 and the
#: core field is within 0.27% of the M = 96 reference the study writes.
#:
#: The fit does not become ill-conditioned beyond ~52, as this note used to
#: claim. That was true at the old Tikhonov weight of 1e-10 and before the
#: design matrix was column-equilibrated; the residual now falls monotonically
#: to at least M = 96. The order is capped because the core field has stopped
#: moving, not because the solve degrades.
#:
#: Every call site must take this default. Both the auto wrapper below and the
#: production suite once carried their own ``M=24`` literal, which pinned every
#: published field at 24 no matter what this said.
DEFAULT_M = 48

#: Tikhonov weight for the boundary fit. Small because the design matrix is
#: column-equilibrated before the solve; see the note in
#: :func:`fit_orthotropic_airy_disk`. Every call site must take this default.
DEFAULT_LAM = 1e-18

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
    M: int = DEFAULT_M,
    Nbd: int = 480,
    beta_deg: float = 10.0,
    smooth_deg: float = 4.0,
    mu: float = 0.0,
    lam: float = DEFAULT_LAM,
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

    # Column equilibration before regularising.
    #
    # The basis is (z/R)^(m-2) with |z/R| bounded by |p|, so the column norms
    # grow like |p|^M. For the complex-root materials |p| is about 1.2 and at
    # M = 48 that is a spread of ~10^3, which a uniform Tikhonov weight
    # tolerates. A material whose shear modulus is low enough to drive the
    # characteristic equation into its real-root regime has a larger |p|: the
    # psammitic schist at 0 degrees, with a measured G12 half the isotropic
    # estimate, gives p = 1.816i and 1.068i and a spread of ~10^12. The columns
    # then span thirty orders of magnitude, a single weight annihilates the
    # small ones, the effective rank falls from 187 to 74 of 188, and the fit
    # returns a field of zeros with a boundary residual of 0.78 against 0.006
    # everywhere else.
    #
    # Scaling each column to unit norm makes the penalty act uniformly on the
    # coefficients rather than on their arbitrary basis scaling, and the
    # solution is unscaled afterwards, so nothing about the model changes. With
    # the system equilibrated the weight can also be far smaller, and every
    # specimen then reaches the same residual.
    col_norm = np.linalg.norm(A, axis=0)
    col_norm[col_norm == 0.0] = 1.0
    A_scaled = A / col_norm

    if lam and lam > 0:
        lam = float(lam)
        reg = np.sqrt(lam) * np.eye(nunk, dtype=float)
        A_aug = np.vstack([A_scaled, reg])
        b_aug = np.concatenate([b, np.zeros(nunk)])
    else:
        A_aug, b_aug = A_scaled, b

    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    x = x / col_norm

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
    M: int = DEFAULT_M,
    Nbd: int = 480,
    Nbd_arc_each: int = 1200,
):
    """
    Light-weight "auto" wrapper. (You can extend it later to sweep M/lam.)

    This is the production path, so ``M`` must track :data:`DEFAULT_M`.
    """
    return fit_orthotropic_airy_disk(
        E1, E2, nu12, G12,
        R=R, t=t, P=P,
        alpha=alpha,
        M=int(M),
        Nbd=Nbd,
        beta_deg=beta_deg,
        smooth_deg=smooth_deg,
        mu=mu,
        lam=DEFAULT_LAM,
        Nbd_arc_each=Nbd_arc_each,
    )