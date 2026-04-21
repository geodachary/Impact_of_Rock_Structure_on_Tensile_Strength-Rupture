import numpy as np
from cracked_disk_ddm import (
    orthotropic_Q_plane_stress, rotate_Q, compliance_from_Q,
    reduced_compliance_components, p_roots_quartic, stroh_LA_inplane
)

def H_energy_barnett_lothe(E1, E2, nu12, G12, alpha_const, psi_tip_global):
    """
    Build a Stroh/Barnett–Lothe-based energy matrix H such that:
        G = [KI, KII] @ H @ [KI, KII]^T
    """

    # crack direction in MATERIAL coords
    psi_m = float(psi_tip_global) - float(alpha_const)

    # stiffness in crack-local frame (material -> crack local)
    Qm = orthotropic_Q_plane_stress(E1, E2, nu12, G12)
    Qbar = rotate_Q(Qm, psi_m)
    Sbar = compliance_from_Q(Qbar)
    S11, S22, S12, S16, S26, S66 = reduced_compliance_components(Sbar)

    # Stroh roots and eigenvectors
    p1, p2 = p_roots_quartic(S11, S22, S12, S16, S26, S66)

    # Your function returns (L, A). In standard notation, treat:
    #   B ≈ L  (traction eigenvectors),  A ≈ A (displacement eigenvectors)
    B, A = stroh_LA_inplane(S11, S22, S12, S16, S26, S66, p1, p2)

    # Barnett–Lothe L tensor from traction eigenvectors
    # (should be real symmetric under proper Stroh normalization)
    L_BL = np.real(-2j * (B @ B.T))

    # Energy matrix
    H = 0.5 * np.linalg.inv(L_BL)

    return H

def G_from_K(KI, KII, H):
    k = np.array([float(KI), float(KII)], dtype=float)
    return float(k @ H @ k)
