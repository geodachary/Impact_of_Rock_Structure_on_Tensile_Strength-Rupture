import numpy as np

# =========================
# Orthotropic compliance + Lekhnitskii roots
# =========================
def orthotropic_compliances_plane_stress(E1, E2, nu12, G12):
    s11 = 1.0/E1
    s22 = 1.0/E2
    s12 = -nu12/E1
    s66 = 1.0/G12
    return s11, s22, s12, s66


def lekh_roots_p(E1, E2, nu12, G12):
    s11, s22, s12, s66 = orthotropic_compliances_plane_stress(E1, E2, nu12, G12)
    a = float(s11)
    b = float(2*s12 + s66)
    c = float(s22)
    u = np.roots([a, b, c]).astype(np.complex128)
    p_candidates = []
    for ui in u:
        p = np.sqrt(ui)
        p_candidates.extend([p, -p])
    p_candidates = [p for p in p_candidates if np.isfinite(p.real) and np.isfinite(p.imag)]
    uniq = []
    for p in p_candidates:
        if all(abs(p-q) > 1e-10 for q in uniq):
            uniq.append(p)
    p_pos = [p for p in uniq if np.imag(p) > 1e-12]
    if len(p_pos) >= 2:
        p_pos = sorted(p_pos, key=lambda z: np.imag(z), reverse=True)
        return p_pos[0], p_pos[1]
    uniq = sorted(uniq, key=lambda z: (abs(np.imag(z)), abs(np.real(z))), reverse=True)
    if len(uniq) < 2:
        raise RuntimeError("Could not obtain two valid Lekhnitskii roots p1,p2.")
    return uniq[0], uniq[1]