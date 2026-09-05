"""Plane-stress orthotropic compliance and the Lekhnitskii characteristic roots.

The roots are properties of the elastic constants alone. They are computed in
the material frame, where the foliation lies along the axes and the shear
coupling terms vanish, so the characteristic equation reduces to a biquadratic
in ``p``. Loading orientation never enters here: it is carried by the boundary
tractions, which are rotated into the material frame before the Airy fit, and
by the rotation of the interior stresses back to the global frame afterwards.
One pair of roots therefore serves every orientation of a given lithology.
"""
import numpy as np

# =========================
# Orthotropic compliance + Lekhnitskii roots
# =========================

# Isotropy makes the characteristic equation degenerate. With s11 = s22 and
# 2*s12 + s66 = 2*s11, the quadratic in u = p^2 collapses to (u + 1)^2 = 0, a
# repeated root at p = i. Lekhnitskii's two-potential form presumes distinct
# roots; the degenerate case needs the separate formulation built on F1(z1)
# and conj(z1)*F2(z1). Carrying that second formulation is not worth it for a
# limit no real rock occupies, but leaving the degeneracy unhandled is worse:
# the root search then finds only one root in the upper half plane and the
# fallback returned the conjugate pair (i, -i), which forces sxx = -syy
# identically and silently produces a field that is not the Brazilian
# solution at all.
#
# The degeneracy is instead broken by a small relative perturbation of s22.
# The perturbed solution converges: the Brazilian centre-stress ratio comes
# out at -3.09 against the classical -3, and is stable for perturbations from
# 1e-5 down to 1e-7, so the residual offset is the finite platen arc rather
# than the perturbation itself.
DEGENERACY_TOL = 1e-9          # relative discriminant treated as a repeated root
DEGENERACY_PERTURBATION = 1e-6  # relative separation applied to s22


def orthotropic_compliances_plane_stress(E1, E2, nu12, G12):
    s11 = 1.0/E1
    s22 = 1.0/E2
    s12 = -nu12/E1
    s66 = 1.0/G12
    return s11, s22, s12, s66


def _roots_from_quadratic(a, b, c):
    """The two upper-half-plane roots p of a*p^4 + b*p^2 + c = 0, if they exist."""
    u = np.roots([a, b, c]).astype(np.complex128)
    cand = []
    for ui in u:
        p = np.sqrt(ui)
        cand.extend([p, -p])
    cand = [p for p in cand if np.isfinite(p.real) and np.isfinite(p.imag)]
    uniq = []
    for p in cand:
        if all(abs(p - q) > 1e-10 for q in uniq):
            uniq.append(p)
    pos = [p for p in uniq if np.imag(p) > 1e-12]
    return sorted(pos, key=lambda z: (np.imag(z), np.real(z)), reverse=True)


def lekh_roots_p(E1, E2, nu12, G12):
    """Return the two Lekhnitskii roots, both in the upper half plane.

    Both roots must have positive imaginary part for the two-potential
    representation to span the solution space. A pair straddling the real axis
    is not a valid basis, so it is rejected rather than returned.
    """
    s11, s22, s12, s66 = orthotropic_compliances_plane_stress(E1, E2, nu12, G12)
    a = float(s11)
    b = float(2*s12 + s66)
    c = float(s22)

    # A vanishing discriminant means p is a repeated root; see the note above.
    disc = b*b - 4.0*a*c
    scale = b*b + 4.0*abs(a*c)
    if scale > 0.0 and abs(disc) <= DEGENERACY_TOL * scale:
        c *= (1.0 + DEGENERACY_PERTURBATION)

    pos = _roots_from_quadratic(a, b, c)
    if len(pos) >= 2:
        return pos[0], pos[1]
    raise RuntimeError(
        "Could not obtain two Lekhnitskii roots with positive imaginary part "
        f"for E1={E1!r}, E2={E2!r}, nu12={nu12!r}, G12={G12!r}. The "
        "two-potential form is not valid for these constants.")
