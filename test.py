#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brazilian disk phase-field fracture (AT2) with orthotropic plane-stress elasticity in FEniCSx (dolfinx)

What this gives you:
- Real fracture evolution (variational regularized Griffith)
- Irreversibility via history field H (hybrid formulation, robust & "publishable baseline")
- Orthotropic stiffness rotated by foliation angle alpha

Boundary condition:
- Displacement-controlled compression on top/bottom platen arcs (±u0 in y)
- Small pin constraints to remove rigid body motion

Outputs:
- XDMF of displacement u and damage d for ParaView

References for patterns:
- dolfinx gmsh mesh + tags: official demo_gmsh pattern.  (see docs) 
- dolfinx BC patterns: official demos (locate_entities, locate_dofs, dirichletbc).
"""

from mpi4py import MPI
import numpy as np

import ufl
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.io import gmshio

try:
    import gmsh  # type: ignore
except ImportError:
    raise RuntimeError("gmsh is required (pip install gmsh, or use a dolfinx docker image).")


# ----------------------------
# Gmsh mesh: disk with arc tags
# ----------------------------
def build_disk_with_platen_arcs(R: float, lc: float, beta_deg: float):
    """
    Build a 2D disk of radius R.
    Mark boundary arcs:
      tag 1 = top platen arc centered at +pi/2
      tag 2 = bottom platen arc centered at -pi/2
      tag 3 = remaining boundary
    """
    beta = np.deg2rad(beta_deg)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model.add("brazilian_disk")
    model.setCurrent("brazilian_disk")

    # Points to split boundary into arcs
    def pt(theta):
        return model.occ.addPoint(R*np.cos(theta), R*np.sin(theta), 0.0, lc)

    # Split angles
    th_top1 = np.pi/2 - beta
    th_top2 = np.pi/2 + beta
    th_bot1 = -np.pi/2 - beta
    th_bot2 = -np.pi/2 + beta

    p0 = model.occ.addPoint(0, 0, 0, lc)
    pA = pt(th_top1)
    pB = pt(th_top2)
    pC = pt(th_bot2)
    pD = pt(th_bot1)

    # Create circle arcs (A->B top arc, B->C side, C->D bottom arc, D->A side)
    # Need center point p0 as circle center in OCC circle arc
    arc_top = model.occ.addCircleArc(pA, p0, pB)
    arc_right = model.occ.addCircleArc(pB, p0, pC)
    arc_bot = model.occ.addCircleArc(pC, p0, pD)
    arc_left = model.occ.addCircleArc(pD, p0, pA)

    cloop = model.occ.addCurveLoop([arc_top, arc_right, arc_bot, arc_left])
    surf = model.occ.addPlaneSurface([cloop])

    model.occ.synchronize()

    # Physical groups: surface + boundary parts
    domain_tag = 1
    top_tag = 1
    bot_tag = 2
    rest_tag = 3

    model.addPhysicalGroup(2, [surf], tag=domain_tag)
    model.setPhysicalName(2, domain_tag, "disk")

    model.addPhysicalGroup(1, [arc_top], tag=top_tag)
    model.setPhysicalName(1, top_tag, "top_platen")

    model.addPhysicalGroup(1, [arc_bot], tag=bot_tag)
    model.setPhysicalName(1, bot_tag, "bottom_platen")

    model.addPhysicalGroup(1, [arc_left, arc_right], tag=rest_tag)
    model.setPhysicalName(1, rest_tag, "free_boundary")

    model.mesh.generate(2)
    return model


# -----------------------------------------
# Orthotropic plane-stress stiffness (Qbar)
# -----------------------------------------
def orthotropic_Qbar_plane_stress(E1, E2, nu12, G12, alpha):
    """
    Returns 3x3 plane-stress stiffness in GLOBAL (x,y) in Voigt form:
      [sxx, syy, txy]^T = Qbar * [exx, eyy, gxy]^T   with gxy = 2*exy

    Uses standard lamina transformation formulas (composite mechanics).
    """
    nu21 = (E2/E1) * nu12
    denom = 1.0 - nu12*nu21
    Q11 = E1/denom
    Q22 = E2/denom
    Q12 = nu12*E2/denom
    Q66 = G12

    m = np.cos(alpha)
    n = np.sin(alpha)

    Qbar11 = Q11*m**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*n**4
    Qbar22 = Q11*n**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*m**4
    Qbar12 = (Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4)
    Qbar16 = (Q11-Q12-2*Q66)*m**3*n - (Q22-Q12-2*Q66)*m*n**3
    Qbar26 = (Q11-Q12-2*Q66)*m*n**3 - (Q22-Q12-2*Q66)*m**3*n
    Qbar66 = (Q11+Q22-2*Q12-2*Q66)*m**2*n**2 + Q66*(m**4+n**4)

    Qbar = np.array([[Qbar11, Qbar12, Qbar16],
                     [Qbar12, Qbar22, Qbar26],
                     [Qbar16, Qbar26, Qbar66]], dtype=float)
    return Qbar


# ----------------------------
# Stress/energy helper in UFL
# ----------------------------
def sigma_from_u(u, Qbar):
    eps = ufl.sym(ufl.grad(u))
    epsv = ufl.as_vector([eps[0, 0], eps[1, 1], 2*eps[0, 1]])   # gxy = 2*exy
    Q = ufl.as_matrix(Qbar)
    sv = ufl.dot(Q, epsv)  # [sxx, syy, txy]
    s = ufl.as_tensor([[sv[0], sv[2]],
                       [sv[2], sv[1]]])
    return s, eps


def principal_stress_positive_energy(sig, eps, tiny=1e-16):
    """
    Compute tensile-driving energy density proxy psi_plus from positive principal stresses:
      - analytical eigenvalues for 2x2 symmetric tensor
      - reconstruct sigma_plus and compute 0.5*sigma_plus:eps
    """
    sxx = sig[0, 0]
    syy = sig[1, 1]
    txy = sig[0, 1]
    sm = 0.5*(sxx + syy)
    sd = 0.5*(sxx - syy)
    rad = ufl.sqrt(sd*sd + txy*txy + tiny)
    s1 = sm + rad
    s2 = sm - rad

    theta = 0.5*ufl.atan_2(2*txy, (sxx - syy))
    c = ufl.cos(theta)
    s = ufl.sin(theta)
    n1 = ufl.as_vector([c, s])
    n2 = ufl.as_vector([-s, c])

    s1p = ufl.max_value(s1, 0.0)
    s2p = ufl.max_value(s2, 0.0)

    sig_plus = s1p*ufl.outer(n1, n1) + s2p*ufl.outer(n2, n2)
    psi_plus = 0.5*ufl.inner(sig_plus, eps)
    return psi_plus


# -------------
# Main solve
# -------------
def main():
    # --- Geometry / mesh ---
    D = 0.054     # m (example)
    R = D/2
    beta_deg = 10.0
    lc = R/35     # mesh size

    # --- Material (units: Pa) ---
    # Use your measured values; below are placeholders.
    E1 = 50e9
    E2 = 30e9
    nu12 = 0.25
    G12 = 12e9
    alpha = 0.0   # foliation x-axis angle in GLOBAL coordinates (radians)

    # --- Phase-field ---
    Gc = 40.0      # J/m^2 (placeholder; you MUST calibrate)
    ell = 0.0008   # m (regularization length; mesh should resolve ~3-5 elements across ell)
    kappa = 1e-8   # residual stiffness

    # --- Loading ---
    nsteps = 40
    u_max = 0.25e-3  # m (total imposed half-displacement)

    # Build gmsh model and convert to dolfinx mesh + facet tags
    model = build_disk_with_platen_arcs(R=R, lc=lc, beta_deg=beta_deg)
    msh, ct, ft = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0)
    gmsh.finalize()

    # Function spaces
    V = fem.functionspace(msh, ("CG", 1, (msh.geometry.dim,)))  # vector
    Q = fem.functionspace(msh, ("CG", 1))                      # scalar

    u = fem.Function(V, name="u")
    d = fem.Function(Q, name="d")   # damage: 0 intact, 1 broken
    d_old = fem.Function(Q, name="d_old")
    H = fem.Function(Q, name="H")   # history field (tensile driving energy)

    v = ufl.TestFunction(V)
    eta = ufl.TestFunction(Q)

    # Measures with facet tags
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    dx = ufl.Measure("dx", domain=msh)

    # Stiffness (global Qbar)
    Qbar = orthotropic_Qbar_plane_stress(E1, E2, nu12, G12, alpha)

    # Degradation
    g = (1.0 - d)**2 + kappa

    # Elastic stress/strain
    sig0, eps = sigma_from_u(u, Qbar)
    psi0 = 0.5*ufl.inner(sig0, eps)

    # Tensile driving proxy for history field
    psi_plus = principal_stress_positive_energy(sig0, eps)

    # ---- Mechanics problem (linear “hybrid” equilibrium) ----
    # a_u(u,v) = ∫ g(d) * sigma0(u):eps(v) dx
    a_u = ufl.inner(g * sig0, ufl.sym(ufl.grad(v))) * dx
    L_u = ufl.dot(ufl.as_vector((0.0, 0.0)), v) * dx

    # ---- Damage problem (linear with history field) ----
    # (Gc/ell + 2H) d eta + Gc ell grad(d)·grad(eta) = 2H eta
    a_d = (Gc/ell + 2.0*H) * d * eta * dx + (Gc*ell) * ufl.dot(ufl.grad(d), ufl.grad(eta)) * dx
    L_d = 2.0*H * eta * dx

    # Boundary tags from gmsh physical groups
    TOP = 1
    BOT = 2

    # Dirichlet BC: compress top/bottom arcs in y
    # Locate dofs on facets with given tag
    fdim = msh.topology.dim - 1
    top_facets = ft.find(TOP)
    bot_facets = ft.find(BOT)

    # Subspace dofs for y-component
    Vy = V.sub(1)
    top_dofs_y = fem.locate_dofs_topological(Vy, fdim, top_facets)
    bot_dofs_y = fem.locate_dofs_topological(Vy, fdim, bot_facets)

    # Rigid body fix: pin ux at leftmost point and uy at center point (or another point)
    # (simple, pragmatic; you can do cleaner constraints later)
    def near(a, b, tol=1e-6):
        return np.isclose(a, b, atol=tol)

    # point at (-R,0)
    def left_point(x):
        return np.logical_and(near(x[0], -R), near(x[1], 0.0))

    # point at (0,0)
    def origin_point(x):
        return np.logical_and(near(x[0], 0.0), near(x[1], 0.0))

    Vx = V.sub(0)
    left_dofs_x = fem.locate_dofs_geometrical(Vx, left_point)
    org_dofs_y = fem.locate_dofs_geometrical(Vy, origin_point)

    # PETSc solvers
    # Mechanics
    problem_u = fem.petsc.LinearProblem(
        fem.form(a_u), fem.form(L_u),
        bcs=[],  # set per step
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # Damage
    problem_d = fem.petsc.LinearProblem(
        fem.form(a_d), fem.form(L_d),
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # Output
    xdmf = io.XDMFFile(msh.comm, "brazilian_phasefield.xdmf", "w")
    xdmf.write_mesh(msh)

    # Load steps + staggered iterations
    for i in range(nsteps + 1):
        u0 = (i / nsteps) * u_max

        # Update BC values
        top_val = fem.Constant(msh, PETSc.ScalarType(-u0))
        bot_val = fem.Constant(msh, PETSc.ScalarType(+u0))
        zero = fem.Constant(msh, PETSc.ScalarType(0.0))

        bc_top = fem.dirichletbc(top_val, top_dofs_y, Vy)
        bc_bot = fem.dirichletbc(bot_val, bot_dofs_y, Vy)
        bc_fixx = fem.dirichletbc(zero, left_dofs_x, Vx)
        bc_fixy = fem.dirichletbc(zero, org_dofs_y, Vy)

        bcs_u = [bc_top, bc_bot, bc_fixx, bc_fixy]

        # Stagger iterations at each load step
        d_old.x.array[:] = d.x.array
        for it in range(50):
            # 1) Solve mechanics with current d
            problem_u.bcs = bcs_u
            u = problem_u.solve()

            # 2) Update history field H = max(H, projection(psi_plus(u)))
            #    We interpolate psi_plus into Q (nodal) then max with old H.
            expr = fem.Expression(psi_plus, Q.element.interpolation_points())
            psi_plus_fun = fem.Function(Q)
            psi_plus_fun.interpolate(expr)
            H.x.array[:] = np.maximum(H.x.array, psi_plus_fun.x.array)

            # 3) Solve damage with updated H (linear)
            d = problem_d.solve()

            # 4) Enforce irreversibility by clamping: d >= d_old
            d.x.array[:] = np.maximum(d.x.array, d_old.x.array)

            # Convergence check (damage change)
            err = np.linalg.norm(d.x.array - d_old.x.array, ord=np.inf)
            d_old.x.array[:] = d.x.array
            if msh.comm.rank == 0:
                print(f"step {i:03d}/{nsteps}  iter {it:02d}  u0={u0:.4e}  ||Δd||inf={err:.3e}")
            if err < 1e-4:
                break

        # Write fields
        xdmf.write_function(u, i)
        xdmf.write_function(d, i)

    xdmf.close()


if __name__ == "__main__":
    main()
