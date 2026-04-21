#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bound_all_helpers.py  (AGGREGATOR / RE-EXPORT ONLY)

Single import hub for final_file_to_date.py

IMPORTANT (to avoid circular imports):
- grid_phys_method.py MUST import angle_diff_periodic/_wrap_pi from geometry_helpers,
  NOT from bound_all_helpers.
- hybrid_crack_path_helpers.py should NOT import bound_all_helpers.
"""

from __future__ import annotations

# ----------------------------
# Geometry / angles
# ----------------------------
from geometry_helpers import (
    _wrap_pi,
    angle_diff_periodic,
    map_angle_to_alpha,
    reduce_angle_0_90,
    points_in_disk,
    create_disk_grid,
    unit_tangent_normal,
)

# ----------------------------
# Rotations
# ----------------------------
from rotation_helpers import (
    rot_to_material,
    rot_to_global,
    vec_rot_to_material,
    vec_rot_to_global,
)

# ----------------------------
# Stress helpers
# ----------------------------
from stress_helpers import (
    stress_material_to_global,
    principal_from_components,
    eval_stress_field_material,
)

# ----------------------------
# Airy fitting
# ----------------------------
from airy_solution import fit_orthotropic_airy_disk, fit_orthotropic_airy_disk_auto


# ----------------------------
# Failure mapping
# ----------------------------
from failure_mapping_helpers import (
    failure_mode_map,
    disc_failure_point_stats,
    specimen_mode_from_crack_path,
    mc_scan_ratio,
    plane_sigma_tau,
    failure_ratios_pointwise, weak_plane_weight_field
)

# ----------------------------
# Crack growth (local, ddm helpers)
# ----------------------------
from crack_helpers import (
    crack_path_single_mirror_physics,
    crack_path_single_mirror_ddm,
    _extract_tip_sifs,
    pick_initial_psi0_by_first, _seg_intersect,
    line_angle_diff, Eprime_equiv, G_from_K_equiv,
    _align_line_direction, _angle_mean_line_scalar,
    _angle_mean_line_vec, _line_angle_diff, Gc_theta_weak_plane,
    H_matrix_from_stroh, G_from_K_orthotropic
)

# ----------------------------
# Grid-phys crack method (re-export)
# ----------------------------
try:
    from grid_phys_method import (
        crack_path_grid_phys_mirror,
        compute_psi_pref_field,
    )
except Exception:
    crack_path_grid_phys_mirror = None
    compute_psi_pref_field = None

# ----------------------------
# Hybrid crack path (grid_phys + local refinement)
# ----------------------------
try:
    from hybrid_crack_path_helpers import crack_path_hybrid_mirror_physics
except Exception:
    crack_path_hybrid_mirror_physics = None

# ----------------------------
# DDM SIF solver (optional re-export)
# ----------------------------
from cracked_disk_ddm import (sif_two_tips_from_crack, traction_from_element,
                              _try_call_sif_two_tips, total_traction_at_point_material,
                              kink_angle_pls_anisotropic_from_traction, solve_cracked_disk_correction_ddm,
                              traction_from_element_u, orthotropic_Q_plane_stress)


# ----------------------------
# Momentum crack path (optional)
# ----------------------------
try:
    from momentum_crack_path_helpers import CrackPropagator
except Exception:
    CrackPropagator = None  # safe fallback

# ----------------------------
# Explicit re-exports
# ----------------------------
__all__ = [
    # geometry
    "_wrap_pi",
    "angle_diff_periodic",
    "map_angle_to_alpha",
    "reduce_angle_0_90",
    "points_in_disk",
    "create_disk_grid",
    "unit_tangent_normal",

    # rotations
    "rot_to_material",
    "rot_to_global",
    "vec_rot_to_material",
    "vec_rot_to_global",

    # stress
    "stress_material_to_global",
    "principal_from_components",
    "eval_stress_field_material",

    # airy
    "fit_orthotropic_airy_disk",
    "fit_orthotropic_airy_disk_auto",

    # failure
    "failure_mode_map",
    "disc_failure_point_stats",
    "specimen_mode_from_crack_path",
    "mc_scan_ratio",
    "plane_sigma_tau",
    "failure_ratios_pointwise",
    "weak_plane_weight_field",

    # crack growth (local + ddm)
    "crack_path_single_mirror_physics",
    "crack_path_single_mirror_ddm",
    "_extract_tip_sifs",
    "pick_initial_psi0_by_first",
    "traction_from_element",
    "traction_from_element_u",
    "total_traction_at_point_material",
    "kink_angle_pls_anisotropic_from_traction",
    "solve_cracked_disk_correction_ddm",
    "line_angle_diff", "Eprime_equiv", "G_from_K_equiv",
    "_align_line_direction", "_angle_mean_line_scalar", "_angle_mean_line_vec",
    "_line_angle_diff", "Gc_theta_weak_plane", "orthotropic_Q_plane_stress",

    # grid-phys crack method
    "crack_path_grid_phys_mirror",
    "compute_psi_pref_field",

    # hybrid crack method
    "crack_path_hybrid_mirror_physics",

    # ddm solver
    "sif_two_tips_from_crack",

    # momentum crack
    "CrackPropagator",
]
