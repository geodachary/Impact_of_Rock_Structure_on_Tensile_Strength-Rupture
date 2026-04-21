import numpy as np

from rotation_helpers import rot_to_material, rot_to_global
from stress_helpers import eval_stress_field_material, stress_material_to_global, principal_from_components
from failure_mapping_helpers import failure_mode_map, mc_scan_ratio, plane_sigma_tau, failure_ratios_pointwise

# =========================
# NOVEL: MOMENTUM CRACK PATH
# =========================
class CrackPropagator:
    def __init__(self, R, alpha, fit, props):
        self.R = R
        self.alpha = alpha
        self.fit = fit
        self.props = props # [Tm, Coh, Phi]
        self.momentum_vec = np.array([0.0, 0.0])

    def get_util(self, x, y):
        xm, ym = rot_to_material(x, y, self.alpha)
        sxx_m, syy_m, txy_m = eval_stress_field_material(xm, ym, self.R, self.fit["p1"], self.fit["p2"], self.fit["a1"], self.fit["a2"])
        sxx, syy, txy = stress_material_to_global(sxx_m, syy_m, txy_m, self.alpha)
        s1, _, _ = principal_from_components(sxx, syy, txy)
        Rt, Rs = failure_ratios_pointwise(sxx, syy, txy, s1, *self.props, self.alpha)
        return max(Rt, Rs)

    def solve(self, ds_frac=0.01, max_steps=400):
        # Start at center, move slightly towards Max Tensile Stress
        xs, ys = [0.0], [0.0]
        ds = self.R * ds_frac
        
        # Initial direction (30 deg fix: ensure we start vertical-ish)
        curr_x, curr_y = 0.0, 0.0
        angle = np.pi/2 # Start vertical
        
        for _ in range(max_steps):
            best_score = -1e9
            best_angle = angle
            
            # NOVEL: Dynamic Search Cone + Momentum
            # We look at candidates in a 60 degree cone
            for da in np.linspace(-np.deg2rad(30), np.deg2rad(30), 20):
                cand_a = angle + da
                nx, ny = curr_x + ds*np.cos(cand_a), curr_y + ds*np.sin(cand_a)
                
                if (nx**2 + ny**2) > (self.R * 0.98)**2: continue
                
                # Utilization at candidate
                u = self.get_util(nx, ny)
                
                # Penalty for turning (Momentum)
                turn_penalty = abs(da) * 0.2
                
                # NOVEL: Look-ahead check (prevents getting stuck at 30 deg)
                future_x, future_y = nx + ds*np.cos(cand_a), ny + ds*np.sin(cand_a)
                u_future = self.get_util(future_x, future_y)
                
                score = (u * 0.6 + u_future * 0.4) - turn_penalty
                
                if score > best_score:
                    best_score = score
                    best_angle = cand_a
            
            angle = best_angle
            curr_x += ds * np.cos(angle)
            curr_y += ds * np.sin(angle)
            xs.append(curr_x); ys.append(curr_y)
            
            if best_score < 0.8: break # Termination
            
        # Mirror for Brazilian symmetry
        xs = np.array(xs); ys = np.array(ys)
        return np.concatenate([-xs[::-1], xs]), np.concatenate([-ys[::-1], ys])