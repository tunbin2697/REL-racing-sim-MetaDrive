import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from game_env.racing_env import MetaRacingEnv
from agents.optimal_agent import OptimalRacingAgent

class FastOptimalAgent(OptimalRacingAgent):
    def __init__(self, a_brake_safe=5.80, max_scan_m=52.0, maint_thr=0.45, exit_boost=True):
        super().__init__()
        self.a_brake_safe = float(a_brake_safe)
        self.max_scan_m = float(max_scan_m)
        self.maint_thr = float(maint_thr)
        self.exit_boost = bool(exit_boost)

    def select_action(self, obs=None, vehicle=None, pos_2d=None, heading=None, speed_km_h=None, deterministic=True):
        if vehicle is not None:
            p_curr = np.array(vehicle.position[:2], dtype=np.float32)
            th_curr = float(vehicle.heading_theta)
            vel_vec = np.array(vehicle.velocity[:2], dtype=np.float32)
            v_curr_kmh = float(np.linalg.norm(vel_vec) * 3.6)
        else:
            return super().select_action(obs, vehicle, pos_2d, heading, speed_km_h, deterministic)

        window = 100
        indices = np.arange(self.last_hint_idx - 20, self.last_hint_idx + window) % self.num_points
        d_sq = np.sum((self.optimal_line_xy[indices] - p_curr) ** 2, axis=1)
        min_local = int(np.argmin(d_sq))
        idx_near = int(indices[min_local])
        self.last_hint_idx = idx_near

        wheelbase = 2.75
        ds = max(0.5, self.circuit_length / self.num_points)
        is_grid_launch = bool(self.track_s[idx_near] < 180.0 and v_curr_kmh < 110.0)
        p_near = self.optimal_line_xy[idx_near]
        lat_err = float(np.linalg.norm(p_curr - p_near))
        base_min_ld = 14.0 if is_grid_launch else (max(4.2, 2.2 * lat_err) if v_curr_kmh < 50.0 else 4.2)
        L_d = float(np.clip(max(base_min_ld, 3.8 + 0.10 * v_curr_kmh), 4.2, 20.0))

        lookahead_steps = int(round(L_d / ds))
        idx_target = (idx_near + lookahead_steps) % self.num_points
        p_target = self.optimal_line_xy[idx_target]

        dx = float(p_target[0] - p_curr[0])
        dy = float(p_target[1] - p_curr[1])
        cos_th = float(np.cos(th_curr))
        sin_th = float(np.sin(th_curr))
        x_body = cos_th * dx + sin_th * dy
        y_body = -sin_th * dx + cos_th * dy
        actual_lookahead_sq = max(4.0, x_body ** 2 + y_body ** 2)

        curvature = 2.0 * y_body / actual_lookahead_sq
        steer_rad = np.arctan(curvature * wheelbase)
        steer_norm = float(np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0))

        # Phanh siêu muộn (Super Late Braking)
        n_scan = min(self.num_points - 1, max(5, int(round(self.max_scan_m / ds))))
        scan_indices = (idx_near + np.arange(1, n_scan + 1)) % self.num_points
        scan_speeds_ms = self.optimal_speed_kmh[scan_indices] / 3.6
        scan_dists_m = np.arange(1, n_scan + 1) * ds

        allowed_speeds_ms = np.sqrt(scan_speeds_ms ** 2 + 2.0 * self.a_brake_safe * scan_dists_m)
        v_target_ms = min(float(self.optimal_speed_kmh[idx_near] / 3.6), float(np.min(allowed_speeds_ms)))
        v_target_kmh = float(v_target_ms * 3.6)
        v_err_kmh = v_target_kmh - v_curr_kmh

        steer_mag = abs(steer_norm)
        avail_long_grip = float(np.sqrt(max(0.10, 1.0 - min(0.90, (0.90 * steer_mag) ** 2))))
        traction_factor = float(np.sqrt(max(0.45, 1.0 - min(0.75, (0.75 * steer_mag) ** 2))))

        if is_grid_launch:
            target_brake = 0.0
            target_throttle = 1.0
        elif v_err_kmh >= 0.0:
            target_brake = 0.0
            # Mở ga thoát cua siêu dứt khoát
            if (steer_mag < 0.50 and v_curr_kmh < 100.0) or (steer_mag < 0.28 and v_curr_kmh < 119.5):
                target_throttle = 1.0
            else:
                target_throttle = float(np.clip(self.maint_thr + (v_err_kmh / 3.5) * (1.0 - self.maint_thr), 0.0, 1.0)) * traction_factor
        else:
            overspeed_kmh = -v_err_kmh
            if overspeed_kmh <= 2.2:
                target_throttle = 0.0
                target_brake = 0.0
            else:
                target_throttle = 0.0
                brk_fraction = float(np.clip((overspeed_kmh - 2.2) / 5.0, 0.0, 1.0))
                max_brk_pedal = getattr(self, "max_brk_pedal", 0.95)
                target_brake = float(brk_fraction * max_brk_pedal * avail_long_grip)

        alpha_thr = 0.90
        alpha_brk = 0.85
        self.filtered_throttle = float((1.0 - alpha_thr) * self.filtered_throttle + alpha_thr * target_throttle)
        self.filtered_brake = float((1.0 - alpha_brk) * self.filtered_brake + alpha_brk * target_brake)

        throttle_action = float(np.clip(self.filtered_throttle * 2.0 - 1.0, -1.0, 1.0))
        brake_action = float(np.clip(self.filtered_brake * 2.0 - 1.0, -1.0, 1.0))
        action = np.array([steer_norm, throttle_action, brake_action], dtype=np.float32)
        net_pedal = float(self.filtered_throttle - self.filtered_brake)
        return action, steer_norm, net_pedal

def run_experiment(a_brake_safe=5.80, max_scan_m=52.0, maint_thr=0.45):
    env = MetaRacingEnv(use_render=False, max_laps=2)
    obs, info = env.reset(seed=42)
    
    agent = FastOptimalAgent(a_brake_safe=a_brake_safe, max_scan_m=max_scan_m, maint_thr=maint_thr)
    agent.reset()

    lap1_time = 0.0
    lap2_time = 0.0
    max_lat_err = 0.0
    in_lap2 = False

    for step in range(1, 3000):
        veh = env.car.raw_vehicle
        
        # Override select action logic trực tiếp cho thực nghiệm
        action, _, _ = agent.select_action(vehicle=veh)
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        if step_info.get("completed_lap_number", 0) == 1 and not in_lap2:
            in_lap2 = True
            lap1_time = step_info.get("completed_lap_time", 0.0)

        if in_lap2:
            pos = env.car.position
            idx, s = env._get_track_s(pos)
            dist = float(np.linalg.norm(pos - env.track_xy[idx]))
            if dist > max_lat_err:
                max_lat_err = dist
                
            if step_info.get("completed_lap_number", 0) == 2:
                lap2_time = step_info.get("completed_lap_time", 0.0)
                break

        if terminated or truncated:
            break

    env.close()
    return lap1_time, lap2_time, max_lat_err

if __name__ == "__main__":
    l1, l2, max_lat = run_experiment(a_brake_safe=5.80, max_scan_m=55.0, maint_thr=0.45)
    print(f"EXPERIMENT RESULT: Lap 1 = {l1:.2f}s | Lap 2 (Flying) = {l2:.2f}s | Max Lat Error = {max_lat:.2f}m")
