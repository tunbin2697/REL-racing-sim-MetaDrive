import sys
from pathlib import Path
import os
import io

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from agents.optimal_agent import OptimalRacingAgent

def evaluate_candidate(params, seed=42):
    race_car = get_vehicle("gt3")
    env = MetaRacingEnv(use_render=False, vehicle=race_car)
    agent = OptimalRacingAgent()
    agent.reset()

    def custom_select_action(obs=None, vehicle=None, pos_2d=None, heading=None, speed_km_h=None, deterministic=True):
        if vehicle is not None:
            p_curr = np.array(vehicle.position[:2], dtype=np.float32)
            th_curr = float(vehicle.heading_theta)
            vel_vec = np.array(vehicle.velocity[:2], dtype=np.float32)
            v_curr_kmh = float(np.linalg.norm(vel_vec) * 3.6)
        elif pos_2d is not None and heading is not None and speed_km_h is not None:
            p_curr = np.array(pos_2d[:2], dtype=np.float32)
            th_curr = float(heading)
            v_curr_kmh = float(speed_km_h)
        else:
            return np.array([0.0, 1.0, -1.0], dtype=np.float32), 0.0, 0.0

        window = 100
        indices = np.arange(agent.last_hint_idx - 20, agent.last_hint_idx + window) % agent.num_points
        d_sq = np.sum((agent.optimal_line_xy[indices] - p_curr) ** 2, axis=1)
        min_local = int(np.argmin(d_sq))
        idx_near = int(indices[min_local])
        agent.last_hint_idx = idx_near

        wheelbase = 2.75
        ds = max(0.5, agent.circuit_length / agent.num_points)
        is_grid_launch = bool(agent.track_s[idx_near] < 180.0 and v_curr_kmh < 110.0)
        p_near = agent.optimal_line_xy[idx_near]
        k_near = abs(float(agent.optimal_line_kappa[idx_near]))

        # Adaptive Lookahead
        if is_grid_launch:
            L_d = 14.0
        else:
            raw_ld = params["ld_base"] + params["ld_speed_gain"] * v_curr_kmh
            curv_damp = 1.0 + params["curv_damping"] * k_near
            L_d = float(np.clip(raw_ld / curv_damp, 4.2, 19.0))

        lookahead_steps = int(round(L_d / ds))
        idx_target = (idx_near + lookahead_steps) % agent.num_points
        p_target = agent.optimal_line_xy[idx_target]

        dx = float(p_target[0] - p_curr[0])
        dy = float(p_target[1] - p_curr[1])
        cos_th = float(np.cos(th_curr))
        sin_th = float(np.sin(th_curr))
        x_body = cos_th * dx + sin_th * dy
        y_body = -sin_th * dx + cos_th * dy
        actual_lookahead_sq = max(4.0, x_body ** 2 + y_body ** 2)

        curvature = 2.0 * y_body / actual_lookahead_sq
        steer_rad = np.arctan(curvature * wheelbase)
        steer_norm = float(np.clip(steer_rad / agent.max_steer_rad, -1.0, 1.0))

        # Dynamic Braking Horizon
        max_scan_m = params["max_scan_m"]
        n_scan = min(agent.num_points - 1, max(5, int(round(max_scan_m / ds))))
        scan_indices = (idx_near + np.arange(1, n_scan + 1)) % agent.num_points
        scan_speeds_ms = agent.optimal_speed_kmh[scan_indices] / 3.6
        scan_dists_m = np.arange(1, n_scan + 1) * ds

        a_brake = params["a_brake_safe"]
        allowed_speeds_ms = np.sqrt(scan_speeds_ms ** 2 + 2.0 * a_brake * scan_dists_m)
        v_target_ms = min(float(agent.optimal_speed_kmh[idx_near] / 3.6), float(np.min(allowed_speeds_ms)))
        v_target_kmh = float(v_target_ms * 3.6)
        v_err_kmh = v_target_kmh - v_curr_kmh

        steer_mag = abs(steer_norm)
        avail_long_grip = float(np.sqrt(max(0.10, 1.0 - min(0.90, (0.90 * steer_mag) ** 2))))
        traction_factor = float(np.sqrt(max(0.40, 1.0 - min(0.80, (0.80 * steer_mag) ** 2))))

        if is_grid_launch:
            target_brake = 0.0
            target_throttle = 1.0
        elif v_err_kmh >= 0.0:
            target_brake = 0.0
            maint_thr = params["maint_thr"] if v_curr_kmh > 40.0 else 0.22
            thr_factor = float(np.clip(v_err_kmh / params["thr_ramp_range"], 0.0, 1.0))
            target_throttle = float(maint_thr + thr_factor * (1.0 - maint_thr)) * traction_factor
        else:
            overspeed_kmh = -v_err_kmh
            if overspeed_kmh <= params["coast_margin"]:
                # Thả trôi tự nhiên (Lift-and-Coast) kết hợp giữ ga nhẹ theo quán tính
                target_throttle = float(params["maint_thr"] * max(0.0, 1.0 - overspeed_kmh / params["coast_margin"])) * traction_factor
                target_brake = 0.0
            else:
                target_throttle = 0.0
                brk_fraction = float(np.clip((overspeed_kmh - params["coast_margin"]) / 5.0, 0.0, 1.0))
                max_brk_pedal = 0.95
                target_brake = float(brk_fraction * max_brk_pedal * avail_long_grip)

        alpha_thr = 0.85
        alpha_brk = 0.85
        agent.filtered_throttle = float((1.0 - alpha_thr) * agent.filtered_throttle + alpha_thr * target_throttle)
        agent.filtered_brake = float((1.0 - alpha_brk) * agent.filtered_brake + alpha_brk * target_brake)

        throttle_action = float(np.clip(agent.filtered_throttle * 2.0 - 1.0, -1.0, 1.0))
        brake_action = float(np.clip(agent.filtered_brake * 2.0 - 1.0, -1.0, 1.0))
        action = np.array([steer_norm, throttle_action, brake_action], dtype=np.float32)
        net_pedal = float(agent.filtered_throttle - agent.filtered_brake)
        return action, steer_norm, net_pedal

    agent.select_action = custom_select_action

    obs, info = env.reset(seed=seed)
    bench_engine = env.get_benchmark_engine()
    opt_line_xy = bench_engine.optimal_line_xy
    
    step = 0
    terminated = False
    truncated = False
    line_errors = []
    speeds = []
    
    while not (terminated or truncated) and step < 2000:
        action, _, _ = agent.select_action(vehicle=env.car.raw_vehicle)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step += 1
        
        curr_pos = np.array(env.car.position[:2])
        idx_near, _ = env._get_track_s(curr_pos)
        line_pt = opt_line_xy[min(idx_near, len(opt_line_xy)-1)]
        err = float(np.linalg.norm(curr_pos - line_pt))
        line_errors.append(err)
        speeds.append(float(env.car.speed_km_h))

    lap_time = step * 0.1
    completed = bool(step_info.get("lap_completed", False))
    final_dist = float(env.telemetry.cumulative_distance)
    mean_err = float(np.mean(line_errors)) if line_errors else 99.0
    max_err = float(np.max(line_errors)) if line_errors else 99.0
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    
    env.close()
    return {
        "completed": completed,
        "lap_time": lap_time,
        "final_dist": final_dist,
        "mean_err": mean_err,
        "max_err": max_err,
        "mean_speed": mean_speed,
        "term_reason": step_info.get("termination_reason", "Running"),
    }

def optimize_loop():
    print("=" * 80)
    print(">> BAT DAU VONG LAP QUET THAM SO OPTIMAL AGENT (GRID SWEEP)")
    print("=" * 80)
    
    # Danh mục cấu hình thử nghiệm
    candidates = [
        # Candidate 1: Scan 80m, a_brake 5.8 m/s2
        {"name": "C1 (Scan 80m, a_brk 5.8, Coast 2.0)", "max_scan_m": 80.0, "a_brake_safe": 5.80, "ld_base": 4.2, "ld_speed_gain": 0.10, "curv_damping": 12.0, "thr_ramp_range": 6.0, "maint_thr": 0.40, "coast_margin": 2.0},
        # Candidate 2: Scan 90m, a_brake 6.2 m/s2
        {"name": "C2 (Scan 90m, a_brk 6.2, Coast 2.5)", "max_scan_m": 90.0, "a_brake_safe": 6.20, "ld_base": 4.2, "ld_speed_gain": 0.10, "curv_damping": 10.0, "thr_ramp_range": 5.5, "maint_thr": 0.42, "coast_margin": 2.5},
        # Candidate 3: Scan 100m, a_brake 6.5 m/s2
        {"name": "C3 (Scan 100m, a_brk 6.5, Coast 3.0)", "max_scan_m": 100.0, "a_brake_safe": 6.50, "ld_base": 4.5, "ld_speed_gain": 0.11, "curv_damping": 9.0, "thr_ramp_range": 5.0, "maint_thr": 0.44, "coast_margin": 3.0},
        # Candidate 4: Scan 110m, a_brake 6.8 m/s2 (Aggressive GT3 Late Braking)
        {"name": "C4 (Scan 110m, a_brk 6.8, Coast 2.5)", "max_scan_m": 110.0, "a_brake_safe": 6.80, "ld_base": 4.5, "ld_speed_gain": 0.11, "curv_damping": 8.0, "thr_ramp_range": 4.5, "maint_thr": 0.45, "coast_margin": 2.5},
        # Candidate 5: Scan 115m, a_brake 7.0 m/s2 (Ultra Late Braking)
        {"name": "C5 (Scan 115m, a_brk 7.0, Coast 2.0)", "max_scan_m": 115.0, "a_brake_safe": 7.00, "ld_base": 4.5, "ld_speed_gain": 0.11, "curv_damping": 8.0, "thr_ramp_range": 4.0, "maint_thr": 0.46, "coast_margin": 2.0},
    ]

    best_res = None
    best_cand = None

    for i, c in enumerate(candidates, 1):
        print(f"\n[Test {i}/{len(candidates)}] Dang danh gia {c['name']}...")
        res = evaluate_candidate(c)
        status_sym = "[FINISH]" if res["completed"] else f"[FAIL: {res['term_reason']}]"
        print(f"  -> Ket qua: {status_sym} | LapTime: {res['lap_time']:.2f}s | V_avg: {res['mean_speed']:.1f} km/h | Line MAE: {res['mean_err']:.2f}m (Max: {res['max_err']:.2f}m)")
        
        if res["completed"]:
            if best_res is None or res["lap_time"] < best_res["lap_time"]:
                best_res = res
                best_cand = c

    print("\n" + "=" * 80)
    print(">> CAU HINH TOI UU NHAT:")
    if best_cand is not None:
        print(f"  - Ten: {best_cand['name']}")
        print(f"  - Thoi gian 1 vong: {best_res['lap_time']:.2f}s (Can ly thuyet: 111.35s)")
        print(f"  - Van toc trung binh: {best_res['mean_speed']:.1f} km/h")
        print(f"  - Sai so Racing Line MAE: {best_res['mean_err']:.2f}m")
        print(f"  - Tham so: {best_cand}")
    else:
        print("  Khong co cau hinh nao hoan thanh tron ven!")
    print("=" * 80)

if __name__ == "__main__":
    optimize_loop()
