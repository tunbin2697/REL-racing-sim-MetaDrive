"""
evaluation/export_telemetry.py: Xuất dữ liệu quỹ đạo và telemetry của cả 4 mạch đua F1 (Monza, Spa, Suzuka, Silverstone).
Lưu dữ liệu tại: data/telemetry_data.json và data/telemetry_data.js phục vụ trực quan hóa web 60FPS.
"""

import sys
import os
import json
import math
import numpy as np

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import RacingEnv, CircuitProfiler

def export_all_telemetry():
    all_data = {}

    for trk in ['Monza', 'Spa', 'Suzuka', 'Silverstone']:
        print(f"Đang mô phỏng và trích xuất dữ liệu mạch đua: {trk}...")
        env = RacingEnv(trk)
        prof = CircuitProfiler(tire_friction=0.9, lookahead_distance=25.0, track_width=env.track["width"], default_center=env.track["center"])
        state = env.reset()
        
        if trk == 'Silverstone':
            diffs = np.diff(np.vstack([env.center, env.center[0]]), axis=0)
            ds = np.maximum(np.linalg.norm(diffs, axis=1), 1e-4)
            thetas = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
            d_theta = np.diff(np.concatenate([thetas, [thetas[0]]]))
            d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
            r_local = 1.0 / np.maximum(np.abs(d_theta / ds), 1e-4)
            lap1_speeds = np.clip(np.sqrt(0.70 * 9.81 * r_local), 6.5, 11.0)
            for _ in range(3):
                for i in range(len(env.center) - 1, -1, -1):
                    next_i = (i + 1) % len(env.center)
                    allowed_v = math.sqrt(lap1_speeds[next_i]**2 + 2.0 * 4.5 * ds[i])
                    if allowed_v < lap1_speeds[i]: 
                        lap1_speeds[i] = allowed_v
        
        track_info = {
            'name': trk,
            'width': float(env.track['width']),
            'center': [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in env.center],
            'left': [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in env.left],
            'right': [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in env.right],
            'racing_line': []
        }
        
        laps_data = {1: [], 2: [], 3: []}
        lap = 1
        c_idx = 0
        
        for step in range(15000):
            x, y, yaw, v, steer = env.state
            if lap == 1:
                prof.record_step(x, y)
                target_v = lap1_speeds[c_idx] if trk == 'Silverstone' else 11.0
                ref_path = env.track["center"]
            else:
                target_v = prof.get_target_speed(x, y, lap)
                ref_path = prof.get_reference_path(lap)
                
            dists_ref = np.linalg.norm(ref_path - np.array([x, y]), axis=1)
            c_idx = int(np.argmin(dists_ref))
            n_ref = len(ref_path)
            
            if trk == 'Silverstone':
                if lap == 1:
                    target_v = lap1_speeds[c_idx]
                    lookahead_m = min(max(6.0, v * 0.45), r_local[c_idx] * 0.75 + 4.0)
                    lookahead_steps = max(2, int(round(lookahead_m / 4.0)))
                else:
                    lookahead_steps = max(14, int(v * 1.1))
            else:
                lookahead_steps = max(5, int(v * 0.70))
                
            lookahead_idx = (c_idx + lookahead_steps) % n_ref
            tx, ty = ref_path[lookahead_idx]
            desired_yaw = math.atan2(ty - y, tx - x)
            yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            
            if trk == 'Silverstone' and lap == 1:
                p0 = ref_path[c_idx]
                p1 = ref_path[(c_idx + 1) % n_ref]
                cross = (p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0])
                cte = dists_ref[c_idx] * (1.0 if cross > 0 else -1.0)
                steer_cmd = np.clip(yaw_err * 1.6 - math.atan2(1.2 * cte, max(2.0, v)), -1.0, 1.0)
            else:
                steer_cmd = np.clip(yaw_err * 1.5, -1.0, 1.0)
            
            if v < target_v - 0.5:
                accel_cmd = 1.0
            elif v > target_v + 0.5:
                accel_cmd = -0.90
            else:
                accel_cmd = 0.10
                
            # Mẫu hóa mỗi 2 bước (dt = 0.10s) để tối ưu dung lượng JSON
            if step % 2 == 0:
                laps_data[lap].append({
                    'x': round(float(x), 2),
                    'y': round(float(y), 2),
                    'yaw': round(float(yaw), 3),
                    'v': round(float(v), 2),
                    'steer': round(float(steer), 3),
                    'accel': round(float(accel_cmd), 2),
                    't': round(float(env.lap_time), 2),
                    'lidar': [round(float(d), 2) for d in env._compute_lidar()]
                })
                
            action = [steer_cmd, accel_cmd]
            next_state, reward, done, info = env.step(action)
            
            if info['lap_completed']:
                lap_t = info['lap_history'][-1]
                print(f"  [OK] Hoàn thành Lap {lap} ({trk}) trong {lap_t:.2f}s!")
                if lap == 1:
                    prof.finalize_lap_1()
                    if prof.racing_line is not None:
                        track_info['racing_line'] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in prof.racing_line]
                lap += 1
                if lap > 3:
                    break
            if done:
                break
                
        all_data[trk] = {
            'track': track_info,
            'laps': laps_data
        }

    data_dir = os.path.join(ROOT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "telemetry_data.json")
    js_path = os.path.join(data_dir, "telemetry_data.js")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f)

    with open(js_path, 'w', encoding='utf-8') as f:
        f.write('window.F1_TELEMETRY = ' + json.dumps(all_data) + ';')

    print(f"\n[THÀNH CÔNG] Đã lưu telemetry tại:\n  - {json_path}\n  - {js_path}")

if __name__ == "__main__":
    export_all_telemetry()
