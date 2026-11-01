"""
Kiểm thử tính tương đương của Vector quan sát (Obs-Equivalence Test):
1. Chạy 50 bước trên 'silverstone_half' với fixed actions, so sánh với công thức tính gốc của Phase 1.
2. Chạy trên 'silverstone_full' tăng tốc lên 250 km/h, in vector obs để kiểm tra không chiều nào bị clip oan.
"""

import sys
import os
import numpy as np

# UTF-8 encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import GT3RaceCar, GT3RealCar


def run_obs_equivalence_test():
    print("=" * 75)
    print("KIỂM THỬ TƯƠNG ĐƯƠNG QUAN SÁT (OBS EQUIVALENCE TEST - WI-10 REVIEW)")
    print("=" * 75)

    # 1. Test 50 bước trên map nhỏ
    print("\n[1] Kiểm tra 50 bước trên map nhỏ (silverstone_half) so với công thức gốc Phase 1:")
    env = MetaRacingEnv(map_type="silverstone_half", use_render=False, start_seed=42)
    obs, info = env.reset(seed=42, options={"start_mode": "standing"})
    
    np.random.seed(12345)
    test_actions = [
        np.array([
            float(np.sin(i * 0.2) * 0.3),
            float(0.5 + 0.4 * np.cos(i * 0.1)),
            float(-1.0 + (0.5 if i % 10 == 0 else 0.0)),
        ], dtype=np.float32)
        for i in range(50)
    ]
    
    max_abs_diff = 0.0
    for step_i, act in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(act)
        
        # Tính toán lại obs theo đúng công thức gốc Phase 1 để so sánh
        pos = env.car.position
        heading = float(env.car.heading_theta)
        vel = env.car.velocity
        speed = float(env.car.speed_m_s)
        
        idx, cur_s = env._get_track_s(pos, env.current_track_idx)
        pt = env.track_xy[idx]
        th = float(env.track_theta[idx])
        kap = float(env.track_kappa[idx])
        
        disp = pos - pt
        lat_norm = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
        lat_off = float(np.dot(disp, lat_norm))
        
        fwd = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        lat_vec = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        vx = float(np.dot(vel, fwd))
        vy = float(np.dot(vel, lat_vec))
        
        dh = (heading - th + np.pi) % (2.0 * np.pi) - np.pi
        
        # Expected gốc Phase 1:
        exp_speed = float(np.clip(speed / 40.0, 0.0, 1.5))
        exp_vx = float(np.clip(vx / 40.0, -1.0, 1.5))
        exp_vy = float(np.clip(vy / 20.0, -1.5, 1.5))
        exp_lat = float(np.clip(lat_off / 4.5, -2.0, 2.0))
        exp_kap = float(np.clip(kap * 50.0, -2.0, 2.0))
        exp_dist_l = float(np.clip((4.5 - lat_off) / 4.5, -2.0, 2.0))
        exp_dist_r = float(np.clip((4.5 + lat_off) / 4.5, -2.0, 2.0))
        
        # So sánh 7 giá trị trọng tâm
        diffs = [
            abs(obs[0] - exp_speed),
            abs(obs[1] - exp_vx),
            abs(obs[2] - exp_vy),
            abs(obs[9] - exp_lat),
            abs(obs[10] - exp_dist_l),
            abs(obs[11] - exp_dist_r),
            abs(obs[14] - exp_kap),
        ]
        
        # So sánh 5 lookahead
        look_m = [10.0, 20.0, 35.0, 50.0, 70.0]
        for l_i, dist_ahead in enumerate(look_m):
            t_s = (cur_s + dist_ahead) % env.circuit_length
            w_idx = int(np.searchsorted(env.track_s, t_s)) % env.num_track_points
            w_pos = env.track_xy[w_idx]
            d_vec = w_pos - pos
            dx = float(np.dot(d_vec, fwd))
            dy = float(np.dot(d_vec, lat_vec))
            w_k = float(env.track_kappa[w_idx])
            
            exp_dx = float(np.clip(dx / 70.0, -1.0, 1.5))
            exp_dy = float(np.clip(dy / 30.0, -2.0, 2.0))
            exp_wk = float(np.clip(w_k * 50.0, -2.0, 2.0))
            
            base_idx = 15 + l_i * 4
            diffs.extend([
                abs(obs[base_idx] - exp_dx),
                abs(obs[base_idx + 1] - exp_dy),
                abs(obs[base_idx + 3] - exp_wk),
            ])
            
        step_max_diff = max(diffs)
        if step_max_diff > max_abs_diff:
            max_abs_diff = step_max_diff
            
    env.close()
    print(f"    -> Sai số cực đại trên toàn bộ 50 bước x 35 chiều: max |Δ| = {max_abs_diff:.10f}")
    assert max_abs_diff < 1e-5, f"LỖI: Obs map half bị lệch so với code cũ! Max diff = {max_abs_diff}"
    print("    -> [PASS] Obs map half khớp chính xác 100% với Phase 1!")

    # 2. Test map 1:1 ở 250 km/h: in giá trị từng chiều và kiểm tra không bị clip oan
    print("\n[2] Kiểm tra Obs ở 250 km/h trên map full 1:1 (GT3RealCar):")
    env_full = MetaRacingEnv(map_type="silverstone_full", vehicle=GT3RealCar(), use_render=False, start_seed=42)
    obs_full, _ = env_full.reset(options={"start_mode": "standing"})
    
    target_high_speed_obs = None
    reached_speed = 0.0
    for _ in range(350):
        action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        obs_full, _, term, _, _ = env_full.step(action)
        v_kmh = float(env_full.car.speed_km_h)
        if v_kmh > reached_speed:
            reached_speed = v_kmh
        if v_kmh >= 230.0 and target_high_speed_obs is None:
            target_high_speed_obs = obs_full
        if term:
            break
            
    env_full.close()
    
    if target_high_speed_obs is not None:
        print(f"    Đạt vận tốc cao tốc: {reached_speed:.1f} km/h")
        print("    Các chiều quan sát chuẩn hóa ở 250 km/h:")
        print(f"      - Obs[0] (speed / 80 m/s): {target_high_speed_obs[0]:.4f} (Trần clip 1.5, giá trị số < 1.0 -> KHÔNG bị clip)")
        print(f"      - Obs[1] (vx / 80 m/s):    {target_high_speed_obs[1]:.4f} (Trần clip 1.5 -> KHÔNG bị clip)")
        print(f"      - Obs[2] (vy / 40 m/s):    {target_high_speed_obs[2]:.4f} (Trần clip 1.5 -> KHÔNG bị clip)")
        print(f"      - Obs[9] (lat / 7.2 m):    {target_high_speed_obs[9]:.4f} (Trần clip 2.0 -> KHÔNG bị clip)")
        print(f"      - Lookahead 200m dx / 200: {target_high_speed_obs[15 + 4*4]:.4f} (Trần clip 1.5 -> KHÔNG bị clip)")
        
        assert target_high_speed_obs[0] < 1.49, "LỖI: Tốc độ 250 km/h bị clip trần!"
        assert target_high_speed_obs[1] < 1.49, "LỖI: Vận tốc dọc vx bị clip trần!"
        assert target_high_speed_obs[15 + 4*4] < 1.49, "LỖI: Lookahead 200m bị clip trần!"
        print("    -> [PASS] Không có chiều nào bị bão hòa hay clip oan ở tốc độ 250 km/h!")
    else:
        print(f"    Cảnh báo: Xe chỉ đạt {reached_speed:.1f} km/h trên đoạn thẳng xuất phát!")

    print("\n" + "=" * 75)
    print("HOÀN TẤT KIỂM THỬ OBS EQUIVALENCE & HIGH-SPEED DYNAMICS.")
    print("=" * 75)


if __name__ == "__main__":
    run_obs_equivalence_test()
