"""
demo_racing.py: Trực quan hóa và Trình chiếu Vòng đua Thích ứng trên các Mạch đua F1.

Cách chạy:
    python demo_racing.py --track monza
    python demo_racing.py --track spa
    python demo_racing.py --track suzuka
"""

import argparse
import sys
import io
import os
import math
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows không bị UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core import RacingEnv, PPOAgent, CircuitProfiler

def run_visual_demo(track_name="Monza", weights_path="models/f1_racing_agent.pt"):
    print(f"================================================================")
    print(f" TRỰC QUAN HÓA CUỘC ĐUA TRÊN MẠCH ĐUA: {track_name.upper()}")
    print("================================================================")
    
    env = RacingEnv(track_name=track_name)
    agent = PPOAgent(state_dim=16, action_dim=2)
    
    # Nạp trọng số nếu có
    import os
    if not os.path.exists(weights_path) and os.path.exists("f1_racing_agent.pt"):
        weights_path = "f1_racing_agent.pt"
        
    if os.path.exists(weights_path):
        agent.load(weights_path)
        print(f"[OK] Đã nạp trọng số PPO từ: {weights_path}")
    else:
        print("[!] Chưa có file trọng số đã train, sử dụng bộ điều khiển thích ứng kết hợp.")
        
    profiler = CircuitProfiler(tire_friction=0.9, lookahead_distance=25.0, track_width=env.track["width"], default_center=env.track["center"])
    
    # Precompute safe lap 1 speed limit for Silverstone if needed
    is_silverstone = track_name.lower() in ['silverstone', 'silverstone circuit']
    if is_silverstone:
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
                if allowed_v < lap1_speeds[i]: lap1_speeds[i] = allowed_v

    lap_trajectories = {1: [], 2: [], 3: []}
    lap_speeds = {1: [], 2: [], 3: []}
    
    state = env.reset()
    current_lap = 1
    done = False
    profiler_feats = (0.0, 0.0)
    
    print("\nBắt đầu cuộc đua 3 vòng:")
    print(" - Vòng 1: Làm quen đường & Lập bản đồ độ cong (Reconnaissance Lap)")
    print(" - Vòng 2: Tấn công thời gian với Racing Line (Time Attack Lap)")
    print(" - Vòng 3: Tối ưu hóa quỹ đạo Out-In-Out kịch khung (Optimized Lap)")
    
    while current_lap <= 3 and not done:
        x, y, yaw, v, steer = env.state
        lap_trajectories[current_lap].append([x, y])
        lap_speeds[current_lap].append(v)
        
        # 1. Quản lý Profiler & Reference Path
        if current_lap == 1:
            profiler.record_step(x, y)
            profiler_feats = (0.0, 0.0)
            ref_path = env.track["center"]
            target_v = lap1_speeds[c_idx] if is_silverstone and 'c_idx' in locals() else 11.0
        else:
            profiler_feats = profiler.get_lookahead_features(x, y, v)
            ref_path = profiler.get_reference_path(current_lap)
            target_v = profiler.get_target_speed(x, y, current_lap)
            
        # 2. Điều khiển Lái theo Racing Line Out-In-Out
        dists_ref = np.linalg.norm(ref_path - np.array([x, y]), axis=1)
        c_idx = int(np.argmin(dists_ref))
        n_ref = len(ref_path)
        
        if is_silverstone:
            if current_lap == 1:
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
        
        if is_silverstone and current_lap == 1:
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
            
        action = [steer_cmd, accel_cmd]
        next_state, reward, done, info = env.step(action, external_profiler_features=profiler_feats)
        state = next_state
        
        if info["lap_completed"]:
            t = info["lap_history"][-1]
            print(f"  🏁 [HOÀN THÀNH LAP {current_lap}] Thời gian: {t:.2f}s | Tốc độ TB: {np.mean(lap_speeds[current_lap])*3.6:.1f} km/h")
            if current_lap == 1:
                profiler.finalize_lap_1()
                print("     -> Đã tối ưu hóa Racing Line và tạo bản đồ vận tốc Bellman OVP!")
            current_lap += 1
            
        if info["is_collision"]:
            print("  [!] Xe va chạm ngoài ranh giới.")
            break

    # Vẽ quỹ đạo đường đua và so sánh đường chạy (Racing Line Comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Bản đồ Circuit & Quỹ đạo xe
    track = env.track
    ax1.plot(track["left"][:, 0], track["left"][:, 1], 'k-', lw=1.5, label='Ranh giới đường đua')
    ax1.plot(track["right"][:, 0], track["right"][:, 1], 'k-', lw=1.5)
    ax1.plot(track["center"][:, 0], track["center"][:, 1], '--', color='gray', alpha=0.5, label='Tim đường')
    if profiler.racing_line is not None:
        ax1.plot(profiler.racing_line[:, 0], profiler.racing_line[:, 1], ':', color='#ffd700', lw=2.0, label='Optimal Racing Line (Out-In-Out)')
    
    colors = {1: '#e74c3c', 2: '#f39c12', 3: '#2ecc71'}
    labels = {1: 'Vòng 1: Làm quen (Center)', 2: 'Vòng 2: Time Attack', 3: 'Vòng 3: Racing Line'}
    
    for lp in [1, 2, 3]:
        pts = np.array(lap_trajectories[lp])
        if len(pts) > 0:
            ax1.plot(pts[:, 0], pts[:, 1], color=colors[lp], lw=2.0, label=labels[lp], alpha=0.85)
            
    ax1.set_title(f'So sánh Quỹ đạo Đua xe trên {track_name} (Multi-Lap Trajectory)', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Biểu đồ Vận tốc theo chiều dài quãng đường (Telemetry Speed Profile)
    for lp in [1, 2, 3]:
        spd = np.array(lap_speeds[lp]) * 3.6 # km/h
        if len(spd) > 0:
            ax2.plot(np.linspace(0, 100, len(spd)), spd, color=colors[lp], lw=2.0, label=f'Lap {lp}')
            
    ax2.set_xlabel('Phần trăm Hoàn thành Vòng đua (%)', fontsize=12)
    ax2.set_ylabel('Vận tốc Xe (km/h)', fontsize=12)
    ax2.set_title('Telemetry: Hồ Sơ Vận Tốc So Sánh Giữa Các Vòng', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_img = os.path.join(plots_dir, f"demo_{track_name.lower()}_telemetry.png")
    plt.savefig(output_img, dpi=200)
    plt.close()
    print(f"\n[THÀNH CÔNG] Đã xuất bản đồ quỹ đạo và biểu đồ telemetry tại: {output_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="Monza", help="Tên mạch đua: Monza, Spa, Suzuka")
    args = parser.parse_args()
    run_visual_demo(track_name=args.track)
