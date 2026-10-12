"""
SCRIPT CHẨN ĐOÁN TELEMETRY ĐỘC LẬP (STANDALONE SIM RACING TELEMETRY DIAGNOSTIC)
Chạy độc lập khi cần debug chuyên sâu hành vi lái, góc cua, chân ga/phanh và quỹ đạo xe.

Cách chạy:
python scripts/plot_telemetry_diagnostic.py --model models/best_agent.pt --save-path plots/telemetry_diagnostic.png
"""

import os
import sys
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game_env.racing_env import MetaRacingEnv
from agents.ppo_agent import PPOAgent

# Danh sách góc cua chính trên đường đua 2186m
TRACK_CORNERS = [
    (297, "Chicane 1"),
    (850, "90° Turn"),
    (1231, "180° Hairpin"),
    (1511, "Sharp Left"),
    (1650, "S-Turn / Apex (75.4%)"),
    (1950, "Final Sweeper"),
    (2186, "Finish Line"),
]

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Telemetry Diagnostic Tool")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="Checkpoint cần chẩn đoán")
    parser.add_argument("--save-path", type=str, default="plots/telemetry_diagnostic.png", help="Đường dẫn lưu biểu đồ")
    parser.add_argument("--seed", type=int, default=42, help="Seed mô phỏng")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 75)
    print("🏎️  CHẨN ĐOÁN SIM RACING TELEMETRY ĐỘC LẬP")
    print("=" * 75)
    print(f"  - Model checkpoint: {args.model}")
    print(f"  - File xuất: {args.save_path}")
    print(f"  - Seed: {args.seed}")
    
    env = MetaRacingEnv(use_render=False, milestone_respawn=False)
    obs_shape = getattr(env.observation_space, "shape", None)
    act_shape = getattr(env.action_space, "shape", None)
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 35
    act_dim = int(act_shape[0]) if act_shape is not None else 2
    
    agent = PPOAgent(state_dim=obs_dim, action_dim=act_dim)
    
    if os.path.exists(args.model):
        try:
            agent.load(args.model)
            print(f">> [INFO] Đã nạp thành công checkpoint: {args.model} (State Dim: {obs_dim})")
        except Exception as e:
            print(f">> [CẢNH BÁO] Không thể nạp checkpoint: {e}")
            print(">> Sử dụng trọng số mặc định hiện tại để chẩn đoán.")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy file {args.model}. Sử dụng chính sách ngẫu nhiên.")
        
    obs, info = env.reset(seed=args.seed, options={"force_start": True})
    step = 0
    terminated = False
    truncated = False
    term_reason = "Running"
    
    while not (terminated or truncated):
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step += 1
        
        if terminated:
            if step_info.get("arrive_dest", False):
                term_reason = "Về đích trọn vẹn (Finish Lap) 🏁"
            elif step_info.get("crash", False) or step_info.get("out_of_road", False):
                term_reason = f"Đâm xe / Chệch đường tại s = {env.telemetry.cumulative_distance:.1f}m 💥"
            elif step_info.get("is_stalled", False):
                term_reason = "Xe chết máy / Đứng yên tại chỗ 🛑"
            else:
                term_reason = "Kết thúc episode (Terminated)"
        elif truncated:
            term_reason = "Hết số bước tối đa (Truncated Timeout) ⏱️"
            
    samples = env.telemetry.current_lap_data
    track_xy = env.track_xy.copy() if hasattr(env, "track_xy") else None
    track_theta = env.track_theta.copy() if hasattr(env, "track_theta") else None
    track_len = env.circuit_length
    
    # Lấy lịch sử vị trí 2D thực tế của xe từ env.episode_history
    if hasattr(env, "episode_history") and len(env.episode_history) > 0:
        ego_x = np.array([h["pos"][0] for h in env.episode_history])
        ego_y = np.array([h["pos"][1] for h in env.episode_history])
    elif samples and "pos" in samples[0]:
        ego_x = np.array([s["pos"][0] for s in samples])
        ego_y = np.array([s["pos"][1] for s in samples])
    else:
        ego_x = None
        ego_y = None
    
    env.close()
    
    if not samples or len(samples) < 2:
        print(">> [LỖI] Không đủ dữ liệu mẫu telemetry để trích xuất biểu đồ.")
        return
        
    times = np.array([s["time"] for s in samples])
    dists = np.array([s["distance"] for s in samples])
    speeds = np.array([s["speed_km_h"] for s in samples])
    steer = np.array([s.get("steering", 0.0) for s in samples])
    throttle = np.array([s.get("throttle", 0.0) for s in samples]) * 100.0
    brake = np.array([s.get("brake", 0.0) for s in samples]) * 100.0
    acc_long = np.array([s.get("acc_long_g", 0.0) for s in samples])
    acc_lat = np.array([s.get("acc_lat_g", 0.0) for s in samples])
    
    # Tính khoảng cách lệch tâm d_lat chuẩn xác
    d_lat = []
    if track_xy is not None and len(track_xy) > 0 and ego_x is not None and ego_y is not None:
        for ex, ey in zip(ego_x, ego_y):
            diffs = track_xy - np.array([ex, ey], dtype=np.float32)
            dists_pt = np.sum(diffs**2, axis=1)
            min_idx = np.argmin(dists_pt)
            d_val = float(np.sqrt(dists_pt[min_idx]))
            if track_theta is not None and min_idx < len(track_theta):
                th = float(track_theta[min_idx])
                # Vector tiếp tuyến: (cos(th), sin(th)). Vector pháp tuyến trái: (-sin(th), cos(th))
                # Sai số vị trí: dx, dy
                cross = -np.sin(th) * float(diffs[min_idx, 0]) + np.cos(th) * float(diffs[min_idx, 1])
                d_val = -d_val if cross > 0 else d_val
            d_lat.append(d_val)
        d_lat = np.array(d_lat)
    else:
        d_lat = np.zeros_like(dists)

    final_dist = dists[-1]
    final_time = times[-1]
    max_spd = np.max(speeds)
    mean_spd = np.mean(speeds)
    completion_pct = (final_dist / track_len) * 100.0

    print("-" * 75)
    print("📊 KẾT QUẢ CHẨN ĐOÁN:")
    print(f"  • Quãng đường hoàn thành: {final_dist:.1f}m / {track_len:.1f}m ({completion_pct:.1f}%)")
    print(f"  • Thời gian chạy: {final_time:.2f}s ({step} physics steps)")
    print(f"  • Vận tốc cao nhất (V-max): {max_spd:.1f} km/h")
    print(f"  • Vận tốc trung bình (V-avg): {mean_spd:.1f} km/h")
    print(f"  • Trạng thái kết thúc: {term_reason}")
    print("-" * 75)

    # Bỏ emoji khi đưa vào tiêu đề matplotlib để tránh cảnh báo missing font
    clean_reason = term_reason.replace("🏁", "[FINISH]").replace("💥", "[CRASH]").replace("🛑", "[STALL]").replace("⏱️", "[TIMEOUT]")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=300)
    fig.suptitle(
        f"Sim Racing Telemetry Diagnostic Dashboard\n"
        f"Model: {os.path.basename(args.model)} | Dist: {final_dist:.1f}m ({completion_pct:.1f}%) | Time: {final_time:.2f}s | Result: {clean_reason}",
        fontsize=14, fontweight="bold"
    )

    # -------------------------------------------------------------
    # 1. Speed Trace (v vs s) với Mốc Cua
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(dists, speeds, color="#007acc", lw=2.2, label=f"Speed (Max: {max_spd:.1f} km/h)")
    for corner_dist, corner_name in TRACK_CORNERS:
        if corner_dist <= max(final_dist + 50, 100):
            ax1.axvline(corner_dist, color="crimson", linestyle=":", lw=1.5, alpha=0.85)
            y_pos = max_spd * 0.75 if max_spd > 20 else 50
            ax1.text(corner_dist + 5, y_pos, corner_name, fontsize=8, color="crimson", rotation=90, fontweight="bold")
    ax1.set_title("1. Speed Trace (v vs s) & Corner Apexes", fontweight="bold")
    ax1.set_xlabel("Track Distance s (meters)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # 2. Quỹ Đạo Đua 2D (2D Trajectory vs Circuit Map)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    if track_xy is not None and len(track_xy) > 0:
        ax2.plot(track_xy[:, 0], track_xy[:, 1], color="lightgray", lw=4.0, linestyle="--", label="Circuit Centerline")
    if ego_x is not None and ego_y is not None and len(ego_x) > 0 and len(ego_y) > 0:
        scatter_map = ax2.scatter(ego_x, ego_y, c=speeds, cmap="turbo", s=14, alpha=0.9, label="Vehicle Trajectory")
        cb_map = fig.colorbar(scatter_map, ax=ax2, pad=0.02)
        cb_map.set_label("Speed (km/h)", fontsize=8)
        # Điểm xuất phát
        ax2.plot(ego_x[0], ego_y[0], marker="o", color="green", markersize=10, label="Start Point")
        # Điểm dừng / đâm
        term_marker = "[FINISH]" if completion_pct >= 98.0 else "[STOP/CRASH]"
        ax2.plot(ego_x[-1], ego_y[-1], marker="X", color="red", markersize=12, label=f"Terminated ({term_marker})")
    ax2.set_title("2. 2D Circuit Map & Trajectory Pace", fontweight="bold")
    ax2.set_xlabel("X Coordinate (m)")
    ax2.set_ylabel("Y Coordinate (m)")
    ax2.axis("equal")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=8)

    # -------------------------------------------------------------
    # 3. Góc Đánh Lái & Gia Tốc Ngang (Steering & Lateral G)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(dists, steer, color="#ff7f0e", lw=1.8, label="Steering Input [-1, 1]")
    ax3.axhline(0, color="black", lw=0.8, linestyle="-", alpha=0.4)
    ax3.fill_between(dists, 0, steer, where=(steer > 0), color="orange", alpha=0.2, label="Right Turn Zone")
    ax3.fill_between(dists, 0, steer, where=(steer < 0), color="blue", alpha=0.2, label="Left Turn Zone")
    ax3.set_title("3. Steering Profile: Corner Turn-in & Apex Stability", fontweight="bold")
    ax3.set_xlabel("Track Distance s (meters)")
    ax3.set_ylabel("Steering Input [-1 to 1]")
    ax3.set_ylim(-1.05, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.6)
    
    # Twin axis hiển thị Lateral G
    ax3_twin = ax3.twinx()
    ax3_twin.plot(dists, acc_lat, color="#8c564b", lw=1.2, linestyle="--", alpha=0.7, label="Lateral Acc (G)")
    ax3_twin.set_ylabel("Lateral G-Force", color="#8c564b")
    ax3_twin.tick_params(axis="y", labelcolor="#8c564b")
    
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3t, labels_3 + labels_3t, loc="lower right", fontsize=8)

    # -------------------------------------------------------------
    # 4. Hành Vi Chân Ga & Phanh (Pedal Profile: Throttle vs Brake)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.plot(dists, throttle, color="#2ca02c", lw=1.8, label="Throttle (%)")
    ax4.plot(dists, brake, color="#d62728", lw=1.8, label="Brake (%)")
    ax4.fill_between(dists, 0, brake, where=(brake > 10), color="red", alpha=0.2, label="Heavy Braking Zone")
    ax4.set_title("4. Pedal Profile: Throttle Aggressiveness & Trail Braking", fontweight="bold")
    ax4.set_xlabel("Track Distance s (meters)")
    ax4.set_ylabel("Pedal Input (%)")
    ax4.set_ylim(-5, 108)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # 5. G-G Friction Circle Diagram (Vòng tròn ma sát)
    # -------------------------------------------------------------
    ax5 = axes[2, 0]
    scatter_gg = ax5.scatter(acc_lat, acc_long, c=speeds, cmap="plasma", alpha=0.7, s=14, edgecolors="none")
    cb_gg = fig.colorbar(scatter_gg, ax=ax5, orientation="vertical", pad=0.02)
    cb_gg.set_label("Speed (km/h)", fontsize=8)
    theta = np.linspace(0, 2 * np.pi, 100)
    for g_lim in [0.5, 1.0, 1.5]:
        ax5.plot(g_lim * np.cos(theta), g_lim * np.sin(theta), linestyle="--", color="gray", alpha=0.5, lw=1)
        ax5.text(0.05, g_lim, f"{g_lim:.1f}G", color="gray", fontsize=8)
    ax5.axhline(0, color="gray", lw=0.6, alpha=0.5)
    ax5.axvline(0, color="gray", lw=0.6, alpha=0.5)
    ax5.set_title("5. G-G Diagram: Grip Utilization & Friction Limits", fontweight="bold")
    ax5.set_xlabel("Lateral Acc (G) [Cornering]")
    ax5.set_ylabel("Longitudinal Acc (G) [Accel / Brake]")
    ax5.set_xlim(-1.8, 1.8)
    ax5.set_ylim(-1.8, 1.8)
    ax5.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # 6. Độ Lệch Ngang Tim Đường (Frenet Lateral Deviation d_lat)
    # -------------------------------------------------------------
    ax6 = axes[2, 1]
    ax6.plot(dists, d_lat, color="#9467bd", lw=1.8, label="Ego Lateral Offset (m)")
    ax6.axhline(0, color="black", linestyle="-", lw=1.0, alpha=0.6, label="Centerline (0m)")
    ax6.axhline(4.0, color="red", linestyle="--", lw=1.2, alpha=0.7, label="FIA Road Limit (+4m)")
    ax6.axhline(-4.0, color="red", linestyle="--", lw=1.2, alpha=0.7, label="FIA Road Limit (-4m)")
    ax6.fill_between(dists, -4.0, 4.0, color="gray", alpha=0.1, label="Track Surface")
    ax6.set_title("6. Lateral Deviation (Track Width Utilization)", fontweight="bold")
    ax6.set_xlabel("Track Distance s (meters)")
    ax6.set_ylabel("Offset from Centerline (m)")
    ax6.set_ylim(-7.0, 7.0)
    ax6.grid(True, linestyle="--", alpha=0.6)
    ax6.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    plt.close()
    print(f">> [TELEMETRY] Đã xuất biểu đồ chẩn đoán đối chiếu tại: {args.save_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
