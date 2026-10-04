"""
evaluation/compare_policies.py: Chạy Benchmark Đối Đầu Trực Tiếp Giữa Các Policy (A/B Testing).
So sánh:
1. Centerline Safe Policy (Chạy đều an toàn khảo sát)
2. Centerline Speed Policy (Quy hoạch động Bellman OVP trên tim đường)
3. Racing Line Optimal Policy (Out-In-Out Racing Line + Bellman OVP)
4. Pure PPO Policy (Mạng nơ-ron Actor-Critic phản xạ thuần qua 11 tia LiDAR)
"""

import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào path nếu cần
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.racing_env import RacingEnv
from core.circuit_profiler import CircuitProfiler
from policies import (
    CenterlineSafePolicy,
    CenterlineSpeedPolicy,
    RacingLineOptimalPolicy,
    PPOAgentPolicy
)

def run_policy_benchmark(track_name="Silverstone", max_steps=4000, save_plot=True):
    print("=" * 80)
    print(f" THỬ NGHIỆM ĐỐI ĐẦU ĐA POLICY (A/B TESTING) TRÊN MẠCH ĐUA: {track_name.upper()}")
    print("=" * 80)

    # 1. Khởi tạo môi trường và Profiler
    env = RacingEnv(track_name=track_name)
    profiler = CircuitProfiler(
        tire_friction=0.9, 
        lookahead_distance=25.0, 
        track_width=env.track["width"], 
        default_center=env.track["center"]
    )

    # 2. Danh sách các Policy đem ra so sánh
    policies = [
        CenterlineSafePolicy(target_speed_kmh=38.0, name="Centerline Safe", color="#ef4444"),
        CenterlineSpeedPolicy(profiler=profiler, name="Centerline Speed (Bellman)", color="#f59e0b"),
        RacingLineOptimalPolicy(profiler=profiler, lap_stage=3, name="Racing Line (Out-In-Out)", color="#10b981"),
        PPOAgentPolicy(name="Pure PPO (LiDAR Reactive)", color="#8b5cf6")
    ]

    results = []

    for p_idx, policy in enumerate(policies):
        print(f"\n>>> Đang chạy thử nghiệm Policy: [{policy.name}] <<<")
        print(f"    Mô tả: {policy.description}")

        state = env.reset()
        policy.reset(env.track)
        
        trajectory = []
        speeds = []
        distances = []
        steer_angles = []
        total_dist = 0.0
        prev_pos = None

        done = False
        step_count = 0
        collision_step = None
        lap_time = 0.0
        finished = False

        while not done and step_count < max_steps:
            x, y, yaw, v, steer = env.state
            trajectory.append([x, y])
            speeds.append(v * 3.6) # Đổi ra km/h
            steer_angles.append(steer)

            # Nếu là policy đầu tiên (Centerline Safe), ghi nhận tọa độ để xây dựng bản đồ
            if p_idx == 0:
                profiler.record_step(x, y)

            if prev_pos is not None:
                total_dist += float(np.hypot(x - prev_pos[0], y - prev_pos[1]))
            distances.append(total_dist)
            prev_pos = (x, y)

            # Lấy hành động từ policy
            info_dict = {"car_state": env.state, "closest_waypoint": env.track["center"]}
            action = policy.select_action(state, info=info_dict)

            # Thực thi bước mô phỏng
            next_state, reward, done, info = env.step(action)
            state = next_state
            step_count += 1

            if info["is_collision"]:
                collision_step = step_count
                print(f"  [X] VA CHẠM (Đâm rào) tại step {step_count}! Quãng đường đi được: {total_dist:.1f}m")
                break

            if info["lap_completed"]:
                lap_time = info["lap_history"][-1]
                finished = True
                print(f"  [OK] HOÀN THÀNH VÒNG ĐUA! Thời gian: {lap_time:.2f}s | Tốc độ TB: {np.mean(speeds):.1f} km/h")
                
                # Hoàn tất xử lý bản đồ vi phân sau vòng khảo sát an toàn
                if p_idx == 0:
                    profiler.finalize_lap_1()
                    print("     -> Đã xây dựng hoàn chỉnh bản đồ vi phân và Quỹ đạo Racing Line Out-In-Out!")
                break

        res = {
            "policy": policy,
            "name": policy.name,
            "color": policy.color,
            "finished": finished,
            "lap_time": lap_time if finished else None,
            "steps": step_count,
            "collision_step": collision_step,
            "total_distance": total_dist,
            "speeds": np.array(speeds),
            "trajectory": np.array(trajectory),
            "distances": np.array(distances),
            "steer_angles": np.array(steer_angles),
            "v_avg": float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
            "v_max": float(np.max(speeds)) if len(speeds) > 0 else 0.0,
            "v_min": float(np.min(speeds[10:])) if len(speeds) > 10 else 0.0 # Bỏ qua lúc xuất phát
        }
        results.append(res)

    # 3. In bảng kết quả tổng hợp
    print("\n" + "=" * 88)
    print(f" BẢNG TỔNG HỢP SO SÁNH CÁC POLICY TRÊN MẠCH ĐUA: {track_name.upper()}")
    print("=" * 88)
    print(f"{'Tên Policy':<28} | {'Trạng thái':<14} | {'Lap Time (s)':<13} | {'V_avg (km/h)':<13} | {'V_max (km/h)':<13}")
    print("-" * 88)
    for r in results:
        status_str = "Hoàn thành" if r["finished"] else f"Đâm ở st.{r['collision_step']}"
        time_str = f"{r['lap_time']:.2f}s" if r["finished"] else "N/A"
        print(f"{r['name']:<28} | {status_str:<14} | {time_str:<13} | {r['v_avg']:<13.1f} | {r['v_max']:<13.1f}")
    print("=" * 88)

    # 4. Vẽ biểu đồ so sánh đa chiều nếu được yêu cầu
    if save_plot:
        plots_dir = os.path.join(ROOT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"policy_comparison_{track_name.lower()}.png")
        _plot_comparison(env, profiler, results, track_name, plot_path)
        print(f"\n[THÀNH CÔNG] Đã xuất biểu đồ so sánh tại: {plot_path}")

    return results

def _plot_comparison(env, profiler, results, track_name, output_path):
    fig = plt.figure(figsize=(18, 9), dpi=180)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22)

    # Panel 1 (Trái): Bản đồ đường đua và Quỹ đạo xe của các Policy
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#05070e")

    # Vẽ rào chắn mặt đường
    ax_map.plot(env.left_border[:, 0], env.left_border[:, 1], color="#475569", lw=2.0, label="Ranh giới đường đua")
    ax_map.plot(env.right_border[:, 0], env.right_border[:, 1], color="#475569", lw=2.0)
    ax_map.plot(env.center[:, 0], env.center[:, 1], color="#334155", lw=1.2, linestyle="--", label="Tim đường (Centerline)")

    # Vẽ vạch đích
    p0 = env.center[0]
    p_next = env.center[1]
    norm_vec = np.array([-(p_next[1] - p0[1]), p_next[0] - p0[0]])
    norm_vec = norm_vec / np.linalg.norm(norm_vec) * (env.track["width"] / 2.0)
    ax_map.plot([p0[0] - norm_vec[0], p0[0] + norm_vec[0]], 
                [p0[1] - norm_vec[1], p0[1] + norm_vec[1]], 
                color="#ffffff", lw=3.0, label="Vạch Xuất phát / Đích")

    # Vẽ quỹ đạo của từng policy
    for r in results:
        traj = r["trajectory"]
        if len(traj) > 0:
            lw = 2.5 if "Racing Line" in r["name"] else 1.8
            ls = "-" if r["finished"] else ":"
            alpha = 0.95 if r["finished"] else 0.7
            label_text = f"{r['name']}: {r['lap_time']:.2f}s" if r["finished"] else f"{r['name']}: Đâm rào"
            ax_map.plot(traj[:, 0], traj[:, 1], color=r["color"], lw=lw, linestyle=ls, alpha=alpha, label=label_text)

    ax_map.set_title(f"So sánh Quỹ đạo Đua xe giữa các Policy trên {track_name.upper()}", color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax_map.axis("equal")
    ax_map.grid(True, linestyle=":", alpha=0.25, color="#64748b")
    ax_map.tick_params(colors="#94a3b8")
    for spine in ax_map.spines.values():
        spine.set_color("#334155")
    ax_map.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)

    # Panel 2 (Góc trên bên phải): Hồ sơ Vận tốc so sánh theo % Vòng đua
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.set_facecolor("#0b0f19")
    
    for r in results:
        if r["finished"] and len(r["speeds"]) > 1:
            norm_dist = np.linspace(0, 100, len(r["speeds"]))
            lw = 2.4 if "Racing Line" in r["name"] else 1.8
            ax_speed.plot(norm_dist, r["speeds"], color=r["color"], lw=lw, label=f"{r['name']} (V_tb: {r['v_avg']:.1f} km/h)")

    ax_speed.set_title("Hồ sơ Vận tốc v(s) Dọc theo Chiều dài Vòng đua", color="#f8fafc", fontsize=12, fontweight="bold")
    ax_speed.set_xlabel("Phần trăm hoàn thành vòng đua (%)", color="#94a3b8", fontsize=10)
    ax_speed.set_ylabel("Vận tốc Xe (km/h)", color="#94a3b8", fontsize=10)
    ax_speed.grid(True, linestyle=":", alpha=0.25, color="#64748b")
    ax_speed.tick_params(colors="#94a3b8")
    for spine in ax_speed.spines.values():
        spine.set_color("#334155")
    ax_speed.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    # Panel 3 (Góc dưới bên phải): Biểu đồ Cột So sánh Thời gian Hoàn thành Vòng đua (Lap Time)
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor("#0b0f19")

    finished_results = [r for r in results if r["finished"]]
    names = [r["name"].replace(" (Out-In-Out)", "").replace(" (Bellman)", "") for r in finished_results]
    times = [r["lap_time"] for r in finished_results]
    colors = [r["color"] for r in finished_results]

    if len(finished_results) > 0:
        bars = ax_bar.barh(names, times, color=colors, height=0.55, edgecolor="#ffffff", linewidth=0.8, alpha=0.9)
        for bar, t in zip(bars, times):
            delta_best = t - min(times)
            label = f"{t:.2f}s (+{delta_best:.2f}s)" if delta_best > 0.05 else f"{t:.2f}s (BEST)"
            ax_bar.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, label,
                        va='center', color="#f8fafc", fontsize=10, fontweight="bold")

        ax_bar.set_xlim(0, max(times) * 1.25)

    ax_bar.set_title("So sánh Thời gian Hoàn thành Vòng đua (Lap Time)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax_bar.set_xlabel("Thời gian (giây) - Càng ngắn càng tối ưu", color="#94a3b8", fontsize=10)
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.25, color="#64748b")
    ax_bar.tick_params(colors="#94a3b8")
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark đối đầu các Policy đua xe F1.")
    parser.add_argument("--track", type=str, default="Silverstone", help="Tên mạch đua: Silverstone, Monza, Spa, Suzuka")
    parser.add_argument("--max_steps", type=int, default=4000, help="Số bước tối đa mỗi vòng")
    args = parser.parse_args()

    run_policy_benchmark(track_name=args.track, max_steps=args.max_steps)
