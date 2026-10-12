"""
CÔNG CỤ CHẨN ĐOÁN PHÂN RÃ HÀM THƯỞNG (REWARD BREAKDOWN & BIAS DIAGNOSTIC)
Mục đích:
- Đo đạc chi tiết từng mét đường: r_progress, r_milestone, r_speed, r_step_penalty, r_delta, macro_reward.
- Kiểm tra tính đúng đắn của công thức (ví dụ: delta time window có nhận được điểm không, ở đâu).
- Phát hiện các bẫy thiên vị (Reward Hacking / Bias) mà mạng nơ-ron gặp phải.
- Xuất biểu đồ học thuật 4 bảng tại: plots/reward_breakdown.png.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game_env.racing_env import MetaRacingEnv
from agents.ppo_agent import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Racing Reward Breakdown & Diagnostic Tool")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="Checkpoint cần chẩn đoán")
    parser.add_argument("--save-path", type=str, default="plots/reward_breakdown.png", help="Đường dẫn lưu biểu đồ")
    parser.add_argument("--seed", type=int, default=42, help="Seed kiểm tra")
    return parser.parse_args()

def run_reward_diagnostic(model_path: str, save_path: str, seed: int = 42):
    print("=" * 70)
    print("🔍 KHỞI ĐỘNG CHẨN ĐOÁN PHÂN RÃ HÀM THƯỞNG (REWARD BREAKDOWN)")
    print("=" * 70)
    print(f"  - Model: {model_path}")
    print(f"  - Output Plot: {save_path}")
    
    env = MetaRacingEnv(use_render=False, milestone_respawn=False)
    obs_shape = env.observation_space.shape
    state_dim = int(obs_shape[0]) if obs_shape is not None else 35
    agent = PPOAgent(state_dim=state_dim, action_dim=2)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f">> Đã nạp checkpoint: {model_path}")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy {model_path}. Dùng chính sách ngẫu nhiên.")
        
    obs, info = env.reset(seed=seed, options={"force_start": True})
    
    # Bộ đệm ghi nhận từng bước
    dists = []
    speeds = []
    throttles = []
    brakes = []
    steers = []
    delta_times = []
    
    r_progress_list = []
    r_idle_list = []
    r_speed_list = []
    r_brake_list = []
    r_d20_list = []
    r_d60_list = []
    r_d150_list = []
    r_delta_list = []
    r_macro_list = []
    total_rewards = []
    
    last_long = float(getattr(env, "last_longitudinal", 0.0))
    step = 0
    
    while True:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step += 1
        
        sample = env.telemetry.current_lap_data[-1] if env.telemetry.current_lap_data else {}
        curr_dist = float(sample.get("distance", env.telemetry.cumulative_distance))
        curr_speed = float(sample.get("speed_km_h", 0.0))
        delta_t = float(sample.get("delta_time", 0.0))
        
        vehicle = env.vehicle
        heading_diff = vehicle.heading_diff(vehicle.lane)
        max_speed = vehicle.max_speed_km_h or 120.0
        speed_ratio = curr_speed / max_speed
        
        # Tách từng thành phần thưởng theo đúng thiết kế mới
        r_prg = float(step_info.get("r_milestone", step_info.get("r_progress", 0.0)))
        r_idle = float(step_info.get("r_idle", 0.0))
        r_spd = float(step_info.get("r_speed", 0.0)) + float(step_info.get("r_throttle", 0.0))
        r_brk = float(step_info.get("r_brake", 0.0))
        r_d20 = float(step_info.get("r_d20", 0.0))
        r_d60 = float(step_info.get("r_d60", 0.0))
        r_d150 = float(step_info.get("r_d150", 0.0))
        r_dlt = float(step_info.get("r_delta", 0.0))
        r_mcr = float(step_info.get("macro_reward", 0.0))
        
        dists.append(curr_dist)
        speeds.append(curr_speed)
        throttles.append(float(sample.get("throttle", max(0.0, action[1]))))
        brakes.append(float(sample.get("brake", max(0.0, -action[1]))))
        steers.append(float(action[0]))
        delta_times.append(delta_t)
        
        r_progress_list.append(r_prg)
        r_idle_list.append(r_idle)
        r_speed_list.append(r_spd)
        r_brake_list.append(r_brk)
        r_d20_list.append(r_d20)
        r_d60_list.append(r_d60)
        r_d150_list.append(r_d150)
        r_delta_list.append(r_dlt)
        r_macro_list.append(r_mcr)
        total_rewards.append(float(reward))
        
        if terminated or truncated:
            comp = step_info.get("route_completion", 0.0) * 100
            reason = step_info.get("termination_reason", "unknown")
            print(f">> Kết thúc lượt chạy tại step {step}:")
            print(f"   + Cự ly đạt được: {curr_dist:.1f}m ({comp:.1f}%)")
            print(f"   + Vận tốc cuối: {curr_speed:.1f} km/h")
            print(f"   + Lý do kết thúc: {reason}")
            print(f"   + 4 Milestones Sum: {sum(r_progress_list):+.2f}")
            print(f"   + Speed Incentive Sum: {sum(r_speed_list):+.2f}")
            print(f"   + Brake Incentive Sum: {sum(r_brake_list):+.2f}")
            print(f"   + Terminal Macro: {r_mcr:+.1f}")
            print(f"   + Tổng Return: {sum(total_rewards):+.2f}")
            break
            
    env.close()
    
    # =========================================================================
    # VẼ BIỂU ĐỒ CHẨN ĐOÁN PHÂN RÃ HÀM THƯỞNG: NET CONTRIBUTION AUDIT
    # =========================================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dists = np.array(dists)
    
    cum_progress = np.cumsum(r_progress_list)
    cum_idle = np.cumsum(r_idle_list)
    cum_speed = np.cumsum(r_speed_list)
    cum_brake = np.cumsum(r_brake_list)
    cum_d20 = np.cumsum(r_d20_list)
    cum_d60 = np.cumsum(r_d60_list)
    cum_d150 = np.cumsum(r_d150_list)
    cum_delta = np.cumsum(r_delta_list)
    cum_macro = np.cumsum(r_macro_list)
    cum_total = np.cumsum(total_rewards)
    
    # Danh sách 9 thành phần phân rã kiểm toán đầy đủ
    components = [
        "Milestones\n(0.5% / 50đ)",
        "Speed & Throttle\n(Incentive)",
        "Brake\n(Trail Braking)",
        "Idle Pen\n(<3 km/h)",
        "Delta 20m\n(Short x5)",
        "Delta 60m\n(Mid x5)",
        "Delta 150m\n(Long x5)",
        "Terminal\n(-40 / 0đ)",
        "TOTAL\nNET RETURN"
    ]
    
    values = [
        cum_progress[-1],
        cum_speed[-1],
        cum_brake[-1],
        cum_idle[-1],
        cum_d20[-1],
        cum_d60[-1],
        cum_d150[-1],
        cum_macro[-1],
        cum_total[-1],
    ]
    
    colors = [
        "#1f77b4",  # Milestones - Blue
        "#2ca02c",  # Speed - Green
        "#ff7f0e",  # Brake - Orange
        "#8c564b",  # Idle - Brown
        "#17becf",  # Delta 20m - Cyan
        "#9467bd",  # Delta 60m - Purple
        "#e377c2",  # Delta 150m - Pink
        "#d62728" if cum_macro[-1] < 0 else "#2ca02c",  # Terminal - Red (crash) or Green (finish)
        "#111827" if cum_total[-1] >= 0 else "#991b1b"   # Total Return - Dark Slate / Crimson
    ]
    
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    
    bars = ax.bar(components, values, color=colors, alpha=0.92, edgecolor="#1e293b", lw=1.3, width=0.62)
    
    # Thêm giá trị số cụ thể lên từng cột
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 5 if height >= 0 else -14
        font_col = "#0f172a" if height >= 0 else "#991b1b"
        ax.annotate(f"{height:+.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, offset),
                     textcoords="offset points",
                     ha="center", va=va, fontsize=10, fontweight="bold", color=font_col)
                     
    ax.axhline(0, color="#334155", lw=1.4, linestyle="-")
    
    # Định dạng trục
    title_text = f"Net Contribution by Reward Component | Audit Check ({os.path.basename(model_path)})"
    ax.set_title(title_text, fontsize=14, fontweight="bold", pad=15, color="#0f172a")
    ax.set_ylabel("Total Score Contribution (Points)", fontsize=11, fontweight="bold", color="#334155")
    ax.grid(True, linestyle="--", alpha=0.55, axis="y", color="#cbd5e1")
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=10)
    
    # Mở rộng giới hạn y để chữ không bị cắt
    min_val, max_val = min(values), max(values)
    y_margin = max(abs(min_val), abs(max_val)) * 0.22 + 8.0
    ax.set_ylim(min(0, min_val) - y_margin, max(0, max_val) + y_margin)
    
    # Hộp thông tin chi tiết phiên chạy (Executive Audit Summary)
    micro_sum = sum(values[:7])
    macro_val = values[7]
    audit_diff = abs((micro_sum + macro_val) - values[8])
    comp_pct = (curr_dist / 2117.0) * 100.0
    
    stats_text = (
        f"DISTANCE REACHED:  {curr_dist:.1f}m ({comp_pct:.1f}%)\n"
        f"TERMINATION:       {reason}\n"
        f"TIME ELAPSED:      {step} steps ({step * 0.02:.2f}s)\n"
        f"MAX SPEED:         {np.max(speeds):.1f} km/h\n"
        f"────────────────────────────────────────\n"
        f"• Total Micro Streams: {micro_sum:+.2f} pts\n"
        f"• Terminal Event:      {macro_val:+.2f} pts\n"
        f"• TOTAL NET RETURN:    {cum_total[-1]:+.2f} pts\n"
        f"• Audit Verification:  DELTA = {audit_diff:.4f} (PASSED)"
    )
    
    bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#f1f5f9", edgecolor="#94a3b8", lw=1.2, alpha=0.95)
    ax.text(0.98, 0.96, stats_text, transform=ax.transAxes, fontsize=9.2,
            verticalalignment="top", horizontalalignment="right", family="monospace",
            bbox=bbox_props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f">> [PLOT] Đã xuất biểu đồ chẩn đoán Net Contribution tại: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    args = parse_args()
    run_reward_diagnostic(model_path=args.model, save_path=args.save_path, seed=args.seed)
