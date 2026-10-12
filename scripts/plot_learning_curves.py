"""
SCRIPT XUẤT ĐỒ THỊ LEARNING CURVES NGOẠI TUYẾN (STANDALONE OFFLINE PLOTTER)
Dùng khi bạn cần trích xuất file ảnh PNG để viết báo cáo hoặc nộp bài, 
đọc trực tiếp từ logs/training_metrics.json mà không ảnh hưởng tới tiến trình train.

Cách chạy:
python scripts/plot_learning_curves.py --metrics logs/training_metrics.json --save-path plots/learning_curves.png
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Learning Curves Plotter")
    parser.add_argument("--metrics", type=str, default="logs/training_metrics.json", help="File JSON chứa metrics")
    parser.add_argument("--save-path", type=str, default="plots/learning_curves.png", help="Đường dẫn lưu file ảnh")
    parser.add_argument("--smooth-window", type=int, default=3, help="Cửa sổ làm mượt đường xu hướng (SMA)")
    return parser.parse_args()

def moving_average(data, window_size=3):
    if len(data) < window_size or window_size <= 1:
        return data
    window = np.ones(int(window_size)) / float(window_size)
    # Mode same để giữ nguyên độ dài
    return np.convolve(data, window, mode="same")

def safe_extract(metrics, key):
    val = metrics.get(key, [])
    if isinstance(val, list):
        return np.array(val, dtype=np.float32)
    return np.array([], dtype=np.float32)

def main():
    args = parse_args()
    if not os.path.exists(args.metrics):
        print(f">> [LỖI] Không tìm thấy file: {args.metrics}")
        return
        
    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    steps = safe_extract(metrics, "steps")
    if len(steps) == 0:
        print(">> [CẢNH BÁO] Không có dữ liệu steps trong metrics.")
        return
        
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Thiết lập giao diện biểu đồ chuyên nghiệp
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    total_finishes = metrics.get("total_finishes", 0)
    fig.suptitle(
        f"PPO Autonomous Racing Agent Learning Diagnostics\n"
        f"(Total Steps: {int(steps[-1]):,} | Completed Laps: {total_finishes})", 
        fontsize=15, fontweight="bold"
    )

    def align_and_plot(ax, x, y, color, label, marker=None, lw=2.0, ls="-", alpha=1.0):
        n = min(len(x), len(y))
        if n == 0:
            return None
        return ax.plot(x[:n], y[:n], color=color, label=label, marker=marker, lw=lw, linestyle=ls, alpha=alpha)

    # -------------------------------------------------------------
    # 1. Episode Return (Learning Convergence)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ret_arr = safe_extract(metrics, "eval_return")
    if len(ret_arr) > 0:
        n1 = min(len(steps), len(ret_arr))
        x1, y1 = steps[:n1], ret_arr[:n1]
        ax1.plot(x1, y1, color="#1f77b4", alpha=0.35, lw=1.2, marker="o", markersize=4, label="Eval Return (Raw)")
        if len(y1) >= args.smooth_window and args.smooth_window > 1:
            y1_smooth = moving_average(y1, args.smooth_window)
            ax1.plot(x1, y1_smooth, color="#1f77b4", lw=2.5, label=f"Trend (SMA-{args.smooth_window})")
        max_ret = np.max(y1)
        last_ret = y1[-1]
        ax1.axhline(max_ret, color="#1f77b4", linestyle=":", alpha=0.6, label=f"Max: {max_ret:.2f}")
        ax1.text(0.03, 0.92, f"Latest Return: {last_ret:.2f}\nPeak Return: {max_ret:.2f}", 
                 transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.8))
                 
    ax1.set_title("1. Mean Episodic Return (Reward Convergence)", fontweight="bold")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Total Return")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8)

    # -------------------------------------------------------------
    # 2. Track Completion Rate (%) & Milestones
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    comp_arr = safe_extract(metrics, "eval_completion")
    if len(comp_arr) > 0:
        n2 = min(len(steps), len(comp_arr))
        x2, y2 = steps[:n2], comp_arr[:n2]
        ax2.plot(x2, y2, color="#2ca02c", lw=2.2, marker="s", markersize=4, label="Track Completion %")
        max_c = np.max(y2)
        ax2.axhline(max_c, color="#2ca02c", linestyle=":", lw=1.8, label=f"Best: {max_c:.1f}%")
        # Key track sectors
        ax2.axhline(50.0, color="gray", linestyle="--", alpha=0.4, label="Mid-track (50%)")
        ax2.axhline(75.4, color="orange", linestyle="--", alpha=0.6, label="Sector 3 Corner (75.4%)")
        ax2.axhline(100.0, color="gold", linestyle="--", alpha=0.7, label="Finish Line (100%)")
        ax2.text(0.03, 0.88, f"Max Progress: {max_c:.1f}%\nTotal Finishes: {total_finishes}",
                 transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.8))

    ax2.set_title("2. Track Completion Rate (%) & Circuit Progression", fontweight="bold")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Completion (%)")
    ax2.set_ylim(-5, 108)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="center right", fontsize=8)

    # -------------------------------------------------------------
    # 3. Policy Loss, Value Loss (Twin Axes) & Entropy
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    val_loss = safe_extract(metrics, "value_loss")
    pol_loss = safe_extract(metrics, "policy_loss")
    entropy = safe_extract(metrics, "entropy")
    
    # Left axis: Value Loss (MSE)
    if len(val_loss) > 0:
        n_v = min(len(steps), len(val_loss))
        ax3.plot(steps[:n_v], val_loss[:n_v], color="#ff7f0e", lw=2.0, label="Value Loss (MSE)")
    ax3.set_title("3. Critic (Value) & Actor (Policy) Loss Convergence", fontweight="bold")
    ax3.set_xlabel("Environment Steps")
    ax3.set_ylabel("Value Loss (MSE)", color="#ff7f0e", fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#ff7f0e")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Right axis: Policy Loss & Policy Entropy
    ax3_twin = ax3.twinx()
    if len(pol_loss) > 0:
        n_p = min(len(steps), len(pol_loss))
        ax3_twin.plot(steps[:n_p], pol_loss[:n_p], color="#d62728", lw=1.8, linestyle="-", label="Policy Loss")
    if len(entropy) > 0 and np.any(entropy > 0):
        n_e = min(len(steps), len(entropy))
        ax3_twin.plot(steps[:n_e], entropy[:n_e], color="#9467bd", lw=1.5, linestyle=":", label="Entropy")
    ax3_twin.set_ylabel("Policy Loss / Entropy", color="#d62728", fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#d62728")
    
    # Merge legends for ax3 and ax3_twin
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3t, labels_3 + labels_3t, loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # 4. Racing Speed & Safety (Crash Rate)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    color_speed = "#007acc"
    color_crash = "#d9534f"
    
    speed_arr = safe_extract(metrics, "eval_speed")
    if len(speed_arr) > 0:
        n_s = min(len(steps), len(speed_arr))
        ax4.plot(steps[:n_s], speed_arr[:n_s], color=color_speed, lw=2.2, label="Mean Speed (km/h)")
        max_s = np.max(speed_arr[:n_s])
        ax4.axhline(max_s, color=color_speed, linestyle=":", alpha=0.6, label=f"Max: {max_s:.1f} km/h")
        ax4.text(0.03, 0.88, f"Top Eval Speed: {max_s:.1f} km/h\nLatest: {speed_arr[n_s-1]:.1f} km/h",
                 transform=ax4.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_speed, alpha=0.8))

    ax4.set_title("4. Speed Progression vs. Safety Index", fontweight="bold")
    ax4.set_xlabel("Environment Steps")
    ax4.set_ylabel("Speed (km/h)", color=color_speed, fontweight="bold")
    ax4.tick_params(axis="y", labelcolor=color_speed)

    ax4_twin = ax4.twinx()
    crash_arr = safe_extract(metrics, "crash_rate")
    if len(crash_arr) > 0:
        n_c = min(len(steps), len(crash_arr))
        crash_pct = crash_arr[:n_c] * 100.0  # Chuyển sang %
        ax4_twin.plot(steps[:n_c], crash_pct, color=color_crash, label="Crash Rate (%)", lw=2, linestyle="--")
        ax4_twin.set_ylabel("Crash Rate (%)", color=color_crash, fontweight="bold")
        ax4_twin.tick_params(axis="y", labelcolor=color_crash)
        ax4_twin.set_ylim(-5, 105)
        
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    lines_4t, labels_4t = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_4 + lines_4t, labels_4 + labels_4t, loc="center right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Đánh dấu các mốc Resume trên tất cả các subplot
    # -------------------------------------------------------------
    for mark in metrics.get("resume_marks", []):
        m_step = mark.get("step", 0)
        m_label = mark.get("source", "Resume")
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(m_step, color="crimson", linestyle=":", alpha=0.8, lw=1.6)
        ax1.text(m_step, ax1.get_ylim()[0] + 0.15 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), 
                 f" ⟲ {m_label}", color="crimson", fontsize=8, fontweight="bold", rotation=90)
        ax2.text(m_step, 45, f" ⟲ {m_label}", color="crimson", fontsize=8, fontweight="bold", rotation=90)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    plt.close()
    print(f">> [PLOT] Đã xuất biểu đồ học thuật tại: {args.save_path}")

if __name__ == "__main__":
    main()
