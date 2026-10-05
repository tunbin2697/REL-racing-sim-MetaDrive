"""
Script Xuất Bộ 4 Biểu Đồ Sim Racing Telemetry Học Thuật:
1. Speed Trace (v vs s): Phân tích đỉnh cua V-min, góc cua chữ V (Chicane) và chữ U (Sweeper).
2. Delta Time Channel: Phân tích chênh lệch thời gian thực trên từng mét đường.
3. Pedal Trace: Phân tích kỹ thuật Trail Braking (tam giác dốc) và chân ga mượt mà.
4. G-G Diagram: Vòng tròn ma sát khai thác độ bám lốp tối đa (Friction Circle).
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Sim Racing Telemetry Analysis")
    parser.add_argument("--lap-file", type=str, default="logs/reference_lap.json", help="Đường dẫn file telemetry vòng đua")
    parser.add_argument("--save-path", type=str, default="plots/telemetry_analysis.png", help="File ảnh lưu biểu đồ")
    return parser.parse_args()

def plot_telemetry(lap_data, save_path):
    samples = lap_data.get("samples", [])
    if not samples:
        print(f">> [LỖI] File không có dữ liệu mẫu telemetry: {save_path}")
        return
        
    distances = np.array([s["distance"] for s in samples])
    speeds = np.array([s["speed_km_h"] for s in samples])
    times = np.array([s["time"] for s in samples])
    throttles = np.array([s.get("throttle", 0.0) for s in samples])
    brakes = np.array([s.get("brake", 0.0) for s in samples])
    acc_longs = np.array([s.get("acc_long_g", 0.0) for s in samples])
    acc_lats = np.array([s.get("acc_lat_g", 0.0) for s in samples])
    steerings = np.array([s.get("steering", 0.0) for s in samples])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Thiết lập giao diện biểu đồ chuyên nghiệp chuẩn thể thao
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1.2])
    fig.suptitle(f"Sim Racing Telemetry Analysis | Lap Time: {lap_data.get('lap_time', times[-1]):.2f}s | Distance: {distances[-1]:.1f}m", 
                 fontsize=16, fontweight="bold", y=0.98)

    # -------------------------------------------------------------
    # 1. SPEED TRACE v(s)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(distances, speeds, color="#00ffff" if plt.style.available else "blue", lw=2.2, label="Vehicle Speed (km/h)")
    ax1.set_title("1. Speed Trace: V-min Analysis Across Corners (V-shape vs U-shape Apex)", fontweight="bold")
    ax1.set_ylabel("Speed (km/h)", fontweight="bold")
    ax1.set_xlabel("Track Distance (meters)", fontweight="bold")
    
    # Đánh dấu các điểm V-min (đỉnh cua)
    from scipy.signal import find_peaks
    valleys, _ = find_peaks(-speeds, distance=30, prominence=5)
    for v_idx in valleys:
        ax1.plot(distances[v_idx], speeds[v_idx], "ro", markersize=6)
        ax1.annotate(f"Vmin: {speeds[v_idx]:.0f}", (distances[v_idx], speeds[v_idx]),
                     textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, color="red")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # 2. PEDAL INPUT TRACE (Brake & Throttle)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(distances, throttles * 100.0, color="#00ff00" if plt.style.available else "green", lw=1.8, label="Throttle (%)")
    ax2.plot(distances, brakes * 100.0, color="#ff3333" if plt.style.available else "red", lw=2.0, label="Brake (%) - Trail Braking Profile")
    ax2.set_title("2. Pedal Inputs: Trail Braking & Throttle Progression", fontweight="bold")
    ax2.set_ylabel("Pedal Input (%)", fontweight="bold")
    ax2.set_xlabel("Track Distance (meters)", fontweight="bold")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # 3. G-G FRICTION CIRCLE DIAGRAM
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    scatter = ax3.scatter(acc_lats, acc_longs, c=speeds, cmap="plasma", alpha=0.6, s=15, edgecolors="none")
    cb = fig.colorbar(scatter, ax=ax3, orientation="vertical", pad=0.02)
    cb.set_label("Speed (km/h)", fontsize=9)
    
    # Vẽ vòng tròn ma sát giới hạn 1.0G và 1.5G
    theta = np.linspace(0, 2 * np.pi, 200)
    for g_lim in [0.5, 1.0, 1.5]:
        ax3.plot(g_lim * np.cos(theta), g_lim * np.sin(theta), linestyle="--", color="gray", alpha=0.6, lw=1)
        ax3.text(0.05, g_lim, f"{g_lim:.1f}G", color="gray", fontsize=8)
        
    ax3.set_title("3. G-G Diagram: Friction Circle Exploitation", fontweight="bold")
    ax3.set_xlabel("Lateral Acceleration (G) [Cornering]", fontweight="bold")
    ax3.set_ylabel("Longitudinal Acc (G) [Accel (+) / Brake (-)]", fontweight="bold")
    ax3.set_xlim(-1.8, 1.8)
    ax3.set_ylim(-1.8, 1.8)
    ax3.axhline(0, color="gray", lw=0.8, alpha=0.5)
    ax3.axvline(0, color="gray", lw=0.8, alpha=0.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # 4. STEERING TRACE & CORNERING INTENSITY
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(distances, steerings, color="#ff9900", lw=1.8, label="Steering Angle [-1, 1]")
    ax4.set_title("4. Steering Angle Profile Along Track", fontweight="bold")
    ax4.set_ylabel("Steering Input [-1 to 1]", fontweight="bold")
    ax4.set_xlabel("Track Distance (meters)", fontweight="bold")
    ax4.set_ylim(-1.1, 1.1)
    ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f">> [PLOT] Đã xuất thành công bộ 4 biểu đồ Telemetry tại: {save_path}")

def main():
    args = parse_args()
    if not os.path.exists(args.lap_file):
        print(f">> [LỖI] Không tìm thấy file {args.lap_file}. Hãy chạy tạo reference lap trước!")
        sys.exit(1)
        
    with open(args.lap_file, "r", encoding="utf-8") as f:
        lap_data = json.load(f)
        
    plot_telemetry(lap_data, args.save_path)

if __name__ == "__main__":
    main()
