"""
Script Vẽ Biểu Đồ Báo Cáo Đánh Giá Học Thuật (WI-21: Academic Evaluation & Ablation Report)
Đọc các file JSON kết quả từ `training/evaluate.py` (`logs/eval_*.json`) và tạo bộ biểu đồ chuẩn báo cáo vào `plots/`:
  1. `plots/eval_lap_time_hist.png`: Histogram phân bố thời gian hoàn thành vòng sạch (Clean Lap Time) của 3 nhánh:
     - Optimal Ghost (Benchmark lý thuyết / xe đối chứng)
     - PPO Pure (`--obs-mode pure`)
     - PPO + Racing Line (`--obs-mode line`, prior đặc quyền)
  2. `plots/eval_robustness_bars.png`: Biểu đồ cột tỷ lệ hoàn thành vòng sạch (Clean Lap Rate %) theo mức nhiễu quan sát
     (`--obs-noise` sigma in {0, 0.01, 0.02, 0.05}) và độ trễ hành động (`--action-delay` k in {0, 1, 2}).
  3. `plots/eval_speed_trace_and_sectors.png`: Đường tốc độ v(s) dọc vòng đua đặt chồng lên v*(s) của Ghost
     kèm biểu đồ cột Delta t từng khu vực (Sector 1 / Sector 2 / Sector 3).
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_clean_lap_histogram(eval_dir: str, out_dir: str):
    """
    1. Histogram thời gian vòng của 3 nhánh (Optimal / Pure PPO / Line PPO).
    """
    pure_data = load_json_if_exists(os.path.join(eval_dir, "eval_pure_clean.json"))
    line_data = load_json_if_exists(os.path.join(eval_dir, "eval_line_clean.json"))
    opt_data = load_json_if_exists(os.path.join(eval_dir, "eval_optimal_clean.json"))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    has_any = False

    if pure_data and pure_data.get("clean_lap_times"):
        times = pure_data["clean_lap_times"]
        ax.hist(times, bins=10, alpha=0.65, color="#e67e22", edgecolor="black",
                label=f"PPO Pure (35D) — n={len(times)}, μ={np.mean(times):.2f}s ± {np.std(times):.2f}s")
        has_any = True

    if line_data and line_data.get("clean_lap_times"):
        times = line_data["clean_lap_times"]
        ax.hist(times, bins=10, alpha=0.65, color="#2980b9", edgecolor="black",
                label=f"PPO + Line Prior (47D) — n={len(times)}, μ={np.mean(times):.2f}s ± {np.std(times):.2f}s")
        has_any = True

    if opt_data and opt_data.get("clean_lap_times"):
        times = opt_data["clean_lap_times"]
        ax.axvline(float(np.mean(times)), color="#27ae60", linestyle="--", linewidth=2.2,
                   label=f"Optimal Ghost Car — {np.mean(times):.2f}s")
        has_any = True
    else:
        # Vẽ mốc lý thuyết T* từ kết quả eval
        ref_t = None
        for d in (line_data, pure_data):
            if d and d.get("theoretical_best_lap_time"):
                ref_t = float(d["theoretical_best_lap_time"])
                break
        if ref_t is not None:
            ax.axvline(ref_t, color="#27ae60", linestyle="--", linewidth=2.2,
                       label=f"Theoretical Minimum Time T* — {ref_t:.2f}s")
            has_any = True

    if not has_any:
        ax.text(0.5, 0.5, "Chưa có dữ liệu clean_lap_times trong eval_pure_clean.json / eval_line_clean.json",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.set_title("Phân bố Thời gian Hoàn thành Vòng Sạch (Clean Lap Time Distribution)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Lap Time (s)", fontsize=11)
    ax.set_ylabel("Số vòng đua hoàn thành (Count)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    if has_any:
        ax.legend(loc="upper right", fontsize=9.5)

    out_path = os.path.join(out_dir, "eval_lap_time_hist.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">> Đã lưu biểu đồ Histogram Lap Time: {out_path}")


def plot_robustness_bars(eval_dir: str, out_dir: str):
    """
    2. Biểu đồ cột tỷ lệ hoàn thành vòng theo mức nhiễu quan sát và độ trễ hành động.
    Đọc các file:
      - eval_{branch}_clean.json (sigma=0, delay=0)
      - eval_{branch}_noise001.json, eval_{branch}_noise002.json, eval_{branch}_noise005.json
      - eval_{branch}_delay1.json, eval_{branch}_delay2.json
    """
    branches = [
        ("pure", "PPO Pure (35D)", "#e67e22"),
        ("line", "PPO + Line Prior (47D)", "#2980b9"),
    ]
    noise_configs = [
        ("clean", "σ = 0.00\n(Clean)"),
        ("noise001", "σ = 0.01\n(1% Noise)"),
        ("noise002", "σ = 0.02\n(2% Noise)"),
        ("noise005", "σ = 0.05\n(5% Noise)"),
    ]
    delay_configs = [
        ("clean", "k = 0\n(0 ms)"),
        ("delay1", "k = 1\n(100 ms)"),
        ("delay2", "k = 2\n(200 ms)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)

    # Panel 1: Observation Noise
    x1 = np.arange(len(noise_configs))
    width = 0.34
    for b_idx, (b_key, b_label, b_color) in enumerate(branches):
        rates = []
        for suffix, _ in noise_configs:
            data = load_json_if_exists(os.path.join(eval_dir, f"eval_{b_key}_{suffix}.json"))
            rate = float(data.get("clean_lap_rate", data.get("success_rate", 0.0)) * 100.0) if data else 0.0
            rates.append(rate)
        offset = (b_idx - 0.5) * width
        bars = ax1.bar(x1 + offset, rates, width, label=b_label, color=b_color, edgecolor="black", alpha=0.85)
        for bar, r in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5, f"{r:.0f}%",
                     ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl in noise_configs])
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("Clean Lap Completion Rate (%)", fontsize=11)
    ax1.set_title("Độ bền với Nhiễu Cảm biến (Observation Noise)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.5)

    # Panel 2: Action Delay
    x2 = np.arange(len(delay_configs))
    for b_idx, (b_key, b_label, b_color) in enumerate(branches):
        rates = []
        for suffix, _ in delay_configs:
            data = load_json_if_exists(os.path.join(eval_dir, f"eval_{b_key}_{suffix}.json"))
            rate = float(data.get("clean_lap_rate", data.get("success_rate", 0.0)) * 100.0) if data else 0.0
            rates.append(rate)
        offset = (b_idx - 0.5) * width
        bars = ax2.bar(x2 + offset, rates, width, label=b_label, color=b_color, edgecolor="black", alpha=0.85)
        for bar, r in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5, f"{r:.0f}%",
                     ha="center", va="bottom", fontsize=8.5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in delay_configs])
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Clean Lap Completion Rate (%)", fontsize=11)
    ax2.set_title("Độ bền với Độ trễ Cơ cấu Chấp hành (Action Delay)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9.5)

    out_path = os.path.join(out_dir, "eval_robustness_bars.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">> Đã lưu biểu đồ Độ bền (Noise & Delay): {out_path}")


def collect_single_lap_trace(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Chạy 1 vòng deterministic từ checkpoint (nếu có) để thu thập đường tốc độ v(s) dọc vòng đua.
    """
    if not model_path or not os.path.exists(model_path):
        return None

    from game_env.racing_env import MetaRacingEnv
    from game_env.vehicle import get_vehicle
    from agents.ppo_agent import PPOAgent

    meta = PPOAgent.peek_meta(model_path)
    map_name = meta.get("map", "silverstone_full")
    veh_name = meta.get("vehicle", "gt3_real")
    obs_mode = meta.get("obs_mode", "pure")

    car = get_vehicle(veh_name)
    env = MetaRacingEnv(
        map_type=map_name,
        use_render=False,
        num_scenarios=1,
        start_seed=100,
        vehicle=car,
        max_laps=1,
        obs_mode=obs_mode,
    )
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    agent = PPOAgent(state_dim=obs_dim, action_dim=act_dim, logstd_max=env.preset.logstd_max)
    agent.load(model_path)

    obs, _ = env.reset(seed=100, options={"force_start": True, "start_mode": "standing"})
    s_list: List[float] = []
    v_list: List[float] = []
    t_list: List[float] = []

    max_steps = int(3000 * env.preset.time_scale)
    for step in range(max_steps):
        s_list.append(float(env.last_s))
        v_list.append(float(env.car.speed_km_h))
        t_list.append(float(step * env.dt))
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            s_list.append(float(env.last_s))
            v_list.append(float(env.car.speed_km_h))
            t_list.append(float((step + 1) * env.dt))
            break

    bench = env.get_benchmark_engine()
    s_opt = np.asarray(bench.track_s, dtype=np.float32).copy()
    v_opt_kmh = (bench.optimal_speed_profile * 3.6).copy()
    circuit_len = float(env.circuit_length)
    env.close()

    return {
        "s": np.asarray(s_list, dtype=np.float32),
        "v_kmh": np.asarray(v_list, dtype=np.float32),
        "t": np.asarray(t_list, dtype=np.float32),
        "s_opt": s_opt,
        "v_opt_kmh": v_opt_kmh,
        "circuit_length": circuit_len,
    }


def _compute_sector_times(s_arr: np.ndarray, t_arr: np.ndarray, circuit_len: float) -> List[float]:
    """Chia đường đua thành 3 Sector đều nhau và nội suy thời gian từng Sector."""
    boundaries = [0.0, circuit_len / 3.0, 2.0 * circuit_len / 3.0, circuit_len]
    if len(s_arr) < 2 or float(np.max(s_arr)) < boundaries[1]:
        return [0.0, 0.0, 0.0]

    # Đảm bảo s_arr đơn điệu tăng để nội suy
    s_mono = np.maximum.accumulate(s_arr)
    max_s = float(s_mono[-1])
    sec_times = []
    for i in range(3):
        s_start, s_end = boundaries[i], boundaries[i + 1]
        if max_s >= s_end * 0.98:
            t0 = float(np.interp(s_start, s_mono, t_arr))
            t1 = float(np.interp(min(s_end, max_s), s_mono, t_arr))
            sec_times.append(max(0.0, t1 - t0))
        else:
            sec_times.append(0.0)
    return sec_times


def plot_speed_trace_and_sectors(
    out_dir: str,
    pure_model: Optional[str] = None,
    line_model: Optional[str] = None,
    npz_profile: str = "scripts/silverstone_full_gt3_realcar_optimal_profile.npz",
):
    """
    3. Đường tốc độ v(s) dọc vòng đua đặt chồng lên v*(s) của Ghost + biểu đồ cột Delta t từng Sector.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.5, 7.5), dpi=150, gridspec_kw={"height_ratios": [1.6, 1.0]})

    s_opt = None
    v_opt_kmh = None
    circuit_len = 5891.0

    if os.path.exists(npz_profile):
        data = np.load(npz_profile)
        v_opt_mps = data["optimal_speed_profile"].astype(np.float32)
        v_opt_kmh = v_opt_mps * 3.6
        if "s_grid" in data:
            s_opt = data["s_grid"].astype(np.float32)
            circuit_len = float(s_opt[-1])
        else:
            s_opt = np.linspace(0.0, circuit_len, len(v_opt_kmh), dtype=np.float32)

    pure_trace = collect_single_lap_trace(pure_model) if pure_model else None
    line_trace = collect_single_lap_trace(line_model) if line_model else None

    for tr in (line_trace, pure_trace):
        if tr is not None:
            s_opt = tr["s_opt"]
            v_opt_kmh = tr["v_opt_kmh"]
            circuit_len = tr["circuit_length"]
            break

    t_opt_sectors = [0.0, 0.0, 0.0]
    if s_opt is not None and v_opt_kmh is not None:
        ax_top.plot(s_opt, v_opt_kmh, color="#27ae60", linestyle="--", linewidth=2.0, label="Optimal Reference v*(s)")
        ds = np.diff(s_opt, prepend=0.0)
        v_mps_safe = np.maximum(v_opt_kmh / 3.6, 5.0)
        t_opt_cum = np.cumsum(ds / v_mps_safe)
        t_opt_sectors = _compute_sector_times(s_opt, t_opt_cum, circuit_len)

    if pure_trace is not None:
        ax_top.plot(pure_trace["s"], pure_trace["v_kmh"], color="#e67e22", linewidth=1.6, alpha=0.85, label="PPO Pure (35D)")
    if line_trace is not None:
        ax_top.plot(line_trace["s"], line_trace["v_kmh"], color="#2980b9", linewidth=1.6, alpha=0.90, label="PPO + Line Prior (47D)")

    # Đánh dấu ranh giới Sector 1 / 2 / 3
    for sec_idx, s_bound in enumerate([circuit_len / 3.0, 2.0 * circuit_len / 3.0], start=1):
        ax_top.axvline(s_bound, color="gray", linestyle=":", alpha=0.7)
        ax_top.text(s_bound - 120, 255, f"S{sec_idx}|S{sec_idx+1}", fontsize=8.5, color="dimgray")

    ax_top.set_title("Biên dạng Tốc độ v(s) Dọc Đường đua Silverstone Full vs. Quỹ đạo Tối ưu v*(s)", fontsize=12.5, fontweight="bold")
    ax_top.set_xlabel("Quãng đường dọc tâm đường s (m)", fontsize=10.5)
    ax_top.set_ylabel("Tốc độ (km/h)", fontsize=10.5)
    ax_top.set_xlim(0, circuit_len)
    ax_top.set_ylim(0, 280)
    ax_top.grid(True, linestyle=":", alpha=0.6)
    ax_top.legend(loc="lower right", fontsize=9.5)

    # Panel dưới: Sector Delta t so với v*(s)
    sectors = ["Sector 1\n(Start -> Maggotts)", "Sector 2\n(Becketts -> Stowe)", "Sector 3\n(Vale -> Finish)"]
    x_sec = np.arange(len(sectors))
    width = 0.32

    if pure_trace is not None:
        p_sec = _compute_sector_times(pure_trace["s"], pure_trace["t"], circuit_len)
        p_delta = [p_sec[i] - t_opt_sectors[i] if p_sec[i] > 0 else 0.0 for i in range(3)]
        bars_p = ax_bot.bar(x_sec - width / 2, p_delta, width, color="#e67e22", edgecolor="black", label="PPO Pure Δt vs v*")
        for bar, dt_val in zip(bars_p, p_delta):
            if dt_val != 0.0:
                ax_bot.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{dt_val:+.2f}s",
                            ha="center", va="bottom", fontsize=8.5)

    if line_trace is not None:
        l_sec = _compute_sector_times(line_trace["s"], line_trace["t"], circuit_len)
        l_delta = [l_sec[i] - t_opt_sectors[i] if l_sec[i] > 0 else 0.0 for i in range(3)]
        bars_l = ax_bot.bar(x_sec + width / 2, l_delta, width, color="#2980b9", edgecolor="black", label="PPO + Line Prior Δt vs v*")
        for bar, dt_val in zip(bars_l, l_delta):
            if dt_val != 0.0:
                ax_bot.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{dt_val:+.2f}s",
                            ha="center", va="bottom", fontsize=8.5)

    if pure_trace is None and line_trace is None:
        ax_bot.bar(x_sec, t_opt_sectors, width, color="#27ae60", edgecolor="black", alpha=0.75, label="Theoretical Sector Time T* (s)")
        for i, val in enumerate(t_opt_sectors):
            ax_bot.text(x_sec[i], val + 0.5, f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        ax_bot.set_ylabel("Thời gian lý thuyết T* (s)", fontsize=10.5)
    else:
        ax_bot.axhline(0.0, color="black", linewidth=1.0)
        ax_bot.set_ylabel("Độ lệch thời gian Δt = t_agent - t* (s)", fontsize=10.5)

    ax_bot.set_xticks(x_sec)
    ax_bot.set_xticklabels(sectors)
    ax_bot.set_title("Phân tích Thời gian theo Khu vực Đường đua (Sector Breakdown)", fontsize=12, fontweight="bold")
    ax_bot.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_bot.legend(loc="upper right", fontsize=9.5)

    out_path = os.path.join(out_dir, "eval_speed_trace_and_sectors.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">> Đã lưu biểu đồ Speed Trace & Sector Breakdown: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Academic Evaluation Report Figures (WI-21)")
    parser.add_argument("--eval-dir", type=str, default="logs", help="Thư mục chứa các file eval_*.json")
    parser.add_argument("--out-dir", type=str, default="plots", help="Thư mục xuất ảnh PNG")
    parser.add_argument("--pure-model", type=str, default=None, help="Checkpoint PPO Pure để vẽ trace v(s) (tùy chọn)")
    parser.add_argument("--line-model", type=str, default=None, help="Checkpoint PPO + Line để vẽ trace v(s) (tùy chọn)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_clean_lap_histogram(args.eval_dir, args.out_dir)
    plot_robustness_bars(args.eval_dir, args.out_dir)
    plot_speed_trace_and_sectors(args.out_dir, pure_model=args.pure_model, line_model=args.line_model)
    print(">> Hoàn tất xuất toàn bộ biểu đồ WI-21 vào thư mục:", args.out_dir)


if __name__ == "__main__":
    main()

