"""
Phân Tích Thói Quen Lái Xe Của Checkpoint (Driving Behavior Trace & Telemetry Analysis)
Được phát triển từ scripts/drive_trace.py theo kế hoạch Phase 2 (WI-12 / WI-P1C).
Đo đạc và phân tích định lượng thói quen lái xe của một hoặc nhiều checkpoint PPO qua các vòng đua:
- Tốc độ trung bình, tốc độ tối đa, % thời gian ở vận tốc bão hòa (v_max - 5 km/h).
- Tần suất đảo chiều vô lăng (reversals / 100 steps), độ lệch chuẩn vô lăng trên đoạn thẳng (|kappa| < 0.004).
- Độ lệch chuẩn quỹ đạo (lateral offset std) và độ lệch tâm trung bình trên đoạn thẳng.
- Hành vi bàn đạp: Ga trung bình, % hết ga (>95%), % phanh (>5%), % thả trôi (coast: thr < 50% & không phanh).
Trích xuất trực tiếp tọa độ hình học và độ lệch ngang từ env (không giải mã ngược từ obs).
"""

import sys
import os
import argparse
from typing import List, Tuple
import numpy as np

# Thiết lập UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Driving Trace of RL Checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["models/p1_main/best_agent.pt", "models/ref_111s/best_agent.pt"],
        help="Danh sách các file checkpoint cần đánh giá (cách nhau bởi khoảng trắng)",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="silverstone_half",
        choices=["silverstone", "silverstone_half", "silverstone_full"],
        help="Bản đồ đua ('silverstone', 'silverstone_half', 'silverstone_full')",
    )
    parser.add_argument(
        "--vehicle",
        type=str,
        default="gt3",
        choices=["gt3", "gt3_anchor", "gt3_real", "sedan"],
        help="Loại xe ('gt3', 'gt3_real', 'sedan')",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=1,
        help="Số vòng đua kiểm tra",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed tạo môi trường",
    )
    return parser.parse_args()


def trace_model(model_path: str, args):
    if not os.path.exists(model_path):
        print(f"⚠️ [BỎ QUA] Không tìm thấy checkpoint: {model_path}")
        return

    model_name = os.path.basename(model_path)
    race_car = get_vehicle(args.vehicle)
    env = MetaRacingEnv(
        map_type=args.map,
        vehicle=race_car,
        use_render=False,
        max_laps=args.laps,
        start_seed=args.seed,
    )

    obs_shape = getattr(env.observation_space, "shape", (35,))
    act_shape = getattr(env.action_space, "shape", (3,))
    obs_dim = int(obs_shape[0])
    act_dim = int(act_shape[0])

    agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        device="cpu",
        logstd_max=env.preset.logstd_max,
    )
    agent.load(model_path)

    obs, info = env.reset(seed=args.seed, options={"start_mode": "standing"})
    max_steps = int(3500 * env.preset.time_scale * args.laps)

    steer_list = []
    action_list = []
    speed_list = []
    lat_off_list = []
    kappa_list = []
    last_info = {}

    for _ in range(max_steps):
        act, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, last_info = env.step(act)

        # Lấy trực tiếp từ thực thể xe và hình học sa bàn trong env
        pos = env.car.position
        v_kmh = float(env.car.speed_km_h)
        steer_cmd = float(act[0])

        idx, cur_s = env._get_track_s(pos, env.current_track_idx)
        pt = env.track_xy[idx]
        th = float(env.track_theta[idx])
        lat_norm = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
        lat_off = float(np.dot(pos - pt, lat_norm))
        kappa = float(env.track_kappa[idx])

        steer_list.append(steer_cmd)
        action_list.append(act.copy())
        speed_list.append(v_kmh)
        lat_off_list.append(lat_off)
        kappa_list.append(kappa)

        if terminated or truncated:
            break

    env.close()

    if len(steer_list) < 10:
        print(f"❌ {model_name}: Crash quá sớm (< 10 bước).")
        return

    S = np.array(steer_list, dtype=np.float32)
    A = np.array(action_list, dtype=np.float32)
    V = np.array(speed_list, dtype=np.float32)
    L = np.array(lat_off_list, dtype=np.float32)
    K = np.abs(np.array(kappa_list, dtype=np.float32))

    dS = np.diff(S)
    reversals = int(np.sum((np.sign(dS[1:]) != np.sign(dS[:-1])) & (np.abs(dS[1:]) > 0.02)))
    rev_per_100 = (reversals / len(S)) * 100.0

    scale_curv = float(1.0 / (env.preset.length_scale if hasattr(env.preset, "length_scale") else 1.0))
    straight = K < (0.004 * scale_curv)
    steer_straight_std = float(S[straight].std()) if np.any(straight) else 0.0

    thr = 0.5 * (A[:, 1] + 1.0)
    brk = 0.5 * (A[:, 2] + 1.0)
    v_cap = race_car.cfg.max_speed_km_h

    pct_high_speed = float(np.mean(V >= (v_cap - 5.0)) * 100.0)
    pct_full_thr = float(np.mean(thr > 0.95) * 100.0)
    pct_brk = float(np.mean(brk > 0.05) * 100.0)
    pct_coast = float(np.mean((thr < 0.5) & (brk <= 0.05)) * 100.0)

    term_reason = last_info.get("termination_reason", "running")
    lap_time = float(last_info.get("completed_lap_time", last_info.get("lap_time", 0.0)))

    print(f"\n========================================================")
    print(f"📌 CHECKPOINT: {model_name} (Map: {args.map}, Xe: {args.vehicle})")
    print(f"========================================================")
    print(f"  • Bước chạy: {len(S)} | Lý do kết thúc: {term_reason} | Lap time: {lap_time:.2f}s")
    print(f"  • Tốc độ: TB = {V.mean():.1f} km/h | Max = {V.max():.1f} km/h | Time >= {v_cap-5:.0f} km/h: {pct_high_speed:.1f}%")
    print(f"  • Vô lăng: Đảo chiều = {rev_per_100:.1f}/100 bước | Std trên đoạn thẳng = {steer_straight_std:.4f}")
    print(f"  • Quỹ đạo: Độ lệch chuẩn lệch tâm = {L.std():.2f}m | Lệch tâm TB đoạn thẳng = {np.abs(L[straight]).mean():.2f}m")
    print(f"  • Chân ga/phanh: Ga TB = {thr.mean():.2f} | Full ga (>95%) = {pct_full_thr:.1f}% | Có phanh (>5%) = {pct_brk:.1f}% | Thả trôi = {pct_coast:.1f}%")


def main():
    args = parse_args()
    print("=" * 65)
    print("🚗 PHÂN TÍCH TELEMETRY & THÓI QUEN LÁI XE (DRIVE TRACE)")
    print("=" * 65)
    for m in args.models:
        trace_model(m, args)
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()

