"""
So sánh nhiều policy trên cùng đường đua (mỗi policy chạy N vòng deterministic từ vạch xuất phát).
Mỗi policy dùng một env mới để cùng chuỗi episode, tránh thiên lệch do trạng thái engine tích lũy.

Cách chạy:
    python scripts/benchmark_policies.py rule rule-fast models/final_agent.pt models/v2/best_agent.pt --episodes 20
"""

import os
import sys
import json
import argparse
import numpy as np

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from agents.policy_loader import load_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark nhiều policy (rule / PPO v1 / PPO v2)")
    parser.add_argument("policies", nargs="+", help="'rule', 'rule-fast' hoặc đường dẫn checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=str, default="logs/benchmark_policies.json")
    return parser.parse_args()


def run_policy(spec, episodes):
    env = MetaRacingEnv(use_render=False)
    policy = load_policy(spec)
    finishes, completions, lap_times, speeds, crash_at = 0, [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset(seed=42, options={"force_start": True})
        ep_speeds = []
        while True:
            action, _, _ = policy.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep_speeds.append(info["speed_km_h"])
            if terminated or truncated:
                break
        completions.append(info["route_completion"] * 100.0)
        speeds.append(float(np.mean(ep_speeds)))
        if info.get("termination_reason") == "lap_completed":
            finishes += 1
            lap_times.append(info["lap_time"])
        else:
            crash_at.append(int(round(info["track_s"])))
    env.close()
    return {
        "policy": spec,
        "episodes": episodes,
        "finish_rate": finishes / episodes,
        "mean_completion": float(np.mean(completions)),
        "mean_speed_kmh": float(np.mean(speeds)),
        "mean_lap_time": float(np.mean(lap_times)) if lap_times else None,
        "best_lap_time": float(min(lap_times)) if lap_times else None,
        "crash_locations": sorted(crash_at),
    }


def main():
    args = parse_args()
    results = []
    for spec in args.policies:
        res = run_policy(spec, args.episodes)
        results.append(res)
        lap = f"{res['mean_lap_time']:.1f}s (best {res['best_lap_time']:.1f}s)" if res["mean_lap_time"] else "-"
        print(
            f"{spec:<32} finish {res['finish_rate'] * 100:5.1f}% | comp {res['mean_completion']:5.1f}% | "
            f"speed {res['mean_speed_kmh']:4.1f} km/h | lap {lap} | crashes {res['crash_locations']}",
            flush=True,
        )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f">> Đã lưu kết quả: {args.out}")


if __name__ == "__main__":
    main()
