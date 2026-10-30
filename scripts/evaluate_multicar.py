"""
Script Đánh Giá Hàng Loạt Checkpoint Đa Tác Tử (6 xe self-play)
- Mỗi lượt eval khi train chỉ chạy 1 cuộc đua -> dao động rất mạnh (cùng model có thể 0/6 hoặc 5/6 về đích).
- Script này chạy 1 cuộc đua tất định + N cuộc đua ngẫu nhiên cho mỗi checkpoint rồi lấy trung bình.

Cách chạy:
    python scripts/evaluate_multicar.py --vehicle f1 --models "models/multicar_f1c/agent_*k.pt" models/multicar_f1c/best_agent.pt
"""

import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import torch

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.multicar_env import MultiCarRacingEnv
from agents.ppo_agent import PPOAgent
from training.train_multicar import run_race


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-car PPO checkpoints over many races")
    parser.add_argument("--models", type=str, nargs="+", default=["models/multicar_v3/agent_*k.pt", "models/multicar_v3/best_agent.pt"],
                        help="Đường dẫn / glob các checkpoint")
    parser.add_argument("--races", type=int, default=5, help="Số cuộc đua ngẫu nhiên (stochastic) mỗi checkpoint")
    parser.add_argument("--num-cars", type=int, default=6)
    parser.add_argument("--vehicle", choices=["gt3", "f1", "sedan"], default="gt3")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--respawn", action="store_true", help="Xe văng hồi sinh tại chỗ (đứng ngoài 3s) thay vì bị loại")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed cho việc lấy mẫu hành động")
    parser.add_argument("--out", type=str, default=None, help="File JSON lưu kết quả")
    return parser.parse_args()


def expand(patterns):
    def step_key(p):
        m = re.search(r"(\d+)k\.pt$", p)
        return (0, int(m.group(1))) if m else (1, 0)
    paths = []
    for pat in patterns:
        hits = glob.glob(pat)
        for p in sorted(hits, key=step_key) if hits else ([pat] if os.path.exists(pat) else []):
            if p not in paths:
                paths.append(p)
    return paths


def summarize(races):
    laps = [r["mean_lap"] for r in races if r["finished"] > 0]
    return {
        "finished": float(np.mean([r["finished"] for r in races])),
        "outs": float(np.mean([r["outs"] for r in races])),
        "stalls": float(np.mean([r["stalls"] for r in races])),
        "contacts": float(np.mean([r["contacts"] for r in races])),
        "respawns": float(np.mean([r["respawns"] for r in races])),
        "race_time": float(np.mean([r["mean_race_time"] for r in races if r["finished"] > 0])) if laps else 0.0,
        "completion": float(np.mean([r["mean_completion"] for r in races])),
        "mean_lap": float(np.mean(laps)) if laps else 0.0,
        "best_lap": float(min(r["best_lap"] for r in races if r["finished"] > 0)) if laps else 0.0,
    }


def main():
    args = parse_args()
    paths = expand(args.models)
    if not paths:
        print(">> Không tìm thấy checkpoint nào.")
        return
    env = MultiCarRacingEnv(num_cars=args.num_cars, vehicle=args.vehicle, max_laps=args.laps, max_steps=2000 * args.laps)
    respawn = "inplace" if args.respawn else False
    agent = PPOAgent(state_dim=env.obs_dim, action_dim=3)
    results = {}
    n = args.num_cars
    try:
        for path in paths:
            agent.load(path)
            torch.manual_seed(args.seed)
            det = summarize([run_race(env, agent, deterministic=True, respawn=respawn)])
            sto = summarize([run_race(env, agent, deterministic=False, respawn=respawn) for _ in range(args.races)])
            results[path] = {"deterministic": det, "stochastic": sto}
            print(f"{os.path.basename(path):18s} | DET: về đích {det['finished']:.0f}/{n} văng {det['outs']:.0f} "
                  f"đi được {det['completion']:5.1f}% lap {det['mean_lap']:6.1f}s | "
                  f"STO x{args.races}: về đích {sto['finished']:.1f}/{n} văng {sto['outs']:.1f} dừng {sto['stalls']:.1f} "
                  f"hồi sinh {sto['respawns']:.1f} đua TB {sto['race_time']:6.1f}s chạm {sto['contacts']:5.1f} đi được {sto['completion']:5.1f}% lap TB {sto['mean_lap']:6.1f}s (best {sto['best_lap']:.1f}s)",
                  flush=True)
    finally:
        env.close()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f">> Đã lưu kết quả vào: {args.out}")


if __name__ == "__main__":
    main()
