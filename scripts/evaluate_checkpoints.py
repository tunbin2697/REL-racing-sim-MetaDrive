"""
Script Đánh Giá Hàng Loạt Checkpoint (Deterministic vs Stochastic Policy)
- PPO tối ưu chính sách ngẫu nhiên pi(a|s); hành động trung bình tanh(mu) có thể khác xa khi sigma còn lớn.
- Đánh giá mỗi checkpoint ở cả 2 chế độ để báo cáo trung thực tỉ lệ cán đích và lap time.
- Chế độ Deterministic chỉ cần 2 episode (Standing + Flying) vì cùng seed sẽ lặp lại y hệt.
"""

import os
import re
import sys
import glob
import json
import argparse
import torch

# UTF-8 encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from agents.ppo_agent import PPOAgent
from training.evaluator import RacingEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints (deterministic vs stochastic)")
    parser.add_argument("--models", type=str, nargs="+", default=["models/run_v6/agent_step_*k.pt", "models/run_v6/best_agent.pt"], help="Đường dẫn / glob các checkpoint")
    parser.add_argument("--stochastic-episodes", type=int, default=10, help="Số episode đánh giá chính sách ngẫu nhiên")
    parser.add_argument("--det-episodes", type=int, default=2, help="Số episode đánh giá chính sách tất định (Standing + Flying)")
    parser.add_argument("--vehicle", type=str, default="gt3", choices=["gt3", "sedan"], help="Loại xe đánh giá")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed cho việc lấy mẫu hành động")
    parser.add_argument("--out", type=str, default="logs/run_v6/checkpoint_eval.json", help="File lưu kết quả")
    return parser.parse_args()


def checkpoint_sort_key(path: str):
    """Sắp xếp milestone theo số bước, các file không có số bước (best/last) xếp cuối"""
    match = re.search(r"agent_step_(\d+)k", os.path.basename(path))
    return (0, int(match.group(1))) if match else (1, os.path.basename(path))


def main():
    args = parse_args()
    paths = sorted({p for pattern in args.models for p in glob.glob(pattern)}, key=checkpoint_sort_key)
    if not paths:
        print(">> [CẢNH BÁO] Không tìm thấy checkpoint nào khớp với --models")
        return

    env = MetaRacingEnv(use_render=False, traffic_density=0.0, map_config="silverstone", start_seed=42, vehicle=get_vehicle(args.vehicle), max_laps=1)
    agent = PPOAgent(state_dim=40, action_dim=3, device="cpu")
    evaluator = RacingEvaluator(env=env, agent=agent)

    all_results = {}
    for path in paths:
        try:
            agent.load(path)
        except Exception as e:
            print(f">> [BỎ QUA] {path}: {e}")
            continue
        sigma = torch.exp(agent.ac.actor_logstd.detach()).squeeze(0).tolist()
        torch.manual_seed(args.seed)
        det = evaluator.evaluate(num_episodes=args.det_episodes, deterministic=True)
        sto = evaluator.evaluate(num_episodes=args.stochastic_episodes, deterministic=False)
        all_results[path] = {"sigma": sigma, "deterministic": det, "stochastic": sto}

        # mean_time chỉ là lap time khi có ít nhất 1 vòng cán đích (ngược lại là thời gian trước khi đâm)
        det_lap = f"{det['best_lap_time']:6.1f}s" if det["best_lap_time"] > 0 else "     -"
        sto_lap = f"best {sto['best_lap_time']:6.1f}s mean {sto['mean_time']:6.1f}s" if sto["best_lap_time"] > 0 else "best      - mean      -"
        print(
            f"{os.path.basename(path):24s} sigma={max(sigma):.2f} | "
            f"DET: finish {det['success_rate']*100:3.0f}% comp {det['mean_completion']:5.1f}% lap {det_lap} | "
            f"STO: finish {sto['success_rate']*100:3.0f}% comp {sto['mean_completion']:5.1f}% {sto_lap}",
            flush=True,
        )

    env.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f">> Đã lưu kết quả vào: {args.out}")


if __name__ == "__main__":
    main()
