"""
Script Đánh Giá Toàn Diện (Comprehensive Benchmark & Evaluation)
Thực hiện đánh giá mô hình PPO trên nhiều vòng đua với các seed và địa hình ngẫu nhiên.
Xuất số liệu thống kê chuẩn mực để đưa vào báo cáo và slide bảo vệ đồ án REL.
"""

import os
import sys
import argparse

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
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Agent")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="File weights cần đánh giá")
    parser.add_argument("--episodes", type=int, default=10, help="Số vòng đua thử nghiệm")
    parser.add_argument("--vehicle", type=str, default="gt3", choices=["gt3", "f1", "sedan"], help="Loại xe đánh giá ('gt3' hoặc 'sedan')")
    parser.add_argument("--out", type=str, default="logs/evaluation_results.json", help="File lưu kết quả")
    return parser.parse_args()

def main():
    args = parse_args()
    race_car = get_vehicle(args.vehicle)
    
    print("=" * 70)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ CHỈ SỐ HỌC THUẬT (ACADEMIC BENCHMARK EVALUATION)")
    print("=" * 70)
    print(f"  - Model file: {args.model}")
    print(f"  - Vehicle Spec: {race_car.cfg.name}")
    print(f"  - Số vòng đua thử nghiệm: {args.episodes}")
    print("=" * 70)

    env = MetaRacingEnv(use_render=False, num_scenarios=args.episodes, start_seed=100, vehicle=race_car)
    obs_shape = getattr(env.observation_space, "shape", None)
    act_shape = getattr(env.action_space, "shape", None)
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 40
    act_dim = int(act_shape[0]) if act_shape is not None else 3

    agent = PPOAgent(state_dim=obs_dim, action_dim=act_dim)
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f">> Nạp thành công weights từ {args.model} (State Dim: {obs_dim})")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy file {args.model}! Đánh giá agent ngẫu nhiên (chưa huấn luyện)...")

    evaluator = RacingEvaluator(env=env, agent=agent)
    results = evaluator.evaluate(num_episodes=args.episodes, deterministic=True, verbose=True)
    env.close()

    evaluator.print_summary(results)
    evaluator.save_json(results, args.out)
    print(f">> Dữ liệu chi tiết đã lưu vào: {args.out}\n")

if __name__ == "__main__":
    main()
