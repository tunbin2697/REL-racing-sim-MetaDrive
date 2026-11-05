"""
Script Đánh Giá Toàn Diện (Comprehensive Benchmark & Evaluation)
Thực hiện đánh giá mô hình PPO trên nhiều vòng đua với các seed và địa hình ngẫu nhiên.
Xuất số liệu thống kê chuẩn mực để đưa vào báo cáo và slide bảo vệ đồ án REL.
"""

import os
import sys
import json
import argparse
import numpy as np

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
from agents.optimal_agent import OptimalRacingAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Agent")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="File weights cần đánh giá")
    parser.add_argument("--episodes", type=int, default=10, help="Số vòng đua thử nghiệm")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "optimal"], help="'ppo' hoặc 'optimal' (ghost, nên dùng với --vehicle gt3_anchor)")
    parser.add_argument("--vehicle", type=str, default="gt3", choices=["gt3", "gt3_anchor", "sedan"], help="Loại xe đánh giá ('gt3' hoặc 'sedan')")
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
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 35
    act_dim = int(act_shape[0]) if act_shape is not None else 2

    agent = PPOAgent(state_dim=obs_dim, action_dim=act_dim)
    ghost = None
    if args.agent == "optimal":
        env.reset(seed=env.start_seed)
        ghost = OptimalRacingAgent.from_benchmark(env.get_benchmark_engine(), vehicle_cfg=race_car.cfg, v_norm=40.0)
        print(f">> Ghost: T* = {ghost.theoretical_best_lap_time:.2f}s")
    elif os.path.exists(args.model):
        agent.load(args.model)
        print(f">> Nạp thành công weights từ {args.model} (State Dim: {obs_dim})")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy file {args.model}!")
        return
    
    returns = []
    lengths = []
    completions = []
    mean_speeds = []
    max_speeds = []
    crashes = 0
    successes = 0
    lap_times = []
    crash_locations = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=env.start_seed + ep)
        ep_ret = 0.0
        ep_len = 0
        speeds = []
        
        while True:
            if ghost is not None and ep_len == 0:
                ghost.reset()
            if ghost is not None:
                action, _, _ = ghost.select_action(obs, vehicle=env.car)
            else:
                action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            ep_ret += reward
            ep_len += 1
            speeds.append(step_info.get("speed_km_h", step_info.get("velocity", 0.0) * 3.6))
            
            if terminated or truncated:
                crashed = (
                    step_info.get("crash", False)
                    or step_info.get("crash_sidewalk", False)
                    or step_info.get("crash_vehicle", False)
                    or step_info.get("crash_object", False)
                    or (step_info.get("termination_reason") == "crash_or_track_limit")
                )
                arrived = step_info.get("arrive_dest", False)
                comp = step_info.get("route_completion", 0.0) * 100.0
                
                if crashed:
                    crashes += 1
                    crash_locations.append(float(step_info.get("track_s", 0.0)))
                if arrived:
                    successes += 1
                    lap_times.append(float(step_info.get("lap_time", 0.0)))
                    
                returns.append(ep_ret)
                lengths.append(ep_len)
                completions.append(comp)
                mean_speeds.append(np.mean(speeds) if speeds else 0.0)
                max_speeds.append(np.max(speeds) if speeds else 0.0)
                
                status_str = "ARRIVED" if arrived else ("CRASH_TRACK_LIMIT" if crashed else "OUT_OF_TIME")
                print(f"  - Round {ep+1:02d}: Return: {ep_ret:+6.2f} | Progress: {comp:5.1f}% | Avg Speed: {np.mean(speeds):4.1f} km/h | Status: {status_str}")
                break

    env.close()

    results = {
        "num_episodes": args.episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_completion_rate": float(np.mean(completions)),
        "success_rate": float(successes / args.episodes),
        "crash_rate": float(crashes / args.episodes),
        "mean_speed_kmh": float(np.mean(mean_speeds)),
        "max_speed_kmh": float(np.max(max_speeds)),
        "mean_episode_length": float(np.mean(lengths)),
        "agent": args.agent,
        "model": args.model,
        "lap_times": lap_times,
        "lap_time_mean": float(np.mean(lap_times)) if lap_times else None,
        "lap_time_std": float(np.std(lap_times, ddof=1)) if len(lap_times) > 1 else None,
        "crash_locations": crash_locations,
        "lap_definition": "ket thuc khi dat 98.5% quang duong, xuat phat dung",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ CHUẨN MỰC (FINAL BENCHMARK SUMMARY)")
    print("=" * 70)
    print(f"  - Mean Episodic Return:        {results['mean_return']:+.2f} ± {results['std_return']:.2f}")
    print(f"  - Average Track Completion:    {results['mean_completion_rate']:.1f}%")
    print(f"  - Lap Success Rate:            {results['success_rate']*100:.1f}%")
    print(f"  - Crash Rate:                  {results['crash_rate']*100:.1f}%")
    print(f"  - Average Velocity:            {results['mean_speed_kmh']:.1f} km/h (Max: {results['max_speed_kmh']:.1f} km/h)")
    print(f"  - Mean Steps per Episode:      {results['mean_episode_length']:.1f}")
    print(f">> Dữ liệu chi tiết đã lưu vào: {args.out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
