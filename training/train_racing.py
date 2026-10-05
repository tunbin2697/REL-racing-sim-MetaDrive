"""
Pipeline Huấn Luyện Thuật Toán PPO trên Môi Trường MetaDrive 3D Racing
Tự động ghi nhận telemetry, tính toán hàm phần thưởng tự nhiên,
đánh giá hiệu năng và xuất đồ thị học thuật chuẩn rubric môn REL.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

# Thêm thư mục gốc vào path để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from agents.ppo_agent import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Racing Training in MetaDrive 3D")
    parser.add_argument("--total-steps", type=int, default=50000, help="Tổng số bước mô phỏng")
    parser.add_argument("--rollout-len", type=int, default=1024, help="Độ dài rollout thu thập trước mỗi lần update")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước minibatch PPO")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (fine-tuning)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.15, help="PPO clipping parameter")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy bonus coefficient")
    parser.add_argument("--eval-freq", type=int, default=2, help="Tần suất đánh giá (theo số vòng update)")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Số episode đánh giá")
    parser.add_argument("--map", type=str, default="O", help="Bản đồ đua ('O' cho oval track hoặc '3' cho block circuit)")
    parser.add_argument("--warmup-model", type=str, default="models/warmup_racing_agent.pt", help="File checkpoint warmup")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models", help="Thư mục lưu weights")
    parser.add_argument("--log-dir", type=str, default="logs", help="Thư mục lưu log số liệu")
    return parser.parse_args()

def evaluate_agent(env, agent, num_episodes=2):
    """Đánh giá agent ở chế độ deterministic (không lấy mẫu ngẫu nhiên)"""
    eval_returns = []
    eval_lengths = []
    eval_completions = []
    eval_speeds = []
    crash_locations = []
    crash_count = 0
    
    for ep in range(num_episodes):
        target_seed = env.start_seed + (ep % max(1, getattr(env, "num_scenarios", 1)))
        obs, info = env.reset(seed=target_seed)
        ep_return = 0.0
        ep_len = 0
        speeds = []
        
        while True:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            speeds.append(info.get("speed_km_h", info.get("speed_kmh", 0.0)))
            
            if terminated or truncated:
                is_crash = info.get("crash", False) or info.get("crash_sidewalk", False) or (info.get("termination_reason") == "crash_or_track_limit")
                if is_crash:
                    crash_count += 1
                    dist = info.get("crash_distance", info.get("route_completion", 0.0) * 2117.0)
                    crash_locations.append(f"{dist:.0f}m")
                break
                
        eval_returns.append(ep_return)
        eval_lengths.append(ep_len)
        eval_completions.append(info.get("route_completion", 0.0) * 100.0)
        eval_speeds.append(np.mean(speeds) if speeds else 0.0)
        
    return {
        "mean_return": float(np.mean(eval_returns)),
        "mean_length": float(np.mean(eval_lengths)),
        "mean_completion": float(np.mean(eval_completions)),
        "mean_speed": float(np.mean(eval_speeds)),
        "crash_rate": float(crash_count / max(1, num_episodes)),
        "crash_locations": crash_locations,
    }

def plot_training_curves(metrics, plot_path):
    """Vẽ và lưu đồ thị các chỉ số huấn luyện học thuật"""
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    steps = metrics["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Continuous 3D Racing Agent Performance", fontsize=16, fontweight="bold")
    
    # 1. Episode Return (Learning Curve)
    ax1 = axes[0, 0]
    ax1.plot(steps, metrics["eval_return"], color="#1f77b4", lw=2, marker="o", label="Eval Return")
    ax1.set_title("Mean Episodic Return (Reward Convergence)")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Total Return")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # 2. Route Completion Rate (%)
    ax2 = axes[0, 1]
    ax2.plot(steps, metrics["eval_completion"], color="#2ca02c", lw=2, marker="s", label="Track Completion %")
    ax2.set_title("Track Completion Rate (%)")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Completion (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    # 3. Policy & Value Loss
    ax3 = axes[1, 0]
    ax3.plot(steps, metrics["policy_loss"], color="#d62728", lw=1.8, label="Policy Loss")
    ax3.plot(steps, metrics["value_loss"], color="#ff7f0e", lw=1.8, label="Value Loss")
    ax3.set_title("Actor & Critic Loss Convergence")
    ax3.set_xlabel("Environment Steps")
    ax3.set_ylabel("Loss Value")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()
    
    # 4. Mean Speed & Crash Rate
    ax4 = axes[1, 1]
    color_speed = "#9467bd"
    color_crash = "#8c564b"
    ax4.plot(steps, metrics["eval_speed"], color=color_speed, lw=2, label="Mean Speed (km/h)")
    ax4.set_title("Speed Progression & Safety")
    ax4.set_xlabel("Environment Steps")
    ax4.set_ylabel("Speed (km/h)", color=color_speed)
    ax4.tick_params(axis="y", labelcolor=color_speed)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(steps, metrics["crash_rate"], color=color_crash, lw=2, linestyle="--", label="Crash Rate")
    ax4_twin.set_ylabel("Crash Rate (0-1)", color=color_crash)
    ax4_twin.tick_params(axis="y", labelcolor=color_crash)
    ax4_twin.set_ylim(-0.05, 1.05)
    ax4.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> [PLOT] Đã xuất biểu đồ học thuật tại: {plot_path}")

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("=" * 70)
    print(">> BAT DAU HUAN LUYEN PPO CONTINUOUS 3D RACING AGENT (METADRIVE)")
    print("=" * 70)
    print(f"  - Device: {'CUDA (GPU)' if PPOAgent().device.type == 'cuda' else 'CPU'}")
    print(f"  - Total Steps: {args.total_steps}")
    print(f"  - Rollout Length: {args.rollout_len}")
    print(f"  - Minibatch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print("=" * 70)
    
    # Khởi tạo môi trường mô phỏng (MetaDrive sử dụng Singleton Engine)
    env = MetaRacingEnv(use_render=False, traffic_density=0.0, map_config=args.map, start_seed=args.seed)
    
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 259
    act_dim = int(act_shape[0]) if act_shape is not None else 2
    
    agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        minibatch_size=args.batch_size,
    )
    
    if args.warmup_model and os.path.exists(args.warmup_model):
        print(f">> Nạp trọng số khởi đầu từ Warmup Model: {args.warmup_model}")
        agent.load(args.warmup_model)
    
    # Bộ đệm Rollout Storage
    states = np.zeros((args.rollout_len, obs_dim), dtype=np.float32)
    actions = np.zeros((args.rollout_len, act_dim), dtype=np.float32)
    log_probs = np.zeros(args.rollout_len, dtype=np.float32)
    rewards = np.zeros(args.rollout_len, dtype=np.float32)
    values = np.zeros(args.rollout_len, dtype=np.float32)
    dones = np.zeros(args.rollout_len, dtype=np.float32)
    
    metrics = {
        "steps": [],
        "eval_return": [],
        "eval_completion": [],
        "eval_speed": [],
        "crash_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }
    
    print(">> Đánh giá ban đầu (Step 0 / Warmup baseline)...")
    eval_res = evaluate_agent(env, agent, num_episodes=args.eval_episodes)
    metrics["steps"].append(0)
    metrics["eval_return"].append(eval_res["mean_return"])
    metrics["eval_completion"].append(eval_res["mean_completion"])
    metrics["eval_speed"].append(eval_res["mean_speed"])
    metrics["crash_rate"].append(eval_res["crash_rate"])
    metrics["policy_loss"].append(0.0)
    metrics["value_loss"].append(0.0)
    metrics["entropy"].append(0.0)
    print(
        f"[Step 000000/{args.total_steps}] "
        f"Eval Return: {eval_res['mean_return']:+.2f} | "
        f"Track Completion: {eval_res['mean_completion']:.1f}% | "
        f"Speed: {eval_res['mean_speed']:.1f} km/h | "
        f"Crash Rate: {eval_res['crash_rate']*100:.0f}%"
    )
    plot_training_curves(metrics, os.path.join("plots", "learning_curves.png"))
    
    global_step = 0
    update_idx = 0
    best_completion = eval_res["mean_completion"]
    start_time = time.time()
    
    obs, info = env.reset()
    
    while global_step < args.total_steps:
        # 1. Thu thập Rollout
        for step in range(args.rollout_len):
            global_step += 1
            action, log_prob, value = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            states[step] = obs
            actions[step] = action
            log_probs[step] = log_prob
            rewards[step] = reward
            values[step] = value
            is_done = bool(terminated) or bool(truncated)
            dones[step] = 1.0 if is_done else 0.0
            
            obs = next_obs
            if is_done:
                obs, info = env.reset()
                
        # 2. Ước tính giá trị trạng thái kế tiếp để tính GAE
        with_deterministic = True
        _, _, next_value = agent.select_action(obs, deterministic=True)
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        # 3. Cập nhật mạng Actor-Critic (Decoupled optimization)
        train_info = agent.update(states, actions, log_probs, advantages, returns, values)
        update_idx += 1
        
        # 4. Đánh giá định kỳ
        if update_idx % args.eval_freq == 0 or global_step >= args.total_steps:
            eval_res = evaluate_agent(env, agent, num_episodes=args.eval_episodes)
            elapsed = time.time() - start_time
            fps = int(global_step / max(1e-5, elapsed))
            
            metrics["steps"].append(global_step)
            metrics["eval_return"].append(eval_res["mean_return"])
            metrics["eval_completion"].append(eval_res["mean_completion"])
            metrics["eval_speed"].append(eval_res["mean_speed"])
            metrics["crash_rate"].append(eval_res["crash_rate"])
            metrics["policy_loss"].append(train_info["policy_loss"])
            metrics["value_loss"].append(train_info["value_loss"])
            metrics["entropy"].append(train_info["entropy"])
            
            crash_info_str = f" | Crashes at: {eval_res['crash_locations']}" if eval_res['crash_locations'] else " | Clean Lap"
            print(
                f"[Step {global_step:06d}/{args.total_steps}] "
                f"FPS: {fps} | "
                f"Eval Return: {eval_res['mean_return']:+.2f} | "
                f"Completion: {eval_res['mean_completion']:.1f}% | "
                f"Speed: {eval_res['mean_speed']:.1f} km/h | "
                f"Loss(P/V): {train_info['policy_loss']:.3f}/{train_info['value_loss']:.3f} | "
                f"ExpVar: {train_info.get('explained_var', 0.0):.2f}"
                f"{crash_info_str}"
            )
            
            # Lưu checkpoint tốt nhất dựa trên tỷ lệ hoàn thành chặng đua và điểm thưởng
            if eval_res["mean_completion"] >= best_completion:
                best_completion = eval_res["mean_completion"]
                best_model_path = os.path.join(args.save_dir, "best_agent.pt")
                agent.save(best_model_path)
                print(f"  [RECORD] Checkpoint ky luc moi da luu: {best_model_path} (Completion: {best_completion:.1f}%)")
                
            # Lưu checkpoint mới nhất
            latest_model_path = os.path.join(args.save_dir, "latest_agent.pt")
            agent.save(latest_model_path)
            
            # Lưu log và vẽ đồ thị cập nhật
            with open(os.path.join(args.log_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            plot_training_curves(metrics, os.path.join("plots", "learning_curves.png"))
            
            # Reset lại env để tiếp tục thu thập rollout cho vòng lặp kế tiếp
            obs, info = env.reset()

    env.close()
    print("=" * 70)
    print(">> HOAN THANH QUA TRINH HUAN LUYEN THANH CONG!")
    print(f">> Best Model: {os.path.join(args.save_dir, 'best_agent.pt')}")
    print(f">> Biểu đồ học thuật: plots/learning_curves.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
