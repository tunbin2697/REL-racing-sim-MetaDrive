"""
Pipeline Huấn Luyện Thuật Toán PPO trên Môi Trường MetaDrive 3D Racing
Tự động ghi nhận telemetry, tính toán hàm phần thưởng tự nhiên,
đánh giá hiệu năng và xuất đồ thị học thuật chuẩn rubric môn REL.
"""

import os
import sys
import time
import json
import copy
import argparse
import concurrent.futures
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

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

def save_metrics_json(metrics_data, log_dir="logs"):
    """Ghi log JSON số liệu huấn luyện để hỗ trợ resume (siêu nhẹ, không tốn tài nguyên)"""
    try:
        json_path = os.path.join(log_dir, "training_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
    except Exception as e:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Racing Training in MetaDrive 3D")
    parser.add_argument("--total-steps", type=int, default=200000, help="Tổng số bước mô phỏng tối đa (mặc định 200k steps)")
    parser.add_argument("--rollout-len", type=int, default=1024, help="Độ dài rollout thu thập trước mỗi lần update")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước minibatch PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (3e-4 cho train from scratch, 1e-4 cho fine-tuning)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.15, help="PPO clipping parameter")
    parser.add_argument("--ent-coef", type=float, default=0.015, help="Entropy bonus coefficient (đẩy cao khám phá)")
    parser.add_argument("--eval-freq", type=int, default=1, help="Tần suất đánh giá và xuất log định kỳ (mặc định mỗi 1 vòng update)")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Số episode đánh giá")
    parser.add_argument("--map", type=str, default="O", help="Bản đồ đua ('O' cho oval track hoặc '3' cho block circuit)")
    parser.add_argument("--warmup-model", type=str, default=None, help="File checkpoint warmup (mặc định None để train tự bơi from scratch)")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện (ví dụ models/best_agent.pt)")
    parser.add_argument("--reset-logstd", type=float, default=None, help="Reset độ lệch chuẩn khám phá actor_logstd khi resume (ví dụ -0.8 cho sigma ~ 0.45)")
    parser.add_argument("--steps-to-run", type=int, default=None, help="Số bước muốn chạy thêm từ thời điểm hiện tại khi resume")
    parser.add_argument("--milestone-respawn", action="store_true", help="Bật cơ chế hồi sinh Milestone Respawn để đột phá khúc cua trái 71.4%%")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models", help="Thư mục lưu weights")
    parser.add_argument("--log-dir", type=str, default="logs", help="Thư mục lưu log số liệu")
    parser.add_argument("--tb-logdir", type=str, default="runs", help="Thư mục lưu TensorBoard logs")
    return parser.parse_args()

def evaluate_agent(env, agent, num_episodes=2):
    """Đánh giá agent ở chế độ deterministic (không lấy mẫu ngẫu nhiên)"""
    eval_returns = []
    eval_lengths = []
    eval_completions = []
    eval_speeds = []
    eval_milestones = []
    eval_speeds_rew = []
    eval_brakes_rew = []
    eval_idles_pen = []
    eval_deltas_rew = []
    eval_macros_rew = []
    crash_locations = []
    crash_count = 0
    best_lap_samples = []
    max_comp = -1.0
    
    for ep in range(num_episodes):
        target_seed = env.start_seed + (ep % max(1, getattr(env, "num_scenarios", 1)))
        obs, info = env.reset(seed=target_seed, options={"force_start": True})
        ep_return = 0.0
        ep_len = 0
        ep_milestone = 0.0
        ep_speed_rew = 0.0
        ep_brake_rew = 0.0
        ep_idle_pen = 0.0
        ep_delta_rew = 0.0
        ep_macro_rew = 0.0
        speeds = []
        
        while True:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            ep_milestone += float(info.get("r_milestone", 0.0))
            ep_speed_rew += float(info.get("r_speed", 0.0))
            ep_brake_rew += float(info.get("r_brake", 0.0))
            ep_idle_pen += float(info.get("r_idle", 0.0))
            ep_delta_rew += float(info.get("r_delta", 0.0))
            ep_macro_rew += float(info.get("macro_reward", 0.0))
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
        cur_comp = float(info.get("route_completion", 0.0) * 100.0)
        eval_completions.append(cur_comp)
        eval_speeds.append(np.mean(speeds) if speeds else 0.0)
        eval_milestones.append(ep_milestone)
        eval_speeds_rew.append(ep_speed_rew)
        eval_brakes_rew.append(ep_brake_rew)
        eval_idles_pen.append(ep_idle_pen)
        eval_deltas_rew.append(ep_delta_rew)
        eval_macros_rew.append(ep_macro_rew)
        if cur_comp > max_comp:
            max_comp = cur_comp
            best_lap_samples = list(env.telemetry.current_lap_data)
        
    return {
        "mean_return": float(np.mean(eval_returns)),
        "mean_length": float(np.mean(eval_lengths)),
        "mean_completion": float(np.mean(eval_completions)),
        "mean_speed": float(np.mean(eval_speeds)),
        "mean_milestone": float(np.mean(eval_milestones)),
        "mean_speed_reward": float(np.mean(eval_speeds_rew)),
        "mean_brake_reward": float(np.mean(eval_brakes_rew)),
        "mean_idle_penalty": float(np.mean(eval_idles_pen)),
        "mean_delta_reward": float(np.mean(eval_deltas_rew)),
        "mean_terminal_macro": float(np.mean(eval_macros_rew)),
        "crash_rate": float(crash_count / max(1, num_episodes)),
        "crash_locations": crash_locations,
    }



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
    
    # Khởi tạo TensorBoard SummaryWriter
    tb_run_name = f"ppo_racing_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_dir = os.path.join(args.tb_logdir, tb_run_name)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"  - TensorBoard Dir: {tb_dir}")
    print(f"  - Dashboard Command: tensorboard --logdir {args.tb_logdir}")
    print("=" * 70)
    
    # Khởi tạo môi trường mô phỏng (MetaDrive sử dụng Singleton Engine)
    env = MetaRacingEnv(
        use_render=False,
        traffic_density=0.0,
        map_config=args.map,
        start_seed=args.seed,
        milestone_respawn=args.milestone_respawn,
    )
    
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
    
    metrics_file = os.path.join(args.log_dir, "training_metrics.json")
    global_step = 0
    update_idx = 0
    best_completion = 0.0
    best_fastest_speed = 0.0
    best_fastest_comp = 0.0
    total_finishes = 0

    metrics: Dict[str, Any] = {
        "steps": [],
        "eval_return": [],
        "eval_completion": [],
        "eval_speed": [],
        "crash_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_finishes": 0,
        "resume_marks": [],
    }

    # Xử lý Resume checkpoint hoặc Nạp Warmup
    if args.resume and os.path.exists(args.resume):
        print(f">> RESUME: Nạp trọng số từ checkpoint: {args.resume}")
        agent.load(args.resume)
        
        if args.reset_logstd is not None:
            agent.ac.actor_logstd.data.fill_(float(args.reset_logstd))
            curr_std = float(torch.exp(agent.ac.actor_logstd[0, 0]).item())
            print(f">> EXPLORATION BOOST: Đã reset actor_logstd = {args.reset_logstd} (Action std sigma = {curr_std:.3f})")
        
        # Nạp lại lịch sử metrics cũ nếu có
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                if metrics.get("steps"):
                    global_step = int(metrics["steps"][-1])
                if metrics.get("eval_completion"):
                    best_completion = float(max(metrics["eval_completion"]))
                total_finishes = int(metrics.get("total_finishes", 0))
                print(f">> RESUME: Đã nạp lại lịch sử metrics! Bắt đầu từ Step {global_step} | Finishes cũ: {total_finishes} | Best Completion cũ: {best_completion:.1f}%")
            except Exception as e:
                print(f">> [CẢNH BÁO] Không thể nạp metrics_file cũ: {e}")
                
        # Ghi nhận điểm can thiệp resume vào metrics để vẽ đánh dấu lên đồ thị
        if "resume_marks" not in metrics:
            metrics["resume_marks"] = []
        metrics["resume_marks"].append({
            "step": global_step,
            "source": os.path.basename(args.resume)
        })
                
        # Tính toán target total steps
        if args.steps_to_run is not None:
            target_total_steps = global_step + args.steps_to_run
        elif args.total_steps > global_step:
            target_total_steps = args.total_steps
        else:
            target_total_steps = global_step + 50000
            
    elif args.warmup_model and os.path.exists(args.warmup_model):
        print(f">> Nạp trọng số khởi đầu từ Warmup Model: {args.warmup_model}")
        agent.load(args.warmup_model)
        if args.reset_logstd is not None:
            agent.ac.actor_logstd.data.fill_(float(args.reset_logstd))
            curr_std = float(torch.exp(agent.ac.actor_logstd[0, 0]).item())
            print(f">> EXPLORATION TUNING: Đã thiết lập actor_logstd = {args.reset_logstd} (Action std sigma = {curr_std:.3f})")
        # Khởi tạo lại optimizer để xóa sạch momentum đệm (m, v) từ pha pretrain, bảo vệ policy tránh giật tham số đột ngột
        agent.actor_optimizer = torch.optim.Adam(agent.actor_params, lr=agent.lr_actor, eps=1e-5)
        agent.critic_optimizer = torch.optim.Adam(agent.critic_params, lr=agent.lr_critic, eps=1e-5)
        print(">> Đã khởi tạo lại Adam Optimizer (Fresh Momentum) để fine-tuning ổn định.")
        target_total_steps = args.total_steps
    else:
        print(">> KHỞI TẠO TỰ BƠI (FROM SCRATCH): Huấn luyện từ đầu với trọng số ngẫu nhiên Orthogonal Init.")
        if args.reset_logstd is not None:
            agent.ac.actor_logstd.data.fill_(float(args.reset_logstd))
            curr_std = float(torch.exp(agent.ac.actor_logstd[0, 0]).item())
            print(f">> EXPLORATION TUNING: Đã thiết lập actor_logstd = {args.reset_logstd} (Action std sigma = {curr_std:.3f})")
        target_total_steps = args.total_steps

    # Bộ đệm Rollout Storage
    states = np.zeros((args.rollout_len, obs_dim), dtype=np.float32)
    actions = np.zeros((args.rollout_len, act_dim), dtype=np.float32)
    log_probs = np.zeros(args.rollout_len, dtype=np.float32)
    rewards = np.zeros(args.rollout_len, dtype=np.float32)
    values = np.zeros(args.rollout_len, dtype=np.float32)
    dones = np.zeros(args.rollout_len, dtype=np.float32)

    print(f">> Đánh giá ban đầu (Step {global_step})...")
    eval_res = evaluate_agent(env, agent, num_episodes=args.eval_episodes)
    if not args.resume or not metrics.get("steps") or metrics["steps"][-1] != global_step:
        metrics["steps"].append(global_step)
        metrics["eval_return"].append(eval_res["mean_return"])
        metrics["eval_completion"].append(eval_res["mean_completion"])
        metrics["eval_speed"].append(eval_res["mean_speed"])
        metrics["crash_rate"].append(eval_res["crash_rate"])
        metrics["policy_loss"].append(0.0)
        metrics["value_loss"].append(0.0)
        metrics["entropy"].append(0.0)
        
    best_completion = max(best_completion, eval_res["mean_completion"])
    print(
        f"[Step {global_step:06d}/{target_total_steps}] "
        f"Eval Return: {eval_res['mean_return']:+.2f} | "
        f"Track Completion: {eval_res['mean_completion']:.1f}% | "
        f"Speed: {eval_res['mean_speed']:.1f} km/h | "
        f"Crash Rate: {eval_res['crash_rate']*100:.0f}%"
    )
    # Lưu metrics JSON
    save_metrics_json(metrics, log_dir=args.log_dir)
    
    # Log ban đầu vào TensorBoard
    writer.add_scalar("Performance/Eval_Return", eval_res["mean_return"], global_step)
    writer.add_scalar("Performance/Track_Completion_Pct", eval_res["mean_completion"], global_step)
    writer.add_scalar("Performance/Mean_Speed_kmh", eval_res["mean_speed"], global_step)
    writer.add_scalar("Performance/Crash_Rate", eval_res["crash_rate"], global_step)
    writer.add_scalar("Performance/Episode_Length", eval_res["mean_length"], global_step)
    writer.add_scalar("Performance/Total_Finishes", total_finishes, global_step)
    writer.add_scalars("Reward_Breakdown/Streams", {
        "Milestones": eval_res["mean_milestone"],
        "Speed": eval_res["mean_speed_reward"],
        "Brake": eval_res["mean_brake_reward"],
        "Idle": eval_res["mean_idle_penalty"],
        "Delta_SIL": eval_res["mean_delta_reward"],
        "Terminal": eval_res["mean_terminal_macro"],
    }, global_step)
    writer.flush()
    
    start_time = time.time()
    obs, info = env.reset()
    
    while global_step < target_total_steps:
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
            
            # Ghi nhận nếu hoàn thành trọn vẹn vòng đua
            if step_info.get("termination_reason") == "lap_completed" or step_info.get("route_completion", 0.0) >= 0.98:
                total_finishes += 1
                lap_time = step_info.get("lap_time", 0.0)
                speed_fin = step_info.get("speed_km_h", 0.0)
                print(f"  🏁 [FINISH #{total_finishes}] Step {global_step} | Lap Time: {lap_time:.2f}s | Speed: {speed_fin:.1f} km/h")
            
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
        if update_idx % args.eval_freq == 0 or global_step >= target_total_steps:
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
            metrics["total_finishes"] = total_finishes
            
            crash_info_str = f" | Crashes at: {eval_res['crash_locations']}" if eval_res['crash_locations'] else " | Clean Lap"
            print(
                f"[Step {global_step:06d}/{target_total_steps} | Finishes: {total_finishes}] "
                f"FPS: {fps} | "
                f"Eval Return: {eval_res['mean_return']:+.2f} | "
                f"Completion: {eval_res['mean_completion']:.1f}% | "
                f"Speed: {eval_res['mean_speed']:.1f} km/h | "
                f"Loss(P/V): {train_info['policy_loss']:.3f}/{train_info['value_loss']:.3f} | "
                f"ExpVar: {train_info.get('explained_var', 0.0):.2f}"
                f"{crash_info_str}"
            )
            
            # 1. Lưu checkpoint tốt nhất dựa trên tỷ lệ hoàn thành chặng đua (Best Agent)
            if eval_res["mean_completion"] >= best_completion:
                best_completion = eval_res["mean_completion"]
                best_model_path = os.path.join(args.save_dir, "best_agent.pt")
                agent.save(best_model_path)
                print(f"  [RECORD - BEST] Checkpoint ky luc moi da luu: {best_model_path} (Completion: {best_completion:.1f}%)")
                
            # 2. Lưu checkpoint có Completion cao VÀ Vận tốc cao nhất (Fastest Agent)
            cur_comp = eval_res["mean_completion"]
            cur_speed = eval_res["mean_speed"]
            if cur_comp >= 50.0:
                if (cur_comp >= best_fastest_comp - 3.0 and cur_speed > best_fastest_speed) or (cur_comp > best_fastest_comp and cur_speed >= best_fastest_speed * 0.95):
                    best_fastest_speed = max(best_fastest_speed, cur_speed)
                    best_fastest_comp = max(best_fastest_comp, cur_comp)
                    fastest_model_path = os.path.join(args.save_dir, "fastest_agent.pt")
                    agent.save(fastest_model_path)
                    print(f"  [RECORD - FASTEST] Checkpoint toc do cao nhat da luu: {fastest_model_path} (Comp: {cur_comp:.1f}%, Speed: {cur_speed:.1f} km/h)")

            # 3. Lưu checkpoint mới nhất
            latest_model_path = os.path.join(args.save_dir, "latest_agent.pt")
            agent.save(latest_model_path)
            
            # 4. Lưu metrics JSON định kỳ
            save_metrics_json(metrics, log_dir=args.log_dir)
            
            # 5. Ghi log liên tục lên TensorBoard Dashboard
            writer.add_scalar("Performance/Eval_Return", eval_res["mean_return"], global_step)
            writer.add_scalar("Performance/Track_Completion_Pct", eval_res["mean_completion"], global_step)
            writer.add_scalar("Performance/Mean_Speed_kmh", eval_res["mean_speed"], global_step)
            writer.add_scalar("Performance/Crash_Rate", eval_res["crash_rate"], global_step)
            writer.add_scalar("Performance/Episode_Length", eval_res["mean_length"], global_step)
            writer.add_scalar("Performance/Total_Finishes", total_finishes, global_step)
            
            # Ghi nhận phân bổ các dòng thưởng (Reward Breakdown) trực tiếp trên Web Dashboard
            writer.add_scalars("Reward_Breakdown/Streams", {
                "Milestones": eval_res["mean_milestone"],
                "Speed": eval_res["mean_speed_reward"],
                "Brake": eval_res["mean_brake_reward"],
                "Idle": eval_res["mean_idle_penalty"],
                "Delta_SIL": eval_res["mean_delta_reward"],
                "Terminal": eval_res["mean_terminal_macro"],
            }, global_step)
            writer.add_scalar("Reward_Breakdown/Total_Return", eval_res["mean_return"], global_step)
            
            # Ghi nhận các hàm Loss
            writer.add_scalar("Loss/Policy_Loss", train_info["policy_loss"], global_step)
            writer.add_scalar("Loss/Value_Loss", train_info["value_loss"], global_step)
            writer.add_scalar("Loss/Entropy", train_info["entropy"], global_step)
            writer.add_scalar("Loss/Explained_Variance", train_info.get("explained_var", 0.0), global_step)
            writer.add_scalar("Loss/Approx_KL", train_info.get("approx_kl", 0.0), global_step)
            writer.add_scalar("Diagnostics/FPS", fps, global_step)
            writer.flush()
            
            # Reset lại env để tiếp tục thu thập rollout cho vòng lặp kế tiếp
            obs, info = env.reset()

    writer.close()
    env.close()
    print("=" * 70)
    print(f">> HOÀN THÀNH QUÁ TRÌNH HUẤN LUYỆN ({global_step}/{target_total_steps} steps, {total_finishes} finishes)!")
    print(f">> Best Model (Kỷ lục Completion): {os.path.join(args.save_dir, 'best_agent.pt')}")
    print(f">> Fastest Model (Tốc độ cao nhất): {os.path.join(args.save_dir, 'fastest_agent.pt')}")
    print(f">> Latest Model (Checkpoint cuối): {os.path.join(args.save_dir, 'latest_agent.pt')}")
    print(f">> Biểu đồ học thuật: plots/learning_curves.png")
    print(f">> Chẩn đoán Telemetry: plots/telemetry_diagnostic.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
