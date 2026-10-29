"""
Pipeline Huấn Luyện Thuật Toán PPO trên Môi Trường MetaDrive 3D Racing
Tuân thủ chuẩn Clean Architecture & OOP:
- Thu thập Rollout đa chiều, tính toán GAE Advantage.
- Cập nhật mạng Actor-Critic Squashed Gaussian (Tanh Normal).
- Tích hợp bộ đánh giá dùng chung RacingEvaluator (DRY - Single Source of Truth).
- Quản lý siêu tham số linh hoạt với các cờ CLI (LR Decay, Entropy Decay).
- Xuất telemetry và metrics chuẩn rubric môn REL.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Optional
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
from game_env.vehicle import get_vehicle
from agents.ppo_agent import PPOAgent
from training.evaluator import RacingEvaluator


def save_metrics_json(metrics_data: Dict[str, Any], log_dir: str = "logs"):
    """Ghi log JSON số liệu huấn luyện để hỗ trợ resume (siêu nhẹ, không tốn tài nguyên)"""
    try:
        json_path = os.path.join(log_dir, "training_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Racing Training in MetaDrive 3D")
    parser.add_argument("--total-steps", type=int, default=1000000, help="Tổng số bước mô phỏng tối đa (mặc định 1M steps)")
    parser.add_argument("--rollout-len", type=int, default=2048, help="Độ dài rollout thu thập trước mỗi lần update (mặc định 2048)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước minibatch PPO")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate ban đầu")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.90, help="GAE lambda (0.90 giảm phương sai)")
    parser.add_argument("--clip-coef", type=float, default=0.15, help="PPO clipping parameter (0.15)")
    parser.add_argument("--ent-coef", type=float, default=0.010, help="Hệ số thưởng Entropy ban đầu (0.010)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá và xuất log định kỳ (mỗi 5 update = 10,240 steps)")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Số episode đánh giá (mặc định 3 episodes)")
    parser.add_argument("--map", type=str, default="silverstone", help="Bản đồ đua ('silverstone', 'silverstone_half', hoặc 'silverstone_full')")
    parser.add_argument("--time-penalty", type=float, default=0.01, help="Linear time pressure penalty per step (mặc định 0.01)")
    parser.add_argument("--finish-bonus", type=str, default="exp", choices=["exp", "cubic"], help="Loại hàm thưởng cán đích: 'exp' (hàm mũ dốc) hoặc 'cubic' (lũy thừa bậc 3)")
    parser.add_argument("--no-logstd-projection", action="store_true", default=False, help="Tắt phép chiếu logstd.clamp_() trong PPO (dành cho ablation study)")
    parser.add_argument("--warmup-model", type=str, default=None, help="File checkpoint warmup")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện")
    parser.add_argument("--reset-logstd", type=float, default=None, help="Reset độ lệch chuẩn khám phá actor_logstd khi resume")
    parser.add_argument("--steps-to-run", type=int, default=None, help="Số bước muốn chạy thêm khi resume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--vehicle", type=str, default="gt3", choices=["gt3", "gt3_real", "sedan"], help="Loại xe đua ('gt3', 'gt3_real' cho map full, hoặc 'sedan' đối chứng)")
    parser.add_argument("--max-laps", type=int, default=2, help="Số vòng đua tối đa mỗi episode (mặc định 2 laps: Lap 1 Standing + Lap 2 Flying)")
    
    # Cờ điều khiển lịch trình suy giảm tham số (Schedules) - Mặc định TẮT (giữ nguyên LR và Entropy cố định)
    parser.add_argument("--lr-decay", action="store_true", default=False, help="Bật giảm dần Learning Rate tuyến tính từ --lr về --min-lr (mặc định: TẮT)")
    parser.add_argument("--min-lr", type=float, default=5e-5, help="Learning rate sàn khi dùng --lr-decay (mặc định 5e-5)")
    parser.add_argument("--ent-decay", action="store_true", default=False, help="Bật giảm dần Entropy Coefficient tuyến tính từ --ent-coef về --min-ent-coef (mặc định: TẮT)")
    parser.add_argument("--min-ent-coef", type=float, default=0.001, help="Hệ số Entropy sàn khi dùng --ent-decay (mặc định 0.001)")

    parser.add_argument("--milestone-freq", type=int, default=500000, help="Khoảng bước lưu milestone snapshot (mặc định 500k steps)")
    parser.add_argument("--save-dir", type=str, default="models", help="Thư mục lưu weights")
    parser.add_argument("--log-dir", type=str, default="logs", help="Thư mục lưu log số liệu")
    parser.add_argument("--tb-logdir", type=str, default="runs", help="Thư mục lưu TensorBoard logs")
    return parser.parse_args()


def log_tensorboard_metrics(writer: SummaryWriter, eval_res: Dict[str, Any], global_step: int, train_info: Optional[Dict[str, Any]] = None, agent: Optional[PPOAgent] = None, total_finishes: int = 0, fps: int = 0):
    """Ghi nhận toàn bộ chỉ số đánh giá và huấn luyện vào TensorBoard"""
    writer.add_scalar("Performance/Eval_Return", eval_res["mean_return"], global_step)
    writer.add_scalar("Performance/Track_Completion_Pct", eval_res["mean_completion"], global_step)
    writer.add_scalar("Performance/Mean_Speed_kmh", eval_res["mean_speed"], global_step)
    writer.add_scalar("Performance/Crash_Rate", eval_res["crash_rate"], global_step)
    writer.add_scalar("Performance/Episode_Length", eval_res["mean_length"], global_step)
    writer.add_scalar("Performance/Total_Finishes", total_finishes, global_step)
    
    if eval_res["mean_lap1_time"] > 0:
        writer.add_scalar("Performance/Lap1_Standing_Time", eval_res["mean_lap1_time"], global_step)
    if eval_res["mean_lap2_time"] > 0:
        writer.add_scalar("Performance/Lap2_Flying_Time", eval_res["mean_lap2_time"], global_step)
    if eval_res["best_lap_time"] > 0:
        writer.add_scalar("Performance/Best_Lap_Time", eval_res["best_lap_time"], global_step)

    # Chỉ số Benchmark Học thuật
    writer.add_scalar("Benchmark/Racing_Line_MAE_m", eval_res["racing_line_mae_m"], global_step)
    writer.add_scalar("Benchmark/Apex_Accuracy_Pct", eval_res["apex_accuracy_pct"], global_step)
    writer.add_scalar("Benchmark/Corner_Delta_V_kmh", eval_res["corner_delta_v_kmh"], global_step)
    writer.add_scalar("Benchmark/Friction_Utilization_Pct", eval_res["friction_utilization_pct"], global_step)
    writer.add_scalar("Benchmark/Lap_Efficiency_Pct", eval_res["lap_efficiency_pct"], global_step)

    writer.add_scalars("Reward_Breakdown/Streams", {
        "Progress": eval_res["mean_progress_reward"],
        "Steer_Smooth": eval_res["mean_steer_smooth_reward"],
        "Idle": eval_res["mean_idle_penalty"],
        "Macro": eval_res["mean_macro_reward"],
    }, global_step)

    if agent is not None:
        writer.add_scalar("Hyperparameters/Learning_Rate", agent.get_lr(), global_step)
        writer.add_scalar("Hyperparameters/Entropy_Coef", agent.get_ent_coef(), global_step)
        with torch.no_grad():
            sigmas = torch.exp(agent.ac.actor_logstd).detach().cpu().numpy()[0]
            writer.add_scalar("Exploration/Sigma_Steer", float(sigmas[0]), global_step)
            writer.add_scalar("Exploration/Sigma_Throttle", float(sigmas[1]), global_step)
            if len(sigmas) >= 3:
                writer.add_scalar("Exploration/Sigma_Brake", float(sigmas[2]), global_step)

    if train_info is not None:
        writer.add_scalar("Loss/Policy_Loss", train_info["policy_loss"], global_step)
        writer.add_scalar("Loss/Value_Loss", train_info["value_loss"], global_step)
        writer.add_scalar("Loss/Entropy", train_info["entropy"], global_step)
        writer.add_scalar("Loss/Explained_Variance", train_info.get("explained_var", 0.0), global_step)
        writer.add_scalar("Loss/Approx_KL", train_info.get("approx_kl", 0.0), global_step)
        writer.add_scalar("Diagnostics/FPS", fps, global_step)

    writer.flush()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 70)
    print(">> BẮT ĐẦU HUẤN LUYỆN PPO CONTINUOUS 3D RACING AGENT (METADRIVE)")
    print("=" * 70)
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"  - Device: {device_name}")
    print(f"  - Total Steps: {args.total_steps}")
    print(f"  - Rollout Length: {args.rollout_len}")
    print(f"  - Minibatch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr} (Decay: {args.lr_decay}, Min: {args.min_lr})")
    print(f"  - Entropy Coef: {args.ent_coef} (Decay: {args.ent_decay}, Min: {args.min_ent_coef})")
    print(f"  - GAE Lambda: {args.gae_lambda} | Clip: {args.clip_coef} | Gamma: {args.gamma}")

    # Khởi tạo TensorBoard
    tb_run_name = f"ppo_racing_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_dir = os.path.join(args.tb_logdir, tb_run_name)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"  - TensorBoard Dir: {tb_dir}")
    print(f"  - Dashboard Command: tensorboard --logdir {args.tb_logdir}")
    print("=" * 70)

    # Khởi tạo mô hình xe đua theo OOP
    race_car = get_vehicle(args.vehicle)
    print(f"  - Vehicle Spec: {race_car.cfg.name} (Brake: {race_car.cfg.max_brake_force}N, Grip: {race_car.cfg.wheel_friction}, Steering Lock: {race_car.cfg.max_steering_deg}°)")
    print("=" * 70)

    # Khởi tạo môi trường mô phỏng
    map_config_val = "silverstone" if args.map in ["silverstone", "silverstone_half"] else args.map
    env = MetaRacingEnv(
        use_render=False,
        traffic_density=0.0,
        map_config=map_config_val,
        start_seed=args.seed,
        vehicle=race_car,
        max_laps=args.max_laps,
        time_penalty=args.time_penalty,
        finish_bonus=args.finish_bonus,
    )

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 35
    act_dim = int(act_shape[0]) if act_shape is not None else 3

    agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        minibatch_size=args.batch_size,
        enable_logstd_projection=not args.no_logstd_projection,
        logstd_max=env.preset.logstd_max,
    )

    # Khởi tạo Bộ đánh giá OOP dùng chung (DRY)
    evaluator = RacingEvaluator(env=env, agent=agent)

    metrics_file = os.path.join(args.log_dir, "training_metrics.json")
    global_step = 0
    update_idx = 0
    best_eval_completion = 0.0
    best_eval_lap_time = float("inf")
    best_eval_flying_time = float("inf")
    best_eval_speed = 0.0
    total_finishes = 0

    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_commit = "unknown"

    metrics: Dict[str, Any] = {
        "steps": [],
        "eval_return": [],
        "eval_completion": [],
        "eval_speed": [],
        "eval_lap1_time": [],
        "eval_lap2_time": [],
        "eval_best_lap_time": [],
        "crash_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "sigma_steer": [],
        "sigma_throttle": [],
        "sigma_brake": [],
        "racing_line_mae_m": [],
        "apex_accuracy_pct": [],
        "corner_delta_v_kmh": [],
        "friction_utilization_pct": [],
        "lap_efficiency_pct": [],
        "total_finishes": 0,
        "resume_marks": [],
        "git_commit": git_commit,
        "args": vars(args),
    }

    # Xử lý Resume checkpoint hoặc Nạp Warmup
    if args.resume and os.path.exists(args.resume):
        print(f">> RESUME: Nạp trọng số từ checkpoint: {args.resume}")
        agent.load(args.resume)
        agent.set_lr(args.lr)

        if args.reset_logstd is not None:
            agent.ac.actor_logstd.data.fill_(float(args.reset_logstd))
            curr_std = float(torch.exp(agent.ac.actor_logstd[0, 0]).item())
            print(f">> EXPLORATION BOOST: Đã reset actor_logstd = {args.reset_logstd} (Action std sigma = {curr_std:.3f})")

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                if metrics.get("steps"):
                    global_step = int(metrics["steps"][-1])
                if metrics.get("eval_completion"):
                    best_eval_completion = float(max(metrics["eval_completion"]))
                # Khôi phục kỷ lục lap time để eval đầu tiên sau resume không ghi đè best_agent.pt bằng vòng chậm hơn
                # Chỉ lấy các lần eval sạch (crash 0%, completion >= 98%), đúng tiêu chí lưu best. Các list có thể lệch
                # 1 phần tử ở đầu (eval ban đầu không ghi lap time) nên căn theo đuôi.
                prev_laps = metrics.get("eval_best_lap_time", [])
                n_laps = len(prev_laps)
                prev_crash = metrics.get("crash_rate", [])[-n_laps:] if n_laps else []
                prev_comp = metrics.get("eval_completion", [])[-n_laps:] if n_laps else []
                clean_laps = [
                    t for t, c, p in zip(prev_laps, prev_crash, prev_comp)
                    if t and t > 0 and c == 0.0 and p >= 98.0
                ]
                if clean_laps:
                    best_eval_lap_time = float(min(clean_laps))
                total_finishes = int(metrics.get("total_finishes", 0))
                print(f">> RESUME: Đã nạp lại lịch sử metrics! Bắt đầu từ Step {global_step} | Finishes cũ: {total_finishes} | Best Completion cũ: {best_eval_completion:.1f}%")
            except Exception as e:
                print(f">> [CẢNH BÁO] Không thể nạp metrics_file cũ: {e}")

        if "resume_marks" not in metrics:
            metrics["resume_marks"] = []
        metrics["resume_marks"].append({
            "step": global_step,
            "source": os.path.basename(args.resume),
            "git_commit": git_commit,
            "args": vars(args),
        })

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

    if args.steps_to_run is not None:
        target_total_steps = global_step + args.steps_to_run

    # Bộ đệm Rollout Storage
    states = np.zeros((args.rollout_len, obs_dim), dtype=np.float32)
    actions = np.zeros((args.rollout_len, act_dim), dtype=np.float32)
    raw_us = np.zeros((args.rollout_len, act_dim), dtype=np.float32)  # u trước tanh, dùng tính lại log_prob khi update (F13)
    log_probs = np.zeros(args.rollout_len, dtype=np.float32)
    rewards = np.zeros(args.rollout_len, dtype=np.float32)
    values = np.zeros(args.rollout_len, dtype=np.float32)
    dones = np.zeros(args.rollout_len, dtype=np.float32)

    # Đánh giá ban đầu
    print(f">> Đánh giá ban đầu (Step {global_step})...")
    eval_res = evaluator.evaluate(num_episodes=args.eval_episodes, deterministic=True)
    if not args.resume or not metrics.get("steps") or metrics["steps"][-1] != global_step:
        metrics["steps"].append(global_step)
        metrics["eval_return"].append(eval_res["mean_return"])
        metrics["eval_completion"].append(eval_res["mean_completion"])
        metrics["eval_speed"].append(eval_res["mean_speed"])
        metrics["crash_rate"].append(eval_res["crash_rate"])
        metrics["policy_loss"].append(0.0)
        metrics["value_loss"].append(0.0)
        metrics["entropy"].append(0.0)
        metrics.setdefault("racing_line_mae_m", []).append(eval_res["racing_line_mae_m"])
        metrics.setdefault("apex_accuracy_pct", []).append(eval_res["apex_accuracy_pct"])
        metrics.setdefault("corner_delta_v_kmh", []).append(eval_res["corner_delta_v_kmh"])
        metrics.setdefault("friction_utilization_pct", []).append(eval_res["friction_utilization_pct"])
        metrics.setdefault("lap_efficiency_pct", []).append(eval_res["lap_efficiency_pct"])

    best_eval_completion = max(best_eval_completion, eval_res["mean_completion"])
    print(
        f"[Step {global_step:06d}/{target_total_steps}] "
        f"Eval Return: {eval_res['mean_return']:+.2f} | "
        f"Track Completion: {eval_res['mean_completion']:.1f}% | "
        f"Speed: {eval_res['mean_speed']:.1f} km/h | "
        f"Crash Rate: {eval_res['crash_rate']*100:.0f}%"
    )
    print(
        f"  📊 [INITIAL BENCHMARK] "
        f"Line MAE: {eval_res['racing_line_mae_m']:.2f}m | "
        f"Apex Acc: {eval_res['apex_accuracy_pct']:.1f}% | "
        f"Corner ΔV: {eval_res['corner_delta_v_kmh']:+.1f} km/h | "
        f"Grip: {eval_res['friction_utilization_pct']:.1f}% | "
        f"Pace Eff: {eval_res['lap_efficiency_pct']:.1f}%"
    )

    save_metrics_json(metrics, log_dir=args.log_dir)
    log_tensorboard_metrics(writer, eval_res, global_step, agent=agent, total_finishes=total_finishes)

    start_time = time.time()
    obs, info = env.reset()

    while global_step < target_total_steps:
        # 1. Thu thập Rollout
        for step in range(args.rollout_len):
            global_step += 1
            action, log_prob, value, raw_u = agent.select_action(obs, deterministic=False, return_raw=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)

            states[step] = obs
            actions[step] = action
            raw_us[step] = raw_u
            log_probs[step] = log_prob
            rewards[step] = reward
            values[step] = value
            is_done = bool(terminated) or bool(truncated)
            dones[step] = 1.0 if is_done else 0.0

            if step_info.get("lap_completed", False) or step_info.get("termination_reason") == "lap_completed" or step_info.get("arrive_dest", False):
                total_finishes += 1
                comp_num = int(step_info.get("completed_lap_number", step_info.get("current_lap", 1)))
                lap_time = float(step_info.get("completed_lap_time", step_info.get("lap_time", 0.0)))
                mean_spd = float(step_info.get("mean_speed_km_h", step_info.get("speed_km_h", 0.0)))
                lap_tag = "Flying Lap 🚀" if comp_num >= 2 else "Standing Lap 🏁"
                print(f"  🏁 [TRAIN FINISH #{total_finishes}] Step {global_step} | Lap {comp_num} ({lap_tag}) | Time: {lap_time:.2f}s | Speed: {mean_spd:.1f} km/h")

            obs = next_obs
            if is_done:
                obs, info = env.reset()

        # 2. Ước tính giá trị trạng thái kế tiếp để tính GAE
        _, _, next_value = agent.select_action(obs, deterministic=True)
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # 3. Cập nhật Learning Rate & Entropy theo lịch trình Linear Decay
        if target_total_steps > 0:
            frac = max(0.0, min(1.0, 1.0 - (global_step / target_total_steps)))
            if args.lr_decay:
                current_lr = args.min_lr + frac * (args.lr - args.min_lr)
                agent.set_lr(current_lr)
            if args.ent_decay:
                current_ent = args.min_ent_coef + frac * (args.ent_coef - args.min_ent_coef)
                agent.set_ent_coef(current_ent)

        # 4. Cập nhật mạng Actor-Critic
        train_info = agent.update(states, actions, log_probs, advantages, returns, values, b_raw_us=raw_us)
        update_idx += 1
        if train_info.get("skipped_actor_updates", 0) > 0:
            print(f"  ⚠️ [NaN GUARD] Step {global_step}: bỏ qua {train_info['skipped_actor_updates']} minibatch actor có loss không hữu hạn")

        # 5. Đánh giá định kỳ (Evaluation)
        if update_idx % args.eval_freq == 0 or global_step >= target_total_steps:
            eval_res = evaluator.evaluate(num_episodes=args.eval_episodes, deterministic=True)
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
            with torch.no_grad():
                sigmas = torch.exp(agent.ac.actor_logstd).detach().cpu().numpy()[0]
                s_steer = float(sigmas[0])
                s_thr = float(sigmas[1])
                s_brk = float(sigmas[2]) if len(sigmas) >= 3 else 0.0
            metrics.setdefault("sigma_steer", []).append(s_steer)
            metrics.setdefault("sigma_throttle", []).append(s_thr)
            metrics.setdefault("sigma_brake", []).append(s_brk)
            metrics["total_finishes"] = total_finishes
            metrics.setdefault("racing_line_mae_m", []).append(eval_res["racing_line_mae_m"])
            metrics.setdefault("apex_accuracy_pct", []).append(eval_res["apex_accuracy_pct"])
            metrics.setdefault("corner_delta_v_kmh", []).append(eval_res["corner_delta_v_kmh"])
            metrics.setdefault("friction_utilization_pct", []).append(eval_res["friction_utilization_pct"])
            metrics.setdefault("lap_efficiency_pct", []).append(eval_res["lap_efficiency_pct"])

            lap1_val = eval_res["mean_lap1_time"]
            lap2_val = eval_res["mean_lap2_time"]
            best_lap_val = eval_res["best_lap_time"]

            metrics.setdefault("eval_lap1_time", []).append(lap1_val)
            metrics.setdefault("eval_lap2_time", []).append(lap2_val)
            metrics.setdefault("eval_best_lap_time", []).append(best_lap_val)

            if eval_res["crash_locations"]:
                crash_info_str = f" | Crashes at: {eval_res['crash_locations']}"
            elif best_lap_val > 0 or eval_res["mean_completion"] >= 98.0:
                lap_details = []
                if lap1_val > 0:
                    lap_details.append(f"Lap 1: {lap1_val:.2f}s")
                if lap2_val > 0:
                    lap_details.append(f"Lap 2 (Flying): {lap2_val:.2f}s")
                if not lap_details:
                    lap_details.append(f"Time: {eval_res['mean_time']:.1f}s")
                crash_info_str = f" | Clean 🏁 ({', '.join(lap_details)}, Mean Speed: {eval_res['mean_speed']:.1f} km/h)"
            else:
                crash_info_str = " | Stalled on track"

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
            print(
                f"  📊 [ACADEMIC BENCHMARK] "
                f"Line MAE: {eval_res['racing_line_mae_m']:.2f}m (RMSE: {eval_res['racing_line_rmse_m']:.2f}m) | "
                f"Apex: {eval_res['apex_accuracy_pct']:.1f}% | "
                f"Corner ΔV: {eval_res['corner_delta_v_kmh']:+.1f} km/h | "
                f"Grip: {eval_res['friction_utilization_pct']:.1f}% | "
                f"Pace Eff: {eval_res['lap_efficiency_pct']:.1f}%"
            )

            # Lưu Best Model: Ưu tiên Tuyệt Đối Độ Tin Cậy (Zero Crash cả 3/3 vòng) trước, Lap Time sau
            cur_comp = eval_res["mean_completion"]
            cur_speed = eval_res["mean_speed"]
            cur_crash_rate = eval_res.get("crash_rate", 1.0)
            cur_time = best_lap_val if best_lap_val > 0 else eval_res["mean_time"]
            is_clean_benchmark = bool(cur_crash_rate == 0.0 and cur_comp >= 98.0)
            is_new_best = False
            record_msg = ""

            if is_clean_benchmark:
                # Trạng thái đạt 100% tin cậy (Zero Crash cả 3/3 episode đánh giá)
                if best_eval_completion < 98.0 or best_eval_lap_time == float("inf"):
                    best_eval_completion = 100.0
                    best_eval_lap_time = cur_time
                    best_eval_speed = cur_speed
                    if lap2_val > 0:
                        best_eval_flying_time = lap2_val
                    is_new_best = True
                    record_msg = f"Kỷ lục Clean Lap 100% Đầu Tiên (Zero Crash) 🏁 {cur_time:.2f}s | Speed: {cur_speed:.1f} km/h"
                elif cur_time < best_eval_lap_time:
                    best_eval_lap_time = cur_time
                    best_eval_speed = max(best_eval_speed, cur_speed)
                    if lap2_val > 0 and lap2_val < best_eval_flying_time:
                        best_eval_flying_time = lap2_val
                    is_new_best = True
                    record_msg = f"Kỷ lục Lap Time Sạch Mới 🚀 {cur_time:.2f}s (Speed: {cur_speed:.1f} km/h)"
                elif lap2_val > 0 and lap2_val < best_eval_flying_time:
                    best_eval_flying_time = lap2_val
                    is_new_best = True
                    record_msg = f"Kỷ lục Flying Lap Sạch 🚀 {lap2_val:.2f}s | Speed: {cur_speed:.1f} km/h"
            else:
                # Giai đoạn xe chưa đạt được 100% Clean Lap: chỉ ghi nhận kỷ lục quãng đường xa nhất trước khi đâm
                if best_eval_completion < 98.0 and cur_comp > best_eval_completion:
                    best_eval_completion = cur_comp
                    is_new_best = True
                    record_msg = f"Quãng đường kỷ lục: {best_eval_completion:.1f}%"

            if is_new_best:
                best_model_path = os.path.join(args.save_dir, "best_agent.pt")
                agent.save(best_model_path)
                print(f"  🏆 [VALID RECORD - BEST AGENT SAVED] {best_model_path} ({record_msg})")

            # Luôn lưu checkpoint mới nhất
            last_model_path = os.path.join(args.save_dir, "last_agent.pt")
            agent.save(last_model_path)

            # Lưu Milestone Snapshot định kỳ
            if args.milestone_freq > 0:
                prev_milestone = (global_step - args.rollout_len * args.eval_freq) // args.milestone_freq
                curr_milestone = global_step // args.milestone_freq
                if curr_milestone > prev_milestone and curr_milestone > 0:
                    milestone_step = curr_milestone * args.milestone_freq
                    milestone_path = os.path.join(args.save_dir, f"agent_step_{milestone_step // 1000}k.pt")
                    agent.save(milestone_path)
                    print(f"  💾 [MILESTONE SAVED] Đã lưu snapshot tại: {milestone_path}")

            # Lưu metrics JSON định kỳ
            save_metrics_json(metrics, log_dir=args.log_dir)

            # Ghi log liên tục lên TensorBoard Dashboard
            log_tensorboard_metrics(writer, eval_res, global_step, train_info=train_info, agent=agent, total_finishes=total_finishes, fps=fps)

            # Reset lại env để tiếp tục thu thập rollout cho vòng lặp kế tiếp
            obs, info = env.reset()

    writer.close()
    env.close()
    print("=" * 70)
    print(f">> HOÀN THÀNH HUẤN LUYỆN ({global_step}/{target_total_steps} steps, {total_finishes} finishes)!")
    print(f">> Best Agent: {os.path.join(args.save_dir, 'best_agent.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
