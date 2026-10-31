"""
Huấn luyện PPO Self-Play Đa Tác Tử (Parameter Sharing) trên MultiCarRacingEnv
- Phát triển trực tiếp từ nhánh feature/111s-honest-anchor, warm-start từ models/ref_111s/best_agent.pt (35D -> 60D).
- Kế thừa đầy đủ bản vá F13 (lưu raw_u vào buffer) và D19 (logstd_max = [0.0, -0.7, -0.7]).
- Hỗ trợ phạt mượt lái (c1 = steer_smooth_coef) và mượt bàn đạp (c2 = pedal_smooth_coef) theo §4.2:
    + Bộ A (--preset-smooth A): c1 = 1.0, c2 = 0.05
    + Bộ B (--preset-smooth B): c1 = 0.5, c2 = 0.02
- Tự động đo và log các chỉ số §6 (pedal_sw/100, steer_rev/100, beta_p95, số xe về đích, số lần chạm, số lần vượt).
"""

import os
import sys
import time
import json
import subprocess
import argparse
from typing import Dict, Any, List
import numpy as np
import torch

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.multicar_env import MultiCarRacingEnv, BASE_OBS_DIM
from agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-car PPO self-play racing (finetune from ref_111s)")
    parser.add_argument("--total-samples", type=int, default=300000, help="Tổng số mẫu (cộng dồn mọi xe; 200-300k cho vòng thử ngắn, 1M cho bản chốt)")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Số bước mô phỏng mỗi rollout (x số xe = số mẫu)")
    parser.add_argument("--num-cars", type=int, default=6)
    parser.add_argument("--lane-num", type=int, default=3, help="Số làn (3 làn x 4.5m = 13.5m, hoặc 2 làn = 9.0m)")
    parser.add_argument("--vehicle", type=str, default="gt3_anchor", choices=["gt3_anchor", "gt3"], help="Thực thể xe (mặc định gt3_anchor)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=4.0)
    parser.add_argument("--preset-smooth", choices=["none", "A", "B"], default="none",
                        help="Bộ hệ số phạt mượt §4.2: A (c1=1.0, c2=0.05), B (c1=0.5, c2=0.02)")
    parser.add_argument("--steer-smooth-coef", type=float, default=None, help="Hệ số c1 phạt Δlái^2 (ghi đè --preset-smooth)")
    parser.add_argument("--pedal-smooth-coef", type=float, default=None, help="Hệ số c2 phạt Δpedal^2 (ghi đè --preset-smooth)")
    parser.add_argument("--center-coef", type=float, default=0.0, help="Hệ số phạt lệch tim đường r_center (§4.3)")
    parser.add_argument("--grid", choices=["abreast", "staggered"], default="abreast", help="Kiểu xếp xe xuất phát")
    parser.add_argument("--init-model", type=str, default="models/ref_111s/best_agent.pt", help="Checkpoint đơn tác tử 35D để warm start")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint đa tác tử 60D để train tiếp")
    parser.add_argument("--eval-freq", type=int, default=5, help="Đánh giá sau mỗi N lần update")
    parser.add_argument("--milestone-freq", type=int, default=100000, help="Lưu snapshot mỗi N mẫu")
    parser.add_argument("--save-dir", type=str, default="models/multicar_demo")
    parser.add_argument("--log-dir", type=str, default="logs/multicar_demo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_smooth_coefs(args):
    c1, c2 = 0.10, 0.0
    if args.preset_smooth == "A":
        c1, c2 = 1.0, 0.05
    elif args.preset_smooth == "B":
        c1, c2 = 0.5, 0.02
    if args.steer_smooth_coef is not None:
        c1 = float(args.steer_smooth_coef)
    if args.pedal_smooth_coef is not None:
        c2 = float(args.pedal_smooth_coef)
    return c1, c2


def warm_start_from_single(agent: PPOAgent, path: str, base_dim: int = BASE_OBS_DIM):
    """Nạp checkpoint đơn tác tử (35D của ref_111s) vào mạng đa tác tử (35 + 5K = 60D):
    copy 35 cột đầu của lớp đầu tiên, khởi tạo 0 cho 25 cột đối thủ."""
    ckpt = torch.load(path, map_location=agent.device)
    old_sd = ckpt.get("model_state_dict", ckpt)
    new_sd = agent.ac.state_dict()
    for key, val in old_sd.items():
        if key not in new_sd:
            continue
        if new_sd[key].shape == val.shape:
            new_sd[key] = val.clone()
        elif key in ("actor_mean.0.weight", "critic.0.weight") and val.shape[1] <= new_sd[key].shape[1]:
            in_cols = min(base_dim, val.shape[1])
            widened = torch.zeros_like(new_sd[key])
            widened[:, :in_cols] = val[:, :in_cols]
            new_sd[key] = widened
    agent.ac.load_state_dict(new_sd)
    agent._project_logstd()


def policy_act(agent: PPOAgent, obs: np.ndarray, deterministic: bool):
    """Chọn hành động theo batch cho các xe đang chạy, trả về cả raw_u trước tanh (bản vá F13)"""
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        action, log_prob, _, value, raw_u = agent.ac.get_action_and_value(
            x, deterministic=deterministic, return_raw=True
        )
    return (
        np.clip(action.cpu().numpy(), -1.0, 1.0),
        log_prob.cpu().numpy(),
        value.view(-1).cpu().numpy(),
        raw_u.cpu().numpy(),
    )


def run_race(env: MultiCarRacingEnv, agent: PPOAgent, deterministic: bool = True) -> Dict[str, Any]:
    """Chạy 1 cuộc đua đầy đủ, tất cả các xe do policy điều khiển (không respawn: xe bị loại là rời cuộc đua)"""
    train_respawn = env.respawn
    env.respawn = False
    obs, info = env.reset()
    actions = np.zeros((env.num_cars, 3), dtype=np.float32)
    while True:
        active = env.active_mask()
        if active.any():
            act, _, _, _ = policy_act(agent, obs[active], deterministic)
            actions[active] = act
        obs, _, _, truncated, _ = env.step(actions)
        if truncated or not env.active_mask().any():
            break
    env.respawn = train_respawn
    racers = env.racers
    laps = [st.lap_times[-1] for st in racers if st.finished]
    style = env.compute_driving_style_metrics()
    return {
        "finished": int(sum(st.finished for st in racers)),
        "outs": int(sum(st.termination_reason == "crash_or_track_limit" for st in racers)),
        "stalls": int(sum(st.termination_reason == "stalled" for st in racers)),
        "contacts": int(sum(st.contacts for st in racers)),
        "overtakes": int(sum(st.overtakes for st in racers)),
        "mean_completion": float(np.mean([min(1.0, max(0.0, env.race_distance(st)) / env.circuit_length) for st in racers]) * 100.0),
        "mean_lap": float(np.mean(laps)) if laps else 0.0,
        "best_lap": float(np.min(laps)) if laps else 0.0,
        "pedal_sw_per_100": style["pedal_sw_per_100"],
        "steer_rev_per_100": style["steer_rev_per_100"],
        "beta_p95_deg": style["beta_p95_deg"],
        "standings": env.standings(),
    }


def write_run_readme(args, c1: float, c2: float, obs_dim: int):
    """Ghi README ngắn trong thư mục log theo §5.1"""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        commit, branch = "unknown", "unknown"
    cmd_str = " ".join(sys.argv)
    warm_src = args.resume if args.resume else args.init_model
    content = (
        f"# Run Log: {os.path.basename(args.log_dir)}\n\n"
        f"- **Branch**: `{branch}`\n"
        f"- **Commit**: `{commit}`\n"
        f"- **Warm-start / Resume**: `{warm_src}`\n"
        f"- **Vehicle**: `{args.vehicle}` | **Lanes**: `{args.lane_num}` | **Cars**: `{args.num_cars}` | **Obs**: `{obs_dim}D`\n"
        f"- **Smoothness Coefs (§4.2)**: `c1 (steer_smooth_coef) = {c1}`, `c2 (pedal_smooth_coef) = {c2}`, `center_coef = {args.center_coef}`\n"
        f"- **Collision Penalty**: `{args.collision_penalty}` (with proximity buffer & WI-L5 clean overtake bonus)\n"
        f"- **Command**:\n```powershell\npython {cmd_str}\n```\n"
    )
    with open(os.path.join(args.log_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(content)


def main():
    args = parse_args()
    c1, c2 = resolve_smooth_coefs(args)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = MultiCarRacingEnv(
        num_cars=args.num_cars,
        lane_num=args.lane_num,
        vehicle_name=args.vehicle,
        time_penalty=args.time_penalty,
        collision_penalty=args.collision_penalty,
        steer_smooth_coef=c1,
        pedal_smooth_coef=c2,
        center_coef=args.center_coef,
        start_seed=args.seed,
        respawn=True,
        grid=args.grid,
    )
    obs_dim = int(env.obs_dim)
    write_run_readme(args, c1, c2, obs_dim)

    # D19: trần logstd [0.0, -0.7, -0.7] giữ sigma ga/phanh <= 0.5 để không giật pedal trong rollout
    agent = PPOAgent(
        state_dim=obs_dim,
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        minibatch_size=args.batch_size,
        logstd_max=[0.0, -0.7, -0.7],
    )

    if args.resume:
        agent.load(args.resume)
        agent.set_lr(args.lr)
        agent._project_logstd()
        print(f">> RESUME từ checkpoint đa tác tử: {args.resume}")
    elif args.init_model and os.path.exists(args.init_model):
        warm_start_from_single(agent, args.init_model, base_dim=BASE_OBS_DIM)
        print(f">> WARM START từ ref_111s: {args.init_model} ({BASE_OBS_DIM}D -> {obs_dim}D)")
    else:
        print(">> Huấn luyện từ đầu (không có checkpoint khởi tạo)")

    print("=" * 80)
    print(f">> MULTI-CAR PPO SELF-PLAY | {args.num_cars} xe ({args.vehicle}) | {args.lane_num} làn ({env.half_width * 2:.1f}m) | obs {obs_dim}D | device {agent.device}")
    print(f"   Smoothness: c1(steer)={c1}, c2(pedal)={c2}, center={args.center_coef} | lr={args.lr} | collision={args.collision_penalty}")
    print("=" * 80)

    metrics: Dict[str, List] = {k: [] for k in [
        "samples", "eval_finished", "eval_outs", "eval_stalls", "eval_contacts", "eval_overtakes", "eval_completion",
        "eval_mean_lap", "eval_best_lap", "pedal_sw_per_100", "steer_rev_per_100", "beta_p95_deg",
        "policy_loss", "value_loss", "entropy", "explained_var", "train_laps"]}

    def log_eval(samples: int, res: Dict[str, Any], train_info: Dict[str, float], train_laps: int):
        for key, val in [
            ("samples", samples),
            ("eval_finished", res["finished"]),
            ("eval_outs", res["outs"]),
            ("eval_stalls", res["stalls"]),
            ("eval_contacts", res["contacts"]),
            ("eval_overtakes", res["overtakes"]),
            ("eval_completion", res["mean_completion"]),
            ("eval_mean_lap", res["mean_lap"]),
            ("eval_best_lap", res["best_lap"]),
            ("pedal_sw_per_100", res["pedal_sw_per_100"]),
            ("steer_rev_per_100", res["steer_rev_per_100"]),
            ("beta_p95_deg", res["beta_p95_deg"]),
            ("policy_loss", train_info.get("policy_loss", 0.0)),
            ("value_loss", train_info.get("value_loss", 0.0)),
            ("entropy", train_info.get("entropy", 0.0)),
            ("explained_var", train_info.get("explained_var", 0.0)),
            ("train_laps", train_laps),
        ]:
            metrics[key].append(val)
        with open(os.path.join(args.log_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def score(res: Dict[str, Any]):
        # Khi >= 5 xe về đích, ưu tiên giảm mạnh va chạm (contacts) và đạt 3 chỉ số mượt §6
        finish_tier = min(res["finished"], 5)
        smooth_ok = (
            int(res["pedal_sw_per_100"] <= 5.0)
            + int(res["steer_rev_per_100"] <= 40.0)
            + int(res["beta_p95_deg"] <= 4.0)
        )
        return (
            finish_tier,
            smooth_ok,
            -(res["contacts"] // 10),
            res["finished"],
            -res["mean_lap"] if res["mean_lap"] > 0 else -1e9,
            res["mean_completion"],
        )

    res = run_race(env, agent, deterministic=True)
    best_score = (-1,)
    print(
        f"[Samples 0] Eval: về đích {res['finished']}/{args.num_cars} | văng {res['outs']} | chạm {res['contacts']} | vượt {res['overtakes']} | "
        f"lap TB {res['mean_lap']:.1f}s | "
        f"§6[pedal_sw={res['pedal_sw_per_100']:.1f} (<=5), steer_rev={res['steer_rev_per_100']:.1f} (<=40), β_p95={res['beta_p95_deg']:.1f}° (<=4°)]"
    )
    log_eval(0, res, {}, 0)
    agent.save(os.path.join(args.save_dir, "best_agent.pt"))
    agent.save(os.path.join(args.save_dir, "init_warmstart_60d.pt"))

    samples = 0
    update_idx = 0
    train_laps = 0
    next_milestone = args.milestone_freq
    start_time = time.time()
    obs, info = env.reset()
    actions = np.zeros((args.num_cars, 3), dtype=np.float32)

    while samples < args.total_samples:
        # 1. Thu thập rollout: mỗi xe một chuỗi riêng (lưu cả raw_u cho bản vá F13)
        buf = [{"obs": [], "act": [], "logp": [], "rew": [], "val": [], "done": [], "raw_u": []} for _ in range(args.num_cars)]
        for _ in range(args.rollout_steps):
            active = env.active_mask()
            idx = np.flatnonzero(active)
            act, logp, val, raw_u = policy_act(agent, obs[idx], deterministic=False)
            actions[idx] = act
            next_obs, rewards, dones, truncated, infos = env.step(actions)
            for j, car in enumerate(idx):
                b = buf[car]
                b["obs"].append(obs[car])
                b["act"].append(act[j])
                b["logp"].append(logp[j])
                b["rew"].append(rewards[car])
                b["val"].append(val[j])
                b["raw_u"].append(raw_u[j])
                b["done"].append(1.0 if (dones[car] or truncated) else 0.0)
                if "completed_lap_time" in infos[car]:
                    train_laps += 1
            obs = next_obs
            if truncated or not env.active_mask().any():
                obs, info = env.reset()

        # 2. GAE từng xe, bootstrap bằng V(s) hiện tại nếu xe vẫn đang chạy
        active = env.active_mask()
        _, _, last_vals, _ = policy_act(agent, obs, deterministic=True)
        all_s, all_a, all_lp, all_adv, all_ret, all_v, all_u = [], [], [], [], [], [], []
        for car in range(args.num_cars):
            b = buf[car]
            if not b["rew"]:
                continue
            next_value = float(last_vals[car]) if active[car] else 0.0
            adv, ret = agent.compute_gae(
                np.array(b["rew"], dtype=np.float32),
                np.array(b["val"], dtype=np.float32),
                np.array(b["done"], dtype=np.float32),
                next_value,
            )
            all_s.append(np.array(b["obs"], dtype=np.float32))
            all_a.append(np.array(b["act"], dtype=np.float32))
            all_lp.append(np.array(b["logp"], dtype=np.float32))
            all_adv.append(adv)
            all_ret.append(ret)
            all_v.append(np.array(b["val"], dtype=np.float32))
            all_u.append(np.array(b["raw_u"], dtype=np.float32))
        batch_n = int(sum(len(x) for x in all_s))
        samples += batch_n
        train_info = agent.update(
            np.concatenate(all_s),
            np.concatenate(all_a),
            np.concatenate(all_lp),
            np.concatenate(all_adv),
            np.concatenate(all_ret),
            np.concatenate(all_v),
            b_raw_us=np.concatenate(all_u),
        )
        update_idx += 1

        # 3. Đánh giá định kỳ bằng 1 cuộc đua tất định đủ N xe
        if update_idx % args.eval_freq == 0 or samples >= args.total_samples:
            res = run_race(env, agent, deterministic=True)
            sps = int(samples / max(1e-5, time.time() - start_time))
            log_eval(samples, res, train_info, train_laps)
            print(
                f"[Samples {samples:07d}/{args.total_samples} | {sps} mẫu/s | lap train: {train_laps}] "
                f"Eval: về đích {res['finished']}/{args.num_cars} | văng {res['outs']} | chạm {res['contacts']} | vượt {res['overtakes']} | "
                f"lap TB {res['mean_lap']:.1f}s (best {res['best_lap']:.1f}s) | "
                f"§6[pedal_sw={res['pedal_sw_per_100']:.1f}, steer_rev={res['steer_rev_per_100']:.1f}, β_p95={res['beta_p95_deg']:.1f}°]"
            )
            agent.save(os.path.join(args.save_dir, "last_agent.pt"))
            if score(res) > best_score:
                best_score = score(res)
                agent.save(os.path.join(args.save_dir, "best_agent.pt"))
                print(
                    f"  🏆 [BEST SAVED] về đích {res['finished']}/{args.num_cars}, chạm {res['contacts']}, vượt {res['overtakes']}, lap TB {res['mean_lap']:.1f}s, "
                    f"pedal_sw={res['pedal_sw_per_100']:.1f}, steer_rev={res['steer_rev_per_100']:.1f}, β_p95={res['beta_p95_deg']:.1f}°"
                )
            if samples >= next_milestone:
                agent.save(os.path.join(args.save_dir, f"agent_{next_milestone // 1000}k.pt"))
                next_milestone += args.milestone_freq
            obs, info = env.reset()

    env.close()
    print(f">> HOÀN THÀNH: {samples} mẫu, {train_laps} vòng khi train. Best: {os.path.join(args.save_dir, 'best_agent.pt')}")


if __name__ == "__main__":
    main()
