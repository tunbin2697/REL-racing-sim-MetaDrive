"""
Huấn luyện PPO Self-Play Đa Tác Tử (Parameter Sharing) trên MultiCarRacingEnv
- Một mạng PPO dùng chung cho cả N xe: mỗi bước mô phỏng thu được tối đa N mẫu.
- Mỗi xe có chuỗi rollout riêng để tính GAE đúng ranh giới episode của chính nó.
- Warm start từ checkpoint đơn tác tử 35D (mặc định ref_111s): copy trọng số 35 cột đầu, cột quan sát đối thủ
  khởi tạo 0 -> ban đầu hành vi y hệt agent đơn, sau đó học thêm né / vượt xe khác (handoff §5.3).
- Lưu u trước tanh vào buffer để PPO tính lại log_prob đúng điểm đã sample (F13).
- Eval: 1 cuộc đua tất định + N cuộc đua ngẫu nhiên, kèm chỉ số kiểu lái handoff §6.
"""

import os
import sys
import time
import json
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

from game_env.multicar_env import MultiCarRacingEnv, driving_style
from agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-car PPO self-play racing (MetaDrive 3D)")
    parser.add_argument("--total-samples", type=int, default=1000000, help="Tổng số mẫu (cộng dồn mọi xe)")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Số bước mô phỏng mỗi rollout (x số xe = số mẫu)")
    parser.add_argument("--num-cars", type=int, default=6)
    parser.add_argument("--map", type=str, default="silverstone_half_3lane", help="Preset map (game_env/track_presets.py)")
    parser.add_argument("--vehicle", type=str, default="gt3_anchor")
    parser.add_argument("--laps", type=int, default=1, help="Số vòng mỗi cuộc đua")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=2.0)
    parser.add_argument("--steer-smooth", type=float, default=0.10, help="c1: r_steer_smooth = -c1 * Δlái^2 (0.10 = hành vi cũ)")
    parser.add_argument("--pedal-smooth", type=float, default=0.0, help="c2: r_pedal_smooth = -c2 * Δpedal^2 (0 = tắt)")
    parser.add_argument("--straight-steer", type=float, default=0.0,
                        help="c3: r_straight = -c3 * lái^2 khi 60m phía trước là đường thẳng (0 = tắt)")
    parser.add_argument("--respawn", choices=["random", "inplace"], default="random",
                        help="Khi train: random = đặt lại ở chỗ trống ngẫu nhiên; inplace = đứng ngoài vài giây rồi quay lại chỗ văng")
    parser.add_argument("--respawn-delay", type=float, default=3.0, help="Thời gian phạt (s) trước khi hồi sinh tại chỗ")
    parser.add_argument("--race-reset", type=float, nargs=2, default=None, metavar=("MIN_S", "MAX_S"),
                        help="Khi train: cả cuộc đua bắt đầu lại từ vạch xuất phát sau ngẫu nhiên MIN_S..MAX_S giây "
                             "(luyện pha xuất phát / cua đầu khi 6 xe còn chạy thành cụm); mặc định: hết cuộc đua")
    parser.add_argument("--full-race-prob", type=float, default=0.0,
                        help="Với --race-reset: tỉ lệ cuộc đua train chạy trọn (không cắt). Không có thì xe không bao giờ "
                             "chạy hết vòng khi train -> thiếu thưởng về đích và dữ liệu cuối vòng")
    parser.add_argument("--crash-penalty-scale", type=float, default=1.0,
                        help="Nhân phạt văng -10 - 20 (v/v_max)^2 (1.0 = như cũ)")
    parser.add_argument("--center-coef", type=float, default=0.0, help="r_center = -c * (lat / half_width)^2 (ref_111s: 0.04)")
    parser.add_argument("--grid", choices=["abreast", "staggered"], default="abreast", help="Kiểu xếp xe xuất phát")
    parser.add_argument("--init-model", type=str, default="models/ref_111s/best_agent.pt", help="Checkpoint đơn tác tử 35D để warm start")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint đa tác tử để train tiếp")
    parser.add_argument("--eval-freq", type=int, default=10, help="Đánh giá sau mỗi N lần update")
    parser.add_argument("--eval-races", type=int, default=2, help="Số cuộc đua ngẫu nhiên mỗi lượt eval (cộng 1 cuộc đua tất định)")
    parser.add_argument("--milestone-freq", type=int, default=250000, help="Lưu snapshot mỗi N mẫu")
    parser.add_argument("--save-dir", type=str, default="models/multicar")
    parser.add_argument("--log-dir", type=str, default="logs/multicar")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def warm_start_from_single(agent: PPOAgent, path: str, base_dim: int):
    """Nạp checkpoint đơn tác tử base_dim chiều vào mạng (base_dim + 5K)D: lớp đầu copy base_dim cột, các cột đối thủ = 0"""
    ckpt = torch.load(path, map_location=agent.device)
    old_sd = ckpt.get("model_state_dict", ckpt)
    new_sd = agent.ac.state_dict()
    for key, val in old_sd.items():
        if key not in new_sd:
            continue
        if new_sd[key].shape == val.shape:
            new_sd[key] = val.clone()
        elif key in ("actor_mean.0.weight", "critic.0.weight") and val.shape[1] == base_dim:
            widened = torch.zeros_like(new_sd[key])
            widened[:, :base_dim] = val
            new_sd[key] = widened
        else:
            raise ValueError(f"Không nới được {key}: checkpoint {tuple(val.shape)} vs mạng {tuple(new_sd[key].shape)}")
    agent.ac.load_state_dict(new_sd)
    agent._project_logstd()


def policy_act(agent: PPOAgent, obs: np.ndarray, deterministic: bool):
    """Chọn hành động theo batch cho các xe đang chạy; trả thêm u trước tanh để lưu vào buffer"""
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        action, log_prob, _, value, raw_u = agent.ac.get_action_and_value(x, deterministic=deterministic, return_raw=True)
    return action.cpu().numpy(), log_prob.cpu().numpy(), value.view(-1).cpu().numpy(), raw_u.cpu().numpy()


def run_race(env: MultiCarRacingEnv, agent: PPOAgent, deterministic: bool = True, max_steps: int = None) -> Dict[str, Any]:
    """Chạy 1 cuộc đua đầy đủ, tất cả các xe do policy điều khiển (không respawn: xe bị loại là rời cuộc đua)"""
    train_respawn = env.respawn
    env.respawn = False
    env.record_style = True
    train_max_steps = env.max_steps
    if max_steps is not None:
        env.max_steps = int(max_steps)
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
    env.max_steps = train_max_steps
    env.record_style = False
    racers = env.racers
    laps = [t for st in racers for t in st.lap_times]
    res = {
        "finished": int(sum(st.finished for st in racers)),
        "outs": int(sum(st.termination_reason == "crash_or_track_limit" for st in racers)),
        "stalls": int(sum(st.termination_reason == "stalled" for st in racers)),
        "contacts": int(sum(st.contacts for st in racers)),
        "mean_completion": float(np.mean([min(1.0, max(0.0, env.race_distance(st)) / (env.circuit_length * env.max_laps))
                                          for st in racers]) * 100.0),
        "mean_lap": float(np.mean(laps)) if laps else 0.0,
        "best_lap": float(np.min(laps)) if laps else 0.0,
        "standings": env.standings(),
    }
    res.update(driving_style(racers))
    return res


STAT_KEYS = ["finished", "outs", "stalls", "contacts", "mean_completion", "pedal_flips_per100",
             "steer_reversals_per100", "beta_p95_deg"]


def evaluate(env: MultiCarRacingEnv, agent: PPOAgent, stochastic_races: int, max_steps: int = None) -> Dict[str, Any]:
    """1 cuộc đua tất định + N cuộc đua ngẫu nhiên, lấy trung bình (1 cuộc đua quá nhiễu để chọn model)"""
    races = [run_race(env, agent, deterministic=True, max_steps=max_steps)] +         [run_race(env, agent, deterministic=False, max_steps=max_steps) for _ in range(stochastic_races)]
    out = {k: float(np.mean([r[k] for r in races])) for k in STAT_KEYS}
    laps = [r["mean_lap"] for r in races if r["mean_lap"] > 0]
    out["mean_lap"] = float(np.mean(laps)) if laps else 0.0
    out["best_lap"] = float(min(r["best_lap"] for r in races if r["best_lap"] > 0)) if laps else 0.0
    out["races"] = len(races)
    return out


def describe(res: Dict[str, Any], num_cars: int) -> str:
    return (f"về đích {res['finished']:.1f}/{num_cars} | văng {res['outs']:.1f} | dừng {res['stalls']:.1f} | chạm {res['contacts']:.1f} | "
            f"đi được {res['mean_completion']:.1f}% | lap TB {res['mean_lap']:.1f}s (best {res['best_lap']:.1f}s) | "
            f"pedal đổi {res['pedal_flips_per100']:.1f} lái đổi {res['steer_reversals_per100']:.1f} /100b β95 {res['beta_p95_deg']:.1f}°")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    race_steps = 3000 * args.laps
    env = MultiCarRacingEnv(num_cars=args.num_cars, map_type=args.map, vehicle=args.vehicle, max_laps=args.laps,
                            time_penalty=args.time_penalty, collision_penalty=args.collision_penalty,
                            steer_smooth_coef=args.steer_smooth, pedal_smooth_coef=args.pedal_smooth,
                            straight_steer_coef=args.straight_steer, crash_penalty_scale=args.crash_penalty_scale,
                            center_coef=args.center_coef,
                            max_steps=race_steps, start_seed=args.seed, respawn=args.respawn,
                            respawn_delay=args.respawn_delay, grid=args.grid)
    obs_dim = int(env.obs_dim)

    def new_train_race():
        """Bắt đầu lại cả cuộc đua từ vạch xuất phát; độ dài ngẫu nhiên để luyện nhiều pha chạy thành cụm"""
        if args.race_reset and np.random.rand() >= args.full_race_prob:
            lo, hi = args.race_reset
            env.max_steps = int(np.random.uniform(lo, hi) / env.dt)
        else:
            env.max_steps = race_steps
        return env.reset()
    agent = PPOAgent(state_dim=obs_dim, action_dim=3, lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                     clip_coef=args.clip_coef, ent_coef=args.ent_coef, minibatch_size=args.batch_size,
                     logstd_max=env.preset.logstd_max)

    if args.resume:
        agent.load(args.resume)
        agent.set_lr(args.lr)
        print(f">> RESUME từ checkpoint đa tác tử: {args.resume}")
    elif args.init_model and os.path.exists(args.init_model):
        warm_start_from_single(agent, args.init_model, env.base_obs_dim)
        print(f">> WARM START từ checkpoint đơn tác tử: {args.init_model} ({env.base_obs_dim}D -> {obs_dim}D)")
    else:
        print(">> Huấn luyện từ đầu (không có checkpoint khởi tạo)")

    meta = {"kind": "multicar", "map": args.map, "vehicle": args.vehicle, "num_cars": args.num_cars, "obs_dim": obs_dim,
            "base_obs_dim": env.base_obs_dim, "steer_smooth": args.steer_smooth, "pedal_smooth": args.pedal_smooth,
            "straight_steer": args.straight_steer, "respawn": args.respawn,
            "crash_penalty_scale": args.crash_penalty_scale,
            "race_reset": args.race_reset, "full_race_prob": args.full_race_prob,
            "center_coef": args.center_coef, "collision_penalty": args.collision_penalty, "time_penalty": args.time_penalty,
            "init_model": args.resume or args.init_model}
    with open(os.path.join(args.log_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "args": vars(args)}, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f">> MULTI-CAR PPO SELF-PLAY | {args.num_cars} xe {args.vehicle} | {args.map} ({env.preset.half_width * 2:.1f}m) | "
          f"obs {obs_dim}D | device {agent.device}")
    print(f"   lr {args.lr} | gamma {args.gamma} | lambda {args.gae_lambda} | ent {args.ent_coef} | time {args.time_penalty} | "
          f"collision {args.collision_penalty}")
    print(f"   phạt mượt: lái c1 {args.steer_smooth} | pedal c2 {args.pedal_smooth} | lái đường thẳng c3 {args.straight_steer} | "
          f"center {args.center_coef} | phạt văng x{args.crash_penalty_scale} | respawn {args.respawn} | race reset {args.race_reset or 'hết cuộc đua'} (trọn cuộc đua {args.full_race_prob:.0%})")
    print(f"   "
          f"{args.laps} vòng | grid {args.grid} | eval 1 tất định + {args.eval_races} ngẫu nhiên / {args.eval_freq} update")
    print("=" * 70)

    metrics: Dict[str, List] = {k: [] for k in ["samples", "train_laps", "policy_loss", "value_loss", "entropy", "explained_var"]
                                + [f"eval_{k}" for k in STAT_KEYS + ["mean_lap", "best_lap"]]}

    def log_eval(samples: int, res: Dict[str, Any], train_info: Dict[str, float], train_laps: int):
        metrics["samples"].append(samples)
        metrics["train_laps"].append(train_laps)
        for k in ["policy_loss", "value_loss", "entropy", "explained_var"]:
            metrics[k].append(float(train_info.get(k, 0.0)))
        for k in STAT_KEYS + ["mean_lap", "best_lap"]:
            metrics[f"eval_{k}"].append(res[k])
        with open(os.path.join(args.log_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def score(res: Dict[str, Any]):
        # Ưu tiên số xe về đích, sau đó lap time trung bình, sau đó quãng đường
        return (round(res["finished"], 2), -res["mean_lap"] if res["mean_lap"] > 0 else -1e9, res["mean_completion"])

    res = evaluate(env, agent, args.eval_races, max_steps=race_steps)
    best_score = score(res)
    print(f"[Samples 0] Eval: {describe(res, args.num_cars)}", flush=True)
    log_eval(0, res, {}, 0)
    agent.save(os.path.join(args.save_dir, "best_agent.pt"), meta={**meta, "samples": 0})

    samples = 0
    update_idx = 0
    train_laps = 0
    next_milestone = args.milestone_freq
    start_time = time.time()
    obs, info = new_train_race()
    actions = np.zeros((args.num_cars, 3), dtype=np.float32)

    while samples < args.total_samples:
        # 1. Thu thập rollout: mỗi xe một chuỗi riêng
        buf = [{"obs": [], "act": [], "u": [], "logp": [], "rew": [], "val": [], "done": []} for _ in range(args.num_cars)]
        for _ in range(args.rollout_steps):
            idx = np.flatnonzero(env.active_mask())
            act, logp, val, raw_u = policy_act(agent, obs[idx], deterministic=False)
            actions[idx] = act
            next_obs, rewards, dones, truncated, infos = env.step(actions)
            if truncated and args.race_reset:
                # Cắt cuộc đua giữa chừng không phải trạng thái kết thúc thật: bootstrap r += gamma * V(s')
                alive = [car for car in idx if not dones[car] and env.racers[car].active]
                if alive:
                    _, _, boot, _ = policy_act(agent, next_obs[alive], deterministic=True)
                    rewards = rewards.copy()
                    rewards[alive] += agent.gamma * boot
            for j, car in enumerate(idx):
                b = buf[car]
                b["obs"].append(obs[car])
                b["act"].append(act[j])
                b["u"].append(raw_u[j])
                b["logp"].append(logp[j])
                b["rew"].append(rewards[car])
                b["val"].append(val[j])
                # Truncate theo horizon cũng cắt chuỗi (không bootstrap) giống bản đơn tác tử
                b["done"].append(1.0 if (dones[car] or truncated) else 0.0)
                if "completed_lap_time" in infos[car]:
                    train_laps += 1
            obs = next_obs
            if truncated or env.race_over():
                obs, info = new_train_race()

        # 2. GAE từng xe, bootstrap bằng V(s) hiện tại nếu xe vẫn đang chạy
        active = env.active_mask()
        _, _, last_vals, _ = policy_act(agent, obs, deterministic=True)
        cols = {k: [] for k in ["obs", "act", "u", "logp", "adv", "ret", "val"]}
        for car in range(args.num_cars):
            b = buf[car]
            if not b["rew"]:
                continue
            next_value = float(last_vals[car]) if active[car] else 0.0
            adv, ret = agent.compute_gae(np.array(b["rew"], dtype=np.float32), np.array(b["val"], dtype=np.float32),
                                         np.array(b["done"], dtype=np.float32), next_value)
            for k in ["obs", "act", "u", "logp", "val"]:
                cols[k].append(np.array(b[k], dtype=np.float32))
            cols["adv"].append(adv)
            cols["ret"].append(ret)
        cat = {k: np.concatenate(v) for k, v in cols.items()}
        samples += int(len(cat["obs"]))
        train_info = agent.update(cat["obs"], cat["act"], cat["logp"], cat["adv"], cat["ret"], cat["val"], b_raw_us=cat["u"])
        update_idx += 1

        # 3. Đánh giá định kỳ bằng nhiều cuộc đua đủ N xe
        if update_idx % args.eval_freq == 0 or samples >= args.total_samples:
            res = evaluate(env, agent, args.eval_races, max_steps=race_steps)
            sps = int(samples / max(1e-5, time.time() - start_time))
            log_eval(samples, res, train_info, train_laps)
            print(f"[Samples {samples:07d}/{args.total_samples} | {sps} mẫu/s | lap khi train: {train_laps}] "
                  f"Eval: {describe(res, args.num_cars)} | Loss P/V {train_info['policy_loss']:.3f}/{train_info['value_loss']:.3f} | "
                  f"ExpVar {train_info['explained_var']:.2f}", flush=True)
            agent.save(os.path.join(args.save_dir, "last_agent.pt"), meta={**meta, "samples": samples})
            if score(res) > best_score:
                best_score = score(res)
                agent.save(os.path.join(args.save_dir, "best_agent.pt"), meta={**meta, "samples": samples})
                print(f"  🏆 [BEST SAVED] về đích {res['finished']:.1f}/{args.num_cars}, lap TB {res['mean_lap']:.1f}s, "
                      f"đi được {res['mean_completion']:.1f}%", flush=True)
            if samples >= next_milestone:
                agent.save(os.path.join(args.save_dir, f"agent_{next_milestone // 1000}k.pt"), meta={**meta, "samples": samples})
                next_milestone += args.milestone_freq
            obs, info = new_train_race()

    env.close()
    print(f">> HOÀN THÀNH: {samples} mẫu, {train_laps} vòng khi train. Best: {os.path.join(args.save_dir, 'best_agent.pt')}")


if __name__ == "__main__":
    main()
