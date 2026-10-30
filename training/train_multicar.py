"""
Huấn luyện PPO Self-Play Đa Tác Tử (Parameter Sharing) trên MultiCarRacingEnv
- Một mạng PPO dùng chung cho cả N xe: mỗi bước mô phỏng thu được tối đa N mẫu.
- Mỗi xe có chuỗi rollout riêng để tính GAE đúng ranh giới episode của chính nó.
- Warm start từ checkpoint đơn tác tử (40D GT3 / 52D F1): copy các cột cũ, cột quan sát đối thủ khởi tạo 0
  -> ban đầu hành vi y hệt agent đơn, sau đó học thêm né / vượt xe khác.
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

from game_env.multicar_env import MultiCarRacingEnv
from agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-car PPO self-play racing (MetaDrive 3D)")
    parser.add_argument("--total-samples", type=int, default=1500000, help="Tổng số mẫu (cộng dồn mọi xe)")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Số bước mô phỏng mỗi rollout (x số xe = số mẫu)")
    parser.add_argument("--num-cars", type=int, default=6)
    parser.add_argument("--lane-num", type=int, default=3, help="Số làn (3 làn x 4.5m = 13.5m)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=2.0)
    parser.add_argument("--stall-penalty", type=float, default=0.0, help="Phạt khi xe đứng yên (chống mẹo dừng trước cua khó)")
    parser.add_argument("--laps", type=int, default=1, help="Số vòng mỗi cuộc đua (train + eval)")
    parser.add_argument("--respawn", choices=["random", "inplace"], default="random",
                        help="Khi train: random = đặt lại ở chỗ trống ngẫu nhiên; inplace = đứng ngoài vài giây rồi quay lại chỗ văng")
    parser.add_argument("--respawn-delay", type=float, default=3.0, help="Thời gian phạt (s) trước khi xe hồi sinh tại chỗ")
    parser.add_argument("--full-race-prob", type=float, default=0.0,
                        help="Với --race-reset: xác suất một cuộc đua train chạy trọn (giữ thưởng về đích)")
    parser.add_argument("--race-reset", type=float, nargs=2, default=None, metavar=("MIN_S", "MAX_S"),
                        help="Khi train: cả cuộc đua bắt đầu lại từ vạch xuất phát sau ngẫu nhiên MIN_S..MAX_S giây (mặc định: hết cuộc đua)")
    parser.add_argument("--grid", choices=["abreast", "staggered"], default="abreast", help="Kiểu xếp xe xuất phát")
    parser.add_argument("--vehicle", choices=["gt3", "f1", "sedan"], default="gt3", help="Loại xe (f1: trần 340 km/h)")
    parser.add_argument("--init-model", type=str, default="models/best_agent.pt", help="Checkpoint đơn tác tử (40D / 52D) để warm start")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint đa tác tử để train tiếp")
    parser.add_argument("--eval-races", type=int, default=0,
                        help="Số cuộc đua ngẫu nhiên thêm vào mỗi lần eval (ngoài 1 cuộc tất định) để chọn best ít nhiễu hơn")
    parser.add_argument("--target-kl", type=float, default=0.02, help="Dừng sớm epoch PPO khi KL > 1.5 * target_kl (<= 0 để tắt)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Đánh giá sau mỗi N lần update")
    parser.add_argument("--milestone-freq", type=int, default=250000, help="Lưu snapshot mỗi N mẫu")
    parser.add_argument("--save-dir", type=str, default="models/multicar")
    parser.add_argument("--log-dir", type=str, default="logs/multicar")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def policy_act(agent: PPOAgent, obs: np.ndarray, deterministic: bool):
    """Chọn hành động theo batch cho các xe đang chạy"""
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        action, log_prob, _, value = agent.ac.get_action_and_value(x, deterministic=deterministic)
    return action.cpu().numpy(), log_prob.cpu().numpy(), value.view(-1).cpu().numpy()


def run_race(env: MultiCarRacingEnv, agent: PPOAgent, deterministic: bool = True, respawn=False,
             max_steps: int = None) -> Dict[str, Any]:
    """
    Chạy 1 cuộc đua đầy đủ (env.max_laps vòng), tất cả các xe do policy điều khiển.
    respawn=False: xe bị loại rời cuộc đua; "inplace": xe văng đứng ngoài vài giây rồi quay lại chỗ văng.
    """
    train_respawn, train_max_steps = env.respawn, env.max_steps
    env.respawn = respawn
    if max_steps is not None:
        env.max_steps = int(max_steps)
    obs, info = env.reset()
    actions = np.zeros((env.num_cars, 3), dtype=np.float32)
    while True:
        active = env.active_mask()
        if active.any():
            act, _, _ = policy_act(agent, obs[active], deterministic)
            actions[active] = act
        obs, _, _, truncated, _ = env.step(actions)
        if truncated or env.race_over():
            break
    env.respawn, env.max_steps = train_respawn, train_max_steps
    racers = env.racers
    laps = [t for st in racers for t in st.lap_times]
    race_times = [sum(st.lap_times) for st in racers if st.finished]
    return {
        "finished": int(sum(st.finished for st in racers)),
        "outs": int(sum(st.crashes for st in racers)),
        "stalls": int(sum(st.stalls for st in racers)),
        "respawns": int(sum(st.respawns for st in racers)),
        "mean_race_time": float(np.mean(race_times)) if race_times else 0.0,
        "contacts": int(sum(st.contacts for st in racers)),
        "mean_completion": float(np.mean([min(1.0, max(0.0, env.race_distance(st)) / (env.circuit_length * env.max_laps))
                                          for st in racers]) * 100.0),
        "mean_lap": float(np.mean(laps)) if laps else 0.0,
        "best_lap": float(np.min(laps)) if laps else 0.0,
        "standings": env.standings(),
    }


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Horizon đủ cho cả cuộc đua eval (tối đa ~200s/vòng kể cả thời gian phạt hồi sinh)
    race_steps = 2000 * args.laps
    env = MultiCarRacingEnv(num_cars=args.num_cars, lane_num=args.lane_num, time_penalty=args.time_penalty,
                            collision_penalty=args.collision_penalty, start_seed=args.seed, respawn=args.respawn,
                            respawn_delay=args.respawn_delay, grid=args.grid, vehicle=args.vehicle,
                            stall_penalty=args.stall_penalty, max_laps=args.laps, max_steps=race_steps)
    # Eval: cùng luật hồi sinh tại chỗ nếu train inplace (mọi xe đều về đích -> so tổng thời gian đua)
    eval_respawn = "inplace" if args.respawn == "inplace" else False

    def new_train_race():
        """Bắt đầu lại cả cuộc đua từ vạch xuất phát; độ dài ngẫu nhiên để luyện nhiều pha chạy thành cụm"""
        if args.race_reset:
            if np.random.random() < args.full_race_prob:
                env.max_steps = race_steps
            else:
                lo, hi = args.race_reset
                env.max_steps = int(np.random.uniform(lo, hi) / env.dt)
        return env.reset()
    obs_dim = int(env.obs_dim)
    agent = PPOAgent(state_dim=obs_dim, action_dim=3, lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                     clip_coef=args.clip_coef, ent_coef=args.ent_coef, minibatch_size=args.batch_size,
                     target_kl=args.target_kl if args.target_kl > 0 else None)

    if args.resume:
        agent.load(args.resume)
        agent.set_lr(args.lr)
        print(f">> RESUME từ checkpoint đa tác tử: {args.resume}")
    elif args.init_model and os.path.exists(args.init_model):
        old_dim = agent.load_widened(args.init_model)
        print(f">> WARM START từ checkpoint đơn tác tử: {args.init_model} ({old_dim}D -> {obs_dim}D)")
    else:
        print(">> Huấn luyện từ đầu (không có checkpoint khởi tạo)")

    print("=" * 70)
    print(f">> MULTI-CAR PPO SELF-PLAY | {args.num_cars} xe | {args.lane_num} làn ({env.half_width * 2:.1f}m) | obs {obs_dim}D | device {agent.device}")
    print(f"   lr {args.lr} | gamma {args.gamma} | lambda {args.gae_lambda} | ent {args.ent_coef} | time {args.time_penalty} | collision {args.collision_penalty}")
    reset_txt = (f"{args.race_reset[0]:.0f}-{args.race_reset[1]:.0f}s (trọn cuộc đua {args.full_race_prob:.0%})"
                 if args.race_reset else "hết cuộc đua")
    print(f"   {args.laps} vòng | respawn {args.respawn} ({args.respawn_delay:.0f}s) | stall {args.stall_penalty} | race reset {reset_txt} | grid {args.grid}")
    print(f"   eval 1 tất định + {args.eval_races} ngẫu nhiên / {args.eval_freq} update | target KL {args.target_kl}")
    print("=" * 70)

    metrics: Dict[str, List] = {k: [] for k in [
        "samples", "eval_finished", "eval_outs", "eval_stalls", "eval_contacts", "eval_completion",
        "eval_mean_lap", "eval_best_lap", "eval_race_time", "eval_respawns", "policy_loss", "value_loss", "entropy", "explained_var", "train_laps"]}

    def log_eval(samples: int, res: Dict[str, Any], train_info: Dict[str, float], train_laps: int):
        for key, val in [("samples", samples), ("eval_finished", res["finished"]), ("eval_outs", res["outs"]),
                         ("eval_stalls", res["stalls"]), ("eval_contacts", res["contacts"]), ("eval_completion", res["mean_completion"]),
                         ("eval_mean_lap", res["mean_lap"]), ("eval_best_lap", res["best_lap"]),
                         ("eval_race_time", res["mean_race_time"]), ("eval_respawns", res["respawns"]),
                         ("policy_loss", train_info.get("policy_loss", 0.0)), ("value_loss", train_info.get("value_loss", 0.0)),
                         ("entropy", train_info.get("entropy", 0.0)), ("explained_var", train_info.get("explained_var", 0.0)),
                         ("train_laps", train_laps)]:
            metrics[key].append(val)
        with open(os.path.join(args.log_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def score(res: Dict[str, Any]):
        # Ưu tiên số xe về đích, sau đó tổng thời gian đua TB (đã gồm thời gian phạt hồi sinh), sau đó quãng đường
        return (res["finished"], -res["mean_race_time"] if res["mean_race_time"] > 0 else -1e9, res["mean_completion"])

    def eval_race():
        """1 cuộc đua tất định + eval_races cuộc ngẫu nhiên, lấy trung bình (1 cuộc duy nhất quá nhiễu để chọn best)"""
        races = [run_race(env, agent, deterministic=True, respawn=eval_respawn, max_steps=race_steps)]
        races += [run_race(env, agent, deterministic=False, respawn=eval_respawn, max_steps=race_steps)
                  for _ in range(args.eval_races)]
        if len(races) == 1:
            return races[0]
        agg = {k: float(np.mean([r[k] for r in races]))
               for k in ("finished", "outs", "stalls", "respawns", "contacts", "mean_completion")}
        timed = [r for r in races if r["mean_race_time"] > 0]
        agg["mean_race_time"] = float(np.mean([r["mean_race_time"] for r in timed])) if timed else 0.0
        agg["mean_lap"] = float(np.mean([r["mean_lap"] for r in timed])) if timed else 0.0
        agg["best_lap"] = float(min(r["best_lap"] for r in timed)) if timed else 0.0
        return agg

    def describe(res):
        return (f"về đích {res['finished']:.1f}/{args.num_cars} (TB {res['mean_race_time']:.1f}s) | văng {res['outs']:.1f} | dừng {res['stalls']:.1f} | "
                f"hồi sinh {res['respawns']:.1f} | chạm {res['contacts']:.0f} | đi được {res['mean_completion']:.1f}% | "
                f"lap TB {res['mean_lap']:.1f}s (best {res['best_lap']:.1f}s)")

    res = eval_race()
    best_score = score(res)
    print(f"[Samples 0] Eval: {describe(res)}")
    log_eval(0, res, {}, 0)
    agent.save(os.path.join(args.save_dir, "best_agent.pt"))

    samples = 0
    update_idx = 0
    train_laps = 0
    next_milestone = args.milestone_freq
    start_time = time.time()
    obs, info = new_train_race()
    actions = np.zeros((args.num_cars, 3), dtype=np.float32)

    while samples < args.total_samples:
        # 1. Thu thập rollout: mỗi xe một chuỗi riêng
        buf = [{"obs": [], "act": [], "logp": [], "rew": [], "val": [], "done": []} for _ in range(args.num_cars)]
        for _ in range(args.rollout_steps):
            active = env.active_mask()
            idx = np.flatnonzero(active)
            act, logp, val = policy_act(agent, obs[idx], deterministic=False)
            actions[idx] = act
            next_obs, rewards, dones, truncated, infos = env.step(actions)
            if truncated and args.race_reset:
                # Cắt cuộc đua giữa chừng không phải trạng thái kết thúc thật: bootstrap r += gamma * V(s')
                alive = [car for car in idx if not dones[car]]
                if alive:
                    _, _, boot = policy_act(agent, next_obs[alive], deterministic=True)
                    rewards = rewards.copy()
                    rewards[alive] += agent.gamma * boot
            for j, car in enumerate(idx):
                b = buf[car]
                b["obs"].append(obs[car])
                b["act"].append(act[j])
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
        _, _, last_vals = policy_act(agent, obs, deterministic=True)
        all_s, all_a, all_lp, all_adv, all_ret, all_v = [], [], [], [], [], []
        for car in range(args.num_cars):
            b = buf[car]
            if not b["rew"]:
                continue
            next_value = float(last_vals[car]) if active[car] else 0.0
            adv, ret = agent.compute_gae(np.array(b["rew"], dtype=np.float32), np.array(b["val"], dtype=np.float32),
                                         np.array(b["done"], dtype=np.float32), next_value)
            all_s.append(np.array(b["obs"], dtype=np.float32))
            all_a.append(np.array(b["act"], dtype=np.float32))
            all_lp.append(np.array(b["logp"], dtype=np.float32))
            all_adv.append(adv)
            all_ret.append(ret)
            all_v.append(np.array(b["val"], dtype=np.float32))
        batch_n = int(sum(len(x) for x in all_s))
        samples += batch_n
        train_info = agent.update(np.concatenate(all_s), np.concatenate(all_a), np.concatenate(all_lp),
                                  np.concatenate(all_adv), np.concatenate(all_ret), np.concatenate(all_v))
        update_idx += 1

        # 3. Đánh giá định kỳ bằng 1 cuộc đua tất định đủ 6 xe
        if update_idx % args.eval_freq == 0 or samples >= args.total_samples:
            res = eval_race()
            sps = int(samples / max(1e-5, time.time() - start_time))
            log_eval(samples, res, train_info, train_laps)
            print(f"[Samples {samples:07d}/{args.total_samples} | {sps} mẫu/s | lap khi train: {train_laps}] "
                  f"Eval: {describe(res)} | "
                  f"Loss P/V {train_info['policy_loss']:.3f}/{train_info['value_loss']:.3f} | ExpVar {train_info['explained_var']:.2f} | "
                  f"KL {train_info.get('approx_kl', 0.0):.4f} ({train_info.get('epochs', 0)} ep)")
            agent.save(os.path.join(args.save_dir, "last_agent.pt"))
            if score(res) > best_score:
                best_score = score(res)
                agent.save(os.path.join(args.save_dir, "best_agent.pt"))
                print(f"  🏆 [BEST SAVED] về đích {res['finished']:.1f}/{args.num_cars}, đua TB {res['mean_race_time']:.1f}s, "
                      f"văng {res['outs']:.1f}, đi được {res['mean_completion']:.1f}%")
            if samples >= next_milestone:
                agent.save(os.path.join(args.save_dir, f"agent_{next_milestone // 1000}k.pt"))
                next_milestone += args.milestone_freq
            obs, info = new_train_race()

    env.close()
    print(f">> HOÀN THÀNH: {samples} mẫu, {train_laps} vòng khi train. Best: {os.path.join(args.save_dir, 'best_agent.pt')}")


if __name__ == "__main__":
    main()
