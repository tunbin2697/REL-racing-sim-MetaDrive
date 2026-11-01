"""Đo phong cách lái của 1 checkpoint PPO (hoặc ghost) trong 1 vòng deterministic (F20, F23).

In ra: lap, tốc độ TB, số lần đổi ga<->phanh và đảo lái /100 bước, |Δlái| TB, góc trượt β (v > 40 km/h),
a_lat p95/max và các đoạn có |β| > 5°.

Ví dụ:
  python scripts/analysis/drive_style.py --model models/ref_111s/best_agent.pt --map silverstone --vehicle gt3_anchor --ghost
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from game_env.racing_env import MetaRacingEnv  # noqa: E402
from game_env.vehicle import get_vehicle  # noqa: E402
from agents.ppo_agent import PPOAgent  # noqa: E402
from agents.optimal_agent import OptimalRacingAgent  # noqa: E402


def run(policy: str, args) -> None:
    meta = PPOAgent.peek_meta(args.model) if (policy == "ppo" and args.model) else {}
    car = get_vehicle(args.vehicle or meta.get("vehicle", "gt3"))
    env = MetaRacingEnv(
        use_render=False, map_config=args.map or meta.get("map", "silverstone"), vehicle=car, max_laps=1,
        obs_mode=meta.get("obs_mode", "pure"),
        action_mode=meta.get("action_mode", "split") if policy == "ppo" else "split",
        offtrack_mode=args.offtrack_mode or meta.get("offtrack_mode", "terminate"),
    )
    obs, _ = env.reset(seed=args.seed, options={"start_mode": "standing"})
    if policy == "ppo":
        ag = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_dim, device="cpu")
        ag.load(args.model)
        act = lambda o: ag.select_action(o, deterministic=True)[0]
    else:
        opt = OptimalRacingAgent.from_benchmark(env.get_benchmark_engine(), vehicle_cfg=car.cfg, v_norm=float(env.preset.v_norm))
        opt.reset()
        act = lambda o: opt.select_action(o, vehicle=env.car)[0]

    rec = []
    while True:
        obs, _, te, tr, info = env.step(np.asarray(act(obs), dtype=np.float32))
        v = np.asarray(env.car.velocity, dtype=float)
        h = float(env.car.heading_theta)
        vx = v[0] * np.cos(h) + v[1] * np.sin(h)
        vy = -v[0] * np.sin(h) + v[1] * np.cos(h)
        beta = np.degrees(np.arctan2(vy, max(abs(vx), 1.0)))
        steer = float(env.car.last_action[0]) if hasattr(env.car, "last_action") else 0.0
        rec.append((info["track_s"], info["speed_km_h"], info["net_pedal"], steer, beta, abs(env.last_acc_lat_g)))
        if te or tr or len(rec) > 4000:
            break
    env.close()

    s, spd, ped, steer, beta, alat = np.array(rec).T
    m = spd > 40
    flips_ped = np.sum(np.abs(np.diff(np.sign(np.where(np.abs(ped) < 0.05, 0, ped)))) > 1) / len(ped) * 100
    flips_st = np.sum(np.diff(np.sign(np.diff(steer))) != 0) / len(steer) * 100
    print(f"== {policy}: end={info.get('termination_reason')} lap={info.get('lap_time', 0):.1f}s v_mean={spd.mean():.0f} km/h")
    print(f"   pedal: đổi ga<->phanh {flips_ped:.1f}/100 bước | |pedal|>0.9 {np.mean(np.abs(ped) > 0.9) * 100:.0f}% | phanh {np.mean(ped < -0.05) * 100:.0f}%")
    print(f"   lái: đổi chiều {flips_st:.1f}/100 bước | |Δlái| TB {np.mean(np.abs(np.diff(steer))):.3f}")
    print(f"   β (v>40): TB {np.mean(np.abs(beta[m])):.2f}°, p95 {np.percentile(np.abs(beta[m]), 95):.2f}°, max {np.max(np.abs(beta[m])):.1f}°, %|β|>5° {np.mean(np.abs(beta[m]) > 5) * 100:.1f}%")
    print(f"   a_lat p95 {np.percentile(alat[m], 95):.2f} G, max {alat[m].max():.2f} G")
    big = np.where(np.abs(beta) > 5)[0]
    if len(big):
        segs, st, pv = [], s[big[0]], s[big[0]]
        for i in big[1:]:
            if s[i] - pv > 40:
                segs.append((round(st), round(pv)))
                st = s[i]
            pv = s[i]
        segs.append((round(st), round(pv)))
        print(f"   đoạn |β|>5° (m): {segs[:12]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default=None, help="Checkpoint PPO (bỏ trống nếu chỉ đo ghost)")
    p.add_argument("--map", type=str, default=None, help="Mặc định: metadata checkpoint hoặc 'silverstone'")
    p.add_argument("--vehicle", type=str, default=None, help="Mặc định: metadata checkpoint hoặc 'gt3'")
    p.add_argument("--offtrack-mode", type=str, default=None, choices=["terminate", "penalty"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ghost", action="store_true", help="Đo thêm ghost trên cùng map và xe")
    args = p.parse_args()
    if args.model:
        run("ppo", args)
    if args.ghost or not args.model:
        run("ghost", args)


if __name__ == "__main__":
    main()
