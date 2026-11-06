"""
Kiểm thử hồi quy môi trường đa xe: chạy lại đúng một đoạn đua và so với vết đã ghi trong golden_env_trace.npz.

    python tests/test_env_regression.py             # so sánh với vết đã ghi
    python tests/test_env_regression.py --record    # ghi lại vết sau khi CỐ Ý đổi env, reward hoặc vật lý

Vết ghi obs 60D, reward, vị trí và cờ kết thúc của 6 xe do ref_111s điều khiển trong 250 bước, trên 2 cấu hình:
3 làn so le có hồi sinh, và 2 làn ngang hàng không hồi sinh.
Vật lý Bullet chỉ lặp lại được trên cùng một máy và cùng bản thư viện; khác môi trường thì ghi lại vết trước khi sửa code.
"""

import argparse
import os
import sys

import numpy as np
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from game_env.multicar_env import MultiCarRacingEnv
from agents.ppo_agent import PPOAgent
from training.train_multicar import warm_start_from_single, policy_act

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_env_trace.npz")
CASES = [("l3_stag", 3, "staggered", True), ("l2_abr", 2, "abreast", False)]
STEPS = 250


def run_case(lane: int, grid: str, respawn: bool):
    env = MultiCarRacingEnv(
        num_cars=6, lane_num=lane, grid=grid, respawn=respawn, start_seed=42,
        collision_penalty=4.0, steer_smooth_coef=1.0, pedal_smooth_coef=0.05,
    )
    agent = PPOAgent(state_dim=env.obs_dim, action_dim=3, logstd_max=[0.0, -0.7, -0.7], device="cpu")
    warm_start_from_single(agent, "models/ref_111s/best_agent.pt")
    env.np_random = np.random.default_rng(7)
    obs, _ = env.reset()
    all_obs, rewards, positions, dones = [obs.copy()], [], [], []
    actions = np.zeros((env.num_cars, 3), dtype=np.float32)
    for _ in range(STEPS):
        active = env.active_mask()
        if active.any():
            act, _, _, _ = policy_act(agent, obs[active], True)
            actions[active] = act
        obs, reward, done, _, _ = env.step(actions)
        all_obs.append(obs.copy())
        rewards.append(reward.copy())
        dones.append(done.copy())
        positions.append(np.array([st.car.position for st in env.racers], dtype=np.float32))
    env.close()
    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "rew": np.array(rewards, dtype=np.float32),
        "pos": np.array(positions, dtype=np.float32),
        "done": np.array(dones),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="Ghi lại vết thay vì so sánh")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)
    trace = {}
    for name, lane, grid, respawn in CASES:
        for key, val in run_case(lane, grid, respawn).items():
            trace[f"{name}_{key}"] = val

    if args.record:
        np.savez_compressed(GOLDEN, **trace)
        print(f"Đã ghi {GOLDEN}")
        return

    golden = np.load(GOLDEN)
    failed = [key for key in golden.files if not np.array_equal(golden[key], trace[key])]
    for key in failed:
        print(f"LỆCH {key}: max|Δ| = {np.abs(golden[key].astype(float) - trace[key].astype(float)).max():.6g}")
    if failed:
        sys.exit(1)
    print(f"PASS: {len(golden.files)} mảng khớp tuyệt đối với vết đã ghi")


if __name__ == "__main__":
    main()
