"""
Pretrain PPO v2 bằng Behavior Cloning từ Policy luật (RuleRacingPolicy):
  1. Thu thập: xe chạy theo hành động expert + nhiễu Gaussian (và thỉnh thoảng nhồi ga/lệch lái) để phủ
     các trạng thái lệch quỹ đạo; NHÃN luôn là hành động sạch của expert tại trạng thái đó (DAgger-style).
  2. Actor: hồi quy tanh(mu(s)) về hành động expert (MSE). Critic: hồi quy về return Monte Carlo (gamma).
  3. Lưu checkpoint "arch": "v2" để train tiếp bằng training/train_racing_v2.py --warmup-model.

Cách chạy:
    python training/pretrain_bc_v2.py --steps 40000 --save-path models/v2/bc_warmup.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from agents.rule_policy import RuleRacingPolicy
from agents.ppo_agent_v2 import PPOAgentV2


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior Cloning warmup cho PPO v2 từ Policy luật")
    parser.add_argument("--steps", type=int, default=40000, help="Số bước thu thập dữ liệu")
    parser.add_argument("--noise", type=float, default=0.25, help="Độ lệch chuẩn nhiễu hành động khi thu thập")
    parser.add_argument("--perturb-prob", type=float, default=0.02, help="Xác suất bắt đầu một đợt nhiễu mạnh (nhồi ga / lệch lái)")
    parser.add_argument("--epochs", type=int, default=40, help="Số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-logstd", type=str, default="-3.5,-1.2", help="log_std khởi đầu cho pha RL theo từng hành động 'steer,throttle' (σ ~ 0.03, 0.30)")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="models/v2/bc_warmup.pt")
    return parser.parse_args()


def discounted_returns(rewards, dones, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    return returns


def collect(env, expert, num_steps, noise, perturb_prob, rng):
    states, labels, rewards, dones = [], [], [], []
    reasons = {}
    obs, _ = env.reset(options={"force_start": True})
    perturb_left, perturb_action = 0, np.zeros(2, dtype=np.float32)
    for _ in tqdm(range(num_steps), desc="Thu thập expert data"):
        label = expert.act(obs)
        action = label + rng.normal(0.0, noise, size=2).astype(np.float32)
        if perturb_left == 0 and rng.random() < perturb_prob:
            perturb_left = int(rng.integers(5, 20))
            perturb_action = np.array([rng.uniform(-0.6, 0.6), rng.uniform(0.5, 1.0)], dtype=np.float32)
        if perturb_left > 0:
            action = perturb_action
            perturb_left -= 1
        next_obs, reward, terminated, truncated, info = env.step(np.clip(action, -1.0, 1.0))
        done = bool(terminated or truncated)
        states.append(obs)
        labels.append(label)
        rewards.append(reward)
        dones.append(1.0 if done else 0.0)
        obs = next_obs
        if done:
            reason = info.get("termination_reason", "truncated")
            reasons[reason] = reasons.get(reason, 0) + 1
            obs, _ = env.reset(options={"force_start": True})
            perturb_left = 0
    dones[-1] = 1.0
    print(f">> Kết thúc episode khi thu thập: {reasons}")
    return np.array(states), np.array(labels), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32)


def evaluate(env, agent, episodes):
    for ep in range(episodes):
        obs, _ = env.reset(options={"force_start": True})
        while True:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print(f"  - Eval {ep + 1}: {info.get('termination_reason')} tại {info['track_s']:.0f}m | lap {info['lap_time']:.1f}s")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = MetaRacingEnv(use_render=False, start_seed=args.seed)
    expert = RuleRacingPolicy()
    states, labels, rewards, dones = collect(env, expert, args.steps, args.noise, args.perturb_prob, rng)
    returns = discounted_returns(rewards, dones, args.gamma)
    print(f">> Dataset: {len(states)} mẫu | return MC: mean {returns.mean():.2f}, std {returns.std():.2f}")

    agent = PPOAgentV2(state_dim=states.shape[1], action_dim=labels.shape[1])
    init_logstd = [float(v) for v in args.init_logstd.split(",")]
    agent.ac.actor_logstd.data.copy_(torch.tensor([init_logstd], dtype=torch.float32).expand_as(agent.ac.actor_logstd))
    device = agent.device
    X = torch.as_tensor(states, dtype=torch.float32, device=device)
    Y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    R = torch.as_tensor(returns, dtype=torch.float32, device=device)

    actor_params = list(agent.ac.actor_trunk.parameters()) + list(agent.ac.actor_mean_head.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=args.lr)
    critic_opt = torch.optim.Adam(agent.critic_params, lr=args.lr)

    n = len(X)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        a_loss_sum, v_loss_sum, batches = 0.0, 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            u_mean, _ = agent.ac.actor_forward(X[idx])
            a_loss = F.mse_loss(torch.tanh(u_mean), Y[idx])
            actor_opt.zero_grad()
            a_loss.backward()
            actor_opt.step()

            v_loss = F.mse_loss(agent.ac.get_value(X[idx]).view(-1), R[idx])
            critic_opt.zero_grad()
            v_loss.backward()
            critic_opt.step()
            a_loss_sum += a_loss.item()
            v_loss_sum += v_loss.item()
            batches += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch + 1:03d}/{args.epochs} | Actor MSE: {a_loss_sum / batches:.5f} | Critic MSE: {v_loss_sum / batches:.3f}")

    agent.save(args.save_path)
    print(f">> Đã lưu checkpoint BC v2: {args.save_path}")
    print(">> Đánh giá BC policy (deterministic):")
    evaluate(env, agent, args.eval_episodes)
    env.close()


if __name__ == "__main__":
    main()
