"""
Pretrain Actor-Critic qua Học Bắt Chước (Imitation Learning / Behavior Cloning)
Thu thập quỹ đạo mẫu chuẩn từ MetaDrive Expert Policy để khởi tạo tri thức ban đầu cho Agent
giúp Agent biết bám làn, vào cua, điều tiết chân ga và phanh trước khi tiếp tục học tăng cường (RL Fine-tuning).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from metadrive.examples.ppo_expert.numpy_expert import expert
from agents.ppo_agent import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain RL Agent using Expert Demonstrations on RacingMap")
    parser.add_argument("--steps", type=int, default=6000, help="Số bước thu thập từ chuyên gia")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch huấn luyện Behavior Cloning")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học")
    parser.add_argument("--save-path", type=str, default="models/warmup_racing_agent.pt", help="File lưu checkpoint")
    return parser.parse_args()

def collect_expert_data(num_steps):
    print(f">> Đang thu thập {num_steps} bước dữ liệu từ Expert Policy trên MetaRacingEnv...")
    env = MetaRacingEnv(use_render=False)
    
    obs, info = env.reset()
    states = []
    actions = []
    rewards = []
    dones = []
    
    for _ in tqdm(range(num_steps), desc="Thu thập expert data"):
        act_raw = expert(env.vehicle)
        act: np.ndarray = np.asarray(act_raw[0] if isinstance(act_raw, tuple) else act_raw, dtype=np.float32)
        act_clipped = np.clip(act, -1.0, 1.0)
        
        states.append(obs)
        actions.append(act_clipped)
        
        obs, reward, term, trunc, info = env.step(act_clipped)
        rewards.append(reward)
        dones.append(term or trunc)
        
        if term or trunc:
            obs, info = env.reset()
            
    env.close()
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=bool)
    
    # Tính toán Monte-Carlo discounted returns để khởi tạo giá trị Critic V(s)
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    gamma = 0.99
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
        
    print(f">> Thu thập thành công: {states.shape[0]} mẫu dữ liệu (State shape: {states.shape[1]}).")
    return states, actions, returns

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print("=" * 70)
    print(">> KHOI DONG TIEN HUAN LUYEN TRI THUC CHO AI (ACTOR & CRITIC WARM START)")
    print("=" * 70)
    
    # 1. Thu thập dữ liệu
    states, actions, returns = collect_expert_data(args.steps)
    
    # 2. Khởi tạo Agent
    agent = PPOAgent(state_dim=259, action_dim=2, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Thiet bi huan luyen: {agent.device}")
    
    # 3. Chuyển đổi dữ liệu sang PyTorch Tensor
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(returns, dtype=torch.float32),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(agent.ac.parameters(), lr=args.lr)
    
    # 4. Huấn luyện đồng thời cả Actor (hành động) và Critic (giá trị trạng thái)
    print(f">> Bat dau huan luyen Behavior Cloning & Value Fitting ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        for b_states, b_actions, b_returns in dataloader:
            b_states = b_states.to(agent.device)
            b_actions = b_actions.to(agent.device)
            b_returns = b_returns.to(agent.device)
            
            pred_actions = agent.ac.actor_mean(b_states)
            pred_values = agent.ac.critic(b_states).squeeze(-1)
            
            actor_loss = criterion(pred_actions, b_actions)
            critic_loss = criterion(pred_values, b_returns)
            total_loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_actor_loss += actor_loss.item() * b_states.size(0)
            epoch_critic_loss += critic_loss.item() * b_states.size(0)
            
        epoch_actor_loss /= len(dataset)
        epoch_critic_loss /= len(dataset)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [Epoch {epoch+1:03d}/{args.epochs:03d}] Actor Loss: {epoch_actor_loss:.5f} | Critic Loss: {epoch_critic_loss:.5f}")
            
    # Hiệu chỉnh độ lệch chuẩn khám phá an toàn
    with torch.no_grad():
        agent.ac.actor_logstd.fill_(-1.6)
        
    # 5. Lưu trọng số
    agent.save(args.save_path)
    best_path = os.path.join(os.path.dirname(args.save_path), "best_agent.pt")
    agent.save(best_path)
    print("=" * 70)
    print(f">> [THANH CONG] Da luu model khoi tao tri thuc tai: {args.save_path} & {best_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
