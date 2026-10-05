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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import SingleRacingMetaDrive
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
    print(f">> Đang thu thập {num_steps} bước dữ liệu từ Expert Policy trên RacingMap (12 Blocks Circuit)...")
    env = SingleRacingMetaDrive(dict(
        use_render=False,
        traffic_density=0.0,
        map_config=dict(lane_num=2, lane_width=4.5),
        vehicle_config=dict(lidar=dict(num_lasers=240, distance=50)),
    ))
    
    obs, info = env.reset()
    states = []
    actions = []
    
    for _ in tqdm(range(num_steps), desc="Thu thập expert data"):
        act = expert(env.agent)
        states.append(obs)
        actions.append(np.clip(act, -1.0, 1.0))
        
        obs, reward, term, trunc, info = env.step(act)
        if term or trunc:
            obs, info = env.reset()
            
    env.close()
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    print(f">> Thu thập thành công: {states.shape[0]} mẫu dữ liệu (State shape: {states.shape[1]}).")
    return states, actions

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print("=" * 70)
    print(">> KHOI DONG TIEN HUAN LUYEN TRI THUC CHO AI (BEHAVIOR CLONING WARM START)")
    print("=" * 70)
    
    # 1. Thu thập dữ liệu
    states, actions = collect_expert_data(args.steps)
    
    # 2. Khởi tạo Agent
    agent = PPOAgent(state_dim=259, action_dim=2, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Thiet bi huan luyen: {agent.device}")
    
    # 3. Chuyển đổi dữ liệu sang PyTorch Tensor
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(agent.ac.parameters(), lr=args.lr)
    
    # 4. Huấn luyện mạng Actor (học phán đoán góc lái và chân ga chính xác)
    print(f">> Bat dau huan luyen Behavior Cloning ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for b_states, b_actions in dataloader:
            b_states = b_states.to(agent.device)
            b_actions = b_actions.to(agent.device)
            
            # Dự đoán góc lái và ga từ Actor
            pred_actions = agent.ac.actor_mean(b_states)
            loss = criterion(pred_actions, b_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * b_states.size(0)
            
        epoch_loss /= len(dataset)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [Epoch {epoch+1:03d}/{args.epochs:03d}] MSE Loss: {epoch_loss:.5f}")
            
    # 5. Lưu trọng số
    agent.save(args.save_path)
    print("=" * 70)
    print(f">> [THANH CONG] Da luu model khoi tao tri thuc tai: {args.save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
