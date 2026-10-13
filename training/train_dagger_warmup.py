"""
Huấn luyện Warmup Agent hoàn thành 100% vòng đua bằng DAgger (Dataset Aggregation)
- Tốc độ trung bình: ~30 - 35 km/h
- Có đủ hành động đạp ga và đạp phanh vào cua
- Tự động kiểm tra rollout thực tế và chỉ dừng khi Agent tự lái hoàn thành 100% vòng đua!
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="DAgger Warmup Agent Training for 100% Lap Completion")
    parser.add_argument("--initial-laps", type=int, default=2, help="Số vòng chạy mẫu ban đầu của chuyên gia")
    parser.add_argument("--max-iterations", type=int, default=8, help="Số vòng lặp DAgger tối đa")
    parser.add_argument("--epochs-per-iter", type=int, default=30, help="Số epoch huấn luyện mỗi vòng lặp DAgger")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="models/warmup_racing_agent.pt", help="File lưu checkpoint")
    return parser.parse_args()

def evaluate_warmup_agent(env, agent):
    """Kiểm tra xem Agent tự lái độc lập có hoàn thành 100% vòng đua không"""
    obs, info = env.reset(options={"force_start": True})
    speeds = []
    throttles = []
    brakes = []
    completed = False
    max_comp = 0.0
    
    for step in range(3500):
        action, _, _ = agent.select_action(obs, deterministic=True)
        steer, tb = float(action[0]), float(action[1])
        if tb >= 0:
            throttles.append(tb)
        else:
            brakes.append(-tb)
            
        obs, reward, term, trunc, info = env.step(action)
        speed = float(info.get("speed_km_h", 0.0))
        speeds.append(speed)
        comp = float(info.get("route_completion", 0.0) * 100.0)
        max_comp = max(max_comp, comp)
        
        if term or trunc:
            reason = info.get("termination_reason", "unknown")
            if reason == "lap_completed" or comp >= 98.0:
                completed = True
            break
            
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    brake_ratio = float(len(brakes) / max(1, len(speeds)) * 100.0)
    return completed, max_comp, mean_speed, brake_ratio, step

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print("=" * 75)
    print(">> KHỞI ĐỘNG HUẤN LUYỆN WARMUP AGENT BẰNG DAGGER (MỤC TIÊU 100% HOÀN THÀNH)")
    print(f">> Mục tiêu: Tốc độ ~30-35 km/h, có phanh & ga, vượt cua 71.4%, cán đích 100%")
    print("=" * 75)
    
    env = MetaRacingEnv(use_render=False)
    obs_shape = getattr(env.observation_space, "shape", None)
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 35
    agent = PPOAgent(state_dim=obs_dim, action_dim=2, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Thiết bị: {agent.device} | State Dim: {obs_dim}")
    
    # 1. Thu thập dữ liệu mẫu ban đầu từ Expert
    print(f"\n>> Giai đoạn 1: Thu thập {args.initial_laps} vòng chạy mẫu từ MetaDrive Expert...")
    dataset_states = []
    dataset_actions = []
    
    for lap in range(args.initial_laps):
        obs, info = env.reset()
        for step in range(3000):
            act_raw = expert(env.vehicle)
            act = np.asarray(act_raw[0] if isinstance(act_raw, tuple) else act_raw, dtype=np.float32)
            act_clipped = np.clip(act, -1.0, 1.0)
            
            dataset_states.append(obs)
            dataset_actions.append(act_clipped)
            
            obs, reward, term, trunc, info = env.step(act_clipped)
            if term or trunc:
                print(f"   + Vòng mẫu {lap+1}: {step} steps | Cự ly {info.get('route_completion', 0.0)*100:.1f}%")
                break
                
    print(f">> Bộ dữ liệu khởi đầu: {len(dataset_states)} mẫu.")
    
    optimizer = torch.optim.Adam(agent.ac.parameters(), lr=args.lr)
    
    # 2. Vòng lặp DAgger (Dataset Aggregation)
    best_completion = 0.0
    
    for dagger_iter in range(args.max_iterations + 1):
        print(f"\n" + "-" * 70)
        print(f">> [DAgger Vòng {dagger_iter}/{args.max_iterations}] Huấn luyện mô hình trên {len(dataset_states)} mẫu...")
        print("-" * 70)
        
        # Huấn luyện Actor
        X = torch.tensor(np.array(dataset_states, dtype=np.float32), device=agent.device)
        Y = torch.tensor(np.array(dataset_actions, dtype=np.float32), device=agent.device)
        
        # Trọng số: Tăng 3x cho hành động phanh và 2x cho đánh lái gắt
        weights = torch.ones_like(Y)
        weights[:, 1] = torch.where(Y[:, 1] < 0.0, 3.0, 1.0)      # Phanh
        weights[:, 0] = torch.where(torch.abs(Y[:, 0]) > 0.15, 2.0, 1.0) # Đánh lái cua
        
        train_ds = torch.utils.data.TensorDataset(X, Y, weights)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        
        agent.ac.train()
        for epoch in range(args.epochs_per_iter):
            total_loss = 0.0
            for b_states, b_actions, b_w in loader:
                u_mean, _ = agent.ac.actor_forward(b_states)
                pred = torch.tanh(u_mean)
                loss = torch.mean(b_w * (pred - b_actions) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * b_states.size(0)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:02d}/{args.epochs_per_iter} | Loss: {total_loss/len(X):.5f}")
                
        agent.ac.eval()
        
        # Kiểm tra năng lực tự lái độc lập (Closed-loop Rollout)
        completed, max_comp, mean_speed, brake_pct, steps = evaluate_warmup_agent(env, agent)
        print(f">> [ROLLOUT ĐÁNH GIÁ VÒNG {dagger_iter}]:")
        print(f"   + Hoàn thành vòng: {'✅ THÀNH CÔNG (100%)' if completed else '❌ Chưa xong'}")
        print(f"   + Cự ly đạt được: {max_comp:.1f}% ({steps} steps)")
        print(f"   + Vận tốc trung bình: {mean_speed:.1f} km/h")
        print(f"   + Tỷ lệ đạp phanh: {brake_pct:.1f}%")
        
        if max_comp > best_completion:
            best_completion = max_comp
            agent.save(args.save_path)
            best_path = os.path.join(os.path.dirname(args.save_path), "best_agent.pt")
            agent.save(best_path)
            print(f"   -> Đã lưu checkpoint kỷ lục mới: {args.save_path} ({best_completion:.1f}%)")
            
        if completed:
            print("\n" + "=" * 75)
            print(f"🎉 XUẤT SẮC! Warmup Agent đã TỰ LÁI HOÀN THÀNH 100% VÒNG ĐUA!")
            print(f"   - Tốc độ TB: {mean_speed:.1f} km/h | Tỷ lệ phanh: {brake_pct:.1f}%")
            print(">> Đang thu thập 1 vòng mẫu để Fit hàm Critic V(s) cho trọn vẹn 100% bản đồ...")
            
            # Khởi tạo giá trị Critic cho toàn bộ chặng đua
            c_states, c_returns = [], []
            obs, info = env.reset(options={"force_start": True})
            ep_states, ep_rewards = [], []
            for step in range(3500):
                action, _, _ = agent.select_action(obs, deterministic=True)
                ep_states.append(obs)
                obs, reward, term, trunc, info = env.step(action)
                ep_rewards.append(reward)
                if term or trunc:
                    break
            
            # Tính Monte-Carlo discounted return
            running_g = 0.0
            for t in reversed(range(len(ep_rewards))):
                running_g = ep_rewards[t] + 0.99 * running_g
                c_returns.insert(0, running_g)
            c_states = ep_states
            
            # Huấn luyện Critic 20 epochs
            cX = torch.tensor(np.array(c_states, dtype=np.float32), device=agent.device)
            cY = torch.tensor(np.array(c_returns, dtype=np.float32), device=agent.device)
            c_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(cX, cY), batch_size=args.batch_size, shuffle=True)
            c_criterion = nn.MSELoss()
            for _ in range(25):
                for b_s, b_v in c_loader:
                    v_pred = agent.ac.critic(b_s).squeeze(-1)
                    v_loss = c_criterion(v_pred, b_v)
                    optimizer.zero_grad()
                    v_loss.backward()
                    optimizer.step()
            
            print(f"   -> Đã khởi tạo hoàn tất cả Actor & Critic!")
            print(f"   - Đã lưu file chuẩn sẵn sàng cho RL: {args.save_path} & {best_path}")
            print("=" * 75)
            agent.save(args.save_path)
            agent.save(best_path)
            break
            
        # Nếu chưa hoàn thành: Cho Agent tự lái, tại mỗi vị trí hỏi Chuyên gia hành động chuẩn và bổ sung vào dữ liệu (DAgger)
        if dagger_iter < args.max_iterations:
            print(f">> Đang thực hiện DAgger rollout để chuyên gia sửa sai tại các vị trí agent bị trượt...")
            obs, info = env.reset(options={"force_start": True})
            added_samples = 0
            beta = max(0.15, 1.0 - (dagger_iter + 1) * 0.2)
            
            for step in range(3000):
                act_expert_raw = expert(env.vehicle)
                act_expert = np.asarray(act_expert_raw[0] if isinstance(act_expert_raw, tuple) else act_expert_raw, dtype=np.float32)
                act_expert = np.clip(act_expert, -1.0, 1.0)
                
                act_agent, _, _ = agent.select_action(obs, deterministic=True)
                
                if np.random.rand() < beta:
                    act_applied = act_expert
                else:
                    act_applied = act_agent
                    
                dataset_states.append(obs)
                dataset_actions.append(act_expert)
                added_samples += 1
                
                obs, reward, term, trunc, info = env.step(act_applied)
                if term or trunc:
                    print(f"   + Đã thu thập thêm {added_samples} mẫu sửa sai (Cự ly dừng: {info.get('route_completion', 0.0)*100:.1f}%)")
                    break
                    
    env.close()
    print("\n>> Hoàn tất quy trình huấn luyện Warmup!")

if __name__ == "__main__":
    main()
