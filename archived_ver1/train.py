"""
train.py: Kịch bản Huấn luyện PPO Continuous Actor-Critic trên các đường đua ngẫu nhiên (Procedural Tracks).
Lưu mô hình đã học tại: models/f1_racing_agent.pt.
"""

import os
import sys
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core import RacingEnv, PPOAgent, CircuitProfiler

def train_agent(num_episodes=40, steps_per_update=512, weights_path="models/f1_racing_agent.pt"):
    """Huấn luyện Agent trên đường đua ngẫu nhiên để có tính khái quát hóa cao."""
    print("=" * 70)
    print(" HUẤN LUYỆN CONTINUOUS ACTOR-CRITIC (PPO) TRÊN PROCEDURAL TRACKS")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    env = RacingEnv(track_name="random")
    agent = PPOAgent(state_dim=16, action_dim=2, lr=3e-4)
    
    total_steps = 0
    episode_rewards = []
    
    for ep in range(1, num_episodes + 1):
        # Đổi hạt giống ngẫu nhiên mỗi 5 episode để tăng tính khái quát hóa
        if ep % 5 == 0:
            env.load_track("random")
            
        state = env.reset()
        ep_reward = 0.0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, done, value)
            state = next_state
            ep_reward += reward
            total_steps += 1
            
            if len(agent.states) >= steps_per_update:
                _, _, next_val = agent.select_action(next_state)
                agent.update(next_val)
                
        episode_rewards.append(ep_reward)
        if ep % 5 == 0 or ep == num_episodes:
            mean_r = np.mean(episode_rewards[-5:])
            print(f"[Episode {ep:02d}/{num_episodes:02d}] Total Steps: {total_steps:5d} | Mean Reward (last 5): {mean_r:7.1f}")
            
    agent.save(weights_path)
    print(f"\n[THÀNH CÔNG] Đã lưu trọng số mô hình PPO tại: {weights_path}")
    return agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Huấn luyện PPO Continuous Actor-Critic trên procedural tracks.")
    parser.add_argument("--episodes", type=int, default=40, help="Số lượng episodes huấn luyện (mặc định: 40)")
    parser.add_argument("--weights_path", type=str, default="models/f1_racing_agent.pt", help="Đường dẫn lưu trọng số")
    args = parser.parse_args()
    
    train_agent(num_episodes=args.episodes, weights_path=args.weights_path)
