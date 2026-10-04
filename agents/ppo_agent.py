"""
Thuật toán PPO (Proximal Policy Optimization) - Continuous Action Space
Dựa trên kiến trúc chuẩn Clipped Surrogate Objective & Generalized Advantage Estimation (GAE)
Đúng theo khung lý thuyết giáo trình Học tăng cường (REL).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .networks import ActorCritic

class PPOAgent:
    def __init__(
        self,
        state_dim: int = 259,
        action_dim: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        
    def select_action(self, state, deterministic: bool = False):
        """
        Chọn hành động từ trạng thái hiện tại.
        Trả về: action (numpy array), log_prob, value
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.ac.get_action_and_value(state_tensor, deterministic=deterministic)
            action_np = action.cpu().numpy()[0]
            action_clipped = np.clip(action_np, -1.0, 1.0)
            return action_clipped, log_prob.cpu().item(), value.cpu().item()

    def evaluate_states(self, states, actions):
        """Đánh giá lại log_prob, entropy và value của batch"""
        return self.ac.get_action_and_value(states, actions)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Tính toán Generalized Advantage Estimation (GAE) và Returns:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
        """
        num_steps = len(rewards)
        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae_lam = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values
        return advantages, returns

    def update(self, b_states, b_actions, b_log_probs, b_advantages, b_returns):
        """
        Cập nhật trọng số mạng bằng thuật toán PPO Minibatch SGD
        """
        b_states = torch.as_tensor(b_states, dtype=torch.float32, device=self.device)
        b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=self.device)
        b_log_probs = torch.as_tensor(b_log_probs, dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(b_advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(b_returns, dtype=torch.float32, device=self.device)
        
        # Chuẩn hoá Advantages theo từng batch (Advantage Normalization)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        batch_size = b_states.size(0)
        indices = np.arange(batch_size)
        
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                _, new_log_prob, entropy, new_value = self.ac.get_action_and_value(
                    b_states[mb_idx], b_actions[mb_idx]
                )
                
                log_ratio = new_log_prob - b_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)
                
                # PPO Policy Loss (Clipped Objective)
                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss (MSE)
                v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_idx]) ** 2).mean()
                
                # Entropy Bonus
                entropy_loss = entropy.mean()
                
                # Tổng hàm mất mát cần tối thiểu hóa
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1
                
        return {
            "policy_loss": total_pg_loss / max(1, num_updates),
            "value_loss": total_v_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }

    def save(self, filepath: str):
        """Lưu checkpoint mô hình"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, filepath)

    def load(self, filepath: str):
        """Nạp checkpoint mô hình"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.ac.eval()
