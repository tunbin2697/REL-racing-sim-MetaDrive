"""
PPO Agent v2: dùng lại toàn bộ thuật toán PPO của v1 (GAE, clipped surrogate, decoupled Actor/Critic optimizer),
chỉ thay mạng bằng ActorCriticV2 (đặc trưng vật lý + std phụ thuộc trạng thái).
Checkpoint lưu thêm khóa "arch": "v2" để policy_loader tự nhận diện.
"""

from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .ppo_agent import PPOAgent
from .networks_v2 import ActorCriticV2
from .racing_features import physics_features_torch


class PPOAgentV2(PPOAgent):
    ARCH = "v2"

    def __init__(
        self,
        state_dim: int = 35,
        action_dim: int = 2,
        lr: float = 1.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.15,
        ent_coef: float = 0.005,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        device: Optional[str] = None,
        critic_warmup_updates: int = 5,
        bc_coef: float = 5.0,
        bc_decay: float = 0.98,
        residual: bool = True,
    ):
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_coef=clip_coef, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
            update_epochs=update_epochs, minibatch_size=minibatch_size, device=device,
        )
        # Thay mạng v1 bằng v2 và dựng lại optimizer trên tham số mới
        self.residual = residual
        self.ac = ActorCriticV2(state_dim, action_dim, residual=residual).to(self.device)
        self.actor_params = (
            list(self.ac.actor_trunk.parameters())
            + list(self.ac.actor_mean_head.parameters())
            + list(self.ac.actor_logstd_head.parameters())
            + [self.ac.actor_logstd]
        )
        self.critic_params = list(self.ac.critic.parameters())
        self.actor_optimizer = optim.Adam(self.actor_params, lr=self.lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=self.lr_critic, eps=1e-5)

        # Chống "quên" kiến thức BC khi chuyển sang RL:
        # - critic_warmup_updates: vài lần update đầu chỉ học Critic, đóng băng Actor (advantage còn sai lệch)
        # - bc_coef: phạt lệch khỏi hành động Policy luật (tính trực tiếp từ obs), giảm dần theo bc_decay
        self.critic_warmup_updates = critic_warmup_updates
        self.bc_coef = bc_coef
        self.bc_decay = bc_decay
        self.num_updates_done = 0

    def update(self, b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values=None):
        b_states = torch.as_tensor(b_states, dtype=torch.float32, device=self.device)
        b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=self.device)
        b_log_probs = torch.as_tensor(b_log_probs, dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(b_advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(b_returns, dtype=torch.float32, device=self.device)
        b_values_tensor = torch.as_tensor(b_values, dtype=torch.float32, device=self.device) if b_values is not None else b_returns
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        with torch.no_grad():
            b_expert = physics_features_torch(b_states)[:, 2:4]

        train_actor = self.num_updates_done >= self.critic_warmup_updates
        bc_coef = self.bc_coef * (self.bc_decay ** max(0, self.num_updates_done - self.critic_warmup_updates))
        batch_size = b_states.size(0)
        indices = np.arange(batch_size)
        totals = {"pg": 0.0, "v": 0.0, "ent": 0.0, "bc": 0.0, "kl": 0.0}
        num_mb = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                _, new_log_prob, entropy, new_value = self.ac.get_action_and_value(b_states[mb_idx], b_actions[mb_idx])
                log_ratio = new_log_prob - b_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)
                pg_loss = torch.max(
                    -b_advantages[mb_idx] * ratio,
                    -b_advantages[mb_idx] * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef),
                ).mean()
                entropy_loss = entropy.mean()
                u_mean, _ = self.ac.actor_forward(b_states[mb_idx])
                bc_loss = F.mse_loss(torch.tanh(u_mean), b_expert[mb_idx])

                if train_actor:
                    actor_loss = pg_loss - self.ent_coef * entropy_loss + bc_coef * bc_loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                    self.actor_optimizer.step()

                v_pred = new_value.view(-1)
                v_clipped = b_values_tensor[mb_idx] + torch.clamp(v_pred - b_values_tensor[mb_idx], -10.0, 10.0)
                v_loss = 0.5 * torch.max((v_pred - b_returns[mb_idx]) ** 2, (v_clipped - b_returns[mb_idx]) ** 2).mean()
                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.critic_optimizer.step()

                totals["pg"] += pg_loss.item()
                totals["v"] += v_loss.item()
                totals["ent"] += entropy_loss.item()
                totals["bc"] += bc_loss.item()
                totals["kl"] += ((ratio - 1.0) - log_ratio).mean().item()
                num_mb += 1

        self.num_updates_done += 1
        y_true = b_returns.cpu().numpy()
        y_pred = b_values_tensor.cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 0 else 0.0
        return {
            "policy_loss": totals["pg"] / max(1, num_mb),
            "value_loss": totals["v"] / max(1, num_mb),
            "entropy": totals["ent"] / max(1, num_mb),
            "bc_loss": totals["bc"] / max(1, num_mb),
            "bc_coef": bc_coef if train_actor else 0.0,
            "approx_kl": totals["kl"] / max(1, num_mb),
            "explained_var": explained_var,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save({
            "arch": self.ARCH,
            "residual": self.residual,
            "model_state_dict": self.ac.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, filepath)
