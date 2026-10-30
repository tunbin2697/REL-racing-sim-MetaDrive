"""
Thuật toán PPO (Proximal Policy Optimization) - Continuous Action Space
Dựa trên kiến trúc chuẩn Clipped Surrogate Objective & Generalized Advantage Estimation (GAE)
Đúng theo khung lý thuyết giáo trình Học tăng cường (REL).
"""

from typing import Optional, Any, Dict, Tuple, overload, Literal
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .networks import ActorCritic

class PPOAgent:
    def __init__(
        self,
        state_dim: int = 40,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.15,
        ent_coef: float = 0.005,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        device: Optional[str] = None,
        enable_logstd_projection: bool = True,
        logstd_max: Optional[Any] = None,
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
        self.enable_logstd_projection = enable_logstd_projection
        
        if logstd_max is not None:
            self.logstd_max_tensor = torch.as_tensor(logstd_max, dtype=torch.float32, device=self.device).view(1, -1)
        else:
            self.logstd_max_tensor = None
        
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.actor_params = list(self.ac.actor_mean.parameters()) + [self.ac.actor_logstd]
        self.critic_params = list(self.ac.critic.parameters())
        
        self.lr_actor = lr
        self.lr_critic = lr
        self.actor_optimizer = optim.Adam(self.actor_params, lr=self.lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=self.lr_critic, eps=1e-5)
        self.loaded_meta: Dict[str, Any] = {}

    def _project_logstd(self):
        """Phép chiếu logstd (In-place projection) theo D2 và D19"""
        if not self.enable_logstd_projection:
            return
        with torch.no_grad():
            self.ac.actor_logstd.clamp_(min=self.ac.LOGSTD_MIN)
            if self.logstd_max_tensor is not None:
                max_tensor = self.logstd_max_tensor.to(self.ac.actor_logstd.device)
                self.ac.actor_logstd.data = torch.min(self.ac.actor_logstd.data, max_tensor)
            else:
                self.ac.actor_logstd.clamp_(max=self.ac.LOGSTD_MAX)

    def set_lr(self, lr: float):
        """Cập nhật learning rate động cho Actor và Critic optimizers"""
        self.lr_actor = float(lr)
        self.lr_critic = float(lr)
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = float(lr)
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = float(lr)

    def get_lr(self) -> float:
        """Lấy learning rate hiện hành của Actor optimizer"""
        return float(self.actor_optimizer.param_groups[0]["lr"])

    def set_ent_coef(self, ent_coef: float):
        """Cập nhật hệ số thưởng Entropy động theo lịch trình annealing"""
        self.ent_coef = float(ent_coef)

    def get_ent_coef(self) -> float:
        """Lấy hệ số thưởng Entropy hiện hành"""
        return float(self.ent_coef)
        
    @property
    def network(self):
        """Thuộc tính tương thích cho mạng ActorCritic"""
        return self.ac

    @overload
    def select_action(
        self, state, deterministic: bool = False, return_raw: Literal[False] = False
    ) -> Tuple[np.ndarray, float, float]: ...

    @overload
    def select_action(
        self, state, deterministic: bool = False, return_raw: Literal[True] = True
    ) -> Tuple[np.ndarray, float, float, np.ndarray]: ...

    def select_action(self, state, deterministic: bool = False, return_raw: bool = False):
        """
        Chọn hành động từ trạng thái hiện tại.
        Trả về: action (numpy array), log_prob, value (+ raw_u trước tanh khi return_raw=True, để lưu vào buffer)
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value, raw_u = self.ac.get_action_and_value(
                state_tensor, deterministic=deterministic, return_raw=True
            )
            action_np = action.cpu().numpy()[0]
            action_clipped = np.clip(action_np, -1.0, 1.0)
            if return_raw:
                return action_clipped, log_prob.cpu().item(), value.cpu().item(), raw_u.cpu().numpy()[0]
            return action_clipped, log_prob.cpu().item(), value.cpu().item()

    def evaluate_states(self, states, actions):
        """Đánh giá lại log_prob, entropy và value của batch"""
        return self.ac.get_action_and_value(states, actions)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Tính toán Generalized Advantage Estimation (GAE) và Returns:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        Returns_t = A_t + V(s_t)
        """
        num_steps = len(rewards)
        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae_lam = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = float(next_value)
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = float(values[t + 1])
                
            delta = float(rewards[t]) + self.gamma * next_val * next_non_terminal - float(values[t])
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values, b_raw_us=None):
        """
        Huấn luyện PPO qua nhiều epoch trên các mini-batch ngẫu nhiên.
        Cập nhật đồng thời Actor Loss, Critic Value Loss, và Entropy Bonus.
        b_raw_us: u trước tanh đã sample lúc rollout; khi có thì log_prob được tính lại đúng tại điểm đó (F13).
        """
        # Chuyển đổi dữ liệu sang Tensor PyTorch
        if not isinstance(b_states, torch.Tensor):
            b_states = torch.as_tensor(b_states, dtype=torch.float32, device=self.device)
            b_actions = torch.as_tensor(b_actions, dtype=torch.float32, device=self.device)
            b_log_probs = torch.as_tensor(b_log_probs, dtype=torch.float32, device=self.device)
            b_advantages = torch.as_tensor(b_advantages, dtype=torch.float32, device=self.device)
            b_returns = torch.as_tensor(b_returns, dtype=torch.float32, device=self.device)
            b_values_tensor = torch.as_tensor(b_values, dtype=torch.float32, device=self.device)
        else:
            b_values_tensor = b_values
        if b_raw_us is not None:
            b_raw_us = torch.as_tensor(b_raw_us, dtype=torch.float32, device=self.device)
            
        # Chuẩn hóa Advantages (Zero-mean, Unit-variance) theo chuẩn PPO
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std()
        b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)
        
        batch_size = b_states.shape[0]
        indices = np.arange(batch_size)
        
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        skipped_actor_updates = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                if b_raw_us is not None:
                    _, new_log_prob, entropy, new_value = self.ac.get_action_and_value(
                        b_states[mb_idx], raw_u=b_raw_us[mb_idx]
                    )
                else:
                    _, new_log_prob, entropy, new_value = self.ac.get_action_and_value(
                        b_states[mb_idx], b_actions[mb_idx]
                    )

                # Lưới an toàn: kẹp log_ratio để exp() không tràn số (ratio <= e^20) khi có mẫu bất thường
                log_ratio = torch.clamp(new_log_prob - b_log_probs[mb_idx], -20.0, 20.0)
                ratio = torch.exp(log_ratio)
                
                # 1. PPO Policy Loss (Clipped Objective)
                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 2. Entropy Bonus
                entropy_loss = entropy.mean()
                actor_loss = pg_loss - self.ent_coef * entropy_loss
                
                # Cập nhật riêng mạng Actor (bỏ qua minibatch nếu loss không hữu hạn để không ghi NaN vào trọng số)
                self.actor_optimizer.zero_grad()
                if torch.isfinite(actor_loss):
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                    self.actor_optimizer.step()
                else:
                    skipped_actor_updates += 1
                
                # Phép chiếu logstd (In-place projection) chống đóng băng khám phá
                self._project_logstd()
                
                # 3. PPO Clipped Value Loss (Ổn định Critic, chống spike khi va chạm)
                v_pred = new_value.view(-1)
                v_clipped = b_values_tensor[mb_idx] + torch.clamp(
                    v_pred - b_values_tensor[mb_idx], -10.0, 10.0
                )
                v_loss_unclipped = (v_pred - b_returns[mb_idx]) ** 2
                v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                # Cập nhật riêng mạng Critic
                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_pg_loss += pg_loss.item() if torch.isfinite(pg_loss) else 0.0
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1
                
        # Explained Variance
        y_true = b_returns.cpu().numpy()
        y_pred = b_values_tensor.cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 0 else 0.0
                
        return {
            "policy_loss": total_pg_loss / max(1, num_updates),
            "value_loss": total_v_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "explained_var": explained_var,
            "skipped_actor_updates": skipped_actor_updates,
        }

    def save(self, filepath: str, meta: Optional[Dict[str, Any]] = None):
        """Lưu checkpoint mô hình kèm metadata cấu hình (D34)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload: Dict[str, Any] = {
            "model_state_dict": self.ac.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "meta": dict(meta) if meta is not None else {},
        }
        torch.save(payload, filepath)

    @staticmethod
    def peek_meta(filepath: str) -> Dict[str, Any]:
        """Đọc trước metadata (map, vehicle, obs_mode, spawn_mode) từ file checkpoint nếu có (D34)"""
        if not filepath or not os.path.exists(filepath):
            return {}
        try:
            ckpt = torch.load(filepath, map_location="cpu")
            if isinstance(ckpt, dict) and isinstance(ckpt.get("meta"), dict):
                return dict(ckpt["meta"])
        except Exception:
            pass
        return {}

    def load(self, filepath: str):
        """Nạp checkpoint mô hình (Tự động thích ứng với cấu hình 2D hoặc 3D action space)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
            self.loaded_meta = dict(checkpoint["meta"])
        else:
            self.loaded_meta = {}
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        
        # Tự động suy luận action_dim và state_dim từ metadata hoặc từ shape của trọng số
        ckpt_action_dim = checkpoint.get("action_dim")
        if ckpt_action_dim is None and "actor_logstd" in state_dict:
            ckpt_action_dim = state_dict["actor_logstd"].shape[1]
        ckpt_state_dim = checkpoint.get("state_dim")
        if ckpt_state_dim is None and "actor_mean.0.weight" in state_dict:
            ckpt_state_dim = state_dict["actor_mean.0.weight"].shape[1]
            
        # Kiểm tra tính tương thích số chiều quan sát (D26)
        if ckpt_state_dim is not None and int(ckpt_state_dim) != int(self.state_dim):
            raise ValueError(
                f"Lệch số chiều quan sát (D26): Checkpoint '{filepath}' có state_dim={ckpt_state_dim}, "
                f"nhưng tác tử hiện tại được cấu hình với state_dim={self.state_dim}. "
                f"Không được nạp checkpoint lệch dimension trực tiếp để tránh lỗi shape của torch."
            )

        if ckpt_action_dim is not None and int(ckpt_action_dim) != int(self.action_dim):
            self.action_dim = int(ckpt_action_dim)
            self.ac = ActorCritic(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.actor_params = list(self.ac.actor_mean.parameters()) + [self.ac.actor_logstd]
            self.critic_params = list(self.ac.critic.parameters())
            self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=self.lr_actor, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=self.lr_critic, eps=1e-5)

        self.ac.load_state_dict(state_dict)
        if "actor_optimizer_state_dict" in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            except Exception:
                pass
        if "critic_optimizer_state_dict" in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            except Exception:
                pass
                
        self._project_logstd()
        self.ac.eval()
