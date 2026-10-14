"""
Mạng Actor-Critic v2 (Physics-informed Squashed Gaussian Policy)
So với v1 (agents/networks.py):
  1. Lớp đặc trưng vật lý cố định: ghép thêm 4 tín hiệu [v_allow, vượt tốc, steer Pure Pursuit, ga gợi ý]
     tính từ chính obs 35 chiều -> mạng không phải tự khám phá khái niệm "tốc độ an toàn vào cua".
  2. LayerNorm đầu vào để cân bằng thang đo giữa các nhóm đặc trưng.
  3. Độ lệch chuẩn phụ thuộc trạng thái: log_std = actor_logstd (bias toàn cục) + head(s) (khởi tạo 0),
     cho phép agent tự giảm nhiễu ở khúc cua gắt và khám phá nhiều hơn ở đoạn thẳng.
  4. Residual policy (mặc định): hành động = Policy luật + hiệu chỉnh giới hạn (chống PPO kéo lệch khi fine-tune).
  5. Giữ nguyên giao diện get_action_and_value / actor_logstd để dùng lại PPOAgent và train_racing.py.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .networks import layer_init
from .racing_features import physics_features_torch

NUM_PHYSICS_FEATURES = 4
LOG_STD_MIN, LOG_STD_MAX = -3.5, 0.0  # Lái cực nhạy với nhiễu (σ 0.1 đã văng) -> cho phép σ_steer ~ 0.03


class ActorCriticV2(nn.Module):
    def __init__(
        self,
        state_dim: int = 35,
        action_dim: int = 2,
        hidden_dim: int = 256,
        init_log_std: float = -2.0,
        residual: bool = True,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        in_dim = state_dim + NUM_PHYSICS_FEATURES
        # Residual policy: u_mean = atanh(hành động Policy luật) + residual_scale * tanh(head(s))
        # -> khởi đầu đúng bằng Policy luật, PPO chỉ học phần hiệu chỉnh có biên độ giới hạn
        self.residual = residual
        self.residual_scale = residual_scale

        self.critic = nn.Sequential(
            nn.LayerNorm(in_dim),
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        self.actor_trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor_mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.actor_logstd_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        nn.init.zeros_(self.actor_logstd_head.weight)
        # Bias log_std toàn cục: train_racing.py --reset-logstd ghi đè trực tiếp tham số này
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), float(init_log_std)))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, physics_features_torch(x)], dim=1)

    def actor_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về (u_mean, log_std) trước phép squash tanh"""
        feats = self._features(x)
        h = self.actor_trunk(feats)
        if self.residual:
            prior = torch.atanh(feats[:, -2:].clamp(-0.95, 0.95))
            u_mean = prior + self.residual_scale * torch.tanh(self.actor_mean_head(h))
        else:
            u_mean = self.actor_mean_head(h)
        log_std = (self.actor_logstd + self.actor_logstd_head(h)).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return u_mean, log_std

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(x))

    def get_action_and_value(self, x, action=None, deterministic=False):
        """Squashed Gaussian + hiệu chỉnh Jacobian, giống hệt v1"""
        action_mean, action_logstd = self.actor_forward(x)
        probs = Normal(action_mean, torch.exp(action_logstd))

        if action is None:
            raw_u = action_mean if deterministic else probs.rsample()
            u = torch.clamp(raw_u, -5.0, 5.0)
            action = torch.tanh(u)
        else:
            clamped_action = torch.clamp(action, -0.999, 0.999)
            u = torch.clamp(torch.atanh(clamped_action), -5.0, 5.0)
            action = clamped_action

        jacobian = torch.clamp(1.0 - action.pow(2), min=1e-4)
        log_prob = (probs.log_prob(u) - torch.log(jacobian)).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.get_value(x)
        return action, log_prob, entropy, value
