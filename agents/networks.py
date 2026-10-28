from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Khởi tạo trọng số Orthogonal theo chuẩn OpenAI Baselines / CleanRL"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    Mạng Actor-Critic liên tục (Continuous Actor-Critic)
    Actor: Ước lượng vector trung bình mu(s), kết hợp tham số log_std để lấy mẫu phân phối Gaussian.
    Critic: Ước lượng hàm giá trị trạng thái V(s).
    """
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256, init_log_std: float = -0.8):
        super().__init__()
        
        # Mạng Critic V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Mạng Actor mu(s)
        actor_out = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        # Khởi tạo thiên vị chân ga ban đầu (Forward Drive Bias):
        # Action 0 = Steering (0.0: thẳng)
        # Action 1 = Throttle (+0.5: giúp xe lăn bánh xuất phát)
        # Action 2 = Brake (-0.8: ban đầu không kẹt phanh, nhưng sigma rộng cho phép khám phá phanh)
        if actor_out.bias is not None:
            actor_out.bias.data[0] = 0.0
            actor_out.bias.data[1] = 0.5
            if action_dim >= 3:
                actor_out.bias.data[2] = -0.8
            
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            actor_out,
        )
        
        # Tham số log_std: Khởi tạo ở mức -0.8 (std ~ 0.45) để tạo biên độ khám phá rộng cho chân phanh và vô lăng
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), float(init_log_std)))

    def actor_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về (u_mean, action_logstd)"""
        u_mean = self.actor_mean(x)
        action_logstd = torch.clamp(self.actor_logstd.expand_as(u_mean), min=-2.3, max=0.0)
        return u_mean, action_logstd

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Dự đoán giá trị trạng thái V(s)"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Lấy mẫu hành động qua Squashed Gaussian (Tanh Normal) với hiệu chỉnh Jacobian log-determinant.
        Hành động sinh ra luôn nằm nghiêm ngặt trong [-1.0, 1.0], triệt tiêu lỗi Action Boundary Mismatch.
        """
        action_mean = self.actor_mean(x)
        # Kẹp logstd trong [-2.3, 0.0] (tương ứng sigma trong [0.10, 1.00]) để bảo toàn khám phá mà chống nổ phương sai
        action_logstd = torch.clamp(self.actor_logstd.expand_as(action_mean), min=-2.3, max=0.0)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                u = torch.clamp(action_mean, -5.0, 5.0)
                action = torch.tanh(u)
            else:
                raw_u = probs.rsample()
                u = torch.clamp(raw_u, -5.0, 5.0)
                action = torch.tanh(u)
        else:
            # Nghịch đảo tanh: u = atanh(clamp(action))
            clamped_action = torch.clamp(action, -0.999, 0.999)
            u = torch.clamp(torch.atanh(clamped_action), -5.0, 5.0)
            action = clamped_action
                
        # Hiệu chỉnh Jacobian log-determinant ổn định số học (chống bùng nổ -inf / NaN khi PPO importance sampling)
        jacobian = torch.clamp(1.0 - action.pow(2), min=1e-4)
        log_prob = (probs.log_prob(u) - torch.log(jacobian)).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)
        
        return action, log_prob, entropy, value
