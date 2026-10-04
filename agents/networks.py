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
    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 256):
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
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        
        # Tham số log_std độc lập với trạng thái (Learnable state-independent log standard deviation)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Dự đoán giá trị trạng thái V(s)"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Lấy mẫu hành động a ~ N(mu, sigma) và tính log_prob, entropy cùng giá trị V(s).
        Nếu deterministic=True (dùng khi inference/đua thực tế), trả về chính xác mu(s).
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = torch.clamp(action_mean, -1.0, 1.0)
            else:
                action = probs.sample()
                
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)
        
        return action, log_prob, entropy, value
