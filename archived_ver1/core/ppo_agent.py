"""
ppo_agent.py: Thuật toán Continuous Actor-Critic (PPO - Proximal Policy Optimization).

Kế thừa trực tiếp từ nền tảng lý thuyết môn học:
- Slide 3.11: Kiến trúc Actor-Critic.
- Slide 3.12: Tham số hóa Chính sách Gaussian cho Không gian Hành động Liên tục.
- Cơ chế PPO-Clip: Chuẩn công nghiệp kiểm soát gradient cập nhật an toàn ở tốc độ cao.
"""

import numpy as np
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.normal import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class ActorCriticNetwork(nn.Module):
        def __init__(self, state_dim=16, action_dim=2, hidden_dim=128):
            super().__init__()
            # 1. Critic Network V(s) (Đánh giá giá trị trạng thái - Slide 3.11)
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
            # 2. Actor Network mu(s) (Học kỳ vọng hành động - Slide 3.12)
            self.actor_mean = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh() # Đưa về [-1, 1]
            )
            
            # Độ lệch chuẩn log_std của phân phối Gaussian (Slide 3.12)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        def forward(self, state):
            value = self.critic(state)
            mean = self.actor_mean(state)
            std = torch.exp(self.actor_log_std.expand_as(mean))
            return mean, std, value

        def get_action_and_value(self, state, action=None):
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return action, log_prob, entropy, value


class PPOAgent:
    """
    Tác nhân PPO Continuous Actor-Critic.
    """
    def __init__(self, state_dim=16, action_dim=2, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        
        self.device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu") if HAS_TORCH else None
        
        if HAS_TORCH:
            self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        else:
            # Fallback numpy linear-gaussian controller
            self.actor_weights = np.zeros((action_dim, state_dim), dtype=np.float32)
            self.critic_weights = np.zeros(state_dim, dtype=np.float32)
            
        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state, deterministic=False):
        """
        Chọn hành động liên tục [steer, accel].
        deterministic=True khi chạy đua kiểm thử (lấy kỳ vọng mean mu(s)).
        """
        if HAS_TORCH:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mean, std, value = self.network(state_t)
                if deterministic:
                    action = mean
                    log_prob = torch.zeros(1)
                else:
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                action_clamped = torch.clamp(action, -1.0, 1.0)
                return action_clamped.cpu().numpy()[0], log_prob.item(), value.item()
        else:
            # Heuristic continuous controller based on LiDAR and Lookahead
            steer = np.dot(self.actor_weights[0], state)
            accel = np.dot(self.actor_weights[1], state)
            return np.clip([steer, accel], -1.0, 1.0), 0.0, 0.0

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def update(self, next_value, batch_size=64, epochs=10):
        """
        Cập nhật mạng Actor và Critic bằng PPO Clipped Objective & GAE.
        """
        if not HAS_TORCH or len(self.states) < 32:
            self.reset_buffer()
            return 0.0, 0.0

        # 1. Tính toán Generalized Advantage Estimation (GAE)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [next_value])
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values[:-1]
        
        # 2. Chuyển sang PyTorch Tensors
        b_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        b_actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        b_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        b_advantages = torch.FloatTensor(advantages).to(self.device)
        b_returns = torch.FloatTensor(returns).to(self.device)
        
        # Chuẩn hóa Advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # 3. Vòng lặp tối ưu hóa PPO qua các Minibatches
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    b_states[batch_idx], b_actions[batch_idx]
                )
                
                # Tỷ số xác suất r_t(theta) = pi_theta(a|s) / pi_old(a|s)
                log_ratio = new_log_prob - b_log_probs[batch_idx]
                ratio = torch.exp(log_ratio)
                
                # Clipped Surrogate Objective
                surr1 = ratio * b_advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * b_advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Loss (Mean Squared Error)
                critic_loss = 0.5 * ((new_value.view(-1) - b_returns[batch_idx]) ** 2).mean()
                
                # Entropy Bonus khuyến khích khám phá
                entropy_loss = entropy.mean()
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        self.reset_buffer()
        return total_actor_loss, total_critic_loss

    def save(self, filepath):
        if HAS_TORCH:
            torch.save(self.network.state_dict(), filepath)

    def load(self, filepath):
        if HAS_TORCH:
            self.network.load_state_dict(torch.load(filepath, map_location=self.device))
