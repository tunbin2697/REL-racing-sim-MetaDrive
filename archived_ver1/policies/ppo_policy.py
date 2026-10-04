"""
policies/ppo_policy.py: Chính sách Học Tăng cường PPO Deep RL (Continuous Actor-Critic).
Sử dụng mạng nơ-ron sâu đã huấn luyện (f1_racing_agent.pt), điều khiển chỉ bằng cảm biến LiDAR 180 độ.
"""

import os
import numpy as np
from .base_policy import BasePolicy

class PPOAgentPolicy(BasePolicy):
    """
    Chính sách Mạng Nơ-ron PPO Học Sâu (Deep RL Actor-Critic Policy).
    - Điều khiển dựa trên 11 tia LiDAR và vận tốc xe (không cần bản đồ toàn cục - Map-free).
    - Dùng làm đối chứng (Baseline) cho bài toán phản xạ cục bộ (Reactive Control).
    """
    def __init__(self, weights_path=None, name="Pure PPO Neural Net", color="#8b5cf6"):
        super().__init__(
            name=name,
            description="Mạng nơ-ron sâu Actor-Critic (Gaussian Policy), phản xạ thuần qua 11 tia LiDAR",
            color=color
        )
        self.agent = None
        self.weights_loaded = False
        
        # Tìm đường dẫn weights
        if weights_path is None:
            search_paths = [
                os.path.join(os.path.dirname(__file__), "..", "models", "f1_racing_agent.pt"),
                os.path.join(os.path.dirname(__file__), "..", "f1_racing_agent.pt"),
                os.path.join(os.getcwd(), "models", "f1_racing_agent.pt"),
                os.path.join(os.getcwd(), "f1_racing_agent.pt")
            ]
            for p in search_paths:
                if os.path.exists(p):
                    weights_path = p
                    break
                    
        try:
            from core.ppo_agent import PPOAgent
            self.agent = PPOAgent(state_dim=16, action_dim=2)
            if weights_path and os.path.exists(weights_path):
                self.agent.load(weights_path)
                self.weights_loaded = True
        except Exception as e:
            print(f"[!] Không thể nạp mạng PPO: {e}")

    def reset(self, track_info=None):
        pass

    def select_action(self, state, info=None):
        """Suy luận trực tiếp từ mạng nơ-ron (Inference, No Training / No Gradient)."""
        if self.agent is not None:
            action, _, _ = self.agent.select_action(state, deterministic=True)
            return [float(action[0]), float(action[1])]
        return [0.0, 0.0]
