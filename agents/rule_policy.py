"""
Policy luật (Rule-based Racing Expert) - không học, chỉ đọc vector quan sát 35 chiều:
  - Lái: Pure Pursuit  delta = atan(2 * L * sin(alpha) / Ld), điểm ngắm cách xe Ld = k_ld * v + ld_min.
  - Ga/Phanh: bám tốc độ mục tiêu = tỷ lệ an toàn * v_allow (đường bao phanh theo bán kính cua).
Dùng làm baseline so sánh với PPO và làm Expert cho Behavior Cloning khởi tạo PPO v2.
Giao diện select_action giống PPOAgent để dùng chung các script đánh giá.
"""

import numpy as np
from .racing_features import allowed_speed_kmh, lookahead_point


class RuleRacingPolicy:
    def __init__(
        self,
        wheelbase: float = 2.47,
        max_steering_deg: float = 40.0,
        ld_gain: float = 0.6,
        ld_min: float = 8.0,
        a_lat: float = 0.8 * 9.81,  # 0.85g (8.34) nhanh nhất còn an toàn, 0.9g bắt đầu văng
        a_brake: float = 5.0,
        speed_gain: float = 0.15,
        speed_margin: float = 1.0,
    ):
        self.wheelbase = wheelbase
        self.max_steering = np.deg2rad(max_steering_deg)
        self.ld_gain = ld_gain
        self.ld_min = ld_min
        self.a_lat = a_lat
        self.a_brake = a_brake
        self.speed_gain = speed_gain
        self.speed_margin = speed_margin

    def target_speed_kmh(self, obs):
        return self.speed_margin * allowed_speed_kmh(obs, self.a_lat, self.a_brake)

    def act(self, obs):
        speed_ms = float(obs[0]) * 40.0

        # 1. Lái Pure Pursuit (steer dương = rẽ trái, dy dương = bên trái xe)
        ld = self.ld_gain * speed_ms + self.ld_min
        dx, dy = lookahead_point(obs, ld)
        alpha = np.arctan2(dy, max(dx, 1e-3))
        dist = max(np.hypot(dx, dy), 1e-3)
        delta = np.arctan(2.0 * self.wheelbase * np.sin(alpha) / dist)
        steer = float(np.clip(delta / self.max_steering, -1.0, 1.0))

        # 2. Ga/Phanh tỷ lệ theo sai lệch tốc độ mục tiêu
        error_kmh = self.target_speed_kmh(obs) - speed_ms * 3.6
        throttle = float(np.clip(self.speed_gain * error_kmh, -1.0, 1.0))
        return np.array([steer, throttle], dtype=np.float32)

    def select_action(self, state, deterministic: bool = True):
        """Tương thích PPOAgent.select_action: trả về (action, log_prob, value)"""
        return self.act(state), 0.0, 0.0
