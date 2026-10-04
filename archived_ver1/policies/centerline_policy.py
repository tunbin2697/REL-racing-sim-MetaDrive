"""
policies/centerline_policy.py: Các Chính sách Bám Tim Đường (Centerline Policies).

Gồm 2 chính sách:
1. CenterlineSafePolicy: Khảo sát an toàn ở tốc độ thấp, giữ xe chính giữa đường (Behavior Policy).
2. CenterlineSpeedPolicy: Tấn công tốc độ với Quy hoạch Động Bellman OVP trên tim đường (w = 0).
"""

import numpy as np
import math
from .base_policy import BasePolicy

class CenterlineSafePolicy(BasePolicy):
    """
    Chính sách Khảo sát Tim đường An toàn (Survey / Reconnaissance Policy).
    - Giữ xe chính giữa đường đua (Cross Track Error -> 0).
    - Duy trì vận tốc an toàn (~36 - 38 km/h), tự động giảm tốc ở các góc cua gắt theo giới hạn ma sát.
    """
    def __init__(self, target_speed_kmh=38.0, name="Centerline Safe", color="#ef4444"):
        super().__init__(
            name=name,
            description=f"Chạy an toàn giữa đường ({target_speed_kmh} km/h), phục vụ khảo sát bản đồ",
            color=color
        )
        self.target_speed_kmh = target_speed_kmh
        self.center = None
        self.n = 0
        self.target_speeds = None
        self.r_local = None

    def reset(self, track_info=None):
        if track_info is not None:
            self.center = np.array(track_info["center"])
            self.n = len(self.center)
            
            # Tính toán độ cong vi phân của tim đường
            diffs = np.diff(np.vstack([self.center, self.center[0]]), axis=0)
            ds = np.maximum(np.linalg.norm(diffs, axis=1), 1e-4)
            thetas = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
            d_theta = np.diff(np.concatenate([thetas, [thetas[0]]]))
            d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
            kappa = np.abs(d_theta / ds)
            self.r_local = 1.0 / np.maximum(kappa, 1e-4)
            
            # Tốc độ an toàn qua cua theo bán kính tim đường
            max_v = self.target_speed_kmh / 3.6
            self.target_speeds = np.clip(np.sqrt(0.70 * 9.81 * self.r_local), 6.5, max_v)
            
            # Quét lùi Bellman để phanh đón đầu trước góc cua
            a_brake = 4.5
            for _ in range(3):
                for i in range(self.n - 1, -1, -1):
                    next_i = (i + 1) % self.n
                    allowed = math.sqrt(self.target_speeds[next_i]**2 + 2.0 * a_brake * ds[i])
                    if allowed < self.target_speeds[i]:
                        self.target_speeds[i] = allowed

    def select_action(self, state, info=None):
        car_state = info["car_state"] if (info and "car_state" in info) else state
        x, y, yaw, v = car_state[0], car_state[1], car_state[2], car_state[3]
        
        if self.center is None or self.target_speeds is None:
            return [0.0, 0.4]
            
        dists = np.linalg.norm(self.center - np.array([x, y]), axis=1)
        c_idx = int(np.argmin(dists))
        target_v = self.target_speeds[c_idx]
        
        # Lookahead thích ứng theo độ cong và vận tốc
        lookahead_m = min(max(6.0, v * 0.45), self.r_local[c_idx] * 0.75 + 4.0)
        lookahead_steps = max(2, int(round(lookahead_m / 4.0)))
        t_idx = (c_idx + lookahead_steps) % self.n
        tx, ty = self.center[t_idx]
        
        desired_yaw = math.atan2(ty - y, tx - x)
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        
        # Cross Track Error
        p0 = self.center[c_idx]
        p1 = self.center[(c_idx + 1) % self.n]
        cross = (p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0])
        cte = dists[c_idx] * (1.0 if cross > 0 else -1.0)
        
        steer_cmd = float(np.clip(yaw_err * 1.6 - math.atan2(1.2 * cte, max(2.0, v)), -1.0, 1.0))
        
        if v < target_v - 0.5:
            accel_cmd = 1.0
        elif v > target_v + 0.5:
            accel_cmd = -0.90
        else:
            accel_cmd = 0.10
            
        return [steer_cmd, accel_cmd]


class CenterlineSpeedPolicy(BasePolicy):
    """
    Chính sách Tấn công Tốc độ trên Tim đường (Centerline Speed Attack Policy).
    - Xe giải bài toán Quy hoạch Động Bellman OVP để tối đa hóa vận tốc,
      NHƯNG bị ép buộc chỉ được chạy trên tim đường (w = 0, không được mở cua Ngoài - Trong - Ngoài).
    - Cho thấy rõ giới hạn vật lý của việc không tối ưu Racing Line!
    """
    def __init__(self, profiler=None, name="Centerline Speed", color="#f59e0b"):
        super().__init__(
            name=name,
            description="Tối ưu ga/phanh Bellman nhưng BỊ ÉP CHẠY TIM ĐƯỜNG (không mở cua Ngoài - Trong - Ngoài)",
            color=color
        )
        self.profiler = profiler
        self.center = None
        self.n = 0
        self.target_speeds = None
        self.r_local = None

    def reset(self, track_info=None):
        if track_info is not None:
            self.center = np.array(track_info["center"])
            self.n = len(self.center)
            
            diffs = np.diff(np.vstack([self.center, self.center[0]]), axis=0)
            ds = np.maximum(np.linalg.norm(diffs, axis=1), 1e-4)
            thetas = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
            d_theta = np.diff(np.concatenate([thetas, [thetas[0]]]))
            d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
            kappa = np.abs(d_theta / ds)
            self.r_local = 1.0 / np.maximum(kappa, 1e-4)
            
            # Tốc độ đỉnh cua tim đường theo ma sát cao hơn
            self.target_speeds = np.clip(np.sqrt(0.82 * 9.81 * self.r_local), 7.0, 22.0)
            
            # Quét lùi Bellman (phanh muộn tối ưu)
            a_brake = 6.5
            for _ in range(3):
                for i in range(self.n - 1, -1, -1):
                    next_i = (i + 1) % self.n
                    allowed = math.sqrt(self.target_speeds[next_i]**2 + 2.0 * a_brake * ds[i])
                    if allowed < self.target_speeds[i]:
                        self.target_speeds[i] = allowed
                        
            # Quét tiến Bellman (tăng tốc tối ưu)
            a_accel = 5.0
            for _ in range(2):
                for i in range(self.n):
                    prev_i = (i - 1) % self.n
                    allowed = math.sqrt(self.target_speeds[prev_i]**2 + 2.0 * a_accel * ds[prev_i])
                    if allowed < self.target_speeds[i]:
                        self.target_speeds[i] = allowed

    def select_action(self, state, info=None):
        car_state = info["car_state"] if (info and "car_state" in info) else state
        x, y, yaw, v = car_state[0], car_state[1], car_state[2], car_state[3]
        
        if self.center is None or self.target_speeds is None:
            return [0.0, 0.5]
            
        dists = np.linalg.norm(self.center - np.array([x, y]), axis=1)
        c_idx = int(np.argmin(dists))
        target_v = self.target_speeds[c_idx]
        
        lookahead_m = min(max(7.0, v * 0.55), self.r_local[c_idx] * 0.75 + 5.0)
        lookahead_steps = max(2, int(round(lookahead_m / 4.0)))
        t_idx = (c_idx + lookahead_steps) % self.n
        tx, ty = self.center[t_idx]
        
        desired_yaw = math.atan2(ty - y, tx - x)
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        
        p0 = self.center[c_idx]
        p1 = self.center[(c_idx + 1) % self.n]
        cross = (p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0])
        cte = dists[c_idx] * (1.0 if cross > 0 else -1.0)
        
        steer_cmd = float(np.clip(yaw_err * 1.6 - math.atan2(1.2 * cte, max(2.5, v)), -1.0, 1.0))
        
        if v < target_v - 0.5:
            accel_cmd = 1.0
        elif v > target_v + 0.5:
            accel_cmd = -0.90
        else:
            accel_cmd = 0.10
            
        return [steer_cmd, accel_cmd]
