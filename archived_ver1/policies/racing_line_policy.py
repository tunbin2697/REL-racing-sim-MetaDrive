"""
policies/racing_line_policy.py: Chính sách Đua xe Tối ưu Ngoài - Trong - Ngoài (Racing Line Policy).
Tận dụng toàn bộ bề rộng 16m mặt đường để mở rộng bán kính góc cua (+30%), kết hợp Quy hoạch Động Bellman OVP.
"""

import numpy as np
import math
from .base_policy import BasePolicy

class RacingLineOptimalPolicy(BasePolicy):
    """
    Chính sách Đua xe Tối ưu Out-In-Out Racing Line (Policy Improvement).
    - Vòng 1 làm quen tạo ra quỹ đạo hình học Out-In-Out mở rộng bán kính cua.
    - Giải bài toán quy hoạch động Bellman OVP (Quét Lùi phanh muộn + Quét Tiến kéo ga).
    - Bộ điều khiển lái thích ứng Lookahead Controller bám sát vạch Racing Line vàng.
    """
    def __init__(self, profiler=None, lap_stage=3, name="Racing Line Optimal", color="#10b981"):
        super().__init__(
            name=name,
            description="Tối ưu hóa hình học Ngoài - Trong - Ngoài (Out-In-Out) + Bellman OVP, mở rộng bán kính cua",
            color=color
        )
        self.profiler = profiler
        self.lap_stage = lap_stage # 2 (Time Attack) hoặc 3 (Max Out-In-Out)
        self.ref_path = None
        self.n_ref = 0
        self.track_name = ""

    def reset(self, track_info=None):
        if track_info is not None:
            self.track_name = track_info.get("name", "").lower()
            if self.profiler is not None:
                if not self.profiler.is_profiled:
                    self.profiler.profile_track_center(track_info["center"])
                self.ref_path = self.profiler.get_reference_path(self.lap_stage, policy_type="racing_line")
                self.n_ref = len(self.ref_path)

    def select_action(self, state, info=None):
        car_state = info["car_state"] if (info and "car_state" in info) else state
        x, y, yaw, v = car_state[0], car_state[1], car_state[2], car_state[3]
        
        if self.ref_path is None or self.n_ref == 0:
            return [0.0, 0.5]
            
        # Tìm điểm gần nhất trên Racing Line
        dists = np.linalg.norm(self.ref_path - np.array([x, y]), axis=1)
        c_idx = int(np.argmin(dists))
        
        # Tốc độ mục tiêu theo Bellman OVP trên Racing Line
        target_v = self.profiler.get_target_speed(x, y, lap_number=self.lap_stage, policy_type="racing_line")
        
        # Lookahead thích ứng theo đặc tính đường đua
        if "silverstone" in self.track_name:
            lookahead_steps = max(14, int(v * 1.1))
        else:
            lookahead_steps = max(5, int(v * 0.70))
            
        target_idx = (c_idx + lookahead_steps) % self.n_ref
        tx, ty = self.ref_path[target_idx]
        
        desired_yaw = math.atan2(ty - y, tx - x)
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        
        steer_cmd = float(np.clip(yaw_err * 1.5, -1.0, 1.0))
        
        if v < target_v - 0.5:
            accel_cmd = 1.0  # Tăng tốc hết ga
        elif v > target_v + 0.5:
            accel_cmd = -0.90 # Phanh gấp
        else:
            accel_cmd = 0.10
            
        return [steer_cmd, accel_cmd]

    def get_profiler_features(self):
        return 0.0, 0.0
