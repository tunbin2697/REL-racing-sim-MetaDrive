"""
core/dynamics.py: Mô hình Động lực học Xe (Kinematic Bicycle Model).
Bao gồm góc trượt (slip angle), ma sát lốp, giới hạn gia tốc và cản khí động học.
"""

import numpy as np
import math

class CarDynamics:
    """
    Mô hình Động lực học Xe (Kinematic Bicycle Model).
    Bao gồm quán tính, ma sát lốp, góc trượt thân xe (slip angle) và giới hạn gia tốc.
    """
    def __init__(self, wheelbase=2.5, max_steer=0.65, max_speed=40.0, max_accel=6.0, max_brake=10.0):
        self.wheelbase = wheelbase          # Chiều dài cơ sở (m)
        self.max_steer = max_steer          # Góc lái tối đa (~37 độ)
        self.max_speed = max_speed          # Tốc độ tối đa (m/s ~ 144 km/h)
        self.max_accel = max_accel          # Gia tốc cực đại (m/s^2)
        self.max_brake = max_brake          # Lực phanh cực đại (m/s^2)
        self.drag = 0.002                   # Lực cản không khí
        self.friction = 0.15                # Ma sát lăn mặt đường

    def step(self, state, steer_action, accel_action, dt=0.05):
        """
        Cập nhật trạng thái xe sau khoảng thời gian dt.
        state: [x, y, yaw (rad), velocity (m/s), steer_angle (rad)]
        steer_action: [-1, 1]
        accel_action: [-1, 1] (dương là đạp ga, âm là phanh)
        """
        x, y, yaw, v, steer = state
        
        # 1. Cập nhật góc lái
        target_steer = np.clip(steer_action, -1.0, 1.0) * self.max_steer
        steer_rate = 3.5 # Tốc độ đánh lái (rad/s)
        steer_diff = target_steer - steer
        steer = steer + np.clip(steer_diff, -steer_rate * dt, steer_rate * dt)
        
        # 2. Tính lực đẩy / phanh
        if accel_action >= 0:
            acceleration = accel_action * self.max_accel
        else:
            acceleration = accel_action * self.max_brake # Phanh gấp hơn tăng tốc
            
        # Trừ lực cản ma sát và khí động học
        net_accel = acceleration - self.drag * (v ** 2) - self.friction * v
        v_next = np.clip(v + net_accel * dt, 0.0, self.max_speed)
        
        # 3. Góc trượt thân xe (Slip Angle beta)
        beta = math.atan(0.5 * math.tan(steer))
        
        # 4. Tích phân vị trí & góc quay
        yaw_rate = (v / self.wheelbase) * math.sin(beta)
        yaw_next = yaw + yaw_rate * dt
        yaw_next = (yaw_next + np.pi) % (2 * np.pi) - np.pi # Chuẩn hóa [-pi, pi]
        
        x_next = x + v * math.cos(yaw + beta) * dt
        y_next = y + v * math.sin(yaw + beta) * dt
        
        return np.array([x_next, y_next, yaw_next, v_next, steer], dtype=np.float32)
