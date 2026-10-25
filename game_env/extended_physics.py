"""
Mô-đun Cơ Chế Vật Lý Mở Rộng (Extended Vehicle Physics)
Áp dụng nguyên lý Single Responsibility Principle (SRP) và Lập trình hướng đối tượng (OOP):

Đóng gói các hiện tượng động lực học đua xe thực tế mà Bullet Physics / MetaDrive chưa mô phỏng:
1. Steering Scrub Drag: Lực cản sinh ra do góc trượt lốp khi bẻ lái, làm tiêu hao động năng
   khiến việc đảo lái liên tục trên đường thẳng bị trừng phạt tự nhiên bằng suy giảm tốc độ.
2. Speed-Sensitive Steering (Variable Steering Ratio & Damping): Giảm góc bẻ lái cực đại và
   tốc độ đánh lái ở vận tốc cao, mô phỏng cơ cấu trợ lực lái biến thiên và ổn định khí động học.
3. Powertrain Inertia (First-order Lag Filter): Mô phỏng quán tính cơ học của bướm ga và áp suất
   thủy lực phanh, triệt tiêu xung đóng ngắt gián đoạn 10Hz (PWM chattering).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ExtendedPhysicsConfig:
    """Tham số cấu hình các cơ chế vật lý mở rộng chuẩn xe đua thể thao"""
    # 1. Steering Scrub Drag (Lực cản do chà lốp khi bẻ lái)
    enable_scrub_drag: bool = True
    scrub_coeff: float = 1.00            # Hệ số cản chà lốp Pacejka chuẩn GT3 (Linear Scrub Model)
    max_steering_rad: float = 0.349066   # 20 độ đổi sang radian (20 * pi / 180)
    ref_top_speed_m_s: float = 33.33     # Vận tốc tham chiếu (120 km/h chuẩn GT3)
    steer_deadband: float = 0.02         # Vùng chết vi mô (0.4 độ) để chạy thẳng tuyệt đối không bị cản

    # 2. Speed-Sensitive Steering (Trợ lực lái đầm chắc theo tốc độ)
    enable_speed_sensitive_steer: bool = True
    v_steer_min_km_h: float = 25.0       # Dưới tốc độ này giữ nguyên độ linh hoạt 100%
    v_steer_max_km_h: float = 110.0      # Đạt độ đầm chắc cực đại tại dải tốc độ cao
    min_steer_lock_ratio: float = 0.50   # Tỷ lệ góc lái tối đa ở tốc độ cao (50% x 20 deg = 10 deg)
    min_steer_rate_ratio: float = 0.75   # Tỷ lệ tốc độ quay vô-lăng ở tốc độ cao (75% x 0.18 = 0.135/step)

    # 3. Powertrain Inertia (Bộ lọc quán tính động cơ và thủy lực phanh)
    enable_powertrain_inertia: bool = True
    throttle_lag_alpha: float = 0.70     # Tốc độ phản hồi bướm ga (0.70 đạt 95% sau ~0.15s)
    brake_lag_alpha: float = 0.85        # Tốc độ phản hồi phanh (phanh thủy lực phản hồi nhanh hơn ga)


class ExtendedVehiclePhysics:
    """
    Lớp quản lý và tính toán các cơ chế vật lý mở rộng cho xe đua.
    Tách biệt hoàn toàn với logic môi trường (SRP) và có thể tái sử dụng cho bất kỳ xe nào.
    """
    def __init__(self, config: Optional[ExtendedPhysicsConfig] = None):
        self.cfg = config or ExtendedPhysicsConfig()
        self.last_applied_pedal: float = 0.0
        self.last_applied_steer: float = 0.0

    def reset(self):
        """Khởi tạo lại trạng thái nội tại của bộ điều khiển vật lý"""
        self.last_applied_pedal = 0.0
        self.last_applied_steer = 0.0

    def calculate_steering_limits(self, speed_km_h: float, base_max_rate: float) -> Tuple[float, float]:
        """
        Tính toán góc lái tối đa cho phép và tốc độ quay vô-lăng (Slew Rate) theo vận tốc tức thời.
        Sử dụng hàm nội suy trơn Hermite (Smoothstep) để đảm bảo không bị giật cục khi chuyển vùng vận tốc.
        Trả về: (max_lock_scale, current_max_rate)
        """
        if not self.cfg.enable_speed_sensitive_steer:
            return 1.0, base_max_rate

        v = float(np.clip(speed_km_h, self.cfg.v_steer_min_km_h, self.cfg.v_steer_max_km_h))
        # Tỷ lệ chuẩn hóa [0, 1]
        t = (v - self.cfg.v_steer_min_km_h) / (self.cfg.v_steer_max_km_h - self.cfg.v_steer_min_km_h + 1e-6)
        # Hàm trơn Hermite: 3t^2 - 2t^3
        smooth_t = t * t * (3.0 - 2.0 * t)

        lock_ratio = 1.0 - smooth_t * (1.0 - self.cfg.min_steer_lock_ratio)
        rate_ratio = 1.0 - smooth_t * (1.0 - self.cfg.min_steer_rate_ratio)

        current_max_rate = float(base_max_rate * rate_ratio)
        return float(lock_ratio), current_max_rate

    def process_steering(
        self,
        raw_steer: float,
        speed_km_h: float,
        base_max_rate: float,
    ) -> Tuple[float, float]:
        """
        Xử lý tín hiệu vô-lăng thô từ Policy:
        - Giới hạn góc bẻ lái cực đại theo dải vận tốc (Speed-Sensitive Lock)
        - Giới hạn tốc độ đánh lái theo độ đầm cơ học (Speed-Sensitive Slew Rate Limit)
        Trả về: (steer_applied, delta_steer)
        """
        raw_steer = float(np.clip(raw_steer, -1.0, 1.0))
        lock_scale, max_rate = self.calculate_steering_limits(speed_km_h, base_max_rate)

        # Giới hạn góc lái tối đa theo vận tốc (tránh bẻ lái 20 độ ở 120 km/h)
        target_steer = float(np.clip(raw_steer, -lock_scale, lock_scale))

        # Slew Rate Limiter theo vận tốc
        delta_steer = target_steer - self.last_applied_steer
        clamped_delta = float(np.clip(delta_steer, -max_rate, max_rate))
        steer_applied = float(np.clip(self.last_applied_steer + clamped_delta, -1.0, 1.0))

        self.last_applied_steer = steer_applied
        return steer_applied, clamped_delta

    def process_pedal(self, target_pedal: float) -> float:
        """
        Lọc quán tính bướm ga và áp suất phanh qua bộ lọc thông thấp bậc nhất (First-Order Lag).
        Triệt tiêu xung đóng mở gián đoạn 10Hz (PWM chattering).
        Trả về: pedal_applied
        """
        target_pedal = float(np.clip(target_pedal, -1.0, 1.0))

        if not self.cfg.enable_powertrain_inertia:
            self.last_applied_pedal = target_pedal
            return target_pedal

        # Phanh phản hồi nhanh hơn ga (áp suất thủy lực so với quán tính trục khuỷu)
        alpha = self.cfg.brake_lag_alpha if target_pedal < self.last_applied_pedal else self.cfg.throttle_lag_alpha
        pedal_applied = float(self.last_applied_pedal + alpha * (target_pedal - self.last_applied_pedal))
        pedal_applied = float(np.clip(pedal_applied, -1.0, 1.0))

        self.last_applied_pedal = pedal_applied
        return pedal_applied

    def apply_steering_scrub_drag(
        self,
        raw_vehicle,
        applied_steer: float,
        speed_m_s: float,
        dt: float,
    ) -> float:
        """
        Tính toán và áp dụng lực cản chà lốp dọc (Steering Scrub Drag Force):
        F_scrub = C_scrub * m * g * sin^2(delta_rad) * (v / v_ref)^2
        Làm suy giảm trực tiếp động năng tiến về phía trước khi xe lắc lái ở tốc độ cao.
        Trả về: tốc độ bị suy giảm trong bước này (km/h)
        """
        if not self.cfg.enable_scrub_drag or raw_vehicle is None or speed_m_s < 0.5:
            return 0.0

        # Giải pháp A (Mô hình cản trượt Pacejka tuyến tính thực tế):
        # Lực cản sinh ra do góc trượt lốp F_drag = F_y * sin(delta).
        # Khi xe đảo lái trên đường thẳng, lực cản tỷ lệ trực tiếp với góc bẻ lái thực tế,
        # có vùng chết vi mô (steer_deadband = 0.02 ~ 0.4 độ) để chạy thẳng không bị cản trở.
        steer_mag = float(abs(applied_steer))
        if steer_mag <= self.cfg.steer_deadband:
            return 0.0

        effective_steer = (steer_mag - self.cfg.steer_deadband) / (1.0 - self.cfg.steer_deadband)

        # Gia tốc hãm do chà lốp:
        # a_scrub = C_scrub * g * effective_steer * (0.5 + 0.5 * (v / v_ref))
        speed_ratio = float(np.clip(speed_m_s / self.cfg.ref_top_speed_m_s, 0.0, 1.5))
        g = 9.80665
        a_scrub = float(self.cfg.scrub_coeff * g * effective_steer * (0.5 + 0.5 * speed_ratio))
        delta_v = float(a_scrub * dt)

        if delta_v <= 1e-4:
            return 0.0

        new_speed_m_s = max(0.0, speed_m_s - delta_v)
        
        # Áp dụng suy giảm vận tốc trực tiếp vào Bullet rigid body node
        if hasattr(raw_vehicle, "velocity") and hasattr(raw_vehicle, "set_velocity"):
            raw_vehicle.set_velocity(raw_vehicle.velocity, value=new_speed_m_s)

        speed_loss_km_h = float(delta_v * 3.6)
        return speed_loss_km_h
