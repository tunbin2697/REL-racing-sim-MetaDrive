import sys
sys.path.insert(0, r"c:\Users\tunbi\Project\school\REL Assignment")
import numpy as np
from panda3d.core import LVector3f
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ExtendedPhysicsConfig:
    # 1. Cơ chế Cản Động Lực Học (Steering Scrub Drag & High-Rate Chatter Drag)
    enable_scrub_drag: bool = True
    scrub_coeff: float = 1.00              # Hệ số cản chà lốp Pacejka chuẩn GT3 (Linear Scrub Model)
    chatter_multiplier: float = 3.5        # Khuếch đại khi đảo lái ngược chiều liên tục (Bang-bang chatter)
    ref_top_speed_m_s: float = 33.33       # Vận tốc tham chiếu (120 km/h)
    steer_deadband: float = 0.02           # Vùng chết vi mô (0.4 độ) để chạy thẳng tuyệt đối không bị cản
    v_scrub_min_km_h: float = 25.0         # Vận tốc tối thiểu để tính cản chà lốp (dưới 25 km/h cho phép khởi hành mượt)

    # 2. Cơ chế Đa Trục: Bất Ổn Định 3 Trục & Văng Đuôi (Dynamic Jerk & Snap Oversteer)
    enable_snap_oversteer: bool = True
    v_instability_min_km_h: float = 55.0   # Ngưỡng vận tốc an toàn: dưới 55 km/h không bị văng đuôi
    instability_threshold: float = 0.45    # Ngưỡng chỉ số giật 3 trục (Jerk + Pedal Shift)
    unstable_rear_friction: float = 0.50   # Ma sát lốp sau khi trượt văng đuôi (bình thường 1.35)
    destabilizing_yaw_gain: float = 1.8    # Cường độ xung mô-men xoay văng đuôi (rad/s)
    oversteer_recovery_steps: int = 5      # Số bước hồi phục độ bám lốp sau (0.5s)

    # 3. Trợ lực lái đầm chắc theo tốc độ (Speed-Sensitive Steering)
    enable_speed_sensitive_steer: bool = True
    v_steer_min_km_h: float = 25.0
    v_steer_max_km_h: float = 110.0
    min_steer_lock_ratio: float = 0.50
    min_steer_rate_ratio: float = 0.75

    # 4. Quán tính truyền động (Powertrain Inertia)
    enable_powertrain_inertia: bool = True
    throttle_lag_alpha: float = 0.70       # Chuẩn GT3
    brake_lag_alpha: float = 0.85

class NewExtendedVehiclePhysics:
    def __init__(self, config: Optional[ExtendedPhysicsConfig] = None):
        self.cfg = config or ExtendedPhysicsConfig()
        self.last_applied_pedal: float = 0.0
        self.prev_applied_pedal: float = 0.0
        self.last_applied_steer: float = 0.0
        self.prev_delta_steer: float = 0.0
        self.last_delta_steer: float = 0.0
        self.oversteer_timer: int = 0
        self.is_oversteering: bool = False

    def reset(self):
        self.last_applied_pedal = 0.0
        self.prev_applied_pedal = 0.0
        self.last_applied_steer = 0.0
        self.prev_delta_steer = 0.0
        self.last_delta_steer = 0.0
        self.oversteer_timer = 0
        self.is_oversteering = False

    def calculate_steering_limits(self, speed_km_h: float, base_max_rate: float) -> Tuple[float, float]:
        if not self.cfg.enable_speed_sensitive_steer:
            return 1.0, base_max_rate

        v = float(np.clip(speed_km_h, self.cfg.v_steer_min_km_h, self.cfg.v_steer_max_km_h))
        t = (v - self.cfg.v_steer_min_km_h) / (self.cfg.v_steer_max_km_h - self.cfg.v_steer_min_km_h + 1e-6)
        smooth_t = t * t * (3.0 - 2.0 * t)

        lock_ratio = 1.0 - smooth_t * (1.0 - self.cfg.min_steer_lock_ratio)
        rate_ratio = 1.0 - smooth_t * (1.0 - self.cfg.min_steer_rate_ratio)
        return float(lock_ratio), float(base_max_rate * rate_ratio)

    def process_steering(self, raw_steer: float, speed_km_h: float, base_max_rate: float) -> Tuple[float, float]:
        raw_steer = float(np.clip(raw_steer, -1.0, 1.0))
        lock_scale, max_rate = self.calculate_steering_limits(speed_km_h, base_max_rate)

        target_steer = float(np.clip(raw_steer, -lock_scale, lock_scale))
        delta_steer = target_steer - self.last_applied_steer
        clamped_delta = float(np.clip(delta_steer, -max_rate, max_rate))
        steer_applied = float(np.clip(self.last_applied_steer + clamped_delta, -1.0, 1.0))

        self.last_delta_steer = clamped_delta
        self.last_applied_steer = steer_applied
        return steer_applied, clamped_delta

    def process_pedal(self, target_pedal: float) -> float:
        target_pedal = float(np.clip(target_pedal, -1.0, 1.0))
        if not self.cfg.enable_powertrain_inertia:
            self.prev_applied_pedal = self.last_applied_pedal
            self.last_applied_pedal = target_pedal
            return target_pedal

        alpha = self.cfg.brake_lag_alpha if target_pedal < self.last_applied_pedal else self.cfg.throttle_lag_alpha
        pedal_applied = float(self.last_applied_pedal + alpha * (target_pedal - self.last_applied_pedal))
        pedal_applied = float(np.clip(pedal_applied, -1.0, 1.0))

        self.prev_applied_pedal = self.last_applied_pedal
        self.last_applied_pedal = pedal_applied
        return pedal_applied

    def apply_steering_scrub_drag(
        self,
        raw_vehicle,
        applied_steer: float,
        speed_m_s: float,
        dt: float,
        base_friction: float = 1.35,
    ) -> float:
        speed_km_h = speed_m_s * 3.6
        if not self.cfg.enable_scrub_drag or raw_vehicle is None or speed_km_h < self.cfg.v_scrub_min_km_h:
            self.prev_delta_steer = self.last_delta_steer
            return 0.0

        current_delta = self.last_delta_steer
        # Phát hiện hành vi quẫy vô-lăng đảo chiều liên tục (Bang-bang chatter)
        is_chattering = bool(current_delta * self.prev_delta_steer < -1e-4)

        steer_mag = float(abs(applied_steer))
        speed_loss_km_h = 0.0

        if steer_mag > self.cfg.steer_deadband:
            effective_steer = (steer_mag - self.cfg.steer_deadband) / (1.0 - self.cfg.steer_deadband)
            speed_ratio = float(np.clip(speed_m_s / self.cfg.ref_top_speed_m_s, 0.0, 1.5))
            
            # Lực cản cơ sở Pacejka
            a_scrub = float(self.cfg.scrub_coeff * 9.80665 * effective_steer * (0.5 + 0.5 * speed_ratio))
            
            # Khi quẫy lái đảo chiều liên tục: Phạt thêm lực cản tốc độ bẻ lái cực mạnh
            if is_chattering:
                steer_rate = abs(current_delta) / max(dt, 1e-4)
                a_scrub *= float(self.cfg.chatter_multiplier * (1.0 + steer_rate))

            delta_v = float(a_scrub * dt)
            if delta_v > 1e-4:
                new_speed_m_s = max(0.0, speed_m_s - delta_v)
                if hasattr(raw_vehicle, "velocity") and hasattr(raw_vehicle, "set_velocity"):
                    raw_vehicle.set_velocity(raw_vehicle.velocity, value=new_speed_m_s)
                speed_loss_km_h = float(delta_v * 3.6)

        # 2. Cơ chế Đa Trục: Bất ổn định 3 trục & Văng đuôi (Dynamic Snap Oversteer)
        if self.cfg.enable_snap_oversteer and speed_km_h >= self.cfg.v_instability_min_km_h:
            delta_pedal = abs(self.last_applied_pedal - self.prev_applied_pedal)
            instability_idx = (abs(current_delta) / 0.18) * (speed_km_h / 120.0) * (1.0 + 1.5 * delta_pedal)

            if instability_idx >= self.cfg.instability_threshold and is_chattering:
                self.oversteer_timer = self.cfg.oversteer_recovery_steps
                self.is_oversteering = True
                
                if hasattr(raw_vehicle, "body") and hasattr(raw_vehicle.body, "get_angular_velocity") and hasattr(raw_vehicle.body, "set_angular_velocity"):
                    yaw_sign = 1.0 if current_delta > 0 else -1.0
                    curr_ang = raw_vehicle.body.get_angular_velocity()
                    raw_vehicle.body.set_angular_velocity(
                        LVector3f(curr_ang[0], curr_ang[1], curr_ang[2] + yaw_sign * self.cfg.destabilizing_yaw_gain)
                    )

            if hasattr(raw_vehicle, "wheels") and len(raw_vehicle.wheels) >= 4:
                if self.oversteer_timer > 0:
                    self.oversteer_timer -= 1
                    raw_vehicle.wheels[2].setFrictionSlip(float(self.cfg.unstable_rear_friction))
                    raw_vehicle.wheels[3].setFrictionSlip(float(self.cfg.unstable_rear_friction))
                else:
                    self.is_oversteering = False
                    raw_vehicle.wheels[2].setFrictionSlip(float(base_friction))
                    raw_vehicle.wheels[3].setFrictionSlip(float(base_friction))

        self.prev_delta_steer = current_delta
        return speed_loss_km_h
