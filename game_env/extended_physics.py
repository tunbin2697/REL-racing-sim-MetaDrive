"""
Mô-đun Động Lực Học Mở Rộng Chuẩn Xe Đua Thể Thao GT3 (Natural Vehicle Dynamics)
Áp dụng nguyên lý Single Responsibility Principle (SRP) và Lập trình Hướng đối tượng (OOP):

Mô hình hóa các quy luật cơ học Newton và lốp xe theo chuẩn quốc tế (TORCS / Pacejka / Fast-Lap):
1. Trục Lái (Steering Axis): Lực cản Lái Chiếu Dọc Tự Nhiên (Induced Steering Drag - Newton Force)
   F_drag = 2 * F_y,front * |sin(delta)|
   Tác dụng trực tiếp lực cản cơ học ngược chiều qua Bullet apply_central_force.
   Không dùng if/else, không can thiệp cưỡng bức vận tốc qua set_velocity. Lạng lách đảo lái liên tục sinh lực cản lớn
   hơn cả lực kéo động cơ, tự động làm xe giảm tốc độ 4 - 6 km/h mỗi giây một cách thuần vật lý.
2. Trục Ga (Throttle Axis): Vòng tròn Ma sát Kamm & Trượt bám Công suất (Drive Wheels Power-Oversteer)
   F_x^2 + F_y^2 <= (mu * F_z)^2
   Thốc ga đột ngột trong cua làm triệt tiêu lực bám ngang của lốp sau, dẫn tới văng đuôi xe tự nhiên.
3. Trục Phanh (Brake Axis): Dịch chuyển Tải trọng Dọc khi Phanh (Dynamic Longitudinal Load Transfer)
   Delta_F_z = m * a_x * (h_cg / L_wheelbase)
   Phanh gấp dồn tải lên cầu trước và nhấc nhẹ cầu sau, làm giảm độ bám lốp sau trong cua.
4. Quán tính Cơ cấu Chấp hành (Actuator Slew Rate & Powertrain Inertia)
   Giới hạn tốc độ vặn vô lăng (max_steer_rate = 0.18) và độ trễ bướm ga/phanh.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from panda3d.core import LVector3f


@dataclass
class ExtendedPhysicsConfig:
    """Tham số cấu hình các cơ chế vật lý mở rộng thuần cơ học chuẩn GT3"""
    # 1. Trục Lái: Lực cản lái chiếu dọc (Induced Steering Drag)
    enable_induced_drag: bool = True
    induced_drag_scale: float = 1.0        # Hệ số khuếch đại lực cản chiếu dọc tự nhiên
    steer_deadband: float = 0.015          # Vùng chết góc lái vi mô (0.3 độ) khi chạy thẳng tuyệt đối
    steer_lock_rad: float = 0.349          # Góc bẻ lái cực đại (20 độ chuẩn GT3)
    
    # 2. Trục Ga: Vòng tròn ma sát Kamm cho bánh sau (Drive Wheels Traction Circle)
    enable_kamm_traction: bool = True
    max_engine_force_n: float = 850.0      # Lực kéo cực đại của động cơ GT3 (N)
    
    # 3. Trục Phanh: Dịch chuyển tải trọng dọc (Longitudinal Pitch Load Transfer)
    enable_load_transfer: bool = True
    vehicle_mass_kg: float = 1100.0        # Khối lượng xe GT3 (kg)
    wheelbase_m: float = 2.75              # Chiều dài cơ sở (m)
    cg_height_m: float = 0.40              # Chiều cao trọng tâm (m)
    
    # 4. Trợ lực lái đầm chắc theo tốc độ (Speed-Sensitive Steering)
    enable_speed_sensitive_steer: bool = True
    v_steer_min_km_h: float = 25.0
    v_steer_max_km_h: float = 110.0
    min_steer_lock_ratio: float = 0.85     # Cho phép 85% góc lái ở tốc độ cao
    min_steer_rate_ratio: float = 0.70     # Hạn chế tốc độ vặn vô lăng ở tốc độ cao
    
    # 5. Quán tính truyền động (Powertrain Inertia)
    enable_powertrain_inertia: bool = True
    throttle_lag_alpha: float = 0.85
    brake_lag_alpha: float = 0.90


class ExtendedVehiclePhysics:
    """
    Lớp quản lý và tính toán các cơ chế vật lý mở rộng thuần cơ học cho xe đua thể thao GT3.
    Tuân thủ SRP, OOP, Định luật Newton và độc lập hoàn toàn với Trọng tài Môi trường.
    """
    def __init__(self, config: Optional[ExtendedPhysicsConfig] = None):
        self.cfg = config or ExtendedPhysicsConfig()
        self.last_applied_pedal: float = 0.0
        self.prev_applied_pedal: float = 0.0
        self.last_applied_steer: float = 0.0
        self.prev_delta_steer: float = 0.0
        self.last_delta_steer: float = 0.0
        self.prev_speed_m_s: float = 0.0
        self.induced_drag_n: float = 0.0
        self.is_oversteering: bool = False

    def reset(self):
        """Khởi tạo lại toàn bộ trạng thái nội tại của bộ điều khiển vật lý"""
        self.last_applied_pedal = 0.0
        self.prev_applied_pedal = 0.0
        self.last_applied_steer = 0.0
        self.prev_delta_steer = 0.0
        self.last_delta_steer = 0.0
        self.prev_speed_m_s = 0.0
        self.induced_drag_n = 0.0
        self.is_oversteering = False

    def calculate_steering_limits(self, speed_km_h: float, base_max_rate: float) -> Tuple[float, float]:
        """
        Tính toán góc lái tối đa cho phép và tốc độ quay vô-lăng (Slew Rate) theo vận tốc tức thời.
        Sử dụng hàm trơn Hermite (Smoothstep) để đảm bảo không bị giật cục.
        """
        if not self.cfg.enable_speed_sensitive_steer:
            return 1.0, base_max_rate

        v = float(np.clip(speed_km_h, self.cfg.v_steer_min_km_h, self.cfg.v_steer_max_km_h))
        t = (v - self.cfg.v_steer_min_km_h) / (self.cfg.v_steer_max_km_h - self.cfg.v_steer_min_km_h + 1e-6)
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

        target_steer = float(np.clip(raw_steer, -lock_scale, lock_scale))
        delta_steer = target_steer - self.last_applied_steer
        clamped_delta = float(np.clip(delta_steer, -max_rate, max_rate))
        steer_applied = float(np.clip(self.last_applied_steer + clamped_delta, -1.0, 1.0))

        self.last_delta_steer = clamped_delta
        self.last_applied_steer = steer_applied
        return steer_applied, clamped_delta

    def process_pedal(self, target_pedal: float) -> float:
        """
        Lọc quán tính bướm ga và áp suất phanh qua bộ lọc thông thấp bậc nhất.
        """
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

    def apply_dynamics(
        self,
        raw_vehicle,
        applied_steer: float,
        applied_pedal: float,
        speed_m_s: float,
        heading_theta: float,
        dt: float,
        base_friction: float = 1.35,
    ) -> float:
        """
        Áp dụng toàn diện 3 cơ chế động lực học thuần cơ học chuẩn quốc tế:
        1. Trục Lái: Lực cản lái chiếu dọc F_drag = 2 * F_y,front * sin(delta) tác dụng qua apply_central_force.
        2. Trục Ga: Bão hòa vòng tròn ma sát Kamm trên bánh sau khi thốc ga (Power-Oversteer).
        3. Trục Phanh: Dịch chuyển tải trọng dọc làm nhẹ cầu sau khi phanh gấp.
        Trả về: Tốc độ bị suy giảm ước tính (km/h) cho bảng telemetry.
        """
        if raw_vehicle is None or not hasattr(raw_vehicle, "body"):
            return 0.0

        # --- 1. TRỤC PHANH: DỊCH CHUYỂN TẢI TRỌNG DỌC (PITCH LOAD TRANSFER) ---
        m = self.cfg.vehicle_mass_kg
        f_z0 = 0.5 * m * 9.80665  # ~5393 N mỗi trục
        if self.cfg.enable_load_transfer:
            # Gia tốc dọc hiệu dụng xác định mượt mà từ hành trình chân ga / chân phanh
            if applied_pedal >= 0.0:
                a_x_net = applied_pedal * (self.cfg.max_engine_force_n / m)
            else:
                a_x_net = applied_pedal * 6.0  # Phanh thể thao GT3 (-6.0 m/s^2)

            delta_f_z = float(m * a_x_net * (self.cfg.cg_height_m / self.cfg.wheelbase_m))
            delta_f_z = float(np.clip(delta_f_z, -0.35 * f_z0, 0.35 * f_z0))
            f_z_front = float(f_z0 - delta_f_z)
            f_z_rear = float(f_z0 + delta_f_z)
        else:
            f_z_front = f_z0
            f_z_rear = f_z0

        # --- 2. TRỤC LÁI: LỰC CẢN LÁI CHIẾU DỌC TỰ NHIÊN (INDUCED STEERING DRAG) ---
        speed_loss_km_h = 0.0
        steer_mag = abs(float(applied_steer))
        if self.cfg.enable_induced_drag and steer_mag > self.cfg.steer_deadband:
            delta_rad = float(applied_steer * self.cfg.steer_lock_rad)
            # Lực bám ngang bánh trước F_y (mô hình Pacejka trơn không bão hòa đột ngột)
            f_y_front = float(base_friction * f_z_front * np.tanh(3.0 * steer_mag))
            # Thành phần cản lùi dọc theo thân xe
            f_drag = float(f_y_front * np.sin(abs(delta_rad)) * self.cfg.induced_drag_scale)
            self.induced_drag_n = f_drag

            # Tác dụng lực cản lùi trực tiếp vào thân xe qua Bullet native physics (áp dụng cho bước mô phỏng tới)
            f_x = float(-f_drag * np.cos(heading_theta))
            f_y = float(-f_drag * np.sin(heading_theta))
            raw_vehicle.body.apply_central_force(LVector3f(f_x, f_y, 0.0))

            # Tính tốc độ hao hụt cơ học để hiển thị telemetry: delta_v = a_drag * dt
            speed_loss_km_h = float((f_drag / m) * dt * 3.6)
        else:
            self.induced_drag_n = 0.0

        # --- 3. TRỤC GA: VÒNG TRÒN MA SÁT KAMM CHO CẦU SAU CHỦ ĐỘNG ---
        if hasattr(raw_vehicle, "wheels") and len(raw_vehicle.wheels) >= 4:
            if self.cfg.enable_kamm_traction:
                # Lực dọc tác dụng lên bánh sau (kéo hoặc phanh)
                if applied_pedal >= 0.0:
                    f_x_rear = float(applied_pedal * self.cfg.max_engine_force_n)
                else:
                    # Phanh GT3: cầu sau chịu ~35% lực phanh tổng
                    f_x_rear = float(abs(applied_pedal) * 650.0 * 0.35)

                f_grip_max_rear = float(base_friction * f_z_rear)
                load_ratio_rear = float(f_z_rear / f_z0)

                # Tỷ số sử dụng ma sát dọc Kamm
                usage_ratio = min(0.95, f_x_rear / max(100.0, f_grip_max_rear))
                # Ma sát ngang khả dụng còn lại của lốp sau
                friction_scale_rear = float(np.sqrt(max(0.15, 1.0 - usage_ratio ** 2)))
                mu_rear_effective = float(np.clip(base_friction * load_ratio_rear * friction_scale_rear, 0.50, 1.35))

                # Cập nhật độ bám vào 2 bánh sau (index 2, 3)
                raw_vehicle.wheels[2].setFrictionSlip(mu_rear_effective)
                raw_vehicle.wheels[3].setFrictionSlip(mu_rear_effective)

                # Cầu trước: Luôn đảm bảo đủ độ bám bẻ lái (tối thiểu 1.35, tăng thêm khi phanh dồn tải)
                load_ratio_front = float(f_z_front / f_z0)
                mu_front_effective = float(base_friction * max(1.0, load_ratio_front))
                raw_vehicle.wheels[0].setFrictionSlip(mu_front_effective)
                raw_vehicle.wheels[1].setFrictionSlip(mu_front_effective)

                # Theo dõi trạng thái trượt văng đuôi xe
                self.is_oversteering = bool(usage_ratio > 0.85 and steer_mag > 0.40)
            else:
                for w in raw_vehicle.wheels:
                    w.setFrictionSlip(float(base_friction))

        self.prev_delta_steer = self.last_delta_steer
        return speed_loss_km_h

    def apply_steering_scrub_drag(
        self,
        raw_vehicle,
        applied_steer: float,
        speed_m_s: float,
        dt: float,
        base_friction: float = 1.35,
        heading_theta: Optional[float] = None,
        applied_pedal: Optional[float] = None,
    ) -> float:
        """
        Hàm tương thích ngược với API cũ của RacingEnv:
        Chuyển tiếp đầy đủ sang apply_dynamics().
        """
        if heading_theta is None:
            heading_theta = float(getattr(raw_vehicle, "heading_theta", 0.0))
        if applied_pedal is None:
            applied_pedal = self.last_applied_pedal

        return self.apply_dynamics(
            raw_vehicle=raw_vehicle,
            applied_steer=applied_steer,
            applied_pedal=applied_pedal,
            speed_m_s=speed_m_s,
            heading_theta=heading_theta,
            dt=dt,
            base_friction=base_friction,
        )
