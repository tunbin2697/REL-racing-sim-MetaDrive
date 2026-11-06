"""
Động Lực Học Xe Đua Thể Thao (Race Vehicle Dynamics & Actuation)
Áp dụng nguyên lý Single Responsibility (SRP) và Lập trình hướng đối tượng (OOP):
- Tách biệt hoàn toàn phần Cơ khí / Động lực học (Vehicle Dynamics) ra khỏi Trọng tài Môi trường (RacingEnv).
- Quản lý lực phanh, lực kéo, độ bám lốp, tỷ số truyền lái, bộ lọc làm mịn góc lái (Slew Rate Limiter).
- GT3RaceCar: xe đua của ref_111s và demo multicar.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VehicleConfig:
    """Thông số kỹ thuật động lực học của xe đua"""
    name: str = "GT3_RaceCar"
    mass: float = 1100.0               # Khối lượng thân xe (kg)
    max_engine_force: float = 850.0    # Lực kéo cực đại của động cơ (N)
    max_brake_force: float = 650.0     # Lực phanh cực đại (N) - Gấp 5.5 lần xe sedan
    max_steering_deg: float = 20.0     # Góc bẻ lái cực đại (độ) - Chuẩn xe đua đầm chắc
    wheel_friction: float = 1.35       # Hệ số ma sát lốp Bullet (Racing Slicks)
    # Ma sát riêng cầu trước (None = bằng wheel_friction). Bản cũ của attach_agent ghi ma sát qua list
    # raw_vehicle.wheels đã lỗi thời nên cầu trước luôn giữ mặc định 0.9 của MetaDrive (F16).
    front_wheel_friction: Optional[float] = None
    max_steer_rate: float = 0.18       # Tốc độ quay vô-lăng tối đa mỗi bước 0.1s (Slew Rate Limit)
    length: float = 4.50               # Chiều dài xe (m)
    width: float = 1.85                # Chiều rộng xe (m)
    max_speed_km_h: float = 120.0      # Vận tốc giới hạn trần (km/h)


class BaseRaceVehicle:
    """Lớp cơ sở đóng gói thực thể phương tiện đua xe (Race Vehicle Entity)"""
    def __init__(self, config: Optional[VehicleConfig] = None):
        self.cfg = config or VehicleConfig()
        self.raw_vehicle = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_steer = 0.0
        self.last_pedal = 0.0

    def reset_state(self):
        """Khởi tạo lại trạng thái điều khiển"""
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_steer = 0.0
        self.last_pedal = 0.0

    def attach_agent(self, vehicle_agent):
        """
        Tiêm thông số vật lý vào MetaDrive Bullet raycast vehicle agent
        Được gọi mỗi khi môi trường reset episode
        """
        self.raw_vehicle = vehicle_agent
        self.reset_state()

        if hasattr(self.raw_vehicle, "config"):
            self.raw_vehicle.config["max_speed_km_h"] = float(self.cfg.max_speed_km_h)
            self.raw_vehicle.config["max_brake_force"] = float(self.cfg.max_brake_force)
            self.raw_vehicle.config["max_engine_force"] = float(self.cfg.max_engine_force)
            self.raw_vehicle.config["max_steering"] = float(self.cfg.max_steering_deg)
            self.raw_vehicle.config["wheel_friction"] = float(self.cfg.wheel_friction)
            self.raw_vehicle.config["mass"] = float(self.cfg.mass)

        self.raw_vehicle.max_steering = float(self.cfg.max_steering_deg)
        # Lấy bánh trực tiếp từ BulletVehicle: list raw_vehicle.wheels lỗi thời sau khi MetaDrive dựng lại xe
        system = getattr(self.raw_vehicle, "system", None)
        if system is not None:
            front_mu = self.cfg.front_wheel_friction if self.cfg.front_wheel_friction is not None else self.cfg.wheel_friction
            for k in range(system.getNumWheels()):
                wheel = system.getWheel(k)
                wheel.setFrictionSlip(float(front_mu if wheel.isFrontWheel() else self.cfg.wheel_friction))

    def process_action(self, raw_action) -> Tuple[np.ndarray, float, float]:
        """
        Xử lý action thô từ policy RL:
        - Giới hạn tốc độ đánh lái (Slew Rate Limiter) chống rung lắc 10Hz
        - Chuyển đổi [steer, throttle, brake] sang định dạng chấp hành của MetaDrive
        Trả về: (action_applied, delta_steer, net_pedal)
        """
        raw_action = np.clip(np.array(raw_action, dtype=np.float32), -1.0, 1.0)
        steer = float(raw_action[0])

        thr = 0.5 * (float(raw_action[1]) + 1.0)
        brk = 0.5 * (float(raw_action[2]) + 1.0)
        net_pedal = thr * (1.0 - brk) - brk if brk > 0.05 else thr
        net_pedal = float(np.clip(net_pedal, -1.0, 1.0))

        # Slew rate limiter: Chống giật cục và triệt tiêu rung chấn vô-lăng
        delta_steer = steer - self.last_steer
        steer_applied = self.last_steer + np.clip(delta_steer, -self.cfg.max_steer_rate, self.cfg.max_steer_rate)
        steer_applied = float(np.clip(steer_applied, -1.0, 1.0))

        self.last_steer = steer_applied
        self.last_pedal = net_pedal
        self.last_action = np.array([steer_applied, net_pedal], dtype=np.float32)

        return self.last_action, float(delta_steer), net_pedal

    def get_wheel_positions(self) -> List[np.ndarray]:
        """
        Tính toán tọa độ 2D của 4 tâm bánh xe (FL, FR, RL, RR) phục vụ kiểm tra giới hạn đường đua chuẩn FIA.
        Đồng dạng hóa theo tỷ lệ kích thước thân xe (length, width).
        """
        if self.raw_vehicle is None:
            return []
        pos = np.array(self.raw_vehicle.position[:2], dtype=np.float32)
        heading = float(self.raw_vehicle.heading_theta)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        fwd = np.array([cos_h, sin_h], dtype=np.float32)
        lat = np.array([-sin_h, cos_h], dtype=np.float32)

        l_front = 0.5 * 2.75 * (self.cfg.length / 4.50)
        l_rear = -0.5 * 2.75 * (self.cfg.length / 4.50)
        w_track = 0.5 * 1.65 * (self.cfg.width / 1.85)

        return [
            pos + l_front * fwd + w_track * lat,  # Bánh trước Trái (Front-Left)
            pos + l_front * fwd - w_track * lat,  # Bánh trước Phải (Front-Right)
            pos + l_rear * fwd + w_track * lat,   # Bánh sau Trái (Rear-Left)
            pos + l_rear * fwd - w_track * lat,   # Bánh sau Phải (Rear-Right)
        ]

    def is_airborne_or_flipped(self) -> bool:
        """Kiểm tra xe có bị lật hoặc rơi rớt khỏi mặt sàn đường đua không"""
        if self.raw_vehicle is None or not hasattr(self.raw_vehicle, "origin"):
            return False
        pos_3d = self.raw_vehicle.origin.get_pos()
        hpr = self.raw_vehicle.origin.get_hpr()
        return bool(pos_3d[2] < -0.1 or pos_3d[2] > 2.0 or abs(hpr[1]) > 35.0 or abs(hpr[2]) > 35.0)

    # --- Properties truy cập nhanh trạng thái động lực học ---
    @property
    def position(self) -> np.ndarray:
        return np.array(self.raw_vehicle.position[:2], dtype=np.float32) if self.raw_vehicle else np.zeros(2)

    @property
    def velocity(self) -> np.ndarray:
        return np.array(self.raw_vehicle.velocity[:2], dtype=np.float32) if self.raw_vehicle else np.zeros(2, dtype=np.float32)

    @property
    def heading_theta(self) -> float:
        return float(self.raw_vehicle.heading_theta) if self.raw_vehicle else 0.0

    @property
    def crash_vehicle(self) -> bool:
        return bool(getattr(self.raw_vehicle, "crash_vehicle", False)) if self.raw_vehicle else False

    @property
    def crash_object(self) -> bool:
        return bool(getattr(self.raw_vehicle, "crash_object", False)) if self.raw_vehicle else False

    @property
    def crash_sidewalk(self) -> bool:
        return bool(getattr(self.raw_vehicle, "crash_sidewalk", False)) if self.raw_vehicle else False


class GT3RaceCar(BaseRaceVehicle):
    """Cấu hình xe đua thể thao chuyên nghiệp chuẩn GT3 (map nhỏ)"""
    def __init__(self, config: Optional[VehicleConfig] = None):
        cfg = config or VehicleConfig(
            name="GT3_RaceCar",
            mass=1100.0,
            max_engine_force=850.0,
            max_brake_force=650.0,
            max_steering_deg=20.0,
            wheel_friction=1.35,
            front_wheel_friction=0.90,  # Giữ đúng vật lý Phase 1 / bản 111s (cầu trước thực tế luôn 0.9, F16)
            max_steer_rate=0.18,
            length=4.50,
            width=1.85,
            max_speed_km_h=120.0,
        )
        super().__init__(cfg)
