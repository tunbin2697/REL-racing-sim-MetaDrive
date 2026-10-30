"""
Động Lực Học Xe Đua Thể Thao (Race Vehicle Dynamics & Actuation)
Áp dụng nguyên lý Single Responsibility (SRP) và Lập trình hướng đối tượng (OOP):
- Tách biệt hoàn toàn phần Cơ khí / Động lực học (Vehicle Dynamics) ra khỏi Trọng tài Môi trường (RacingEnv).
- Quản lý lực phanh, lực kéo, độ bám lốp, tỷ số truyền lái, bộ lọc làm mịn góc lái (Slew Rate Limiter).
- Cung cấp sẵn các mẫu xe: GT3RaceCar (xe đua thể thao), F1RaceCar (xe công thức 340 km/h) và DefaultCivilianSedan (xe dân dụng đối chứng).
"""

import copy
import weakref
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from panda3d.core import Vec3


@dataclass
class VehicleConfig:
    """Thông số kỹ thuật động lực học của xe đua"""
    name: str = "GT3_RaceCar"
    mass: float = 1100.0               # Khối lượng thân xe (kg)
    max_engine_force: float = 850.0    # Lực kéo cực đại của động cơ (N)
    max_brake_force: float = 650.0     # Lực phanh cực đại (N) - Gấp 5.5 lần xe sedan
    max_steering_deg: float = 20.0     # Góc bẻ lái cực đại (độ) - Chuẩn xe đua đầm chắc
    wheel_friction: float = 1.35       # Hệ số ma sát lốp Bullet (Racing Slicks)
    max_steer_rate: float = 0.18       # Tốc độ quay vô-lăng tối đa mỗi bước 0.1s (Slew Rate Limit)
    length: float = 4.50               # Chiều dài xe (m)
    width: float = 1.85                # Chiều rộng xe (m)
    max_speed_km_h: float = 120.0      # Vận tốc giới hạn trần (km/h)
    downforce_coef: float = 0.0        # Lực ép khí động học tại 200 km/h tính theo số lần trọng lượng xe (tỉ lệ v^2)
    max_suspension_force: Optional[float] = None  # Giới hạn lực lò xo mỗi bánh (N); Bullet mặc định 6000N -> chạm đáy khi có downforce
    roll_influence: Optional[float] = None        # Độ nghiêng thân xe khi vào cua (MetaDrive mặc định 0.5)


# Các xe đang có lực ép khí động học; lực được áp ở MỌI bước vật lý 0.02s (Bullet xóa lực sau mỗi bước)
_DOWNFORCE_CARS = weakref.WeakSet()


def _install_downforce_hook(engine):
    """Bọc engine.step_physics_world để áp downforce cho các xe đăng ký trước mỗi bước vật lý (cài 1 lần mỗi engine)"""
    if engine is None or getattr(engine, "_race_downforce_hooked", False):
        return
    step_physics_world = engine.step_physics_world

    def step_with_downforce():
        for car in list(_DOWNFORCE_CARS):
            car.apply_downforce()
        step_physics_world()

    engine.step_physics_world = step_with_downforce
    engine._race_downforce_hooked = True


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

    def clone(self) -> "BaseRaceVehicle":
        """Tạo thực thể xe độc lập mới với cùng cấu hình cơ khí (OOP Prototype pattern)"""
        new_cfg = copy.deepcopy(self.cfg)
        return self.__class__(config=new_cfg)

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
        if hasattr(self.raw_vehicle, "wheels"):
            for wheel in self.raw_vehicle.wheels:
                wheel.setFrictionSlip(float(self.cfg.wheel_friction))
                if self.cfg.max_suspension_force is not None:
                    wheel.setMaxSuspensionForce(float(self.cfg.max_suspension_force))
                if self.cfg.roll_influence is not None:
                    wheel.setRollInfluence(float(self.cfg.roll_influence))

        if self.cfg.downforce_coef > 0.0:
            from metadrive.engine.engine_utils import get_engine
            _DOWNFORCE_CARS.add(self)
            _install_downforce_hook(get_engine())

    def apply_downforce(self):
        """Lực ép xuống F = coef * m * g * (v / 200 km/h)^2 đặt tại trọng tâm (mô phỏng cánh gió F1)"""
        raw = self.raw_vehicle
        body = getattr(raw, "body", None) if raw is not None else None
        if body is None:
            return
        try:
            # Chặn tốc độ ở trần thiết kế: va chạm mạnh có thể làm vận tốc vọt lên, lực ~v^2 sẽ khuếch đại thành NaN
            speed = float(raw.speed_km_h)
            if not np.isfinite(speed):
                return
            ratio = min(speed, self.cfg.max_speed_km_h) / 200.0
            body.applyCentralForce(Vec3(0.0, 0.0, -self.cfg.downforce_coef * self.cfg.mass * 9.81 * ratio * ratio))
        except (AssertionError, AttributeError):
            # Xe đã bị hủy khỏi engine
            pass

    def process_action(self, raw_action) -> Tuple[np.ndarray, float, float]:
        """
        Xử lý action thô từ policy RL:
        - Giới hạn tốc độ đánh lái (Slew Rate Limiter) chống rung lắc 10Hz
        - Chuyển đổi [steer, pedal] sang định dạng chấp hành của MetaDrive
        Trả về: (action_applied, delta_steer, net_pedal)
        """
        raw_action = np.clip(np.array(raw_action, dtype=np.float32), -1.0, 1.0)
        steer = float(raw_action[0])
        
        if len(raw_action) >= 3:
            thr = 0.5 * (float(raw_action[1]) + 1.0)
            brk = 0.5 * (float(raw_action[2]) + 1.0)
            net_pedal = thr * (1.0 - brk) - brk if brk > 0.05 else thr
        else:
            net_pedal = float(raw_action[1])
            
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
        Thông số xe đua GT3:
        - Chiều dài cơ sở (Wheelbase): 2.75m -> Trục trước/sau cách trọng tâm +-1.375m
        - Độ rộng cơ sở (Wheel Track): 1.65m -> Tâm bánh xe cách trục giữa +-0.825m
        - Bề rộng lốp đua: 0.30m (bán kính lốp 0.15m)
        """
        if self.raw_vehicle is None:
            return []
        pos = np.array(self.raw_vehicle.position[:2], dtype=np.float32)
        heading = float(self.raw_vehicle.heading_theta)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        fwd = np.array([cos_h, sin_h], dtype=np.float32)
        lat = np.array([-sin_h, cos_h], dtype=np.float32)
        
        l_front = 1.375
        l_rear = -1.375
        w_track = 0.825
        
        return [
            pos + l_front * fwd + w_track * lat,  # Bánh trước Trái (Front-Left)
            pos + l_front * fwd - w_track * lat,  # Bánh trước Phải (Front-Right)
            pos + l_rear * fwd + w_track * lat,   # Bánh sau Trái (Rear-Left)
            pos + l_rear * fwd - w_track * lat,   # Bánh sau Phải (Rear-Right)
        ]

    def get_bounding_corners(self) -> List[np.ndarray]:
        """
        Tính toán tọa độ 2D của 4 góc cản xe (Bounding Box thân xe 4.50m x 1.85m)
        """
        if self.raw_vehicle is None:
            return []
        pos = np.array(self.raw_vehicle.position[:2], dtype=np.float32)
        heading = float(self.raw_vehicle.heading_theta)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        fwd = np.array([cos_h, sin_h], dtype=np.float32)
        lat = np.array([-sin_h, cos_h], dtype=np.float32)
        l = 0.5 * self.cfg.length
        w = 0.5 * self.cfg.width
        return [
            pos + l * fwd + w * lat,
            pos + l * fwd - w * lat,
            pos - l * fwd + w * lat,
            pos - l * fwd - w * lat,
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
        return np.array(self.raw_vehicle.position[:2], dtype=np.float32) if self.raw_vehicle else np.zeros(2, dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return np.array(self.raw_vehicle.velocity[:2], dtype=np.float32) if self.raw_vehicle else np.zeros(2, dtype=np.float32)

    @property
    def speed_m_s(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def speed_km_h(self) -> float:
        return float(self.speed_m_s * 3.6)

    @property
    def heading_theta(self) -> float:
        return float(self.raw_vehicle.heading_theta) if self.raw_vehicle else 0.0

    @property
    def origin(self):
        return getattr(self.raw_vehicle, "origin", None) if self.raw_vehicle else None

    @property
    def chassis(self):
        return getattr(self.raw_vehicle, "chassis", None) if self.raw_vehicle else None

    @property
    def yaw_rate(self) -> float:
        if self.raw_vehicle is not None and hasattr(self.raw_vehicle, "body") and hasattr(self.raw_vehicle.body, "get_angular_velocity"):
            return float(self.raw_vehicle.body.get_angular_velocity()[2])
        return float(getattr(self.raw_vehicle, "yaw_rate", 0.0)) if self.raw_vehicle else 0.0

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
    """Cấu hình xe đua thể thao chuyên nghiệp chuẩn GT3"""
    def __init__(self, config: Optional[VehicleConfig] = None):
        cfg = config or VehicleConfig(
            name="GT3_RaceCar",
            mass=1100.0,
            max_engine_force=850.0,
            max_brake_force=650.0,
            max_steering_deg=20.0,
            wheel_friction=1.35,
            max_steer_rate=0.18,
            length=4.50,
            width=1.85,
            max_speed_km_h=120.0,
        )
        super().__init__(cfg)


class F1RaceCar(BaseRaceVehicle):
    """Cấu hình xe công thức F1: trần 340 km/h, lực kéo lớn (0-100 ~3.0s, 0-300 ~10s trên đường thẳng)
    MetaDrive không có khí động học và tăng wheel_friction trên ~1.35 không tăng độ bám (trần ~0.86G cua / ~1.5G phanh),
    nên mô phỏng lực ép khí động học (downforce ~ v^2) + nâng giới hạn lò xo + giảm nghiêng thân xe:
    phanh ~3.8G, cua ~1.16G @120 km/h / ~1.59G @180 km/h, thân xe nghiêng tối đa ~5 độ (đo trên đường thẳng dài)."""
    def __init__(self, config: Optional[VehicleConfig] = None):
        cfg = config or VehicleConfig(
            name="F1_RaceCar",
            mass=1100.0,
            max_engine_force=2500.0,
            max_brake_force=650.0,
            max_steering_deg=20.0,
            wheel_friction=1.35,
            max_steer_rate=0.18,
            length=4.50,
            width=1.85,
            max_speed_km_h=340.0,
            downforce_coef=1.0,
            max_suspension_force=60000.0,
            roll_influence=0.2,
        )
        super().__init__(cfg)


class DefaultCivilianSedan(BaseRaceVehicle):
    """Cấu hình xe sedan dân dụng nguyên bản của MetaDrive (dành cho đối chứng / Ablation Study)"""
    def __init__(self, config: Optional[VehicleConfig] = None):
        cfg = config or VehicleConfig(
            name="Default_Civilian_Sedan",
            mass=1100.0,
            max_engine_force=789.2,
            max_brake_force=119.2,
            max_steering_deg=40.0,
            wheel_friction=0.90,
            max_steer_rate=0.35,
            length=4.50,
            width=1.85,
            max_speed_km_h=120.0,
        )
        super().__init__(cfg)


def get_vehicle(name: Optional[str] = "gt3") -> BaseRaceVehicle:
    """
    Factory Function (Mẫu thiết kế Factory):
    Tạo đối tượng xe đua tương ứng theo tên định danh.
    Hỗ trợ: 'gt3', 'racing', 'f1', 'sedan', 'civilian'
    """
    if name is None or str(name).lower() in ["gt3", "gt3_racecar", "racing"]:
        return GT3RaceCar()
    elif str(name).lower() in ["f1", "f1_racecar", "formula1"]:
        return F1RaceCar()
    elif str(name).lower() in ["sedan", "default", "civilian"]:
        return DefaultCivilianSedan()
    else:
        raise ValueError(f"Không nhận diện được loại xe '{name}'. Chọn 'gt3', 'f1' hoặc 'sedan'.")
