"""
GhostCar: Thực thể Xe Bóng Ma Đối Chứng (Ghost Car Entity) trong Mô phỏng 3D
Áp dụng Lập trình Hướng đối tượng (OOP) & Single Responsibility Principle (SRP):
- Đóng gói toàn bộ logic khởi tạo, quản lý vòng đời và điều khiển của xe đối chứng (Ghost Car).
- Tích hợp bộ điều khiển OptimalRacingAgent (Pure Pursuit + Feedforward Braking).
- Thu thập và cung cấp telemetry thời gian thực (Ghost Speed, Delta Time, Track s, Trajectory Error).
- Hoàn toàn độc lập với tác tử PPO, đảm bảo tính liêm chính học thuật (không rò rỉ dữ liệu).
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from agents.optimal_agent import OptimalRacingAgent
from game_env.vehicle import GT3RaceCar, BaseRaceVehicle


class GhostCar:
    """
    Thực thể Ghost Car (Bóng ma đối chứng) biểu diễn trên đường đua 3D Panda3D / Bullet.
    Được điều khiển tự động bởi OptimalRacingAgent dựa trên profile tối ưu Convex QP + Lipp & Boyd.
    Tuân thủ chuẩn OOP & SRP: Sử dụng BaseRaceVehicle để quản lý cơ cấu chấp hành và động lực học.
    """
    def __init__(
        self,
        agent: Optional[OptimalRacingAgent] = None,
        optimal_profile_file: Optional[str] = None,
        vehicle: Optional[BaseRaceVehicle] = None,
    ):
        self.agent: OptimalRacingAgent = agent or OptimalRacingAgent(optimal_profile_file=optimal_profile_file)
        self.car: BaseRaceVehicle = vehicle if vehicle is not None else GT3RaceCar()
        self.raw_vehicle: Optional[DefaultVehicle] = None
        self.active: bool = False
        self.step_count: int = 0
        self.lap_time: float = 0.0
        self.lap_completed: bool = False
        self.initial_s: float = 0.0
        self.last_action: Optional[np.ndarray] = None

    def spawn(
        self,
        engine: Any,
        spawn_lane: Any,
        longitudinal: float = 0.0,
        lateral: float = 0.0,
    ) -> bool:
        """
        Khởi tạo xe Ghost Car vào thế giới vật lý của MetaDrive
        """
        self.engine = engine
        if self.raw_vehicle is not None:
            self.destroy(engine)
            
        try:
            pos = spawn_lane.position(longitudinal, lateral)
            heading = float(spawn_lane.heading_theta_at(longitudinal))
            
            # Cấu hình xe Ghost: sử dụng màu đặc biệt (special color) để phân biệt trực quan
            ghost_config = {
                "use_special_color": True,
                "max_speed_km_h": 120.0,
                "max_engine_force": 850.0,
                "max_brake_force": 650.0,
                "wheel_friction": 1.35,
            }
            
            self.raw_vehicle = engine.spawn_object(
                DefaultVehicle,
                vehicle_config=ghost_config,
                position=pos,
                heading=heading,
            )
            
            # Cấu hình lốp xe bám đường chuẩn GT3
            if self.raw_vehicle is not None:
                self.car.attach_agent(self.raw_vehicle)
                wheels = getattr(self.raw_vehicle, "wheels", None)
                if wheels is not None:
                    for wheel in wheels:
                        if hasattr(wheel, "setFrictionSlip"):
                            wheel.setFrictionSlip(float(self.car.cfg.wheel_friction))
                            
                # Vô hiệu hóa va chạm vật lý: Ghost Car là xe bóng ma đối chứng (Hologram/Phantom)
                # Tuyệt đối không được gây va chạm vật lý hay kích hoạt crash_vehicle lên xe chính
                try:
                    from panda3d.core import BitMask32, TransparencyAttrib
                    chassis = getattr(self.raw_vehicle, "chassis", None)
                    if chassis is not None and hasattr(chassis, "node"):
                        chassis.node().setIntoCollideMask(BitMask32.allOff())
                    
                    # Hiệu ứng mờ ảo bán trong suốt chuẩn Ghost Car
                    origin = getattr(self.raw_vehicle, "origin", None)
                    if origin is not None and hasattr(origin, "setTransparency"):
                        origin.setTransparency(TransparencyAttrib.M_alpha)
                        origin.setAlphaScale(0.70)
                except Exception as ex:
                    print(f">> [GHOST CAR] Cảnh báo cấu hình xuyên thấu: {ex}")
                    
            self.agent.reset()
            self.active = True
            self.step_count = 0
            self.lap_time = 0.0
            self.lap_completed = False
            self.initial_s = float(longitudinal)
            return True
        except Exception as e:
            print(f">> [GHOST CAR] Lỗi khởi tạo Ghost Car: {e}")
            self.active = False
            self.raw_vehicle = None
            return False

    def spawn_from_env(self, env: Any, longitudinal: float = 0.0, lateral: float = 2.25) -> bool:
        """Helper OOP tự động trích xuất engine và spawn_lane từ MetaRacingEnv an toàn tuyệt đối"""
        try:
            raw_env = getattr(env, "env", env)
            engine = getattr(raw_env, "engine", None)
            map_obj = getattr(raw_env, "current_map", None)
            agent = getattr(raw_env, "agent", None)
            if engine is None or map_obj is None or agent is None:
                return False
            road_network = getattr(map_obj, "road_network", None)
            if road_network is None:
                return False
            lane = road_network.get_lane(agent.lane_index)
            if lane is None:
                return False
            return self.spawn(engine, lane, longitudinal=longitudinal, lateral=lateral)
        except Exception as e:
            print(f">> [GHOST CAR] Lỗi spawn_from_env: {e}")
            return False

    def step(self, dt: float = 0.10) -> Optional[np.ndarray]:
        """
        Thực hiện một chu kỳ tính toán và áp dụng hành động lái cho Ghost Car:
        1. Gọi OptimalRacingAgent tính toán action 3 trục chuẩn [steer, throttle, brake]
        2. Ủy quyền cho BaseRaceVehicle xử lý cơ cấu chấp hành (OOP - DRY)
        3. Gửi lệnh chấp hành xuống Bullet raycast vehicle
        """
        if not self.active or self.raw_vehicle is None:
            return None
            
        self.step_count += 1
        self.lap_time += dt
        
        # 1. Tính toán hành động tối ưu 3 trục [steer, throttle, brake]
        action, _, _ = self.agent.select_action(vehicle=self.raw_vehicle)
        self.last_action = action
        
        # 2. Xử lý chuẩn OOP qua BaseRaceVehicle (DRY - Không trùng lặp code)
        action_applied, delta_steer, net_pedal = self.car.process_action(action)
        
        # 3. Áp dụng vào Bullet raycast vehicle
        self.raw_vehicle.before_step(action_applied)
        return action

    def destroy(self, engine: Any = None):
        """Dọn dẹp xe Ghost Car khỏi engine khi kết thúc hoặc reset episode"""
        eng = engine or getattr(self, "engine", None)
        if self.raw_vehicle is not None:
            veh_id = getattr(self.raw_vehicle, "id", None)
            if eng is not None and hasattr(eng, "clear_objects") and veh_id is not None:
                try:
                    if veh_id in getattr(eng, "_spawned_objects", {}):
                        eng.clear_objects([veh_id], force_destroy=True)
                except Exception:
                    pass
            elif hasattr(self.raw_vehicle, "destroy"):
                try:
                    self.raw_vehicle.destroy()
                except Exception:
                    pass
            self.raw_vehicle = None
            self.car.raw_vehicle = None
        self.active = False

    @property
    def position(self) -> np.ndarray:
        return self.car.position

    @property
    def speed_km_h(self) -> float:
        return self.car.speed_km_h

    @property
    def heading_theta(self) -> float:
        return self.car.heading_theta

    def get_telemetry(
        self,
        main_pos: Optional[np.ndarray] = None,
        main_heading: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Trả về thông tin telemetry của Ghost Car phục vụ hiển thị HUD:
        - Tổng khoảng cách gap (m)
        - Vị trí tương quan: Đang bám đuôi phía sau (is_behind) hay chạy phía trước
        """
        telemetry = {
            "ghost_active": self.active,
            "ghost_speed_km_h": self.speed_km_h,
            "ghost_lap_time": self.lap_time,
            "ghost_pos": self.position,
        }
        if main_pos is not None and self.active:
            diff = self.position - np.array(main_pos[:2], dtype=np.float32)
            dist = float(np.linalg.norm(diff))
            telemetry["gap_distance_m"] = dist
            if main_heading is not None:
                cos_h = np.cos(main_heading)
                sin_h = np.sin(main_heading)
                fwd = np.array([cos_h, sin_h], dtype=np.float32)
                long_gap = float(np.dot(diff, fwd))
                telemetry["longitudinal_gap_m"] = long_gap
                telemetry["is_behind"] = bool(long_gap < 0)
        return telemetry
