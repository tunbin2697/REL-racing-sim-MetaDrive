"""
MetaRacingEnv: Môi trường Đua xe 3D chuẩn RL dựa trên MetaDrive Simulator
- Bản đồ: RacingMap khép kín (12 blocks: Straight, Chicanes, Cua 90°, Cua tay áo 180°, Cua chữ S 220°).
- Cấu hình: 2 làn cùng chiều (lane_num=2, lane_width=4.5m), rào thép GUARDRAIL bao quanh.
- Runoff & Track Limits: Chém lề kerb chỉ giảm tốc do ma sát lề, không phạt điểm; chỉ hủy vòng khi 4 bánh ra ngoài.
- Hàm thưởng Hai Tầng:
  + Tầng Vi Mô (Micro): Tiến độ dọc + Vận tốc hiệu dụng hướng đúng tim đường - Phạt thời gian từng step.
  + Tầng Vĩ Mô (Macro): Thưởng lớn hoàn thành vòng (+100.0) + Thưởng phá kỷ lục Lap Time cá nhân.
- Tích hợp TelemetryTracker: Đo đạc tốc độ, chân phanh, chân ga, góc lái, G-G diagram, và Delta Time thời gian thực.
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.envs.marl_envs.marl_racing_env import RacingMapManager, RACING_CONFIG
from metadrive.constants import PGLineType
from game_env.telemetry import TelemetryTracker


class SingleRacingMetaDrive(MetaDriveEnv):
    """Môi trường MetaDrive đơn tác tử nạp trực tiếp RacingMapManager"""
    @classmethod
    def default_config(cls):
        config = super(SingleRacingMetaDrive, cls).default_config()
        config.update(dict(
            cross_yellow_line_done=False,
            crash_sidewalk_done=False,
        ), allow_add_new_key=True)
        return config

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())


class MetaRacingEnv(gym.Env):
    """
    Gym wrapper tối ưu hóa cho bài toán Đua xe Thể thao Tốc độ cao (Sim Racing RL)
    """
    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(
        self,
        use_render: bool = False,
        window_size=(1280, 720),
        horizon: int = 3000,
        manual_control: bool = False,
        start_seed: int = 42,
        num_scenarios: int = 1,
        reference_lap_file: str = "logs/reference_lap.json",
        **kwargs,
    ):
        super().__init__()
        
        self.use_render = use_render
        self.window_size = window_size
        self.horizon = horizon
        self.manual_control = manual_control
        self.start_seed = start_seed
        self.num_scenarios = num_scenarios
        self.dt = 0.02
        
        # Telemetry tracker
        self.telemetry = TelemetryTracker(reference_lap_file=reference_lap_file)
        self.best_lap_time = self.telemetry.ref_best_lap_time
        
        # Cấu hình MetaDrive engine
        self.config = {
            "use_render": use_render,
            "manual_control": manual_control,
            "traffic_density": 0.0,
            "start_seed": start_seed,
            "num_scenarios": 1,
            "horizon": horizon,
            "window_size": window_size,
            "physics_world_step_size": self.dt,
            "map_config": {
                "lane_num": 2,
                "lane_width": 4.5,
            },
            
            # Tắt toàn bộ ép làn và phạt vạch để tự do cắt góc apex & lấn làn
            "use_lateral_reward": False,
            "cross_yellow_line_done": False,
            "on_continuous_line_done": False,
            "out_of_route_done": False,
            "crash_sidewalk_done": False,   # Chạm lề không bị văng game ngay
            "crash_vehicle_done": False,
            "out_of_road_done": False,       # Track limits được quản lý tùy biến bởi wrapper
            
            # Cảm biến & Đồ họa
            "vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": True,
                "show_line_to_dest": False,
                "lidar": {"num_lasers": 240, "distance": 50},
            },
            "show_interface": use_render,
            "show_logo": False,
            "show_fps": True,
        }
        
        self.env = SingleRacingMetaDrive(self.config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Biến trạng thái
        self.current_step = 0
        self.last_longitudinal = 0.0
        self.lap_completed = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
            
        self.current_step = 0
        self.lap_completed = False
        self.telemetry.reset()
        
        vehicle = self.env.agent
        long, _ = vehicle.lane.local_coordinates(vehicle.position)
        self.last_longitudinal = long
        
        info["lap_time"] = 0.0
        info["best_lap_time"] = self.best_lap_time
        info["delta_time"] = 0.0
        info["delta_color"] = "gray"
        return np.asarray(obs, dtype=np.float32), info

    def check_track_limits_and_apply_runoff(self, vehicle):
        """
        Quản lý lề Runoff & Giới hạn đường đua (Track Limits) chuẩn thể thao:
        - Trên bất kỳ làn nào của đường đua: on_lane = True -> hoàn toàn bình thường.
        - Chém lề Runoff: Xe lệch khỏi mép làn ngoài nhưng chưa đâm rào -> giảm 5% vận tốc do ma sát lề, KHÔNG phạt.
        - Đâm rào thép / văng khỏi đường đua: is_full_out = True -> Dừng episode & phạt.
        """
        on_runoff = False
        is_full_out = False
        
        # 1. Xe hoàn toàn nằm trong các làn đường
        if vehicle.on_lane:
            on_runoff = False
            is_full_out = False
        else:
            # Xe đang liếm lề kerb / runoff ngoài vạch sơn
            if vehicle.crash_sidewalk or vehicle.crash_object:
                # Đâm trực diện vào rào hộ lan (Barrier impact)
                is_full_out = True
            else:
                # Chém lề nhẹ (chưa chạm rào chắn)
                on_runoff = True
                vehicle.set_velocity(vehicle.velocity * 0.95)
                
        # Kiểm tra chạy ngược lùi quá xa
        long, _ = vehicle.lane.local_coordinates(vehicle.position)
        if long < -5.0:
            is_full_out = True
            
        return on_runoff, is_full_out

    def step(self, action):
        self.current_step += 1
        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        vehicle = self.env.agent
        current_time = self.current_step * self.dt
        
        # 1. Kiểm tra Runoff & Track Limits
        on_runoff, is_full_out = self.check_track_limits_and_apply_runoff(vehicle)
        
        # 2. Ghi nhận Telemetry chi tiết
        sample = self.telemetry.log_step(
            current_time=current_time,
            position=vehicle.position,
            velocity=vehicle.velocity,
            steering=action[0],
            throttle_brake=action[1],
            dt=self.dt,
            on_runoff=on_runoff,
        )
        
        # 3. Tính toán tiến độ dọc (Progress)
        current_long, _ = vehicle.lane.local_coordinates(vehicle.position)
        delta_long = current_long - self.last_longitudinal
        self.last_longitudinal = current_long
        
        # Xử lý trường hợp chuyển lane / socket
        if delta_long < -10.0 or delta_long > 50.0:
            delta_long = float(np.linalg.norm(vehicle.velocity[:2])) * self.dt
            
        # 4. HÀM THƯỞNG HAI TẦNG (MICRO + MACRO)
        # --- TẦNG VI MÔ (Mỗi step) ---
        r_progress = 1.5 * delta_long
        
        # Vận tốc hiệu dụng (tốc độ cao + đầu xe hướng đúng đường)
        speed_km_h = sample["speed_km_h"]
        max_speed = vehicle.max_speed_km_h or 120.0
        speed_ratio = speed_km_h / max_speed
        
        heading_diff = vehicle.heading_diff(vehicle.lane)
        r_speed = 0.3 * speed_ratio * max(0.0, float(np.cos(heading_diff)))
        
        # Phạt thời gian trôi qua: Tương đương tối ưu Lap Time nhưng loại bỏ bẫy Credit Assignment!
        r_step_penalty = -0.05
        
        micro_reward = r_progress + r_speed + r_step_penalty
        macro_reward = 0.0
        
        # Kiểm tra hoàn thành vòng đua
        route_comp = info.get("route_completion", 0.0)
        arrive_dest = info.get("arrive_dest", False) or route_comp >= 0.98
        
        # --- TẦNG VĨ MÔ (Kết thúc hoặc Về đích) ---
        if is_full_out or vehicle.crash_vehicle or vehicle.crash_object:
            macro_reward = -30.0
            terminated = True
            info["termination_reason"] = "crash_or_track_limit"
            
        elif arrive_dest:
            self.lap_completed = True
            terminated = True
            info["termination_reason"] = "lap_completed"
            
            # Cúp hoàn thành vòng: Đảm bảo tỷ lệ hoàn thành >= 99%
            r_finish = 100.0
            
            # Thưởng phá kỷ lục Lap Time
            r_record_gap = 0.0
            if current_time < self.best_lap_time:
                time_saved = self.best_lap_time - current_time
                r_record_gap = 10.0 * min(time_saved, 20.0)  # Thưởng tối đa 200 điểm khi phá kỷ lục
                self.best_lap_time = current_time
                self.telemetry.save_as_reference_lap()
                info["new_record"] = True
            else:
                info["new_record"] = False
                
            macro_reward = r_finish + r_record_gap
            
        total_reward = micro_reward + macro_reward
        
        info["step_reward"] = float(total_reward)
        info["micro_reward"] = float(micro_reward)
        info["macro_reward"] = float(macro_reward)
        info["lap_time"] = float(current_time)
        info["best_lap_time"] = float(self.best_lap_time)
        info["speed_km_h"] = float(speed_km_h)
        info["delta_time"] = float(sample["delta_time"])
        info["delta_color"] = sample["delta_color"]
        info["on_runoff"] = bool(on_runoff)
        
        return np.asarray(obs, dtype=np.float32), float(total_reward), terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    @property
    def vehicle(self):
        return self.env.agent
