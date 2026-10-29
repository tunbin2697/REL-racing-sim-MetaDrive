"""
MetaRacingEnv: Môi trường Đua xe 3D chuẩn RL tối ưu hóa cao cho MetaDrive Simulator
- Không gian quan sát 40 chiều (Compact Racing State - Zero Data Leakage):
  + 16 chiều Ego Kinematics, IMU & Controls: Vận tốc dọc/ngang, Yaw rate, Body slip beta, Gia tốc dọc/ngang IMU (G), Roll/Pitch, Hành động bước trước, Tốc độ xoay vô lăng, Cờ lấn lề, Độ lệch ngang tim đường, Sin/Cos góc lệch hướng.
  + 24 chiều Tầm nhìn góc cua (Lookahead Horizon): 6 mốc [10m, 25m, 45m, 70m, 105m, 150m] với tọa độ tương đối dx, dy, sin góc lệch hướng, và độ cong cua kappa.
- Track Limits chuẩn thể thao FIA 4 bánh:
  + Bỏ hoàn toàn gờ Kerb ảo, lấy chuẩn đúng mép đường nhựa thật 4.5m.
  + Vòng chạy hợp lệ khi còn ít nhất 1 lốp xe trên mặt nhựa (<= 4.5m), chỉ dừng cuộc đua khi cả 4 lốp hoàn toàn rơi khỏi mặt nhựa.
- Đo đạc quãng đường chuẩn Frenet s: Miễn nhiễm với hiện tượng trượt drift hoặc quay vòng.
- Hàm thưởng Markovian: Thưởng tiến độ Frenet s, phạt lắc tay lái (-0.10 * delta_steer^2), phạt đứng yên, áp lực thời gian tùy chọn (time_penalty/bước), và thưởng cán đích hàm bậc 3 theo thời gian vòng đua.
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.constants import TerrainProperty
TerrainProperty.map_region_size = 2048

from metadrive.envs.marl_envs.marl_racing_env import RacingMapManager
from game_env.silverstone_map import SilverstoneMapManager
from game_env.telemetry import TelemetryTracker
from game_env.vehicle import BaseRaceVehicle, GT3RaceCar, DefaultCivilianSedan
from game_env.benchmark import RacingBenchmarkEngine
from typing import Optional, Dict


class SingleRacingMetaDrive(MetaDriveEnv):
    """Môi trường MetaDrive đơn tác tử hỗ trợ nạp SilverstoneMapManager hoặc RacingMapManager"""
    @classmethod
    def default_config(cls):
        config = super(SingleRacingMetaDrive, cls).default_config()
        config.update(dict(
            show_sidewalk=False,
            cross_yellow_line_done=False,
            crash_sidewalk_done=False,
            cull_lanes_outside_map=False,
            map_type="silverstone",
        ), allow_add_new_key=True)
        return config

    def setup_engine(self):
        super().setup_engine()
        map_type = self.config.get("map_type", "silverstone")
        if map_type == "silverstone":
            self.engine.update_manager("map_manager", SilverstoneMapManager())
        else:
            self.engine.update_manager("map_manager", RacingMapManager())

    @staticmethod
    def _is_arrive_destination(vehicle):
        # Trọng tài MetaRacingEnv quản lý hoàn thành vòng đua độc lập qua vạch đích Start/Finish
        return False

    def reset(self, *args, **kwargs):
        # Fail-safe: Tự động dọn dẹp các đối tượng ngoại lai (như Ghost Car) trước khi reset engine
        if hasattr(self, "engine") and self.engine is not None and hasattr(self.engine, "clear_objects"):
            try:
                from metadrive.component.vehicle.base_vehicle import BaseVehicle
                managed_ids = set()
                agent_mgr = getattr(self.engine, "agent_manager", None)
                if agent_mgr is not None and hasattr(agent_mgr, "spawned_objects"):
                    managed_ids.update(getattr(agent_mgr, "spawned_objects", {}).keys())
                extra_objs = [
                    obj_id for obj_id, obj in getattr(self.engine, "_spawned_objects", {}).items()
                    if isinstance(obj, BaseVehicle) and obj_id not in managed_ids
                ]
                if extra_objs:
                    self.engine.clear_objects(extra_objs, force_destroy=True)
            except Exception:
                pass

        ret = super().reset(*args, **kwargs)
        # Đưa tâm của Terrain và Semantic texture về đúng trọng tâm của Silverstone Circuit (411m, -33m)
        if self.current_map is not None and hasattr(self.engine, "terrain") and self.engine.terrain is not None:
            center_p = self.current_map.get_center_point()
            self.engine.terrain.reset(center_p)
        return ret


class MetaRacingEnv(gym.Env):
    """
    Gym wrapper tối ưu hóa cho bài toán Đua xe Thể thao Tốc độ cao (Sim Racing RL)
    """
    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(
        self,
        config: Optional[dict] = None,
        use_render: bool = False,
        window_size=(1280, 720),
        horizon: int = 6000,
        manual_control: bool = False,
        start_seed: int = 42,
        num_scenarios: int = 1,
        map_type: str = "silverstone",
        max_speed_km_h: float = 120.0,
        vehicle: Optional[BaseRaceVehicle] = None,
        max_laps: int = 1,
        time_penalty: float = 0.0,
        lane_num: int = 2,
        lane_width: float = 4.5,
        **kwargs,
    ):
        super().__init__()
        # Hình học mặt đường: làn 0 ở lateral 0, các làn sau xếp về phía lateral dương
        # -> tim đường tại lateral (N-1)*w/2, mép nhựa cách tim N*w/2 (mặc định 2 làn: 2.25m / 4.5m)
        self.lane_num = int(lane_num)
        self.lane_width = float(lane_width)
        self.center_lat = 0.5 * (self.lane_num - 1) * self.lane_width
        self.half_width = 0.5 * self.lane_num * self.lane_width
        # Áp lực thời gian r_time = -time_penalty mỗi bước 0.1s: tổng phạt tỉ lệ thuận với lap time
        self.time_penalty = float(time_penalty)
        
        # Thực thể phương tiện đua xe chuyên biệt (OOP Race Car Entity)
        self.car: BaseRaceVehicle = vehicle if vehicle is not None else GT3RaceCar()
        
        self.use_render = use_render
        self.window_size = window_size
        self.horizon = horizon
        self.manual_control = manual_control
        self.start_seed = start_seed
        self.num_scenarios = num_scenarios
        self.dt = 0.10
        self.max_speed_km_h = float(self.car.cfg.max_speed_km_h)
        if "map_config" in kwargs and isinstance(kwargs["map_config"], str):
            map_type = kwargs["map_config"]
        self.map_type = map_type
        
        # Telemetry tracker & Kỷ lục
        self.telemetry = TelemetryTracker()
        self.best_lap_time = float("inf")
        self.total_finishes = 0
        self._benchmark_engine: Optional[RacingBenchmarkEngine] = None
        
        # Cấu hình MetaDrive engine (Tắt 240 tia LiDAR chết để tăng tốc độ mô phỏng)
        self.config = {
            "use_render": use_render,
            "manual_control": manual_control,
            "traffic_density": 0.0,
            "start_seed": start_seed,
            "num_scenarios": num_scenarios,
            "horizon": horizon,
            "window_size": window_size,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "cull_lanes_outside_map": False,
            "map_type": map_type,
            "map_config": {
                "lane_num": self.lane_num,
                "lane_width": self.lane_width,
            },
            
            # Tắt tường bê tông rào chắn và phạt vạch để áp dụng cơ chế Runoff thể thao
            "show_sidewalk": False,
            "use_lateral_reward": False,
            "cross_yellow_line_done": False,
            "on_continuous_line_done": False,
            "out_of_route_done": False,
            "crash_sidewalk_done": False,
            "crash_vehicle_done": False,
            "out_of_road_done": False,
            
            # Cảm biến & Đồ họa
            "vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": True,
                "show_line_to_dest": False,
                "lidar": {"num_lasers": 0, "distance": 0},
            },
            "show_interface": use_render,
            "show_logo": False,
            "show_fps": True,
        }
        
        self.env = SingleRacingMetaDrive(self.config)
        
        # 40 chiều Compact Racing State (16 Ego Kinematics/IMU/Controls + 24 Centerline Lookahead)
        self.obs_dim = 40
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # 3 chiều Continuous Racing Action: [steer, throttle, brake]
        # steer: [-1, 1], throttle: [-1, 1] -> [0, 1], brake: [-1, 1] -> [0, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        
        # Dữ liệu hình học đường đua dạng Polyline
        self.circuit_initialized = False
        self.track_xy: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.track_theta: np.ndarray = np.zeros(1, dtype=np.float32)
        self.track_kappa: np.ndarray = np.zeros(1, dtype=np.float32)
        self.track_s: np.ndarray = np.zeros(1, dtype=np.float32)
        self.circuit_length = 2852.3
        self.num_track_points = 1
        self.current_track_idx = 0
        self.last_longitudinal = 0.0
        
        # Biến trạng thái & Đua đa vòng (Multi-lap Racing)
        self.max_laps = int(config.get("max_laps", max_laps)) if isinstance(config, dict) else int(max_laps)
        self.current_lap = 1
        self.lap_start_step = 0
        self.last_pos = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        self.last_s = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_acc_long_g = 0.0
        self.last_acc_lat_g = 0.0
        self.lap_completed = False
        self.episode_history = []
        self.last_crash_history = []
        self.stagnant_steps = 0
        self.is_flying_start = False

    def _init_circuit_polyline(self):
        """Khởi tạo mảng tọa độ tim đường đua (Dense Centerline Polyline) để tính toán nhanh trong O(1)"""
        map_obj = self.env.current_map
        if map_obj is None or not hasattr(map_obj, "road_network") or map_obj.road_network is None:
            return
            
        vehicle = self.env.agent
        if vehicle is None or not hasattr(vehicle, "navigation") or vehicle.navigation is None:
            return
            
        ckpts = vehicle.navigation.checkpoints
        route_lanes = []
        for i in range(len(ckpts) - 1):
            lane = map_obj.road_network.get_lane((ckpts[i], ckpts[i + 1], 0))
            if lane is not None:
                route_lanes.append(lane)
                
        if not route_lanes:
            return
            
        self.circuit_length = float(sum(l.length for l in route_lanes))
        track_points = []
        for lane in route_lanes:
            s_samples = np.linspace(0, lane.length, max(2, int(lane.length)), endpoint=False)
            kappa = 0.0
            if hasattr(lane, "radius") and lane.radius > 0:
                is_cw = lane.is_clockwise() if callable(getattr(lane, "is_clockwise", None)) else bool(getattr(lane, "is_clockwise", False))
                r_mid = lane.radius + self.center_lat
                kappa = (1.0 / float(r_mid)) * (-1.0 if is_cw else 1.0)

            for s_l in s_samples:
                # Tim đường thực tế của mặt đường N làn (mặc định 2 làn: làn 0 từ -2.25 đến +2.25, làn 1 từ +2.25 đến +6.75)
                # Điểm tim chuẩn chính xác nằm tại lateral = center_lat (2 làn: 2.25)
                pos = lane.position(s_l, self.center_lat)
                theta = float(lane.heading_theta_at(s_l))
                track_points.append({
                    "pos": np.array(pos[:2], dtype=np.float32),
                    "theta": theta,
                    "kappa": float(kappa),
                })
            
        # Định vị vạch xuất phát / cán đích (Start/Finish Grid) tại x = 60.0m trên Hamilton Straight
        # Khớp chuẩn với vị trí vạch xuất phát thực tế của F1 Silverstone GP
        dists_to_grid = np.sum((np.array([p["pos"] for p in track_points]) - np.array([60.0, self.center_lat])) ** 2, axis=1)
        grid_k = int(dists_to_grid.argmin())
        if grid_k > 0:
            track_points = track_points[grid_k:] + track_points[:grid_k]

        self.track_xy = np.array([p["pos"] for p in track_points], dtype=np.float32)
        self.track_theta = np.array([p["theta"] for p in track_points], dtype=np.float32)
        self.track_kappa = np.array([p["kappa"] for p in track_points], dtype=np.float32)
        
        # Khoảng cách s Frenet chuẩn xác theo chiều dài hình học tim đường
        diffs = np.linalg.norm(np.diff(self.track_xy, axis=0), axis=1)
        self.track_s = np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)
        self.circuit_length = float(self.track_s[-1])
        self.num_track_points = len(track_points)
        self.circuit_initialized = True

    def _get_track_s(self, pos, hint_idx=None):
        """Tìm chỉ số điểm tim đường gần nhất và trả về tọa độ Frenet s"""
        if not self.circuit_initialized or self.num_track_points <= 1:
            self._init_circuit_polyline()
            
        pos_2d = np.array(pos[:2], dtype=np.float32)
        if hint_idx is None or self.num_track_points <= 1:
            dists = np.sum((self.track_xy - pos_2d) ** 2, axis=1)
            idx = int(dists.argmin())
        elif hint_idx < 40:
            # Gần vạch xuất phát: KHÔNG cho phép tìm lùi vòng ra đuôi đường đua (chống bug về đích ảo)
            indices = np.arange(0, min(self.num_track_points, hint_idx + 60))
            dists = np.sum((self.track_xy[indices] - pos_2d) ** 2, axis=1)
            idx = int(indices[dists.argmin()])
        else:
            window = 60
            indices = np.arange(hint_idx - 15, hint_idx + window) % self.num_track_points
            dists = np.sum((self.track_xy[indices] - pos_2d) ** 2, axis=1)
            idx = int(indices[dists.argmin()])
            
        return idx, float(self.track_s[idx])

    def get_compact_racing_obs(
        self,
        vehicle,
        action_applied=None,
        delta_steer: float = 0.0,
        on_runoff: bool = False,
        state=None,
    ) -> np.ndarray:
        """
        Xây dựng vector quan sát 40 chiều (Compact Racing State) - Zero Data Leakage:
        - 16 chiều Ego Kinematics, IMU & Controls:
          + [vx/40, vy/20, yaw_rate/pi, body_slip_beta]
          + [acc_long_g, acc_lat_g, roll/0.35, pitch/0.35]
          + [prev_steer, prev_throttle, prev_brake, delta_steer]
          + [runoff_flag, lat_offset/4.5, sin_dh, cos_dh]
        - 24 chiều Lookahead Horizon dọc tim đường (6 waypoints x 4 values):
          + [dx_rel/150, dy_rel/40, sin(d_theta), kappa*50] tại các mốc [10, 25, 45, 70, 105, 150]m
        state: đối tượng giữ trạng thái riêng của từng xe (current_track_idx, last_action, last_acc_long_g,
          last_acc_lat_g); mặc định là chính env (đơn tác tử). Môi trường nhiều xe truyền trạng thái của từng xe.
        """
        if state is None:
            state = self
        if action_applied is None:
            action_applied = state.last_action

        pos = np.array(vehicle.position[:2], dtype=np.float32)
        heading = float(vehicle.heading_theta)
        vel = np.array(vehicle.velocity[:2], dtype=np.float32)
        speed_m_s = float(np.linalg.norm(vel))

        # Tìm vị trí tim đường hiện tại
        idx, current_s = self._get_track_s(pos, state.current_track_idx)
        state.current_track_idx = idx
        track_pt = self.track_xy[idx]
        track_heading = float(self.track_theta[idx])
        
        # Hệ trục xe (Ego Body Frame)
        cos_head = np.cos(heading)
        sin_head = np.sin(heading)
        fwd_vec = np.array([cos_head, sin_head], dtype=np.float32)
        lat_vec = np.array([-sin_head, cos_head], dtype=np.float32)
        
        # Vận tốc dọc và ngang
        vx = float(np.dot(vel, fwd_vec))
        vy = float(np.dot(vel, lat_vec))
        
        # Góc lệch hướng so với tiếp tuyến tim đường
        delta_heading = (heading - track_heading + np.pi) % (2.0 * np.pi) - np.pi
        cos_dh = float(np.cos(delta_heading))
        sin_dh = float(np.sin(delta_heading))
        
        # Khoảng cách lệch ngang so với tim đường
        disp_center = pos - track_pt
        track_lat_normal = np.array([-np.sin(track_heading), np.cos(track_heading)], dtype=np.float32)
        lat_offset = float(np.dot(disp_center, track_lat_normal))
        
        # Yaw rate
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0))
        
        # Góc trượt thân xe beta (body slip angle)
        if speed_m_s > 0.5:
            body_slip = float(np.arctan2(vy, max(0.1, vx)) / (np.pi / 4.0))
        else:
            body_slip = 0.0
        body_slip = float(np.clip(body_slip, -1.5, 1.5))
        
        # Góc roll và pitch từ Panda3D origin
        roll_val = 0.0
        pitch_val = 0.0
        if hasattr(vehicle, "origin") and vehicle.origin is not None:
            try:
                hpr = vehicle.origin.get_hpr()
                pitch_val = float(np.radians(hpr[1]))
                roll_val = float(np.radians(hpr[2]))
            except Exception:
                pass
                
        # Hành động bước trước
        act_arr = np.array(state.last_action, dtype=np.float32)
        prev_steer = float(act_arr[0]) if len(act_arr) > 0 else 0.0
        prev_throttle = float(act_arr[1]) if len(act_arr) > 1 else 0.0
        prev_brake = float(act_arr[2]) if len(act_arr) > 2 else 0.0
        
        # --- 16 CHIỀU EGO KINEMATICS, IMU & CONTROLS ---
        ego_state = [
            float(np.clip(vx / 40.0, -1.0, 1.5)),               # 1. Vận tốc dọc thân xe
            float(np.clip(vy / 20.0, -1.5, 1.5)),               # 2. Vận tốc ngang thân xe
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),      # 3. Yaw rate quanh trục Z
            float(body_slip),                                   # 4. Góc trượt thân xe beta
            float(np.clip(state.last_acc_long_g, -2.5, 2.5)),   # 5. Gia tốc dọc IMU (G)
            float(np.clip(state.last_acc_lat_g, -2.5, 2.5)),    # 6. Gia tốc ngang IMU (G)
            float(np.clip(roll_val / 0.35, -2.0, 2.0)),         # 7. Góc nghiêng thân xe roll
            float(np.clip(pitch_val / 0.35, -2.0, 2.0)),        # 8. Góc chúc/ngóc đầu pitch
            float(np.clip(prev_steer, -1.0, 1.0)),              # 9. Vô lăng bước trước
            float(np.clip(prev_throttle, -1.0, 1.0)),           # 10. Chân ga bước trước
            float(np.clip(prev_brake, -1.0, 1.0)),              # 11. Chân phanh bước trước
            float(np.clip(delta_steer, -1.0, 1.0)),             # 12. Tốc độ xoay vô lăng
            float(1.0 if on_runoff else 0.0),                   # 13. Cờ lấn lề Runoff
            float(np.clip(lat_offset / self.half_width, -2.0, 2.0)),  # 14. Độ lệch ngang so với tim đường (theo nửa bề rộng)
            float(sin_dh),                                      # 15. Sin góc lệch hướng
            float(cos_dh),                                      # 16. Cos góc lệch hướng
        ]
        
        # --- 24 CHIỀU LOOKAHEAD HORIZON (6 WAYPOINTS X 4) ---
        lookahead_distances = [10.0, 25.0, 45.0, 70.0, 105.0, 150.0]
        lookahead_features = []
        for dist_ahead in lookahead_distances:
            target_s = (current_s + dist_ahead) % self.circuit_length
            wpt_idx = int(np.searchsorted(self.track_s, target_s)) % self.num_track_points
            wpt_pos = self.track_xy[wpt_idx]
            wpt_theta = float(self.track_theta[wpt_idx])
            wpt_kappa = float(self.track_kappa[wpt_idx])
            
            d_vec = wpt_pos - pos
            dx_rel = float(np.dot(d_vec, fwd_vec))
            dy_rel = float(np.dot(d_vec, lat_vec))
            d_theta = (wpt_theta - heading + np.pi) % (2.0 * np.pi) - np.pi
            
            lookahead_features.extend([
                float(np.clip(dx_rel / 150.0, -1.0, 1.5)),
                float(np.clip(dy_rel / 40.0, -2.0, 2.0)),
                float(np.sin(d_theta)),
                float(np.clip(wpt_kappa * 50.0, -2.0, 2.0)),
            ])
            
        obs = np.array(ego_state + lookahead_features, dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _, info = self.env.reset(seed=seed)
        else:
            _, info = self.env.reset()
            
        if not self.circuit_initialized:
            self._init_circuit_polyline()
            
        self.current_step = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_acc_long_g = 0.0
        self.last_acc_lat_g = 0.0
        self.lap_completed = False
        self.telemetry.reset()
        self.telemetry.last_completed_lap_data = []
        self.episode_history = []
        self.stagnant_steps = 0
        self.current_track_idx = 0
        self.current_lap = 1
        self.lap_start_step = 0
        vehicle = self.env.agent
        assert vehicle is not None, "Vehicle agent was not properly spawned in MetaDrive"
        self.car.attach_agent(vehicle)
        
        # Xác định chế độ xuất phát:
        # - Nếu truyền options["start_mode"], lấy theo options
        # - Khi demo / biểu diễn 3D (use_render=True), luôn xuất phát đứng "standing" từ 0 km/h cùng vị trí với Ghost Car
        # - Khi huấn luyện (use_render=False), random 50/50 giữa "standing" (0 km/h) và "flying" (108 km/h)
        start_mode = "standing"
        if options and "start_mode" in options:
            start_mode = str(options["start_mode"]).lower()
        elif not self.use_render:
            start_mode = "flying" if (getattr(self, "np_random", np.random).random() < 0.5) else "standing"

        self.is_flying_start = bool(start_mode == "flying")

        # Đặt xe vào vị trí xuất phát Grid Slot P1 (x = 60.0m, y = tim đường trên Hamilton Straight)
        vehicle.set_position([60.0, self.center_lat])
        if self.is_flying_start:
            vehicle.set_velocity([30.0, 0.0])  # 30 m/s = 108 km/h đà bay
        else:
            vehicle.set_velocity([0.0, 0.0])   # Xuất phát đứng từ 0 km/h

        if hasattr(vehicle, "navigation") and vehicle.navigation is not None:
            vehicle.navigation.reset(vehicle)
        self.last_pos = np.array(vehicle.position[:2], dtype=np.float32)
            
        # Lấy Frenet s ban đầu (tại x = 60.0m, s sẽ là 0.0m)
        _, current_s = self._get_track_s(self.car.position)
        self.last_s = current_s
        self.telemetry.cumulative_distance = current_s
        
        obs = self.get_compact_racing_obs(
            vehicle,
            action_applied=self.last_action,
            delta_steer=0.0,
            on_runoff=False,
        )
        
        info["lap_time"] = 0.0
        info["current_lap"] = 1
        info["max_laps"] = self.max_laps
        info["best_lap_time"] = self.best_lap_time
        info["delta_time"] = 0.0
        info["delta_color"] = "gray"
        info["track_s"] = current_s
        info["mean_speed_km_h"] = 0.0
        info["start_mode"] = "flying" if self.is_flying_start else "standing"
        info["is_flying_start"] = bool(self.is_flying_start)
        return obs, info

    def check_track_limits_and_apply_runoff(self, state=None, vehicle_contact_is_out: bool = True):
        """
        Quản lý lề Runoff & Giới hạn đường đua (Track Limits) chuẩn thể thao FIA 4 bánh:
        - Tim đường thực tế nằm ở giữa mặt đường (mặc định 2 làn rộng 9.0m, khoảng cách tới mép nhựa +-half_width = 4.5m).
        - Bỏ hoàn toàn gờ Kerb ảo: lấy chuẩn đúng mép đường nhựa thật để khớp 100% với hiển thị 3D.
        - Luật 4 bánh FIA:
          + Một bánh xe (bề rộng 0.30m, nửa bán kính 0.15m) được coi là còn trên đường nhựa nếu mép trong lốp xe dist_inner <= half_width.
          + Vòng chạy HOÀN TOÀN HỢP LỆ nếu còn ít nhất 1 bánh xe trên mặt nhựa (wheels_on_track >= 1).
          + Bật cờ lấn lề (on_runoff = True) khi có từ 1 đến 3 bánh rơi khỏi mặt nhựa (> half_width).
          + Xử thua (is_full_out = True -> terminated) khi CẢ 4 BÁNH đều hoàn toàn rơi khỏi mặt nhựa (wheels_on_track == 0).
          + Hoặc xe bị lật / va chạm vật thể (is_airborne_or_flipped, crash_object) và va chạm xe khác nếu vehicle_contact_is_out.
        state: trạng thái riêng từng xe (car, current_track_idx); mặc định là chính env (đơn tác tử).
        vehicle_contact_is_out: đơn tác tử coi va chạm xe là bị loại; đua nhiều xe chỉ phạt va chạm ở hàm thưởng.
        """
        if state is None:
            state = self
        wheels = state.car.get_wheel_positions()
        if not wheels:
            return False, False

        tire_half_width = 0.15
        asphalt_limit = self.half_width

        wheels_on_track = 0
        for w_pos in wheels:
            idx, _ = self._get_track_s(w_pos, state.current_track_idx)
            dist_to_center = float(np.linalg.norm(w_pos - self.track_xy[idx]))
            # Mép trong lốp xe gần tim đường hơn tâm bánh xe 0.15m
            dist_inner_edge = max(0.0, dist_to_center - tire_half_width)
            if dist_inner_edge <= asphalt_limit:
                wheels_on_track += 1
                
        is_full_out = False
        on_runoff = False
        is_airborne_or_flipped = state.car.is_airborne_or_flipped()

        if (
            wheels_on_track == 0
            or is_airborne_or_flipped
            or state.car.crash_object
            or (vehicle_contact_is_out and state.car.crash_vehicle)
            or state.car.crash_sidewalk
        ):
            is_full_out = True
        elif wheels_on_track < 4:
            on_runoff = True
            
        return on_runoff, is_full_out

    def _check_lap_completion(self, current_s: float, prev_s: float) -> bool:
        """
        Kiểm tra điều kiện cán đích an toàn theo vạch đích khép kín (Frenet Wrap-around)
        Vạch đích Start/Finish nằm tại x = 60.0m trên Hamilton Straight (-2.5 <= y <= 7.0).
        Xe hoàn thành vòng khi:
        1. Đã hoàn thành phần lớn vòng đua (> 85% chiều dài mạch s theo prev_s).
        2. Vượt qua vạch đích x >= 60.0m trên Hamilton Straight HOẶC Frenet s wrap-around qua vạch đích.
        """
        has_completed_majority = bool(prev_s > 0.85 * self.circuit_length)
        if not has_completed_majority:
            return False
            
        curr_x = float(self.car.position[0])
        last_x = float(self.last_pos[0])
        curr_y = float(self.car.position[1])
        
        # Kiểm tra xe đang trên đoạn thẳng Hamilton Straight (mặc định 2 làn: -2.5 <= y <= 7.0)
        y_min = -0.5 * self.lane_width - 0.25
        y_max = 2.0 * self.center_lat + 0.5 * self.lane_width + 0.25
        on_finish_straight = bool(y_min <= curr_y <= y_max)
        
        # Cán qua mặt phẳng x = 60.0m từ phía sau tới (last_x < 60.0 <= curr_x)
        crossed_finish_line = bool(on_finish_straight and last_x < 60.0 and curr_x >= 60.0)
        
        # Frenet s wrap-around qua vạch đích từ cuối mạch về đầu mạch
        frenet_wrap = bool(on_finish_straight and prev_s > 0.90 * self.circuit_length and current_s < 0.10 * self.circuit_length)
        
        # Hoặc đạt ngưỡng 99.95% Frenet s (fail-safe)
        frenet_100pct = bool(current_s >= 0.9995 * self.circuit_length)
        
        return bool(crossed_finish_line or frenet_wrap or frenet_100pct)

    def step(self, action):
        self.current_step += 1
        
        # 1. Cơ cấu chấp hành xe đua (Xử lý pedal, slew rate limit chống rung lắc)
        action_applied, delta_steer, net_pedal = self.car.process_action(action)
        throttle = max(0.0, net_pedal)
        brake = max(0.0, -net_pedal)
        
        _, _, terminated, truncated, info = self.env.step(action_applied)
        
        vehicle = self.env.agent
        assert vehicle is not None, "Vehicle agent was lost during step"
        
        # 2. Kiểm tra Runoff & Track Limits theo chuẩn FIA 4 bánh
        on_runoff, is_full_out = self.check_track_limits_and_apply_runoff()
        
        # 3. Tính toán tọa độ dọc tim làn đường Frenet s
        idx, current_s = self._get_track_s(self.car.position, self.current_track_idx)
        self.current_track_idx = idx
        
        # 4. Ghi nhận Telemetry chi tiết với Frenet s
        current_time = float(self.current_step * self.dt)
        sample = self.telemetry.log_step(
            current_time=current_time,
            position=self.car.position,
            velocity=self.car.velocity,
            steering=float(action_applied[0]),
            throttle_brake=float(net_pedal),
            dt=self.dt,
            on_runoff=on_runoff,
        )
        self.telemetry.cumulative_distance = float(current_s)
        self.last_acc_long_g = float(sample.get("acc_long_g", 0.0))
        self.last_acc_lat_g = float(sample.get("acc_lat_g", 0.0))
        
        self.episode_history.append({
            "pos": [float(self.car.position[0]), float(self.car.position[1])],
            "heading": float(self.car.heading_theta),
            "vel": [float(self.car.velocity[0]), float(self.car.velocity[1])],
            "track_s": float(current_s),
            "step": int(self.current_step),
        })
        
        speed_km_h = float(sample["speed_km_h"])
        
        is_stalled = False
        if speed_km_h < 10.0:
            self.stagnant_steps += 1
            if (current_s < 20.0 and self.stagnant_steps >= 200) or (current_s >= 20.0 and self.stagnant_steps >= 120):
                is_stalled = True
        else:
            self.stagnant_steps = 0
            
        # Vector quan sát 40 chiều
        obs = self.get_compact_racing_obs(
            vehicle,
            action_applied=action_applied,
            delta_steer=float(delta_steer),
            on_runoff=on_runoff,
        )
        
        # --- HÀM THƯỞNG CHUẨN HÓA THỂ THAO GT3 (MARKOVIAN REWARD) ---
        # a. Thưởng tiến độ dọc tim đường Frenet s
        prev_s = float(self.last_s)
        delta_s = current_s - prev_s
        if delta_s < -0.5 * self.circuit_length:
            # Wrap-around qua vạch đích từ cuối vòng về đầu vòng
            delta_s += self.circuit_length
        self.last_s = current_s
        r_progress = 0.05 * max(0.0, float(delta_s))

        # b. Phạt rung lắc đánh lái (Steering Smoothness Regularization từ feature/clean-111s-optimal-ghost)
        r_steer_smooth = -0.10 * float(delta_steer ** 2)

        # c. Bỏ phạt lệch tim đường để xe tự do mở góc cua Out-In-Out và chém Apex
        r_center = 0.0

        # d. Phạt đứng yên / quay mòng mòng tại chỗ (< 10 km/h)
        r_idle = -0.15 if speed_km_h < 10.0 else 0.0

        # e. Áp lực thời gian (Time Pressure): phạt hằng số mỗi bước, không phụ thuộc tốc độ
        #    nên không biến thành thưởng đạp tốc mù quáng; xe chậm chịu tổng phạt lớn hơn
        r_time = -self.time_penalty

        # Hàm thưởng vi mô thuần túy Markov:
        micro_reward = r_progress + r_steer_smooth + r_idle + r_time
        macro_reward = 0.0
        
        self.last_action = np.array(action, dtype=np.float32)
        
        # Kiểm tra hoàn thành vòng đua (cán đích) qua hàm kiểm tra vạch đích an toàn
        arrive_dest = self._check_lap_completion(current_s=current_s, prev_s=prev_s)
        self.last_pos = np.array(self.car.position[:2], dtype=np.float32)
        
        # Tiến trình hoàn thành vòng đua (cán đích đạt 100%)
        progress_ratio = 1.0 if arrive_dest else float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        
        # --- TẦNG VĨ MÔ (Macro Reward) ---
        if is_full_out or self.car.crash_vehicle or self.car.crash_object:
            terminated = True
            info["termination_reason"] = "crash_or_track_limit"
            info["crash_distance"] = float(current_s)
            info["crash_speed"] = float(speed_km_h)
            self.last_crash_history = list(self.episode_history)
            
            # Phạt va chạm theo vận tốc: tốc độ càng cao phạt càng nặng (-10đ đến -30đ)
            speed_ratio = float(np.clip(speed_km_h / 120.0, 0.0, 1.0))
            macro_reward = -10.0 - 20.0 * (speed_ratio ** 2)
            
        elif is_stalled:
            terminated = True
            info["termination_reason"] = "stalled"
            info["crash_distance"] = float(current_s)
            info["crash_speed"] = float(speed_km_h)
            macro_reward = 0.0
            
        elif arrive_dest:
            self.total_finishes += 1
            self.last_crash_history = []
            lap_time = float((self.current_step - self.lap_start_step) * self.dt)
            
            # Thưởng cán đích cơ bản (50đ) + Thưởng mốc thời gian lũy thừa bậc 3 (Cubic Time Bonus)
            r_finish = 50.0
            eta = float(np.clip((180.0 - lap_time) / 90.0, 0.0, 1.0))
            time_bonus = 100.0 * float(eta ** 3)
            macro_reward = r_finish + time_bonus
            
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
                
            info["lap_completed"] = True
            info["completed_lap_number"] = int(self.current_lap)
            info["completed_lap_time"] = float(lap_time)
            info["lap_time"] = float(lap_time)
            
            if self.current_lap >= self.max_laps:
                self.lap_completed = True
                terminated = True
                info["termination_reason"] = "lap_completed"
                info["arrive_dest"] = True
            else:
                # Chuyển sang Lap tiếp theo trong chế độ đua liên tục nhiều lap
                self.telemetry.last_completed_lap_data = list(self.telemetry.current_lap_data)
                self.telemetry.reset()
                self.current_lap += 1
                self.lap_start_step = self.current_step
                self.current_track_idx = 0
                self.last_s = current_s - self.circuit_length
                if hasattr(vehicle, "navigation") and vehicle.navigation is not None:
                    vehicle.navigation.reset(vehicle)
                
        total_reward = micro_reward + macro_reward
        
        info["step_reward"] = float(total_reward)
        info["micro_reward"] = float(micro_reward)
        info["macro_reward"] = float(macro_reward)
        info["r_progress"] = float(r_progress)
        info["r_steer_smooth"] = float(r_steer_smooth)
        info["r_center"] = 0.0
        info["r_idle"] = float(r_idle)
        info["r_time"] = float(r_time)
        info["throttle"] = float(throttle)
        info["brake"] = float(brake)
        info["net_pedal"] = float(net_pedal)
        if not arrive_dest:
            info["lap_time"] = float((self.current_step - self.lap_start_step) * self.dt)
        info["race_time"] = float(current_time)
        info["current_lap"] = int(self.current_lap)
        info["max_laps"] = int(self.max_laps)
        info["best_lap_time"] = float(self.best_lap_time)
        info["speed_km_h"] = float(speed_km_h)
        mean_lap_speed = float(np.mean([s["speed_km_h"] for s in self.telemetry.current_lap_data])) if self.telemetry.current_lap_data else speed_km_h
        info["mean_speed_km_h"] = float(mean_lap_speed)
        info["track_s"] = float(current_s)
        info["route_completion"] = float(progress_ratio)
        info["on_runoff"] = bool(on_runoff)
        
        return obs, float(total_reward), terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    @property
    def vehicle(self):
        return self.env.agent

    def get_benchmark_engine(self) -> RacingBenchmarkEngine:
        """Khởi tạo hoặc trả về động cơ phân tích học thuật (RacingBenchmarkEngine) đã cache"""
        if self._benchmark_engine is None:
            if not self.circuit_initialized or self.num_track_points <= 1:
                self._init_circuit_polyline()
            self._benchmark_engine = RacingBenchmarkEngine(
                track_xy=self.track_xy,
                track_theta=self.track_theta,
                track_kappa=self.track_kappa,
                track_s=self.track_s,
                vehicle_cfg=self.car.cfg,
                track_width=9.0,
            )
        return self._benchmark_engine

    def evaluate_current_lap_benchmark(self) -> Dict[str, float]:
        """Đánh giá vòng chạy hiện tại với RacingBenchmarkEngine qua Telemetry OOP"""
        engine = self.get_benchmark_engine()
        return self.telemetry.evaluate_with_benchmark(engine)
