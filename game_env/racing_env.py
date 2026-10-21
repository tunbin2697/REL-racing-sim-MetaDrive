"""
MetaRacingEnv: Môi trường Đua xe 3D chuẩn RL tối ưu hóa cao cho MetaDrive Simulator
- Không gian quan sát 35 chiều (Compact Racing State):
  + 10 chiều Ego Kinematics: Vận tốc dọc/ngang, Yaw rate, Góc lái, Throttle/Brake trước, Cos/Sin góc lệch hướng, Độ lệch tim đường.
  + 5 chiều Giới hạn & Tiến trình: Khoảng cách lề trái/phải, Cờ chém lề Kerb, Tỷ lệ tiến trình Frenet s, Độ cong hiện tại.
  + 20 chiều Tầm nhìn góc cua (Lookahead Horizon): 5 mốc [10m, 20m, 35m, 50m, 70m] với tọa độ tương đối dx, dy, sin góc lệch hướng, và độ cong cua kappa.
- Track Limits chuẩn thể thao FIA 4 bánh:
  + Cho phép chém gờ kerb để mở góc cua Apex tối ưu.
  + Chỉ dừng cuộc đua khi cả 4 góc thân xe hoàn toàn văng khỏi lề Runoff (> 6.8m từ tim đường).
- Đo đạc quãng đường chuẩn Frenet s: Miễn nhiễm với hiện tượng trượt drift hoặc quay vòng.
- Sector-based Delta Time: Chia 10 Sector chốt chặn; chỉ thưởng khi vượt qua ranh giới nhanh hơn kỷ lục phân đoạn cũ.
- Áp lực thời gian r_step_penalty = -0.01/step và Thưởng cán đích lớn +100.0đ.
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
from typing import Optional


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

    def reset(self, *args, **kwargs):
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
        use_render: bool = False,
        window_size=(1280, 720),
        horizon: int = 6000,
        manual_control: bool = False,
        start_seed: int = 42,
        num_scenarios: int = 1,
        reference_lap_file: str = "logs/reference_lap.json",
        milestone_respawn: bool = False,
        map_type: str = "silverstone",
        max_speed_km_h: float = 120.0,
        vehicle: Optional[BaseRaceVehicle] = None,
        **kwargs,
    ):
        super().__init__()
        
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
                "lane_num": 2,
                "lane_width": 4.5,
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
        
        # 35 chiều Compact Racing State
        self.obs_dim = 35
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # 2 chiều Continuous Racing Action: [steer, pedal] (pedal > 0 là ga, pedal < 0 là phanh)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        
        # Dữ liệu hình học đường đua dạng Polyline
        self.circuit_initialized = False
        self.track_xy: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.track_theta: np.ndarray = np.zeros(1, dtype=np.float32)
        self.track_kappa: np.ndarray = np.zeros(1, dtype=np.float32)
        self.track_s: np.ndarray = np.zeros(1, dtype=np.float32)
        self.circuit_length = 2838.7
        self.num_track_points = 1
        self.current_track_idx = 0
        self.last_longitudinal = 0.0
        
        # Biến trạng thái
        self.current_step = 0
        self.last_s = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.lap_completed = False
        self.episode_history = []
        self.last_crash_history = []
        self.stagnant_steps = 0

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
                r_mid = lane.radius + 2.25
                kappa = (1.0 / float(r_mid)) * (-1.0 if is_cw else 1.0)
                
            for s_l in s_samples:
                # Tim đường thực tế của mặt đường 2 làn (làn 0 từ -2.25 đến +2.25, làn 1 từ +2.25 đến +6.75)
                # Điểm tim chuẩn chính xác giữa 2 làn nằm tại lateral = 2.25
                pos = lane.position(s_l, 2.25)
                theta = float(lane.heading_theta_at(s_l))
                track_points.append({
                    "pos": np.array(pos[:2], dtype=np.float32),
                    "theta": theta,
                    "kappa": float(kappa),
                })
            
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
            idx = int(np.argmin(dists))
        elif hint_idx < 40:
            # Gần vạch xuất phát: KHÔNG cho phép tìm lùi vòng ra đuôi đường đua (chống bug về đích ảo)
            indices = np.arange(0, min(self.num_track_points, hint_idx + 60))
            dists = np.sum((self.track_xy[indices] - pos_2d) ** 2, axis=1)
            idx = int(indices[np.argmin(dists)])
        else:
            window = 60
            indices = np.arange(hint_idx - 15, hint_idx + window) % self.num_track_points
            dists = np.sum((self.track_xy[indices] - pos_2d) ** 2, axis=1)
            idx = int(indices[np.argmin(dists)])
            
        return idx, float(self.track_s[idx])

    def get_compact_racing_obs(self, vehicle, action_applied=None, on_runoff=False):
        """
        Xây dựng vector quan sát 35 chiều (Compact Racing State):
        - 10 chiều Ego Kinematics & Controls
        - 5 chiều Track Limits & Progress
        - 20 chiều Lookahead Horizon (5 waypoints x 4 values)
        """
        if action_applied is None:
            action_applied = self.last_action
            
        pos = np.array(vehicle.position[:2], dtype=np.float32)
        heading = float(vehicle.heading_theta)
        vel = np.array(vehicle.velocity[:2], dtype=np.float32)
        speed_m_s = float(np.linalg.norm(vel))
        
        # Tìm vị trí tim đường hiện tại
        idx, current_s = self._get_track_s(pos, self.current_track_idx)
        self.current_track_idx = idx
        track_pt = self.track_xy[idx]
        track_heading = float(self.track_theta[idx])
        track_curvature = float(self.track_kappa[idx])
        
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
        
        # Khoảng cách tới mép đường trái và phải (đường rộng 9m -> mép tại +-4.5m)
        dist_left = float(np.clip((4.5 - lat_offset) / 4.5, -2.0, 2.0))
        dist_right = float(np.clip((4.5 + lat_offset) / 4.5, -2.0, 2.0))
        
        # Yaw rate
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0))
        
        # Xử lý chân ga/phanh bước trước hỗ trợ cả 2D và 3D
        if len(self.last_action) >= 3:
            t = 0.5 * (float(self.last_action[1]) + 1.0)
            b = 0.5 * (float(self.last_action[2]) + 1.0)
            last_pedal = t * (1.0 - b) - b if b > 0.05 else t
        else:
            last_pedal = float(self.last_action[1])
            
        cur_steer = float(action_applied[0])
        
        # --- 10 CHIỀU EGO KINEMATICS ---
        ego_state = [
            float(np.clip(speed_m_s / 40.0, 0.0, 1.5)),          # Vận tốc chuẩn hóa
            float(np.clip(vx / 40.0, -1.0, 1.5)),                # Vận tốc dọc
            float(np.clip(vy / 20.0, -1.5, 1.5)),                # Vận tốc trượt ngang
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),       # Yaw rate
            float(cur_steer),                                    # Góc lái hiện tại
            float(self.last_action[0]),                          # Góc lái bước trước
            float(np.clip(last_pedal, -1.0, 1.0)),               # Ga/phanh bước trước
            float(cos_dh),                                       # Cos góc lệch hướng
            float(sin_dh),                                       # Sin góc lệch hướng
            float(np.clip(lat_offset / 4.5, -2.0, 2.0)),         # Độ lệch ngang tim đường
        ]
        
        # --- 5 CHIỀU TRACK LIMITS & TIẾN TRÌNH ---
        progress_ratio = float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        track_info = [
            dist_left,
            dist_right,
            1.0 if on_runoff else 0.0,
            progress_ratio,
            float(np.clip(track_curvature * 50.0, -2.0, 2.0)),  # Độ cong chuẩn hóa
        ]
        
        # --- 20 CHIỀU LOOKAHEAD HORIZON (5 WAYPOINTS X 4) ---
        lookahead_distances = [10.0, 20.0, 35.0, 50.0, 70.0]
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
                float(np.clip(dx_rel / 70.0, -1.0, 1.5)),
                float(np.clip(dy_rel / 30.0, -2.0, 2.0)),
                float(np.sin(d_theta)),
                float(np.clip(wpt_kappa * 50.0, -2.0, 2.0)),
            ])
            
        obs = np.array(ego_state + track_info + lookahead_features, dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _, info = self.env.reset(seed=seed)
        else:
            _, info = self.env.reset()
            
        if not self.circuit_initialized:
            self._init_circuit_polyline()
            
        self.current_step = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.lap_completed = False
        self.telemetry.reset()
        self.episode_history = []
        self.stagnant_steps = 0
        self.current_track_idx = 0
        
        vehicle = self.env.agent
        assert vehicle is not None, "Vehicle agent was not properly spawned in MetaDrive"
        self.car.attach_agent(vehicle)
            
        # Lấy Frenet s ban đầu
        _, current_s = self._get_track_s(self.car.position)
        self.last_s = current_s
        self.telemetry.cumulative_distance = current_s
        
        obs = self.get_compact_racing_obs(vehicle, action_applied=self.last_action, on_runoff=False)
        
        info["lap_time"] = self.current_step * self.dt
        info["best_lap_time"] = self.best_lap_time
        info["delta_time"] = 0.0
        info["delta_color"] = "gray"
        info["track_s"] = current_s
        info["mean_speed_km_h"] = 0.0
        return obs, info

    def check_track_limits_and_apply_runoff(self):
        """
        Quản lý lề Runoff & Giới hạn đường đua (Track Limits) chuẩn thể thao FIA 4 bánh:
        - Tim đường thực tế nằm ở giữa 2 làn (đường nhựa rộng 9.0m, khoảng cách tới mép nhựa +-4.5m).
        - Gờ Kerb / Runoff: mở rộng thêm 1.0m (bán kính tới 5.5m từ tim đường).
        - Xe được phép chém lề Kerb để mở cua Apex tối ưu.
        - Xử thua (terminated = True) khi:
          + Cả 4 góc thân xe hoàn toàn rơi khỏi lề Runoff (> 4.5m từ tim đường).
          + Hoặc xe bị lật / văng lơ lửng ngoài không gian (roll/pitch > 35 độ hoặc rơi khỏi mặt đường z < -0.1).
        """
        corners = self.car.get_bounding_corners()
        corner_offsets = []
        for c in corners:
            idx, _ = self._get_track_s(c, self.current_track_idx)
            dist_to_center = float(np.linalg.norm(c - self.track_xy[idx]))
            corner_offsets.append(dist_to_center)
            
        max_offset = max(corner_offsets) if corner_offsets else 0.0
        min_offset = min(corner_offsets) if corner_offsets else 0.0
        
        is_full_out = False
        on_runoff = False
        is_airborne_or_flipped = self.car.is_airborne_or_flipped()
        limit_threshold = 4.5
        
        if (
            min_offset > limit_threshold
            or is_airborne_or_flipped
            or self.car.crash_object
            or self.car.crash_vehicle
            or self.car.crash_sidewalk
        ):
            is_full_out = True
        elif max_offset > 4.5:
            on_runoff = True
            
        return on_runoff, is_full_out

    def step(self, action):
        self.current_step += 1
        
        # 1. Cơ cấu chấp hành xe đua (Xử lý pedal, slew rate limit chống rung lắc)
        action_applied, delta_steer, net_pedal = self.car.process_action(action)
        throttle = max(0.0, net_pedal)
        brake = max(0.0, -net_pedal)
        
        _, _, terminated, truncated, info = self.env.step(action_applied)
        
        vehicle = self.env.agent
        assert vehicle is not None, "Vehicle agent was lost during step"
        current_time = self.current_step * self.dt
        
        # 2. Kiểm tra Runoff & Track Limits theo chuẩn FIA 4 bánh
        on_runoff, is_full_out = self.check_track_limits_and_apply_runoff()
        
        # 3. Tính toán tọa độ dọc tim làn đường Frenet s
        idx, current_s = self._get_track_s(self.car.position, self.current_track_idx)
        self.current_track_idx = idx
        
        # 4. Ghi nhận Telemetry chi tiết với Frenet s
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
        
        self.episode_history.append({
            "pos": [float(self.car.position[0]), float(self.car.position[1])],
            "heading": float(self.car.heading_theta),
            "vel": [float(self.car.velocity[0]), float(self.car.velocity[1])],
            "track_s": float(current_s),
            "step": int(self.current_step),
        })
        
        speed_km_h = float(sample["speed_km_h"])
        
        is_stalled = False
        if speed_km_h < 2.0:
            self.stagnant_steps += 1
            if (current_s < 20.0 and self.stagnant_steps >= 400) or (current_s >= 20.0 and self.stagnant_steps >= 150):
                is_stalled = True
        else:
            self.stagnant_steps = 0
            
        # Vector quan sát 35 chiều
        obs = self.get_compact_racing_obs(vehicle, action_applied=action_applied, on_runoff=on_runoff)
        
        # --- HÀM THƯỞNG CHUẨN HÓA THỂ THAO GT3 (MARKOVIAN REWARD) ---
        # a. Thưởng tiến độ dọc tim đường Frenet s
        delta_s = current_s - self.last_s
        self.last_s = current_s
        r_progress = 0.05 * max(0.0, float(delta_s))

        # b. Phạt rung lắc đánh lái để vô-lăng mượt mà, ổn định trên đường thẳng
        r_steer_smooth = -0.10 * float(delta_steer ** 2)

        # c. Hướng dẫn phanh trước cua gắt & đạp ga trên đường thẳng (Chuẩn Markov MDP thuần túy)
        # Giữ cố định xuyên suốt để không làm biến đổi phân phối Value Function của Critic
        r_guide = 0.0
        upcoming_kappa = max(abs(obs[18]), abs(obs[22]), abs(obs[26]), abs(obs[30]))
        if upcoming_kappa > 0.28 and speed_km_h > 50.0 and net_pedal < -0.05:
            r_guide = 0.08  # Thưởng phanh giảm tốc an toàn chuẩn bị ôm cua gắt
        elif upcoming_kappa < 0.10 and net_pedal > 0.25:
            r_guide = 0.03  # Thưởng đạp ga tăng tốc trên đường thẳng

        # d. Phạt lệch tim đường (Racing Line Centering): khuyến khích đi giữa lòng đường, tránh bám mép cỏ
        lat_offset_val = float(obs[9]) * 4.5  # Khôi phục từ ego_state lat_offset / 4.5
        r_center = -0.04 * float((lat_offset_val / 4.5) ** 2)

        # e. Phạt đứng yên / quay mòng mòng tại chỗ (< 10 km/h)
        r_idle = -0.15 if speed_km_h < 10.0 else 0.0
        
        micro_reward = r_progress + r_steer_smooth + r_guide + r_center + r_idle
        macro_reward = 0.0
        
        self.last_action = np.array([action_applied[0], net_pedal], dtype=np.float32)
        
        # Kiểm tra hoàn thành vòng đua (cán đích)
        progress_ratio = float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        has_completed_majority = bool(current_s > 0.85 * self.circuit_length)
        arrive_dest = bool((info.get("arrive_dest", False) or progress_ratio >= 0.985) and has_completed_majority)
        
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
            self.lap_completed = True
            terminated = True
            self.total_finishes += 1
            info["termination_reason"] = "lap_completed"
            info["arrive_dest"] = True
            info["lap_completed"] = True
            self.last_crash_history = []
            
            # Thưởng cán đích cơ bản + Thưởng mốc thời gian bậc thang (mỗi mốc 20s)
            r_finish = 100.0
            if current_time < 140.0:
                time_bonus = 50.0
            elif current_time < 160.0:
                time_bonus = 40.0
            elif current_time < 180.0:
                time_bonus = 30.0
            elif current_time < 200.0:
                time_bonus = 20.0
            elif current_time < 220.0:
                time_bonus = 10.0
            else:
                time_bonus = 0.0
            macro_reward = r_finish + time_bonus
            
            if current_time < self.best_lap_time:
                self.best_lap_time = current_time
                
        total_reward = micro_reward + macro_reward
        
        info["step_reward"] = float(total_reward)
        info["micro_reward"] = float(micro_reward)
        info["macro_reward"] = float(macro_reward)
        info["r_progress"] = float(r_progress)
        info["r_steer_smooth"] = float(r_steer_smooth)
        info["r_guide"] = float(r_guide)
        info["r_idle"] = float(r_idle)
        info["throttle"] = float(throttle)
        info["brake"] = float(brake)
        info["net_pedal"] = float(net_pedal)
        info["lap_time"] = float(current_time)
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
