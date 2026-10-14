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
from metadrive.envs.marl_envs.marl_racing_env import RacingMapManager
from game_env.telemetry import TelemetryTracker


class SingleRacingMetaDrive(MetaDriveEnv):
    """Môi trường MetaDrive đơn tác tử nạp trực tiếp RacingMapManager, tắt hoàn toàn LiDAR raycast dư thừa"""
    @classmethod
    def default_config(cls):
        config = super(SingleRacingMetaDrive, cls).default_config()
        config.update(dict(
            show_sidewalk=False,
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
        horizon: int = 4500,
        manual_control: bool = False,
        start_seed: int = 42,
        num_scenarios: int = 1,
        reference_lap_file: str = "logs/reference_lap.json",
        milestone_respawn: bool = False,
        overspeed_a_lat: float = 0.65 * 9.81,
        **kwargs,
    ):
        super().__init__()
        
        self.use_render = use_render
        self.window_size = window_size
        self.horizon = horizon
        self.manual_control = manual_control
        self.start_seed = start_seed
        self.num_scenarios = num_scenarios
        # MetaDrive gộp decision_repeat bước vật lý vào mỗi lần step() -> thời gian thực mỗi step = 0.02 * 5 = 0.1s
        self.physics_dt = 0.02
        self.decision_repeat = 5
        self.dt = self.physics_dt * self.decision_repeat
        self.milestone_respawn = milestone_respawn
        # Gia tốc ngang dùng để tính tốc độ cho phép trong phạt vượt tốc (0.65g mặc định; ~0.9g là giới hạn văng thực tế)
        self.overspeed_a_lat = overspeed_a_lat
        
        # Telemetry tracker & Kỷ lục
        self.telemetry = TelemetryTracker()
        self.agent_best_lap_time = float("inf")
        self.best_lap_time = float("inf")
        self.agent_best_completion = 0.0
        
        # Kỷ lục 10 Sectors (mỗi Sector ~218.6m)
        self.num_sectors = 10
        self.sector_best_times = {}
        self.current_sector = 0
        self.sector_start_time = 0.0
        
        # Cấu hình MetaDrive engine (Tắt 240 tia LiDAR chết để tăng tốc độ mô phỏng)
        self.config = {
            "use_render": use_render,
            "manual_control": manual_control,
            "traffic_density": 0.0,
            "start_seed": start_seed,
            "num_scenarios": num_scenarios,
            "horizon": horizon,
            "window_size": window_size,
            "physics_world_step_size": self.physics_dt,
            "decision_repeat": self.decision_repeat,
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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        
        # Dữ liệu hình học đường đua dạng Polyline
        self.circuit_initialized = False
        self.track_xy: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.track_theta: np.ndarray = np.zeros(1, dtype=np.float32)
        self.track_kappa: np.ndarray = np.zeros(1, dtype=np.float32)
        self.track_s: np.ndarray = np.zeros(1, dtype=np.float32)
        self.circuit_length = 2186.3
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
        self.cleared_milestones = set()
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

    def get_allowed_speed(self, current_s, preview=70.0, a_lat=0.65 * 9.81, a_brake=5.0):
        """
        Tốc độ tối đa cho phép tại vị trí hiện tại (km/h) theo đường bao phanh:
        v_allow = min_d sqrt(v_corner(d)^2 + 2 * a_brake * d), với v_corner = sqrt(a_lat * R).
        """
        ahead = np.arange(0.0, preview, 2.0)
        target_s = (current_s + ahead) % self.circuit_length
        idx = np.searchsorted(self.track_s, target_s) % self.num_track_points
        kappa = np.abs(self.track_kappa[idx])
        v_corner_sq = np.where(kappa > 1e-6, a_lat / np.maximum(kappa, 1e-6), np.inf)
        v_allow = float(np.sqrt(np.min(v_corner_sq + 2.0 * a_brake * ahead)))
        return min(v_allow * 3.6, 200.0)

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
        
        # --- 10 CHIỀU EGO KINEMATICS ---
        ego_state = [
            float(np.clip(speed_m_s / 40.0, 0.0, 1.5)),          # Vận tốc chuẩn hóa
            float(np.clip(vx / 40.0, -1.0, 1.5)),                # Vận tốc dọc
            float(np.clip(vy / 20.0, -1.5, 1.5)),                # Vận tốc trượt ngang
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),       # Yaw rate
            float(action_applied[0]),                            # Góc lái hiện tại
            float(self.last_action[0]),                          # Góc lái bước trước
            float(self.last_action[1]),                          # Ga/phanh bước trước
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
        self.cleared_milestones = set()
        self.stagnant_steps = 0
        self.current_track_idx = 0
        self.current_sector = 0
        self.sector_start_time = 0.0
        
        vehicle = self.env.agent
        
        # Cơ chế Hồi Sinh Reverse Curriculum (Milestone Respawn)
        if self.milestone_respawn and not (options and options.get("force_start", False)):
            roll = np.random.rand()
            if roll < 0.50 and len(self.last_crash_history) >= 20:
                max_back = min(len(self.last_crash_history) - 2, 120)
                min_back = min(20, max_back)
                back_steps = int(np.random.randint(min_back, max_back + 1)) if min_back < max_back else min_back
                idx_saved = max(0, len(self.last_crash_history) - 1 - back_steps)
                
                saved = self.last_crash_history[idx_saved]
                vehicle.set_position(saved["pos"])
                vehicle.set_heading_theta(saved["heading"])
                vehicle.set_velocity(saved["vel"])
                self.current_step = int(saved["step"])
                
                # Khóa các milestone / sector đã vượt qua trước thời điểm tua ngược
                saved_s = float(saved.get("track_s", 0.0))
                sector_size = self.circuit_length / float(self.num_sectors)
                past_sector_idx = min(self.num_sectors - 1, int(saved_s / sector_size))
                self.current_sector = past_sector_idx
                for s_m in range(past_sector_idx + 1):
                    self.cleared_milestones.add(s_m)
                    
        # Lấy Frenet s ban đầu
        # Tìm toàn cục rồi cập nhật hint, tránh việc dò quanh vạch xuất phát khi xe vừa được hồi sinh giữa đường
        idx, current_s = self._get_track_s(vehicle.position)
        self.current_track_idx = idx
        self.last_s = current_s
        self.telemetry.cumulative_distance = current_s
        
        obs = self.get_compact_racing_obs(vehicle, action_applied=self.last_action, on_runoff=False)
        
        info["lap_time"] = self.current_step * self.dt
        info["best_lap_time"] = self.best_lap_time
        info["delta_time"] = 0.0
        info["delta_color"] = "gray"
        info["track_s"] = current_s
        return obs, info

    def check_track_limits_and_apply_runoff(self, vehicle):
        """
        Quản lý lề Runoff & Giới hạn đường đua (Track Limits) chuẩn thể thao FIA 4 bánh:
        - Tim đường thực tế nằm ở giữa 2 làn (đường nhựa rộng 9.0m, khoảng cách tới mép nhựa +-4.5m).
        - Gờ Kerb / Runoff: mở rộng thêm 1.0m (bán kính tới 5.5m từ tim đường).
        - Xe được phép chém lề Kerb để mở cua Apex tối ưu.
        - Xử thua (terminated = True) khi:
          + Cả 4 góc thân xe hoàn toàn rơi khỏi lề Runoff (> 5.5m từ tim đường).
          + Hoặc xe bị lật / văng lơ lửng ngoài không gian (roll/pitch > 35 độ hoặc rơi khỏi mặt đường z < -0.1).
        """
        length = float(getattr(vehicle, "LENGTH", 4.5))
        width = float(getattr(vehicle, "WIDTH", 1.85))
        pos = np.array(vehicle.position[:2], dtype=np.float32)
        heading = float(vehicle.heading_theta)
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        fwd = np.array([cos_h, sin_h], dtype=np.float32)
        lat = np.array([-sin_h, cos_h], dtype=np.float32)
        
        corners = [
            pos + 0.5 * length * fwd + 0.5 * width * lat,
            pos + 0.5 * length * fwd - 0.5 * width * lat,
            pos - 0.5 * length * fwd + 0.5 * width * lat,
            pos - 0.5 * length * fwd - 0.5 * width * lat,
        ]
        
        corner_offsets = []
        for c in corners:
            idx, _ = self._get_track_s(c, self.current_track_idx)
            dist_to_center = float(np.linalg.norm(c - self.track_xy[idx]))
            corner_offsets.append(dist_to_center)
            
        max_offset = max(corner_offsets)
        min_offset = min(corner_offsets)
        
        is_full_out = False
        on_runoff = False
        
        # Kiểm tra trạng thái lật xe / rơi rớt khỏi không gian 3D
        is_airborne_or_flipped = False
        if hasattr(vehicle, "origin"):
            pos_3d = vehicle.origin.get_pos()
            hpr = vehicle.origin.get_hpr()
            if pos_3d[2] < -0.1 or pos_3d[2] > 2.0 or abs(hpr[1]) > 35.0 or abs(hpr[2]) > 35.0:
                is_airborne_or_flipped = True
        
        if (
            min_offset > 5.5
            or is_airborne_or_flipped
            or vehicle.crash_object
            or vehicle.crash_vehicle
            or vehicle.crash_sidewalk
        ):
            is_full_out = True
        elif max_offset > 4.5:
            on_runoff = True
            
        return on_runoff, is_full_out

    def step(self, action):
        self.current_step += 1
        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        
        steer_applied = 0.7 * float(action[0]) + 0.3 * float(self.last_action[0])
        action_applied = np.array([steer_applied, action[1]], dtype=np.float32)
        
        _, _, terminated, truncated, info = self.env.step(action_applied)
        
        vehicle = self.env.agent
        current_time = self.current_step * self.dt
        
        # 1. Kiểm tra Runoff & Track Limits theo chuẩn FIA 4 bánh
        on_runoff, is_full_out = self.check_track_limits_and_apply_runoff(vehicle)
        
        # 2. Tính toán tọa độ dọc tim làn đường Frenet s
        idx, current_s = self._get_track_s(vehicle.position, self.current_track_idx)
        self.current_track_idx = idx
        
        # 3. Ghi nhận Telemetry chi tiết với Frenet s
        sample = self.telemetry.log_step(
            current_time=current_time,
            position=vehicle.position,
            velocity=vehicle.velocity,
            steering=float(action_applied[0]),
            throttle_brake=float(action_applied[1]),
            dt=self.dt,
            on_runoff=on_runoff,
        )
        self.telemetry.cumulative_distance = float(current_s)
        
        self.episode_history.append({
            "pos": [float(vehicle.position[0]), float(vehicle.position[1])],
            "heading": float(vehicle.heading_theta),
            "vel": [float(vehicle.velocity[0]), float(vehicle.velocity[1])],
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
            
        # Vector quan sát 35 chiều thế hệ mới
        obs = self.get_compact_racing_obs(vehicle, action_applied=action_applied, on_runoff=on_runoff)
        
        # 4. HÀM THƯỞNG ĐỘT PHÁ (BREAKTHROUGH REWARD)
        # a. Áp lực thời gian: Mỗi step xe tồn tại đều bị trừ điểm (-0.5đ/giây)
        r_time = -0.01
        
        # b. Thưởng tiến độ theo trục cung Frenet s (nhân hệ số vận tốc để thúc đẩy chạy nhanh)
        delta_s = current_s - self.last_s
        self.last_s = current_s
        speed_ratio = min(speed_km_h / 80.0, 1.5)
        r_progress = 0.02 * max(0.0, float(delta_s)) * (0.5 + 0.5 * speed_ratio)
        
        # c. Thưởng tốc độ thực tế (Bình phương vận tốc)
        is_forward = delta_s > 0
        if is_forward:
            speed_norm = speed_km_h / 120.0
            r_speed = 0.03 * float(speed_norm ** 2)
        else:
            r_speed = 0.0
            
        # d. Phạt vượt tốc độ an toàn theo vật lý (Physics-based Overspeed Penalty)
        # Tốc độ cho phép = min trên các điểm phía trước của sqrt(v_cua^2 + 2*a_brake*d), với v_cua = sqrt(a_lat*R)
        v_allow_km_h = self.get_allowed_speed(current_s, a_lat=self.overspeed_a_lat)
        excess_km_h = max(0.0, speed_km_h - v_allow_km_h)
        is_braking = action_applied[1] < -0.1
        r_brake = -min(0.5, 0.05 * (excess_km_h / 5.0) ** 2)
        if excess_km_h > 0.0 and is_braking:
            r_brake += 0.02  # Thưởng chủ động phanh khi đang vượt tốc
            
        # e. Sector Gate Rewards (Thưởng mốc từng chặng để bứt phá bẫy cua 1650m)
        sector_size = self.circuit_length / float(self.num_sectors)  # ~218.6m
        current_sector_idx = min(self.num_sectors - 1, int(current_s / sector_size))
        r_sector = 0.0
        if current_sector_idx > self.current_sector and current_sector_idx not in self.cleared_milestones:
            r_sector = 5.0
            self.cleared_milestones.add(current_sector_idx)
            self.current_sector = current_sector_idx

        # f. Phạt đứng yên / quay mòng mòng tại chỗ (< 10 km/h)
        r_idle = -0.15 if speed_km_h < 10.0 else 0.0
        
        micro_reward = r_time + r_progress + r_speed + r_brake + r_sector + r_idle
        macro_reward = 0.0
        
        self.last_action = np.array(action_applied, dtype=np.float32)
        
        # Kiểm tra hoàn thành vòng đua (cán đích)
        # Chỉ cho phép cán đích nếu xe đã thực sự chạy qua phần lớn đường đua (> 85%)
        progress_ratio = float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        has_completed_majority = bool(current_s > 0.85 * self.circuit_length)
        arrive_dest = bool((info.get("arrive_dest", False) or progress_ratio >= 0.985) and has_completed_majority)
        
        # --- TẦNG VĨ MÔ (Macro Reward) ---
        if is_full_out or vehicle.crash_vehicle or vehicle.crash_object:
            terminated = True
            info["termination_reason"] = "crash_or_track_limit"
            info["crash_distance"] = float(current_s)
            info["crash_speed"] = float(speed_km_h)
            self.last_crash_history = list(self.episode_history)
            macro_reward = -30.0
            
        elif is_stalled:
            terminated = True
            info["termination_reason"] = "stalled"
            info["crash_distance"] = float(current_s)
            info["crash_speed"] = float(speed_km_h)
            macro_reward = -30.0  # Bằng phạt crash: chống reward hacking dừng xe trước cua để né phạt đâm tường
            
        elif arrive_dest:
            self.lap_completed = True
            terminated = True
            info["termination_reason"] = "lap_completed"
            info["arrive_dest"] = True
            info["lap_completed"] = True
            self.last_crash_history = []
            
            # CỦ CÀ RỐT CÁN ĐÍCH: Thưởng lớn + Thưởng Lap Time kỷ lục
            r_finish = 100.0
            lap_bonus = max(0.0, (120.0 - current_time) * 1.5)
            macro_reward = r_finish + lap_bonus
            
            if current_time < self.best_lap_time:
                self.best_lap_time = current_time
                
        total_reward = micro_reward + macro_reward
        
        info["step_reward"] = float(total_reward)
        info["micro_reward"] = float(micro_reward)
        info["macro_reward"] = float(macro_reward)
        info["r_time"] = float(r_time)
        info["r_progress"] = float(r_progress)
        info["r_milestone"] = float(r_progress + r_sector)  # Bao gồm cả tiến độ và sector gate
        info["r_sector"] = float(r_sector)
        info["r_speed"] = float(r_speed)
        info["r_brake"] = float(r_brake)
        info["v_allow_km_h"] = float(v_allow_km_h)
        info["r_idle"] = float(r_idle)
        info["r_step_penalty"] = float(r_time)
        info["r_delta"] = 0.0
        info["lap_time"] = float(current_time)
        info["best_lap_time"] = float(self.best_lap_time)
        info["speed_km_h"] = float(speed_km_h)
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
