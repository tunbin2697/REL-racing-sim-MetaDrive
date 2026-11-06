"""
MetaRacingEnv: lớp nền cho môi trường đua trên MetaDrive.
- Dựng engine và sa bàn Silverstone theo TrackPreset.
- Tính tim đường dạng polyline và tọa độ Frenet s.
- Xây dựng quan sát ego 35D của ref_111s: 10 chiều động học, 5 chiều giới hạn đường, 20 chiều lookahead.
reset() và step() do lớp con định nghĩa (xem MultiCarRacingEnv).
"""

from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive import MetaDriveEnv

from game_env.silverstone_map import SilverstoneMapManager
from game_env.track_presets import get_track_preset, TrackPreset
from game_env.vehicle import BaseRaceVehicle, GT3RaceCar


class SingleRacingMetaDrive(MetaDriveEnv):
    """MetaDrive nạp sa bàn Silverstone"""
    @classmethod
    def default_config(cls):
        config = super(SingleRacingMetaDrive, cls).default_config()
        if "map_config" in config and hasattr(config["map_config"], "update"):
            config["map_config"].update({"length_scale": 1.0}, allow_add_new_key=True)
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
        self.engine.update_manager("map_manager", SilverstoneMapManager())

    @staticmethod
    def _is_arrive_destination(vehicle):
        # Lớp con tự quản lý việc hoàn thành vòng đua qua vạch Start/Finish
        return False

    def reset(self, *args, **kwargs):
        # Dọn các xe spawn ngoài agent manager (xe AI của cuộc đua trước) trước khi reset engine
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
    """Gym wrapper chung: engine, hình học đường đua và quan sát ego 35D"""
    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(
        self,
        map_type: str,
        vehicle: Optional[BaseRaceVehicle] = None,
        use_render: bool = False,
        window_size=(1280, 720),
        horizon: int = 6000,
        start_seed: int = 42,
        max_laps: int = 1,
        time_penalty: float = 0.0,
    ):
        super().__init__()
        self.car: BaseRaceVehicle = vehicle if vehicle is not None else GT3RaceCar()
        self.preset: TrackPreset = get_track_preset(map_type)
        self.max_laps = int(max_laps)
        self.horizon = int(horizon)
        self.dt = 0.10
        self.time_penalty = float(time_penalty)

        # Cấu hình MetaDrive engine (Tắt 240 tia LiDAR chết để tăng tốc độ mô phỏng)
        self.config = {
            "use_render": use_render,
            "manual_control": False,
            "traffic_density": 0.0,
            "start_seed": start_seed,
            "num_scenarios": 1,
            "horizon": self.horizon,
            "window_size": window_size,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "cull_lanes_outside_map": False,
            # Vùng vẽ mặt đường lên terrain (m). MetaDrive mặc định 1024 và ghi đè TerrainProperty khi khởi tạo,
            # nên phải đặt qua config.
            "map_region_size": 2048,
            "map_type": "silverstone",
            "map_config": {
                "lane_num": self.preset.lane_num,
                "lane_width": self.preset.lane_width,
                "length_scale": self.preset.length_scale,
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

        # [steer, throttle, brake]; steer: [-1, 1], throttle và brake: [-1, 1] -> [0, 1]
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
        self.current_step = 0
        self.last_action = np.zeros(3, dtype=np.float32)

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
                r_mid = lane.radius + self.preset.center_lat
                kappa = (1.0 / float(r_mid)) * (-1.0 if is_cw else 1.0)

            for s_l in s_samples:
                # Tim đường thực tế của mặt đường tại lateral = center_lat
                pos = lane.position(s_l, self.preset.center_lat)
                theta = float(lane.heading_theta_at(s_l))
                track_points.append({
                    "pos": np.array(pos[:2], dtype=np.float32),
                    "theta": theta,
                    "kappa": float(kappa),
                })

        # Định vị vạch xuất phát / cán đích (Start/Finish Grid) tại grid_x trên Hamilton Straight
        dists_to_grid = np.sum((np.array([p["pos"] for p in track_points]) - np.array([self.preset.grid_x, self.preset.center_lat])) ** 2, axis=1)
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

    def get_compact_racing_obs(self, vehicle, action_applied, on_runoff: bool = False) -> np.ndarray:
        """
        Xây dựng vector quan sát ego 35 chiều của ref_111s:
        - 10 chiều Ego Kinematics & Controls
        - 5 chiều Track Limits & Progress (dist_left, dist_right, runoff, progress, current_kappa)
        - 20 chiều Lookahead Horizon (5 waypoints x 4 values) theo preset.lookahead_m
        """
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

        # Khoảng cách tới mép đường trái và phải theo chuẩn hóa preset
        dist_left = float(np.clip((self.preset.half_width - lat_offset) / self.preset.lat_norm, -2.0, 2.0))
        dist_right = float(np.clip((self.preset.half_width + lat_offset) / self.preset.lat_norm, -2.0, 2.0))

        # Yaw rate
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0))

        # Ga/phanh bước trước
        t = 0.5 * (float(self.last_action[1]) + 1.0)
        b = 0.5 * (float(self.last_action[2]) + 1.0)
        last_pedal = t * (1.0 - b) - b if b > 0.05 else t

        cur_steer = float(action_applied[0])

        # --- 10 CHIỀU EGO KINEMATICS ---
        ego_state = [
            float(np.clip(speed_m_s / self.preset.v_norm, 0.0, 1.5)),            # 1. Vận tốc chuẩn hóa
            float(np.clip(vx / self.preset.v_norm, -1.0, 1.5)),                  # 2. Vận tốc dọc
            float(np.clip(vy / self.preset.vy_norm, -1.5, 1.5)),                 # 3. Vận tốc trượt ngang
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),                       # 4. Yaw rate
            float(cur_steer),                                                    # 5. Góc lái hiện tại
            float(self.last_action[0]),                                          # 6. Góc lái bước trước
            float(np.clip(last_pedal, -1.0, 1.0)),                               # 7. Ga/phanh bước trước
            float(cos_dh),                                                       # 8. Cos góc lệch hướng
            float(sin_dh),                                                       # 9. Sin góc lệch hướng
            float(np.clip(lat_offset / self.preset.lat_norm, -2.0, 2.0)),        # 10. Độ lệch ngang tim đường
        ]

        # --- 5 CHIỀU TRACK LIMITS & TIẾN TRÌNH ---
        progress_ratio = float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        track_info = [
            dist_left,                                                             # 11. Khoảng cách mép trái
            dist_right,                                                            # 12. Khoảng cách mép phải
            1.0 if on_runoff else 0.0,                                             # 13. Cờ lấn lề Runoff
            progress_ratio,                                                        # 14. Tiến trình trên sa bàn
            float(np.clip(track_curvature * self.preset.kappa_norm, -2.0, 2.0)),   # 15. Độ cong tức thời
        ]

        # --- 20 CHIỀU LOOKAHEAD HORIZON (5 WAYPOINTS X 4) ---
        lookahead_features = []
        for dist_ahead in self.preset.lookahead_m:
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
                float(np.clip(dx_rel / self.preset.lookahead_dx_norm, -1.0, 1.5)),
                float(np.clip(dy_rel / self.preset.lookahead_dy_norm, -2.0, 2.0)),
                float(np.sin(d_theta)),
                float(np.clip(wpt_kappa * self.preset.kappa_norm, -2.0, 2.0)),
            ])

        return np.array(ego_state + track_info + lookahead_features, dtype=np.float32)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
