"""
Môi trường Đua xe Đối đầu Đa tác tử (Multi-Agent Racing Environment)
Sử dụng MultiAgentMetaDrive tích hợp RacingMapManager với 2 xe:
  - 'player': Xe Người (điều khiển bằng bàn phím W-A-S-D)
  - 'ai': Xe AI (điều khiển bằng mạng Neural Network PPO)
Cả 2 xe tương tác vật lý trực tiếp với nhau và với rào chắn đường đua trong không gian 3D.
Tích hợp TelemetryTracker tính toán Delta Time (Xanh/Đỏ) thời gian thực.
"""

import numpy as np
from metadrive.envs.marl_envs import MultiAgentMetaDrive
from metadrive.envs.marl_envs.marl_racing_env import RacingMapManager
from metadrive.policy.manual_control_policy import KeyboardController
from game_env.telemetry import TelemetryTracker
from game_env.vehicle import GT3RaceCar


class MultiAgentRacingMetaDrive(MultiAgentMetaDrive):
    """Môi trường đa tác tử nạp trực tiếp RacingMapManager (12 blocks circuit khép kín)"""
    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())


class MultiAgentRacingEnv:
    def __init__(
        self,
        use_render: bool = True,
        window_size=(1280, 720),
        horizon: int = 3000,
        start_seed: int = 42,
        map_config=None,
        vehicle=None,
        **kwargs,
    ):
        self.use_render = use_render
        self.window_size = window_size
        self.horizon = horizon
        self.start_seed = start_seed
        self.dt = 0.02
        self.keyboard_controller = None
        
        from game_env.vehicle import GT3RaceCar
        self.car_player = vehicle if vehicle is not None else GT3RaceCar()
        self.car_ai = GT3RaceCar()
        
        # Telemetry trackers cho cả Player và AI
        self.telemetry_player = TelemetryTracker()
        self.telemetry_ai = TelemetryTracker()
        
        map_cfg = {
            "lane_num": 2,
            "lane_width": 4.5,
        }
        if isinstance(map_config, dict):
            map_cfg.update(map_config)
            
        self.config = {
            "use_render": use_render,
            "num_agents": 2,
            "is_multi_agent": True,
            "allow_respawn": False,
            "horizon": horizon,
            "start_seed": start_seed,
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "window_size": window_size,
            "physics_world_step_size": self.dt,
            "show_fps": True,
            "show_logo": False,
            "map_config": map_cfg,
            
            # Tắt ép bám tâm làn và phạt đè vạch
            "use_lateral_reward": False,
            "on_continuous_line_done": False,
            "out_of_route_done": False,
            "crash_vehicle_done": False,
            
            "vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": True,
                "show_line_to_dest": False,
                "lidar": {"num_lasers": 240, "distance": 50},
                "random_color": False,
            },
            "agent_configs": {
                "agent0": {
                    "spawn_lane_index": (">", ">>", 0),
                    "spawn_longitude": 15.0,
                    "_specified_spawn_lane": True,
                    "use_special_color": True,
                },
                "agent1": {
                    "spawn_lane_index": (">", ">>", 1),
                    "spawn_longitude": 10.0,
                    "_specified_spawn_lane": True,
                    "use_special_color": False,
                },
            },
        }
        
        self.env = MultiAgentRacingMetaDrive(self.config)
        self.compact_obs = True
        self.circuit_initialized = False
        self.circuit_length = 2186.0
        self.track_xy = None
        self.track_theta = None
        self.track_kappa = None
        self.track_s = None
        self.num_track_points = 0
        self.last_ai_action = [0.0, 0.0]

    def _init_circuit_polyline(self):
        map_obj = self.env.current_map
        if map_obj is None or not hasattr(map_obj, "road_network") or map_obj.road_network is None:
            return
            
        v = self.vehicles.get("agent1", self.vehicles.get("agent0"))
        if v is None or not hasattr(v, "navigation") or v.navigation is None:
            return
            
        ckpts = v.navigation.checkpoints
        route_lanes = []
        for i in range(len(ckpts) - 1):
            lane = map_obj.road_network.get_lane((ckpts[i], ckpts[i + 1], 0))
            if lane is not None:
                route_lanes.append(lane)
                
        if not route_lanes:
            return
            
        self.circuit_length = float(sum(l.length for l in route_lanes))
        track_points = []
        s_accum = 0.0
        
        for lane in route_lanes:
            s_samples = np.linspace(0, lane.length, max(2, int(lane.length)), endpoint=False)
            kappa = 0.0
            if hasattr(lane, "radius") and lane.radius > 0:
                is_cw = lane.is_clockwise() if callable(getattr(lane, "is_clockwise", None)) else bool(getattr(lane, "is_clockwise", False))
                kappa = (1.0 / float(lane.radius)) * (-1.0 if is_cw else 1.0)
                
            for s_l in s_samples:
                pos = lane.position(s_l, 0)
                theta = float(lane.heading_theta_at(s_l))
                track_points.append({
                    "pos": np.array(pos[:2], dtype=np.float32),
                    "theta": theta,
                    "kappa": float(kappa),
                    "s": float(s_accum + s_l),
                })
            s_accum += float(lane.length)
            
        self.track_xy = np.array([p["pos"] for p in track_points], dtype=np.float32)
        self.track_theta = np.array([p["theta"] for p in track_points], dtype=np.float32)
        self.track_kappa = np.array([p["kappa"] for p in track_points], dtype=np.float32)
        self.track_s = np.array([p["s"] for p in track_points], dtype=np.float32)
        self.num_track_points = len(track_points)
        self.circuit_initialized = True

    def get_compact_racing_obs(self, vehicle, action_applied=None):
        if not self.circuit_initialized:
            self._init_circuit_polyline()
        if action_applied is None:
            action_applied = self.last_ai_action
            
        pos = np.array(vehicle.position[:2], dtype=np.float32)
        heading = float(vehicle.heading_theta)
        vel = np.array(vehicle.velocity[:2], dtype=np.float32)
        speed_m_s = float(np.linalg.norm(vel))
        
        if (
            self.circuit_initialized
            and self.track_xy is not None
            and self.track_s is not None
            and self.track_theta is not None
            and self.track_kappa is not None
            and self.num_track_points > 1
        ):
            dists = np.sum((self.track_xy - pos) ** 2, axis=1)
            idx = int(dists.argmin())
            current_s = float(self.track_s[idx])
            track_pt = self.track_xy[idx]
            track_heading = float(self.track_theta[idx])
            track_curvature = float(self.track_kappa[idx])
        else:
            current_s = 0.0
            track_pt = pos
            track_heading = heading
            track_curvature = 0.0
            
        cos_head = np.cos(heading)
        sin_head = np.sin(heading)
        fwd_vec = np.array([cos_head, sin_head], dtype=np.float32)
        lat_vec = np.array([-sin_head, cos_head], dtype=np.float32)
        vx = float(np.dot(vel, fwd_vec))
        vy = float(np.dot(vel, lat_vec))
        
        delta_heading = (heading - track_heading + np.pi) % (2.0 * np.pi) - np.pi
        disp_center = pos - track_pt
        track_lat_normal = np.array([-np.sin(track_heading), np.cos(track_heading)], dtype=np.float32)
        lat_offset = float(np.dot(disp_center, track_lat_normal))
        
        dist_left = float(np.clip((4.5 - lat_offset) / 4.5, -2.0, 2.0))
        dist_right = float(np.clip((4.5 + lat_offset) / 4.5, -2.0, 2.0))
        yaw_rate = float(getattr(vehicle, "yaw_rate", 0.0))
        
        ego_state = [
            float(np.clip(speed_m_s / 40.0, 0.0, 1.5)),
            float(np.clip(vx / 40.0, -1.0, 1.5)),
            float(np.clip(vy / 20.0, -1.5, 1.5)),
            float(np.clip(yaw_rate / 3.14159, -2.0, 2.0)),
            float(action_applied[0]),
            float(self.last_ai_action[0]),
            float(self.last_ai_action[1]),
            float(np.cos(delta_heading)),
            float(np.sin(delta_heading)),
            float(np.clip(lat_offset / 4.5, -2.0, 2.0)),
        ]
        
        progress_ratio = float(np.clip(current_s / self.circuit_length, 0.0, 1.0))
        track_info = [
            dist_left,
            dist_right,
            0.0,
            progress_ratio,
            float(np.clip(track_curvature * 50.0, -2.0, 2.0)),
        ]
        
        lookahead_distances = [10.0, 20.0, 35.0, 50.0, 70.0]
        lookahead_features = []
        for dist_ahead in lookahead_distances:
            if (
                self.circuit_initialized
                and self.track_s is not None
                and self.track_xy is not None
                and self.track_theta is not None
                and self.track_kappa is not None
                and self.num_track_points > 1
            ):
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
            else:
                lookahead_features.extend([0.0, 0.0, 0.0, 0.0])
                
        return np.array(ego_state + track_info + lookahead_features, dtype=np.float32)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.telemetry_player.reset()
        self.telemetry_ai.reset()
        self._init_circuit_polyline()
        self.last_ai_action = [0.0, 0.0]
        
        if self.use_render and self.keyboard_controller is None:
            self.keyboard_controller = KeyboardController(pygame_control=False)
            
        # Gắn kết cấu hình động lực học GT3 cho cả 2 xe
        if "agent0" in self.vehicles:
            self.car_player.attach_agent(self.vehicles["agent0"])
        if "agent1" in self.vehicles:
            self.car_ai.attach_agent(self.vehicles["agent1"])

        if self.compact_obs and "agent1" in self.vehicles:
            obs["agent1"] = self.get_compact_racing_obs(self.vehicles["agent1"])
            
        return obs, info

    def get_player_action(self, player_vehicle):
        """Lấy hành động từ bàn phím người chơi (W, A, S, D)"""
        if self.keyboard_controller is not None:
            return self.keyboard_controller.process_input(player_vehicle)
        return [0.0, 0.0]

    def step(self, actions):
        """
        Thực hiện một bước mô phỏng vật lý đa tác tử
        actions: dict gồm {'agent0': action0, 'agent1': action1}
        """
        self.current_step += 1
        processed_actions = dict(actions)
        if "agent0" in actions:
            processed_actions["agent0"] = self.car_player.process_action(actions["agent0"])
        if "agent1" in actions:
            processed_actions["agent1"] = self.car_ai.process_action(actions["agent1"])
            self.last_ai_action = list(processed_actions["agent1"])
            
        obs, rewards, terminations, truncations, infos = self.env.step(processed_actions)
        current_time = self.current_step * self.dt
        
        # Cập nhật Telemetry cho cả 2 xe
        for aid, tracker in [("agent0", self.telemetry_player), ("agent1", self.telemetry_ai)]:
            if aid in self.vehicles and aid in actions and aid in infos:
                v = self.vehicles[aid]
                act = actions[aid]
                sample = tracker.log_step(
                    current_time=current_time,
                    position=v.position,
                    velocity=v.velocity,
                    steering=act[0],
                    throttle_brake=act[1],
                    dt=self.dt,
                )
                infos[aid]["speed_km_h"] = sample["speed_km_h"]
                infos[aid]["lap_time"] = current_time
                infos[aid]["delta_time"] = sample["delta_time"]
                infos[aid]["delta_color"] = sample["delta_color"]
                
        if self.compact_obs and "agent1" in self.vehicles:
            obs["agent1"] = self.get_compact_racing_obs(self.vehicles["agent1"], actions.get("agent1", [0.0, 0.0]))
            
        return obs, rewards, terminations, truncations, infos

    def render(self, text=None):
        return self.env.render(text=text)

    def close(self):
        return self.env.close()

    @property
    def vehicles(self):
        return self.env.vehicles

    @property
    def agents(self):
        return self.env.agents

    def set_vehicle_colors(self, player_id="agent0", ai_id="agent1"):
        """
        Cố định màu xe cho từng vòng đua:
          - Player: Xanh lá cây thể thao (Green Racing)
          - AI Agent: Đỏ rực thể thao (Ferrari / Red Bull)
        """
        try:
            from panda3d.core import Vec4
            if player_id in self.vehicles and self.vehicles[player_id].origin:
                self.vehicles[player_id].origin.setColorScale(Vec4(0.1, 0.9, 0.2, 1.0))
            if ai_id in self.vehicles and self.vehicles[ai_id].origin:
                self.vehicles[ai_id].origin.setColorScale(Vec4(0.95, 0.1, 0.1, 1.0))
        except Exception:
            pass
