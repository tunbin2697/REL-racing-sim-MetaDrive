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
        reference_lap_file: str = "logs/reference_lap.json",
    ):
        self.use_render = use_render
        self.window_size = window_size
        self.horizon = horizon
        self.start_seed = start_seed
        self.dt = 0.02
        
        # Telemetry trackers cho cả Player và AI
        self.telemetry_player = TelemetryTracker(reference_lap_file=reference_lap_file)
        self.telemetry_ai = TelemetryTracker(reference_lap_file=reference_lap_file)
        
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
            "map_config": {
                "lane_num": 2,
                "lane_width": 4.5,
            },
            
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
        self.keyboard_controller = None
        self.current_step = 0

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.telemetry_player.reset()
        self.telemetry_ai.reset()
        
        if self.use_render and self.keyboard_controller is None:
            self.keyboard_controller = KeyboardController(pygame_control=False)
            
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
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
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
