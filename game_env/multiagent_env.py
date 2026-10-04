"""
Môi trường Đua xe Đối đầu Đa tác tử (Multi-Agent Racing Environment)
Sử dụng MultiAgentMetaDrive với 2 xe:
  - 'player': Xe Người (điều khiển bằng bàn phím W-A-S-D)
  - 'ai': Xe AI (điều khiển bằng mạng Neural Network PPO)
Cả 2 xe tương tác vật lý trực tiếp với nhau và với rào chắn đường đua trong không gian 3D.
"""

import numpy as np
from metadrive.envs.marl_envs import MultiAgentMetaDrive
from metadrive.policy.manual_control_policy import KeyboardController

class MultiAgentRacingEnv:
    def __init__(
        self,
        use_render: bool = True,
        map_config: str = "O",          # 'O' = Vòng đua Oval khép kín, hoặc map số
        window_size=(1280, 720),
        horizon: int = 2500,
        start_seed: int = 42,
    ):
        self.use_render = use_render
        self.map_config = map_config
        self.window_size = window_size
        self.horizon = horizon
        self.start_seed = start_seed
        
        self.config = {
            "use_render": use_render,
            "num_agents": 2,
            "is_multi_agent": True,
            "allow_respawn": False,
            "map": map_config,
            "horizon": horizon,
            "start_seed": start_seed,
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "window_size": window_size,
            "physics_world_step_size": 0.02,
            "show_fps": True,
            "show_logo": False,
            "vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": True,
                "show_line_to_dest": False,
                "lidar": {"num_lasers": 240},
                "random_color": False,
            },
            "agent_configs": {
                "agent0": {
                    "spawn_lane_index": (">", ">>", 0),
                    "spawn_longitude": 12.0,
                    "_specified_spawn_lane": True,
                    "use_special_color": True,
                },
                "agent1": {
                    "spawn_lane_index": (">", ">>", 1),
                    "spawn_longitude": 8.0,
                    "_specified_spawn_lane": True,
                    "use_special_color": False,
                },
            },
        }
        
        self.env = MultiAgentMetaDrive(self.config)
        self.keyboard_controller = None

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
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
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
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

    def set_vehicle_colors(self, player_id, ai_id):
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
