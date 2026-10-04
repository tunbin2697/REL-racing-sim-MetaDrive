"""
MetaRacingEnv: Môi trường Đua xe 3D chuẩn RL dựa trên MetaDrive Simulator
Thiết kế MDP chuẩn mực (Sutton & Barto), Continuous State & Action Spaces,
không mớm đường đua (Không cung cấp waypoints/racing line vẽ sẵn).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive import MetaDriveEnv

class MetaRacingEnv(gym.Env):
    """
    Wrapper môi trường đua xe 3D sử dụng engine MetaDrive / Panda3D.
    
    Không gian Trạng thái (Observation Space):
        Box(259,) chứa:
        - 240 tia LiDAR quét 360 độ xung quanh xe (nhận diện mép đường, rào chắn, xe khác).
        - 19 chiều telemetry nội tại: vận tốc dài vx, vận tốc ngang vy, gia tốc, góc lái hiện tại,
          vector điều hướng đường cơ bản.
        
    Không gian Hành động (Action Space):
        Box(-1.0, 1.0, (2,)) gồm:
        - action[0]: Góc đánh lái vô lăng (Steering) [-1: rẽ trái hết cỡ, +1: rẽ phải hết cỡ].
        - action[1]: Chân ga / Thắng (Throttle / Brake) [-1: thắng gấp / lùi, +1: đạp kịch ga].
        
    Hàm Phần thưởng Tự nhiên (Natural Reward Function):
        R_t = R_progress + R_speed + R_crash + R_out + R_success + R_step
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(
        self,
        use_render: bool = False,
        traffic_density: float = 0.0,
        map_config: int = 3,
        start_seed: int = 42,
        num_scenarios: int = 20,
        horizon: int = 1500,
        manual_control: bool = False,
        window_size=(1280, 720),
    ):
        super().__init__()
        
        self.use_render = use_render
        self.traffic_density = traffic_density
        self.map_config = map_config
        self.start_seed = start_seed
        self.num_scenarios = num_scenarios
        self.horizon = horizon
        self.manual_control = manual_control
        
        # Cấu hình MetaDrive engine
        self.config = {
            "use_render": use_render,
            "manual_control": manual_control,
            "traffic_density": traffic_density,
            "start_seed": start_seed,
            "num_scenarios": num_scenarios,
            "map": map_config,
            "horizon": horizon,
            "window_size": window_size,
            "use_lateral_reward": True,
            "driving_reward": 1.5,
            "speed_reward": 0.2,
            "out_of_road_penalty": 10.0,
            "crash_vehicle_penalty": 10.0,
            "crash_sidewalk_penalty": 10.0,
            "success_reward": 25.0,
            "vehicle_config": {
                "show_navi_mark": False,          # Tắt mũi tên dẫn đường trên mặt đất
                "show_dest_mark": True,
                "show_line_to_dest": False,       # Tắt vạch kẻ mớm đường chạy
            },
            "show_interface": use_render,
            "show_logo": False,
        }
        
        self.env = MetaDriveEnv(self.config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.last_route_completion = 0.0
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
            
        self.last_route_completion = info.get("route_completion", 0.0)
        self.current_step = 0
        return obs.astype(np.float32), info

    def step(self, action):
        self.current_step += 1
        
        # Đảm bảo action là mảng numpy float32 trong [-1, 1]
        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Trích xuất telemetry từ simulator
        route_comp = info.get("route_completion", 0.0)
        velocity = info.get("velocity", 0.0)             # m/s
        speed_kmh = velocity * 3.6
        crashed = info.get("crash", False) or info.get("crash_sidewalk", False) or info.get("crash_object", False)
        out_of_road = info.get("out_of_road", False)
        arrived = info.get("arrive_dest", False)
        
        # MetaDrive base_reward đã tích hợp tối ưu:
        # 1.5 * delta_longitudinal * lateral_factor + speed_reward - penalties
        r_step = -0.005
        custom_reward = base_reward + r_step
        
        # Điều kiện kết thúc episode
        if crashed or out_of_road or arrived:
            terminated = True
            
        info["custom_reward"] = custom_reward
        info["speed_kmh"] = speed_kmh
        info["base_reward"] = base_reward
        
        return obs.astype(np.float32), float(custom_reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
