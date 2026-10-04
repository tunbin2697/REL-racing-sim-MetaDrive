"""
core/racing_env.py: Môi trường Đua xe F1 2D Gym API.
Hỗ trợ cảm biến Raycast LiDAR 180 độ, kiểm tra va chạm, đo Lap time và tính toán phần thưởng RL.
"""

import numpy as np
import math
from .dynamics import CarDynamics
from .track_generator import TrackGenerator

class RacingEnv:
    """
    Môi trường Đua xe 2D chuẩn RL với Cảm biến Raycast LiDAR (F1TENTH Style).
    """
    def __init__(self, track_name="Monza", num_lidar_beams=11, lidar_range=30.0, dt=0.05):
        self.num_lidar_beams = num_lidar_beams
        self.lidar_range = lidar_range
        self.dt = dt
        self.car = CarDynamics()
        self.lidar_angles = np.linspace(-np.pi / 2, np.pi / 2, num_lidar_beams) # Quét 180 độ phía trước
        
        self.load_track(track_name)
        self.reset()

    def load_track(self, track_name):
        """Nạp mạch đua theo tên."""
        name_lower = track_name.lower()
        if "monza" in name_lower:
            self.track = TrackGenerator.generate_monza()
        elif "spa" in name_lower:
            self.track = TrackGenerator.generate_spa()
        elif "suzuka" in name_lower:
            self.track = TrackGenerator.generate_suzuka()
        elif "silverstone" in name_lower:
            self.track = TrackGenerator.generate_silverstone()
        elif "random" in name_lower or "procedural" in name_lower:
            self.track = TrackGenerator.generate_random_circuit()
        else:
            self.track = TrackGenerator.generate_monza()
            
        self.center = self.track["center"]
        self.left_border = self.track["left"]
        self.right_border = self.track["right"]
        self.num_waypoints = len(self.center)
        self.track_name = self.track["name"]

    def reset(self):
        """Khởi tạo lại trạng thái xuất phát."""
        p0 = self.center[0]
        p1 = self.center[1]
        initial_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        
        self.state = np.array([p0[0], p0[1], initial_yaw, 5.0, 0.0], dtype=np.float32)
        self.lap_time = 0.0
        self.total_time = 0.0
        self.steps_in_lap = 0
        self.lap_count = 0
        self.lap_history = []
        self._halfway_flag = False
        
        lidar_readings = self._compute_lidar()
        return self._get_observation(lidar_readings)

    def step(self, action, external_profiler_features=None):
        """
        Thực hiện một bước thời gian trong môi trường.
        action: [steer, accel/brake]
        external_profiler_features: tuple (lookahead_curvature, speed_delta) từ Profiler
        """
        steer_action = float(np.clip(action[0], -1.0, 1.0))
        accel_action = float(np.clip(action[1], -1.0, 1.0))
        
        # 1. Cập nhật động lực học xe
        self.state = self.car.step(self.state, steer_action, accel_action, self.dt)
        self.lap_time += self.dt
        self.total_time += self.dt
        self.steps_in_lap += 1
        
        x, y, yaw, v, steer = self.state
        
        # 2. Bắn tia Raycast LiDAR
        lidar_distances = self._compute_lidar()
        min_lidar = np.min(lidar_distances)
        
        # 3. Tìm waypoint gần nhất
        dists = np.linalg.norm(self.center - np.array([x, y]), axis=1)
        closest_idx = int(np.argmin(dists))
        dist_to_center = dists[closest_idx]
        
        # 4. Kiểm tra va chạm (Đâm rào khi cảm biến < 0.08m hoặc chệch ra ngoài mép đường đua)
        is_collision = (dist_to_center > (self.track["width"] * 0.5 - 0.3)) or (min_lidar <= 0.08)
        
        # 5. Kiểm tra hoàn thành vòng đua (Finish Lap)
        lap_completed = False
        if closest_idx > self.num_waypoints * 0.9 and not self._halfway_flag:
            self._halfway_flag = True
        elif closest_idx < self.num_waypoints * 0.1 and self._halfway_flag:
            lap_completed = True
            self.lap_count += 1
            self.lap_history.append(self.lap_time)
            self._halfway_flag = False
            self.lap_time = 0.0
            self.steps_in_lap = 0
            
        # 6. Thiết kế Hàm Phần Thưởng (Reward Function)
        wp_prev = self.center[(closest_idx - 1) % self.num_waypoints]
        wp_next = self.center[(closest_idx + 1) % self.num_waypoints]
        track_angle = math.atan2(wp_next[1] - wp_prev[1], wp_next[0] - wp_prev[0])
        angle_diff = (yaw - track_angle + np.pi) % (2 * np.pi) - np.pi
        forward_velocity = v * math.cos(angle_diff)
        
        if is_collision:
            reward = -100.0
            done = True
        else:
            edge_margin = (self.track["width"] * 0.5) - dist_to_center
            edge_penalty = -0.5 if edge_margin < 1.0 else 0.0
            reward = forward_velocity * 0.2 \
                     + edge_penalty \
                     - abs(steer_action) * 0.01 \
                     - 0.01 # Phạt thời gian trôi (ép lap time)
            if lap_completed:
                reward += 200.0 # Thưởng hoàn thành vòng đua
            done = is_collision or (self.steps_in_lap > 5000)
            
        obs = self._get_observation(lidar_distances, external_profiler_features)
        
        info = {
            "lap_completed": lap_completed,
            "lap_count": self.lap_count,
            "lap_time": self.lap_time,
            "lap_history": self.lap_history,
            "is_collision": is_collision,
            "closest_waypoint": closest_idx,
            "speed_kmh": v * 3.6,
            "lidar": lidar_distances,
            "dist_to_center": dist_to_center
        }
        
        return obs, reward, done, info

    def _get_observation(self, lidar_readings, external_profiler_features=None):
        """
        Tạo vector quan sát 16 chiều cho Policy / Agent:
        - 11 chiều: Khoảng cách cảm biến LiDAR chuẩn hóa [0, 1]
        - 1 chiều: Vận tốc hiện tại / max_speed
        - 1 chiều: Góc đánh lái hiện tại / max_steer
        - 1 chiều: Khoảng cách lệch tim đường chuẩn hóa
        - 2 chiều: Tín hiệu Thích ứng (Lookahead Curvature, Speed Delta)
        """
        x, y, yaw, v, steer = self.state
        norm_lidar = lidar_readings / self.lidar_range
        norm_v = v / self.car.max_speed
        norm_steer = steer / self.car.max_steer
        
        dists = np.linalg.norm(self.center - np.array([x, y]), axis=1)
        norm_dist_center = np.clip(np.min(dists) / (self.track["width"] / 2), 0.0, 1.0)
        
        if external_profiler_features is not None:
            f1, f2 = external_profiler_features
        else:
            f1, f2 = 0.0, 0.0
            
        obs = np.concatenate([
            norm_lidar,
            [norm_v, norm_steer, norm_dist_center, f1, f2]
        ]).astype(np.float32)
        
        return obs

    def _compute_lidar(self):
        """Bắn tia LiDAR đa hướng tìm giao điểm gần nhất với rào chắn."""
        x, y, yaw, _, _ = self.state
        origin = np.array([x, y])
        lidar_readings = np.full(self.num_lidar_beams, self.lidar_range, dtype=np.float32)
        
        # Tìm các đoạn rào chắn ở gần vị trí xe trong bán kính (lidar_range + 5m)
        left_pts = self.left_border
        right_pts = self.right_border
        d_left = np.linalg.norm(left_pts - origin, axis=1)
        d_right = np.linalg.norm(right_pts - origin, axis=1)
        
        search_rad = self.lidar_range + 5.0
        nearby_l_idx = np.where(d_left < search_rad)[0]
        nearby_r_idx = np.where(d_right < search_rad)[0]
        
        segments = []
        n_l = len(left_pts)
        for idx in nearby_l_idx:
            segments.append((left_pts[idx], left_pts[(idx + 1) % n_l]))
            
        n_r = len(right_pts)
        for idx in nearby_r_idx:
            segments.append((right_pts[idx], right_pts[(idx + 1) % n_r]))
            
        if not segments:
            return lidar_readings
            
        for b_idx, angle_offset in enumerate(self.lidar_angles):
            beam_angle = yaw + angle_offset
            dir_vec = np.array([math.cos(beam_angle), math.sin(beam_angle)])
            end_pt = origin + dir_vec * self.lidar_range
            
            min_dist = self.lidar_range
            for p1, p2 in segments:
                hit, dist = self._ray_segment_intersection(origin, end_pt, p1, p2)
                if hit and dist < min_dist:
                    min_dist = dist
                    
            lidar_readings[b_idx] = min_dist
            
        return lidar_readings

    @staticmethod
    def _ray_segment_intersection(p, p_end, q, q_end):
        """Thuật toán 2D Ray-Segment Intersection kiểm tra tia cắt đoạn thẳng."""
        r = p_end - p
        s = q_end - q
        r_cross_s = r[0] * s[1] - r[1] * s[0]
        q_minus_p = q - p
        
        if abs(r_cross_s) < 1e-8:
            return False, 0.0
            
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / r_cross_s
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / r_cross_s
        
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            dist = t * np.linalg.norm(r)
            return True, dist
            
        return False, 0.0
