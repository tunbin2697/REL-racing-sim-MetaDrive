"""
core/circuit_profiler.py: Module Khảo sát Hình học Vi phân & Quy hoạch Động Bellman OVP.

Chức năng:
1. Tính toán độ cong vi phân kappa(s) và bán kính góc cua R(s).
2. Tối ưu hóa quỹ đạo đua Ngoài - Trong - Ngoài (Out-In-Out Racing Line) bằng lọc Hanning.
3. Giải bài toán Quy hoạch Động Bellman (Bellman Optimal Velocity Profile):
   - Cho Quỹ đạo Racing Line (Out-In-Out).
   - Cho Tim đường (Centerline constrained) để so sánh thực nghiệm.
"""

import numpy as np
import math

class CircuitProfiler:
    def __init__(self, tire_friction=0.9, gravity=9.81, lookahead_distance=25.0, track_width=16.0, default_center=None):
        self.mu = tire_friction              # Hệ số bám của lốp xe đua
        self.g = gravity                     # Gia tốc trọng trường (m/s^2)
        self.lookahead_dist = lookahead_distance # Khoảng cách nhìn xa (m)
        self.track_width = track_width       # Bề rộng đường đua (m)
        self.default_center = np.array(default_center) if default_center is not None else None
        self.is_profiled = False
        self.track_length = 0.0
        
        # Quỹ đạo và tốc độ Racing Line (Out-In-Out)
        self.racing_line = None
        self.racing_curvatures = None
        self.optimal_speeds = None
        
        # Tốc độ tối ưu giới hạn ở tim đường (Centerline Constrained)
        self.centerline_curvatures = None
        self.centerline_optimal_speeds = None
        
        self.recorded_points = []
        self.curvatures = None
        self.apex_speeds = None
        self.distances = None

    def record_step(self, x, y):
        """Ghi nhận tọa độ trong Vòng 1 (Reconnaissance Lap)."""
        if not self.is_profiled:
            if len(self.recorded_points) == 0:
                self.recorded_points.append([x, y])
            else:
                last_p = self.recorded_points[-1]
                if (x - last_p[0])**2 + (y - last_p[1])**2 >= 0.09:
                    self.recorded_points.append([x, y])

    def profile_track_center(self, center_pts):
        """Khởi tạo nhanh profiler từ tọa độ tim đường có sẵn với nội suy mượt."""
        pts = np.array(center_pts)
        n = len(pts)
        dense_n = max(1200, n * 3)
        t_orig = np.linspace(0, 1, n, endpoint=False)
        t_dense = np.linspace(0, 1, dense_n, endpoint=False)
        x_dense = np.interp(t_dense, t_orig, pts[:, 0], period=1.0)
        y_dense = np.interp(t_dense, t_orig, pts[:, 1], period=1.0)
        self.recorded_points = np.column_stack([x_dense, y_dense]).tolist()
        return self.finalize_lap_1()

    def finalize_lap_1(self):
        """
        Xử lý chuỗi tọa độ để tính toán:
        1. Curvature Profile của tim đường.
        2. Quỹ đạo Out-In-Out Racing Line.
        3. Bellman OVP trên Racing Line.
        4. Bellman OVP trên Centerline (để so sánh chính sách).
        """
        if len(self.recorded_points) < 50:
            return False
            
        pts = np.array(self.recorded_points)
        n = len(pts)
        
        # 1. Độ dài tích lũy
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.distances = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self.track_length = float(self.distances[-1])
        
        # 2. Hướng tiếp tuyến theta tại mỗi điểm
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        thetas = np.arctan2(dy, dx)
        thetas_unwrapped = np.unwrap(thetas)
        
        # 3. Độ cong hình học tim đường: kappa = |d_theta / d_s|
        d_theta = np.gradient(thetas_unwrapped)
        d_s = np.clip(np.gradient(self.distances), 0.1, None)
        self.curvatures = np.abs(d_theta / d_s)
        
        kernel = np.ones(7) / 7.0
        self.curvatures = np.convolve(self.curvatures, kernel, mode='same')
        self.centerline_curvatures = np.copy(self.curvatures)
        
        # 4. Tối ưu hóa Quỹ đạo Đua xe (Out-In-Out Racing Line)
        diffs_pts = np.diff(np.vstack([pts, pts[0]]), axis=0)
        lens_pts = np.linalg.norm(diffs_pts, axis=1)
        tangents = diffs_pts / np.maximum(lens_pts[:, None], 1e-6)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        
        signed_curv = d_theta / d_s
        r_local = 1.0 / (np.abs(signed_curv) + 1e-4)
        local_max_offset = np.minimum((self.track_width / 2.0) - 2.5, r_local * 0.45)
        raw_offsets = np.clip(signed_curv * 50.0, -local_max_offset, local_max_offset)
        
        filter_size = min(31, max(11, n // 15))
        if filter_size % 2 == 0: 
            filter_size += 1
        h_kernel = np.hanning(filter_size)
        h_kernel /= np.sum(h_kernel)
        smoothed_offsets = np.convolve(np.tile(raw_offsets, 3), h_kernel, mode='same')[n:2*n]
        self.racing_line = pts + normals * smoothed_offsets[:, None]
        
        # 5. Độ cong hiệu dụng của Racing Line
        dx_r = np.gradient(self.racing_line[:, 0])
        dy_r = np.gradient(self.racing_line[:, 1])
        thetas_r = np.unwrap(np.arctan2(dy_r, dx_r))
        self.racing_curvatures = np.convolve(np.abs(np.gradient(thetas_r) / d_s), np.ones(5)/5.0, mode='same')
        
        # 6. Quy hoạch động Bellman OVP trên RACING LINE
        raw_speeds_rl = np.sqrt((0.80 * self.g) / np.maximum(self.racing_curvatures, 0.003))
        self.apex_speeds = np.clip(raw_speeds_rl, 6.5, 32.0)
        self.optimal_speeds = self._solve_bellman_ovp(self.apex_speeds, self.distances)
        
        # 7. Quy hoạch động Bellman OVP trên TIM ĐƯỜNG (CENTERLINE)
        raw_speeds_cl = np.sqrt((0.80 * self.g) / np.maximum(self.centerline_curvatures, 0.003))
        cl_apex_speeds = np.clip(raw_speeds_cl, 6.5, 32.0)
        self.centerline_optimal_speeds = self._solve_bellman_ovp(cl_apex_speeds, self.distances)
        
        self.is_profiled = True
        return True

    def _solve_bellman_ovp(self, apex_speeds, distances, a_brake=6.5, a_accel=5.0):
        """Giải bài toán quy hoạch động Bellman với Quét Lùi (phanh) và Quét Tiến (ga)."""
        speeds = np.copy(apex_speeds)
        n = len(speeds)
        
        # Quét lùi (Backward Bellman Sweep): Giới hạn phanh
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                next_idx = (i + 1) % n
                ds = max(0.1, abs(distances[next_idx] - distances[i]))
                allowed_v = math.sqrt(speeds[next_idx]**2 + 2.0 * a_brake * ds)
                if allowed_v < speeds[i]:
                    speeds[i] = allowed_v
                    
        # Quét tiến (Forward Bellman Sweep): Giới hạn lực kéo
        for _ in range(2):
            for i in range(n):
                prev_idx = (i - 1) % n
                ds = max(0.1, abs(distances[i] - distances[prev_idx]))
                allowed_v = math.sqrt(speeds[prev_idx]**2 + 2.0 * a_accel * ds)
                if allowed_v < speeds[i]:
                    speeds[i] = allowed_v
                    
        return speeds

    def get_reference_path(self, lap_number=1, policy_type="racing_line"):
        """Trả về quỹ đạo tham chiếu theo số vòng và loại chính sách."""
        if policy_type == "centerline" or lap_number == 1 or not self.is_profiled or self.racing_line is None:
            if self.default_center is not None:
                return self.default_center
            if len(self.recorded_points) > 5:
                return np.array(self.recorded_points)
            return np.zeros((2, 2))
        return self.racing_line

    def get_target_speed(self, current_x, current_y, lap_number=2, policy_type="racing_line"):
        """Truy vấn tốc độ tối ưu quy hoạch động Bellman theo vị trí."""
        if not self.is_profiled:
            return 11.0 # 40 km/h an toàn
            
        if policy_type == "centerline":
            if self.centerline_optimal_speeds is None:
                return 11.0
            ref = self.get_reference_path(1, "centerline")
            dists = np.linalg.norm(ref - np.array([current_x, current_y]), axis=1)
            curr_idx = int(np.argmin(dists))
            return float(self.centerline_optimal_speeds[curr_idx])
            
        # Racing line speeds
        if self.optimal_speeds is None:
            return 11.0
        ref = self.get_reference_path(lap_number, "racing_line")
        dists = np.linalg.norm(ref - np.array([current_x, current_y]), axis=1)
        curr_idx = int(np.argmin(dists))
        v_opt = self.optimal_speeds[curr_idx]
        
        if lap_number == 2:
            return min(v_opt, 22.0) # Vòng 2: Time Attack (~80 km/h)
        else:
            return min(v_opt, 28.0) # Vòng 3: Full Racing Line Attack (~100 km/h)

    def get_lookahead_features(self, current_x, current_y, current_v):
        """Trích xuất 2 đặc trưng thích ứng nhìn xa (Lookahead Curvature & Speed Delta)."""
        if not self.is_profiled or len(self.recorded_points) == 0:
            return 0.0, 0.0
            
        pts = np.array(self.recorded_points)
        dists = np.linalg.norm(pts - np.array([current_x, current_y]), axis=1)
        curr_idx = int(np.argmin(dists))
        n = len(pts)
        
        dynamic_lookahead = max(18.0, current_v * 1.5)
        accum_dist = 0.0
        horizon_indices = [curr_idx]
        for step in range(1, n):
            idx = (curr_idx + step) % n
            prev = (idx - 1) % n
            accum_dist += np.linalg.norm(pts[idx] - pts[prev])
            horizon_indices.append(idx)
            if accum_dist >= dynamic_lookahead:
                break
                
        kappa_ahead = float(np.max(self.curvatures[horizon_indices]))
        target_v = float(self.optimal_speeds[curr_idx]) if self.optimal_speeds is not None else 11.0
        
        f1 = float(np.clip(kappa_ahead * 10.0, 0.0, 2.0))
        f2 = float(np.clip((target_v - current_v) / 40.0, -1.0, 1.0))
        return f1, f2
