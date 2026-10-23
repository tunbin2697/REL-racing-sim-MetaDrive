"""
Tác Tử Đua Xe Tối Ưu Lý Thuyết (Theoretical Dynamic Optimal Agent)
Áp dụng Lập trình Hướng đối tượng (OOP) & Điều khiển học Tự hành (Autonomous Racing Control):
- Nhận dữ liệu Ground Truth từ RacingBenchmarkEngine:
  + Quỹ đạo Racing Line tối ưu hình học (Convex QP Minimum Curvature)
  + Profile vận tốc tới hạn thời gian (Lipp & Boyd 3-Pass Time-Optimal Velocity Profile)
- Thuật toán điều khiển lái: Stanley Steering Controller với Lookahead thích ứng theo vận tốc.
- Thuật toán điều khiển chân ga/phanh: Feedforward Braking Anticipation + P-Controller.
- Xuất không gian hành động 3 chiều chuẩn GT3: [steer, throttle, brake].
- Đóng vai trò là Mẫu Đối Chứng Cận Trên (Theoretical Upper Bound Benchmark) & Ghost Car.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class OptimalRacingAgent:
    """
    Tác tử Đua xe Chuẩn mực (Ground Truth Dynamic Optimal Benchmark Agent)
    Sử dụng mô hình điều khiển học Stanley kết hợp Feedforward Braking để bám sát
    Racing Line và Profile tốc độ tối ưu thời gian.
    """
    def __init__(
        self,
        optimal_profile_file: Optional[str] = None,
        custom_track_xy: Optional[np.ndarray] = None,
        custom_track_kappa: Optional[np.ndarray] = None,
        custom_speed_kmh: Optional[np.ndarray] = None,
        k_lat: float = 1.4,
        k_head: float = 1.2,
        max_steer_rad: float = 0.349,  # 20.0 độ chuẩn GT3
    ):
        self.k_lat = float(k_lat)
        self.k_head = float(k_head)
        self.max_steer_rad = float(max_steer_rad)
        
        if custom_track_xy is not None:
            # Sử dụng track polyline tùy biến (ví dụ Oval circuit trong MultiAgentRacingEnv)
            self.optimal_line_xy = np.array(custom_track_xy, dtype=np.float32)
            self.num_points = len(self.optimal_line_xy)
            
            if custom_track_kappa is not None:
                self.optimal_line_kappa = np.array(custom_track_kappa, dtype=np.float32)
            else:
                self.optimal_line_kappa = np.zeros(self.num_points, dtype=np.float32)
                
            if custom_speed_kmh is not None:
                self.optimal_speed_kmh = np.array(custom_speed_kmh, dtype=np.float32)
            else:
                # Ước tính vận tốc an toàn theo bán kính cua: v = min(120, sqrt(mu * g / max(0.001, kappa)) * 3.6)
                safe_speeds = []
                for kp in self.optimal_line_kappa:
                    k_abs = abs(float(kp))
                    if k_abs > 0.005:
                        v_c = min(110.0, np.sqrt((1.35 * 9.81) / k_abs) * 3.6 * 0.90)
                    else:
                        v_c = 120.0
                    safe_speeds.append(v_c)
                self.optimal_speed_kmh = np.array(safe_speeds, dtype=np.float32)
                
            self.theoretical_best_lap_time = 0.0
        else:
            # Nạp dữ liệu tối ưu từ cache hoặc file
            if optimal_profile_file is None:
                optimal_profile_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts",
                    "silverstone_optimal_profile.npz"
                )
                
            assert os.path.exists(optimal_profile_file), f"Không tìm thấy file profile tối ưu: {optimal_profile_file}"
            data = np.load(optimal_profile_file)
            self.optimal_line_xy = np.array(data["optimal_line_xy"], dtype=np.float32)
            self.optimal_line_kappa = np.array(data["optimal_line_kappa"], dtype=np.float32)
            self.optimal_speed_kmh = np.array(data["optimal_speed_kmh"], dtype=np.float32)
            self.theoretical_best_lap_time = float(data["theoretical_best_lap_time"])
            self.num_points = len(self.optimal_line_xy)
        
        # Tính tiếp tuyến góc hướng của từng điểm trên racing line tối ưu
        opt_dx = np.gradient(self.optimal_line_xy[:, 0])
        opt_dy = np.gradient(self.optimal_line_xy[:, 1])
        self.optimal_headings = np.arctan2(opt_dy, opt_dx).astype(np.float32)
        
        # Tính khoảng cách s tích lũy
        diffs = np.linalg.norm(np.diff(self.optimal_line_xy, axis=0), axis=1)
        self.track_s = np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)
        self.circuit_length = float(self.track_s[-1])
        
        self.last_hint_idx = 0
        self.filtered_throttle = 0.5
        self.filtered_brake = 0.0

    def reset(self):
        """Khởi tạo lại trạng thái của Agent khi bắt đầu vòng đua mới"""
        self.last_hint_idx = 0
        self.filtered_throttle = 0.5
        self.filtered_brake = 0.0

    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        vehicle: Optional[Any] = None,
        pos_2d: Optional[np.ndarray] = None,
        heading: Optional[float] = None,
        speed_km_h: Optional[float] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """
        Giao diện đồng nhất với PPOAgent:
        Trả về: (action, log_prob, value)
        Trong đó action = [steer, throttle, brake] in [-1, 1]^3.
        """
        # 1. Trích xuất trạng thái xe từ vehicle hoặc pos/heading/speed
        if vehicle is not None:
            p_curr = np.array(vehicle.position[:2], dtype=np.float32)
            th_curr = float(vehicle.heading_theta)
            vel_vec = np.array(vehicle.velocity[:2], dtype=np.float32)
            v_curr_kmh = float(np.linalg.norm(vel_vec) * 3.6)
        elif pos_2d is not None and heading is not None and speed_km_h is not None:
            p_curr = np.array(pos_2d[:2], dtype=np.float32)
            th_curr = float(heading)
            v_curr_kmh = float(speed_km_h)
        elif obs is not None and len(obs) >= 14:
            # Ước tính từ vector quan sát compact obs
            v_curr_kmh = float(obs[0] * 40.0 * 3.6)
            # Dùng hint index để ước lượng vị trí
            p_curr = self.optimal_line_xy[self.last_hint_idx]
            th_curr = self.optimal_headings[self.last_hint_idx]
        else:
            return np.array([0.0, 1.0, -1.0], dtype=np.float32), 0.0, 0.0

        # 2. Tìm điểm gần nhất trên Racing Line tối ưu
        window = 100
        indices = np.arange(self.last_hint_idx - 20, self.last_hint_idx + window) % self.num_points
        d_sq = np.sum((self.optimal_line_xy[indices] - p_curr) ** 2, axis=1)
        min_local = int(np.argmin(d_sq))
        idx_near = int(indices[min_local])
        self.last_hint_idx = idx_near

        # 3. Điều khiển lái ngang: Pure Pursuit Controller chuẩn xe đua tự hành với Lookahead thích ứng
        wheelbase = 2.75  # Chiều dài cơ sở GT3 (m)
        ds = max(0.5, self.circuit_length / self.num_points)
        
        # Xử lý đề-pa xuất phát (Grid Launch Transition):
        # Khi xuất phát ở lề phải mà racing line ở lề trái, mở rộng khoảng cách ngắm để tiếp cận thoai thoải,
        # tránh bẻ hết lái ở tốc độ thấp làm lết bánh trước và triệt tiêu gia tốc dọc
        p_near = self.optimal_line_xy[idx_near]
        lat_err = float(np.linalg.norm(p_curr - p_near))
        min_ld = max(4.8, 3.2 * lat_err) if v_curr_kmh < 50.0 else 4.8
        L_d = float(np.clip(max(min_ld, 4.0 + 0.11 * v_curr_kmh), 4.8, 22.0))
        
        lookahead_steps = int(round(L_d / ds))
        idx_target = (idx_near + lookahead_steps) % self.num_points
        p_target = self.optimal_line_xy[idx_target]

        # Chuyển đổi tọa độ điểm ngắm sang hệ trục tọa độ của xe (Vehicle Body Frame)
        dx = float(p_target[0] - p_curr[0])
        dy = float(p_target[1] - p_curr[1])
        cos_th = float(np.cos(th_curr))
        sin_th = float(np.sin(th_curr))
        
        # Tọa độ dọc x_body và ngang y_body trong hệ quy chiếu thân xe
        x_body = cos_th * dx + sin_th * dy
        y_body = -sin_th * dx + cos_th * dy
        actual_lookahead_sq = max(4.0, x_body ** 2 + y_body ** 2)
        
        # Công thức độ cong Pure Pursuit: curvature = 2 * y_body / L^2
        curvature = 2.0 * y_body / actual_lookahead_sq
        steer_rad = np.arctan(curvature * wheelbase)
        steer_norm = float(np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0))

        # 4. Điều khiển dọc: Lift-and-Coast + Kamm's Friction Circle Trail-Braking + Slew-Rate Limiter
        v_curr_ms = max(0.1, v_curr_kmh / 3.6)
        
        # Quét trước một cự ly ngắm thích ứng (0.35s) để đón đầu điểm giảm tốc
        lookahead_m = float(np.clip(0.35 * v_curr_ms, 3.5, 14.0))
        steps_ahead = max(1, int(round(lookahead_m / ds)))
        idx_speed = (idx_near + steps_ahead) % self.num_points
        v_target_kmh = float(self.optimal_speed_kmh[idx_speed])
        
        v_err_kmh = v_target_kmh - v_curr_kmh

        # Trail-Braking qua Elip ma sát Kamm (Friction Circle):
        steer_mag = abs(steer_norm)
        avail_long_grip = float(np.sqrt(max(0.10, 1.0 - min(0.90, (0.90 * steer_mag) ** 2))))
        traction_factor = float(np.sqrt(max(0.35, 1.0 - min(0.85, (0.85 * steer_mag) ** 2))))

        if v_err_kmh >= 0.0:
            # Tăng tốc (Acceleration): nhả phanh, mở ga
            target_brake = 0.0
            if (steer_mag < 0.35 and v_curr_kmh < 80.0) or (steer_mag < 0.15 and v_curr_kmh < 118.0):
                # Đoạn thẳng hoặc đề-pa xuất phát: full ga
                target_throttle = 1.0
            else:
                # Trong cua hoặc thoát cua: ga tăng mượt theo sai số tốc độ kết hợp lực bám ngang
                maint_thr = 0.20 if v_curr_kmh > 40.0 else 0.0
                target_throttle = float(np.clip(maint_thr + (v_err_kmh / 8.0) * (1.0 - maint_thr), 0.0, 1.0)) * traction_factor
        else:
            # Giảm tốc (Deceleration):
            overspeed_kmh = -v_err_kmh
            if overspeed_kmh <= 2.5:
                # Lift-and-Coast: Chỉ nhả bớt ga, TUYỆT ĐỐI KHÔNG CHẠM PHANH
                target_throttle = 0.0
                target_brake = 0.0
            else:
                # Phanh lũy tiến liên tục theo sai số tốc độ và độ bám dọc còn lại của Elip Kamm
                target_throttle = 0.0
                brk_fraction = float(np.clip((overspeed_kmh - 2.5) / 10.0, 0.0, 1.0))
                max_brk_pedal = 0.35
                target_brake = float(brk_fraction * max_brk_pedal * avail_long_grip)

        # Bộ lọc thông thấp bậc 1 (1st-Order Low-Pass Filter / EMA, alpha=0.20):
        # Làm mượt độc lập cả 2 kênh chân ga và chân phanh
        alpha = 0.20
        self.filtered_throttle = float((1.0 - alpha) * self.filtered_throttle + alpha * target_throttle)
        self.filtered_brake = float((1.0 - alpha) * self.filtered_brake + alpha * target_brake)

        # Chuẩn hóa về [-1, 1]^3 khớp định dạng 3 trục của Gym Action Space
        throttle_action = float(np.clip(self.filtered_throttle * 2.0 - 1.0, -1.0, 1.0))
        brake_action = float(np.clip(self.filtered_brake * 2.0 - 1.0, -1.0, 1.0))

        action = np.array([steer_norm, throttle_action, brake_action], dtype=np.float32)
        net_pedal = float(self.filtered_throttle - self.filtered_brake)
        return action, steer_norm, net_pedal
