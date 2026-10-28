"""
Tác Tử Đua Xe Tối Ưu Lý Thuyết (Theoretical Dynamic Optimal Agent)
Áp dụng Lập trình Hướng đối tượng (OOP) & Điều khiển học Tự hành (Autonomous Racing Control):
- Nhận dữ liệu Ground Truth từ RacingBenchmarkEngine:
  + Quỹ đạo Racing Line tối ưu hình học Out-In-Out (1D Frenet Minimum Curvature QP + C2 Spline)
  + Profile vận tốc tới hạn thời gian đơn điệu (Monotonic Lipp & Boyd 3-Pass Velocity Profile)
- Thuật toán điều khiển lái: Adaptive Pure Pursuit Controller khớp động lực học Bullet Physics.
- Thuật toán điều khiển chân ga/phanh: Longitudinal Racing MPC (Convex QP) kết hợp Predictive Braking Horizon
  và tách biệt dứt khoát giữa Điều tiết ga (Throttle Feathering) và Vùng phanh thật (Threshold/Trail-Braking).
- Xuất không gian hành động 3 chiều chuẩn GT3: [steer, throttle, brake].
- Đóng vai trò là Mẫu Đối Chứng Cận Trên (Theoretical Upper Bound Benchmark) & Ghost Car.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import lsq_linear


class LongitudinalRacingMPC:
    """
    Bộ điều khiển Dự báo Mô hình Dọc (Longitudinal Model Predictive Control)
    Áp dụng Tối ưu hóa Lồi Quadratic Programming (Convex QP) để tìm chuỗi gia tốc a_x tối ưu
    trong cửa sổ dự báo tương lai N steps, giải quyết triệt để hiện tượng nhấp nhả phanh và lỡ trớn.
    """
    def __init__(
        self,
        horizon: int = 15,
        dt: float = 0.10,
        max_brake_acc: float = 6.5,
        max_engine_acc: float = 3.5,
        mu: float = 1.35,
    ):
        self.N = horizon
        self.dt = dt
        self.max_brake = max_brake_acc
        self.max_engine = max_engine_acc
        self.mu = mu
        self.g = 9.81

        # Trọng số tối ưu hóa (Cost Function Weights)
        self.q_v = 1.8        # Phạt sai số vận tốc so với mục tiêu (m/s)
        self.r_a = 0.04       # Phạt công suất gia tốc
        self.w_da = 1.6       # Phạt độ giật (Jerk rate: Delta a_x) -> Mượt mà, chống nhấp nhả ga/phanh

        N = self.N
        # Ma trận động lực học dọc: v_k = v_0 + dt * sum_{j=0}^{k-1} a_j
        self.L_lower = np.tril(np.ones((N, N))) * self.dt

        # Toán tử biến thiên gia tốc Delta a_k = a_k - a_{k-1}
        self.M = np.eye(N)
        for i in range(1, N):
            self.M[i, i - 1] = -1.0
        self.M0 = np.zeros(N)
        self.M0[0] = 1.0

        # Ma trận bài toán Bounded Least Squares: min || A_ls * A_vec - b_ls ||_2^2
        self.A_ls = np.vstack([
            np.sqrt(self.q_v) * self.L_lower,
            np.sqrt(self.r_a) * np.eye(N),
            np.sqrt(self.w_da) * self.M
        ])

    def solve(
        self,
        v_curr_ms: float,
        target_v_ms: np.ndarray,
        target_kappa: np.ndarray,
        a_prev: float,
        steer_norm: float = 0.0,
    ) -> float:
        """
        Giải bài toán QP lồi trong thời gian < 0.1 ms bằng trf bounded least-squares
        Trả về gia tốc tối ưu bước đầu tiên a_opt (m/s^2)
        """
        N = self.N
        err_v = target_v_ms - v_curr_ms
        b_ls = np.concatenate([
            np.sqrt(self.q_v) * err_v,
            np.zeros(N),
            np.sqrt(self.w_da) * (self.M0 * a_prev)
        ])

        # Ràng buộc Kamm Friction Circle Ellipse
        mu_g = self.mu * self.g * 0.95
        steer_mag = abs(steer_norm)
        avail_long_grip = float(np.sqrt(max(0.12, 1.0 - min(0.88, (0.85 * steer_mag) ** 2))))

        lb = np.zeros(N)
        ub = np.zeros(N)
        for k in range(N):
            v_targ = max(4.0, float(target_v_ms[k]))
            kap = abs(float(target_kappa[k]))
            a_lat = (v_targ ** 2) * kap
            max_a_long = np.sqrt(max(0.25, mu_g ** 2 - min(mu_g ** 2 - 0.25, a_lat ** 2)))
            lb[k] = -min(self.max_brake, max_a_long * avail_long_grip)
            ub[k] = min(self.max_engine, max_a_long)

        res = lsq_linear(self.A_ls, b_ls, bounds=(lb, ub), method='trf', max_iter=15, tol=1e-3)
        return float(res.x[0])


class OptimalRacingAgent:
    """
    Tác tử Đua xe Chuẩn mực (Ground Truth Dynamic Optimal Benchmark Agent)
    Sử dụng mô hình điều khiển học Pure Pursuit bám Racing Line kết hợp Bộ điều khiển
    Dự báo Mô hình Dọc (Longitudinal MPC) để bám sát Profile tốc độ tối ưu thời gian.
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
        max_brk_pedal: float = 0.95,   # Khai thác 95% hành trình phanh thể thao chuẩn GT3
        use_mpc: bool = True,          # Bật bộ điều khiển Longitudinal MPC
    ):
        self.k_lat = float(k_lat)
        self.k_head = float(k_head)
        self.max_steer_rad = float(max_steer_rad)
        self.max_brk_pedal = float(max_brk_pedal)
        self.use_mpc = bool(use_mpc)
        
        if custom_track_xy is not None:
            self.optimal_line_xy = np.array(custom_track_xy, dtype=np.float32)
            self.num_points = len(self.optimal_line_xy)
            
            if custom_track_kappa is not None:
                self.optimal_line_kappa = np.array(custom_track_kappa, dtype=np.float32)
            else:
                self.optimal_line_kappa = np.zeros(self.num_points, dtype=np.float32)
                
            if custom_speed_kmh is not None:
                self.optimal_speed_kmh = np.array(custom_speed_kmh, dtype=np.float32)
            else:
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
        
        opt_dx = np.gradient(self.optimal_line_xy[:, 0])
        opt_dy = np.gradient(self.optimal_line_xy[:, 1])
        self.optimal_headings = np.arctan2(opt_dy, opt_dx).astype(np.float32)
        
        diffs = np.linalg.norm(np.diff(self.optimal_line_xy, axis=0), axis=1)
        self.track_s = np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)
        self.circuit_length = float(self.track_s[-1])
        
        self.last_hint_idx = 0
        self.step_count = 0
        self.filtered_throttle = 0.5
        self.filtered_brake = 0.0
        self.last_a_opt = 0.0
        self.mu = 1.35
        if self.use_mpc:
            self.mpc = LongitudinalRacingMPC(
                horizon=15,
                dt=0.10,
                max_brake_acc=6.5,
                max_engine_acc=3.5,
                mu=self.mu,
            )

    def reset(self):
        """Khởi tạo lại trạng thái của Agent khi bắt đầu vòng đua mới"""
        self.last_hint_idx = 0
        self.step_count = 0
        self.filtered_throttle = 0.5
        self.filtered_brake = 0.0
        self.last_a_opt = 0.0

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
            v_curr_kmh = float(obs[0] * 40.0 * 3.6)
            p_curr = self.optimal_line_xy[self.last_hint_idx]
            th_curr = self.optimal_headings[self.last_hint_idx]
        else:
            return np.array([0.0, 1.0, -1.0], dtype=np.float32), 0.0, 0.0

        # 2. Tìm điểm gần nhất trên Racing Line tối ưu
        window = 100
        indices = np.arange(self.last_hint_idx - 20, self.last_hint_idx + window) % self.num_points
        d_sq = np.sum((self.optimal_line_xy[indices] - p_curr) ** 2, axis=1)
        min_local = int(d_sq.argmin())
        idx_near = int(indices[min_local])
        self.last_hint_idx = idx_near

        # 3. Điều khiển lái ngang: Pure Pursuit Controller chuẩn xe đua tự hành với Lookahead thích ứng
        wheelbase = 2.75  # Chiều dài cơ sở GT3 (m)
        ds = max(0.5, self.circuit_length / self.num_points)
        self.step_count += 1
        
        is_grid_launch = bool(self.step_count <= 60 and v_curr_kmh < 85.0)
        p_near = self.optimal_line_xy[idx_near]
        lat_err = float(np.linalg.norm(p_curr - p_near))
        k_near = abs(float(self.optimal_line_kappa[idx_near]))
        k_ahead = abs(float(self.optimal_line_kappa[(idx_near + 8) % self.num_points]))
        k_ahead_far = abs(float(self.optimal_line_kappa[(idx_near + 14) % self.num_points]))
        k_eff = max(k_near, 0.88 * k_ahead, 0.72 * k_ahead_far)
        
        if is_grid_launch:
            L_d = 15.0
        else:
            raw_ld = max(5.5, 4.8 + 0.115 * v_curr_kmh)
            curv_damping = 1.0 + 11.5 * k_eff
            L_d = float(np.clip(raw_ld / curv_damping, 4.8, 19.0))
        
        lookahead_steps = max(1, int(round(L_d / ds)))
        idx_target = (idx_near + lookahead_steps) % self.num_points
        p_target = self.optimal_line_xy[idx_target]

        dx = float(p_target[0] - p_curr[0])
        dy = float(p_target[1] - p_curr[1])
        cos_th = float(np.cos(th_curr))
        sin_th = float(np.sin(th_curr))
        
        x_body = cos_th * dx + sin_th * dy
        y_body = -sin_th * dx + cos_th * dy
        actual_lookahead_sq = max(4.0, x_body ** 2 + y_body ** 2)
        
        # Công thức độ cong Pure Pursuit kèm hệ số bù góc trượt lốp động lực học (Understeer Gradient K_us * v^2):
        # δ = (1.10 + K_us * v^2) * arctan(L * κ), giúp xe không bị văng understeer ở các góc cua tốc độ cao (Copse, Maggots, Stowe)
        v_curr_ms = max(0.1, v_curr_kmh / 3.6)
        curvature = 2.0 * y_body / actual_lookahead_sq
        slip_gain = float(np.clip(1.10 + 0.00055 * (v_curr_ms ** 2), 1.12, 1.50))
        steer_rad = slip_gain * np.arctan(curvature * wheelbase)
        steer_norm = float(np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0))

        # 4. Điều khiển dọc: Bám trực tiếp Profile Lipp & Boyd Pass 2 qua Longitudinal MPC
        # (Bỏ quét phanh kép dư thừa và bỏ lat_speed_scale gây sụt tốc độ mục tiêu đột ngột)
        v_target_kmh = float(self.optimal_speed_kmh[idx_near])
        v_target_ms = v_target_kmh / 3.6
        v_err_kmh = v_target_kmh - v_curr_kmh

        # Nhận diện Vùng Phanh Tiến Vào Đỉnh Cua (Monotonic Braking-Zone Latch):
        # Khi profile phía trước đang giảm tốc về đáy Apex (v_ahead_8 < v_target_kmh) và xe vẫn đang nhanh hơn
        # tốc độ đáy cua phía trước (v_curr_kmh > v_min_ahead + 2.0), khóa trần gia tốc a_opt <= 0.0
        # để xe chỉ nhả phanh trôi mượt vào Apex chứ TUYỆT ĐỐI KHÔNG nhồi ga giữa vùng phanh!
        ahead_scan_idx = (idx_near + np.arange(1, 36)) % self.num_points
        v_min_ahead_kmh = float(np.min(self.optimal_speed_kmh[ahead_scan_idx]))
        v_ahead_8_kmh = float(self.optimal_speed_kmh[(idx_near + 8) % self.num_points])
        in_braking_zone = bool((v_ahead_8_kmh < v_target_kmh - 0.4) and (v_curr_kmh > v_min_ahead_kmh + 2.0))

        # Trail-Braking & Traction Control qua Elip ma sát Kamm (Friction Circle):
        steer_mag = abs(steer_norm)
        avail_long_grip = float(np.sqrt(max(0.12, 1.0 - min(0.88, (0.88 * steer_mag) ** 2))))
        if v_curr_kmh < 25.0 or is_grid_launch:
            traction_factor = 1.0
        else:
            traction_factor = float(np.clip(1.0 - 0.50 * (steer_mag ** 1.5), 0.45, 1.0))

        # Lực ga cân bằng tĩnh (Equilibrium Maintenance Throttle tại a_x = 0):
        # Trên đường thẳng (steer_mag = 0), lực cản lăn của Bullet rất nhỏ (~0.12 m/s^2) nên chỉ cần 7% ga.
        # Khi ôm cua (steer_mag > 0), lực cản trượt ngang của lốp tăng theo góc lái nên cần 7% -> 43% ga.
        maint_thr = float(0.07 + 0.36 * (steer_mag ** 1.2))

        if is_grid_launch:
            target_brake = 0.0
            target_throttle = 1.0
            self.last_a_opt = 3.5
        elif self.use_mpc:
            indices_fut = []
            idx_cursor = idx_near
            for k in range(self.mpc.N):
                v_est = max(4.0, float(self.optimal_speed_kmh[idx_cursor] / 3.6))
                step_adv = max(1, int(round((v_est * self.mpc.dt) / ds)))
                idx_cursor = (idx_cursor + step_adv) % self.num_points
                indices_fut.append(idx_cursor)

            target_v_ms = (self.optimal_speed_kmh[indices_fut] / 3.6).copy()
            target_kappa = self.optimal_line_kappa[indices_fut]

            a_opt = self.mpc.solve(v_curr_ms, target_v_ms, target_kappa, self.last_a_opt, steer_norm)
            if in_braking_zone and a_opt > 0.0:
                a_opt = 0.0
            self.last_a_opt = a_opt

            # Vùng điều tiết bướm ga / nhả ga trôi tự nhiên (Coasting / Throttle Feathering Band = 0.38 m/s^2):
            # - Khi a_opt >= 0: mở ga mượt từ maint_thr lên 1.0 (nếu đang trong vùng phanh thì khóa tại maint_thr).
            # - Khi -0.38 <= a_opt < 0: nhả bướm ga mượt từ maint_thr về 0.0, không nhấp phanh.
            # - Khi a_opt < -0.38: đóng ga 0.0 và chia theo đúng hiệu suất phanh cực đại thực tế của Bullet (12.0 m/s^2)
            #   để lực phanh êm, đúng điểm và không bao giờ bị hãm quá đà!
            coast_band = 0.38
            full_brake_decel = 12.0
            is_top_speed_cruise = bool(v_target_kmh >= 118.5 and v_curr_kmh <= 121.5)
            if a_opt >= 0.0:
                thr_factor = float(np.clip(a_opt / self.mpc.max_engine, 0.0, 1.0))
                target_throttle = float(maint_thr + thr_factor * (1.0 - maint_thr)) * traction_factor
                target_brake = 0.0
            else:
                overspeed_acc = -a_opt
                if overspeed_acc <= coast_band or is_top_speed_cruise:
                    lift_ratio = float(np.clip(overspeed_acc / coast_band, 0.0, 1.0))
                    target_throttle = float(maint_thr * (1.0 - lift_ratio)) * traction_factor
                    target_brake = 0.0
                else:
                    target_throttle = 0.0
                    brk_norm = float(np.clip((overspeed_acc - coast_band) / (full_brake_decel - coast_band), 0.0, 1.0))
                    target_brake = float(brk_norm * avail_long_grip * self.max_brk_pedal)
        elif v_err_kmh >= -1.5:
            target_brake = 0.0
            thr_factor = float(np.clip(v_err_kmh / 4.5, -1.0, 1.0))
            if thr_factor >= 0.0:
                target_throttle = float(maint_thr + thr_factor * (1.0 - maint_thr)) * traction_factor
            else:
                target_throttle = float(maint_thr * (1.0 + thr_factor)) * traction_factor
        else:
            target_throttle = 0.0
            excess_speed = -v_err_kmh - 1.5
            brk_fraction = float(np.clip(excess_speed / 4.5, 0.0, 1.0))
            target_brake = float(brk_fraction * self.max_brk_pedal * avail_long_grip)

        alpha_thr = 0.88
        alpha_brk = 0.88
        self.filtered_throttle = float((1.0 - alpha_thr) * self.filtered_throttle + alpha_thr * target_throttle)
        if target_brake == 0.0 and self.filtered_brake < 0.06:
            self.filtered_brake = 0.0
        else:
            self.filtered_brake = float((1.0 - alpha_brk) * self.filtered_brake + alpha_brk * target_brake)

        throttle_action = float(np.clip(self.filtered_throttle * 2.0 - 1.0, -1.0, 1.0))
        brake_action = float(np.clip(self.filtered_brake * 2.0 - 1.0, -1.0, 1.0))

        action = np.array([steer_norm, throttle_action, brake_action], dtype=np.float32)
        net_pedal = float(self.filtered_throttle - self.filtered_brake)
        return action, steer_norm, net_pedal

