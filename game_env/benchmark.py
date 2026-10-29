"""
Hệ Thống Đánh Giá Học Thuật & Động Lực Học Đua Xe (Racing Benchmark Engine)
Áp dụng Lập trình Hướng đối tượng (OOP) & Tối ưu hóa lồi (Convex Optimization):
1. Tính toán Racing Line tối ưu hình học (Minimum Curvature Line qua Sparse QP + C2 Periodic CubicSpline)
   với biên độ mở rộng sát Track Limit (+-3.20m, >58% đường đua mở rộng trên 2.5m) chuẩn Out-In-Out.
2. Tính toán Profile Tốc độ Tới hạn thời gian Đơn điệu Từng Pha (Monotonic Lipp & Boyd 3-Pass Velocity Profile)
   được hiệu chuẩn khớp chính xác với động lực học vật lý Bullet nguyên bản của thư viện MetaDrive.
3. Cung cấp bộ đo đạc định lượng chuẩn xác: MAE/RMSE lệch Racing line, Apex Accuracy, Delta-V góc cua, G-G Friction Utilization.
4. Tương thích hoàn toàn với TelemetryTracker và quy chuẩn học thuật REL (Không gây Data Leak).
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from game_env.vehicle import VehicleConfig, GT3RaceCar


class RacingBenchmarkEngine:
    """
    Động cơ phân tích chuẩn mực (Ground Truth Benchmark Engine)
    Giải các bài toán vật lý & hình học để thiết lập cận trên/dưới lý thuyết cho xe đua trên track.
    """
    def __init__(
        self,
        track_xy: np.ndarray,
        track_theta: np.ndarray,
        track_kappa: np.ndarray,
        track_s: np.ndarray,
        vehicle_cfg: Optional[VehicleConfig] = None,
        track_width: float = 9.0,
        preset_name: Optional[str] = None,
        force_recompute: bool = False,
    ):
        self.track_xy = np.array(track_xy, dtype=np.float32)
        self.track_theta = np.array(track_theta, dtype=np.float32)
        self.track_kappa = np.array(track_kappa, dtype=np.float32)
        self.track_s = np.array(track_s, dtype=np.float32)
        self.num_points = len(track_xy)
        self.circuit_length = float(track_s[-1]) if len(track_s) > 0 else 2838.7
        self.track_width = float(track_width)
        self.preset_name = preset_name
        self.scale = float(self.circuit_length / 2851.8) if self.circuit_length > 3000.0 else 1.0
        
        self.cfg = vehicle_cfg if vehicle_cfg is not None else GT3RaceCar().cfg
        self.mu = float(self.cfg.wheel_friction)
        # v22: hệ số khớp tay legacy (map nhỏ, T* = 112.6s). v23: envelope đo trên Bullet (WI-11)
        self.profile_version = 23.0 if getattr(self.cfg, "envelope_a_lat", None) is not None else 22.0
        self.envelope = np.array([float(getattr(self.cfg, k, None) or 0.0) for k in ("envelope_a_lat", "envelope_a_brake", "envelope_a_engine")])
        self.g = 9.81
        self.v_max_m_s = float(self.cfg.max_speed_km_h / 3.6)
        
        # Đặt biên độ lệch tâm tối đa của Racing Line (chuẩn Out-In-Out sát mép nhựa, trừ 0.45m đệm an toàn)
        if self.circuit_length > 4000.0 or (self.preset_name and "full" in self.preset_name):
            self.max_lat_bound = min(6.75, max(1.0, 0.5 * self.track_width - 0.45))
        else:
            self.max_lat_bound = min(4.05, max(1.0, 0.5 * self.track_width - 0.45))
        
        # Đường dẫn lưu cache phân tách theo sa bàn và loại xe (WI-L4)
        if self.preset_name is not None:
            map_tag = self.preset_name
        else:
            map_tag = "silverstone_full" if self.circuit_length > 4000.0 else "silverstone_half"
        veh_tag = getattr(self.cfg, "name", "gt3").lower()
        if map_tag == "silverstone_half" and ("real" not in veh_tag):
            cache_file = "silverstone_optimal_profile.npz"
        else:
            cache_file = f"{map_tag}_{veh_tag}_optimal_profile.npz"
        cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", cache_file)
        loaded_from_cache = False
        if not force_recompute and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                if (
                    data["optimal_line_xy"].shape[0] == self.num_points
                    and abs(float(data["mu"]) - self.mu) < 1e-3
                    and np.linalg.norm(data["optimal_line_xy"][0] - self.track_xy[0]) < 4.5
                    and float(data.get("profile_version", 0)) == self.profile_version
                    and np.allclose(data["envelope"] if "envelope" in data.files else np.zeros(3), self.envelope)
                ):
                    self.optimal_line_xy = data["optimal_line_xy"]
                    self.optimal_line_kappa = data["optimal_line_kappa"]
                    self.optimal_speed_profile = data["optimal_speed_profile"]
                    self.optimal_speed_kmh = data["optimal_speed_kmh"]
                    self.theoretical_best_lap_time = float(data["theoretical_best_lap_time"])
                    loaded_from_cache = True
            except Exception:
                pass
                
        if not loaded_from_cache:
            # 1. Tính toán Racing Line tối ưu (Minimum Curvature QP + C2 Periodic Spline mở rộng +-4.05m)
            self.optimal_line_xy, self.optimal_line_kappa = self._compute_optimal_racing_line()
            
            # 2. Tính toán Profile Vận tốc tới hạn đơn điệu (Monotonic Lipp & Boyd 3-Pass)
            self.optimal_speed_profile = self._compute_optimal_speed_profile()
            self.optimal_speed_kmh = self.optimal_speed_profile * 3.6
            
            # 3. Tính Cận dưới Thời gian vòng đua lý thuyết (Theoretical Best Lap Time)
            diff_s = np.linalg.norm(np.diff(self.optimal_line_xy, axis=0), axis=1)
            self.diff_s = np.concatenate([diff_s, [diff_s[0]]]).astype(np.float32)
            self.theoretical_best_lap_time = float(np.sum(self.diff_s / np.maximum(self.optimal_speed_profile, 1.0)))
            
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    optimal_line_xy=self.optimal_line_xy,
                    optimal_line_kappa=self.optimal_line_kappa,
                    optimal_speed_profile=self.optimal_speed_profile,
                    optimal_speed_kmh=self.optimal_speed_kmh,
                    theoretical_best_lap_time=self.theoretical_best_lap_time,
                    mu=self.mu,
                    profile_version=self.profile_version,
                    envelope=self.envelope,
                )
            except Exception:
                pass

        # 4. Trích xuất danh sách các góc cua chính (Apexes)
        self.corner_apexes = self._detect_corner_apexes()

    def _compute_optimal_racing_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Giải bài toán Convex Quadratic Programming (Heilmeier et al. 2020):
        - Tối thiểu hóa bình phương độ cong toàn cục 2D Cartesian trên không gian ma trận thưa (Sparse QP)
          với hệ số chính quy hóa tâm siêu nhỏ (w_center = 2.5e-9, tức sqrt(w_center) = 5e-5 < (pi/L_straight)^2)
          và biên độ +-4.05m (2 bánh ngoài đè qua vạch trắng rìa đường ~0.48m chuẩn F1/GT3).
        - Giải phóng hoàn toàn quỹ đạo trên các đoạn thẳng dài (Hamilton, Wellington, National Pit, Hangar, Vale)
          để xe mở rộng kịch biên +-4.05m tại cả điểm vào cua (Entry), đỉnh cua (Apex) và lối thoát (Exit)
          mà vẫn giảm 35%-50% độ cong hiệu dụng trong góc cua.
        """
        N = self.num_points
        if N < 10:
            return self.track_xy.copy(), self.track_kappa.copy()

        ds = self.circuit_length / max(1, N)

        # Lấy mẫu thưa ~3.0m để giải nghiệm QP tuần hoàn
        step_sub = max(1, int(round(3.0 / ds)))
        idx_sub = np.arange(0, N, step_sub)
        N_sub = len(idx_sub)
        ds_sub = ds * step_sub
        xy_sub = self.track_xy[idx_sub]
        th_sub = self.track_theta[idx_sub]

        x0 = xy_sub[:, 0].astype(np.float64)
        y0 = xy_sub[:, 1].astype(np.float64)
        nx_sub = -np.sin(th_sub).astype(np.float64)
        ny_sub = np.cos(th_sub).astype(np.float64)

        import scipy.sparse as sp
        row, col, val = [], [], []
        for i in range(N_sub):
            row.extend([i, i, i])
            col.extend([i, (i + 1) % N_sub, (i - 1) % N_sub])
            val.extend([-2.0 / (ds_sub ** 2), 1.0 / (ds_sub ** 2), 1.0 / (ds_sub ** 2)])

        D2 = sp.csr_matrix((val, (row, col)), shape=(N_sub, N_sub))

        x0_ddot = D2.dot(x0)
        y0_ddot = D2.dot(y0)

        D2_Nx = D2.dot(sp.diags(nx_sub))
        D2_Ny = D2.dot(sp.diags(ny_sub))

        # Hệ số sqrt(w_center) = 5e-5 / (scale^2) nhỏ hơn trị riêng (pi/L)^2 của các đoạn thẳng,
        # giúp Racing Line dạt kịch biên ở đầu/cuối mọi đoạn thẳng thay vì bị kéo về tim đường.
        w_center_sqrt = 5e-5 / (self.scale ** 2)
        I = sp.eye(N_sub)

        A_qp = sp.vstack([D2_Nx, D2_Ny, w_center_sqrt * I])
        b_qp = np.concatenate([-x0_ddot, -y0_ddot, np.zeros(N_sub)])

        bounds = (-self.max_lat_bound, self.max_lat_bound)
        try:
            res = lsq_linear(
                A_qp,
                b_qp,
                bounds=bounds,
                method="trf",
                lsmr_tol="auto",
                max_iter=2000,
                tol=1e-6,
                verbose=0,
            )
            alpha_sub = np.asarray(res.x, dtype=np.float64) if res.x is not None else np.zeros(N_sub, dtype=np.float64)
        except Exception:
            alpha_sub = np.zeros(N_sub, dtype=np.float64)

        # Nội suy bằng Periodic CubicSpline C2 và làm mịn nhẹ theo scale để giữ nguyên đỉnh Out-In-Out
        idx_ext = np.append(idx_sub, N).astype(np.float64)
        alpha_ext = np.append(alpha_sub, alpha_sub[0])
        cs = CubicSpline(idx_ext, alpha_ext, bc_type="periodic")
        alpha_full_raw = cs(np.arange(N, dtype=np.float64))
        alpha_smooth = np.asarray(
            gaussian_filter1d(alpha_full_raw, sigma=1.8 * self.scale, mode="wrap"),
            dtype=np.float64,
        )
        alpha_full = np.clip(
            alpha_smooth,
            -self.max_lat_bound,
            self.max_lat_bound,
        ).astype(np.float32)

        nx_full = -np.sin(self.track_theta)
        ny_full = np.cos(self.track_theta)
        opt_x = self.track_xy[:, 0] + nx_full * alpha_full
        opt_y = self.track_xy[:, 1] + ny_full * alpha_full
        opt_xy = np.column_stack([opt_x, opt_y]).astype(np.float32)

        dx = np.gradient(opt_x)
        dy = np.gradient(opt_y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        raw_kappa = (dx * ddy - dy * ddx) / np.maximum(1e-5, (dx ** 2 + dy ** 2) ** 1.5)
        opt_kappa = np.asarray(
            gaussian_filter1d(raw_kappa.astype(np.float64), sigma=2.5 * self.scale, mode="wrap"),
            dtype=np.float32,
        )

        return opt_xy, opt_kappa

    def _compute_optimal_speed_profile(self) -> np.ndarray:
        """
        Thuật toán Lipp & Boyd 3-Pass Time-Optimal Velocity Profile Đơn điệu Từng Pha (Monotonic Corner Phases):
        1. Pass 1: Tính vận tốc giới hạn ngang v_crit(s) trực tiếp từ độ cong trơn C2 opt_kappa(s)
           (không làm bẹt đáy cua) để xe lập tức mở ga thoát cua ngay khi qua đỉnh Apex.
        2. Pass 2 (Backward Braking): Tích phân lùi với gia tốc phanh hiệu chỉnh.
        3. Pass 3 (Forward Acceleration): Tích phân tiến với gia tốc kéo hiệu chỉnh.
        """
        N = self.num_points
        if N < 5:
            return np.full(N, self.v_max_m_s, dtype=np.float32)

        diff_s = np.linalg.norm(np.diff(self.optimal_line_xy, axis=0), axis=1)
        ds_opt = np.concatenate([diff_s, [diff_s[0]]]).astype(np.float32)

        # Gia tốc kéo/phanh: envelope đo trên Bullet nếu xe có (WI-11), ngược lại dùng hệ số legacy v22
        envelope_lat = getattr(self.cfg, "envelope_a_lat", None)
        a_engine = float(getattr(self.cfg, "envelope_a_engine", None) or 3.40)
        a_brake = float(getattr(self.cfg, "envelope_a_brake", None) or 6.20)

        scale = self.scale
        dk_profile = np.zeros(N, dtype=np.float32)
        k_env = np.zeros(N, dtype=np.float32)
        k_abs_arr = np.abs(self.optimal_line_kappa)
        win_chicane = int(round(12 * scale))
        idx_back = int(round(-4 * scale))
        idx_ahead = int(round(23 * scale))
        for i in range(N):
            dk_profile[i] = abs(float(self.optimal_line_kappa[(i + 2) % N] - self.optimal_line_kappa[(i - 2) % N])) / (4.0 * max(0.5, ds_opt[i]))
            # Bao hình độ cong bất đối xứng "Slow In, Fast Out" (quét trước vào cua, chỉ giữ sau đỉnh cua):
            local_indices = (i + np.arange(idx_back, idx_ahead)) % N
            chicane_indices = (i + np.arange(-win_chicane, win_chicane + 1)) % N
            k_chicane_win = self.optimal_line_kappa[chicane_indices]
            is_inflection = bool(np.min(k_chicane_win) < (-0.004 / scale) and np.max(k_chicane_win) > (0.004 / scale))
            slew_k = (17.5 * scale * float(dk_profile[i])) if is_inflection else 0.0
            k_env[i] = max(
                float(k_abs_arr[i]),
                0.88 * float(np.max(k_abs_arr[local_indices])),
                slew_k,
            )

        def get_a_lat_limit(k_val: float, dk_val: float) -> float:
            if envelope_lat is not None:
                return float(envelope_lat)  # Trần bám ngang đo được, không phụ thuộc loại cua
            if k_val > (0.0170 / scale) or dk_val > (0.00055 / (scale ** 2)):
                return 8.85   # Đỉnh cua gắt & Chicane đổi hướng
            elif k_val > (0.0090 / scale) or dk_val > (0.00035 / (scale ** 2)):
                return 9.10   # Cua trung bình-nhanh 90°
            elif k_val > (0.0065 / scale):
                return 10.20  # Cua nhanh
            else:
                return 11.80  # Đoạn cong thoai thoải tốc độ cao (Aintree / Hangar kink): ~1.20G

        def kamm_ratio(a_lat_curr: float, a_lat_lim: float) -> float:
            # Phần bám dọc còn lại theo elip Kamm. Envelope đo được: elip chặt (sàn 0.05), vì phanh/ga khi đã dùng
            # hết bám ngang làm xe xoay. Legacy v22: giữ sàn 0.15 và trần 0.92 như cũ.
            if envelope_lat is not None:
                return float(np.sqrt(max(0.05, 1.0 - min(0.9975, (a_lat_curr / a_lat_lim) ** 2))))
            return float(np.sqrt(max(0.15, 1.0 - min(0.92, (a_lat_curr / a_lat_lim) ** 2))))

        # Pass 1: Vận tốc tới hạn góc cua từ độ cong C2 opt_kappa
        v_prof = np.full(N, self.v_max_m_s, dtype=np.float32)
        for i in range(N):
            k_eff = float(k_env[i])
            dk = float(dk_profile[i])
            if k_eff > 1e-4 or dk > 1e-4:
                a_lat_eff = get_a_lat_limit(k_eff, dk)
                v_crit = np.sqrt(a_lat_eff / max(1e-4, k_eff))
                v_prof[i] = min(v_prof[i], v_crit)

        # Pass 2: Quét lùi (Backward integration - Braking với elip ma sát Kamm)
        for _ in range(4):
            for i in range(N - 1, -1, -1):
                nxt = (i + 1) % N
                d_s = ds_opt[i]
                k_curr = float(k_env[i])
                dk_curr = float(dk_profile[i])
                a_lat_lim = get_a_lat_limit(k_curr, dk_curr)
                a_lat_curr = (v_prof[i] ** 2) * k_curr
                avail_ratio = kamm_ratio(a_lat_curr, a_lat_lim)
                v_brk = np.sqrt(v_prof[nxt] ** 2 + 2.0 * (a_brake * avail_ratio) * d_s)
                v_prof[i] = min(v_prof[i], v_brk)

        # Pass 3: Quét tiến (Forward integration - Acceleration với elip ma sát Kamm)
        for _ in range(4):
            for i in range(N):
                nxt = (i + 1) % N
                d_s = ds_opt[i]
                k_curr = float(k_env[i])
                dk_curr = float(dk_profile[i])
                a_lat_lim = get_a_lat_limit(k_curr, dk_curr)
                a_lat_curr = (v_prof[i] ** 2) * k_curr
                avail_ratio = kamm_ratio(a_lat_curr, a_lat_lim)
                v_acc = np.sqrt(v_prof[i] ** 2 + 2.0 * (a_engine * avail_ratio) * d_s)
                v_prof[nxt] = min(v_prof[nxt], v_acc)

        # Lọc làm mịn Gaussian tuần hoàn (sigma=2.0) để profile vận tốc khả vi trơn tru cho MPC
        v_smooth = np.asarray(
            gaussian_filter1d(v_prof.astype(np.float64), sigma=2.0, mode="wrap"),
            dtype=np.float64,
        )

        return np.clip(v_smooth, 12.0, self.v_max_m_s).astype(np.float32)

    def _detect_corner_apexes(self) -> List[Dict[str, Any]]:
        """
        Nhận diện các đỉnh cua chính (Corner Apexes) trên đường đua.
        Áp dụng thuật toán Non-Maximum Suppression (NMS) với cự ly tối thiểu giữa các góc cua (min_dist = 45.0m)
        để loại bỏ gợn sóng nhiễu của spline và gom về đúng các góc cua thực tế của trường đua.
        """
        raw_candidates = []
        N = self.num_points
        k_abs = np.abs(self.optimal_line_kappa)
        
        for i in range(2, N - 2):
            if k_abs[i] > (0.010 / self.scale):
                if k_abs[i] >= k_abs[i - 1] and k_abs[i] >= k_abs[i + 1]:
                    raw_candidates.append({
                        "idx": i,
                        "s": float(self.track_s[i]),
                        "pos": self.optimal_line_xy[i],
                        "kappa": float(k_abs[i]),
                        "target_speed_kmh": float(self.optimal_speed_kmh[i]),
                    })

        if not raw_candidates:
            return []

        raw_candidates.sort(key=lambda x: x["kappa"], reverse=True)
        selected_apexes = []
        min_separation_m = 45.0 * self.scale

        for cand in raw_candidates:
            too_close = False
            for sel in selected_apexes:
                dist_s = abs(cand["s"] - sel["s"])
                circular_diff = min(dist_s, self.circuit_length - dist_s)
                if circular_diff < min_separation_m:
                    too_close = True
                    break
            if not too_close:
                selected_apexes.append(cand)

        selected_apexes.sort(key=lambda x: x["s"])
        return selected_apexes

    def evaluate_lap(self, lap_samples: List[Dict]) -> Dict[str, float]:
        """
        Đánh giá toàn diện một vòng chạy của Agent dựa trên dữ liệu Telemetry:
        Trả về đầy đủ các chỉ số khoa học: MAE, RMSE, Apex Accuracy, Speed Delta, Friction %.
        """
        if not lap_samples or len(lap_samples) < 5:
            return {
                "racing_line_mae_m": 0.0,
                "racing_line_rmse_m": 0.0,
                "apex_cutting_accuracy_pct": 0.0,
                "mean_corner_delta_v_kmh": 0.0,
                "friction_circle_utilization_pct": 0.0,
                "lap_efficiency_pct": 0.0,
                "theoretical_best_lap_time": float(self.theoretical_best_lap_time),
            }
            
        ego_xy = np.array([[s.get("pos_x", 0.0), s.get("pos_y", 0.0)] for s in lap_samples], dtype=np.float32)
        ego_speeds = np.array([s.get("speed_km_h", 0.0) for s in lap_samples], dtype=np.float32)
        ego_dists = np.array([s.get("distance", 0.0) for s in lap_samples], dtype=np.float32)
        
        dists_to_line = []
        for p in ego_xy:
            diffs = self.optimal_line_xy - p
            d_sq = np.sum(diffs ** 2, axis=1)
            dists_to_line.append(float(np.sqrt(np.min(d_sq))))
        dists_to_line = np.array(dists_to_line, dtype=np.float32)
        mae_line = float(np.mean(dists_to_line))
        rmse_line = float(np.sqrt(np.mean(dists_to_line ** 2)))
        
        corner_speed_deltas = []
        apex_clipped_count = 0
        for apex in self.corner_apexes:
            dists_to_apex = np.linalg.norm(ego_xy - apex["pos"], axis=1)
            min_dist_apex = float(np.min(dists_to_apex))
            if min_dist_apex <= 15.0:
                idx_best = int(np.argmin(dists_to_apex))
                v_actual = float(ego_speeds[idx_best])
                v_target = float(apex["target_speed_kmh"])
                corner_speed_deltas.append(v_actual - v_target)
                
                if min_dist_apex <= 2.20:
                    apex_clipped_count += 1
                    
        mean_delta_v = float(np.mean(corner_speed_deltas)) if corner_speed_deltas else 0.0
        apex_acc_pct = float(apex_clipped_count / max(1, len(self.corner_apexes))) * 100.0
        
        g_longs = np.array([s.get("acc_long_g", 0.0) for s in lap_samples], dtype=np.float32)
        g_lats = np.array([s.get("acc_lat_g", 0.0) for s in lap_samples], dtype=np.float32)
        g_totals = np.sqrt(g_longs ** 2 + g_lats ** 2)
        speed_mask = ego_speeds > 40.0
        if np.any(speed_mask):
            mean_g = float(np.mean(g_totals[speed_mask]))
            friction_util_pct = float(np.clip(mean_g / self.mu, 0.0, 1.2) * 100.0)
        else:
            friction_util_pct = 0.0
            
        actual_lap_time = float(lap_samples[-1].get("time", 120.0))
        lap_eff_pct = float(np.clip(self.theoretical_best_lap_time / max(1e-2, actual_lap_time), 0.0, 1.2) * 100.0)
        
        return {
            "racing_line_mae_m": mae_line,
            "racing_line_rmse_m": rmse_line,
            "apex_cutting_accuracy_pct": apex_acc_pct,
            "mean_corner_delta_v_kmh": mean_delta_v,
            "friction_circle_utilization_pct": friction_util_pct,
            "lap_efficiency_pct": lap_eff_pct,
            "theoretical_best_lap_time": float(self.theoretical_best_lap_time),
        }

