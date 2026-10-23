"""
Hệ Thống Sim Racing Telemetry Chuyên Sâu cho MetaDrive:
- Đo đạc chi tiết từng mét đường: Vận tốc, Chân ga, Chân phanh, Góc lái, Gia tốc G dọc/ngang.
- Kênh Delta Time thời gian thực so sánh với Reference Lap (Ghost Lap / Personal Best).
- Cung cấp dữ liệu phục vụ trực tiếp hàm thưởng RL và xuất đồ thị học thuật.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np


class TelemetryTracker:
    def __init__(self, **kwargs):
        # Dữ liệu của vòng chạy hiện tại
        self.current_lap_data: List[Dict] = []
        self.cumulative_distance: float = 0.0
        self.last_position = None
        self.last_velocity = None
        
        # Bảng kỷ lục phân đoạn cá nhân đa quy mô (Personal Best Windows):
        # key = (window_meters, grid_key), value = thời gian tốt nhất (s)
        self.personal_best_windows: Dict[Tuple[int, int], float] = {}

    def reset(self):
        """Reset bộ ghi cho vòng chạy mới (giữ lại personal_best_windows để tiếp tục so sánh Self-Play)"""
        self.current_lap_data = []
        self.cumulative_distance = 0.0
        self.last_position = None
        self.last_velocity = None

    def log_step(
        self,
        current_time: float,
        position: np.ndarray,
        velocity: np.ndarray,
        steering: float,
        throttle_brake: float,
        dt: float = 0.02,
        on_runoff: bool = False,
    ) -> Dict:
        """
        Ghi nhận telemetry tại mỗi bước vật lý:
        Tính gia tốc dọc/ngang (G-G), quãng đường tích lũy, và delta time.
        """
        pos = np.array(position[:2], dtype=np.float32)
        vel = np.array(velocity[:2], dtype=np.float32)
        speed_m_s = float(np.linalg.norm(vel))
        speed_km_h = speed_m_s * 3.6
        
        # Tính khoảng cách tích lũy dọc đường đua
        if self.last_position is not None:
            step_dist = float(np.linalg.norm(pos - self.last_position))
            self.cumulative_distance += step_dist
            
        # Tính gia tốc dọc & ngang (G-G Acceleration)
        acc_long = 0.0
        acc_lat = 0.0
        if self.last_velocity is not None and dt > 0:
            acc_vec = (vel - self.last_velocity) / dt
            # Hướng xe chạy
            if speed_m_s > 0.5:
                forward_dir = vel / speed_m_s
                lateral_dir = np.array([-forward_dir[1], forward_dir[0]])
                acc_long = float(np.dot(acc_vec, forward_dir)) / 9.81  # Đơn vị G
                acc_lat = float(np.dot(acc_vec, lateral_dir)) / 9.81   # Đơn vị G
                
        self.last_position = pos
        self.last_velocity = vel
        
        # Tách lực phanh và ga
        throttle = float(max(0.0, throttle_brake))
        brake = float(max(0.0, -throttle_brake))
        
        # Tính Delta Time theo 3 Cửa sổ trượt đa quy mô: Ngắn (20m), Vừa (60m), Dài (150m)
        delta_20, color_20 = self.get_window_delta(self.cumulative_distance, current_time, window_meters=20.0)
        delta_60, color_60 = self.get_window_delta(self.cumulative_distance, current_time, window_meters=60.0)
        delta_150, color_150 = self.get_window_delta(self.cumulative_distance, current_time, window_meters=150.0)
        
        sample = {
            "time": float(current_time),
            "distance": float(self.cumulative_distance),
            "speed_km_h": float(speed_km_h),
            "steering": float(steering),
            "throttle": float(throttle),
            "brake": float(brake),
            "acc_long_g": float(acc_long),
            "acc_lat_g": float(acc_lat),
            "delta_time": float(delta_60),  # Đại diện chung
            "delta_short": float(delta_20),
            "delta_mid": float(delta_60),
            "delta_long": float(delta_150),
            "delta_color": color_60,
            "on_runoff": bool(on_runoff),
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "vel_x": float(vel[0]),
            "vel_y": float(vel[1]),
        }
        
        self.current_lap_data.append(sample)
        return sample

    def get_window_delta(self, current_distance: float, current_time: float, window_meters: float = 40.0) -> Tuple[float, str]:
        """
        Tính chênh lệch thời gian cục bộ theo cửa sổ trượt (Sliding Window Delta) so với
        Kỷ lục Tốt nhất của Chính Mình (Personal Best - SIL):
        - delta_win < 0: Nhanh hơn kỷ lục cá nhân cũ trong window_meters qua (Xanh lá - Phá kỷ lục phân đoạn)
        - delta_win > 0: Chậm hơn kỷ lục cá nhân (Đỏ)
        """
        if current_distance < window_meters:
            return 0.0, "gray"
            
        s_start = current_distance - window_meters
        s_end = current_distance
        grid_key = (int(round(window_meters)), int(round(s_end / 5.0) * 5))
        
        # 1. Thời gian của agent trên đoạn [s_start, s_end] từ current_lap_data
        t_agent_window = None
        for sample in reversed(self.current_lap_data):
            if sample["distance"] <= s_start:
                t_agent_window = current_time - sample["time"]
                break
                
        if t_agent_window is None or t_agent_window <= 0.01:
            return 0.0, "gray"
            
        # 2. So sánh với Kỷ lục cá nhân (Personal Best) của phân đoạn này
        t_pb = self.personal_best_windows.get(grid_key)
        if t_pb is None:
            # Khai phá phân đoạn mới chưa từng có kỷ lục -> Lưu thành tích làm chuẩn ban đầu
            self.personal_best_windows[grid_key] = t_agent_window
            return -0.05, "green"
            
        delta_window = t_agent_window - t_pb
        if delta_window < 0:
            # Phá kỷ lục cá nhân tại phân đoạn này: Cập nhật chuẩn mới (Monotonic ratcheting)
            self.personal_best_windows[grid_key] = t_agent_window
            color = "green"
        else:
            color = "red"
            
        return float(delta_window), color

    def evaluate_with_benchmark(self, benchmark_engine) -> Dict[str, float]:
        """
        Đánh giá vòng chạy của xe qua Động cơ phân tích chuẩn mực (RacingBenchmarkEngine):
        - Đo MAE/RMSE lệch Racing line tối ưu (Minimum Curvature).
        - Đo tỷ lệ cắt đỉnh cua Apex chuẩn xác.
        - Đo sai số tốc độ tại các góc cua (Corner Speed Delta).
        - Đo mức độ khai thác vòng tròn ma sát lốp (G-G Friction Utilization).
        - Đo hiệu suất thời gian so với cận dưới lý thuyết.
        """
        if benchmark_engine is None:
            return {
                "racing_line_mae_m": 0.0,
                "racing_line_rmse_m": 0.0,
                "apex_cutting_accuracy_pct": 0.0,
                "mean_corner_delta_v_kmh": 0.0,
                "friction_circle_utilization_pct": 0.0,
                "lap_efficiency_pct": 0.0,
                "theoretical_best_lap_time": 0.0,
            }
        return benchmark_engine.evaluate_lap(self.current_lap_data)
