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
    def __init__(self, reference_lap_file: Optional[str] = "logs/reference_lap.json"):
        self.reference_lap_file = reference_lap_file
        self.ref_distances: np.ndarray = np.array([])
        self.ref_times: np.ndarray = np.array([])
        self.ref_speeds: np.ndarray = np.array([])
        self.ref_best_lap_time: float = float("inf")
        
        # Dữ liệu của vòng chạy hiện tại
        self.current_lap_data: List[Dict] = []
        self.cumulative_distance: float = 0.0
        self.last_position = None
        self.last_velocity = None
        
        # Load reference lap nếu có
        self.load_reference_lap()

    def reset(self):
        """Reset bộ ghi cho vòng chạy mới"""
        self.current_lap_data = []
        self.cumulative_distance = 0.0
        self.last_position = None
        self.last_velocity = None

    def load_reference_lap(self, filepath: Optional[str] = None):
        """Nạp dữ liệu vòng chạy mẫu (Ghost Lap) để so sánh delta thời gian thực"""
        path = filepath or self.reference_lap_file
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples = data.get("samples", [])
                if samples:
                    self.ref_distances = np.array([s["distance"] for s in samples], dtype=np.float32)
                    self.ref_times = np.array([s["time"] for s in samples], dtype=np.float32)
                    self.ref_speeds = np.array([s["speed_km_h"] for s in samples], dtype=np.float32)
                    self.ref_best_lap_time = data.get("lap_time", float(self.ref_times[-1]) if len(self.ref_times) > 0 else float("inf"))
            except Exception as e:
                print(f"[TelemetryTracker] Không thể nạp reference lap từ {path}: {e}")

    def save_as_reference_lap(self, filepath: Optional[str] = None):
        """Lưu vòng chạy hiện tại làm Reference Lap mới"""
        path = filepath or self.reference_lap_file
        if not self.current_lap_data:
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        final_time = self.current_lap_data[-1]["time"]
        data = {
            "lap_time": float(final_time),
            "total_distance": float(self.current_lap_data[-1]["distance"]),
            "samples": self.current_lap_data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        # Nạp lại ngay vào bộ nhớ
        self.load_reference_lap(path)

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
        
        # Tính Delta Time so với Reference Lap tại cùng cự ly s
        delta_time, delta_color = self.get_realtime_delta(self.cumulative_distance, current_time)
        
        sample = {
            "time": float(current_time),
            "distance": float(self.cumulative_distance),
            "speed_km_h": float(speed_km_h),
            "steering": float(steering),
            "throttle": float(throttle),
            "brake": float(brake),
            "acc_long_g": float(acc_long),
            "acc_lat_g": float(acc_lat),
            "delta_time": float(delta_time) if delta_time is not None else 0.0,
            "delta_color": delta_color,
            "on_runoff": bool(on_runoff),
        }
        
        self.current_lap_data.append(sample)
        return sample

    def get_realtime_delta(self, current_distance: float, current_time: float) -> Tuple[Optional[float], str]:
        """
        Tính chênh lệch thời gian thực (Delta Bar) so với Reference Lap tại vị trí s:
        Δt = t_now - t_ref(s)
        - Δt < 0: Xanh lá (Nhanh hơn kỷ lục)
        - Δt > 0: Đỏ (Chậm hơn kỷ lục)
        """
        if len(self.ref_distances) < 2:
            return 0.0, "gray"
            
        # Tìm mốc cự ly gần nhất trên reference lap
        idx = np.searchsorted(self.ref_distances, current_distance)
        if idx >= len(self.ref_distances):
            idx = len(self.ref_distances) - 1
            
        ref_time_at_s = float(self.ref_times[idx])
        delta = current_time - ref_time_at_s
        color = "green" if delta < 0 else "red"
        return delta, color
