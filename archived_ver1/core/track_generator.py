"""
core/track_generator.py: Bộ sinh và nạp các đường đua F1 thật và đường đua ngẫu nhiên.
Hỗ trợ: Monza, Spa, Suzuka, Silverstone, và Procedural Circuits.
"""

import numpy as np
import os

class TrackGenerator:
    """
    Bộ sinh các đường đua F1 thật và đường đua ngẫu nhiên.
    Đường đua được biểu diễn bằng chuỗi waypoints khép kín (centerline, left_border, right_border).
    """
    @staticmethod
    def generate_monza(num_points=400, track_width=16.0):
        """Mô phỏng đường đua Monza (Italy): Đoạn thẳng chính dài xuất phát và 3 cụm chicane."""
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + 0.534
        x = 220 * np.cos(t) + 16 * np.cos(3 * t)
        y = 90 * np.sin(t) - 14 * np.sin(2 * t)
        return TrackGenerator._build_borders(x, y, track_width, "Monza Autodromo")

    @staticmethod
    def generate_spa(num_points=500, track_width=18.0):
        """Mô phỏng đường đua Spa-Francorchamps (Bỉ): Khúc cua liên hoàn phức tạp."""
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = 200 * np.cos(t) + 50 * np.sin(2 * t) + 10 * np.cos(3 * t)
        y = 120 * np.sin(t) + 40 * np.cos(2 * t)
        return TrackGenerator._build_borders(x, y, track_width, "Spa-Francorchamps")

    @staticmethod
    def generate_suzuka(num_points=450, track_width=16.0):
        """Mô phỏng đường đua Suzuka (Nhật Bản): Đường đua hình số 8 kỹ thuật."""
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = 150 * np.sin(t)
        y = 90 * np.sin(2 * t) + 15 * np.cos(t)
        return TrackGenerator._build_borders(x, y, track_width, "Suzuka Circuit")

    @staticmethod
    def generate_silverstone(track_width=16.0):
        """Mô phỏng đường đua Silverstone Grand Prix Circuit (Vương quốc Anh)."""
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "data", "silverstone_center.npy"),
            os.path.join(os.path.dirname(__file__), "..", "silverstone_center.npy"),
            os.path.join(os.getcwd(), "data", "silverstone_center.npy"),
            os.path.join(os.getcwd(), "silverstone_center.npy"),
        ]
        coords = None
        for p in search_paths:
            if os.path.exists(p):
                coords = np.load(p)
                break
        if coords is None:
            raise FileNotFoundError("Không tìm thấy file tọa độ silverstone_center.npy trong dự án!")
        x, y = coords[:, 0], coords[:, 1]
        return TrackGenerator._build_borders(x, y, track_width, "Silverstone Circuit")

    @staticmethod
    def generate_random_circuit(num_points=350, track_width=16.0, seed=None):
        """Sinh đường đua ngẫu nhiên để huấn luyện agent có tính khái quát hóa."""
        if seed is not None:
            np.random.seed(seed)
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radius = 120 + 25 * np.sin(2 * t + np.random.uniform(0, np.pi)) \
                     + 20 * np.cos(3 * t + np.random.uniform(0, np.pi)) \
                     + 12 * np.sin(5 * t)
        x = radius * np.cos(t)
        y = radius * np.sin(t) * 0.75
        return TrackGenerator._build_borders(x, y, track_width, "Procedural Circuit")

    @staticmethod
    def _build_borders(x, y, width, name):
        """Từ tim đường (x, y), dựng mép đường đua trái/phải vuông góc với thích ứng bán kính cua hẹp."""
        center = np.column_stack([x, y])
        n = len(center)
        left = np.zeros_like(center)
        right = np.zeros_like(center)
        
        diffs = np.diff(np.vstack([center, center[0]]), axis=0)
        ds = np.maximum(np.linalg.norm(diffs, axis=1), 1e-4)
        thetas = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
        d_theta = np.diff(np.concatenate([thetas, [thetas[0]]]))
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        signed_k = d_theta / ds
        
        w_half = width / 2.0
        for i in range(n):
            prev_p = center[(i - 1) % n]
            next_p = center[(i + 1) % n]
            tangent = next_p - prev_p
            tangent_norm = tangent / (np.linalg.norm(tangent) + 1e-8)
            # Vector pháp tuyến vuông góc với hướng tiếp tuyến (hướng sang trái)
            normal = np.array([-tangent_norm[1], tangent_norm[0]])
            
            k = signed_k[i]
            r = 1.0 / (abs(k) + 1e-5)
            # Nếu cua trái (k > 0): mép trái là phía trong, giới hạn an toàn để không giao cắt tâm cua
            # Nếu cua phải (k < 0): mép phải là phía trong
            w_left = min(w_half, max(4.0, r * 0.80)) if k > 0 else w_half
            w_right = min(w_half, max(4.0, r * 0.80)) if k < 0 else w_half
            
            left[i] = center[i] + normal * w_left
            right[i] = center[i] - normal * w_right
            
        return {
            "name": name,
            "center": center,
            "left": left,
            "right": right,
            "width": width,
            "length": float(np.sum(np.linalg.norm(np.diff(np.vstack([center, center[0]]), axis=0), axis=1)))
        }
