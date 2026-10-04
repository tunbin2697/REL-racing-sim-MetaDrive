"""
policies/base_policy.py: Lớp trừu tượng cơ sở (Base Class) cho mọi Chính sách Đua xe F1.
"""

from abc import ABC, abstractmethod
import numpy as np

class BasePolicy(ABC):
    """
    Lớp trừu tượng đại diện cho một Policy (Chính sách quyết định) trong bài toán Đua xe F1.
    Một Policy nhận trạng thái quan sát state và trả về hành động action = [steer, throttle/brake].
    """
    def __init__(self, name="BasePolicy", description="", color="#3b82f6"):
        self.name = name
        self.description = description
        self.color = color # Mã màu HEX phục vụ trực quan hóa biểu đồ và web

    @abstractmethod
    def reset(self, track_info=None):
        """Khởi tạo lại trạng thái nội tại của Policy trước mỗi vòng đua."""
        pass

    @abstractmethod
    def select_action(self, state, info=None):
        """
        Nhận vector trạng thái state hoặc info từ môi trường,
        trả về action = [steer_cmd, accel_cmd] trong khoảng [-1.0, 1.0].
        """
        pass

    def get_profiler_features(self):
        """Trả về 2 đặc trưng ngoại vi (Lookahead Curvature, Speed Delta) nếu có."""
        return 0.0, 0.0
