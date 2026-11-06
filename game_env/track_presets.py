"""
Cấu hình hình học và chuẩn hóa quan sát theo sa bàn Silverstone bán phần (L ≈ 2838.7m):
- 'silverstone_half': 2 làn 9.0m, sa bàn ref_111s được train
- 'silverstone_half_3lane': 3 làn 13.5m, sa bàn demo multicar
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TrackPreset:
    name: str
    length_scale: float
    lane_num: int
    lane_width: float
    lookahead_m: List[float]
    lookahead_dx_norm: float
    lookahead_dy_norm: float
    kappa_norm: float
    v_norm: float
    vy_norm: float
    lat_norm: float
    progress_coef: float

    @property
    def center_lat(self) -> float:
        """Tọa độ y của tim đường (giữa các làn): 0.5 * (lane_num - 1) * lane_width"""
        return 0.5 * (self.lane_num - 1) * self.lane_width

    @property
    def half_width(self) -> float:
        """Khoảng cách từ tim đường tới mép nhựa: 0.5 * lane_num * lane_width"""
        return 0.5 * self.lane_num * self.lane_width

    @property
    def grid_x(self) -> float:
        """Tọa độ x của vạch xuất phát / cán đích trên Hamilton Straight"""
        return 60.0 * self.length_scale


TRACK_PRESETS: Dict[str, TrackPreset] = {
    "silverstone_half": TrackPreset(
        name="silverstone_half",
        length_scale=1.0,
        lane_num=2,
        lane_width=4.5,
        lookahead_m=[10.0, 20.0, 35.0, 50.0, 70.0],
        lookahead_dx_norm=70.0,
        lookahead_dy_norm=30.0,
        kappa_norm=50.0,
        v_norm=40.0,
        vy_norm=20.0,
        lat_norm=4.5,
        progress_coef=0.05,
    ),
    "silverstone_half_3lane": TrackPreset(
        name="silverstone_half_3lane",
        length_scale=1.0,
        lane_num=3,
        lane_width=4.5,
        lookahead_m=[10.0, 20.0, 35.0, 50.0, 70.0],
        lookahead_dx_norm=70.0,
        lookahead_dy_norm=30.0,
        kappa_norm=50.0,
        v_norm=40.0,
        vy_norm=20.0,
        lat_norm=6.75,     # half_width = 0.5 * 3 * 4.5 = 6.75m
        progress_coef=0.05,
    ),
}


def get_track_preset(name: str) -> TrackPreset:
    """Truy xuất TrackPreset theo tên"""
    if name not in TRACK_PRESETS:
        raise ValueError(f"Không có preset '{name}'. Chọn một trong: {', '.join(TRACK_PRESETS)}")
    return TRACK_PRESETS[name]
