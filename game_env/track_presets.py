"""
Cấu hình chuẩn hóa hình học và tham số môi trường theo sa bàn (Track Presets)
Hỗ trợ cả hai chế độ:
- 'silverstone_half' (mặc định / Phase 1): sa bàn thu nhỏ bán phần (2 làn 9.0m, L ≈ 2838.7m)
- 'silverstone_full' (Phase 2): sa bàn kích thước thật 1:1 (3 làn 14.4m, L ≈ 5891m, length_scale = 2.075)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


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
    finish_T0: float
    finish_tau: float
    time_scale: float
    logstd_max: List[float]
    opponent_range: float

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
        finish_T0=180.0,
        finish_tau=25.0,
        time_scale=1.0,
        logstd_max=[0.0, 0.0, 0.0],
        opponent_range=60.0,
    ),
    "silverstone_full": TrackPreset(
        name="silverstone_full",
        length_scale=2.075,
        lane_num=3,
        lane_width=4.8,
        lookahead_m=[15.0, 40.0, 80.0, 130.0, 200.0],
        lookahead_dx_norm=200.0,
        lookahead_dy_norm=80.0,
        kappa_norm=104.0,  # 50.0 * 2.075 ≈ 103.75 -> 104.0
        v_norm=80.0,
        vy_norm=40.0,
        lat_norm=7.2,      # half_width = 0.5 * 3 * 4.8 = 7.2
        progress_coef=0.0241,  # 0.05 * 2838.7 / 5891 ≈ 0.02409
        finish_T0=186.0,   # D14 v2: 180 x T_ref_full/T_ref_half (tim đường + trần bám đo được + v_max), không dùng profile
        finish_tau=26.0,   # 25 x 186/180
        time_scale=1.45,
        logstd_max=[0.0, -0.7, -0.7],  # D19: khống chế sigma ga/phanh <= 0.5
        opponent_range=124.5,          # 60.0 * 2.075 = 124.5
    ),
}


def get_track_preset(name: Optional[str] = None) -> TrackPreset:
    """
    Truy xuất TrackPreset tương ứng theo tên định danh.
    Hỗ trợ alias:
    - None, 'silverstone', 'silverstone_half', 'half' -> preset 'silverstone_half'
    - 'silverstone_full', 'full', '1:1' -> preset 'silverstone_full'
    """
    if name is None:
        return TRACK_PRESETS["silverstone_half"]
    key = str(name).strip().lower()
    if key in ["silverstone", "silverstone_half", "half"]:
        return TRACK_PRESETS["silverstone_half"]
    elif key in ["silverstone_full", "full", "1:1"]:
        return TRACK_PRESETS["silverstone_full"]
    else:
        import warnings
        warnings.warn(f"Map preset '{name}' khong nam trong danh muc. Tu dong fallback ve 'silverstone_half'.")
        return TRACK_PRESETS["silverstone_half"]

