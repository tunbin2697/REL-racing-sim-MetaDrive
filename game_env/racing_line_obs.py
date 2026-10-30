"""
Mã hóa thông tin Racing Line & Vận tốc tối ưu vào Vector Quan sát (Privileged Optimization Prior Encoder)
Triển khai theo WI-19 (Quyết định D30, D31):
- Tuân thủ Single Responsibility Principle (SRP): Class này độc lập, chỉ chịu trách nhiệm chiếu
  quỹ đạo tối ưu (optimal_line_xy) và vận tốc tối ưu (optimal_speed_profile) từ RacingBenchmarkEngine
  lên hệ tọa độ Frenet của tim đường (track_xy, track_theta, track_s) và trích xuất 12 chiều đặc trưng.
- Chỉ được kích hoạt khi MetaRacingEnv chạy ở chế độ obs_mode="line".
- Tuyệt đối không can thiệp vào hàm thưởng (reward function) hay nhánh obs_mode="pure".
"""

from typing import Optional, Sequence
import numpy as np
from scipy.spatial import cKDTree


class RacingLineObsEncoder:
    """
    Bộ mã hóa 12 chiều đặc trưng Racing Line & V* cho nhánh đối chứng '--obs-mode line':
      - Chiều 0: Độ lệch ngang của xe so với racing line tại điểm hiện tại: (ego_lat - line_offset[idx]) / lat_norm, clip [-2.0, 2.0]
      - Chiều 1: Độ lệch tốc độ của xe so với v* tại điểm hiện tại: (speed_ms - line_v[idx]) / v_norm, clip [-1.0, 1.0]
      - Chiều 2..11: Tại 5 mốc lookahead_m phía trước, mỗi mốc gồm 2 giá trị:
          + line_offset[wpt_idx] / lat_norm, clip [-1.5, 1.5]
          + line_v[wpt_idx] / v_norm, clip [0.0, 1.2]
    """

    FEATURE_DIM: int = 12

    def __init__(
        self,
        benchmark_engine,
        track_xy: np.ndarray,
        track_theta: np.ndarray,
        track_s: np.ndarray,
        lat_norm: float,
        v_norm: float,
        lookahead_m: Sequence[float],
    ):
        self.track_xy = np.asarray(track_xy, dtype=np.float32)
        self.track_theta = np.asarray(track_theta, dtype=np.float32)
        self.track_s = np.asarray(track_s, dtype=np.float32)
        self.num_points = int(len(self.track_xy))
        self.circuit_length = float(self.track_s[-1]) if self.num_points > 0 else 1.0

        self.lat_norm = float(lat_norm)
        self.v_norm = float(v_norm)
        self.lookahead_m = [float(d) for d in lookahead_m]

        opt_xy = np.asarray(benchmark_engine.optimal_line_xy, dtype=np.float32)
        opt_v = np.asarray(benchmark_engine.optimal_speed_profile, dtype=np.float32)

        # Pháp tuyến trái tại mỗi điểm tim đường: n_left = [-sin(theta), cos(theta)]
        # (Đồng bộ 100% quy ước dấu với lat_offset trong MetaRacingEnv.get_compact_racing_obs)
        nx = -np.sin(self.track_theta)
        ny = np.cos(self.track_theta)
        self.normals_left = np.column_stack([nx, ny]).astype(np.float32)

        self.line_offset = np.zeros(self.num_points, dtype=np.float32)
        self.line_v = np.zeros(self.num_points, dtype=np.float32)

        if len(opt_xy) == self.num_points:
            # Trong RacingBenchmarkEngine, optimal_line_xy[i] được dựng trực tiếp trên pháp tuyến của track_xy[i].
            # Kiểm tra sai số phương tiếp tuyến; nếu khớp 1-1 thì chiếu trực tiếp để đạt sai số hình học = 0.0m.
            disp = opt_xy - self.track_xy
            lat_proj = np.sum(disp * self.normals_left, axis=1)
            reconstructed = self.track_xy + self.normals_left * lat_proj[:, None]
            max_recon_err = float(np.max(np.linalg.norm(reconstructed - opt_xy, axis=1)))
            if max_recon_err < 0.25:
                self.line_offset = lat_proj.astype(np.float32)
                self.line_v = opt_v.astype(np.float32)
            else:
                self._project_via_kdtree(opt_xy, opt_v)
        else:
            self._project_via_kdtree(opt_xy, opt_v)

    def _project_via_kdtree(self, opt_xy: np.ndarray, opt_v: np.ndarray):
        """Dự phòng: Tìm điểm racing line gần nhất với từng điểm tim đường qua KD-Tree rồi chiếu lên pháp tuyến trái"""
        tree = cKDTree(opt_xy)
        _, nearest_idx = tree.query(self.track_xy, k=1)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64)
        disp = opt_xy[nearest_idx] - self.track_xy
        self.line_offset = np.sum(disp * self.normals_left, axis=1).astype(np.float32)
        self.line_v = opt_v[nearest_idx].astype(np.float32)

    def reconstruct_line_xy(self) -> np.ndarray:
        """Dựng lại tọa độ 2D của Racing Line từ tim đường và line_offset (dùng cho Smoke Test WI-19)"""
        return (self.track_xy + self.normals_left * self.line_offset[:, None]).astype(np.float32)

    def features(
        self,
        idx: int,
        ego_lat: float,
        speed_ms: float,
        current_s: Optional[float] = None,
    ) -> np.ndarray:
        """
        Trích xuất 12 chiều đặc trưng Racing Line & V* tại vị trí hiện tại của xe.
        Tất cả các chiều đều được chuẩn hóa theo preset và kẹp (clip) chặt chẽ.
        """
        idx_clamped = int(idx) % self.num_points
        s_now = float(self.track_s[idx_clamped]) if current_s is None else float(current_s)

        # 1. Độ lệch ngang của xe so với racing line tại chỗ
        lat_err_norm = float(np.clip((float(ego_lat) - float(self.line_offset[idx_clamped])) / self.lat_norm, -2.0, 2.0))

        # 2. Độ lệch vận tốc của xe so với v* tại chỗ
        v_err_norm = float(np.clip((float(speed_ms) - float(self.line_v[idx_clamped])) / self.v_norm, -1.0, 1.0))

        out = [lat_err_norm, v_err_norm]

        # 3. 5 mốc lookahead phía trước (mỗi mốc 2 chiều: line_offset chuẩn hóa và line_v chuẩn hóa)
        for dist_ahead in self.lookahead_m:
            target_s = (s_now + dist_ahead) % self.circuit_length
            wpt_idx = int(np.searchsorted(self.track_s, target_s)) % self.num_points
            off_norm = float(np.clip(float(self.line_offset[wpt_idx]) / self.lat_norm, -1.5, 1.5))
            v_opt_norm = float(np.clip(float(self.line_v[wpt_idx]) / self.v_norm, 0.0, 1.2))
            out.append(off_norm)
            out.append(v_opt_norm)

        return np.asarray(out, dtype=np.float32)

