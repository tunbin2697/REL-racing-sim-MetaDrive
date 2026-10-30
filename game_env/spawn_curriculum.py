"""
Chương Trình Xuất Phát Học Từ Lỗi (Mistake-Learning Spawn Curriculum)
Triển khai theo WI-20 (Quyết định D32, dựa trên cơ chế Mistake Learning của GT Sophy - Wurman et al., Nature 2022):
- Tuân thủ Single Responsibility Principle (SRP): Class SpawnCurriculum chỉ quản lý bộ nhớ đệm các trạng thái
  trước khi mất lái (crash snapshots) của chính tác tử và lấy mẫu chế độ xuất phát cho env.reset().
- Tuyệt đối không dùng dữ liệu tối ưu (optimal profile), hợp lệ cho cả nhánh 'pure' và 'line'.
- Các tỷ lệ phân phối là hằng số định danh rõ ràng ở đầu file:
    + 50% xuất phát đứng tại Grid (standing)
    + 35% xuất phát từ trạng thái ~3 giây (30 bước) trước một lần crash gần đây (snapshot)
    + 15% xuất phát đứng yên tại tọa độ Frenet s ngẫu nhiên trên tim đường (random_s)
"""

from collections import deque
from typing import Deque, Dict, List, Any, Optional
import numpy as np

# Hằng số cấu hình phân phối Spawn Curriculum (WI-20 / D32)
PROB_STANDING: float = 0.50
PROB_SNAPSHOT: float = 0.35
PROB_RANDOM_S: float = 0.15
MAX_SNAPSHOTS: int = 32
STEPS_BEFORE_CRASH: int = 30  # 30 bước x 0.1s = 3.0 giây trước khi mất lái


class SpawnCurriculum:
    """
    Bộ lập lịch điểm xuất phát học từ lỗi (Mistake Learning Curriculum).
    Lưu trữ tối đa 32 trạng thái trước khi crash gần nhất trong deque và sinh options cho MetaRacingEnv.reset().
    """

    def __init__(
        self,
        circuit_length: float = 5919.4,
        maxlen: int = MAX_SNAPSHOTS,
        steps_before_crash: int = STEPS_BEFORE_CRASH,
    ):
        self.circuit_length = float(max(1.0, circuit_length))
        self.maxlen = int(maxlen)
        self.steps_before_crash = int(steps_before_crash)
        self.snapshots: Deque[Dict[str, Any]] = deque(maxlen=self.maxlen)

        # Bộ đếm thống kê phục vụ theo dõi ở mỗi kỳ Eval (WI-20)
        self.spawn_counts: Dict[str, int] = {
            "standing": 0,
            "snapshot": 0,
            "random_s": 0,
        }
        self.segment_completed_count: int = 0

    def update_circuit_length(self, circuit_length: float):
        """Cập nhật chiều dài đường đua chính xác sau khi env khởi tạo polyline"""
        if circuit_length > 10.0:
            self.circuit_length = float(circuit_length)

    def record_crash(self, crash_history: List[Dict[str, Any]], start_mode: str = "standing") -> bool:
        """
        Trích xuất trạng thái ~30 bước (3s) trước khi crash từ episode_history của env để đưa vào deque.
        Áp dụng Lưu ý 3 (plan/plan_notices.md):
        - Nếu lịch sử đủ >= 30 bước: lấy phần tử tại -30.
        - Nếu lịch sử ngắn hơn 30 bước:
            + Nếu xuất phát từ 'snapshot' mà crash lại ngay trong < 30 bước: bỏ qua, không nạp đè
              trạng thái sát tường vào deque (chống hiệu ứng cascading unrecoverable snapshots).
            + Nếu xuất phát từ 'standing' hoặc 'random_s': lấy phần tử đầu tiên history[0].
        """
        if not crash_history:
            return False

        n_steps = len(crash_history)
        if n_steps >= self.steps_before_crash:
            chosen = crash_history[-self.steps_before_crash]
        else:
            if str(start_mode).lower() == "snapshot":
                return False
            chosen = crash_history[0]

        snap = {
            "pos": [float(chosen["pos"][0]), float(chosen["pos"][1])],
            "heading": float(chosen["heading"]),
            "vel": [float(chosen["vel"][0]), float(chosen["vel"][1])],
            "track_s": float(chosen.get("track_s", 0.0)),
        }
        self.snapshots.append(snap)
        return True

    def record_segment_completed(self):
        """Ghi nhận một episode xuất phát giữa vòng (snapshot / random_s) đã cán vạch đích thành công"""
        self.segment_completed_count += 1

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Lấy mẫu chế độ xuất phát cho env.reset(options=...).
        - 50% standing
        - 35% snapshot (nếu deque rỗng thì chuyển về standing)
        - 15% random_s (s ~ U(0, L), vận tốc 0)
        """
        if rng is None:
            u = float(np.random.rand())
        else:
            u = float(rng.random())

        if u < PROB_STANDING:
            self.spawn_counts["standing"] += 1
            return {"start_mode": "standing"}

        elif u < (PROB_STANDING + PROB_SNAPSHOT):
            if len(self.snapshots) == 0:
                self.spawn_counts["standing"] += 1
                return {"start_mode": "standing"}
            if rng is None:
                idx = int(np.random.randint(0, len(self.snapshots)))
            else:
                idx = int(rng.integers(0, len(self.snapshots)))
            snap = self.snapshots[idx]
            self.spawn_counts["snapshot"] += 1
            return {
                "start_mode": "snapshot",
                "pos": list(snap["pos"]),
                "heading": float(snap["heading"]),
                "vel": list(snap["vel"]),
                "track_s": float(snap["track_s"]),
            }

        else:
            if rng is None:
                s_rand = float(np.random.uniform(0.0, self.circuit_length))
            else:
                s_rand = float(rng.uniform(0.0, self.circuit_length))
            self.spawn_counts["random_s"] += 1
            return {
                "start_mode": "random_s",
                "s": s_rand,
            }

    def get_stats_summary(self) -> Dict[str, int]:
        """Trả về thống kê số lượng episode theo loại spawn, kích thước deque và số segment_completed"""
        return {
            "standing": int(self.spawn_counts["standing"]),
            "snapshot": int(self.spawn_counts["snapshot"]),
            "random_s": int(self.spawn_counts["random_s"]),
            "deque_size": int(len(self.snapshots)),
            "segment_completed": int(self.segment_completed_count),
        }

