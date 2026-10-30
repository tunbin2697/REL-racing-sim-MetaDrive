# Lưu Ý Kỹ Thuật Khi Triển Khai Plan v11.2 (Implementation Notices)

> **Vị trí**: `scratch/plan_notices.md` · **Ngày tạo**: 30/10/2026
> **Mục đích**: Ghi nhận các phân tích cơ chế bổ sung cho F17 và 5 điểm bẫy kỹ thuật trong mã nguồn cần tuân thủ nghiêm ngặt khi triển khai **WI-19**, **WI-20**, **WI-21**, **WI-22**, **WI-23**.

---

## 1. Giải thích cơ chế bổ sung cho F17: Vì sao pedal deterministic dao động chu kỳ 2 bước?

Trong `game_env/racing_env.py` (dòng 350–367), chiều thứ 7 của vector quan sát (`obs[6]`) đưa trực tiếp giá trị `last_pedal` của bước $t-1$ vào mạng Actor:
```python
last_pedal = t * (1.0 - b) - b if b > 0.05 else t
```
Tuy nhiên, hàm thưởng vi mô (`micro_reward`) chỉ phạt độ giật góc lái (`r_steer_smooth = -0.1 * (delta_steer ** 2)`), hoàn toàn **không có khoản phạt độ giật chân ga/phanh (`r_pedal_smooth = 0`)**.

- **Cơ chế hình thành bộ điều khiển đóng–ngắt (Bang-Bang / PWM chu kỳ 2 bước)**:
  1. Do mỗi lần văng khỏi đường đều bị phạt nặng và kết thúc episode ngay lập tức, kết hợp với việc $\sigma_{\text{brake}}$ bị đẩy kịch trần $-0.7$ ($\sigma \approx 0.48$) làm nhiễu các pha rà phanh mịn (trail-braking), mạng MLP học cách giữ tốc độ an toàn ~140–160 km/h trên mọi đoạn thẳng.
  2. Thay vì xuất một mức ga trung bình cố định (vốn dễ bị lệch bởi nhiễu Gaussian lúc rollout), mạng Actor tận dụng chính phản hồi `obs[6]` (`last_pedal`) để tạo thành mạch dao động nghịch đảo mỗi bước ($\Delta t = 0.1\text{s}$, tần số $5\text{ Hz}$):
     - Bước chẵn: thấy `last_pedal < 0` $\rightarrow$ xuất lệnh ga dương (`+0.3` đến `+0.7`).
     - Bước lẻ: thấy `last_pedal > 0` $\rightarrow$ xuất lệnh phanh nhẹ (`-0.1` đến `-0.3`).
  3. Trung bình trượt qua 2 bước tạo ra gia tốc dọc hiệu dụng bằng 0 ở quanh ~150 km/h.
- **Hệ quả vật lý**:
  - Việc nhấp ga – nhấp phanh liên tục ở tần số $5\text{ Hz}$ gây ra hiện tượng **dồn tải dọc (longitudinal weight transfer) liên tục** giữa cầu trước và cầu sau.
  - Vì `GT3RealCar` có cấu hình ma sát bất đối xứng ($\mu_{\text{trước}} = 1.55 < \mu_{\text{sau}} = 1.80$, thiết kế để chống xoay khi trail-braking), nhịp giật pedal ngay lúc vào cua làm lực bám ngang cầu trước dao động mạnh, dẫn tới văng understeer/oversteer ngay cả khi tốc độ thực tế (136–149 km/h) thấp hơn $v^*$ (151–177 km/h).
- **Quyết định thiết kế**:
  - **Không sửa hàm thưởng** ở WI-22 (`P2-line`) và WI-23 (`mistake fine-tune`) để giữ nguyên tính đối chứng khoa học tuyệt đối (chỉ thay đổi đúng 1 biến số giữa các nhánh theo D30 và D22).
  - Ghi nhận cơ chế phản hồi qua `obs[6]` này vào phần thảo luận F17 của báo cáo.

---

## 2. Năm (05) Điểm Bẫy Kỹ Thuật Bắt Buộc Xử Lý Khi Code WI-19 → WI-21

### Lưu ý 1 (WI-20 · `MetaRacingEnv.reset`): Bắt buộc đặt `heading_theta` và xóa vận tốc góc khi spawn giữa đường
- **Hiện trạng**: Trong `racing_env.py` hiện tại, xe luôn xuất phát tại Hamilton Straight (hướng đường đua bằng $0\text{ rad}$), nên code chỉ gọi `vehicle.set_position(...)` và `vehicle.set_velocity(...)` mà **chưa gọi `vehicle.set_heading_theta(...)`**.
- **Rủi ro**: Khi spawn ở chế độ `snapshot` hoặc `random_s` tại các khúc cua có góc hướng bất kỳ $\theta \in [-\pi, \pi]$, nếu không đặt lại `heading_theta`, thân xe sẽ giữ hướng $0\text{ rad}$ của vạch xuất phát trong khi vector vận tốc `vel` hướng theo góc $\theta$. Xe sẽ trượt ngang gần $90^\circ - 180^\circ$ ngay tại bước 0 và văng khỏi đường.
- **Cách xử lý bắt buộc**:
  - Gọi `vehicle.set_heading_theta(heading)` sau `vehicle.set_position(pos)`.
  - Đặt lại vận tốc theo đúng vector 2D `[vx_world, vy_world]`.
  - Xóa vận tốc góc (yaw rate) còn tồn dư trên chassis Bullet (nếu có) để xe không bị xoay tại bước 0.

### Lưu ý 2 (WI-20 · `MetaRacingEnv.reset`): Cập nhật `self.current_track_idx` trước khi tính `obs`
- **Hiện trạng**: Trong `reset()` (dòng 426–460 của `racing_env.py`):
  - Dòng 426 gán `self.current_track_idx = 0`.
  - Dòng 456 gọi `_, current_s = self._get_track_s(self.car.position)` nhưng bỏ qua giá trị `idx` trả về.
  - Dòng 460 gọi `get_compact_racing_obs()`, hàm này lại gọi `self._get_track_s(pos, self.current_track_idx)` với `hint_idx = 0`.
  - Trong `_get_track_s` (dòng 281): khi `hint_idx < 40`, thuật toán khóa cứng cửa sổ tìm kiếm trong đoạn `[0, 60]` đầu đường đua (chống bug về đích ảo).
- **Rủi ro**: Khi spawn tại $s = 3000\text{m}$ (`idx ≈ 3000`), nếu `self.current_track_idx` vẫn bằng `0`, `get_compact_racing_obs()` ở bước `reset` sẽ chiếu vị trí $s = 3000\text{m}$ vào điểm tim đường ở đoạn $[0, 60\text{m}]$, làm `lat_offset` và `obs` bước đầu tiên sai lệch hàng trăm mét và kéo `current_track_idx` về đầu đường đua.
- **Cách xử lý bắt buộc**:
  ```python
  idx, current_s = self._get_track_s(self.car.position, hint_idx=None)
  self.current_track_idx = idx
  ```
  Phải thực hiện ngay trước khi gọi `get_compact_racing_obs()`.

### Lưu ý 3 (WI-20 · `SpawnCurriculum`): Chống hiệu ứng chuỗi snapshot sát tường (Cascading Unrecoverable Snapshots)
- **Rủi ro**: Khi một episode được spawn từ `snapshot` (cách điểm crash 30 bước = 3.0s) nhưng agent vẫn chưa học kịp và crash lại chỉ sau 10–15 bước. Nếu lấy phần tử đầu tiên của episode ngắn ngủi đó nạp tiếp vào `deque(maxlen=32)`, thì snapshot mới chỉ còn cách điểm văng 10–15 bước (lúc xe đã mất lái hoặc chệch quá xa). Sau vài chu kỳ, `deque` sẽ bị lấp đầy bởi các trạng thái "chết chắc" không thể cứu vãn.
- **Cách xử lý bắt buộc**:
  - Khi lấy mẫu từ `last_crash_history`, nếu `len(last_crash_history) >= 30` thì lấy phần tử `last_crash_history[-30]`.
  - Nếu episode ngắn hơn 30 bước (`len < 30`):
    - Với episode xuất phát từ `standing` hoặc `random_s`: lấy `last_crash_history[0]` (theo đúng spec WI-20).
    - Với episode xuất phát từ chính một `snapshot` mà chết sớm trong vòng `< 30` bước: **không ghi đè lại phần tử đầu của nó vào `deque`** (vì bản thân điểm xuất phát đó đã nằm sẵn trong `deque` rồi, nạp thêm một điểm sát tường hơn chỉ làm suy thoái chất lượng bộ nhớ).

### Lưu ý 4 (WI-19 · `RacingLineObsEncoder`): Đồng bộ quy ước dấu pháp tuyến trái và chiếu chính xác
- **Quy ước dấu**: Trong `racing_env.py` (dòng 324–325), pháp tuyến trái tại điểm tim đường $i$ là:
  $$\mathbf{n}_{\text{left}}[i] = [-\sin(\theta[i]), \cos(\theta[i])]$$
  và `ego_lat = dot(pos - track_xy[i], n_left[i])`.
- `RacingLineObsEncoder` phải dùng **chính xác công thức pháp tuyến trái này** để tính `line_offset[i]` nhằm đảm bảo phép trừ `(ego_lat - line_offset[idx]) / lat_norm` khớp dấu 100%.
- Hơn nữa, do `optimal_line_xy` trong `RacingBenchmarkEngine` vốn được sinh ra trực tiếp dọc theo pháp tuyến của từng điểm `track_xy[i]` (cùng kích thước mảng $N$), cần kiểm tra cả tìm điểm gần nhất theo cửa sổ / KD-tree lẫn phép chiếu pháp tuyến để sai số dựng lại đường đua `< 0.05 m` (vượt xa ngưỡng `< 0.5 m` của smoke test WI-19).

### Lưu ý 5 (WI-21 · `EvalPerturbation` & Đồng bộ hằng số D14 v2):
- **Khởi tạo hàng đợi `--action-delay k`**: Khi `k > 0`, hàng đợi FIFO cần được làm mới (reset) ở đầu mỗi episode với `k` phần tử hành động trung lập `[0.0, -1.0, -1.0]` (tay lái thẳng, nhả ga, nhả phanh) để độ dài hàng đợi luôn cố định bằng `k` ngay từ bước đầu tiên.
- **Nhiễu `--obs-noise σ`**: Áp dụng công thức nhân `obs_noisy = obs * (1.0 + rng.normal(0.0, sigma, size=obs.shape))` theo đúng chuẩn Fuchs et al. (2021) thông qua wrapper độc lập trong `evaluator.py`/`evaluate.py`, không sửa `MetaRacingEnv`.
- **Đồng bộ văn bản D14 v2**: Giữ chuẩn `finish_T0 = 186.0, finish_tau = 26.0` trên `silverstone_full` (đã có trong `game_env/track_presets.py` và D14 v2).

