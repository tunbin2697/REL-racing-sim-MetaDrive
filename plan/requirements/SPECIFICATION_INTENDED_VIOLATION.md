# Đối chiếu đặc tả và các sai lệch có chủ đích

> **Cập nhật**: 31/10/2026 · **Nhánh**: `feature/111s-honest-anchor`
> **Đặc tả gốc**: [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md)
> **Đối tượng đối chiếu**: mô hình tham chiếu `ref_111s` (`models/ref_111s/best_agent.pt`, train ở commit `ee99f9d`) và env dùng để eval nó hiện nay.
> Bản trước của file này mô tả nhánh `feature/oop-clean-physics-40d` (obs 40D, không còn là kết quả chính); xem lịch sử git nếu cần.

---

## 1. Bảng đối chiếu

| Mục đặc tả | Đặc tả yêu cầu | Thực tế của `ref_111s` | Loại | Lý do |
| :-- | :-- | :-- | :-- | :-- |
| §1 Mục tiêu | Agent tự học tối ưu lap time bằng thử–sai, không mớm đường | PPO thuần, reward Markov, không có dữ liệu tối ưu trong obs/reward | Tuân thủ | – |
| §2.1 Xe | Giữ mặc định thư viện | Thân xe mặc định MetaDrive (`DefaultVehicle`, 1100 kg), **chỉnh thông số** thành `GT3RaceCar`, xem §2 | **Lệch có chủ đích** | Xe mặc định (40° lái, phanh 119 N/bánh, μ 0.9) không đủ để học phanh và vào cua kiểu đua |
| §2.1 Action | Mặc định `[lái, pedal]` 2D | `[lái, ga, phanh]` 3D qua bộ trộn pedal | **Lệch có chủ đích** | Tách ga và phanh để agent học vùng phanh rõ ràng |
| §2.1 Obs | Không quy định cụ thể | 35D, chỉ dùng hình học tim đường và động học xe | Tuân thủ | Thay 240 tia LiDAR bằng lookahead hình học |
| §2.2 Luật 4 bánh | Chỉ kết thúc khi cả 4 bánh ra ngoài | **Lúc train**: kết thúc khi cả 4 **góc thân xe** ra quá 4.5 m (kerb ảo). **Lúc eval hiện nay**: luật 4 bánh đúng đặc tả | Lệch khi train, tuân thủ khi eval | Kerb ảo là luật của env thời điểm train. Kết quả báo cáo (§3) đo dưới luật 4 bánh |
| §2.2 Không phạt lệch tim đường | Không cài phạt chạy sai làn hay lệch tim đường | **Có** `r_center = −0.04·(lat/4.5)²` mỗi bước | **Lệch** | Khai báo minh bạch. Phạt này kéo xe về giữa đường, tức là **làm chậm** chứ không giúp nhanh hơn; xe vẫn nhanh hơn ghost. Trong thực nghiệm của chủ dự án, đây là cấu hình hội tụ ổn định nhất trên MetaDrive |
| §2.3 Rào chắn, địa hình | Không rào sát mép, kiểm tra bằng hình học | Không rào chắn; kiểm tra bằng hình học | Tuân thủ | Phần ngoài mặt nhựa vẫn là mặt phẳng có cùng ma sát (hạn chế, F22) |
| §3 Tiêu chí F1 | Chỉ dùng làm thước đo, không đưa vào code | Không có reward nào thưởng ga/phanh theo vị trí | Tuân thủ | – |
| §4 Không data leak | Không racing line, không imitation, không dùng kết quả benchmark làm điều kiện train | Đúng. Profile tối ưu chỉ dùng cho ghost và chỉ số đánh giá | Tuân thủ | Nhánh thử nghiệm Phase 2 `--obs-mode line` có dùng thông tin đặc quyền; nhánh đó **không** phải kết quả chính và được ghi riêng |
| §4 Tránh thưởng bám tim đường | Khuyến nghị | `r_center` như trên | Lệch (đã nêu) | – |

## 2. Xe: thân xe mặc định MetaDrive với thông số đã chỉnh

Trong code xe được gọi là `GT3RaceCar` (`game_env/vehicle.py`). Đây **không** phải mô hình xe GT3 thật, mà là thân xe mặc định của MetaDrive được chỉnh thông số theo hướng xe đua:

| Thông số | Mặc định MetaDrive (`DefaultCivilianSedan` trong code) | `GT3RaceCar` (`ref_111s`) |
| :-- | :-- | :-- |
| Lực động cơ mỗi bánh | 789.2 N | 850 N |
| Lực phanh mỗi bánh | 119.2 N | 650 N |
| Góc lái tối đa | 40° | 20° |
| Ma sát lốp | 0.90 | Sau 1.35, **trước 0.90** (bug F16, xem dưới) |
| Giới hạn tốc độ xoay vô lăng | 0.35/bước | 0.18/bước |
| Tốc độ tối đa | 120 km/h trong code (đặc tả ghi mặc định thư viện 80 km/h) | 120 km/h |

Đặc tính đo được trên Bullet (`logs/calibration/calibration_results.json`): 0–100 km/h khoảng 9 s, bám ngang ổn định khoảng 0.90 G, phanh khoảng 1.84 G từ 100 km/h.

**Bug F16**: code cũ ghi ma sát qua danh sách bánh lỗi thời, nên cầu trước luôn giữ 0.9 trong khi cầu sau là 1.35. Xe vì vậy understeer, đuôi ổn định. `GT3RaceCar` giữ tường minh giá trị 0.9 để vật lý trùng với lúc train `ref_111s`. Ghost dùng đúng vật lý này, nên so sánh là công bằng; nhưng xe chưa thật.

**`gt3_anchor`**: vật lý giống hệt `GT3RaceCar`, chỉ thêm các giới hạn gia tốc đo được (envelope) để benchmark và ghost dùng. PPO không đọc các giá trị này.

## 3. Kết quả đo dưới luật đúng đặc tả

Env: Silverstone half, luật 4 bánh, không kerb ảo, `gt3_anchor`.

| | Vòng xuất phát đứng | Vòng flying | Sạch |
| :-- | :-- | :-- | :-- |
| Ghost hiệu chỉnh (T\* 110.62s) | 118.2s | 112.8s | 3/3 |
| `ref_111s`, 20 episode × 2 vòng | 114.8 ± 0.4s | 109.4 ± 0.5s | 18/20 |

Chi tiết và cách hiệu chỉnh ghost: `knowledge/rl_debugging_findings.md` F24.

## 4. Phase 2 (map 1:1) – không phải kết quả chính

Map 1:1 (5.9 km, 3 làn × 4.8 m) với xe `GT3RealCar` đã sửa F16. Các thử nghiệm ở đó lệch đặc tả nhiều hơn (pedal gộp 1 trục, ra ngoài đường thì phạt và chạy tiếp, học từ lỗi, một nhánh có racing line trong obs/reward). Tất cả được khai báo ở `knowledge/ablation_inventory.md` R12–R14b. Phase 2 đã đóng băng.

## 5. Cam kết liêm chính

- Kết quả chính dùng PPO thuần, không có dữ liệu tối ưu trong obs hay reward.
- Mọi điểm lệch đặc tả đều được liệt kê ở §1, kể cả những điểm bất lợi cho kết quả (`r_center`, kerb ảo lúc train, bug F16).
- Mốc so sánh (ghost) được làm mạnh nhất có thể trước khi so, không phải làm yếu đi.
