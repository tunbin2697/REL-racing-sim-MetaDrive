# Vật lý xe, MetaDrive và Bullet: tổng hợp phân tích

> Tập trung mọi phân tích về vật lý xe, cách MetaDrive/Bullet mô phỏng, hiệu chuẩn xe, benchmark giải tích
> và ghost car. Thay thế `knowledge/gt3_calibration.md` (đã gộp vào đây) và các mục F10, F16, I12 cũ trong
> `knowledge/rl_debugging_findings.md`.
>
> Nhãn độ tin cậy: **[Đo]** đo trực tiếp trên Bullet · **[Mã nguồn]** đọc từ code MetaDrive/dự án ·
> **[Suy luận]** lập luận, chưa kiểm chứng bằng thí nghiệm riêng.
>
> Số đo thô: `logs/calibration/calibration_results.json`, `logs/calibration/calibration_run.txt`.
> Script: `scripts/calibrate_vehicle.py`, `scripts/tests/bench_regression.py`, `scripts/tests/ghost_lap_test.py`.

---

## 1. MetaDrive mô phỏng xe như thế nào

### 1.1. Mô hình
- Thân xe là rigid body Bullet, gắn 4 bánh **raycast vehicle**: mỗi bánh là một tia quét xuống, kèm lò xo
  và giảm chấn. Không có mô hình lốp kiểu Pacejka; giới hạn bám đến từ `frictionSlip` × tải bánh. [Mã nguồn]
- Không có khí động học (không cản gió, không downforce), không nhiệt độ hay mòn lốp. [Mã nguồn]
- dt điều khiển = 0.1 s (physics step 0.02 s × decision_repeat 5). [Mã nguồn]

### 1.2. Cách áp lực điều khiển (`base_vehicle._apply_throttle_brake`) [Mã nguồn]
- **Ga** (pedal ≥ 0): `applyEngineForce(max_engine_force × pedal)` cho **cả 4 bánh**, tức dẫn động 4 bánh.
  Đồng thời `setBrake(2.0)`, tương đương một lực cản lăn nhỏ. Khi vượt `max_speed_km_h` thì lực kéo về 0.
- **Phanh** (pedal < 0): lực kéo về 0, `setBrake(|pedal| × max_brake_force)` cho cả 4 bánh.
- **Lái**: chỉ 2 bánh trước, góc = steer × `max_steering`.
- Hệ quả [Đo]: a_x = 4F/m **chính xác** (4 × 2400 / 1100 = 8.727 m/s², khớp số đo đến 3 chữ số) và không
  giảm theo tốc độ. Xe chạm trần tốc độ trên mọi đoạn thẳng đủ dài.

### 1.3. Hình học `DefaultVehicle` (dùng cho mọi xe trong dự án) [Mã nguồn]
| Đại lượng | Giá trị |
| :-- | :-- |
| Khối lượng chassis | **1100 kg, cố định** (`MASS`); `config["mass"]` không được đọc sau khi tạo xe |
| Trọng tâm → cầu trước / cầu sau | 1.052 m / 1.417 m (tải tĩnh trước 3100 N, sau 2298 N mỗi bánh) |
| Nửa vệt bánh | 0.815 m |
| Bán kính lốp | 0.313 m |
| Hành trình / độ cứng treo | 15 cm / 40 |
| Hộp chassis | 4.515 × 1.852 × 1.19 m |
| rollInfluence | 0.5 |

Trọng tâm cao (~0.55 m) và gần cầu trước là nguyên nhân của hiện tượng chổng đuôi (mục 4.3).

---

## 2. Bug và cạm bẫy đã tìm ra

### 2.1. F16: ma sát cầu trước chưa bao giờ được áp [Đo]
- `BaseRaceVehicle.attach_agent()` cũ gọi `setFrictionSlip()` qua list `raw_vehicle.wheels`. List này lỗi thời
  sau khi MetaDrive dựng lại xe: phần tử 0–1 (bánh trước) trỏ vào vùng nhớ rác (đọc ra tải 9e7, 2e30 N,
  `contact = 0`).
- Đọc trực tiếp từ `raw_vehicle.system.getWheel(k)`: ma sát thực là **trước 0.90 (mặc định MetaDrive) /
  sau = `wheel_friction`**, ổn định qua mọi lần reset. Bản 111s và Phase 1 đều chạy với **trước 0.90 / sau 1.35**.
- Đã sửa: `attach_agent` đọc bánh từ `system.getWheel(k)` và hỗ trợ `front_wheel_friction`.
  `GT3RaceCar` đặt tường minh `front_wheel_friction = 0.90` để giữ y nguyên vật lý Phase 1. Đã kiểm chứng:
  bản 111s ra đúng 115.10s, 73.2% full ga; ghost map half đúng 119.5s.
- **Bài học phương pháp**: đọc trạng thái từ chính solver vật lý, không tin vào wrapper. Bản báo cáo hiệu chuẩn
  đầu tiên (do script tự viết kết luận cứng) đã kết luận sai rằng "μ không có tác dụng".

### 2.2. Các cạm bẫy khác
| Cạm bẫy | Hệ quả | Xử lý |
| :-- | :-- | :-- |
| `config["mass"]` không có tác dụng [Đo] | Đặt 1350 kg nhưng Bullet vẫn dùng 1100 | Đặt config = 1100 cho trung thực |
| Đo hiệu chuẩn qua `MetaRacingEnv.step()` | Episode kết thúc khi ra khỏi đường đua, cắt ngang bài phanh/vòng tròn; ra số sai (bám 0.48–0.77 G, quãng phanh 200 km/h ngắn hơn 180 km/h) | Đo bằng `env.env.step()` của MetaDrive trên địa hình tự do (mục 3) |
| `max_brake_force` lớn [Đo] | Giảm tốc bão hòa sớm, phanh thành bật/tắt; với 2200 N chỉ ~10–20% đầu bàn đạp có tác dụng, riêng deadband 5% đã cho ~1 G | Đặt lực phanh dưới ngưỡng bão hòa (mục 4.3) |
| `force_recompute` của benchmark ghi đè cache | File cache map half (được git track) bị đổi | `scripts/tests/bench_regression.py` giữ và khôi phục cache |
| Đo a_y bằng yaw rate của body | Đỉnh giả ±3–4 G khi xe giật/xoay | Chỉ dùng trong bài đo ổn định (lấy trung bình trượt), không dùng làm số bám |

---

## 3. Phương pháp đo (`scripts/calibrate_vehicle.py`)
- Spawn xe qua `MetaRacingEnv`, sau đó gọi thẳng `env.env.step([steer, pedal])` (bỏ qua luật track limit và
  slew limiter). Địa hình ngoài đường là mặt phẳng có va chạm; z ổn định khi chạy ra ngoài. [Đo]
- **Tăng tốc**: hết ga từ đứng yên; mốc 0→100/200/250, a_x theo dải tốc độ.
- **Phanh**: hết ga tới tốc độ vào, rồi phanh hết tới khi dừng; quãng phanh, giảm tốc trung bình.
- **Bám ngang** (constant-speed ramp-steer): giữ tốc độ bằng bộ P, tăng góc lái 0 → 1 trong 25s;
  $a_y = v\dot\psi$, lấy đỉnh trung bình trượt 5 bước; quá 20° góc trượt thì tính là xoay.
- **Ổn định khi phanh**: đếm số bước cả hai bánh sau mất tiếp đất, pitch cực đại, yaw rate cực đại.
- **Ổn định khi phanh trong cua**: giữ cua ổn định 8s ở 1.2–1.4 G (PI bám yaw rate), rồi nhả ga hoặc phanh 30%
  trong 3s, giữ nguyên góc lái.
- **Ổn định cao tốc**: giữ ≥ 250 km/h, lái sin ±0.04; đo Δz, roll, pitch.
- Script chỉ xuất số đo thô; kết luận viết tay ở file này.

---

## 4. Kết quả đo

### 4.1. Bảng tổng hợp [Đo]
| Chỉ số | GT3RaceCar (map half, legacy) | **GT3RealCar (chốt)** | Tham chiếu GT3 thật (ước lượng, cần nguồn) |
| :-- | :-- | :-- | :-- |
| Ma sát thực FL, FR, RL, RR | 0.90, 0.90, 1.35, 1.35 | 1.55, 1.55, 1.80, 1.80 | |
| Engine / brake (N/bánh) | 850 / 650 | 2400 / 160 | |
| a_x hết ga | 3.09 m/s² | 8.73 m/s² | giảm mạnh ở tốc độ cao do cản gió |
| 0→100 / 0→200 km/h | 9.0 s / – | 3.2 s / 6.4 s | ~3–3.5 s / ~7–9 s |
| Tốc độ trần đạt được | 121 km/h | 263 km/h | 253–265 km/h cuối Hangar |
| Bám ngang 60/100/150/200 km/h | 0.88–0.90 G | 1.53 / 1.56 / 1.59 / 1.60 G | ~1.5–2.0 G (có downforce) |
| Phanh hết 100 / 200 / 250 km/h | 1.84 G, **chổng đuôi** (pitch 25°) | 1.63 / 1.65 / 1.66 G, pitch 3.6° | ~1.8–2.0 G |
| Quãng phanh 250 → 0 | – | 150 m | |
| Cao tốc 263 km/h, lái sin | – | Δz 1.4 cm, roll 2.6°, pitch 0.75° | |

### 4.2. Bám ngang do μ cầu trước quyết định [Đo]
| Cấu hình | Bám ngang | Ghi chú |
| :-- | :-- | :-- |
| Legacy, μ trước 0.90 (mọi giá trị μ sau ≥ 0.9) | 0.88–0.93 G | Trần cứng 0.9 G. Nguồn gốc là bug F16 |
| μ sau < 0.9 (cầu trước vẫn 0.9) | xoay đuôi | Cầu sau yếu hơn cầu trước |
| μ 1.35 / 1.65 / 1.95 cả 4 bánh | 1.34–1.51 / 1.62–1.79 / 1.90–2.13 G | Tỉ lệ thuận μ sau khi sửa F16 |
| μ trước 1.55, sau 1.80 | 1.53–1.60 G | Bị giới hạn bởi cầu trước |

Đỉnh a_y đạt ở góc lái rất nhỏ khi tốc độ cao (~0.05 ở 200 km/h, ~0.36 ở 60 km/h); lái thêm thì a_y giảm
nhẹ (understeer).

### 4.3. Phanh: bão hòa và chổng đuôi [Đo]
**Ngưỡng bão hòa.** Giảm tốc 0.6s đầu khi đạp hết bàn đạp không tăng nữa khi lực ≥ ~260 N (μ trước 0.9),
hoặc ≥ ~470 N (μ 1.65). Dưới ngưỡng này, giảm tốc tỉ lệ với bàn đạp.

**Chổng đuôi.** Khi giảm tốc vượt ~1.7 G, cả hai bánh sau nhấc khỏi mặt đất, pitch ~25°, và ở 250 km/h xe tự
xoay (yaw 2–3 rad/s), dù đi thẳng. Theo lập luận lật quanh cầu trước, ngưỡng là $a \approx g \cdot b/h$ =
1.05/0.55 g ≈ 1.9 G, khớp bậc độ lớn với số đo [Suy luận]. Xe map nhỏ cũng chổng đuôi khi phanh hết, nhưng
không xoay vì tốc độ thấp.

| Lực / bánh (μ 1.55/1.80) | 100 km/h | 180 km/h | 250 km/h |
| :-- | :-- | :-- | :-- |
| 120 N | 1.41 G, ổn | 1.43 G | 1.44 G |
| **160 N (chốt)** | **1.63 G, pitch 3.6°** | **1.65 G** | **1.66 G, yaw 0** |
| 180 N | 1.72 G, nhấc 4 bước | 1.75 G | 1.73 G, pitch 7.7°, yaw 1.1 rad/s |
| ≥ 200 N (μ 1.65) | chổng đuôi, pitch 25° | | xoay, yaw 2–3 rad/s |

**Không sửa bằng hình học.** Dời điểm gắn bánh trước ra phía trước (1.05 → 1.40–1.60 m) làm pitch tăng lên
37°; tăng độ cứng treo 40 → 80 thì xe lật (pitch 89°). Mô hình raycast không phản ứng như xe thật, nên giải
pháp là giới hạn lực phanh. [Đo]

### 4.4. Cân bằng trước/sau: ổn định khi phanh trong cua [Đo]
| μ trước / sau | Bám 100 / 200 km/h | Nhả ga giữa cua | Phanh 30% giữa cua (150–230 km/h, 1.2–1.4 G) |
| :-- | :-- | :-- | :-- |
| 1.65 / 1.65 | 1.65 / 1.80 G | ổn | **xoay 68–72°** |
| **1.55 / 1.80** | **1.56 / 1.60 G** | ổn | **ổn, góc trượt ≤ 3.4°** |
| 1.45 / 1.85 | 1.46 / 1.49 G | ổn | ổn, ≤ 1.6° |
| 1.40 / 1.95 | 1.41 / 1.44 G | ổn | ổn, ≤ 1.3° |

Khi phanh, tải dồn lên cầu trước và cầu sau mất bám; với μ bằng nhau, xe nghiêng về oversteer. Xe đua thật
được set up với cầu sau bám hơn để ổn định. Legacy "trước 0.9 / sau 1.35" thiên understeer mạnh, có thể là
một lý do agent Phase 1 không bị xoay khi phanh trễ [Suy luận].

---

## 5. Cấu hình chốt (D27)

| Tham số | `GT3RaceCar` (map half) | `GT3RealCar` (map full) | Lý do (GT3RealCar) |
| :-- | :-- | :-- | :-- |
| mass | 1100 | 1100 | Trùng giá trị Bullet thực dùng |
| max_engine_force | 850 | 2400 | 0→100 3.2s; chạm 260 km/h sau 277m từ đứng yên |
| wheel_friction (sau) / front | 1.35 / **0.90 (giữ legacy)** | 1.80 / 1.55 | Bám cao nhất mà vẫn ổn định khi trail-braking |
| max_brake_force | 650 | 160 | Mạnh nhất không chổng đuôi; bàn đạp tuyến tính |
| max_steering | 20° | 22° | |
| max_speed_km_h | 120 | 260 | |
| envelope (lat / brake / engine) | – (legacy v22) | 13.8 / 15.5 / 8.70 m/s² | Mục 6.3 |

**R4 lookahead** (map full: [15, 40, 80, 130, 200] m, dx_norm 200): quãng phanh 250 → 0 là 150m; 250 → 85 km/h
(cua chậm nhất) khoảng 133m. Mốc 200m bao trọn điểm phanh xa nhất.

---

## 6. Benchmark giải tích (optimal profile, `game_env/benchmark.py`)

### 6.1. Pipeline
1. **Racing line**: minimum curvature tuyến tính hóa (Heilmeier et al. 2020). Lấy mẫu ~3m;
   $\min_\alpha \|D_2(x_0+n_x\alpha)\|^2 + \|D_2(y_0+n_y\alpha)\|^2 + w\|\alpha\|^2$; ràng buộc
   $|\alpha| \le$ W/2 − 0.45 (±4.05m map half, ±6.75m map full); giải bằng `lsq_linear`; spline tuần hoàn rồi
   làm mịn Gaussian. Các tham số theo mét được nhân với scale = L/2851.8.
2. **Speed profile** (3-pass, Lipp & Boyd 2014, elip Kamm): Pass 1 $v=\sqrt{a_{lat}/\kappa_{env}}$ với bao
   độ cong −4/+22 mẫu và phần bù chicane; Pass 2 quét lùi (phanh); Pass 3 quét tiến (tăng tốc); mỗi pass lặp 4
   lần; cuối cùng làm mịn Gaussian σ = 2.
3. $T^* = \sum \Delta s / v$.

### 6.2. Hai chế độ
| | v22 legacy (map half) | v23 envelope (map full) |
| :-- | :-- | :-- |
| a_lat | Bậc khớp tay 8.85 / 9.10 / 10.20 / 11.80 theo κ, dκ | Hằng số đo được 13.8 |
| a_brake / a_engine | 6.20 / 3.40 (viết cứng) | 15.5 / 8.70 (đo) |
| Elip Kamm | sàn 0.15, trần 0.92 | **chặt**: sàn 0.05 |
| $T^*$ | **112.59 s** (kiểm tra hồi quy: tính lại từ đầu, racing line trùng cache 0.0000m) | **124.03 s** (L = 5919m, đỉnh cua 85–187 km/h) |

- Sàn 0.15 của legacy cho phép phanh khoảng 0.4·a_brake ngay cả khi đã dùng hết bám ngang. Về vật lý thì không
  khả thi: √(1.45² + 0.75²) = 1.63 G > trần 1.56 G, và ghost đã xoay ở đúng chỗ đó.
- So với số đo thật, các hệ số legacy lệch theo cả hai chiều: phanh 6.2 m/s² (0.63 G) trong khi Bullet phanh được
  ~18 m/s²; bậc bám 1.04–1.2 G cho cua nhanh thì vượt trần thật 0.9 G.
- Cache đặt tên theo map + xe, lưu kèm `profile_version` và `envelope`; đổi envelope là tự tính lại.
- Profile chỉ dùng để đối chiếu và cho ghost (zero leakage).

### 6.3. Envelope (D28)
`envelope_a_lat = 13.8` (90% của 1.56 G), `envelope_a_brake = 15.5` (mức thấp có biên của 1.63–1.66 G),
`envelope_a_engine = 8.70`. Lưu trong `VehicleConfig`, chỉ benchmark và ghost đọc.

### 6.4. Ghost car (`OptimalRacingAgent`), R8
| Map | Vòng 1 (xuất phát đứng) | Vòng 2 (flying) | T* |
| :-- | :-- | :-- | :-- |
| half | 119.5 s | | 112.59 s |
| full | 128.6 s | **124.9 s** | 124.03 s |

- Ghost map full chạy 2 vòng không crash. 28% số bước có 1–3 bánh lấn vạch do racing line sát mép (thiết kế
  2 bánh đè vạch), vẫn hợp lệ theo luật 4 bánh.
- Những lần thất bại trước khi đạt, theo thứ tự: understeer khi profile đòi 100% bám; giật yaw do chổng đuôi
  khi phanh; xoay do phanh trong cua (vòng tròn ma sát tính theo góc lái); vào cua quá tốc độ (MPC thả trôi khi
  lệch nhỏ).
- Các thay đổi, chỉ áp cho xe có envelope (xe map nhỏ giữ luật legacy, vẫn ra 119.5s):
  1. Vòng tròn ma sát theo tải ngang ước lượng $v^2\kappa/a_{lat,max}$ thay vì theo góc lái, với elip chặt.
  2. Bám tốc độ bằng preview: $a = (v_{pv}^2 - v^2)/(2d) + 0.8(v_{target} - v)$, d = max(6, 0.35v).
  3. Mapping bàn đạp phanh theo `envelope_a_brake`; giới hạn gia tốc MPC theo envelope; góc lái tối đa lấy từ xe.
- `OptimalRacingAgent.from_benchmark()` dùng chung cho ghost và `spectate_ai --agent optimal`, nên luôn bám đúng
  profile của map đang chạy.

---

## 7. Sai số mô hình: vì sao RL có thể nhanh hơn ghost (I12)

- Đây là **sai lệch giữa mô hình và hệ thật** (model–plant mismatch) của điều khiển dựa trên mô hình, không phải
  PID (PID là bộ điều khiển phản hồi không dùng mô hình). Profile được tính trên mô hình chất điểm + elip ma sát;
  mô hình sai ở đâu thì profile sai ở đó. RL không dùng mô hình mà tối ưu trực tiếp trên simulator, nên khai thác
  được cả những gì mô hình bỏ qua (trượt có kiểm soát, chuyển tải, đặc thù của Bullet).
- Lặp lại phép tính nhiều lần ra cùng một kết quả chỉ chứng minh **tính tái lập**, không chứng minh **tính tối ưu**:
  - Racing line là minimum curvature, không phải minimum time.
  - Speed profile chỉ tối ưu thời gian cho chất điểm trên đường cố định, với các hệ số khớp tay.
  - Mô hình chất điểm không có trượt, chuyển tải, quán tính quay.
- **Cận trên, không phải cận dưới**: mọi quỹ đạo chạy được thật trong simulator cho
  $T_{opt} \le T_{chạy được}$. Ghost chạy được 124.9s nên $T_{opt}^{full} \le 124.9$s. Bản 111s chạy được 111.4s
  nên $T_{opt}^{half} \le 111.4$s, một cận chặt hơn ghost. $T^*$ (tích phân profile) và lap time ghost chạy thật
  là hai con số khác nhau; chỉ con số chạy thật mới là cận hợp lệ. Báo cáo nên gọi ghost là **"quỹ đạo tham chiếu
  giải tích khả thi"**.
- Lý do bản 111s nhanh hơn ghost map half [Suy luận]: biên lệch ngang ±4.05m thận trọng; các hệ số legacy lệch
  khỏi vật lý thật (mục 6.2); ghost bám profile không hoàn hảo. Cần đo thêm vòng flying của bản 111s.
- Đề xuất phân tích cho báo cáo: phân rã khoảng chênh 111s − ghost theo bề rộng đường dùng, a_y đỉnh từng cua,
  điểm phanh, góc trượt.

---

## 8. Scale map và các hằng số phụ thuộc vật lý

### 8.1. Phép đồng dạng Froude không áp được [Suy luận]
- Lập luận "Froude vòng tròn" trong tài liệu specs (116.5 × 0.694 = 80.8s, rồi 80.8 × 1.44 = 116.5s) "khớp chuẩn"
  chỉ vì chính phép tính đó, không phải bằng chứng.
- Xe không scale được, vì hình học `DefaultVehicle` cố định; nên phép đồng dạng không chặt chẽ. Đây là lý do chọn
  map 1:1 cùng xe thật (hướng Y). Chi phí bước mỗi vòng: xem F12 trong `rl_debugging_findings.md` (×1.44, không
  phải ×3–4).

### 8.2. Chuẩn hóa obs map full (D23)
Giữ dải giá trị của từng chiều giống map nhỏ: dx/200, dy/80, κ×104 (= 50 × 2.075), v/80, vy/40, lệch ngang /7.2.
Ví dụ: với dx/70, mốc lookahead 130m và 200m luôn bị kẹp ở 1.5; với v/40, tốc độ bị kẹp ở 216 km/h.

### 8.3. Hằng số thưởng đích map full (D14 v2: T0 186s, τ 26s)
- Quy tắc cũ $T_0 = L/(0.475\,v_{max})$ = 172s giả định lap time tỉ lệ với $v_{max}$. Nhưng tốc độ vào cua bị
  chặn bởi trần bám, không tăng theo $v_{max}$.
- Quy tắc mới (người dùng chốt) chỉ dùng **hình học tim đường + trần bám đo được + v_max**, không dùng profile:
  $T_{ref} = \sum \Delta s / \min(v_{max}, \sqrt{a_{lat}/|\kappa_{center}|})$, với a_lat của từng xe (half 8.83,
  full 15.3 m/s²) → 105.6s / 108.8s. Giữ map half 180/25: $T_0^{full} = 180 \times 108.8/105.6 = 186$s,
  $\tau = 26$s. (Lần áp đầu tiên ra 219/30 là do dùng trần bám sai 0.9 G của bug F16.)
- Thưởng ở T = 124 / 135 / 150 / 165s: 43.5 / 28.4 / 15.8 / 8.8 điểm. Map half ở 112.6 / 125 / 140s: 59 / 36 / 20 điểm.

---

## 9. Sửa các khẳng định cũ trong tài liệu dự án

| Khẳng định | Nguồn | Sửa |
| :-- | :-- | :-- |
| Cần góc lái 26° để ôm cua R = 25m | specs | Góc Ackermann chỉ cần atan(2.75/25) ≈ 6.3°; 20° đã dư. Số đo: đỉnh bám ở R ≈ 18m chỉ dùng lái 0.36 × 22° ≈ 8° [Đo] |
| `max_steer_rate` tính theo rad/bước | specs | Đơn vị chuẩn hóa: 0.18 × 20° = 3.6°/bước [Mã nguồn] |
| Khối lượng xe chỉnh được | specs, vehicle.py cũ | Cố định 1100 kg [Đo] |
| "MetaDrive mô phỏng chính xác 100% trọng tâm, quán tính..." | `08_sim_racing_telemetry_and_physics.md` §1.2 | Không đúng: không có khí động học; hình học xe cố định gây chổng đuôi ở ~1.7 G; ma sát chỉ áp đúng khi đi qua `system.getWheel` |
| $F_{brake} = F_{max\_brake} \cdot |a|$ là lực | `08_...` §1.1 | Giảm tốc bão hòa ở vài trăm N/bánh; trên ngưỡng đó, lực phanh không còn tỉ lệ giảm tốc [Đo] |
| Lực kéo cho "bánh dẫn động" | `08_...` §1.1 | Cả 4 bánh [Mã nguồn] |
| GT3RealCar bám 1.85–2.0 G với μ 1.65 | báo cáo hiệu chuẩn đầu tiên | Chuỗi viết cứng trong script, không phải số đo; đã thay bằng mục 4 |

---

## 10. Hạn chế cần nêu trong báo cáo
1. Không có khí động học: a_x không giảm theo tốc độ, bám ngang không tăng theo tốc độ như xe có downforce.
2. Hình học `DefaultVehicle` cố định (trọng tâm cao, gần cầu trước) gây chổng đuôi, nên phải giới hạn phanh ở ~1.65 G.
3. Xe Phase 1 có ma sát cầu trước thực tế 0.9 (F16). Hai phase không cùng cân bằng xe: Phase 1 là understeer
   cưỡng bức với bám 0.9 G; Phase 2 cân bằng hơn với bám 1.56 G nhưng nhạy hơn khi phanh trong cua.
4. Ghost là tham chiếu giải tích dựa trên mô hình chất điểm, không phải giới hạn tối ưu (mục 7).
5. Các số "GT3 thật" ở cột tham chiếu là ước lượng, chưa trích nguồn.
6. **Layout map không phải 1:1 về hình học.**
   - `scripts/silverstone_blocks.json` gồm 23 block (18 cua là cung tròn, 5 đoạn thẳng). Cua mang tên các cua thật, nhưng **bán kính là đoán tay** (25–95 m ở map half).
   - `solve_silverstone_layout.py` chỉ tối ưu độ dài và góc để vòng đua khép kín.
   - Map full là map half **nhân đều ×2.075**. Hệ quả:
     - cua chậm bị phóng to, ví dụ The Loop thành r ≈ 52 m, Village thành r ≈ 62 m, nên chạy nhanh hơn thực tế;
     - giữa các block không có đoạn chuyển tiếp độ cong (clothoid).
   - **Độ rộng đường thì gần đúng.** Map dùng 3 làn × 4.8 m = 14.4 m đều cả vòng. Các nguồn MotoGP/Tech3 ghi Silverstone rộng khoảng 15 m; Honda ghi 17 m, có thể là khu vực grid. Xe MetaDrive rộng ~1.85 m so với GT3 thật ~2.0 m, nên tỉ lệ xe/đường khá sát.
   - Người dùng quyết định không dựng lại map từ centerline thật (30/10). Trong báo cáo gọi đây là "layout xấp xỉ theo topology Silverstone".
7. **Ảnh hưởng của mô hình xe lên profile v\* map full.**
   - Không có cản gió: đoạn thẳng phẳng lì ở limiter 260 km/h.
   - Không có downforce: cua nhanh chậm hơn xe thật.
   - Phanh trần ~1.65 G, không tăng theo tốc độ: vùng phanh từ 260 km/h dài, 150 m cho 250 → 0.
   - T\* 124.0s gần với lap GT3 thật, nhưng đây là **các sai số bù trừ nhau**, không chứng minh mô hình đúng.
