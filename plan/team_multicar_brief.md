# Đặc tả cho team: demo đua nhiều xe "đẹp như tay đua"

> **Tạo**: 31/10/2026 · **Người đọc**: các bạn làm multicar · **Thời hạn**: 2–3 ngày, trước buổi bảo vệ
> **Quyết định gốc**: D40 trong `plan/plan.md`

---

## 1. Bối cảnh trong một đoạn

- Phần **học thuật** của đồ án dựa vào bản **111s** (`ee99f9d`, `models/ref_111s/best_agent.pt`): 1 xe, map half, chạy nhanh hơn ghost giải tích trên cùng simulator. Phần này đã xong, các bạn không cần đụng tới.
- Phần **demo nhiều xe** là của các bạn. Mục tiêu là **cho người xem thấy một cuộc đua giống thật**, không phải để chứng minh điều gì. Cứ làm cho vui, nhưng đừng làm cho có.
- Map full 1:1 (Phase 2) **đóng băng**. Không port multicar sang map full.

## 2. Mục tiêu

Trên khung multicar sẵn có của nhánh `origin/feature/multicar-selfplay` (6 xe, map half 2.84 km, **3 làn × 4.5 m = 13.5 m**), làm cho các xe **đua giống tay đua**:
- lái mượt, không lắc vô lăng trên đoạn thẳng;
- ga và phanh dứt khoát: full ga trên đoạn thẳng, phanh một lần rõ ràng trước cua, nhả phanh dần khi vào cua; **không** đổi ga↔phanh liên tục;
- không drift hay văng đuôi vô cớ;
- vượt xe có chủ đích, ít húc nhau, ít văng ra ngoài.

**Không cần nhanh nhất.** Lap chậm hơn `multicar_best.pt` (TB 154.7s) vài giây mà đẹp hơn thì vẫn là đạt.

## 3. Phạm vi được tự quyết

| Hạng mục | Được làm | Lưu ý |
| :-- | :-- | :-- |
| Reward | **Tự do thiết kế** | Mỗi lần thử chỉ đổi **một nhóm** thay đổi, để còn biết cái gì có tác dụng |
| Xe | **Được tự build** (sửa thông số trong `game_env/vehicle.py` của nhánh) | Xem §5.4. Khuyên giữ xe hiện tại trừ khi thật sự cần |
| Số xe, grid, số vòng | Tùy | Đo FPS trước khi hứa số step |
| Siêu tham số | Tùy | Bộ hiện tại của nhánh (lr 1e-4, γ 0.995, ent 0.001) đã từng cho racing line tốt khi fine-tune |
| Racing line / profile tối ưu trong reward hoặc obs | Được, vì đây là demo | Ghi rõ trong README của run: "có dùng thông tin tối ưu" |

**Không được**: sửa code hay model của nhánh chính (`feature/silverstone-phase2`) và của bản 111s; ghi đè `models/multicar_best.pt` (luôn lưu ra thư mục mới).

## 4. Khởi đầu khuyên dùng (đã có bằng chứng từ dự án)

### 4.1 Chống giật lái và giật pedal: dùng **phạt trong reward**, đừng dùng ràng buộc vật lý
- Dự án đã thử: lọc EMA tay lái, giới hạn góc lái và tốc độ xoay theo tốc độ xe, quán tính ga/phanh (`ExtendedVehiclePhysics`, commit `a79aee7`). **Agent vẫn bang-bang.** Lý do: ràng buộc chỉ thu hẹp không gian action, agent vẫn đánh kịch biên mới; còn bộ lọc thì agent học cách bù ngược. Chỉ phạt trong reward mới đổi được mục tiêu (`knowledge/rl_debugging_findings.md` F23).
- Reward multicar hiện có `r_steer_smooth = −0.10·Δlái²`. Mức này **quá nhẹ** (tối đa khoảng −0.003/bước, so với thưởng tiến độ khoảng +0.1/bước).
- Đề xuất thêm hai cờ, mặc định giữ hành vi cũ:
  - `--steer-smooth-coef` (mặc định 0.10) → `r_steer_smooth = −c·Δlái²`
  - `--pedal-smooth-coef` (mặc định 0.0) → `r_pedal_smooth = −c·Δpedal²`, với pedal = ga − phanh
- Chạy song song **2 bộ**, mỗi bạn một bộ, warm-start từ `models/multicar_best.pt`:
  - **Bộ A**: `--steer-smooth-coef 1.0 --pedal-smooth-coef 0.05`
  - **Bộ B**: `--steer-smooth-coef 0.5 --pedal-smooth-coef 0.02` (nhẹ hơn, đề phòng xe rụt rè khi có đối thủ)
- Vì sao phạt Δ² không cấm phanh gấp: phanh đúng kiểu tay đua khoảng 20 lần đổi hết cỡ mỗi vòng chỉ mất khoảng −4 điểm/vòng ở bộ A; còn kiểu đổi liên tục (khoảng 290 lần/vòng) mất hàng chục điểm. Phạt bình phương còn làm nhả phanh dần rẻ hơn nhả đột ngột.
- Cơ sở: CAPS (Mysore et al., ICRA 2021); bang-bang là hiện tượng chung của RL điều khiển liên tục (Seyde et al., NeurIPS 2021). GT Sophy cũng tự thừa nhận điều khiển của nó kém mượt hơn tay đua. Chi tiết: `knowledge/related_work_racing_rl.md` §5.

### 4.2 Các ứng viên cho lần thử tiếp theo (chỉ thêm khi 4.1 đã ổn)
| Ứng viên | Khi nào dùng | Ghi chú |
| :-- | :-- | :-- |
| `r_center = −c·(lat/half_width)²` kiểu 111s | Xe hay cắt cua phía trong, chạy sát mép | 111s dùng c = 0.04 với nửa bề rộng 4.5 m. Ở map 13.5 m nên chuẩn hóa theo 6.75 m. Nhược điểm: tay đua thật không chạy giữa đường, nên xe sẽ bớt dùng hết bề rộng |
| Phạt trượt lốp kiểu GT Sophy | Còn drift, văng đuôi | Ví dụ phạt khi góc trượt β > 5°, tăng theo tốc độ |
| Thưởng vượt sạch, phạt va chạm đối xứng | Xe húc nhau hoặc không dám vượt | Thiết kế có sẵn ở `plan/plan.md` WI-L5. GT Sophy ghi nhận: phạt va chạm quá mạnh làm agent rụt rè |

### 4.3 Quy trình mỗi lần thử
1. Fine-tune ngắn khoảng 200–300k samples.
2. Xem bằng mắt: `python demo/race_multicar.py --spectate --model <checkpoint>`.
3. Đo chỉ số ở §6.
4. Đạt thì chạy dài khoảng 1M samples; không đạt thì chỉnh **một** hệ số rồi thử lại.

## 5. Lưu ý kỹ thuật

1. **Đo FPS 6 xe trước** khi lên lịch train. Mỗi bước 6 xe tốn hơn 1 xe nhiều.
2. **Sim không tất định**: cùng seed, cùng policy vẫn có thể văng ở chỗ khác. Đánh giá bằng nhiều cuộc đua, không bằng 1 lần.
3. **Lưu metadata** của mỗi run: commit, cờ đã dùng, warm-start từ đâu. Ghi vào README ngắn trong thư mục log của run.
4. **Nếu tự build xe**:
   - Xe nhanh hơn thì khó lái mượt hơn: nhiễu gia tốc ngang khi giật lái tăng theo v². Ở map full, xe GT3 thật chạy khoảng 144 km/h trung bình bị drift rõ (F20), còn 111s chạy 88 km/h thì nhìn mượt (F23).
   - Xe hiện tại (GT3RaceCar) có ma sát cầu trước 0.9, nhỏ hơn cầu sau 1.35 (bug F16). Bug này làm xe understeer, đuôi không văng, nên **dễ lái đẹp hơn**. Sửa bug là xe nhạy hơn và khó hơn.
   - Đổi xe thì phải fine-tune lâu hơn, vì policy cũ học trên động lực học cũ.
   - Kết luận: chỉ build xe mới nếu có thời gian dư sau khi §4.1 đã đạt.

## 6. Tiêu chí đạt

| Chỉ số | Cách đo | Mục tiêu |
| :-- | :-- | :-- |
| Nhìn bằng mắt | 2–3 cuộc đua `--spectate`, quay video ngắn | Người không biết RL xem thấy "giống đua thật" |
| Pedal đổi ga↔phanh /100 bước | Script đo phong cách lái (trợ lý đang viết, `scripts/drive_style.py`) hoặc tự đếm từ telemetry | ≤ 5 (ghost: 0.5; 111s: 4.7) |
| Lái đổi chiều /100 bước | như trên | ≤ 40 (111s: 60; ghost: 16) |
| Góc trượt β p95 | như trên | ≤ 4° |
| Số xe về đích / số lần chạm mỗi cuộc đua | Bảng eval sẵn có của nhánh (README §6.2) | Không tệ hơn `multicar_best.pt` (28/36 về đích, 43 lần chạm) |

## 7. Báo kết quả

Mỗi run đạt hoặc gần đạt, gửi:
- commit và lệnh chạy đầy đủ;
- thư mục log (`training_metrics.json`) và checkpoint tốt nhất;
- bảng chỉ số §6;
- 1 video ngắn (30–60 s) cảnh vào cua và vượt xe.

## 8. Tài liệu tham khảo trong repo

- `knowledge/rl_debugging_findings.md`: F20 (giật lái, drift ở tốc độ cao), F23 (vì sao 111s nhìn mượt; vì sao ràng buộc vật lý không chống được bang-bang).
- `knowledge/related_work_racing_rl.md`: GT Sophy, Fuchs, và §5 (bang-bang, CAPS, racing line trong RL).
- `knowledge/ablation_inventory.md`: R9, R10 (các run multicar cũ).
- `plan/plan.md`: WI-L5 (cơ chế đua nhiều xe), D39–D40.
