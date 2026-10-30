# Bàn giao: demo đua nhiều xe

> **Người đọc**: các bạn trong team phụ trách multicar · **Cập nhật**: 31/10/2026
> **Nhánh xuất phát**: `feature/111s-honest-anchor`. Mỗi bạn tạo nhánh làm việc riêng từ đây (ví dụ `feature/multicar-demo-<tên>`), **không** commit thẳng lên nhánh xuất phát.
> **Liên hệ**: thay đổi nào ngoài phạm vi ở §3 thì hỏi chủ dự án trước khi làm.

---

## 1. Bối cảnh

- Phần học thuật của đồ án đã xong. Kết quả chính là **mô hình tham chiếu `ref_111s`** (`models/ref_111s/best_agent.pt`; tên lấy từ lap 111.4s lúc train). Model này là bản tốt nhất và được ưng ý nhất từ đầu dự án.
- Trên map Silverstone half, cùng luật 4 bánh và cùng vật lý xe, `ref_111s` chạy **114.8s** (xuất phát đứng) và **109.4s** (flying). Ghost tối ưu đã hiệu chỉnh chạy 118.2s và 112.8s (`plan/plan.md` §1).
- Phần còn thiếu là **demo nhiều xe**, giao cho các bạn. Mục tiêu là để người xem thấy **một cuộc đua giống thật**. Không cần chứng minh gì thêm; làm cho vui, nhưng phải đẹp.

Xem nhanh model tham chiếu chạy cùng ghost:
```powershell
python demo/spectate_ai.py --model models/ref_111s/best_agent.pt --map silverstone --vehicle gt3_anchor --ghost
```

## 2. Mục tiêu

Nhiều xe (gợi ý 4–6) đua trên Silverstone half, nhìn giống tay đua thật:
- lái mượt, không lắc vô lăng trên đoạn thẳng;
- ga và phanh dứt khoát: full ga trên đoạn thẳng, phanh rõ ràng trước cua, nhả phanh dần khi vào cua, **không** đổi ga↔phanh liên tục;
- không drift hay văng đuôi vô cớ;
- vượt xe có chủ đích, ít húc nhau, ít văng ra ngoài.

**Không cần nhanh nhất.** Chậm hơn `ref_111s` vài giây mà đẹp hơn vẫn là đạt.

## 3. Phạm vi

### 3.1 Giữ nguyên
| Hạng mục | Ở đâu | Vì sao |
| :-- | :-- | :-- |
| Hình học map Silverstone half (chiều dài, các cua) | `game_env/silverstone_map.py`, `scripts/silverstone_blocks.json` | Để so sánh được với `ref_111s` và ghost |
| Thông số xe | `GT3RaceCar` và `gt3_anchor` trong `game_env/vehicle.py` | Cùng vật lý với `ref_111s` |
| Cách tính profile optimal và ghost | `game_env/benchmark.py`, `agents/optimal_agent.py`, envelope của `gt3_anchor` | Đã hiệu chỉnh kỹ (F24) |

### 3.2 Được tự quyết
| Hạng mục | Lưu ý |
| :-- | :-- |
| **Hàm thưởng phạt** | Tùy ý, miễn phù hợp đua xe. Mỗi lần thử chỉ đổi **một nhóm** thay đổi, để còn biết cái gì có tác dụng |
| **Siêu tham số** | Được tune lại |
| **Nới rộng đường / thêm làn** | Được. Xem §5.2: phải tính lại profile optimal bằng đúng code hiện có, không sửa envelope |
| Số xe, kiểu grid, số vòng | Tùy. Đo FPS trước khi hứa số step |
| Racing line hoặc profile optimal trong obs/reward | Được, vì đây là demo; ghi rõ "có dùng thông tin tối ưu" trong README của run |

### 3.3 Phải hỏi trước
Đổi kiến trúc mạng hoặc thuật toán (ví dụ PPO → SAC), đổi thông số hay profile xe, đổi hình học map, sửa code benchmark/ghost, sửa `models/ref_111s/`.

## 4. Gợi ý khởi đầu (đã có bằng chứng trong dự án)

### 4.1 Mã multicar có sẵn
Nhánh `origin/feature/multicar-selfplay` (`7ab74a5`) đã có đua 6 xe chạy được:
- `game_env/multicar_env.py`, `training/train_multicar.py`, `demo/race_multicar.py`, `game_env/gamepad.py`;
- self-play PPO chung tham số, obs 65D, đường 3 làn × 4.5 m = 13.5 m;
- model `models/multicar_best.pt`: 28/36 lượt về đích, lap TB 154.7s.

Cách nhanh nhất là port 4 file này sang nhánh làm việc mới, cho dùng xe `gt3_anchor` và map của nhánh xuất phát. Các khác biệt cần xử lý khi port: §5.3.

### 4.2 Chống giật lái và giật pedal: phạt trong reward, đừng dùng ràng buộc vật lý
- Dự án đã thử lọc EMA tay lái, giới hạn góc lái và tốc độ xoay theo tốc độ xe, quán tính ga/phanh (`ExtendedVehiclePhysics`, commit `a79aee7`). **Agent vẫn bang-bang.** Lý do: ràng buộc chỉ thu hẹp không gian action nên agent vẫn đánh kịch biên mới; còn với bộ lọc thì agent học cách bù ngược. Chỉ phạt trong reward mới đổi được mục tiêu (`knowledge/rl_debugging_findings.md` F23).
- Reward hiện tại có `r_steer_smooth = −0.10·Δlái²`, mức này **quá nhẹ** (tối đa khoảng −0.003/bước so với thưởng tiến độ khoảng +0.1/bước).
- Gợi ý thêm hai hệ số, mặc định giữ hành vi cũ: `r_steer_smooth = −c₁·Δlái²` và `r_pedal_smooth = −c₂·Δpedal²` (pedal = ga − phanh). Hai bộ để chạy song song:
  - **A**: c₁ = 1.0, c₂ = 0.05;
  - **B**: c₁ = 0.5, c₂ = 0.02 (nhẹ hơn, đề phòng xe rụt rè khi có đối thủ).
- Phạt Δ² không cấm phanh gấp. Phanh kiểu tay đua (khoảng 20 lần đổi hết cỡ mỗi vòng) ở bộ A chỉ mất khoảng −4 điểm/vòng; kiểu đổi liên tục (khoảng 290 lần/vòng) mất hàng chục điểm. Phạt bình phương còn làm việc nhả phanh dần rẻ hơn nhả đột ngột.
- Cơ sở: CAPS (Mysore et al., ICRA 2021); bang-bang là hiện tượng chung của RL điều khiển liên tục (Seyde et al., NeurIPS 2021); GT Sophy cũng thừa nhận điều khiển của nó kém mượt hơn tay đua (`knowledge/related_work_racing_rl.md` §5).

### 4.3 Ứng viên cho các lần thử sau
| Ứng viên | Khi nào | Ghi chú |
| :-- | :-- | :-- |
| `r_center = −c·(lat/half_width)²` (có sẵn cờ `--center-coef` trong `train_racing.py`) | Xe hay cắt cua phía trong, chạy sát mép | `ref_111s` dùng c = 0.04 trên đường 9 m. Nhược điểm: tay đua thật không chạy giữa đường |
| Phạt trượt lốp kiểu GT Sophy | Còn drift | Ví dụ phạt khi góc trượt β > 5°, tăng theo tốc độ |
| Thưởng vượt sạch, phạt va chạm đối xứng | Húc nhau hoặc không dám vượt | Thiết kế ở `plan/archive/plan_v12_full.md` WI-L5. Theo GT Sophy, phạt va chạm quá mạnh làm agent rụt rè |

### 4.4 Quy trình mỗi lần thử
1. Fine-tune ngắn, khoảng 200–300k samples.
2. Xem bằng mắt bằng demo multicar ở chế độ khán giả.
3. Đo các chỉ số ở §6.
4. Đạt thì chạy dài khoảng 1M samples; không đạt thì chỉnh **một** hệ số rồi thử lại.

## 5. Lưu ý kỹ thuật

### 5.1 Chung
- **Sim không tất định**: cùng seed, cùng policy vẫn có thể văng ở chỗ khác. Đánh giá bằng nhiều cuộc đua.
- **Đo FPS** với số xe thật trước khi lên lịch.
- Mỗi run ghi một README ngắn trong thư mục log: commit, lệnh đầy đủ, warm-start từ đâu.
- Không ghi đè model có sẵn; luôn dùng `--save-dir` mới.

### 5.2 Nới đường / thêm làn
- Bề rộng đường đặt trong `game_env/track_presets.py` (`lane_num`, `lane_width`). Nên thêm **preset mới** (ví dụ `silverstone_half_3lane`) thay vì sửa preset `silverstone_half`, để `ref_111s` và ghost hiện tại vẫn chạy đúng.
- Đổi bề rộng thì racing line và T\* thay đổi. Cache profile đã được khóa theo bề rộng đường từ 31/10, nên với bề rộng mới benchmark sẽ tự tính lại và ghi ra file cache riêng. **Không sửa envelope** của `gt3_anchor`.
- Obs của xe chuẩn hóa khoảng cách tới mép theo nửa bề rộng đường, nên model train trên đường 9 m sẽ thấy phân phối obs khác trên đường rộng hơn. Warm-start được, nhưng cần fine-tune.

### 5.3 Port multicar từ `origin/feature/multicar-selfplay`
- Khung đó dùng obs ego 40D cộng thông tin đối thủ (65D), map 13.5 m, xe `GT3RaceCar`. Nhánh xuất phát dùng obs ego 35D.
- Warm-start từ `ref_111s` (35D) sang 65D cần nạp trọng số "nới rộng": copy các cột ứng với 35 chiều chung, khởi tạo 0 cho các chiều mới. Khung cũ đã có cách nạp 40D → 65D; làm tương tự.
- Hoặc warm-start luôn từ `multicar_best.pt` của khung cũ. Cách này nhanh hơn, nhưng xe và obs khác nhánh xuất phát, nên phải ghi rõ.

### 5.4 Nếu cần đổi xe (phải hỏi trước, §3.3)
- Xe nhanh hơn thì khó lái mượt hơn: nhiễu gia tốc ngang khi giật lái tăng theo v². Ở map 1:1, xe chạy trung bình khoảng 144 km/h bị drift rõ (F20); `ref_111s` chạy 88 km/h nhìn mượt (F23).
- `GT3RaceCar` có ma sát cầu trước 0.9 nhỏ hơn cầu sau 1.35 (bug F16). Bug này làm xe understeer, đuôi không văng, nên **dễ lái đẹp hơn**.

## 6. Tiêu chí đạt

| Chỉ số | Cách đo | Mục tiêu |
| :-- | :-- | :-- |
| Nhìn bằng mắt | 2–3 cuộc đua ở chế độ khán giả, quay video ngắn | Người không biết RL xem thấy "giống đua thật" |
| Pedal đổi ga↔phanh /100 bước | Đếm từ telemetry mỗi xe | ≤ 5 (ghost: 0.5; `ref_111s`: 4.7) |
| Lái đổi chiều /100 bước | như trên | ≤ 40 (ghost: 16; `ref_111s`: 60) |
| Góc trượt β p95 | như trên | ≤ 4° (`ref_111s`: 3.5°) |
| Số xe về đích, số lần chạm mỗi cuộc đua | Bảng eval của khung multicar | Không tệ hơn `multicar_best.pt` (28/36 về đích, 43 lần chạm) |

Cách đếm các chỉ số lái giống F20 trong `knowledge/rl_debugging_findings.md`: đổi dấu pedal (bỏ vùng |pedal| < 0.05), đổi chiều của Δlái, β = atan2(v_y, |v_x|) khi tốc độ > 40 km/h.

## 7. Lịch gợi ý (2–3 ngày)

| Mốc | Việc |
| :-- | :-- |
| Ngày 1 | Port khung multicar sang nhánh làm việc, chạy được với `gt3_anchor`; đo FPS; thêm 2 hệ số phạt mượt |
| Ngày 1 tối | 2 bạn chạy song song bộ A và bộ B, mỗi bộ khoảng 1M samples |
| Ngày 2 | Xem demo, đo §6, chọn bộ tốt hơn; nếu cần thì thêm một ứng viên ở §4.3 rồi chạy tiếp |
| Ngày 3 | Chốt model, quay video, gửi kết quả (§8) |

## 8. Báo kết quả

Mỗi run đạt hoặc gần đạt, gửi:
- nhánh, commit và lệnh chạy đầy đủ;
- thư mục log (`training_metrics.json`) và checkpoint tốt nhất;
- bảng chỉ số §6;
- 1 video ngắn (30–60 giây) cảnh vào cua và vượt xe.

## 9. Tồn đọng đã biết

- Bug F16 (μ cầu trước 0.9) còn trong `GT3RaceCar`. Cố ý giữ để cùng vật lý với `ref_111s`.
- MetaDrive coi phần ngoài mặt nhựa như mặt đường thường (chưa chắc có ma sát thấp hơn), nên cắt cua chỉ bị phạt qua reward (F22).
- Map half có bề rộng đều suốt vòng, trong khi Silverstone thật rộng 12–17 m.
- `ref_111s` từng được train dưới luật kerb ảo (4 góc thân xe), giờ eval dưới luật 4 bánh: 18/20 vòng sạch, văng ở 2637 m và 364 m.

## 10. Tài liệu trong repo

| File | Dùng khi |
| :-- | :-- |
| `plan/plan.md` | Trạng thái dự án, kết quả chính |
| `plan/requirements/PROJECT_SPECIFICATION.md` | Đặc tả gốc của đề tài |
| `plan/requirements/SPECIFICATION_INTENDED_VIOLATION.md` | Những điểm lệch có chủ đích so với đặc tả |
| `knowledge/rl_debugging_findings.md` | F20 (giật lái, drift), F22 (cắt cua), F23 (vì sao `ref_111s` nhìn mượt), F24 (so với ghost) |
| `knowledge/vehicle_and_engine_physics.md` | Vật lý xe, hiệu chỉnh, benchmark |
| `knowledge/related_work_racing_rl.md` | GT Sophy (reward, mistake learning, đua nhiều xe), Fuchs, paper về điều khiển mượt |
| `knowledge/ablation_inventory.md` | R9, R10: các run multicar cũ, cách lấy lại artifact |
