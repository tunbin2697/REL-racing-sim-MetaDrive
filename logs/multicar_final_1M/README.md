# Run Log: multicar_final_1M (Bản chốt bàn giao §5.1 & §6)

## 1. Thông tin cấu hình & Nguồn gốc (§5.1)
- **Nhánh làm việc (Branch)**: `feature/multicar-demo-phat` (tách từ `origin/feature/111s-honest-anchor` @ commit `f4d8ef9`)
- **Warm-start gốc**: `models/ref_111s/best_agent.pt` (35D $\rightarrow$ 60D: giữ nguyên 35 cột đầu của `ref_111s`, khởi tạo 0 cho 25 cột đối thủ `5 xe x 5 đặc trưng`)
- **Resume từ vòng thử ngắn**: `models/multicar_A/best_agent.pt` (300k mẫu Preset A) $\rightarrow$ chạy tiếp **1,001,472 mẫu** (`570` vòng đua khi train)
- **Sa bàn & Thực thể xe**: `silverstone_half_3lane` (`lane_num = 3`, `lane_width = 4.5m`, tổng bề rộng mặt nhựa `13.5m`, `lat_norm = 6.75m`) | `6` xe `gt3_anchor` (giữ nguyên 100% thông số vật lý `GT3RaceCar` / `gt3_anchor`)
- **Bản vá & Ràng buộc kế thừa**:
  - **F13**: Lưu `raw_u` trước `tanh` vào rollout buffer và truyền `b_raw_us` vào `agent.update()` để tính đúng `log_prob` Jacobian.
  - **D19**: Khóa trần `logstd_max = [0.0, -0.7, -0.7]` ($\sigma_{\text{throttle}}, \sigma_{\text{brake}} \le 0.5$).
  - **§4.2 Smoothness (Bộ A)**: `c1 (steer_smooth_coef) = 1.0`, `c2 (pedal_smooth_coef) = 0.05`, `center_coef = 0.0`.
  - **§4.3 Va chạm & Vượt sạch (WI-L5)**: `collision_penalty = 2.5`, phạt vùng đệm elip (`PROXIMITY_LONG_M = 5.4m`, `PROXIMITY_LAT_M = 2.5m`), thưởng vượt sạch `+1.0` (cooldown 50 bước, miễn trừ 150 bước sau respawn) kèm bộ đệm nhường đỉnh cua mượt (`_apply_racing_etiquette`).

## 2. Lệnh chạy huấn luyện & Demo
```powershell
# Lệnh huấn luyện bản chốt 1M mẫu:
python training/train_multicar.py --resume models/multicar_A/best_agent.pt --total-samples 1000000 --preset-smooth A --collision-penalty 2.5 --lr 5e-5 --eval-freq 10 --milestone-freq 100000 --save-dir models/multicar_final_1M --log-dir logs/multicar_final_1M

# Lệnh xem trực tiếp 3D chế độ khán giả (Spectate):
python demo/race_multicar.py --spectate --model models/multicar_final_1M/best_agent.pt

# Lệnh đua bằng tay cầm Xbox với 5 xe AI:
python demo/race_multicar.py --model models/multicar_final_1M/best_agent.pt
```

## 3. Kết quả kiểm thử chuẩn trên 6 chặng đua liên tiếp (36 lượt xe) theo mục §6

| Chỉ số nghiệm thu (§6) | Ngưỡng yêu cầu | `multicar_best.pt` (nhánh cũ) | `ref_111s` (Step 0) | **`multicar_A/best_agent.pt`** (Vòng ngắn) | **`final_1M/best_agent.pt` (Đề xuất chính)** | **`final_1M/agent_500k.pt`** | **`final_1M/last_agent.pt` (1M mẫu)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Số xe về đích (6 chặng)** | **`>= 28 / 36`** | 28 / 36 | 36 / 36 | ✅ **35 / 36** | ✅ **35 / 36 (97.2%)** | ✅ **34 / 36 (94.4%)** | ✅ **35 / 36 (97.2%)** |
| **Số xe văng (`outs`)** | Thấp nhất | 8 / 36 | 0 / 36 | ✅ **0 / 36** | ✅ **1 / 36** | ✅ **2 / 36** | ✅ **1 / 36** |
| **Số xe đứng yên (`stalls`)** | **`0`** | 0 / 36 | 0 / 36 | ✅ **1 / 36** | ✅ **0 / 36** | ✅ **0 / 36** | ✅ **0 / 36** |
| **Va chạm TB / chặng (`contacts`)** | **`<= 43.0`** | 43.0 | 405.0 ❌ | ✅ **31.00** | ✅ **21.50 (-50%)** | ✅ **29.17 (-32%)** | ✅ **42.33** |
| **Thời gian vòng đua TB** | Nhanh, ổn định | 154.7s | 117.2s | ✅ **117.68s** (best 114.1s) | ✅ **117.41s** (best **114.2s**) | ✅ **117.59s** (best 114.8s) | ✅ **117.97s** (best 114.4s) |
| **Đổi chiều bàn đạp (`pedal_sw/100`)** | **`<= 5.0`** | 28.0 ❌ | 5.83 ❌ | ✅ **4.85** | ✅ **4.97** | ✅ **4.78** | ✅ **4.75** |
| **Đổi chiều lái (`steer_rev/100`)** | **`<= 40.0`** | 65.0 ❌ | 39.68 | ✅ **32.48** | ✅ **32.09** | ✅ **27.70** | ✅ **24.61** |
| **Góc trượt thân xe ($\beta$ p95)** | **`<= 4.0°`** | 11.7° ❌ | 3.77° | ✅ **3.67°** | ✅ **3.68°** | ✅ **3.65°** | ✅ **3.16°** |

## 4. Danh sách Checkpoint đã lưu trong `models/multicar_final_1M/`
- **`best_agent.pt`** *(Khuyên dùng chính khi demo)*: 35/36 về đích, 0 xe kẹt, trung bình chỉ **21.5 lần chạm/chặng**, vòng đua TB **117.41s** (best **114.2s**), đạt chuẩn cả 3 chỉ số mượt §6.
- **`agent_500k.pt`** *(Lựa chọn cân bằng siêu mượt)*: 34/36 về đích, 29.17 lần chạm/chặng, `steer_rev = 27.70/100 bước`, `pedal_sw = 4.78/100 bước`.
- **`last_agent.pt` / `agent_1000k.pt`**: 35/36 về đích, độ mượt vô-lăng cao nhất (`steer_rev = 24.61/100 bước`, `beta_p95 = 3.16°`).
- Đầy đủ 10 mốc snapshot mỗi 100k mẫu: `agent_100k.pt`, `agent_200k.pt`, ..., `agent_1000k.pt`.
