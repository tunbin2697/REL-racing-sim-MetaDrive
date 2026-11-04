Tôi chưa viết report mẫu, vì `report/REPORT.md` (410 dòng) đã có trên nhánh neo từ 01/11 và đã gộp vào `main` qua PR #6. Thay vào đó tôi kiểm tra cấu hình train của 111s. Cả report lẫn `knowledge/` đều mô tả sai một số điểm.

## 1. Code của `ee99f9d` đúng là code đã train ra bản 111s

Bằng chứng:
- **Model và code cùng một commit.** Model trước đó (`b8e0288`) có action 2 chiều (1.825 MB); model của `ee99f9d` có action 3 chiều (1.828 MB). Chỉ code từ `ee99f9d` trở đi mới có action 3 chiều.
- **Tái hiện được.** Tôi chạy eval ngắn trên đúng bản code đó (trích riêng vào thư mục tạm): 3/3 vòng về đích, 98.5% quãng đường, tốc độ TB 89.1 km/h. Log train ghi 98.5% và 89.3 km/h.
- **Là một run liên tục từ step 0.** Các lần eval cách đều nhau 10.240 step, chỉ có một khoảng 8.192 ở cuối, và không có `resume_mark`. Nếu có resume thì nhịp eval sẽ lệch.
- **Về các commit sau:** commit trong git không bị đè. Các commit sau (`e0738e8` F1/F2, `958c11f` F13, `600f19e`…) chỉ tạo ra code khác, còn `ee99f9d` vẫn nguyên.

## 2. Cấu hình train 111s

| Tham số | Giá trị | Nguồn xác minh |
|---|---|---|
| Tổng step | 4.000.000 (dừng ở 4.001.792 = 1954 rollout) | Suy từ log |
| Rollout / batch / epoch | 2048 / 64 / 10 | Code |
| lr | 3e-4 **cho cả actor và critic** | Code. Commit này đổi từ `0.2×lr` / `1.5×lr` |
| γ / λ / clip / entropy | 0.99 / 0.90 / 0.15 / 0.01 cố định, không decay | Code |
| Khác | `vf_coef` 0.5, clip gradient 0.5, Adam eps 1e-5, chuẩn hóa advantage theo batch, clip value loss ±10 | Code |
| Mạng | 2 lớp ẩn 256 tanh, init trực giao | Code |
| Chính sách | Gaussian nén tanh, init logstd −0.8, **clamp trong forward `[-2.3, 0.5]`**, bias init ga 0.5, phanh −0.8 | Code |
| Eval khi train | Mỗi 5 update (10.240 step), 3 episode, deterministic | Suy từ log (crash_rate là bội của 1/3) |
| Lưu `best_agent.pt` | Khi completion ≥ 98% và lap nhanh hơn hoặc tốc độ cao hơn kỷ lục | Code |
| Seed | 42 (mặc định) | **Không ghi trong log**, chưa xác minh được |

**Reward** (khớp mô tả F9):
- Thưởng tiến độ `0.05·max(0, Δs)` cho mỗi bước.
- Phạt Δlái² hệ số 0.10.
- `r_center = −0.04·(lat/4.5)²`.
- Phạt đứng yên −0.15 khi v < 10 km/h.
- Chạm hoặc ra khỏi đường: kết thúc episode, phạt `−10 − 20·(v/120)²`.
- Kẹt dưới 2 km/h trong 400 bước (150 bước nếu đã qua 20 m): kết thúc, không phạt thêm.
- Về đích: `100 +` bậc thang theo thời gian (+50 nếu dưới 140 s, rồi giảm 10 mỗi 20 s).
- **Không có** phạt thời gian.

**Obs 35D:**
- 10 chiều động học: v, vₓ, v_y, yaw rate, lái hiện tại, lái bước trước, pedal bước trước, cosΔθ, sinΔθ, lệch ngang.
- 5 chiều về đường: khoảng cách tới mép trái và phải, cờ lấn lề, s/L, độ cong.
- 20 chiều lookahead: 5 điểm ở 10/20/35/50/70 m.

## 3. Những chỗ `report/REPORT.md` đang sai so với bản 111s

| Mục | Report viết | Thực tế của `ref_111s` |
|---|---|---|
| §3.5 thưởng tiến độ | $c_{prog} = 0.15$ | **0.05 mỗi mét** (0.15 là mức ước lượng mỗi bước ở khoảng 3 m/bước, nhầm đơn vị) |
| §3.5 phạt đứng yên | −0.05 khi v < 5 km/h | **−0.15 khi v < 10 km/h** |
| §3.5 phạt văng đường | −20 cố định | **−10 đến −30 theo v²** |
| §3.5 thưởng về đích | $50\cdot\exp(-(T-110)/25)$ | **100 + bậc thang**. Hàm mũ là của Phase 1 và 2 |
| §3.5 thiếu | – | Điều kiện kết thúc khi kẹt |
| §3.4 obs | 11 + 4 + 20 chiều, lookahead 10/25/45/70/100 m, có a_x, a_y, roll, pitch | **10 + 5 + 20**, lookahead 10/20/35/50/70 m, không có gia tốc/roll/pitch (đó là obs 40D của nhánh khác) |
| §3.6 | δ_reg = 1e-6 | Code clamp `1 − a²` tối thiểu **1e-4** |
| §6.4 mục 1 | Clamp `max=0.0` | `ref_111s` có `max=0.5`. 0.0 là của `2055469`/`0180988` |
| §6.4 mục 2–4 | Trình bày F2, bias phanh −1.6 như cơ chế chung | Đó là **bản vá Phase 1**. `ref_111s` được train **không có** F1, F2/F13 và bias −1.6 |
| Bảng 4, cột "Tổng bước" | **2.15M** cho `ref_111s`, seed 43 và `no_rcenter` | Cả ba đều chạy **4.0M** |

**Cũng cần nói rõ trong báo cáo:** kết quả chính của `ref_111s` đạt được khi PPO **chưa có** các bản vá F1/F13 mà §6.4 mô tả. Điều này làm câu chuyện mạnh hơn (công thức đơn giản đã đủ), nhưng hiện report đang ngầm nói ngược.

Bảng F9 trong `knowledge/rl_debugging_findings.md` cũng thiếu dòng learning rate (actor/critic từng khác nhau) và dòng lookahead.

## 4. Những điều chưa xác minh được
- **Seed của run 111s** (log không ghi).
- **Lệnh train đã gõ:** tôi suy ra `--total-steps 4000000` và `eval-freq 5`, nhưng không có bản ghi.
- **Code lúc train có khác `ee99f9d` không:** chưa có commit nào ghi lại working tree lúc đó. Nhưng eval tái hiện khớp nên khả năng lệch thấp.
- **Số liệu `no_rcenter` seed 43 trong Bảng 2 của report** (tốc độ 88.75 km/h, lệch line 2.56 m) khác eval tôi chạy ngày 01/11 (83.9 km/h, 3.48 m). Có thể bạn đã eval lại với seed khác (report ghi seed 100–119). Cần đối chiếu với file JSON gốc.

## 5. Đề xuất
Sửa `report/REPORT.md` §3.4–3.6, §6.4 và Bảng 4 theo bảng ở mục 3, thêm một mục "Cấu hình train của `ref_111s`" dùng bảng ở mục 2, và bổ sung hai dòng thiếu vào bảng F9.

Lưu ý khi làm: các file đó nằm trên nhánh neo, còn thư mục hiện ở `feature/multicar-demo-phat-bk-finetune` và có `gamepad.py`, `vehicle.py` đang sửa dở. Tôi không đụng gì vào đó trong lượt này. Nếu bạn đồng ý, tôi sẽ chuyển sang nhánh neo để sửa, nhưng cần bạn cất hoặc commit hai file kia trước. Việc đó cũng cần bạn cho phép, vì tôi chỉ được thao tác git khi bạn yêu cầu.