# TÀI LIỆU TOÀN DIỆN: DIỄN GIẢI DỰ ÁN, ĐỐI QUÁT CÁC NHÁNH & INSIGHT THỰC NGHIỆM
*(Comprehensive Project Explanation, Cross-Branch Experimental Audit & Defense Handbook)*

Tài liệu này được lập ra sau khi rà soát và kiểm tra chéo **100% tất cả các nhánh Git, các commit lịch sử và toàn bộ tệp log huấn luyện JSON** trong toàn bộ kho lưu trữ.

---

## PHẦN I: KHÔNG GIAN QUAN SÁT & CẤU TRÚC KỸ THUẬT MÔ HÌNH CHÍNH

### 1. Không gian Quan sát 35 Chiều Thuần Túy (`35D Pure Observation`)
Tác tử PPO Mô hình Chính **không nhận bất kỳ đường đua mẫu (Racing Line)** nào, chỉ nhận thông tin hình học tim đường và trạng thái xe:
* **Nhóm 1: Động thái tự thân xe (Ego Kinematics - 10 chiều):**
  1. `obs[0] = speed / 40.0`: Vận tốc tổng hợp chuẩn hóa.
  2. `obs[1] = vx / 40.0`: Vận tốc dọc thân xe.
  3. `obs[2] = vy / 10.0`: Vận tốc trượt ngang thân xe (Sideslip velocity).
  4. `obs[3] = yaw_rate / \pi`: Tốc độ quay góc hướng xe.
  5. `obs[4] = cur_steer`: Góc lái sau bộ Slew Rate Limiter ($\in [-1, 1] \iff \pm 20^\circ$).
  6. `obs[5] = last_steer`: Lệnh lái bước trước ($t-1$).
  7. `obs[6] = last_pedal`: Lệnh pedal ròng bước trước ($t-1$, sau bộ trộn ECU: $>0$ ga, $<0$ phanh).
  8. `obs[7] = \cos(\Delta\theta)`: Cosin sai lệch hướng đầu xe với tiếp tuyến tim đường.
  9. `obs[8] = \sin(\Delta\theta)`: Sin sai lệch hướng đầu xe với tiếp tuyến tim đường.
  10. `obs[9] = d_t / 4.5`: Độ lệch ngang từ trọng tâm xe tới tim đường ($w = 4.5\text{m}$).
* **Nhóm 2: Biên đường & Sa bàn (5 chiều):**
  11. `obs[10] = (4.5 - d_t) / 4.5`: Khoảng cách tới mép nhựa bên trái.
  12. `obs[11] = (4.5 + d_t) / 4.5`: Khoảng cách tới mép nhựa bên phải.
  13. `obs[12] = on_runoff`: Cờ nhị phân ($1.0$ khi lấn lề bê tông thoát hiểm).
  14. `obs[13] = s / 2838.7`: Tiến trình hoàn thành vòng đua ($s/L$).
  15. `obs[14] = \kappa(s) \cdot 50.0`: Độ cong tim đường tức thời tại vị trí xe.
* **Nhóm 3: Tầm nhìn xa đa điểm (Lookahead Horizon - 20 chiều):**
  - 5 waypoints phía trước tim đường tại các khoảng cách: $[5, 12, 25, 45, 75]\text{ m}$.
  - Mỗi điểm gồm 4 giá trị: $(\Delta x_{\text{rel}}, \Delta y_{\text{rel}}, \sin\Delta\theta_i, \kappa_i \cdot 50.0)$.
  - Tổng: $5 \times 4 = 20$ chiều.

### 2. Hàm thưởng & Bộ trộn ECU $C^1$
* **Hàm thưởng vi mô:** $r_t = 0.15 \Delta s_t - 0.04 (d_t/w)^2 - 0.10 (\Delta\delta_t)^2 + r_{\text{idle}} + R^{\text{terminal}}$.
* **Phạt đứng yên:** $r_{\text{idle}} = -0.15$ khi $v < 10\text{ km/h}$.
* **Thưởng cán đích gốc (Commit `ee99f9d`):** $R^{\text{finish}} = 100.0 + \text{Bonus bậc thang}$ ($+50$đ nếu $<140\text{s}$, $+40$đ nếu $<160\text{s}$, $+30$đ nếu $<180\text{s}$, $+20$đ nếu $<200\text{s}$, $+10$đ nếu $<220\text{s}$).
* **Bộ trộn ECU $C^1$:**
  $$u^{\text{brk}} = \frac{a^{\text{brk}}+1}{2}, \quad a_{\text{eff}}^{\text{brk}} = \max\left(0, \frac{u^{\text{brk}} - 0.05}{0.95}\right)$$
  $$\text{pedal} = u^{\text{thr}} \cdot (1 - a_{\text{eff}}^{\text{brk}}) - a_{\text{eff}}^{\text{brk}} \in [-1.0, 1.0]$$

---

## PHẦN II: TỔNG KIỂM TOÁN TẤT CẢ CÁC NHÁNH GIT & DỮ LIỆU LOG THỰC NGHIỆM

Bảng dưới đây trích xuất trực tiếp từ các tệp `training_metrics.json` và checkpoint thực tế nằm trong các commit lịch sử của từng nhánh:

| STT | Nhánh Git & Commit Head | Cấu hình & Biến số | Dữ liệu Log Đo được | Best Lap ghi nhận | Nguyên nhân Kỹ thuật & Insight Rút ra |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **R0** | `feature/racing-circuit-and-laptime-reward` (`d52d626`) | Sa bàn RacingMap có sẵn của MetaDrive, 2 làn, Xe mặc định, 35D/2D | $0.20\text{M}$ steps, Comp max $76.3\%$, $v_{\text{max}} = 75.1\text{ km/h}$ | DNF (Chưa hoàn thành) | Thử nghiệm pipeline PPO đầu tiên. Xe chưa cán đích do hàm thưởng sơ khai thiếu thành phần phạt đứng yên và biên đường chưa chuẩn thể thao. |
| **R1** | `origin/v2` (`d1a49da`) | RacingMap, Xe mặc định, $\Delta t = 0.1\text{s}$, so sánh Rule-based vs Residual PPO | $\sim 0.15\text{M}$ steps mỗi run, 20 vòng test | Rule-based: $105.6\text{s}$<br>Residual: $105.1\text{s}$<br>Pure PPO: $115.1\text{s}$ | **Bài học về kiến trúc Policy:** Chính sách lai ghép (Residual PPO học độ lệch trên nền bộ điều khiển luật) hội tụ cực nhanh và chạy rất mượt trên map phẳng, nhưng khó mở rộng khi xe trượt ở giới hạn ma sát cao. |
| **R2** | `experiment/120kmh-braking-dynamics` (`0d755a0`) | Sa bàn Silverstone nửa (lần đầu áp dụng), 2 làn x 4.5m, GT3 ban đầu | $1.00\text{M}$ steps, Comp max $98.5\%$, $v_{\text{max}} = 77.5\text{ km/h}$ | Best sạch trong eval: 8 lần | Lần đầu đưa mô hình lên sa bàn Silverstone Grand Prix. Phát hiện xe nguyên bản của MetaDrive có phanh quá yếu ($119\text{N}$), không thể hãm tốc trước các cua gấp. |
| **R3** | `feature/clean-111s-optimal-ghost` (`ee99f9d`) | **MÔ HÌNH CHÍNH:** Sa bàn nửa 9m, GT3RaceCar ($\mu_f=0.9, \mu_r=1.35$), 35D/3D, $r_{\text{center}}=0.04$, kerb nới lỏng | $4.00\text{M}$ steps ($2.15\text{M}$ mốc chuẩn), Comp max $98.6\%$, $v_{\text{max}} = 90.1\text{ km/h}$ | Eval anchor 4 bánh:<br>Standing: $114.78\text{s}$<br>Flying: **$109.44\text{s}$** (Best $108.60\text{s}$)<br>Sạch $18/20$ ($90\%$) | **Thành quả cốt lõi của đồ án:** PPO tự học kỹ thuật Trail-braking để dồn tải lên trục trước, nén lò xo đạt đỉnh bám tức thời $1.21\text{G}$, chạy nhanh hơn nghiệm giải tích chất điểm Lipp-Boyd ($112.80\text{s}$) tới $3.36\text{s}$. |
| **R7** | `feature/extended-physics-mpc-optimal` (`2055469`) | Sa bàn nửa, **42 chiều observation**, bổ sung vật lý mở rộng ExtendedVehiclePhysics | $4.00\text{M}$ steps, Comp max $86.6\%$, $v_{\text{max}} = 89.2\text{ km/h}$ | $133.0\text{s}$ (cuối run tụt dốc) | **Lỗi F1 (Kẹt trần Log-std):** Lệnh `torch.clamp` trong forward pass triệt tiêu gradient, khiến $\sigma$ của cả 3 chiều lái/ga/phanh kẹt cứng ở trần $0.002$ (tương ứng $\sigma = 1.0$), xe liên tục giật phanh ngẫu nhiên, không thể hội tụ. |
| **R8** | `feature/oop-clean-physics-40d` (`63feaf2` / `0180988`) | Sa bàn nửa, khôi phục Bullet thuần, **40 chiều observation**, đổi thưởng | $3.38\text{M}$ steps, Comp max $42.5\%$, crash rate $100\%$ | $132.4\text{s}$ (DNF $0/20$ sạch) | **Lỗi F2 (Thiếu bù trừ Jacobian):** Sử dụng hàm tanh squashing nhưng tính log-prob trực tiếp từ phân phối Gaussian mà không bù định thức Jacobian, dẫn đến gradient sai lệch nghiêm trọng, xe không hoàn thành được vòng sạch nào. |
| **R10** | `origin/feature/multicar-selfplay` (`caf69a0`) | Sa bàn 3 làn ($13.5\text{m}$), **6 xe tự chơi (Self-play)**, 65D observation | $4.45\text{M}$ samples | Best lap: **$128.2\text{s}$** (về đích 5-6 xe) | Khảo sát đua đối kháng nhiều xe: Phạt va chạm đối xứng giúp các xe học được kỹ năng tránh né và xếp hàng qua góc cua chật hẹp mà không gây nghẽn đường. |
| **R11** | `feature/silverstone-bk-phase1` (`6ec59ed`) | **Phase 1 Main:** Sa bàn nửa, luật 4 bánh khắt khe, đã vá F1 & F2, decay entropy | $2.66\text{M}$ steps, Comp max $100\%$, $v_{\text{max}} = 79.8\text{ km/h}$ | Best sạch: **$125.6\text{s}$** (31 eval sạch) | **Lỗi I9 (Sập Entropy):** Hạ `ent_coef` về $0.001$ khiến phương sai đánh lái sập về $0.10$ sau $150\text{k}$ steps. Mạng Critic học quá sớm rằng xe sẽ đâm ở cua mét 1306, triệt tiêu toàn bộ tín hiệu Advantage $\hat{A}_t$, run bị kẹt $46\%$ crash $100\%$. |
| **R12** | `feature/silverstone-phase2` (`e72ee80`) | **Phase 2 Single v1:** Sa bàn đầy đủ 1:1 ($5.8\text{km}$), GT3RealCar ($\mu=1.8/1.55$, $v > 200\text{ km/h}$) | $4.54\text{M}$ steps, 35D/3D | Lap min: $157.6\text{s}$ (1 eval sạch $218\text{s}$ @ $1.5\text{M}$) | **Lỗi F17 (Choke tốc độ trên đường thẳng):** Tác tử tự giới hạn tốc độ $\sim 150\text{ km/h}$ trên đoạn thẳng có $v^* = 260\text{ km/h}$ do xuất hiện dao động pedal 2 bước và góc lái cực nhạy ở vận tốc cao. |
| **R13** | `feature/silverstone-phase2` (`b783194` / `819d3e1`) | **Phase 2 Single v2:** Sa bàn 1:1, GT3RealCar, **2D pedal gộp**, phạt ra ngoài không cắt episode | $3.32\text{M}$ steps, Comp max $100\%$, $v_{\text{max}} = 151.6\text{ km/h}$ | Best sạch: **$143.4\text{s}$** (@ $2.56\text{M}$) | **Lỗi F20 & F23 (Dao động Pedal 5 Hz):** Phát hiện mạch hồi tiếp 2 bước giữa `last_pedal` trong observation và việc thiếu phạt vi phân $(\Delta\text{pedal})^2$ trong reward. Ở dải tốc độ cao, hiện tượng này kích động mất ổn định quán tính dọc. |
| **R14** | `feature/silverstone-phase2` (`6df6ba9`) | **Phase 2 Line:** Sa bàn 1:1, **47D Obs** (nạp trước sai lệch Racing Line $\Delta d_{\text{line}}$ và $v^*$) | $2.66\text{M}$ steps, Comp max $100\%$, $v_{\text{max}} = 152.2\text{ km/h}$ | Best sạch: **$153.7\text{s}$** (@ $1.74\text{M}$) | **Lỗi F21 (Thoái hóa Tốc độ do Policy Shortcut):** Xe học chạy sạch rất sớm ($1.15\text{M}$ vs $2.15\text{M}$ steps), nhưng sau $2.0\text{M}$ steps lap time thoái hóa về $173 - 181\text{s}$. Tác tử bám thụ động vào line tĩnh mà không học kỹ năng dồn tải, rơi vào cực tiểu bảo thủ. |
| **R17** | `exp/111s-no-rcenter` (`cde2cae`) | **Ablation Bỏ $r_{\text{center}}$:** Giữ nguyên 100% cấu hình Mô hình Chính, chỉ đặt $c_{\text{center}} = 0.0$ | $2.15\text{M}$ steps trên 2 seed (42 & 43) | $112.2\text{s}$ (DNF)<br>Clean: **$0/20$ ($0\%$)** ở S42; **$2/20$ ($10\%$)** ở S43 | **Insight A13:** Hệ số chiết khấu $\gamma=0.99$ khiến tầm nhìn hiệu dụng của mạng hữu hạn ($\approx 100$ bước). Không có $r_{\text{center}}$ dẫn đường, xe ham cắt cua sớm (Early Apex) dẫn đến văng lề không thể cứu vãn tại mét 363. |
| **F1** | `feature/f1-downforce` (`cdfb928`) | Thử nghiệm lực ép khí động học nhân tạo $F_{\text{down}} = mg(v/200)^2$, sa bàn 3 làn | $1.00\text{M}$ steps, $1.5\text{M}$ samples (6 xe) | Đơn: $109.4\text{s}$<br>6 xe: $112.9\text{s}$ | **Khảo sát Downforce:** Lực ép khí động học mở rộng ma sát ngang theo bình phương vận tốc giúp xe ôm cua ổn định ở $301\text{ km/h}$. Tuy nhiên đây là bài toán xe F1, không dùng đối chuẩn với GT3. |

---

## PHẦN III: BỘ CÂU HỎI VẤN ĐÁP KHOA HỌC DÀNH CHO BẢO VỆ ĐỒ ÁN (FAQ)

### 1. Tại sao nhóm khẳng định PPO thắng Ghost Car $3.36\text{s}$ là do vật lý chứ không phải do Ghost bị cấu hình yếu?
* **Trả lời:**
  - Nhóm đã thực hiện **quét lưới 33 cấu hình độc lập** (`scripts/analysis/ghost_envelope_sweep.py`) cho Ghost Car, thử nghiệm gia tốc ngang $a_{\text{lat}}$ từ $0.81\text{G}$ đến $1.10\text{G}$.
  - Kết quả thực nghiệm chứng minh: Khi đặt $a_{\text{lat}} \ge 1.00\text{G}$, do trễ cơ cấu chấp hành và động học quay vòng của thân xe thật, Ghost Car bị trượt bão hòa và **văng lề 100% tại khúc cua mét $2390\text{m}$**.
  - Con số $a_{\text{lat}} = 0.95\text{G}$ là cận bám tối ưu cao nhất mà Ghost Car còn chạy sạch được.
  - PPO vượt qua mốc này vì nó tối ưu hóa trực tiếp trên tương tác đa vật thể của Bullet: PPO tự học kỹ thuật **Trail-braking** (vừa phanh vừa ép lái), dồn tải trọng đè nén lò xo bánh trước ngoài để khai thác đỉnh bám động học tức thời lên tới **$1.21\text{G}$**.

### 2. Bài học rút ra từ việc đưa Racing Line vào quan sát (`p2_line`) là gì?
* **Trả lời:**
  - Đây là đóng góp phủ định rất giá trị: Nạp gợi ý đường chạy tối ưu giúp mô hình giảm số mẫu cần học để có vòng sạch đầu tiên (từ $2.15\text{M}$ xuống $1.15\text{M}$ steps).
  - Tuy nhiên, mạng nơ-ron rơi vào cái bẫy **"Lối tắt chính sách" (Policy Shortcut)**: Nó chỉ học bám an toàn theo tọa độ tĩnh thay vì học cách phối hợp tải trọng động học. Về lâu dài, chính sách bị thoái hóa tốc độ ($153\text{s} \to 181\text{s}$). Điều này khẳng định triết lý: **Zero Prior Model-Free RL** (không mớm đường trước) là phương pháp tối ưu nhất để đạt tới giới hạn vật lý.

### 3. Tại sao bỏ phạt tim đường $r_{\text{center}}$ thì xe lại không thể hoàn thành vòng đua?
* **Trả lời:**
  - Trong RL với $\gamma = 0.99$, trọng số tương lai sau $k$ bước suy giảm theo $\gamma^k$. Với $\Delta t = 0.1\text{s}$, tầm nhìn thực tế của mạng chỉ nhạy cảm trong khoảng $100$ bước ($10\text{s}$, tương ứng $\approx 250\text{m}$).
  - Khi vào góc cua mở rộng, xe không nhìn thấy lối thoát cua phía xa. Nếu không có $r_{\text{center}}$ ghì xe ở khu vực giữa đường, mạng sẽ chọn hành vi tham lam cục bộ: ôm sát lề trong quá sớm (**Early Apex**). Hậu quả hình học là khi ra cua, bán kính quỹ đạo bị bóp hẹp đột ngột, xe bị văng lề $100\%$ tại mét 363. $r_{\text{center}}$ đóng vai trò như một lực thế năng định hướng (Potential field) giúp xe sống sót trong giai đoạn khám phá ban đầu.
