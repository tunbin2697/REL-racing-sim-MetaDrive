# MODULE 8: VẬT LÝ SIM RACING, HỆ THỐNG TELEMETRY & TỐI ƯU HÓA HÀM THƯỞNG TRONG METADRIVE

> **Mục tiêu chuyên sâu**: Kết nối giữa nguyên lý vật lý thể thao tốc độ (Sim Racing Telemetry) và nền tảng toán học của Học tăng cường (Reinforcement Learning - REL), cung cấp cơ sở lý thuyết chuẩn mực phục vụ việc thiết kế hàm thưởng, giám sát hiệu năng vi mô và lập báo cáo học thuật.

---

## 1. Bản Chất Mô Hình Vật Lý Xe Đua Trong MetaDrive Simulator

MetaDrive sử dụng engine vật lý **Panda3D Bullet Physics (`BulletVehicle`)**, mô phỏng động lực học thân xe cứng kết hợp hệ thống treo tia quét (Raycast Suspension) 4 bánh độc lập.

```
                           +------------------------+
                           |  Thân xe (Rigid Body)  |
                           |   Khối lượng & Quán tính|
                           +------------------------+
                               /      |      |      \
                              /       |      |       \
             Raycast Suspension:      |      |        \
             - Độ nén lò xo           v      v         v
             - Giảm chấn (Damping) [FL]   [FR]       [RL]   [RR]
             - Ma sát trượt bánh xe (Coulomb Friction Slip)
```

### 1.1. Các tham số điều khiển cơ học cốt lõi
* **Lực kéo động cơ ($F_{\text{engine}}$)**: Khi hành động ga $a_1 > 0$:
  $$F_i = F_{\text{max\_engine}} \cdot a_1 \quad (\forall i \in \text{bánh dẫn động})$$
* **Lực phanh ($F_{\text{brake}}$)**: Khi hành động phanh $a_1 < 0$:
  $$F_{\text{brake}, i} = F_{\text{max\_brake}} \cdot |a_1| \quad (\forall i \in 4 \text{ bánh})$$
* **Hệ thống đánh lái**: Góc bẻ lái của 2 bánh trước tỷ lệ với $a_0 \in [-1, 1]$ qua $\theta_{\text{steer}} = a_0 \cdot \theta_{\max}$.

### 1.2. Đối chiếu Khả năng Vật lý: MetaDrive vs Thiết bị Đua xe Chuyên nghiệp
1. **Những đặc tính MetaDrive mô phỏng chính xác 100%**:
   * Trọng tâm xe, lực quán tính khi tăng tốc/phanh gấp.
   * Lực ly tâm và gia tốc ngang khi vào cua tốc độ cao ($a_y = v^2 / R$).
   * Hiện tượng trượt rê thân xe (Chassis Drift / Lateral Skid) khi vượt quá giới hạn ma sát.
2. **Những giới hạn cần lưu ý (Không có trong MetaDrive)**:
   * **Nhiệt động học lốp (Tire Thermodynamics)**: Lốp xe không bị nóng, không thay đổi độ bám theo nhiệt độ.
   * **Mòn lốp & Áp suất lốp**: Ma sát mặt đường và bánh xe là hằng số cố định.

---

## 2. Hệ Thống Sim Racing Telemetry Chuyên Sâu

Trong thể thao tốc độ chuyên nghiệp, người ta không đánh giá tay đua bằng cảm giác mà bằng **Telemetry (Dữ liệu đo đạc từ xa)** ghi lại từng mili-giây và từng mét đường đua.

```
                       +------------------------------------------+
                       |       BỘ TELEMETRY ĐUA XE METADRIVE      |
                       +------------------------------------------+
                                            |
         +------------------+---------------+------------------+------------------+
         |                  |                                  |                  |
         v                  v                                  v                  v
   [Speed Trace]      [Delta Time]                     [Pedal & Steering]    [G-G Diagram]
  v(s) chữ V / U    So sánh từng mét                   Phanh (Trail brake)   Vòng tròn ma sát
  Đỉnh cua V-min    với Best Lap (Xanh/Đỏ)             Ga (Hesitation/Lift)  (Khai thác độ bám)
```

### 2.1. Biểu đồ Chân Phanh (Brake Input Trace)
Biểu đồ thể hiện phần trăm lực phanh theo quãng đường $s$:
* **Độ dốc nhát đạp đầu tiên (Initial Bite)**: Đạt 100% lực phanh trong khoảng thời gian ngắn nhất để triệt tiêu động năng nhanh nhất trên đường thẳng.
* **Kỹ thuật Phanh rê vào cua (Trail Braking Profile)**: 
  * *Lái xe thông thường*: Phanh 100% $\to$ nhả hẳn về 0% $\to$ mới bắt đầu đánh lái (gãy khúc).
  * *Kỹ thuật Đua xe (Trail Braking)*: Đạp mạnh 100% ở đầu vùng phanh $\to$ vừa nhả dần phanh (từ 80% $\to$ 40% $\to$ 0%) vừa tăng dần góc đánh lái vào tâm cua (Apex). Đồ thị phanh có dạng hình tam giác dốc một bên.
* **Điểm phanh (Braking Point)**: Tọa độ mét chính xác mà tay đua bắt đầu đạp phanh trước mỗi khúc cua.

### 2.2. Biểu đồ Chân Ga (Throttle Input Trace)
* **Thời điểm bắt đầu mớm ga (Initial Throttle Application)**: Xác định vị trí mở ga lại (trước, tại, hay sau Apex).
* **Độ mượt mà (Throttle Progression) & Hesitation**:
  * Đồ thị chuẩn: Tăng tuyến tính mượt mà từ 0% lên 100%.
  * Lỗi kỹ thuật: Đạp lên 60% rồi phải nhả về 30% do xe bị văng đuôi rồi mới dám đạp lại 100% (Hesitation / Throttle lift).
* **Khoảng thời gian thả trôi (Coasting Phase)**: Khoảng cách mà cả Chân Ga và Chân Phanh đều bằng 0 ($a_1 \approx 0$). Trong đua xe, thời gian thả trôi càng dài chứng tỏ tay đua càng thiếu tự tin và lãng phí thời gian vòng chạy.

### 2.3. Góc Đánh Lái & Biểu Đồ Vòng Tròn Ma Sát (G-G Diagram)
Theo định luật ma sát của lốp xe: Lốp chỉ có một giới hạn bám tổng hợp nhất định ($F_{\text{total}} \le \mu F_z$):
$$a_x^2 + a_y^2 \le G_{\max}^2$$
* Trục tung ($a_x$): Gia tốc dọc (Dương: Tăng tốc, Âm: Phanh).
* Trục hoành ($a_y$): Gia tốc ngang (Rẽ trái / Rẽ phải).
* **Vòng tròn ma sát (Friction Circle)**:
  * Một vòng chạy hoàn hảo của AI sẽ **lấp đầy đường biên ngoài cùng của vòng tròn G-G** (khai thác 100% giới hạn vật lý của xe ở mọi thời điểm: vừa phanh tối đa vừa rẽ, hoặc vừa ôm cua vừa mở ga tối đa).
  * Nếu biểu đồ có khoảng trống lớn ở các góc phần tư, AI đang chạy dưới mức tiềm năng của xe.

### 2.4. Biểu Đồ Tốc Độ (Speed Trace) & Đỉnh Cua $V_{\min}$
* **Khúc cua gắt (Chicane / Hairpin)**: Đồ thị tốc độ $v(s)$ phải tạo thành **hình chữ V nhọn hoắt**. Phanh dứt khoát chạm đáy $V_{\min}$ tại đỉnh cua rồi dốc đứng tăng tốc ngay.
* **Khúc cua dài (Sweeper / Parabolica)**: Đồ thị tạo thành **hình chữ U đáy hẹp**.
* Đáy chữ U bị bè ngang hoặc răng cưa chứng tỏ xe bị dậm chân tại chỗ ở giữa cua.

### 2.5. Kênh Chênh Lệch Thời Gian Thực (Delta Time Channel)
So sánh mốc thời gian tại từng cự ly $s$ với vòng chạy chuẩn (Reference / Best Lap):
$$\Delta t(s) = t_{\text{current}}(s) - t_{\text{ref}}(s)$$
* $\Delta t(s) < 0$ (**Xanh lá**): Xe đang chạy nhanh hơn kỷ lục tại vị trí đó.
* $\Delta t(s) > 0$ (**Đỏ**): Xe đang bị chậm hơn kỷ lục.
* *Quy luật bù trừ*: Nếu vào cua phanh trễ kiếm được $-0.1s$ (báo xanh), nhưng vì vào quá nhanh làm hỏng góc thoát cua khiến đoạn thẳng sau đó mất $+0.3s$ (báo đỏ ngay lập tức) $\implies$ Tay đua nhận ra chiến thuật phanh trễ đó là sai lầm!

---

## 3. Cầu Nối Toán Học Giữa Telemetry Và Học Tăng Cường (RL)

### 3.1. Giải quyết Bài toán Phân bổ Công trạng (Credit Assignment Problem)
* **Bế tắc của Sparse Reward**: Nếu chỉ thưởng khi chạm vạch đích ($T = 2.000$ steps), đạo hàm gradient của mạng nơ-ron có phương sai cực đại (High Variance). AI không thể phân bổ công trạng xem mình nhanh hơn nhờ khúc cua nào.
* **Sức mạnh của Micro Telemetry Feedback**:
  * Khi gắn kênh phản hồi Telemetry vào từng mét đường ($s$), mạng Critic $V(s)$ ước lượng chính xác hàm giá trị tại từng khúc cua.
  * Tín hiệu Advantage $A(s, a) = Q(s, a) - V(s)$ phản ánh trực tiếp hành động đánh lái/phanh tại mét đó có làm cải thiện Delta Time hay không, giúp PPO hội tụ nhanh gấp nhiều lần.

### 3.2. Cấu Trúc Hàm Thưởng Hai Tầng (Two-level Hierarchical Reward)

$$\text{Reward Tổng} = R_{\text{Micro}} + R_{\text{Macro}}$$

#### Tầng 1: Vi mô (Micro Step-level Reward)
Tạo động lực uốn nắn kỹ thuật lái trên từng mét đường:
$$r_t = w_{\text{prog}} \cdot \Delta s_t + w_{\text{speed}} \cdot \frac{v_t}{v_{\max}} \cos(\Delta \theta) - r_{\text{step\_penalty}}$$
* $\Delta s_t$: Quãng đường xe tiến dọc theo tim đường đua (mét).
* $\cos(\Delta \theta)$: Hệ số định hướng (chỉ thưởng khi đầu xe hướng dọc đường đua).
* $r_{\text{step\_penalty}} = -0.05$: Phạt thời gian trôi qua.

#### Tầng 2: Vĩ mô (Macro Episode-level Reward)
Neo chặt mục tiêu tối thượng (Hoàn thành vòng & Phá kỷ lục):
$$R_{\text{terminal}} = R_{\text{Finish}} + R_{\text{Record\_Gap}} - R_{\text{Crash}}$$
* $R_{\text{Finish}} = +100.0$: Thưởng khi hoàn thành trọn vẹn 1 vòng đua (đảm bảo tỷ lệ hoàn thành $\ge 99\%$).
* $R_{\text{Record\_Gap}} = \beta \cdot \max(0, T_{\text{best}} - T_{\text{lap}})$: Thưởng khi phá kỷ lục vòng chạy của toàn bộ quá trình huấn luyện.
* $R_{\text{Crash}} = -30.0$: Phạt khi 4 bánh vượt khỏi đường đua (Track Limits).

### 3.3. Chứng minh Toán học: Phạt Thời Gian Step Tương Đương Với Tối Thiểu Hóa Lap Time
Tổng phần thưởng phạt thời gian trong suốt một vòng đua gồm $N$ bước thời gian với bước nhảy $\Delta t$:
$$\sum_{t=1}^N r_{\text{step\_penalty}} = \sum_{t=1}^N (-c) = -c \cdot N = - \frac{c}{\Delta t} \cdot T_{\text{lap}}$$
Do đó:
$$\max_{\pi} \mathbb{E}\left[ \sum_{t=1}^N r_t \right] \iff \min_{\pi} T_{\text{lap}}$$
Việc áp dụng phạt $r_{\text{step\_penalty}} < 0$ ở mỗi bước **tương đương về mặt toán học với bài toán tối thiểu hóa thời gian chạy một vòng ($T_{\text{lap}}$)** nhưng loại bỏ hoàn toàn bẫy Credit Assignment của phần thưởng thưa!

---

## 4. Danh Mục Biểu Đồ Học Thuật Phục Vụ Báo Cáo Môn Học

Một báo cáo nghiên cứu RL Đua xe chuẩn mực quốc tế cần trình bày 4 biểu đồ sau:
1. **Biểu đồ Hội tụ Lap Time & Completion Rate**: Chứng minh thuật toán đạt tỷ lệ hoàn thành vòng đua $\ge 99\%$ và thời gian vòng chạy giảm dần qua các epochs.
2. **Speed Trace Comparison ($v$ vs $s$)**: Vẽ đường cong vận tốc của AI so sánh với Người chơi/Reference Lap, chỉ rõ các đáy cua hình chữ V tại Chicane.
3. **Biểu đồ Phanh & Ga theo Quãng đường**: Phân tích kỹ thuật Trail Braking mượt mà của AI.
4. **Vòng tròn Ma sát G-G Diagram**: Biểu diễn đám mây điểm $(a_y, a_x)$ lấp đầy đường tròn ma sát tối đa của xe.
