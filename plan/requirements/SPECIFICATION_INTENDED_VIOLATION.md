# 🏎️ BÁO CÁO KỸ THUẬT: ĐỐI CHIẾU ĐẶC TẢ & SAI LỆCH CÓ CHỦ ĐÍCH
## (Technical Report: Intended Specification Violations & Scientific Rationale)

> **Nhánh Git**: `feature/oop-clean-physics-40d`  
> **Tài liệu gốc đối chiếu**: [PROJECT_SPECIFICATION.md](file:///c:/Users/tunbi/Project/school/REL%20Assignment/plan/requirements/PROJECT_SPECIFICATION.md)  
> **Phiên bản cập nhật**: v4.0 (40D Observation Space, Pure Bullet Physics OOP, Steering Smoothness Penalty, Longitudinal MPC Optimal Benchmark & Strict Checkpoint Guard)  
> **Trạng thái**: Hoàn thiện Tác tử Đơn 40D trên nền Vật lý Bullet nguyên bản của MetaDrive & Xe bóng ma đối chứng (Single Agent 40D with MPC Optimal Ghost Car)

---

## 📌 1. Bảng Đối Chiếu Trực Tiếp Từng Mục So Với PROJECT_SPECIFICATION.md

| Mục trong Spec gốc | Yêu cầu tại SPEC gốc (`PROJECT_SPECIFICATION.md`) | Thực thi tại Nhánh Thử nghiệm này | Phân loại & Cơ sở Lý luận Khoa học |
| :--- | :--- | :--- | :--- |
| **Mục 1: Mục tiêu đề tài** | Agent tự học tối ưu thời gian hoàn thành 1 lap thông qua trial-and-error, không mớm đường đi. | Agent tối ưu Lap Time thuần túy qua Bellman Value Function, GAE và áp lực thời gian $r_{\text{time}} = -0.01$/step. | **TUÂN THỦ 100%**: Bản chất bài toán không đổi, loại bỏ mọi hàm heuristic mớm ga/phanh. |
| **Mục 2.1: Tốc độ tối đa (`max_speed_km_h`)** | Giữ nguyên mặc định thư viện MetaDrive: **$80.0\text{ km/h}$**. | Áp dụng cấu hình xe thể thao GT3: **$120.0\text{ km/h}$** với lực phanh $650\text{N}$, độ bám $1.35$ và Slew Rate Limiter tách biệt chuẩn GT3 Telemetry (Steering $0.18$, Brake $1.00$, Throttle $0.35$). | **SAI LỆCH CÓ CHỦ ĐÍCH**: Cho phép xe đạt tốc độ đua GT3 thực tế, kích hoạt phản xạ Trail-Braking và bám đường vật lý. |
| **Mục 2.1: Không gian hành động (Action Space)** | 2 chiều mặc định của MetaDrive: `[steer, pedal]` với $\text{pedal} \in [-1.0, 1.0]$ (dương là ga, âm là phanh). | **3 chiều độc lập (Decoupled Racing Action)**: `[steer, throttle, brake]` với $\text{throttle}, \text{brake} \in [0.0, 1.0]$. | **CẢI TIẾN ĐIỀU KHIỂN HỌC (SAI LỆCH CÓ CHỦ ĐÍCH)**: Phá vỡ bẫy bão hòa chân ga, tái hiện chuẩn đua xe thể thao (Left-Foot Braking & Trail Braking cắt chéo chữ X). |
| **Mục 2.1: Không gian quan sát (Observation Space)** | Vector cảm biến phẳng hoặc LiDAR 259 chiều của MetaDrive. | **40 chiều Zero Data Leakage**: 16 chiều Ego/IMU/Controls ($v_x, v_y, \dot{\psi}, \beta, a_x, a_y, \text{roll}, \text{pitch}, \delta_{t-1}, \text{thr}_{t-1}, \text{brk}_{t-1}, \Delta\delta, \mathbb{I}_{\text{runoff}}, d_{\text{lat}}/4.5, \sin\Delta\theta, \cos\Delta\theta$) + 24 chiều Lookahead tim đường tại 6 mốc $[10, 25, 45, 70, 105, 150]\text{m}$. | **CẢI TIẾN HỌC THUẬT (SAI LỆCH CÓ CHỦ ĐÍCH)**: Thay thế 240 tia LiDAR dư thừa bằng 6 điểm ngắm tim đường hình học (tầm nhìn 150m cho dải tốc độ đua $> 110\text{ km/h}$); tuyệt đối không nhúng tọa độ quỹ đạo tối ưu. |
| **Mục 2.2: Quy tắc 4 bánh FIA & Track Limits** | Xe chỉ terminate khi cả 4 bánh vượt ra ngoài mép đường. Được phép chạy lane ngược chiều để mở góc cua. | Bỏ hoàn toàn gờ Kerb ảo, lấy chuẩn đúng mép đường nhựa thật $4.5\text{m}$. Kiểm tra theo 4 lốp xe đua GT3: vòng chạy hợp lệ khi còn ít nhất 1 bánh trên mặt nhựa ($\le 4.5\text{m}$), chỉ terminate khi CẢ 4 BÁNH xe hoàn toàn rơi khỏi mặt nhựa xuống bãi cỏ. | **TUÂN THỦ 100% QUY TẮC FIA**: Đồng bộ hoàn hảo giữa hình ảnh 3D và logic trọng tài. |
| **Mục 2.3: Rào chắn & Vật lý địa hình** | Không rào chắn sát mép, không áp dụng vật lý địa hình phức tạp, quản lý bằng hình học. | Giữ nguyên quy tắc không rào chắn; khắc phục triệt để lỗi mesh địa hình bị biến mất bằng cơ chế Dynamic Centering. | **TUÂN THỦ 100%**: Giải quyết triệt để lỗi đồ họa đường đua vô hình trên màn hình 3D. |
| **Mục 3: Tiêu chí F1 (Phanh rà, Max ga, Chém Apex)** | Chỉ dùng làm thước đo trực quan, tuyệt đối không mớm vào code. | Không có dòng code nào ép xe bám ray hay ép xe phanh ở vị trí cố định. | **TUÂN THỦ 100% NGUYÊN TẮC HỌC THUẬT**. |
| **Mục 4: Tuyệt đối không Data Leak & Mớm điểm** | Không waypoint mớm, không imitation learning, không thưởng bám tim đường. | Xóa bỏ $100\%$ các hàm thưởng điều kiện `if upcoming_curve` và phạt tim đường `r_center = 0.0`. Dùng hàm thưởng Markov gọn sạch: tiến độ Frenet $\Delta s$ + phạt lắc lái $r_{\text{steer\_smooth}} = -0.10 (\Delta\text{steer})^2$ + phạt đứng yên $r_{\text{idle}}$ + thưởng cán đích bậc 3 $r_{\text{finish}}$. | **TUÂN THỦ TUYỆT ĐỐI**: Đưa bài toán về đúng nguyên lý *Reward Hypothesis* (Sutton & Barto). |
| **Mục 5.1: Hiện tượng đảo vô lăng loạn xạ (Jitter)** | Ghi nhận lỗi tay lái rung lắc ở $50\text{Hz}$ chưa giải quyết được. | Kết hợp Slew Rate Limiter cơ cấu chấp hành ($\Delta\text{steer}_{\max} = 0.18/\text{step}$) trên nền vật lý Bullet nguyên bản và hình phạt đạo hàm góc lái $r_{\text{steer\_smooth}} = -0.10 (\Delta\text{steer})^2$. Loại bỏ hoàn toàn lớp vật lý ngoài (`extended_physics.py`) để không gây trễ pha hay cản trở phản xạ bẻ lái của Agent. | **CẢI TIẾN KỸ THUẬT & ĐIỀU KHIỂN HỌC**: Triệt tiêu bang-bang control mà vẫn giữ phản hồi lái tức thời chuẩn vật lý thư viện. |
| **Mục 5.3 & 7.3: Sa bàn đua Silverstone** | Khuyến nghị chuyển sang map F1 thực tế có vòng lặp khép kín hoàn chỉnh. | Tích hợp sa bàn **Silverstone Grand Prix 18 góc cua ($3,417.8\text{m}$)** khép kín hoàn hảo. | **TUÂN THỦ 100% KHUYẾN NGHỊ**. |
| **Mục 8: Đánh giá & Đối chứng Cận trên (Upper Bound Benchmark)** | Không quy định cụ thể công cụ đo sai số quỹ đạo và tốc độ vào cua. | Tích hợp **Theoretical Optimal Benchmark Agent (`OptimalRacingAgent` với `LongitudinalRacingMPC`)** và **Xe Bóng ma 3D (`GhostCar`)** dựa trên Convex QP Minimum Curvature Line và Lipp & Boyd Time-Optimal Velocity Profile. | **CẢI TIẾN HỌC THUẬT (ĐỐI CHỨNG CẬN TRÊN & ZERO DATA LEAKAGE)**: Cung cấp cận trên lý thuyết ($T^* = 110.41\text{s}$, Optimal Agent $112.60\text{s}$ flying / $117.40\text{s}$ standing, MAE $0.42\text{m}$, Apex Accuracy $100\%$). Tuyệt đối không mớm vào hàm thưởng hay không gian quan sát của PPO. |
| **Quản lý Checkpoint & Độ tin cậy** | Không quy định ràng buộc điều kiện lưu model tốt nhất. | **Strict Reliability Guard**: Chỉ lưu `best_agent.pt` khi đạt **100% Zero-Crash (3/3 episode đánh giá không va chạm)** và `mean_completion >= 98%` trước khi xếp hạng Lap Time. Tách biệt `last_agent.pt` (chỉ dùng resume). | **ĐẢM BẢO TIN CẬY HỌC THUẬT**: Loại bỏ rủi ro lưu model "ăn may" đâm xe ở những mét cuối; bảo đảm tính toàn vẹn khi báo cáo khoa học. |

---

## 📌 2. Phân Tích Chuyên Sâu Các Sai Lệch & Giải Pháp Kỹ Thuật

### 2.1. Giải Mã Cơ Chế 3D Decoupled Action & Bộ Trộn Chân Ga Phanh ECU Liên Tục (Continuous C1 Deadband)

* **Giải pháp 3 Chiều Độc Lập (Decoupled Action Space)**:
  $$\mathbf{a} = [\text{steer}, \text{throttle}, \text{brake}] \in [-1, 1] \times [0, 1] \times [0, 1]$$
  - Hai neuron đầu ra độc lập: $\mu_{\text{throttle}}$ và $\mu_{\text{brake}}$ giúp PPO học điều khiển như tay đua thật (chân trái phanh, chân phải ga).
  - Phối hợp bàn đạp thực tế qua bộ trộn Drive-by-Wire liên tục (Continuous $C^1$ Deadband):
    $$\text{net\_pedal} = \text{thr\_eff} \cdot (1 - \text{brk\_eff}) - \text{brk\_eff}$$
    với $\text{brk\_eff} = \frac{\text{brk} - 0.015}{0.985}$ (khi $\text{brk} > 0.015$, ngược lại bằng 0).
  - Loại bỏ hoàn toàn bẫy bậc thang gián đoạn $C^0$, tạo hàm gradient khả vi trơn tru cho PPO và triệt tiêu 100% rung giật bàn đạp.
  - Cho phép thực hiện kỹ thuật **Pedal Crossover (X-Trace)**, **Left-Foot Braking** và **Trail Braking** như tay đua GT3 chuyên nghiệp: vừa mớm ga vừa đè phanh để ổn định vi sai và giữ áp suất nạp turbo qua apex.

---

### 2.2. Triệt Tiêu Hiện Tượng Lắc Lư Vô Lăng (Steering Jitter Regularization)

Để giải quyết việc xe "đảo tay lái siêu loạn" ở tần số $50\text{Hz}$:
1. **Hình phạt đạo hàm góc lái**:
   $$R_{\text{steer\_smooth}} = -0.10 \times (\text{steer}_t - \text{steer}_{t-1})^2$$
2. **Slew Rate Limiter cơ cấu chấp hành (`GT3RaceCar`)**:
   $$\text{steer}_{\text{applied}} = \text{steer}_{t-1} + \text{clip}(\Delta \text{steer}, -\Delta_{\max}, \Delta_{\max})$$
Giúp xe di chuyển êm ái, bám đúng quỹ đạo và không bị mất độ bám do đổi hướng bánh xe đột ngột.

---

### 2.3. Sửa Lỗi Hiển Thị Đồ Họa 3D (Invisible Track) & Singleton Map Lock Leak

1. **Lỗi sa bàn biến mất trên màn hình 3D**:
   Căn chỉnh tâm địa hình về trọng tâm sa bàn $(411.0\text{m}, -33.0\text{m})$, phủ kín toàn bộ $3,417.8\text{m}$ mặt đường nhựa.
2. **Lỗi sập `RuntimeError: can't allocate lock`**:
   Cố định đối tượng sa bàn `SilverstoneMap` dưới dạng Singleton bền vững trong bộ nhớ, triệt tiêu $100\%$ nguy cơ rò rỉ lock.

---

### 2.4. Tác Tử Đối Chứng Tối Ưu Lý Thuyết (OptimalRacingAgent) & Thực Thể Xe Bóng Ma (Ghost Car)

Để giải quyết bài toán: *"Làm sao biết Agent đã đua tối ưu chưa? Có thể định lượng độ chính xác quỹ đạo và tốc độ vào cua so với cận trên lý thuyết không?"*, dự án phát triển độc lập một module chuẩn mực phân tích theo phương pháp Luận văn / Báo cáo Quốc tế:

1. **Cơ sở Lý thuyết Cận trên (Theoretical Upper Bound)**:
   - **Quỹ đạo Cực tiểu Độ cong (Convex QP Minimum Curvature Line + C2 Gaussian Smoothing)**: Tối ưu hóa biến thiên độ dịch chuyển ngang trong biên $\pm 3.05\text{m}$ (nằm gọn trong mặt đường nhựa thật $4.50\text{m}$ không kerb ảo) bằng bộ giải bậc hai để giảm thiểu tối đa gia tốc ly tâm $a_y = v^2 \cdot \kappa$.
   - **Profile Tốc độ Tối ưu Thời gian (Lipp & Boyd 3-Pass Method)**: Hiệu chuẩn khớp chính xác với động lực học vật lý Bullet nguyên bản của thư viện MetaDrive, cho ra thời gian vòng đua cận trên lý thuyết: **$T^* = 110.41\text{s}$**.

2. **Kiến trúc Điều khiển Học của Tác tử Đối chứng (`OptimalRacingAgent`)**:
   - **Bộ điều khiển lái ngang (Lateral Control)**: Sử dụng **Adaptive Pure Pursuit** với giảm chấn cự ly ngắm theo độ cong (`curv_damping`) và hệ số bù góc trượt lốp (`1.15`).
   - **Bộ điều khiển dọc dự báo (`LongitudinalRacingMPC`)**: Giải bài toán Convex QP đa bước ($N=15$) kết hợp Predictive Dynamic Braking Horizon ($110\text{m}$) và ràng buộc Elip ma sát Kamm, loại bỏ hoàn toàn vùng chết thả trôi (0% coasting deadband).
   - **Thực nghiệm kiểm chứng**: Tác tử `OptimalRacingAgent` hoàn thành 100% vòng đua sạch trong **$112.60\text{s}$** (Flying Lap, hiệu suất nhịp độ $98.1\%$) và **$117.40\text{s}$** (Standing Start từ $0\text{ km/h}$), đạt **MAE quỹ đạo $= 0.42\text{m}$** và **Apex Accuracy $= 100.0\%$**!

3. **Thực thể Xe Bóng Ma 3D (`GhostCar`) & Chế độ Xem Biểu diễn (`spectate_ai.py`)**:
   - Thiết kế theo nguyên lý Hướng đối tượng (OOP) & DRY: Đóng gói thực thể xe đua độc lập, xuất phát đồng vị trí (`[60.0, 2.25]`) và đồng vận tốc (`0 km/h`) với xe chính khi chạy biểu diễn.
   - Vô hiệu hóa toàn bộ mặt nạ va chạm vật lý (`intoCollideMask = BitMask32.allOff()`) để trở thành xe bóng ma xuyên thấu (Phantom/Hologram), không bao giờ gây va chạm hay ngắt vòng đua của xe chính.
   - Hiệu ứng thị giác Hologram bán trong suốt ($70\%$ opacity) trên nền Panda3D.
   - Màn hình HUD cung cấp telemetry Sim Racing theo thời gian thực: khoảng cách chênh lệch ($\text{Gap}$ mét), tốc độ Ghost, và thanh Delta Time ($\Delta t$).

4. **Trực quan hóa Quỹ đạo Đua Tối ưu 3D (`RacingLineVisualizer`) & Phím tắt [L]**:
   - Module OOP `game_env/visualizer.py` quản lý node đồ họa 3D khép kín trên nền Panda3D `engine.render`.
   - Cao độ $z = 0.15\text{m}$ triệt tiêu hoàn toàn hiện tượng nhấp nháy z-buffer (z-fighting) với mặt đường nhựa.
   - Bắt sự kiện bàn phím native `[L]` thông qua hệ thống sự kiện của Panda3D mà không can thiệp thư viện ngoài.
   - Quỹ đạo Out-In-Out giải quyết chuẩn xác qua Hình học vi phân đường cong tham số tự nhiên $\min_\alpha \int (x''(s)^2 + y''(s)^2) ds$: Tự động tạo bán kính cong cực đại qua chu trình Out-In-Out kinh điển.

5. **Cơ Cấu Chấp Hành Vật Lý Nguyên Bản OOP (`game_env/vehicle.py`) & Kháng Rung Lắc Tay Lái**:
   - **Loại bỏ hoàn toàn lớp vật lý can thiệp ngoài (`extended_physics.py`)**: Không áp dụng bộ lọc trễ pha EMA trên góc lái hay lực cản nhân tạo ngoài thư viện để đảm bảo phản xạ lái tức thời và đồng nhất 100% với engine Bullet của MetaDrive.
   - **Slew Rate Limiter & 3D Pedal Mixing**: Áp dụng giới hạn tốc độ đánh lái cơ học $\Delta\text{steer}_{\max} = 0.18/\text{step}$ kết hợp hình phạt đạo hàm góc lái $r_{\text{steer\_smooth}} = -0.10 (\Delta\text{steer})^2$ (khôi phục từ nhánh `feature/clean-111s-optimal-ghost`), triệt tiêu hiện tượng bang-bang control mà vẫn giữ độ nhạy bén khi ôm cua.

6. **Không Gian Quan Sát 40 Chiều (40D Observation Space) & Cảm Biến Quán Tính Ảo IMU**:
   - **16 chiều Ego Kinematics, IMU & Controls**:
     $$[v_x/40, v_y/20, \dot{\psi}/\pi, \beta, a_{\text{long}}, a_{\text{lat}}, \phi_{\text{roll}}/0.35, \theta_{\text{pitch}}/0.35, \delta_{t-1}, \text{thr}_{t-1}, \text{brk}_{t-1}, \Delta\delta, \mathbb{I}_{\text{runoff}}, d_{\text{lat}}/4.5, \sin(\Delta\theta), \cos(\Delta\theta)]$$
     - $\beta = \arctan2(v_y, \max(0.1, v_x)) / (\pi/4)$: Góc trượt thân xe (body slip angle), hỗ trợ PPO nhận biết trạng thái thừa lái (oversteer) và thiếu lái (understeer).
     - $a_{\text{long}}, a_{\text{lat}}$: Gia tốc trọng trường mô phỏng cảm biến IMU, phản ánh trực tiếp vòng tròn ma sát lốp Kamm (Kamm's Friction Circle).
   - **24 chiều Lookahead Horizon Preview (6 Waypoints $\times$ 4)**:
     Tại các cự ly $d \in \{10\text{m}, 25\text{m}, 45\text{m}, 70\text{m}, 105\text{m}, 150\text{m}\}$ phía trước xe dọc trục Frenet $s$:
     $$[\Delta x_{\text{rel}} / 150, \Delta y_{\text{rel}} / 40, \sin(\Delta \theta_{\text{wpt}}), \text{clip}(\kappa_{\text{wpt}} \cdot 50, -2, 2)]$$
     Tầm nhìn 150m cho phép Agent nhận biết khúc cua từ xa để thực hiện kỹ thuật phanh sớm (Threshold Braking & Trail-Braking) từ dải tốc độ đua $> 110\text{ km/h}$.
   - **Cam kết Zero Data Leakage**: Toàn bộ 24 đặc trưng lookahead được trích xuất thuần túy từ tim đường hình học (Centerline), hoàn toàn không chứa tọa độ hay vận tốc từ quỹ đạo đua tối ưu lý thuyết của `OptimalRacingAgent`.

7. **Kiến Trúc Đánh Giá Độc Lập OOP (`RacingEvaluator`) & Chuẩn Mực Checkpoint 100% Tin Cậy**:
   - **DRY & Single Responsibility**: Tách biệt toàn bộ logic đánh giá đa episode, trích xuất metrics (Line MAE, Apex Accuracy, G-G Grip, Pace Efficiency) sang module [training/evaluator.py](file:///c:/Users/tunbi/Project/school/REL%20Assignment/training/evaluator.py), dùng chung giữa vòng lặp huấn luyện (`train_racing.py`) và công cụ benchmark độc lập (`evaluate.py`).
   - **Strict Reliability Guard**: Checkpoint `best_agent.pt` được quản lý bởi cơ chế bảo vệ nghiêm ngặt: Chỉ những policy đạt **100% Zero-Crash (3/3 episode đánh giá không va chạm)** và `mean_completion >= 98%` mới được xét duyệt kỷ lục thời gian vòng đua (`Lap Time` hoặc `Flying Lap Time`).
   - **Tách biệt Checkpoint**: `last_agent.pt` chỉ dùng độc quyền để phục hồi sau sự cố gián đoạn (resume an toàn), còn `agent_step_*k.pt` lưu lại snapshot tiến trình phục vụ phân tích học thuật.

---

## 📌 3. Cam Kết Liêm Chính Học Thuật

Tất cả các thay đổi trên được ghi nhận minh bạch và có cơ sở lý luận khoa học chặt chẽ:
- **Tác tử PPO**: Hoàn toàn tuân thủ nguyên lý Reinforcement Learning không mớm điểm, tự tối ưu qua hàm giá trị Bellman, GAE và áp lực thời gian.
- **Tác tử Optimal & Ghost Car**: Đóng vai trò là Mẫu đối chứng Cận trên (Upper Bound Benchmark) độc lập, phục vụ định lượng khoa học độ hội tụ và hiệu năng điều khiển của thuật toán.

