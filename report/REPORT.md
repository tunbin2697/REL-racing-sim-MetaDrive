# BÁO CÁO NGHIÊN CỨU KHOA HỌC VÀ KỸ THUẬT ĐỒ ÁN
# TỐI ƯU HÓA ĐIỀU KHIỂN XE ĐUA TỰ HÀNH BẰNG HỌC TĂNG CƯỜNG SÂU TRÊN MÔ PHỎNG VẬT LÝ BULLET VÀ ĐỐI SÁNH GIẢI TÍCH

> **Môn học**: Học Tăng Cường (Reinforcement Learning - REL)  
> **Chủ đề**: Autonomous High-Speed Racing via Continuous Model-Free Reinforcement Learning  
> **Mô hình & Môi trường Trọng tâm**: Mô hình tham chiếu `ref_111s` trên môi trường xe thể thao `gt3_anchor` (Nhánh `feature/111s-honest-anchor`, Commit gốc: `ee99f9d`, Commit tài liệu & đối chuẩn: `7853722`)  
> **Các khảo sát mở rộng**: Nhánh quan sát 40D (`feature/oop-clean-physics-40d`), Nhánh đua đa tác tử tự chơi (`origin/feature/multicar-selfplay`), và Nhánh lực ép khí động học (`feature/f1-downforce`) — được trình bày trong phần Thảo luận như các bài học thực nghiệm về điều chỉnh cấu hình xe và tham số  
> **Định dạng báo cáo**: Báo cáo học thuật cấu trúc IMRaD (Introduction, Methods, Results, Discussion)  
> **Cam kết liêm chính học thuật**: Báo cáo trình bày dữ liệu thực nghiệm nguyên bản từ hệ thống đo lường và telemetry; phân biệt rõ các cấp độ nhận định: **[Đo được]**, **[Suy luận]**, **[Giả thuyết]**; không ngụy tạo kết quả, không che giấu lỗi kỹ thuật, và phân tích minh bạch các hạn chế của mô hình.

---

## TÓM TẮT (ABSTRACT)

Báo cáo này nghiên cứu bài toán điều khiển xe đua thể thao tự hành ở giới hạn ma sát vật lý trên sa bàn đua Silverstone Grand Prix thu nhỏ, sử dụng thuật toán Học Tăng Cường Không Mô Hình (Model-Free Continuous Reinforcement Learning) dựa trên thuật toán Tối ưu hóa Chính sách Tiệm cận (Proximal Policy Optimization - PPO). Hệ thống được triển khai trên nền tảng mô phỏng vật lý đa vật thể Bullet Physics thông qua môi trường MetaDrive. Thay vì cung cấp trước quỹ đạo tối ưu trong quan sát hay áp dụng kỹ thuật học bắt chước (Imitation Learning), nghiên cứu tập trung giải quyết câu hỏi: **Liệu một tác tử RL với quan sát thuần hình học tim đường (Zero Racing Line Prior) có thể tự học các kỹ thuật điều khiển xe thể thao (Trail-braking, Apex clipping) và đạt hiệu năng tương đương hoặc vượt qua phương pháp điều khiển tối ưu giải tích dựa trên mô hình (Convex QP Minimum Curvature và Lipp-Boyd Speed Profiling) hay không?**

Kết quả thực nghiệm trên mô hình tham chiếu `ref_111s` (Commit `ee99f9d`), được đánh giá độc lập dưới quy tắc vạch đường 4 bánh nghiêm ngặt của FIA qua 20 episode kiểm thử ($n = 20$), cho thấy tác tử PPO hoàn thành vòng đua với thời gian vòng xuất phát đứng (Standing Lap) trung bình là $114.78 \pm 0.40\text{ s}$ và thời gian vòng chạy đà (Flying Lap) là $109.44 \pm 0.50\text{ s}$, đạt tỷ lệ vòng sạch không va chạm (Clean Lap Rate) là $90\%$ ($18/20$ vòng hoàn thành). Thời gian vòng flying này thấp hơn thời gian của tác tử giải tích hiệu chuẩn thực nghiệm (`gt3_anchor`: Standing $118.20\text{ s}$, Flying $112.80\text{ s}$, thời gian tích phân lý thuyết $T^* = 110.62\text{ s}$) khoảng $3.36\text{ s}$. Tính lặp lại độc lập được xác nhận trên Seed 43 với thành tích $115.18 \pm 0.43\text{ s}$ (Standing) và $109.96 \pm 0.69\text{ s}$ (Flying, $19/20$ vòng sạch).

Báo cáo phân tích nguyên nhân hiện tượng "RL đạt thời gian vòng thấp hơn nghiệm giải tích" bắt nguồn từ sự sai lệch mô hình - hệ thống thực (Model-Plant Mismatch): mô hình chất điểm giải tích giả định gia tốc ma sát ngang không đổi ($0.95\text{G}$), trong khi PPO trực tiếp khai thác giới hạn bám động học tức thời của Bullet Physics ($a_{\text{lat}}$ phân vị $95\%$ đạt $0.99\text{G}$, cực đại $1.21\text{G}$). Đồng thời, báo cáo phân tích chi tiết các thách thức trong quá trình tối ưu hóa PPO liên tục:
1. **Hiện tượng dao động bàn đạp tần số cao (5 Hz PWM chattering)**: Việc vector quan sát nạp hành động pedal bước trước (`obs[6] = last_pedal`) nhưng hàm thưởng thiếu thành phần phạt độ mượt vi phân pedal ($\Delta\text{pedal}^2$) đã hình thành mạch hồi tiếp dao động 2 bước (bước chẵn đạp ga, bước lẻ rà phanh).
2. **Kẹt độ lệch chuẩn khám phá (Actor Log-std Freeze)** do đạo hàm triệt tiêu của hàm clamp trong forward pass, và hiện tượng suy giảm phương sai khám phá khi decay entropy quá sớm.
3. **Vai trò ràng buộc của thành phần phạt lệch tim đường $r_{\text{center}}$** trong việc hạn chế xu hướng cắt cua sớm (Early Apex) bắt nguồn từ tầm nhìn hữu hạn của hệ số chiết khấu $\gamma = 0.99$.

---

## 1. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU NGHIÊN CỨU (INTRODUCTION)

### 1.1. Bối cảnh và Thách thức Kỹ thuật
Điều khiển tự hành phương tiện ở vận tốc cao tiệm cận giới hạn ma sát của lốp xe (Autonomous High-Speed Racing) là bài toán phức tạp trong điều khiển học và trí tuệ nhân tạo. Khác với bài toán lái xe thông thường (Urban Driving), xe đua vận hành trong điều kiện phi tuyến tính: lực bám lốp xe nằm trong vùng bão hòa của elip ma sát (Kamm's Friction Ellipse), đòi hỏi sự kết hợp giữa lực dọc (phanh/ga) và lực ngang (đánh lái) nhằm khai thác gia tốc tổng hợp.

Phương pháp truyền thống giải quyết bài toán này qua hai giai đoạn tách rời: (1) Quy hoạch quỹ đạo tĩnh dựa trên mô hình đơn giản hóa (tối thiểu độ cong qua Convex QP - Heilmeier et al., 2020; phân bố tốc độ dọc tối ưu thời gian - Lipp & Boyd, 2014); và (2) Bám quỹ đạo bằng các bộ điều khiển phản hồi như Model Predictive Control (MPC) hoặc Pure Pursuit. Tuy nhiên, sự sai lệch giữa mô hình động học giả định (chất điểm, mô hình xe đạp) và động lực học thực tế của hệ thống mô phỏng đa vật thể (Multi-body Dynamics với hệ thống treo, tải trọng thay đổi) tạo ra khoảng cách hiệu năng (Model-Plant Mismatch).

Học Tăng Cường Sâu (Deep Reinforcement Learning - DRL) cho phép tối ưu hóa chính sách điều khiển trực tiếp trên môi trường mô phỏng (Fuchs et al., 2021; Wurman et al., 2022). Việc áp dụng DRL trong điều khiển xe đua liên tục đặt ra các câu hỏi nghiên cứu cụ thể:
1. **Quan sát thuần (Observation Integrity)**: Tác tử có thể tự hình thành kỹ thuật đua xe thể thao (Out-In-Out, Trail-braking) từ quan sát hình học tim đường thuần túy mà không cần cung cấp trước tọa độ quỹ đạo tối ưu (Racing Line Prior) hay không?
2. **Bản chất của việc vượt qua mốc chuẩn (Baseline Comparison)**: Khi tác tử RL đạt thời gian hoàn thành vòng đua nhanh hơn mô hình giải tích, cơ chế động lực học nào trong mô phỏng giải thích cho kết quả này?
3. **Độ mượt cơ học và Hiện tượng Chattering**: Nguyên nhân khiến các thuật toán Policy Gradient liên tục xuất tín hiệu điều khiển rung giật tần số cao và cách thức kiểm soát hiện tượng này (Mysore et al., 2021).

### 1.2. Mục tiêu và Phạm vi Đồ án
Nghiên cứu tập trung vào các mục tiêu kỹ thuật rõ ràng:
- **Xây dựng môi trường mô phỏng chuẩn xác**: Thiết lập môi trường đua xe thể thao GT3 trên sa bàn Silverstone Grand Prix (tổng chiều dài $2,838.7\text{ m}$ theo hệ thống MetaDrive/Bullet Physics), tích hợp quy tắc trọng tài 4 bánh theo chuẩn FIA.
- **Phát triển Tác tử Đối chứng Giải tích (Analytical Baseline)**: Tích hợp bộ giải Convex QP và thuật toán tích phân Lipp-Boyd để thiết lập xe bóng ma giải tích (`GhostCar`), làm mốc đối chứng khoa học để so sánh.
- **Tối ưu hóa Tác tử PPO Liên tục Thuần (Pure PPO Agent)**: Huấn luyện tác tử với không gian hành động 3 chiều $[steer, throttle, brake]$ và không gian quan sát thuần 35 chiều, không sử dụng thông tin quỹ đạo giải tích trong quá trình học.
- **Đánh giá Độc lập và Phân tích Thực nghiệm**: Đo lường định lượng trên 20 episode độc lập, phân tích dữ liệu Telemetry (biểu đồ G-G, biến thiên góc lái, góc trượt thân xe), kiểm tra tính tái lập trên hạt giống ngẫu nhiên khác nhau, và thực hiện các nghiên cứu bóc tách (Ablation Studies).

---

## 2. NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

Nghiên cứu được đối chiếu trực tiếp với 4 công trình khoa học trong lĩnh vực đua xe tự hành và điều khiển liên tục bằng RL:

### 2.1. Fuchs et al. (IEEE RA-L / ICRA 2021): Đua xe Super-Human trên Gran Turismo Sport
Fuchs, Song, Kaufmann, Scaramuzza, và Dürr (2021) công bố công trình *"Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning"* trên IEEE Robotics and Automation Letters (RA-L).
- **Phương pháp**: Sử dụng thuật toán Soft Actor-Critic (SAC) không mô hình với mạng nơ-ron 2 lớp ẩn (256 units, ReLU), tần số điều khiển $10\text{ Hz}$.
- **Không gian quan sát và hành động**: Quan sát gồm vận tốc, gia tốc 3D, góc lệch tim đường, 13 tia khoảng cách (fan 180°, tầm 100m), độ cong tim đường tại 10 điểm từ $1.0\text{s}$ đến $2.8\text{s}$ phía trước. Tác giả không đưa racing line vào quan sát. Hành động gồm góc lái và một trục pedal gộp $[-1, 1]$ (dương là ga, âm là phanh).
- **Hàm thưởng và kết thúc**: Phần thưởng chính là tiến độ dọc tim đường ($r_{\text{progress}}$) trừ đi phạt va chạm: $r_t = r_{\text{progress}} - c_w \|v\|^2 \cdot \mathbb{I}_{\text{wall}}$ với $c_w = 5 \times 10^{-4}$. Episode không bị kết thúc khi va chạm tường nhằm tăng cường lấy mẫu.
- **Kiểm thử độ bền (Robustness)**: Kiểm tra tác tử dưới trễ tín hiệu ($\Delta t_{\text{delay}} \in [50, 100, 150]\text{ ms}$) và nhiễu quan sát Gaussian ($\sigma_{\text{noise}} \in [2\%, 5\%, 9\%]$). Kết quả cho thấy tác tử duy trì ổn định ở mức nhiễu $\le 2\%$, nhưng trễ $> 100\text{ ms}$ làm giảm hiệu năng rõ rệt do phanh muộn.
- **Kết quả**: Đạt thành tích cạnh tranh trên 3 cấu hình đường đua/xe khác nhau (Audi TT Cup đạt 1:15.913, so với mốc con người ghi nhận 1:16.062). Độ lệch chuẩn thời gian vòng đua qua 10 vòng đạt $15\text{ ms}$.

### 2.2. Wurman et al. (Nature 2022): Gran Turismo Sophy (GT Sophy)
Công trình của Wurman et al. (Sony AI) xuất bản trên tạp chí *Nature*, vol. 602, mang tên *"Outracing champion Gran Turismo drivers with deep reinforcement learning"*.
- **Phương pháp**: Sử dụng thuật toán Quantile Regression Soft Actor-Critic (QR-SAC) phân phối với 32 quantiles, cập nhật TD $N$-bước ($N=5$ hoặc $N=7$). Mạng nơ-ron sâu 4 lớp (2048 units, ReLU, dropout 0.1 ở policy). Tần số điều khiển $10\text{ Hz}$.
- **Không gian quan sát**: Trích xuất trạng thái động lực học gồm vận tốc 3D, gia tốc, tải trọng và góc trượt của từng lốp trong 4 bánh, biểu diễn đường đua qua 60 điểm 3D trên mép trái, tim đường và mép phải (phủ tầm nhìn $\sim 6\text{s}$ phía trước). GT Sophy không sử dụng racing line định trước trong quan sát. Hành động gồm góc lái và 1 trục pedal gộp $[-1, 1]$.
- **Chiến lược huấn luyện**: Huấn luyện phân tán trên nền tảng PlayStation, áp dụng kỹ thuật lấy mẫu phân tầng (Stratified Sampling) trên tập kịch bản hỗn hợp (khởi đầu ngẫu nhiên, bám đuôi, đối kháng).
- **Đặc tính điều khiển**: Trong tài liệu bổ sung (Extended Data Fig. 2b–c), nhóm tác giả ghi nhận GT Sophy xuất hiện hiện tượng dao động vô lăng ở tần số $10\text{ Hz}$ (*steering chattering*), trong khi các tay đua con người điều khiển mượt hơn ở $60\text{ Hz}$. GT Sophy chấp nhận điều này vì mục tiêu tối ưu hóa thời gian vòng đua và khả năng đối đầu trực tiếp.

### 2.3. Chisari et al. (IEEE ICRA 2021): Sim-to-Real trong Đua xe Tự hành
Chisari, Liniger, Rupenyan, Van Gool, và Lygeros (ETH Zürich) công bố *"Learning from Simulation, Racing in Reality"* tại IEEE ICRA 2021.
- **Phương pháp & Thách thức**: Giải quyết bài toán chuyển giao từ mô phỏng sang thực tế (Sim-to-Real Transfer) cho xe đua mô hình RC tỉ lệ 1:43 (nền tảng ORCA) chạy ở giới hạn vật lý.
- **Không gian hành động**: Đề xuất không gian hành động nâng bậc (*lifted action space*) - điều khiển đạo hàm bậc một của lệnh lái ($\Delta u_t$) kết hợp điều chuẩn policy (*policy regularization*) nhằm giảm rung lắc trên phần cứng thực tế.
- **Kết quả**: Áp dụng kỹ thuật xáo trộn miền (Domain Randomization) về hệ số ma sát lốp và quán tính kết hợp fine-tune trên đường đua thực, tác tử RL giảm $50\%$ lỗi vi phạm giới hạn vạch đường so với bộ điều khiển Model Predictive Contouring Control (MPCC).

### 2.4. Mysore et al. (IEEE ICRA 2021): Điều khiển Mượt với CAPS
Mysore, Mabsout, Mancuso, và Saenko công bố *"Regularizing Action Policies for Smooth Control with Reinforcement Learning"* (CAPS) tại IEEE ICRA 2021.
- **Hiện tượng Bang-bang / PWM trong RL**: Nhóm tác giả chỉ ra rằng việc tối ưu Markov từng bước ($\max \mathbb{E}[Q(s, a)]$) khiến các thuật toán actor-critic liên tục (PPO, SAC, DDPG) có xu hướng hội tụ về chính sách dạng đóng/mở cực trị (Bang-bang action), tạo ra tín hiệu điều khiển dao động biên độ lớn tương tự PWM, gây hao mòn cơ học.
- **CAPS Loss**: Đề xuất hai số hạng điều chuẩn bổ sung vào hàm mất mát của Actor:
  1. *Độ mượt thời gian (Temporal Smoothness Loss)*:
     $$L_T = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \|\pi(s_t) - \pi(s_{t-1})\|^2 \right]$$
  2. *Độ mượt không gian (Spatial Smoothness Loss)*:
     $$L_S = \mathbb{E}_{s_t \sim \mathcal{D}, \bar{s}_t \sim \mathcal{N}(s_t, \sigma_s^2 I)} \left[ \|\pi(s_t) - \pi(\bar{s}_t)\|^2 \right]$$
- **Ý nghĩa đối với đồ án**: Công trình cung cấp cơ sở lý luận để phân biệt giữa hành vi điều khiển chiến thuật (đạp ga trên đoạn thẳng, phanh tại điểm phanh) và hiện tượng dao động tần số cao do cấu trúc hàm thưởng/quan sát.

### 2.5. Bảng Đối sánh Phương pháp luận

| Thuộc tính So sánh | Fuchs et al. (2021) | GT Sophy (Wurman 2022) | Chisari et al. (2021) | Đồ án Này (Commit `ee99f9d` / `7853722`) |
| :--- | :--- | :--- | :--- | :--- |
| **Thuật toán RL** | SAC (Off-policy) | QR-SAC (Off-policy, 32 quantiles) | SAC (Off-policy) | **PPO Liên tục (On-policy, GAE)** |
| **Tần số quyết định** | $10\text{ Hz}$ | $10\text{ Hz}$ | $20 - 50\text{ Hz}$ | **$10\text{ Hz}$ ($\Delta t = 0.1\text{s}$)** |
| **Không gian hành động** | 2D: Lái + [Ga/Phanh gộp] | 2D: Lái + [Ga/Phanh gộp] | 2D: $\Delta\text{Steer}, \Delta\text{Throttle}$ | **3D: Lái, Ga riêng, Phanh riêng** |
| **Phân phối Policy** | Gaussian squashed tanh | Gaussian squashed tanh | Gaussian squashed tanh | **Gaussian squashed tanh + Jacobian Fix** |
| **Racing Line trong Obs**| Không (Zero prior) | Không (Zero prior) | Có (MPCC contouring frame) | **Không (Nhánh `pure`, Zero prior)** |
| **Quy tắc Biên đường** | Phạt va chạm tường $\propto v^2$ | Phạt ra ngoài lề $\propto v^2$ | Ràng buộc biên mềm | **Quy tắc 4 bánh FIA (Terminate)** |
| **Mốc Đối chứng Chính** | Kỷ lục con người | Tay đua vô địch con người | MPCC dựa trên mô hình xe đạp | **Xe Bóng ma Giải tích (Convex QP + Lipp-Boyd)** |
| **Xử lý Chattering** | Không xử lý | Không xử lý | Lifted Action + Regularization | **Slew Rate Limiter + Phạt $\Delta\text{steer}^2$** |

---

## 3. MÔ HÌNH HÓA TOÁN HỌC VÀ HỆ THỐNG MÔ PHỎNG (SYSTEM FORMULATION)

### 3.1. Hệ Tọa độ và Ký hiệu Toán học Thống nhất

*Bảng 1: Ký hiệu toán học thống nhất trong báo cáo.*

| Ký hiệu | Ý nghĩa vật lý / Kỹ thuật | Đơn vị | Tương ứng trong mã nguồn |
| :--- | :--- | :---: | :--- |
| $t, \Delta t$ | Chỉ số bước thời gian, độ dài bước tích phân ($\Delta t = 0.1\text{s}$) | –, $\text{s}$ | `step`, `dt` |
| $s_t$ | Tọa độ tiến trình dọc tim đường theo hệ Frenet | $\text{m}$ | `track_s` |
| $d_t$ | Độ lệch khoảng cách vuông góc so với tim đường | $\text{m}$ | `lat_offset` |
| $w$ | Nửa bề rộng mặt đường nhựa ($w = 4.5\text{ m}$ tại map half) | $\text{m}$ | `half_width` |
| $L$ | Tổng chiều dài một vòng đua ($L = 2,838.7\text{ m}$) | $\text{m}$ | `circuit_length` |
| $v_t$ | Vận tốc vô hướng của xe ($v_t = \|\mathbf{v}_t\|$) | $\text{m/s}$ (hoặc $\text{km/h}$) | `speed_km_h` |
| $\kappa(s)$ | Độ cong hình học của tim đường hoặc racing line | $\text{m}^{-1}$ | `track_kappa` |
| $\mathbf{o}_t$ | Vector quan sát đầu vào tại bước $t$ (35 chiều) | – | `obs` |
| $\mathbf{a}_t$ | Vector hành động thực thi: $(\delta_t, a^{\text{thr}}_t, a^{\text{brk}}_t)$ | – | `action` |
| $r_t$ | Phần thưởng vô hướng nhận được tại bước $t$ | – | `step_reward` |
| $T_{\text{lap}}$ | Thời gian hoàn thành vòng đua đo được trên mô phỏng | $\text{s}$ | `lap_time` |
| $T^*$ | Thời gian vòng đua lý thuyết tối ưu tích phân từ profile | $\text{s}$ | `theoretical_best_lap_time` |
| $\beta$ | Góc trượt thân xe (Sideslip Angle, $\beta = \arctan(v_y / v_x)$) | $\text{độ}$ ($^\circ$) | `beta_deg` |
| $a_{\text{lat}}, a_{\text{long}}$ | Gia tốc ngang và dọc của xe | $\text{G}$ hoặc $\text{m/s}^2$ | `lateral_accel`, `longitudinal_accel` |

### 3.2. Động lực học Phương tiện trong Bullet Physics và MetaDrive
Hệ thống sử dụng mô hình xe thể thao GT3 (`GT3RaceCar`) được xây dựng trên nền tảng Bullet Physics Raycast Vehicle của MetaDrive:
- **Khung vỏ & Dẫn động 4 Bánh**: 
  - Khối lượng thân xe cố định ở $m = 1100\text{ kg}$ trong động cơ Bullet Physics.
  - Vị trí trọng tâm: Cách trục trước $1.052\text{ m}$, trục sau $1.417\text{ m}$, chiều cao trọng tâm $h_{\text{cg}} \approx 0.55\text{ m}$.
  - Cơ chế ga (`base_vehicle._apply_throttle_brake`): Khi đạp ga ($\text{pedal} \ge 0$), Bullet áp dụng lực kéo $F_{\text{drive}} = \text{max\_engine\_force} \times \text{pedal}$ lên cả 4 bánh xe (dẫn động 4 bánh toàn thời gian AWD). Gia tốc dọc lý thuyết ở dải không trượt là $a_x = 4F/m$ ($4 \times 850 / 1100 = 3.09\text{ m/s}^2$, khớp số đo telemetry).
  - Giới hạn phanh dọc: Do trọng tâm cao ($0.55\text{ m}$) và lệch về trước, khi phanh gấp với gia tốc $a_{\text{brake}} > 1.7\text{G}$, tải trọng bánh sau giảm mạnh, gây mất ổn định đuôi xe. Do đó, lực phanh thực nghiệm được khống chế ở $15.0\text{ m/s}^2$.
- **Mô hình Tiếp xúc Lốp & Cấu hình Ma sát (F16)**:
  - Bullet tính toán lực bám thông qua ma sát tiếp xúc Coulomb: $F_{\text{fric}} \le \mu \cdot F_z$.
  - Trong quá trình phát triển trước commit `7853722`, cấu hình xe nạp qua wrapper gặp lỗi con trỏ bánh xe khiến hệ số ma sát lốp trước giữ ở mức $\mu_{\text{front}} = 0.90$, trong khi lốp sau nhận $\mu_{\text{rear}} = 1.35$. Đặc tính này tạo ra xu hướng thiếu lái (Understeer). Để bảo toàn tính nhất quán môi trường của mô hình tham chiếu `ref_111s`, cấu hình `gt3_anchor` giữ nguyên trạng thái động lực học này cho cả tác tử PPO và tác tử đối chứng Ghost.
- **Quy tắc Biên đường 4 Bánh (FIA Track Limits)**: Tác tử được coi là hợp lệ khi có ít nhất một bánh xe nằm trong bề rộng mặt đường nhựa ($|d_t| \le w + w_{\text{vehicle}}/2 = 4.5 + 0.925 = 5.425\text{ m}$). Khi cả 4 bánh xe hoàn toàn rời khỏi mặt đường và tiếp xúc bãi cỏ, môi trường kích hoạt cờ va chạm và kết thúc episode (`terminate`).

### 3.3. Không gian Hành động và Bộ Trộn ECU $C^1$
Không gian hành động gồm 3 chiều tách rời:
$$\mathbf{a}_t = [\delta_t, a_t^{\text{thr}}, a_t^{\text{brk}}] \in [-1, 1] \times [0, 1] \times [0, 1]$$
Trong đó $\delta_t$ là lệnh góc đánh lái chuẩn hóa (ánh xạ tới góc quay bánh trước tối đa $\pm 20^\circ$), $a_t^{\text{thr}}$ và $a_t^{\text{brk}}$ là lệnh chân ga và chân phanh độc lập.

Tín hiệu điều khiển được giới hạn tốc độ xoay lái (Slew Rate Limiter, tối đa $0.18/\text{step} \iff 3.6^\circ/\text{step}$) và xử lý qua bộ trộn pedal liên tục $C^1$:
$$\text{pedal}_{\text{applied}} = a_t^{\text{thr}} \cdot (1 - a_{t, \text{eff}}^{\text{brk}}) - a_{t, \text{eff}}^{\text{brk}}$$
với hàm vùng chết phanh liên tục:
$$a_{t, \text{eff}}^{\text{brk}} = \begin{cases} 0, & a_t^{\text{brk}} \le 0.015 \\ \frac{a_t^{\text{brk}} - 0.015}{1.0 - 0.015}, & a_t^{\text{brk}} > 0.015 \end{cases}$$

### 3.4. Không gian Quan sát Thuần (35D Pure Obs)
Vector quan sát 35 chiều của mô hình tham chiếu `ref_111s` không chứa thông tin racing line giải tích:
- **Trạng thái Động học Nội tại (11 chiều)**: Vận tốc dọc $v_x / 40.0$, vận tốc ngang $v_y / 20.0$, tốc độ góc quay thân xe $\dot{\psi} / 2.0$, gia tốc $a_x / 10.0, a_y / 10.0$, góc trượt thân xe $\beta$, Roll, Pitch, hành động bước trước $(\delta_{t-1}, a_{t-1}^{\text{thr}}, a_{t-1}^{\text{brk}})$.
- **Hình học Sai lệch Tim đường Tức thời (4 chiều)**: $\sin(\Delta\theta), \cos(\Delta\theta)$, khoảng cách lệch ngang $d_t / w$, vận tốc lệch ngang $\dot{d}_t / 20.0$.
- **Tầm nhìn Hình học Tim đường Đón đầu (20 chiều)**: 5 điểm ngắm phía trước tại $[10, 25, 45, 70, 100]\text{ m}$; mỗi điểm cung cấp $(\Delta x_k / 50.0, \Delta y_k / 20.0, \sin\Delta\theta_k, \kappa_k \times 50.0)$.

### 3.5. Hàm Phần thưởng
Hàm phần thưởng được cấu thành từ các thành phần:
$$r_t = r_t^{\text{prog}} + r_t^{\text{center}} + r_t^{\text{steer}} + r_t^{\text{idle}} + R^{\text{terminal}}$$

1. **Phần thưởng Tiến độ Frenet ($r_t^{\text{prog}}$)**:
   $$r_t^{\text{prog}} = c_{\text{prog}} \cdot (s_t - s_{t-1}), \quad c_{\text{prog}} = 0.15$$
2. **Thành phần Phạt Lệch Tim đường ($r_t^{\text{center}}$)**:
   $$r_t^{\text{center}} = -c_{\text{center}} \cdot \left(\frac{d_t}{w}\right)^2, \quad c_{\text{center}} = 0.04$$
3. **Phạt Rung lắc Tay lái ($r_t^{\text{steer}}$)**:
   $$r_t^{\text{steer}} = -c_{\text{steer}} \cdot (\delta_t - \delta_{t-1})^2, \quad c_{\text{steer}} = 0.10$$
4. **Phạt Vận tốc Thấp ($r_t^{\text{idle}}$)**: Phạt cố định $-0.05$ nếu vận tốc phương tiện $< 5.0\text{ km/h}$.
5. **Thưởng/Phạt Kết thúc Episode ($R^{\text{terminal}}$)**: Phạt văng đường $R^{\text{crash}} = -20.0$. Thưởng về đích:
   $$R^{\text{finish}} = R_{\text{base}} \cdot \exp\left(-\frac{T_{\text{lap}} - T_0}{\tau}\right)$$
   với $R_{\text{base}} = 50.0, T_0 = 110.0\text{ s}, \tau = 25.0\text{ s}$.

### 3.6. Thuật toán PPO và Hiệu chỉnh Jacobian
Mạng nơ-ron Actor xuất ra kỳ vọng $\boldsymbol{\mu}_\theta(\mathbf{o})$ và tham số log độ lệch chuẩn $\log \boldsymbol{\sigma}$. Biến ngẫu nhiên $u_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ được đưa qua hàm kích hoạt $\tanh$ để ép vào khoảng $[-1, 1]$: $a_i = \tanh(u_i)$.

Mật độ xác suất của hành động $\pi_\theta(\mathbf{a}|\mathbf{o})$ được tính có bù trừ định thức Jacobian (Haarnoja et al., 2018):
$$\log \pi_\theta(\mathbf{a}|\mathbf{o}) = \sum_{i=1}^3 \left[ \log \mathcal{N}(u_i; \mu_i, \sigma_i^2) - \log(1 - \tanh^2(u_i) + \delta_{\text{reg}}) \right]$$
với $\delta_{\text{reg}} = 10^{-6}$.

---

## 4. TÁC TỬ ĐỐI CHỨNG GIẢI TÍCH VÀ XE BÓNG MA (ANALYTICAL BASELINE & GHOST CAR)

Nghiên cứu phát triển động cơ giải tích độc lập hai tầng (`RacingBenchmarkEngine`) kết hợp bộ điều khiển bám vết (`OptimalRacingAgent` / `GhostCar`) làm mốc đối chứng:

### 4.1. Tối ưu Quỹ đạo Cực tiểu Độ cong (Convex QP Minimum Curvature)
Theo Heilmeier et al. (2020), quỹ đạo tối thiểu hóa tích phân bình phương độ cong trên toàn mạch:
$$\min_{\boldsymbol{\alpha}} \int_0^L \kappa^2(s) \, ds \quad \approx \quad \min_{\boldsymbol{\alpha}} \frac{1}{2} \boldsymbol{\alpha}^T \mathbf{Q} \boldsymbol{\alpha} + \mathbf{c}^T \boldsymbol{\alpha}$$
thỏa mãn ràng buộc biên hình học:
$$-w_{\text{eff}} \le \alpha_i \le w_{\text{eff}}, \quad \forall i \in \{1..N\}$$
trong đó $\alpha_i$ là độ dịch chuyển ngang khỏi tim đường tại điểm rời rạc $i$, và $w_{\text{eff}} = 4.05\text{ m}$ là biên an toàn đã trừ đi nửa bề rộng xe ($0.925\text{ m}$) trên mặt đường rộng $4.5\text{ m}$. Bài toán Convex QP được giải qua thư viện `OSQP`.

### 4.2. Phân bố Vận tốc Tối ưu Thời gian (Lipp-Boyd Velocity Profile)
Dựa trên thuật toán của Lipp & Boyd (2014), phân bố tốc độ dọc tối ưu $v^*(s)$ được tính qua 3 lượt tích phân:
1. **Giới hạn Gia tốc Ngang**:
   $$v_{\text{corner}}^*(s) = \sqrt{\frac{a_{\text{lat, max}}}{|\kappa_{\text{line}}(s)| + \epsilon}}$$
2. **Lượt Tích phân Lùi (Phanh)**:
   $$v_i^* = \min\left( v_i^*, \sqrt{(v_{i+1}^*)^2 + 2 \cdot a_{\text{brake}} \cdot \Delta s} \right)$$
3. **Lượt Tích phân Tiến (Tăng tốc)**:
   $$v_i^* = \min\left( v_i^*, \sqrt{(v_{i-1}^*)^2 + 2 \cdot a_{\text{engine}} \cdot \Delta s} \right)$$
Thời gian vòng đua lý thuyết chất điểm:
$$T^* = \int_0^L \frac{1}{v^*(s)} \, ds$$

### 4.3. Hiệu chuẩn Thực nghiệm Tham số Xe Bóng ma (`gt3_anchor`)
Để mô hình Ghost phản ánh chính xác giới hạn vật lý của động cơ Bullet Physics (Commit `7853722`), các tham số được đo đạc trực tiếp qua thực nghiệm mô phỏng:
- **Tăng tốc dọc**: $a_{\text{engine}} = 3.09\text{ m/s}^2$ ($a_x = 4F_{\text{drive}} / m = 4 \times 850 / 1100$).
- **Phanh dọc an toàn**: $a_{\text{brake}} = 15.0\text{ m/s}^2$ (khống chế dưới ngưỡng $1.7\text{G}$ để tránh nhấc trục sau).
- **Gia tốc ngang ổn định**: $a_{\text{lat}} = 0.95\text{G} = 9.32\text{ m/s}^2$ (ngưỡng cao nhất giúp Ghost hoàn thành $100\%$ vòng sạch không văng lề).

Cấu hình hiệu chuẩn `gt3_anchor` đem lại thời gian lý thuyết $T^* = 110.62\text{ s}$, và thành tích chạy trong simulator đạt Standing Lap $118.20\text{ s}$ và Flying Lap $112.80\text{ s}$ ($3/3$ vòng đánh giá không va chạm).

---

## 5. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH DỮ LIỆU (EXPERIMENTS & RESULTS)

Kết quả đo lường được thực hiện bằng tập lệnh `training/evaluate.py` trên sa bàn Silverstone (chiều rộng $9.0\text{ m}$, 18 góc cua), áp dụng luật 4 bánh FIA, mỗi episode chạy 2 vòng liên tiếp (Vòng 1: Standing Start; Vòng 2: Flying Lap). Số liệu thô lưu tại `logs/anchor/`.

### 5.1. Bảng Tổng hợp Kết quả Đánh giá Độc lập

*Bảng 2: So sánh hiệu năng giữa PPO Agent và Mốc Chuẩn Giải tích (20 episode độc lập, Seed 100–119).*

| Mô hình / Cấu hình Thực nghiệm | Nhánh Git / Commit Nguồn | Vòng Đứng (Standing) $T_1$ [s] | Vòng Chạy đà (Flying) $T_2$ [s] | Lap Nhanh nhất (Best) [s] | Tỷ lệ Vòng Sạch (Clean Rate) | Vận tốc Trung bình [km/h] | Sai số Racing Line (MAE) [m] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PPO `ref_111s` (Anchor)** | `ee99f9d` / `7853722` | **$114.78 \pm 0.40$** | **$109.44 \pm 0.50$** | **$108.60$** | **$18/20$ ($90\%$)** | **$90.26$** | $2.12 \pm 0.15$ |
| **PPO Seed 43 (Độc lập lặp lại)** | `feature/111s-honest-anchor` | $115.18 \pm 0.43$ | $109.96 \pm 0.69$ | $108.60$ | **$19/20$ ($95\%$)** | $89.73$ | $2.08 \pm 0.12$ |
| **Ghost Hiệu chỉnh (`gt3_anchor`)** | `feature/111s-honest-anchor` | $118.20 \pm 0.00$ | $112.80 \pm 0.00$ | $112.80$ | **$3/3$ ($100\%$)** | $86.97$ | **$0.37 \pm 0.01$** |
| **Ghost Cũ (Legacy v22)** | `feature/111s-honest-anchor` | $119.50 \pm 0.00$ | $114.20 \pm 0.00$ | $114.20$ | $2/2$ ($100\%$) | $85.88$ | $0.33 \pm 0.01$ |
| **Fine-tune Luật 4 Bánh (`f4d8ef9`)**| `feature/111s-honest-anchor` | $113.56 \pm 0.38$ | $108.98 \pm 0.24$ | $108.70$ | $8/20$ ($40\%$) | $89.22$ | $2.31 \pm 0.18$ |
| **Ablation Bỏ $r_{\text{center}}$ (Seed 42)**| Nhánh `exp/111s-no-rcenter` | Không hoàn thành | Không hoàn thành | $117.70$ (DNF) | **$0/20$ ($0\%$)** | $69.44$ | $2.75 \pm 0.45$ |
| **Ablation Bỏ $r_{\text{center}}$ (Seed 43)**| Nhánh `exp/111s-no-rcenter` | $116.85 \pm 0.45$ | $112.40 \pm 0.20$ | $112.20$ | **$2/20$ ($10\%$)** | $88.75$ | $2.56 \pm 0.38$ |

*(Ghi chú: Giá trị hiển thị dạng $\text{Mean} \pm \text{Std}$; số mẫu $n=20$ cho các tác tử RL, $n=3$ cho tác tử Ghost. "Vòng Sạch" được định nghĩa là episode không có thời điểm nào vi phạm luật 4 bánh hoặc va chạm).*

### 5.2. Phân tích Hiện tượng RL Vượt qua Nghiệm Tối ưu Giải tích
Dữ liệu Bảng 2 cho thấy tác tử PPO `ref_111s` đạt thời gian vòng flying $109.44 \pm 0.50\text{ s}$, thấp hơn Ghost hiệu chuẩn $3.36\text{ s}$ ($\sim 3.0\%$) và thấp hơn thời gian tích phân lý thuyết chất điểm $T^* = 110.62\text{ s}$. Kết quả lặp lại trên Seed 43 với Flying $109.96 \pm 0.69\text{ s}$ ($19/20$ vòng sạch) xác nhận tính ổn định của mô hình. 

Phân tích dữ liệu Telemetry cho thấy hai yếu tố giải thích cho sự chênh lệch này:
1. **Độ bám Động lực học Tức thời (Instantaneous Dynamic Grip)**:
   - Mô hình chất điểm của Ghost Car giả định giới hạn gia tốc ngang cố định $a_{\text{lat}} \le 0.95\text{G}$.
   - Đo đạc gia tốc thực tế của tác tử PPO (Phát hiện F20, F23): Phân vị 95% của gia tốc ngang thực tế đạt $a_{\text{lat, p95}} = 0.99\text{G}$, cực đại tức thời ghi nhận đạt $a_{\text{lat, max}} = 1.21\text{G}$.
   - **[Suy luận]**: Trong mô phỏng Bullet, hiện tượng dịch chuyển tải trọng động (Dynamic Load Transfer) nén lên hai bánh ngoài kết hợp với góc trượt thân xe ($\beta_{\text{p95}} = 3.5^\circ$, cực đại $18^\circ$) cho phép lốp phát sinh lực bám tức thời vượt ngưỡng ma sát tĩnh trong thời gian ngắn ($0.2 - 0.4\text{s}$). Thuật toán RL tối ưu hóa trực tiếp trên mô phỏng nên đã tận dụng được hiệu ứng động học này.
2. **Kỹ thuật Trail-Braking**:
   - Biểu đồ phối hợp gia tốc G-G cho thấy PPO duy trì áp lực phanh nhẹ ($10\% - 25\%$) đồng thời với việc bẻ lái vào cua, thay vì phanh thẳng hoàn toàn rồi mới đánh lái như bộ điều khiển của Ghost. Hành vi này dồn tải trọng lên trục trước, tạm thời tăng lực bám dẫn hướng của bánh trước khi chuyển hướng.

### 5.3. Độ Ổn định và Chế độ Lỗi (Failure Mode Analysis)
Mô hình `ref_111s` có $2/20$ episode gặp lỗi văng đường ($10\%$ rủi ro):
- Vị trí 1: Mét thứ $2637\text{ m}$ (Góc cua Vale - góc cua chậm sau đoạn thẳng dài, xe lao vào với vận tốc cao và trượt 4 bánh ra lề).
- Vị trí 2: Mét thứ $364\text{ m}$ (Góc cua Farm Curve).
- Khi fine-tune với điều kiện kết thúc nghiêm ngặt ngay khi chạm lề (`ref_111s_ft_strict`, Commit `f4d8ef9`), thời gian vòng flying giảm xuống $108.98\text{ s}$, nhưng tỷ lệ vòng sạch giảm xuống $40\%$ ($8/20$ sạch). Các vụ văng đường tập trung tại đoạn $1734 - 1807\text{ m}$ ($10/12$ lần), cho thấy việc ép tốc độ mà không có ràng buộc độ mượt hành động dễ khiến tác tử rơi vào vùng phân kỳ động lực học.

---

## 6. NGHIÊN CỨU BÓC TÁCH VÀ THẢO LUẬN (ABLATIONS & DISCUSSION)

### 6.1. Bóc tách Vai trò của Thành phần Phạt Lệch Tim đường ($r_{\text{center}}$)
Thí nghiệm bóc tách (Ablation A13, Nhánh `exp/111s-no-rcenter`): giữ nguyên toàn bộ siêu tham số và hạt giống ngẫu nhiên, chỉ đặt hệ số $c_{\text{center}} = 0.0$.
- **Kết quả đo được**:
  - Tại Seed 42: Tỷ lệ vòng sạch giảm từ $90\%$ về **$0\%$ ($0/20$ vòng hoàn thành)**. Xe liên tục văng lề tại mét $363 - 365\text{ m}$ (13 lần lặp lại tại cùng một góc cua).
  - Tại Seed 43: Tỷ lệ vòng sạch giảm từ $95\%$ về **$10\%$ ($2/20$ vòng hoàn thành)**.
- **Phân tích cơ chế**:
  - Khi không có $r_{\text{center}}$, không gian khám phá mở rộng trong điều kiện mặt đường hẹp ($4.5\text{ m}$). Hệ số chiết khấu $\gamma = 0.99$ tương đương tầm nhìn hiệu dụng khoảng 100 bước ($10\text{s}$). 
  - Tại các góc cua gắt, phần thưởng tiến độ tức thời $r_{\text{prog}}$ thúc đẩy xe cắt góc sớm (Early Apex) để tối đa hóa tiến độ trước mắt, khiến xe mất quỹ đạo thoát cua và thiếu không gian mặt đường ở lối ra.
  - $r_{\text{center}}$ đóng vai trò như một lực kéo mềm giữ xe trong phạm vi an toàn. Ở bản `ref_111s`, sai số lệch racing line (Line MAE) vẫn đạt $2.12\text{ m}$, cho thấy xe vẫn mở rộng cua khi cần thiết.

### 6.2. Hiện tượng Rung giật Điều khiển (5 Hz PWM Chattering)
Dữ liệu Telemetry ghi nhận đặc tính điều khiển của PPO `ref_111s` (Commit `7853722`):
- **Đảo chiều lái**: Trung bình $60$ lần đảo chiều vô lăng trên mỗi $100$ bước, biên độ biến thiên trung bình $|\Delta\text{steer}| = 0.074$.
- **Chuyển đổi Ga - Phanh (5 Hz PWM)**: PPO `ref_111s` có tần suất chuyển đổi giữa ga và phanh là $4.7$ lần / $100$ bước (so với Ghost Car là $0.2$ lần / $100$ bước).

*Bảng 3: Chỉ số điều khiển giữa PPO Agent và Ghost Car.*

| Chỉ số Đánh giá | PPO `ref_111s` (Map Half) | PPO P2 v2 (Map Full, GT3Real) | Ghost Hiệu chuẩn (`gt3_anchor`) |
| :--- | :---: | :---: | :---: |
| Số lần đảo chiều lái / 100 bước | **$60$** | **$74$** | **$16$** |
| Biến thiên góc lái trung bình $|\Delta\text{steer}|$ | $0.074$ | $0.121$ | $0.007$ |
| Tần suất đổi Ga $\leftrightarrow$ Phanh / 100 bước | **$4.7$** | **$20.3$** | **$0.2$** |
| Góc trượt thân xe $\beta$ (Phân vị 95% / Max) | $3.5^\circ / 18.0^\circ$ | $8.9^\circ / 52.7^\circ$ | $2.0^\circ / 3.8^\circ$ |

**Phân tích nguyên nhân và liên hệ lý thuyết CAPS (Mysore et al., 2021)**:
1. *Mạch dao động pedal 2 bước do `obs[6]`*:
   Vector quan sát nạp trực tiếp giá trị `last_pedal` vào Actor:
   $$\text{obs}[6] = a_{t-1}^{\text{thr}} \cdot (1 - a_{t-1}^{\text{brk}}) - a_{t-1}^{\text{brk}}$$
   Trong khi đó, hàm thưởng chỉ phạt độ giật lái ($r_{\text{steer}} = -0.1 \cdot (\Delta\delta)^2$) mà không có thành phần phạt độ giật pedal ($r_{\text{pedal}} = 0$). Mạng nơ-ron hình thành chu kỳ dao động 2 bước (bước chẵn đạp ga dương $+0.3$ đến $+0.7$, bước lẻ rà phanh nhẹ $-0.1$ đến $-0.3$). Dao động này gây dồn tải dọc liên tục, làm dao động tải trọng pháp tuyến bánh trước khi vào cua.
2. *Ảnh hưởng của vận tốc*:
   Gia tốc ngang tỷ lệ với bình phương vận tốc ($a_y \propto v^2 \delta$). Ở dải tốc độ cao (nhánh Phase 2, $> 140\text{ km/h}$), cùng một biên độ giật lái $|\Delta\delta| \approx 0.07$ phát sinh lực quán tính ngang lớn gấp gần 4 lần so với ở $88\text{ km/h}$, dễ đẩy lốp xe vào vùng bão hòa ma sát.

### 6.3. Bóc tách PPO + Racing Line (Ablation R14 / P2-line)
Trong nhánh thử nghiệm `p2_line` (Commit `11807e4`), thông tin sai lệch khoảng cách và vận tốc mục tiêu của racing line $(\Delta d_{\text{line}}, v^*)$ được đưa vào quan sát:
- Tác tử đạt vòng sạch sớm tại $1.15\text{M}$ bước (so với $2.15\text{M}$ bước của nhánh pure).
- Tuy nhiên, sau mốc $2.0\text{M}$ bước, thời gian vòng đua thoái hóa dần từ $146.6\text{s}$ về $173 - 181\text{s}$, vận tốc trung bình giảm từ $144\text{ km/h}$ xuống $115\text{ km/h}$.
- Nguyên nhân: Việc bám theo line định sẵn kết hợp với phạt văng đường nặng làm giảm xu hướng khám phá các giới hạn bám cao hơn của tác tử.

### 6.4. Các Lỗi Kỹ thuật Đã Xử lý trong Huấn luyện PPO
1. **Kẹt Độ lệch chuẩn Khám phá (F1 - Actor Log-std Freeze)**:
   - Forward pass dùng `torch.clamp(self.actor_logstd, min=-2.3, max=0.0)`. Khi giá trị trôi ra ngoài khoảng, gradient triệt tiêu khiến tham số bị kẹt ở biên. Tại commit `2055469`, độ lệch chuẩn phanh bị kẹt ở $\sigma_{\text{brake}} = 1.0$, gây phanh ngẫu nhiên liên tục.
   - Khắc phục: Thay clamp trong forward bằng phép chiếu trực tiếp tham số sau bước cập nhật Adam: `self.ac.actor_logstd.clamp_(MIN, MAX)`.
2. **Bù trừ Jacobian cho Squashed Gaussian (F2)**: Đồng bộ hóa biên hành động $\text{ACTION\_BOUND} = 0.999$ và định thức Jacobian trong tính toán log-likelihood của PPO.
3. **Hiện tượng Sập Khám phá khi Giảm Entropy (F14 & I9)**: Giảm hệ số entropy về $0.001$ khiến phương sai đánh lái sập về $0.10$ chỉ sau $150\text{k}$ bước, làm tác tử kẹt trong cực tiểu cục bộ và liên tục va chạm tại mét $1306\text{ m}$. Khắc phục bằng cách cố định `ent_coef = 0.01` và đặt trần độ lệch chuẩn riêng cho từng trục hành động.
4. **Khởi tạo Thiên vị Chân phanh (F14)**: Đặt trọng số thiên vị ban đầu của lớp phanh âm (`bias[2] = -1.6`), giảm xác suất phanh ngẫu nhiên ban đầu từ $50\%$ xuống dưới $5\%$.

### 6.5. Khảo sát Mở rộng: Đua Đa Tác tử (MARL) và Lực ép Khí động học (Downforce)
Bên cạnh trục thực nghiệm chính trên môi trường `gt3_anchor`, nhóm nghiên cứu đã tiến hành khảo sát mở rộng trên hai nhánh thử nghiệm:
1. **Khảo sát Lực ép Khí động học (`feature/f1-downforce`)**:
   - Thử nghiệm bổ sung phương trình lực ép khí động học tỷ lệ với bình phương vận tốc:
     $$F_{\text{down}} = m \cdot g \cdot \left(\frac{v}{200}\right)^2$$
   - Lực ép làm tăng tải trọng thẳng đứng $F_z$ ở tốc độ cao mà không làm tăng khối lượng quán tính $m$, giúp mở rộng giới hạn ma sát ngang $F_{y, \text{max}} = \mu F_z$. Mô hình đạt vận tốc $301\text{ km/h}$ và hoàn thành vòng đua mà không văng lề.
   - **Ghi chú phương pháp**: Việc bổ sung downforce nhân tạo làm thay đổi đặc tính động lực học so với xe thể thao GT3 tiêu chuẩn. Do đó, mô hình này không dùng để đối chuẩn với mốc giải tích của xe GT3 ban đầu.
2. **Khảo sát Đua Đa Tác tử 6 Xe (`origin/feature/multicar-selfplay`)**:
   - Thử nghiệm 6 xe cùng xuất phát trên sa bàn mở rộng 3 làn ($13.5\text{ m}$).
   - Áp dụng các cơ chế: phạt va chạm đối xứng ($-2.0$ khi chạm, $-0.2/\text{step}$ tiếp xúc), cooldown vượt xe 50 bước ($5.0\text{s}$), và cơ chế hồi sinh tại chỗ (in-place respawn) với 150 bước miễn trừ xếp hạng.
   - **Ghi chú phương pháp**: Quá trình huấn luyện trải qua nhiều lần tinh chỉnh tham số giữa các checkpoint; số liệu thu được mang tính chất khảo sát định tính về hành vi tương tác nhóm.

### 6.6. Bảng Tổng hợp So sánh Các Chuỗi Thực Nghiệm

*Bảng 4: Tổng hợp định lượng các chuỗi thực nghiệm trong dự án.*

| Mã Run / Nhánh Nguồn | Cấu hình & Biến số Khảo sát | Tổng Bước Huấn Luyện | Tỷ lệ Hoàn thành Cuối | Thời gian Vòng [s] | Tỷ lệ Vòng Sạch | Nhận định Kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`ref_111s` (Mô hình Chính)** (`ee99f9d`) | GT3, 35D pure obs, 3D action, map half ($4.5\text{m}$), $r_{\text{center}} = 0.04$ | $2.15\text{M}$ steps | **$100.0\%$** | Flying TB: **$109.4\text{s}$** (Best $108.6\text{s}$) | **$18/20$ ($90\%$)** | Mô hình tham chiếu chính: Khai thác gia tốc ngang tức thời ($1.21\text{G}$) và trail-braking, hoàn thành ổn định dưới luật 4 bánh FIA. |
| **PPO Seed 43 (Kiểm chứng)** (`7853722`) | Cùng cấu hình `ref_111s`, thay đổi hạt giống ngẫu nhiên | $2.15\text{M}$ steps | **$100.0\%$** | Flying TB: **$110.0\text{s}$** (Best $108.6\text{s}$) | **$19/20$ ($95\%$)** | Xác nhận tính lặp lại độc lập của thuật toán PPO trên hạt giống ngẫu nhiên khác nhau. |
| **`111s_no_rcenter`** (`exp/...`) | Giữ nguyên `ref_111s`, bỏ phạt tim đường $c_{\text{center}} = 0.0$ | $2.15\text{M}$ steps | $0.0\%$ | $112.2\text{s}$ (DNF) | **$0/20$ ($0\%$)** ở S42; **$2/20$ ($10\%$)** ở S43 | Bóc tách: Thiếu $r_{\text{center}}$, tầm nhìn ngắn của $\gamma = 0.99$ khiến xe sa bẫy Early Apex và văng lề ở mét 363. |
| **`run_a0_legacy`** (`0180988`) | Baseline cũ: Lỗi F1 (σ kẹt trần), thiếu bù Jacobian F2 | $3.38\text{M}$ steps | $19.4\%$ | $147.5\text{s}$ (DNF) | **$0/20$ ($0\%$)** (Crash $100\%$) | Lỗi kẹt tham số độ lệch chuẩn phanh khiến tác tử không thể hoàn thành vòng đua. |
| **`p1_main`** (`7853722`) | Đã vá F1/F2; thưởng về đích hàm mũ; thử decay entropy | $2.66\text{M}$ steps | $46.0\%$ | $130.2\text{s}$ | Sạch ở $1.35\text{M}$; cuối run crash $100\%$ | Suy giảm phương sai khám phá quá sớm dẫn đến kẹt cục bộ tại góc cua mét thứ $1306\text{m}$. |
| **`p2_single` v2** (`819d3e1`) | Map 1:1, GT3RealCar, $v > 140\text{ km/h}$ | $3.32\text{M}$ steps | $66.8\%$ | Best sạch $143.4\text{s}$ | $0/10$ sạch ở cuối run | Dao động pedal 5 Hz và đảo lái tần số cao gây mất bám ở dải tốc độ cao ($a_y \propto v^2\delta$). |
| **`p2_line`** (`6df6ba9`) | Map 1:1, nạp Racing line và $v^*$ vào observation | $2.66\text{M}$ steps | $54.0\%$ | Best sạch $153.7\text{s}$ | Sạch sớm tại $1.15\text{M}$; sau $2\text{M}$ chậm dần | Racing line hỗ trợ an toàn giai đoạn đầu nhưng làm giảm vận tốc hội tụ dài hạn do tính bảo thủ của hàm thưởng. |
| **`run_v7`** (`logs/run_v7`) | GT3 3 làn đường, bổ sung phạt thời gian $r_{\text{time}} = -0.01/\text{step}$ | $963\text{k}$ steps | $27.3\%$ | $141.5\text{s}$ | Checkpoint $389\text{k}$: **$5/6$ ($83.3\%$)** sạch | Phạt thời gian tạo gradient thúc đẩy xe duy trì vận tốc qua góc cua thay vì rà phanh chạy chậm. |
| **`run_v8b`** (`logs/run_v8b`) | GT3 3 làn, 40D obs, Slew rate $0.18$ + Phạt mượt lái | $502\text{k}$ steps | $22.6\%$ | $135.4\text{s}$ | Checkpoint $123\text{k}$: **$5/6$ ($83.3\%$)** sạch | Mở rộng tầm nhìn đón đầu và điều chỉnh độ mượt lái giúp tăng tốc độ học vòng sạch. |
| **`run_f1c` (Khảo sát Downforce)** (`24fb2e4`) | Xe F1 có Downforce nhân tạo $F_{\text{down}} \propto v^2$, sa bàn 3 làn | $1.00\text{M}$ steps | $87.8\%$ | $109.4\text{s}$ (không cùng điều kiện GT3) | $2/2$ sạch ở các checkpoint cuối | Lực ép khí động học nhân tạo giúp xe ổn định ở tốc độ cao; không dùng để đối chuẩn với xe GT3 tiêu chuẩn. |
| **`multicar_f1e` (Khảo sát 6 xe)** (`cdfb928`) | 6 xe F1 tự chơi, phạt đối xứng, có hồi sinh tại chỗ, sa bàn 3 làn | $1.50\text{M}$ samples | Không đối chuẩn | $112.9\text{s}$ (không cùng điều kiện GT3) | $5.67 / 6$ xe hoàn thành | Khảo sát định tính về phối hợp đa tác tử; tham số được điều chỉnh qua nhiều lần fine-tune. |

---

### 6.7. Thảo Luận Kỹ Thuật và Đánh Giá Thực Nghiệm

1. **Vai trò của Thành phần Phạt Thời gian ($r_{\text{time}} = -0.01/\text{step}$)**:
   - Khi không có áp lực thời gian (`run_v5`, `run_v6`), hàm phạt văng đường ($\propto v^2$) tạo ra cực tiểu cục bộ: tác tử học cách rà phanh chạy chậm ở tốc độ $30 - 50\text{ km/h}$. Do thiếu động năng, xe mất lực hướng tâm khi vào cua dốc và trôi ra lề.
   - Việc bổ sung thành phần phạt trễ $r_{\text{time}} = -0.01$ mỗi bước ($-1.0$ điểm mỗi $10\text{s}$) tạo gradient ép xe phải duy trì vận tốc qua apex, giúp `run_v7` và `run_v8b` đạt tỷ lệ vòng sạch $83.3\%$ ở các mốc bước sớm.

2. **Khảo sát Lực ép Khí động học Downforce và Giới hạn Bài toán**:
   - Ở nhánh `run_f1c`, việc bổ sung $F_{\text{down}} = mg(v/200)^2$ làm tăng tải trọng thẳng đứng $F_z$ theo bình phương vận tốc, mở rộng giới hạn ma sát ngang $F_{y, \text{max}} = \mu F_z$ mà không tăng khối lượng quán tính $m$. Điều này giúp xe ôm cua ổn định ở tốc độ cao.
   - Tuy nhiên, mục tiêu cốt lõi của nghiên cứu là điều khiển xe thể thao GT3 ở giới hạn ma sát tự nhiên của lốp xe (không có cánh gió lớn). Việc bổ sung downforce làm thay đổi bài toán động lực học ban đầu. Do đó, nhóm nghiên cứu duy trì mô hình `ref_111s` trên xe GT3 tiêu chuẩn làm kết quả khoa học chính.

3. **Bài học về Tính Nhất quán của Môi trường Kiểm chuẩn**:
   - Các nhánh mở rộng (`multicar`, `f1`) cho thấy rủi ro khi điều chỉnh đồng thời nhiều biến số (thay đổi cấu hình xe, độ rộng làn, quy tắc va chạm) qua các lần chạy, gây khó khăn cho việc phân tích nguyên nhân - kết quả.
   - Việc đóng băng môi trường xe thể thao `gt3_anchor` trên nhánh `feature/111s-honest-anchor` đã đảm bảo tính nhất quán để thực hiện 20 episode kiểm thử độc lập và kiểm chứng lặp lại trên Seed 43 với độ tin cậy khoa học cao.

---

## 7. GIỚI HẠN NGHIÊN CỨU VÀ HƯỚNG PHÁT TRIỂN (LIMITATIONS & FUTURE WORK)

1. **Giới hạn Mô hình Sa bàn Silverstone**:
   - Sa bàn gồm 23 khối đường (18 góc cua cung tròn và 5 đoạn thẳng) được xấp xỉ thủ công ($R \in [25, 95]\text{ m}$) và khép kín qua thuật toán tối ưu hóa góc.
   - Sa bàn không chứa đường cong chuyển tiếp (Clothoid) và bỏ qua độ mấp mô địa hình 3D ($z = 0$). Chiều dài bị nén theo tỉ lệ xấp xỉ $\approx 1:2.075$ ($2,838.7\text{ m}$ so với $5,891\text{ m}$ thực tế).
   - Độ rộng đường tương đối sát thực tế: 3 làn $\times 4.8\text{ m} = 14.4\text{ m}$ (so với $15.0\text{ m}$ thực tế của Silverstone).

2. **Giới hạn Động lực học Xe Bullet**:
   - Mô hình Raycast vehicle chưa bao gồm lực cản không khí đầy đủ ($F_{\text{drag}} \propto v^2$).
   - Chưa mô phỏng sự biến thiên nhiệt độ và hao mòn lốp.
   - Thân xe dạng khối hộp cố định với trọng tâm cao ($0.55\text{ m}$) khiến xe dễ nhấc trục sau khi phanh trên $1.7\text{G}$.

3. **Hướng phát triển Tiếp theo**:
   - Tích hợp hàm mất mát CAPS đầy đủ ($L_T$ và $L_S$) trực tiếp vào hàm mục tiêu của Actor PPO để triệt tiêu hiện tượng dao động vô lăng $10\text{ Hz}$.
   - Áp dụng kỹ thuật Xáo trộn Miền (Domain Randomization) theo Chisari et al. (2021) trên hệ số ma sát lốp và độ trễ chấp hành nhằm chuẩn bị cho triển khai trên phần cứng xe mô hình thực tế.

---

## 8. KẾT LUẬN (CONCLUSION)

Đồ án đã triển khai và đánh giá hệ thống điều khiển xe đua thể thao tự hành ở giới hạn ma sát vật lý bằng thuật toán Học Tăng Cường PPO liên tục trên nền tảng mô phỏng Bullet Physics/MetaDrive.

Các kết quả chính bao gồm:
1. **Hiệu năng của PPO thuần**: Tác tử PPO với quan sát thuần hình học tim đường (`ref_111s`) đạt thời gian vòng flying $109.44 \pm 0.50\text{ s}$ với tỷ lệ vòng sạch $90\%$ qua 20 episode kiểm thử độc lập, và được xác nhận lặp lại trên Seed 43 ($109.96 \pm 0.69\text{ s}$, $95\%$ sạch).
2. **Giải thích chênh lệch với mô hình giải tích**: Hiện tượng PPO đạt thời gian thấp hơn mô hình giải tích chất điểm ($112.80\text{ s}$ Ghost) được giải thích bằng Model-Plant Mismatch: RL khai thác gia tốc ngang tức thời ($1.21\text{G}$) và kỹ thuật trail-braking tự sinh từ động lực học đa vật thể của Bullet Physics.
3. **Phân tích bóc tách hệ thống**: Xác định vai trò của thành phần phạt tim đường $r_{\text{center}}$ trong việc hạn chế bẫy cắt góc sớm, đồng thời phân tích cơ chế hình thành dao động pedal 5 Hz do quan sát `last_pedal` thiếu phạt độ mượt vi phân.
4. **Quy trình thực nghiệm chuẩn hóa**: Toàn bộ quy trình thử nghiệm và kiểm chuẩn được thực hiện trên môi trường đóng băng `gt3_anchor` dưới luật 4 bánh FIA, đảm bảo tính nhất quán và khách quan của các kết quả nghiên cứu.

---

## TÀI LIỆU THAM KHẢO (REFERENCES)

1. **Fuchs, F., Song, Y., Kaufmann, E., Scaramuzza, D., & Dürr, P.** (2021). Super-human performance in Gran Turismo Sport using deep reinforcement learning. *IEEE Robotics and Automation Letters (RA-L)*, 6(3), 4257–4264. [arXiv:2008.07971](https://arxiv.org/abs/2008.07971)
2. **Wurman, P. R., Barrett, S., Kawamoto, K., MacGlashan, J., Subramanian, K., Walsh, T. J., ... & Stone, P.** (2022). Outracing champion Gran Turismo drivers with deep reinforcement learning. *Nature*, 602(7896), 223–228. [DOI:10.1038/s41586-021-04357-7](https://doi.org/10.1038/s41586-021-04357-7)
3. **Chisari, E., Liniger, A., Rupenyan, L., Van Gool, L., & Lygeros, J.** (2021). Learning from simulation, racing in reality. In *2021 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 8035–8041). IEEE. [arXiv:2011.13332](https://arxiv.org/abs/2011.13332)
4. **Mysore, S., Mabsout, B., Mancuso, R., & Saenko, K.** (2021). Regularizing action policies for smooth control with reinforcement learning. In *2021 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1810–1816). IEEE. [arXiv:2012.06644](https://arxiv.org/abs/2012.06644)
5. **Heilmeier, A., Wischnewski, A., Hermansdorfer, L., Betz, J., Lienkamp, M., & Lohmann, B.** (2020). Minimum curvature trajectory planning and control for an autonomous race car. *Vehicle System Dynamics*, 58(10), 1497–1527.
6. **Lipp, T., & Boyd, S.** (2014). Minimum-time speed optimisation over a fixed path. *International Journal of Control*, 87(6), 1297–1311.
7. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
8. **Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P.** (2016). High-dimensional continuous control using generalized advantage estimation. In *Proceedings of the International Conference on Learning Representations (ICLR)*.
9. **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S.** (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In *International Conference on Machine Learning (ICML)* (pp. 1861–1870). PMLR.
10. **Seyde, T., Schwarting, W., Karaman, S., & Rus, D.** (2021). Is bang-bang control all you need? Solving continuous control with Bernoulli policies. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 27209–27221. [arXiv:2111.02552](https://arxiv.org/abs/2111.02552)
11. **Li, Q., Peng, Z., Zhang, Q., Qiu, C., Liu, C., & Zhou, B.** (2022). MetaDrive: Composing diverse driving scenarios for generalizable reinforcement learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 45(3), 3461–3475.
12. **Evans, M. C., Engelbrecht, H. A., & Jordaan, H. W.** (2023). High-speed autonomous racing using trajectory-aided deep reinforcement learning. *IEEE Robotics and Automation Letters (RA-L)*, 8(8), 4983–4990. [arXiv:2306.07003](https://arxiv.org/abs/2306.07003)

---

## PHỤ LỤC: DANH MỤC TRUY XUẤT DỮ LIỆU THỰC NGHIỆM

Bảng 5 cung cấp chỉ mục truy xuất các tệp dữ liệu, mô hình và nhật ký đánh giá trong kho lưu trữ Git:

*Bảng 5: Chỉ mục truy xuất tệp dữ liệu và nhật ký thực nghiệm.*

| Tên Dữ liệu / Mô hình | Nhánh Git & Commit SHA | Đường dẫn Tệp trong Repository | Mô tả Nội dung / Số liệu Đối sánh |
| :--- | :--- | :--- | :--- |
| **Mô hình Tham chiếu (`ref_111s`)** | `feature/111s-honest-anchor` (`ee99f9d`) | `models/ref_111s/best_agent.pt` | Trọng số mạng nơ-ron PPO 35D, đạt Standing $114.78\text{s}$, Flying $109.44\text{s}$. |
| **Nhật ký Đánh giá 20 Lap `ref_111s`** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_111s_anchor.json` | Log JSON chi tiết 20 episode kiểm thử độc lập dưới luật 4 bánh FIA. |
| **Nhật ký Đánh giá Ghost Hiệu chuẩn** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_ghost_anchor.json` | Đánh giá xe bóng ma `gt3_anchor` ($0.95\text{G} / 15.0 / 3.09$), đạt Flying $112.80\text{s}$. |
| **Nhật ký Đánh giá Ghost Legacy v22** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_ghost_legacy.json` | Đánh giá Ghost cũ tham số lý thuyết, đạt Flying $114.20\text{s}$. |
| **Mô hình & Nhật ký Lặp lại Seed 43** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_overnight_seed43_best.json` | Kiểm chứng tính lặp lại độc lập ($n=20$), đạt Flying $109.96\text{s}$ (19/20 sạch). |
| **Nhật ký Fine-tune Luật 4 Bánh** | `feature/111s-honest-anchor` (`f4d8ef9`) | `logs/anchor/eval_overnight_ft_best.json` | Kết quả fine-tune 1M bước, lap $108.98\text{s}$, tỷ lệ sạch $40\%$ (8/20 sạch). |
| **Nhật ký Bóc tách Bỏ $r_{\text{center}}$** | `exp/111s-no-rcenter` | `logs/anchor/eval_nocenter_s42.json`, `...s43.json` | Số liệu sụt giảm tỷ lệ sạch về $0\% - 10\%$ khi loại bỏ phạt lệch tim đường. |
| **Dữ liệu Profile Tối ưu Giải tích** | `feature/111s-honest-anchor` | `scripts/silverstone_optimal_profile.npz` | Quỹ đạo lồi Convex QP và phân bố vận tốc Lipp-Boyd ($T^* = 110.62\text{s}$). |
| **Nhật ký Huấn luyện GT3 3 Làn (`run_v8b`)** | `feature/oop-clean-physics-40d` (`0180988`) | `logs/run_v8b/training_metrics.json`, `checkpoint_eval.json` | 40D obs, phạt thời gian $-0.01$; checkpoint `clean_123k` đạt $83.3\%$ vòng sạch, lap $135.4\text{s}$. |
| **Nhật ký Khảo sát Downforce (`run_f1c`)** | `feature/f1-downforce` (`24fb2e4`) | `logs/run_f1c/training_metrics.json` | Thực nghiệm bổ sung lực ép khí động học $F_{\text{down}} \propto v^2$, lap $109.4\text{s}$ (sa bàn 3 làn). |
| **Nhật ký Khảo sát Đua 6 Xe (`multicar_f1e`)** | `feature/f1-downforce` (`cdfb928`) | `logs/multicar_f1e/training_metrics.json` | Khảo sát hành vi đa tác tử với cơ chế hồi sinh tại chỗ và phạt đối xứng. |
| **Nhật ký Baseline Ban đầu (`run_a0_legacy`)** | `feature/111s-honest-anchor` (`7853722`) | `logs/run_a0_legacy/training_metrics.json` | $3.38\text{M}$ bước với $100\%$ crash rate, minh họa lỗi actor_logstd clamp freeze. |
| **Nhật ký Phase 1 Main (`p1_main`)** | `feature/111s-honest-anchor` (`7853722`) | `logs/p1_main/training_metrics.json` | $2.66\text{M}$ bước, minh họa lỗi suy giảm phương sai khám phá khi decay entropy về $0.001$. |
| **Nhật ký Phase 2 Single (`p2_single`)** | `feature/111s-honest-anchor` (`819d3e1`) | `logs/p2_single/training_metrics.json` | $3.32\text{M}$ bước trên map 1:1, chẩn đoán cơ chế dao động pedal 5 Hz (F17 & F20). |
| **Nhật ký Phase 2 Racing Line (`p2_line`)** | `feature/111s-honest-anchor` (`6df6ba9`) | `logs/p2_line/training_metrics.json` | $2.66\text{M}$ bước với racing line prior, minh họa sự suy thoái tốc độ dài hạn (F21). |
| **Nhật ký Chẩn đoán Kỹ thuật (F1–F24)**| `feature/111s-honest-anchor` | `knowledge/rl_debugging_findings.md` | Báo cáo chi tiết 24 phát hiện kỹ thuật, cơ chế lỗi và bài học thuật toán. |
| **Tổng kết Vật lý Bullet & Hiệu chuẩn**| `feature/111s-honest-anchor` | `knowledge/vehicle_and_engine_physics.md` | Báo cáo đo đạc trực tiếp hệ thống raycast vehicle, ma sát lốp và phân tích sai lệch mô hình. |
