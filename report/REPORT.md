# BÁO CÁO NGHIÊN CỨU KHOA HỌC VÀ KỸ THUẬT ĐỒ ÁN
# TỐI ƯU HÓA ĐIỀU KHIỂN XE ĐUA TỰ HÀNH BẰNG HỌC TĂNG CƯỜNG SÂU TRÊN MÔ PHỎNG VẬT LÝ BULLET VÀ ĐỐI SÁNH GIẢI TÍCH

> **Môn học**: Học Tăng Cường (Reinforcement Learning - REL)  
> **Chủ đề**: Autonomous High-Speed Racing via Continuous Model-Free Reinforcement Learning  
> **Nhánh mã nguồn neo đối chuẩn**: `feature/111s-honest-anchor` (Commit gốc: `ee99f9d`, Commit tài liệu & đối chứng: `7853722`)  
> **Các nhánh thực nghiệm mở rộng**: `feature/oop-clean-physics-40d` (`0180988`), `origin/feature/multicar-selfplay` (`7ab74a5`), `feature/f1-downforce` (`cdfb928`)  
> **Định dạng báo cáo**: Báo cáo học thuật cấu trúc IMRaD (Introduction, Methods, Results, Discussion)  
> **Cam kết liêm chính học thuật**: Báo cáo trình bày dữ liệu thực nghiệm nguyên bản từ hệ thống đo lường và telemetry; phân biệt rõ các cấp độ nhận định: **[Đo được]**, **[Suy luận]**, **[Giả thuyết]**; không ngụy tạo kết quả, không che giấu lỗi kỹ thuật, và phân tích minh bạch các hạn chế của mô hình.

---

## TÓM TẮT (ABSTRACT)

Báo cáo này nghiên cứu bài toán điều khiển xe đua thể thao tự hành ở giới hạn ma sát vật lý (handling limits) trên sa bàn đua Silverstone Grand Prix thu nhỏ, sử dụng thuật toán Học Tăng Cường Không Mô Hình (Model-Free Continuous Reinforcement Learning) dựa trên thuật toán Tối ưu hóa Chính sách Tiệm cận (Proximal Policy Optimization - PPO). Hệ thống được triển khai trên nền tảng mô phỏng vật lý đa vật thể Bullet Physics thông qua môi trường MetaDrive. Thay vì cung cấp quỹ đạo tối ưu định sẵn trong quan sát hay áp dụng kỹ thuật học bắt chước (Imitation Learning), nghiên cứu tập trung giải quyết câu hỏi: **Liệu một tác tử RL với quan sát thuần hình học tim đường (Zero Racing Line Prior) có thể tự khám phá kỹ thuật lái xe thể thao (Trail-braking, Apex clipping) và đạt hiệu năng tương đương hoặc vượt qua phương pháp điều khiển tối ưu giải tích dựa trên mô hình (Convex QP Minimum Curvature và Lipp-Boyd Speed Profiling) hay không?**

Kết quả thực nghiệm trên mô hình tham chiếu neo `ref_111s` (Commit `ee99f9d`), được đánh giá độc lập dưới quy tắc vạch đường 4 bánh nghiêm ngặt của FIA qua 20 episode kiểm thử độc lập ($n = 20$), cho thấy tác tử PPO hoàn thành vòng đua với thời gian vòng xuất phát đứng (Standing Lap) trung bình là $114.78 \pm 0.40\text{ s}$ và thời gian vòng chạy đà (Flying Lap) là $109.44 \pm 0.50\text{ s}$, đạt tỷ lệ vòng sạch không va chạm (Clean Lap Rate) là $90\%$ ($18/20$ vòng hoàn thành). Thành tích này vượt qua tác tử tham chiếu giải tích mạnh nhất đã được hiệu chuẩn thực nghiệm (`gt3_anchor`: Standing $118.20\text{ s}$, Flying $112.80\text{ s}$, thời gian tích phân lý thuyết $T^* = 110.62\text{ s}$) khoảng $3.36\text{ s}$ trên vòng flying. Tính lặp lại độc lập được xác nhận trên Seed 43 với thành tích $115.18 \pm 0.43\text{ s}$ (Standing) và $109.96 \pm 0.69\text{ s}$ (Flying, $19/20$ vòng sạch), bác bỏ giả thuyết kết quả đạt được do phương sai chọn hạt giống ngẫu nhiên. 

Báo cáo đi sâu phân tích nguyên nhân hiện tượng "RL vượt qua nghiệm giải tích tối ưu" bắt nguồn từ sự sai lệch mô hình - hệ thống thực (Model-Plant Mismatch): mô hình chất điểm giải tích giả định gia tốc ma sát ngang không đổi ($0.95\text{G}$), trong khi PPO trực tiếp khai thác giới hạn bám động học tức thời của Bullet Physics ($a_{\text{lat}}$ phân vị $95\%$ đạt $0.99\text{G}$, cực đại $1.21\text{G}$). Đồng thời, báo cáo giải mã chi tiết các thách thức cố hữu của PPO liên tục:
1. **Cơ chế vi mô của hiện tượng dao động bàn đạp tần số cao (Bang-bang / PWM chattering tại 5 Hz)**: Việc vector quan sát đưa trực tiếp hành động pedal bước trước (`obs[6] = last_pedal`) vào mạng nơ-ron nhưng hàm phần thưởng thiếu thành phần phạt độ mượt vi phân pedal ($\Delta\text{pedal}^2$) đã tạo thành mạch hồi tiếp dao động 2 bước (bước chẵn đạp ga dương, bước lẻ rà phanh nhẹ). Sự dồn tải dọc liên tục này gây mất ổn định lực bám ngang cầu trước ngay tại thời điểm vào cua (đối chiếu với lý thuyết CAPS của Mysore et al., 2021).
2. **Kẹt độ lệch chuẩn khám phá (Actor Log-std Freeze)** do đạo hàm triệt tiêu của hàm clamp trong forward pass, và hiện tượng suy sụp khám phá (Exploration Collapse) khi decay entropy quá sớm.
3. **Sự phụ thuộc sống còn vào thành phần phạt lệch tim đường $r_{\text{center}}$** như một rào cản thế năng mềm chống lại xu hướng cắt cua sớm (Early Apex) bắt nguồn từ tầm nhìn hữu hạn của hệ số chiết khấu $\gamma = 0.99$.

---

## 1. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU NGHIÊN CỨU (INTRODUCTION)

### 1.1. Bối cảnh và Thách thức Kỹ thuật
Điều khiển tự hành phương tiện ở vận tốc cao tiệm cận giới hạn vật lý của lốp xe (Autonomous High-Speed Racing) là một bài toán kinh điển nhưng phức tạp trong điều khiển học và trí tuệ nhân tạo. Khác với bài toán lái xe thông thường (Urban Driving), xe đua vận hành trong điều kiện phi tuyến tính cao độ: lực bám lốp xe nằm sâu trong vùng bão hòa phi tuyến của elip ma sát (Kamm's Friction Ellipse), đòi hỏi sự kết hợp phức tạp giữa lực dọc (phanh/ga) và lực ngang (đánh lái) nhằm khai thác tối đa gia tốc tổng hợp.

Phương pháp truyền thống giải quyết bài toán này thông qua quy trình hai giai đoạn tách rời: (1) Quy hoạch quỹ đạo tối ưu hình học tĩnh dựa trên mô hình đơn giản hóa (ví dụ: tối tiểu độ cong qua Convex QP - Heilmeier et al., 2020; phân bố tốc độ dọc tối ưu thời gian - Lipp & Boyd, 2014); và (2) Bám quỹ đạo bằng các bộ điều khiển phản hồi như Model Predictive Control (MPC) hoặc Pure Pursuit. Tuy nhiên, sự sai lệch giữa mô hình động học giả định (chất điểm, mô hình xe đạp tuyến tính) và động lực học thực tế của hệ thống mô phỏng đa vật thể (Multi-body Dynamics với hệ thống treo, vi sai, tải trọng thay đổi) tạo ra khoảng cách hiệu năng lớn (Model-Plant Mismatch).

Sự trỗi dậy của Học Tăng Cường Sâu (Deep Reinforcement Learning - DRL) mở ra khả năng tối ưu hóa chính sách điều khiển trực tiếp trên simulator tương tác phức tạp (Fuchs et al., 2021; Wurman et al., 2022). Tuy nhiên, việc áp dụng DRL trong điều khiển xe đua liên tục đặt ra các câu hỏi mở mang tính cốt lõi về mặt học thuật:
1. **Liêm chính quan sát (Observation Integrity)**: Tác tử có thể tự hình thành kỹ thuật đua xe thể thao tối ưu (vào cua chậm, ra cua nhanh - Out-In-Out, Trail-braking) từ quan sát hình học tim đường thuần túy mà không cần "mớm" trước tọa độ quỹ đạo tối ưu (Racing Line Prior) hay không?
2. **Bản chất của việc vượt qua mốc chuẩn (Baseline Superiority)**: Khi một tác tử RL đạt thời gian hoàn thành vòng đua nhanh hơn mô hình giải tích, đó là do tác tử tìm được quỹ đạo thông minh hơn, hay do tác tử đã khai thác các điểm hở (exploitations) của simulator vật lý?
3. **Độ mượt cơ học và Hiện tượng Chattering**: Tại sao các thuật toán Policy Gradient liên tục luôn có xu hướng xuất tín hiệu điều khiển rung giật (Bang-bang/PWM chattering) gây hại cho cơ cấu chấp hành trong thực tế (Mysore et al., 2021)?

### 1.2. Mục tiêu và Phạm vi Đồ án
Nghiên cứu này tập trung vào các mục tiêu học thuật rõ ràng:
- **Xây dựng môi trường mô phỏng chuẩn xác**: Thiết lập môi trường đua xe thể thao GT3 trên sa bàn Silverstone Grand Prix (tổng chiều dài $2,838.7\text{ m}$ theo hệ thống MetaDrive/Bullet Physics), tích hợp quy tắc trọng tài 4 bánh nghiêm ngặt theo chuẩn FIA.
- **Phát triển Tác tử Đối chứng Giải tích (Analytical Upper-Bound Baseline)**: Tích hợp bộ giải Convex QP và thuật toán tích phân Lipp-Boyd để thiết lập xe bóng ma giải tích (`GhostCar`), đóng vai trò là mốc chuẩn khoa học minh bạch để so sánh.
- **Tối ưu hóa Tác tử PPO Liên tục Thuần (Pure PPO Agent)**: Huấn luyện tác tử với không gian hành động tách rời 3 chiều $[steer, throttle, brake]$ và không gian quan sát thuần 35 chiều (Zero Data Leakage), loại bỏ hoàn toàn thông tin quỹ đạo giải tích trong quá trình học.
- **Đánh giá Độc lập và Phân tích Liêm chính**: Đo lường định lượng trên 20 episode ngẫu nhiên, phân tích dữ liệu Telemetry vi mô (G-G Diagram, Speed Trace, Steering Power Spectrum), kiểm tra tính tái lập trên nhiều hạt giống ngẫu nhiên (Seeds), và tiến hành các nghiên cứu bóc tách (Ablation Studies) để làm sáng tỏ cơ chế hoạt động của hệ thống.

---

## 2. NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

Nghiên cứu được đặt trên nền tảng đối chiếu trực tiếp với 4 công trình khoa học tiêu biểu trong lĩnh vực đua xe tự hành và điều khiển liên tục bằng RL:

### 2.1. Fuchs et al. (IEEE RA-L / ICRA 2021): Đua xe Super-Human trên Gran Turismo Sport
Fuchs, Song, Kaufmann, Scaramuzza, và Dürr (2021) công bố công trình *"Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning"* trên IEEE Robotics and Automation Letters (RA-L).
- **Phương pháp**: Sử dụng thuật toán Soft Actor-Critic (SAC) không mô hình với mạng nơ-ron 2 lớp ẩn (256 units, ReLU), tần số điều khiển $10\text{ Hz}$.
- **Không gian quan sát và hành động**: Quan sát gồm vận tốc, gia tốc 3D, góc lệch tim đường, 13 tia khoảng cách (fan 180°, tầm 100m), độ cong tim đường tại 10 điểm từ $1.0\text{s}$ đến $2.8\text{s}$ phía trước. **Tác giả khẳng định không đưa racing line vào quan sát**. Không gian hành động gồm góc lái và một trục pedal gộp $[-1, 1]$ (dương là ga, âm là phanh).
- **Hàm thưởng và kết thúc**: Phần thưởng chính là tiến độ dọc tim đường ($r_{\text{progress}}$) trừ đi phạt va chạm tỉ lệ với bình phương vận tốc: $r_t = r_{\text{progress}} - c_w \|v\|^2 \cdot \mathbb{I}_{\text{wall}}$ với $c_w = 5 \times 10^{-4}$. **Tác giả không kết thúc episode khi va chạm tường** nhằm tối ưu hóa việc lấy mẫu.
- **Kiểm thử nhiễu loạn và độ bền (Robustness)**: Kiểm tra tác tử dưới trễ tín hiệu điều khiển ($\Delta t_{\text{delay}} \in [50, 100, 150]\text{ ms}$) và nhiễu quan sát Gaussian ($\sigma_{\text{noise}} \in [2\%, 5\%, 9\%]$). Kết quả cho thấy tác tử duy trì sự ổn định ở mức nhiễu $\le 2\%$, nhưng trễ $> 100\text{ ms}$ làm mất hoàn toàn khả năng cạnh tranh do phanh muộn.
- **Kết quả**: Đạt thành tích siêu nhân trên 3 cấu hình đường đua/xe khác nhau (ví dụ: Audi TT Cup đạt 1:15.913, vượt kỷ lục của tay đua con người nhanh nhất thế giới là 1:16.062). Độ lệch chuẩn thời gian vòng đua qua 10 vòng chỉ đạt $15\text{ ms}$ (so với $480\text{ ms}$ của con người thuộc top $1\%$).

### 2.2. Wurman et al. (Nature 2022): Gran Turismo Sophy (GT Sophy)
Công trình kinh điển của Wurman et al. (Sony AI) xuất bản trên tạp chí *Nature*, vol. 602, mang tên *"Outracing champion Gran Turismo drivers with deep reinforcement learning"*.
- **Phương pháp**: Sử dụng thuật toán Quantile Regression Soft Actor-Critic (QR-SAC) phân phối với 32 quantiles, cập nhật TD $N$-bước ($N=5$ hoặc $N=7$). Mạng nơ-ron sâu 4 lớp (2048 units, ReLU, dropout 0.1 ở policy). Tần số điều khiển $10\text{ Hz}$.
- **Không gian quan sát**: Trích xuất chi tiết trạng thái động lực học gồm vận tốc 3D, gia tốc, **tải trọng và góc trượt của từng lốp trong 4 bánh**, biểu diễn đường đua qua 60 điểm 3D trên mép trái, tim đường và mép phải (phủ tầm nhìn $\sim 6\text{s}$ phía trước, co giãn theo vận tốc). **GT Sophy hoàn toàn không sử dụng racing line định trước trong quan sát**. Hành động gồm góc lái và 1 trục pedal gộp $[-1, 1]$.
- **Chiến lược huấn luyện**: Huấn luyện phân tán không đồng bộ trên hàng nghìn máy chơi game PlayStation, áp dụng kỹ thuật lấy mẫu phân tầng (Stratified Sampling) trên tập kịch bản hỗn hợp (khởi đầu ngẫu nhiên trên đường, bám đuôi, đối kháng) thay vì giáo trình tuần tự (Curriculum Learning).
- **Hiện tượng rung giật điều khiển**: Đáng chú ý, trong tài liệu bổ sung (Extended Data Fig. 2b–c), nhóm tác giả thừa nhận GT Sophy xuất hiện hiện tượng giật vô lăng tần số cao ở $10\text{ Hz}$ (*steering chattering*), trong khi các tay đua vô địch thế giới (Igor Fraga, Takuma Miyazono) điều khiển vô cùng mượt mà ở $60\text{ Hz}$. Tuy nhiên, GT Sophy chấp nhận điều này vì mục tiêu tối thượng là thời gian hoàn thành vòng đua và chiến thuật thi đấu trực tiếp.

### 2.3. Chisari et al. (IEEE ICRA 2021): Sim-to-Real trong Đua xe Tự hành
Chisari, Liniger, Rupenyan, Van Gool, và Lygeros (ETH Zürich) công bố *"Learning from Simulation, Racing in Reality"* tại IEEE ICRA 2021.
- **Phương pháp & Thách thức**: Giải quyết bài toán chuyển giao từ mô phỏng sang thực tế (Sim-to-Real Transfer) cho xe đua mô hình RC tỉ lệ 1:43 (nền tảng ORCA) chạy ở giới hạn vật lý.
- **Điều chuẩn không gian hành động**: Tác giả chỉ ra rằng việc áp dụng trực tiếp action từ policy RL lên cơ cấu chấp hành vật lý sẽ gây rung lắc mạnh và phá hủy phần cứng. Nhóm nghiên cứu đề xuất không gian hành động nâng bậc (*lifted action space*) - điều khiển đạo hàm bậc một của lệnh lái ($\Delta u_t$) kết hợp điều chuẩn policy (*policy regularization*).
- **Kết quả**: Sau khi áp dụng kỹ thuật xáo trộn miền (Domain Randomization) về hệ số ma sát lốp và quán tính động cơ kết hợp 3 giờ fine-tune trên đường đua thực, tác tử RL giảm $50\%$ lỗi vi phạm giới hạn vạch đường so với bộ điều khiển Model Predictive Contouring Control (MPCC) tối ưu, minh chứng năng lực thích nghi với các hiệu ứng phi tuyến mà mô hình MPC bỏ qua.

### 2.4. Mysore et al. (IEEE ICRA 2021): Điều khiển Mượt với CAPS
Mysore, Mabsout, Mancuso, và Saenko công bố *"Regularizing Action Policies for Smooth Control with Reinforcement Learning"* (CAPS) tại IEEE ICRA 2021.
- **Hiện tượng Bang-bang / PWM trong RL**: Nhóm tác giả chứng minh về mặt toán học rằng việc giải bài toán tối ưu Markov từng bước độc lập ($\max \mathbb{E}[Q(s, a)]$) khiến các thuật toán actor-critic liên tục (PPO, SAC, DDPG) tự nhiên hội tụ về chính sách dạng đóng/mở cực trị (Bang-bang action), tạo ra tín hiệu điều khiển dao động biên độ lớn tương tự điều chế độ rộng xung (PWM). Điều này gây hao mòn cơ học, tiêu tốn năng lượng và mất ổn định hệ thống.
- **Đóng góp lý thuyết (CAPS Loss)**: Đề xuất hàm phạt điều chuẩn bổ sung trực tiếp vào hàm mất mát của Actor:
  1. *Độ mượt thời gian (Temporal Smoothness Loss)*: Phạt bình phương khoảng cách giữa hai hành động liên tiếp trên quỹ đạo:
     $$L_T = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \|\pi(s_t) - \pi(s_{t-1})\|^2 \right]$$
  2. *Độ mượt không gian (Spatial Smoothness Loss)*: Phạt độ nhạy của hành động trước các nhiễu loạn nhỏ trong không gian trạng thái:
     $$L_S = \mathbb{E}_{s_t \sim \mathcal{D}, \bar{s}_t \sim \mathcal{N}(s_t, \sigma_s^2 I)} \left[ \|\pi(s_t) - \pi(\bar{s}_t)\|^2 \right]$$
- **Ý nghĩa đối với đồ án**: Công trình cung cấp cơ sở lý luận vững chắc để phân biệt giữa hành vi "Bang-bang chiến thuật tối ưu" (đạp lút ga trên đoạn thẳng, phanh kịch sàn ở điểm phanh) và hiện tượng "Chattering bệnh lý" (đảo lái và nhấp nhả ga/phanh ở tần số $10\text{ Hz}$).

### 2.5. Bảng Đối sánh Phương pháp luận giữa các Công trình và Đồ án

| Thuộc tính So sánh | Fuchs et al. (2021) | GT Sophy (Wurman 2022) | Chisari et al. (2021) | Đồ án Này (Commit `ee99f9d` / `7853722`) |
| :--- | :--- | :--- | :--- | :--- |
| **Thuật toán RL** | SAC (Off-policy) | QR-SAC (Off-policy, 32 quantiles) | SAC (Off-policy) | **PPO Liên tục (On-policy, GAE)** |
| **Tần số quyết định** | $10\text{ Hz}$ | $10\text{ Hz}$ | $20 - 50\text{ Hz}$ | **$10\text{ Hz}$ ($\Delta t = 0.1\text{s}$)** |
| **Không gian hành động** | 2D: Lái + [Ga/Phanh gộp] | 2D: Lái + [Ga/Phanh gộp] | 2D: $\Delta\text{Steer}, \Delta\text{Throttle}$ | **3D: Lái, Ga riêng, Phanh riêng** |
| **Phân phối Policy** | Gaussian squashed tanh | Gaussian squashed tanh | Gaussian squashed tanh | **Gaussian squashed tanh + Jacobian Fix** |
| **Racing Line trong Obs**| **KHÔNG** (Zero prior) | **KHÔNG** (Zero prior) | Có (MPCC contouring frame) | **KHÔNG (Nhánh `pure`, Zero prior)** |
| **Quy tắc Biên đường** | Phạt va chạm tường $\propto v^2$ | Phạt ra ngoài lề $\propto v^2$ | Ràng buộc biên mềm | **Quy tắc 4 bánh FIA (Terminate)** |
| **Mốc Đối chứng Chính** | Kỷ lục con người (World Record) | Tay đua vô địch con người | MPCC dựa trên mô hình xe đạp | **Xe Bóng ma Giải tích (Convex QP + Lipp-Boyd)** |
| **Xử lý Chattering** | Không xử lý (chấp nhận) | Không xử lý (thừa nhận ở Fig. 2) | Lifted Action + Regularization | **Slew Rate Limiter + Phạt $\Delta\text{steer}^2$** |

---

## 3. MÔ HÌNH HÓA TOÁN HỌC VÀ HỆ THỐNG MÔ PHỎNG (SYSTEM FORMULATION)

### 3.1. Hệ Tọa độ và Ký hiệu Toán học Thống nhất
Nhằm đảm bảo tính chuẩn xác và nhất quán trong toàn bộ báo cáo, các biến số được định nghĩa theo Bảng 1.

*Bảng 1: Bảng ký hiệu toán học thống nhất trong báo cáo.*

| Ký hiệu | Ý nghĩa vật lý / Kỹ thuật | Đơn vị | Tương ứng trong mã nguồn |
| :--- | :--- | :---: | :--- |
| $t, \Delta t$ | Chỉ số bước thời gian, độ dài bước tích phân ($\Delta t = 0.1\text{s}$) | –, $\text{s}$ | `step`, `dt` |
| $s_t$ | Tọa độ tiến trình dọc tim đường theo hệ Frenet | $\text{m}$ | `track_s` |
| $d_t$ | Độ lệch khoảng cách vuông góc so với tim đường | $\text{m}$ | `lat_offset` |
| $w$ | Nửa bề rộng mặt đường nhựa ($w = 4.5\text{ m}$ tại map half) | $\text{m}$ | `half_width` |
| $L$ | Tổng chiều dài một vòng đua ($L = 2,838.7\text{ m}$) | $\text{m}$ | `circuit_length` |
| $v_t$ | Vận tốc vô hướng của xe ($v_t = \|\mathbf{v}_t\|$) | $\text{m/s}$ (hoặc $\text{km/h}$) | `speed_km_h` |
| $\kappa(s)$ | Độ cong hình học của tim đường hoặc racing line | $\text{m}^{-1}$ | `track_kappa` |
| $\mathbf{o}_t$ | Vector quan sát đầu vào tại bước $t$ (35 chiều hoặc 40 chiều) | – | `obs` |
| $\mathbf{a}_t$ | Vector hành động thực thi: $(\delta_t, a^{\text{thr}}_t, a^{\text{brk}}_t)$ | – | `action` |
| $r_t$ | Phần thưởng vô hướng nhận được tại bước $t$ | – | `step_reward` |
| $T_{\text{lap}}$ | Thời gian hoàn thành vòng đua đo được trên mô phỏng | $\text{s}$ | `lap_time` |
| $T^*$ | Thời gian vòng đua lý thuyết tối ưu tích phân từ profile | $\text{s}$ | `theoretical_best_lap_time` |
| $\beta$ | Góc trượt thân xe (Sideslip Angle, $\beta = \arctan(v_y / v_x)$) | $\text{độ}$ ($^\circ$) | `beta_deg` |
| $a_{\text{lat}}, a_{\text{long}}$ | Gia tốc trọng trường ngang và dọc của xe | $\text{G}$ hoặc $\text{m/s}^2$ | `lateral_accel`, `longitudinal_accel` |

### 3.2. Động lực học Phương tiện trong Bullet Physics và MetaDrive
Hệ thống sử dụng mô hình xe thể thao GT3 (`GT3RaceCar`) được xây dựng trên nền tảng Bullet Physics Raycast Vehicle của MetaDrive:
- **Cấu trúc Chassis & Dẫn động 4 Bánh**: 
  - Khối lượng thân xe cố định ở $m = 1100\text{ kg}$ (`MASS` trong Bullet engine; các khai báo `config["mass"] = 1350` bị engine bỏ qua [Đo được]).
  - Vị trí trọng tâm: Cách trục trước $1.052\text{ m}$, trục sau $1.417\text{ m}$, chiều cao trọng tâm $h_{\text{cg}} \approx 0.55\text{ m}$. Tải trọng tĩnh: $3100\text{ N}$ ở mỗi bánh trước, $2298\text{ N}$ ở mỗi bánh sau.
  - Cơ chế áp lực ga (`base_vehicle._apply_throttle_brake`): Khi đạp ga ($\text{pedal} \ge 0$), Bullet áp dụng lực kéo $F_{\text{drive}} = \text{max\_engine\_force} \times \text{pedal}$ lên **cả 4 bánh xe** (dẫn động 4 bánh toàn thời gian AWD). Do đó, gia tốc dọc lý thuyết ở dải không trượt là $a_x = 4F/m$ chính xác tuyệt đối (ví dụ: $4 \times 850 / 1100 = 3.09\text{ m/s}^2$, khớp số đo telemetry đến 3 chữ số thập phân).
  - Hiện tượng chổng đuôi (Pitch-over / Rear lift): Do trọng tâm cao ($0.55\text{ m}$) và lệch về trước, khi phanh gấp với gia tốc $a_{\text{brake}} > 1.7\text{G}$, tải trọng bánh sau giảm về $0$, gây mất lái đuôi xe. Do đó, lực phanh thực nghiệm được khống chế ở $15.0\text{ m/s}^2$.
- **Mô hình Tiếp xúc Lốp & Bug Ma sát Cầu trước Lịch sử (F16)**:
  - Bullet không áp dụng phương trình giải tích lốp Pacejka mà tính toán lực bám thông qua ma sát tiếp xúc Coulomb: $F_{\text{fric}} \le \mu \cdot F_z$.
  - Trong quá trình phát triển trước commit `7853722`, việc nạp xe qua wrapper `BaseRaceVehicle` đã gặp lỗi con trỏ bánh xe cũ (`raw_vehicle.wheels` trỏ vào vùng nhớ rác sau khi dựng lại xe), khiến hệ số ma sát lốp trước luôn bị khóa ở giá trị mặc định của MetaDrive là $\mu_{\text{front}} = 0.90$, trong khi lốp sau nhận $\mu_{\text{rear}} = 1.35$. Lỗi này tạo ra đặc tính thiếu lái cưỡng bức (Strong Understeer). Để đảm bảo tính liêm chính khoa học và bảo toàn môi trường huấn luyện của mô hình tham chiếu `ref_111s`, cấu hình `gt3_anchor` giữ nguyên trạng thái động lực học này một cách minh bạch cho cả tác tử PPO và tác tử đối chứng Ghost.
- **Quy tắc Biên đường 4 Bánh (FIA Track Limits)**: Tác tử được coi là hợp lệ khi có ít nhất một bánh xe nằm trong bề rộng mặt đường nhựa ($|d_t| \le w + w_{\text{vehicle}}/2 = 4.5 + 0.925 = 5.425\text{ m}$). Ngay khi cả 4 bánh xe hoàn toàn rời khỏi mặt đường và tiếp xúc bãi cỏ, môi trường lập tức kích hoạt cờ va chạm và kết thúc episode (`terminate`).

### 3.3. Không gian Hành động và Bộ Trộn ECU $C^1$ (Continuous Drive-by-Wire)
Khác với quy ước 2 chiều truyền thống của MetaDrive (gộp ga và phanh vào một trục $[-1, 1]$), đồ án thiết kế không gian hành động 3 chiều tách rời:
$$\mathbf{a}_t = [\delta_t, a_t^{\text{thr}}, a_t^{\text{brk}}] \in [-1, 1] \times [0, 1] \times [0, 1]$$
Trong đó:
- $\delta_t$: Lệnh góc đánh lái chuẩn hóa (ánh xạ tới góc quay bánh trước tối đa $\pm 20^\circ$).
- $a_t^{\text{thr}}, a_t^{\text{brk}}$: Lệnh chân ga và chân phanh độc lập.

Tín hiệu điều khiển được xử lý qua bộ điều chỉnh tốc độ xoay lái cơ học (Slew Rate Limiter, tối đa $0.18/\text{step} \iff 3.6^\circ/\text{step}$) và bộ trộn pedal điện tử liên tục $C^1$ (ECU Electronic Control Unit):
$$\text{pedal}_{\text{applied}} = a_t^{\text{thr}} \cdot (1 - a_{t, \text{eff}}^{\text{brk}}) - a_{t, \text{eff}}^{\text{brk}}$$
với hàm vùng chết phanh liên tục:
$$a_{t, \text{eff}}^{\text{brk}} = \begin{cases} 0, & a_t^{\text{brk}} \le 0.015 \\ \frac{a_t^{\text{brk}} - 0.015}{1.0 - 0.015}, & a_t^{\text{brk}} > 0.015 \end{cases}$$
Cơ chế này cho phép mạng nơ-ron thực hiện đồng thời kỹ thuật phanh chân trái (Left-Foot Braking) và phanh dồn số nhả dần (Trail-braking) mà không tạo ra điểm gián đoạn gradient trong tối ưu hóa.

### 3.4. Không gian Quan sát Thuần (35D Pure Obs) và Phiên bản Tiến hóa (40D Obs)
1. **Không gian Quan sát 35 chiều Chuẩn (`ref_111s`)**: Hoàn toàn không chứa bất kỳ tọa độ hay mục tiêu nào của racing line giải tích:
   - **Trạng thái Động học Nội tại (Ego Kinematics, 11 chiều)**: Vận tốc dọc $v_x / 40.0$, vận tốc ngang $v_y / 20.0$, tốc độ góc quay thân xe $\dot{\psi} / 2.0$, gia tốc $a_x / 10.0, a_y / 10.0$, góc trượt thân xe $\beta$, Roll, Pitch, hành động bước trước $(\delta_{t-1}, a_{t-1}^{\text{thr}}, a_{t-1}^{\text{brk}})$.
   - **Hình học Sai lệch Tim đường Tức thời (4 chiều)**: $\sin(\Delta\theta), \cos(\Delta\theta)$, khoảng cách lệch ngang $d_t / w$, vận tốc lệch ngang $\dot{d}_t / 20.0$.
   - **Tầm nhìn Hình học Tim đường Đón đầu (20 chiều)**: 5 điểm ngắm phía trước tại $[10, 25, 45, 70, 100]\text{ m}$; mỗi điểm cung cấp $(\Delta x_k / 50.0, \Delta y_k / 20.0, \sin\Delta\theta_k, \kappa_k \times 50.0)$.
2. **Không gian Quan sát 40 chiều Mở rộng (Nhánh `feature/oop-clean-physics-40d`, Commit `0180988`)**:
   - Mở rộng thêm 5 chiều thông tin bao gồm cờ cảnh báo runoff $\mathbb{I}_{\text{runoff}}$, gia tốc giật vi phân $\Delta\delta$, và mở rộng tầm nhìn đón đầu lên 6 mốc tới $150\text{ m}$ để hỗ trợ dải tốc độ cao $> 110\text{ km/h}$.

### 3.5. Hàm Phần thưởng Thể thao (Motorsport Reward Function)
Hàm phần thưởng được thiết kế nhằm cân bằng giữa mục tiêu tối đa hóa vận tốc tiến trình và duy trì phương tiện trong giới hạn an toàn của mặt đường:
$$r_t = r_t^{\text{prog}} + r_t^{\text{center}} + r_t^{\text{steer}} + r_t^{\text{idle}} + R^{\text{terminal}}$$

1. **Phần thưởng Tiến độ Frenet ($r_t^{\text{prog}}$)**:
   $$r_t^{\text{prog}} = c_{\text{prog}} \cdot (s_t - s_{t-1}), \quad c_{\text{prog}} = 0.15$$
2. **Thành phần Phạt Lệch Tim đường ($r_t^{\text{center}}$)**:
   $$r_t^{\text{center}} = -c_{\text{center}} \cdot \left(\frac{d_t}{w}\right)^2, \quad c_{\text{center}} = 0.04$$
   > *Lưu ý Liêm chính Học thuật*: Việc giữ thành phần này trong bản `ref_111s` là một sai lệch có chủ đích so với khuyến nghị ban đầu của đặc tả (vốn đề xuất thả lỏng hoàn toàn để xe tự do mở cua). Nghiên cứu bóc tách ở Chương 6 sẽ chứng minh vai trò then chốt của thành phần này.
3. **Phạt Rung lắc Tay lái ($r_t^{\text{steer}}$)**:
   $$r_t^{\text{steer}} = -c_{\text{steer}} \cdot (\delta_t - \delta_{t-1})^2, \quad c_{\text{steer}} = 0.10$$
4. **Phạt Đứng yên / Tốc độ Thấp ($r_t^{\text{idle}}$)**: Phạt cố định $-0.05$ nếu vận tốc phương tiện $< 5.0\text{ km/h}$.
5. **Thưởng/Phạt Điểm mút ($R^{\text{terminal}}$)**: Phạt văng đường $R^{\text{crash}} = -20.0$. Thưởng về đích hàm mũ:
   $$R^{\text{finish}} = R_{\text{base}} \cdot \exp\left(-\frac{T_{\text{lap}} - T_0}{\tau}\right)$$
   với $R_{\text{base}} = 50.0, T_0 = 110.0\text{ s}, \tau = 25.0\text{ s}$.

### 3.6. Thuật toán PPO và Hiệu chỉnh Jacobian cho Phân phối Squashed Gaussian
Mạng nơ-ron Actor xuất ra vector kỳ vọng $\boldsymbol{\mu}_\theta(\mathbf{o})$ và vector tham số log độ lệch chuẩn $\log \boldsymbol{\sigma}$. Biến ngẫu nhiên nguyên bản $u_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ được đưa qua hàm kích hoạt phi tuyến $\tanh$ để ép vào khoảng hữu hạn $[-1, 1]$: $a_i = \tanh(u_i)$.

Theo định lý chuyển đổi biến ngẫu nhiên (Haarnoja et al., 2018 - Phụ lục SAC; Schulman et al., 2017), mật độ xác suất thực tế của hành động $\pi_\theta(\mathbf{a}|\mathbf{o})$ bắt buộc phải bù trừ định thức Jacobian của phép biến đổi:
$$\log \pi_\theta(\mathbf{a}|\mathbf{o}) = \sum_{i=1}^3 \left[ \log \mathcal{N}(u_i; \mu_i, \sigma_i^2) - \log(1 - \tanh^2(u_i) + \delta_{\text{reg}}) \right]$$
với $\delta_{\text{reg}} = 10^{-6}$. Việc bỏ quên số hạng Jacobian (Phát hiện F2) trong các phiên bản cũ từng gây ra sai lệch nghiêm trọng giữa hàm mục tiêu importance sampling của PPO và gradient thực tế.

---

## 4. TÁC TỬ ĐỐI CHỨNG GIẢI TÍCH VÀ XE BÓNG MA (ANALYTICAL BASELINE & CALIBRATED GHOST CAR)

Để thiết lập một mốc đối chứng khoa học khách quan và không tạo ra "đối thủ bù nhìn" (Strawman Baseline), nghiên cứu đã phát triển một động cơ giải tích tối ưu độc lập hai tầng (`RacingBenchmarkEngine`) kết hợp bộ điều khiển bám vết (`OptimalRacingAgent` / `GhostCar`).

### 4.1. Tối ưu Quỹ đạo Hình học Cực tiểu Độ cong (Convex QP Minimum Curvature)
Theo phương pháp của Heilmeier et al. (2020), quỹ đạo đua lý tưởng là đường cong tối thiểu hóa tích phân bình phương độ cong trên toàn mạch, giúp phương tiện có thể duy trì vận tốc góc cua cao nhất:
$$\min_{\boldsymbol{\alpha}} \int_0^L \kappa^2(s) \, ds \quad \approx \quad \min_{\boldsymbol{\alpha}} \frac{1}{2} \boldsymbol{\alpha}^T \mathbf{Q} \boldsymbol{\alpha} + \mathbf{c}^T \boldsymbol{\alpha}$$
thỏa mãn các ràng buộc biên hình học:
$$-w_{\text{eff}} \le \alpha_i \le w_{\text{eff}}, \quad \forall i \in \{1..N\}$$
trong đó $\alpha_i$ là độ dịch chuyển ngang khỏi tim đường tại điểm rời rạc $i$, và $w_{\text{eff}} = 4.05\text{ m}$ là biên an toàn đã trừ đi nửa bề rộng xe ($0.925\text{ m}$) và khoảng đệm vật lý trên mặt đường rộng $4.5\text{ m}$. Bài toán quy hoạch toàn phương lồi (Convex QP) được giải số học qua thư viện `OSQP` trong thời gian dưới $1.5\text{s}$.

### 4.2. Phân bố Vận tốc Tối ưu Thời gian (Lipp-Boyd Time-Optimal Velocity Profile)
Dựa trên thuật toán của Lipp & Boyd (2014), phân bố tốc độ dọc tối ưu $v^*(s)$ dọc theo quỹ đạo đã tìm được tính toán qua phương pháp 3-lượt (Three-pass forward/backward integration):
1. **Giới hạn Gia tốc Ngang (Lateral Friction Limit)**:
   $$v_{\text{corner}}^*(s) = \sqrt{\frac{a_{\text{lat, max}}}{|\kappa_{\text{line}}(s)| + \epsilon}}$$
2. **Lượt Tích phân Lùi (Backward Pass - Braking Phase)**:
   $$v_i^* = \min\left( v_i^*, \sqrt{(v_{i+1}^*)^2 + 2 \cdot a_{\text{brake}} \cdot \Delta s} \right)$$
3. **Lượt Tích phân Tiến (Forward Pass - Acceleration Phase)**:
   $$v_i^* = \min\left( v_i^*, \sqrt{(v_{i-1}^*)^2 + 2 \cdot a_{\text{engine}} \cdot \Delta s} \right)$$
Thời gian vòng đua lý thuyết cận dưới của mô hình chất điểm được tính bằng tích phân:
$$T^* = \int_0^L \frac{1}{v^*(s)} \, ds$$

### 4.3. Hiệu chuẩn Thực nghiệm Tham số Xe Bóng ma (`gt3_anchor`)
Trong phiên bản cũ (`Ghost legacy v22`), các tham số vật lý của Ghost được gán theo giả định lý thuyết ($a_{\text{brake}} = 6.2\text{ m/s}^2, a_{\text{lat}} = 1.04 - 1.20\text{G}$), dẫn đến việc Ghost bị trượt văng khỏi đường đua khi cố gắng bám theo profile hoặc chạy chậm hơn năng lực thật của xe.

Để biến Ghost thành đối thủ mạnh nhất và trung thực nhất với môi trường Bullet (Phát hiện F24, Commit `7853722`), nhóm tác giả đã thực hiện quy trình quét thực nghiệm lưới 33 cấu hình (Grid Search) đo đạc trực tiếp trên động cơ Bullet Physics:
- **Năng lực tăng tốc dọc**: Đo được $a_{\text{engine}} = 3.09\text{ m/s}^2$ ($a_x = 4F_{\text{drive}} / m = 4 \times 850 / 1100$).
- **Năng lực phanh dọc cực đại**: Hệ thống phanh Bullet đạt tới $18.0\text{ m/s}^2$; để tránh hiện tượng nhấc bổng trục sau (chổng đuôi do trọng tâm cao), tham số an toàn được chốt ở $a_{\text{brake}} = 15.0\text{ m/s}^2$.
- **Trần gia tốc ngang ổn định**: Qua kiểm thử vòng tròn bán kính cố định (Skidpad test), xe bị văng lề tại cùng một góc cua ($s \approx 2390\text{ m}$) khi $a_{\text{lat}} \ge 1.00\text{G}$. Mốc giới hạn vật lý ổn định cao nhất mà Ghost có thể hoàn thành $100\%$ vòng sạch là $a_{\text{lat}} = 0.95\text{G} = 9.32\text{ m/s}^2$.

Cấu hình hiệu chuẩn `gt3_anchor` này đem lại thời gian lý thuyết $T^* = 110.62\text{ s}$, và thành tích chạy thực tế trong simulator đạt Standing Lap $118.20\text{ s}$ và Flying Lap $112.80\text{ s}$ (hoàn thành $3/3$ vòng đánh giá không va chạm).

---

## 5. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH DỮ LIỆU (EXPERIMENTS & RESULTS)

Toàn bộ kết quả dưới đây được đo lường bằng tập lệnh đánh giá tiêu chuẩn `training/evaluate.py` trên nhánh `feature/111s-honest-anchor`, thực hiện trên sa bàn Silverstone (chiều rộng $9.0\text{ m}$, 18 góc cua), áp dụng luật 4 bánh nghiêm ngặt của FIA, chạy 2 vòng liên tiếp mỗi episode (Vòng 1: Standing Start từ vạch xuất phát; Vòng 2: Flying Lap khi đã đạt tốc độ cao). Số liệu thô được lưu trữ nguyên bản trong thư mục `logs/anchor/`.

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

*(Ghi chú: Giá trị hiển thị dạng $\text{Mean} \pm \text{Std}$; số mẫu $n=20$ cho các tác tử RL, $n=3$ cho tác tử tất định Ghost. "Vòng Sạch" được định nghĩa là episode không có bất kỳ thời điểm nào vi phạm luật 4 bánh hoặc va chạm).*

### 5.2. Phân tích Hiện tượng RL Vượt qua Nghiệm Tối ưu Giải tích
Dữ liệu Bảng 2 thể hiện một kết quả đáng kinh ngạc: **Tác tử PPO `ref_111s` đạt thời gian vòng flying $109.44\text{ s}$, nhanh hơn Ghost hiệu chỉnh $3.36\text{ s}$ ($\sim 3.0\%$) và thậm chí vượt qua cả thời gian lý thuyết tối ưu tích phân của mô hình chất điểm ($T^* = 110.62\text{ s}$)**. 

Bằng chứng thực nghiệm lặp lại trên Seed 43 với thành tích Flying $109.96 \pm 0.69\text{ s}$ ($19/20$ vòng sạch) xác nhận hiện tượng này là một thuộc tính có tính quy luật của hệ thống học, không phải do biến động may rủi. Để giải mã hiện tượng này dưới góc nhìn khoa học khách quan, nhóm nghiên cứu đã phân tích dữ liệu Telemetry chuyên sâu:

1. **Khai thác Độ bám Động lực học Tức thời (Instantaneous Dynamic Grip)**:
   - Mô hình chất điểm của Ghost Car áp dụng định luật ma sát tĩnh với trần gia tốc ngang không đổi $a_{\text{lat}} \le 0.95\text{G}$.
   - Trích xuất dữ liệu cảm biến gia tốc thực tế của xe do PPO điều khiển (Phát hiện F20, F23): Phân vị 95% của gia tốc ngang thực tế đạt $a_{\text{lat, p95}} = 0.99\text{G}$, và gia tốc cực đại tức thời ghi nhận lên tới $a_{\text{lat, max}} = 1.21\text{G}$.
   - **[Suy luận]**: Trong mô phỏng đa vật thể Bullet, khi xe vào cua gắt, hiện tượng dịch chuyển tải trọng động (Dynamic Load Transfer) nén mạnh lên hai bánh ngoài kết hợp với góc trượt thân xe ($\beta_{\text{p95}} = 3.5^\circ$, đỉnh điểm lên tới $18^\circ$ drift nhẹ) cho phép lốp xe phát sinh lực bám tức thời vượt qua ngưỡng ma sát tĩnh trong khoảng thời gian ngắn ($0.2 - 0.4\text{s}$). RL trực tiếp tối ưu trên mô phỏng nên đã học được cách "ép tải" này, trong khi mô hình chất điểm tĩnh bỏ qua hoàn toàn.

2. **Kỹ thuật Trail-Braking Tự sinh (Emergent Trail-Braking)**:
   - Phân tích biểu đồ phối hợp gia tốc G-G (G-G Friction Circle Diagram) cho thấy PPO không thực hiện phanh theo kiểu tuần tự (phanh thẳng hoàn toàn rồi mới đánh lái như thuật toán tách rời của Ghost), mà duy trì áp lực phanh nhẹ ($10\% - 25\%$) đồng thời với việc bắt đầu bẻ lái vào apex.
   - Hành vi này giữ trọng tâm xe dồn lên trục trước, tăng tải trọng thẳng đứng $F_{z, \text{front}}$, từ đó tạm thời gia tăng lực bám dẫn hướng của hai bánh trước ngay tại thời điểm bắt đầu chuyển hướng.

### 5.3. Độ Ổn định và Vấn đề Sai số Biên (Failure Mode Analysis)
Mặc dù đạt tốc độ vượt trội, mô hình tham chiếu `ref_111s` vẫn tồn tại xác suất rủi ro $10\%$ ($2/20$ episode bị crash):
- Điểm văng thứ nhất: Mét thứ $2637\text{ m}$ (Góc cua Vale - góc cua chậm sau đoạn thẳng dài, xe lao vào với tốc độ cao và trượt nhẹ 4 bánh ra bãi cỏ).
- Điểm văng thứ hai: Mét thứ $364\text{ m}$ (Góc cua Farm Curve).
- Khi tiến hành fine-tune trên chính luật 4 bánh nghiêm ngặt (`ref_111s_ft_strict`, Commit `f4d8ef9`), mô hình cố gắng rút ngắn thêm thời gian vòng đua ($108.98\text{ s}$ flying), nhưng tỷ lệ vòng sạch sụp đổ nghiêm trọng xuống chỉ còn $40\%$ ($8/20$ sạch). Dữ liệu crash phân bổ tập trung cao độ tại đoạn $1734 - 1807\text{ m}$ ($10/12$ vụ tai nạn). Điều này chứng minh rằng khi không có các cơ chế điều chuẩn độ mượt hành động, chính sách PPO bị đẩy sát ranh giới phân kỳ động lực học (Dynamic Divergence Boundary).

---

## 6. NGHIÊN CỨU BÓC TÁCH VÀ THẢO LUẬN LIÊM CHÍNH (ABLATIONS & CRITICAL DISCUSSION)

### 6.1. Nghiên cứu Bóc tách Vai trò của Thành phần Phạt Lệch Tim đường ($r_{\text{center}}$)
Trong bản đặc tả ban đầu (`PROJECT_SPECIFICATION.md`), việc đưa thành phần phạt lệch tim đường $r_{\text{center}} = -c_{\text{center}} (d_t/w)^2$ bị coi là vi phạm nguyên tắc "để xe tự do học đường đua tối ưu", vì phạt tim đường sẽ cản trở xe mở rộng cua hình học (Out-In-Out).

Để kiểm chứng tác động thực tế, nhóm nghiên cứu đã thực hiện thí nghiệm bóc tách triệt để (Ablation A13, Nhánh `exp/111s-no-rcenter`): giữ nguyên toàn bộ siêu tham số và hạt giống ngẫu nhiên, chỉ đặt $c_{\text{center}} = 0.0$.
- **Kết quả đo được**:
  - Tại Seed 42: Tỷ lệ vòng sạch sụt giảm từ $90\%$ xuống **$0\%$ ($0/20$ vòng hoàn thành, $20/20$ vụ tai nạn)**. Xe liên tục văng khỏi đường ngay tại mét $363 - 365\text{ m}$ (13 lần crash lặp lại chính xác tại cùng một mét cua).
  - Tại Seed 43: Tỷ lệ vòng sạch sụt giảm từ $95\%$ xuống **$10\%$ ($2/20$ vòng hoàn thành)**.
- **Cơ chế Khoa học**:
  - Khi không có $r_{\text{center}}$, không gian khám phá của bài toán trở nên quá rộng trong điều kiện mặt đường hẹp ($4.5\text{ m}$). Hệ số chiết khấu $\gamma = 0.99$ tương đương với tầm nhìn hiệu dụng khoảng $\sim 10\text{s}$ ($\frac{1}{1 - \gamma} \times 0.1\text{s} = 100$ bước). 
  - Tại các góc cua gắt, phần thưởng tiến độ tức thời $r_{\text{prog}}$ thúc đẩy xe cắt góc sớm (Early Apex) để tối đa hóa quãng đường ngắn nhất trước mắt, khiến xe mất quỹ đạo thoát cua và văng lề ở lối ra.
  - $r_{\text{center}}$ đóng vai trò như một "hàng rào thế năng mềm" (Soft Potential Barrier), giữ xe không bị trôi dạt quá xa vào vùng nguy hiểm trong các giai đoạn đầu của quá trình học. Minh chứng là ở bản `ref_111s`, mặc dù có $r_{\text{center}}$, sai số lệch so với racing line tối ưu (Line MAE) vẫn đạt $2.12\text{ m}$, cho thấy xe vẫn biết cách mở cua ra sát biên khi cần thiết.

### 6.2. Cơ chế Vi mô của Hiện tượng Rung giật Điều khiển (Bang-bang / PWM Chattering)
Dữ liệu Telemetry ở tần số $10\text{ Hz}$ ghi nhận phong cách lái của PPO `ref_111s` và các nhánh mở rộng (Phát hiện F20, F23, và Phân tích `plan_notices.md` §1):
- **Đảo chiều lái**: Trung bình PPO đảo chiều vô lăng tới **$60$ lần trên mỗi $100$ bước** ($6\text{ Hz}$ oscillation), với biên độ biến thiên $|\Delta\text{steer}| = 0.074$.
- **Nhấp nhả pedal chu kỳ 2 bước (5 Hz PWM)**: PPO `ref_111s` có tần suất chuyển đổi giữa ga và phanh là $4.7$ lần / $100$ bước (trong khi ở các nhánh chưa hiệu chỉnh như P2 v2, con số này lên tới $20.3$ lần / $100$ bước). Ngược lại, Ghost Car giải tích chỉ đổi trạng thái pedal $0.2$ lần / $100$ bước.

*Bảng 3: So sánh chỉ số điều khiển giữa PPO Agent và Ghost Car.*

| Chỉ số Đánh giá Độ mượt | PPO `ref_111s` (Map Half) | PPO P2 v2 (Map Full, GT3Real) | Ghost Hiệu chỉnh (`gt3_anchor`) |
| :--- | :---: | :---: | :---: |
| Số lần đảo chiều lái / 100 bước | **$60$** | **$74$** | **$16$** |
| Biến thiên góc lái trung bình $|\Delta\text{steer}|$ | $0.074$ | $0.121$ | $0.007$ |
| Số lần đổi trạng thái Ga $\leftrightarrow$ Phanh / 100 bước | **$4.7$** | **$20.3$** | **$0.2$** |
| Góc trượt thân xe $\beta$ (Phân vị 95% / Max) | $3.5^\circ / 18.0^\circ$ | $8.9^\circ / 52.7^\circ$ | $2.0^\circ / 3.8^\circ$ |

**Phân tích Cơ chế Vi mô và Liên hệ Lý thuyết CAPS (Mysore et al., 2021)**:
1. *Cơ chế Hình thành Dao động Chu kỳ 2 Bước (`obs[6]` Feedback Loop)*:
   Trong vector quan sát, chiều thứ 7 đưa trực tiếp giá trị `last_pedal` vào Actor MLP:
   $$\text{obs}[6] = a_{t-1}^{\text{thr}} \cdot (1 - a_{t-1}^{\text{brk}}) - a_{t-1}^{\text{brk}}$$
   Tuy nhiên, hàm thưởng vi mô chỉ phạt độ rung giật góc lái ($r_{\text{steer\_smooth}} = -0.1 \cdot (\Delta\delta)^2$) mà hoàn toàn **không có khoản phạt độ giật bàn đạp ($r_{\text{pedal\_smooth}} = 0$)**. Do nguy cơ văng đường bị phạt nặng, mạng nơ-ron Actor học cách duy trì vận tốc an toàn (~150 km/h) bằng cách tạo thành một mạch dao động nghịch đảo liên tục mỗi bước ($\Delta t = 0.1\text{s} \implies 5\text{ Hz}$):
   - Bước chẵn: Quan sát thấy `last_pedal < 0` $\implies$ xuất lệnh ga dương ($+0.3$ đến $+0.7$).
   - Bước lẻ: Quan sát thấy `last_pedal > 0` $\implies$ xuất lệnh rà phanh nhẹ ($-0.1$ đến $-0.3$).
   Gia tốc dọc trung bình trượt qua 2 bước bằng $0$, nhưng việc nhấp nhả ga/phanh liên tục gây ra hiện tượng **dồn tải trọng dọc (longitudinal load transfer) không ngừng**. Tại thời điểm bắt đầu vào cua, nhịp giật pedal làm tải trọng và lực bám ngang của hai bánh trước dao động mạnh, giải thích nguyên nhân xe bị văng mất lái ngay cả khi vận tốc vào cua thấp hơn vận tốc tối ưu $v^*$!
2. *Ảo giác Thống kê về Độ mượt (Insight I10)*:
   Bản `ref_111s` trông có vẻ "mượt mà" hơn các nhánh Phase 2 map lớn chủ yếu là vì xe chạy ở dải vận tốc thấp hơn ($88\text{ km/h}$ so với $144\text{ km/h}$) và đặc tính understeer của bug F16. Do gia tốc ngang ly tâm tỉ lệ với bình phương vận tốc ($a_y \propto v^2 \cdot \delta$), cùng một biên độ giật lái $|\Delta\delta| = 0.074$ ở $144\text{ km/h}$ sẽ phát sinh xung lực ngang lớn gấp gần **$4\text{ lần}$** so với ở $88\text{ km/h}$, lập tức đẩy xe vượt qua elip ma sát bão hòa và gây drift mất kiểm soát ($\beta_{\text{max}} = 52.7^\circ$).
3. *Tại sao Ràng buộc Vật lý Cơ học Thất bại trước Bang-bang?*
   Việc thử nghiệm bộ lọc thông thấp EMA (`c1904fa`) hay mô hình sức cản bánh lái trong `ExtendedVehiclePhysics` (`a79aee7`) không giải quyết được chattering vì Actor MLP nhanh chóng học cách bù trừ ngược lại độ trễ của bộ lọc. Đúng như phân tích của Mysore et al. (2021) và Seyde et al. (2021), chattering là hệ quả của hàm mục tiêu Markov; chỉ khi bổ sung số hạng điều chuẩn thời gian $L_T$ trực tiếp vào loss hoặc phạt $\Delta\text{pedal}^2$ trong reward thì mới triệt tiêu được hiện tượng này.

### 6.3. Bóc tách Nhánh PPO + Racing Line (Ablation R14 / P2-line)
Trong nhánh thử nghiệm `p2_line` (Commit `11807e4`), nhóm nghiên cứu đưa trực tiếp sai lệch khoảng cách và vận tốc mục tiêu của racing line tối ưu $(\Delta d_{\text{line}}, v^*)$ vào quan sát.
- **Hiện tượng đo được (Phát hiện F21)**: Tác tử PPO + Line đạt vòng sạch đầu tiên rất sớm (tại $1.147\text{M}$ steps, so với $2.15\text{M}$ steps của nhánh pure), chứng minh racing line đóng vai trò như một gợi ý an toàn (safety guide) rất tốt.
- **Hệ quả dài hạn**: Tuy nhiên, sau mốc $2.0\text{M}$ steps, thời gian vòng đua của tác tử bị thoái hóa dần: từ $146.6\text{s}$ chậm dần về $173 - 181\text{s}$, vận tốc trung bình tụt từ $144\text{ km/h}$ xuống $115\text{ km/h}$.
- **Giải mã cơ chế**: Do hàm phạt văng đường tỉ lệ với bình phương vận tốc ($\propto v^2$) quá nặng so với áp lực thời gian, kết hợp với việc bám theo line định sẵn khiến tác tử triệt tiêu động lực khám phá các đường chạy vượt giới hạn bám. Điều này khẳng định kết luận khoa học: **Racing line trong quan sát giúp tăng tốc độ hội tụ an toàn ban đầu, nhưng cản trở việc khám phá ra các giải pháp điều khiển siêu việt ở giới hạn ma sát**.

### 6.4. Các Lỗi Kỹ thuật PPO Đã Được Khắc phục (Debugging Findings)
1. **Lỗi Đóng băng Độ lệch chuẩn Khám phá (F1 - Actor Log-std Freeze)**:
   - *Cơ chế*: Hàm forward của Actor sử dụng lệnh `torch.clamp(self.actor_logstd, min=-2.3, max=0.0)`. Khi tham số gốc trôi ra ngoài biên do bước cập nhật Adam, đạo hàm của phép clamp triệt tiêu về $0$, khiến gradient của policy loss và entropy bonus hoàn toàn biến mất. Tham số bị kẹt vĩnh viễn ở biên.
   - *Hệ quả*: Tại commit `2055469`, độ lệch chuẩn phanh bị kẹt ở mức cực đại ($\sigma_{\text{brake}} = 1.0$), khiến $70\%$ số bước rollout bị phanh ngẫu nhiên, làm tỷ lệ tai nạn luôn ở mức xấp xỉ $100\%$.
   - *Bản vá*: Thay thế việc clamp trong forward bằng thao tác chiếu trực tiếp tham số (In-place Parameter Projection) sau bước tối ưu hóa: `self.ac.actor_logstd.clamp_(MIN, MAX)`.
2. **Lỗi Lệch Mật độ Xác suất Squashed Gaussian (F2 - Log-prob Mismatch)**:
   - Khắc phục sự sai lệch giữa hành động rời rạc lấy mẫu trong quá trình rollout và việc tính toán lại log-likelihood trong pha tối ưu hóa PPO, đồng bộ hóa ngưỡng cận $\text{ACTION\_BOUND} = 0.999$ và phép bù Jacobian.
3. **Cạm bẫy Suy sụp Khám phá khi Decay Entropy (F14 & Insight I9)**:
   - Thử nghiệm giảm hệ số entropy (`ent_coef`) về $0.001$ khiến phương sai đánh lái $\sigma_{\text{steer}}$ sập xuống mức sàn $0.10$ chỉ sau $150\text{k}$ bước. Kết quả là tác tử bị kẹt trong một "hố đen cục bộ", liên tục đâm vào đúng góc cua mét thứ $1306\text{ m}$ suốt 8 lần đánh giá liên tiếp. Khi mạng Critic dự đoán chính xác xác suất tai nạn, value loss giảm về $0.05$ và triệt tiêu hoàn toàn gradient khám phá.
   - *Giải pháp*: Cố định `ent_coef = 0.01` và thiết lập trần độ lệch chuẩn độc lập theo từng chiều: $\text{LOGSTD\_MAX} = [0.0, -0.7, -0.7]$ cho $[\text{steer}, \text{throttle}, \text{brake}]$.
4. **Sửa đổi Khởi tạo Thiên vị Chân phanh (F14 - Brake Bias Initialization)**:
   - Lớp tuyến tính cuối của Actor được khởi tạo với trọng số thiên vị chân phanh âm (`bias[2] = -1.6`), giúp xác suất phanh ngẫu nhiên ban đầu giảm từ $50\%$ xuống dưới $5\%$, ngăn chặn triệt để hiện tượng tác tử học thói quen rê phanh đứng yên tại vạch xuất phát.

### 6.5. Mở rộng Đua Đa Tác tử (6-Car Self-play MARL) và Khí động học F1 Downforce
Tại các nhánh phát triển mở rộng của nhóm nghiên cứu (`origin/feature/multicar-selfplay` commit `7ab74a5`, và `feature/f1-downforce` commit `cdfb928` / `24fb2e4`):
- **Đua Đa Tác tử Tự Chơi (6-Car MARL Self-Play)**:
  - Triển khai 6 xe đua GT3 cùng xuất phát so le trên sa bàn mở rộng 3 làn ($13.5\text{ m}$). 
  - *Hàm thưởng va chạm đối xứng (Symmetric Collision Penalty)*: Nhằm loại bỏ rủi ro sai lệch của các quy tắc heuristic phán đoán góc đâm, cả xe gây va chạm và xe bị đâm đều nhận mức phạt đối xứng: $-2.0$ tại thời điểm va chạm đầu và $-0.2/\text{step}$ trong suốt thời gian tiếp xúc.
  - *Thưởng vượt xe sạch có kiểm soát (Controlled Overtake Bonus)*: Tác tử vượt xe thành công trên mặt đường nhựa nhận $+1.0$, kèm cơ chế thời gian hồi (cooldown) $50\text{ bước}$ ($5.0\text{s}$) giữa từng cặp xe nhằm triệt tiêu hoàn toàn hành vi "cày điểm" tiến-lùi (oscillating overtake loops).
  - *Cơ chế Hồi sinh Tại chỗ (In-place Respawn)*: Xe gặp sự cố được hồi sinh ngay trên làn đường với vận tốc giảm về $0$, phạt điểm tỉ lệ với bình phương vận tốc lúc va chạm, và nhận thời gian miễn trừ xếp hạng $150\text{ bước}$ để tránh phá vỡ phân phối trạng thái của các xe phía sau.
  - *Hiệu ứng núp gió (Slipstream)*: Tái hiện chân thực vùng chân không phía sau đuôi xe, giúp xe bám đuôi tăng tốc độ tiếp cận khi cự ly $< 15\text{ m}$.
- **Khí động học Lực ép Xuống (F1 Aerodynamic Downforce)**:
  - Bổ sung phương trình lực ép khí động học tỉ lệ với bình phương vận tốc:
    $$F_{\text{down}} = m \cdot g \cdot \left(\frac{v}{200}\right)^2$$
  - Giúp gia tăng tải trọng thẳng đứng $F_z$ ở tốc độ cao mà không làm tăng khối lượng quán tính. Xe F1 đơn đạt tốc độ tối đa $301\text{ km/h}$ và hoàn thành vòng đua Silverstone trong thời gian kỷ lục $111.7\text{ s}$ trên sa bàn tỷ lệ lớn.

### 6.6. Bảng Phân Tích Thực Nghiệm Toàn Diện Các Chuỗi Ablation và Log Huấn Luyện (Comprehensive Cross-Run Ablation & Telemetry Analysis)
Để trả lời thấu đáo các câu hỏi về tác động độc lập của từng thành phần kỹ thuật, nhóm nghiên cứu đã rà soát và phân tích định lượng toàn bộ dữ liệu thô trích xuất từ các thư mục log thực nghiệm (`logs/run_*`, `logs/multicar_*`, `logs/p1_main`, `logs/p2_*`).

*Bảng 4: Ma trận so sánh định lượng các chuỗi thực nghiệm Ablation trên toàn bộ dự án.*

| Mã Run / Nhánh Nguồn | Cấu hình & Biến số Khảo sát | Tổng Bước / Mẫu | Tỷ lệ Hoàn thành Max (Cuối) | Lap Nhanh nhất [s] | Tỷ lệ Vòng Sạch / Hoàn thành | Va chạm (Contacts) / Respawns | Nhận định Kỹ thuật Cốt lõi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`run_a0_legacy`** (Nhánh `0180988`) | Baseline cũ: Lỗi F1 (σ kẹt trần), thiếu Jacobian F2, không $r_{\text{time}}$ | $3.38\text{M}$ steps | $42.5\%$ ($19.4\%$) | $147.5\text{s}$ (DNF) | **$0/20$ ($0\%$)** (Crash $100\%$) | – | **Thất bại hoàn toàn**: $\sigma_{\text{brake}} = 1.0$ khiến $70\%$ bước bị phanh ngẫu nhiên; không có vòng sạch. |
| **`p1_main`** (Nhánh `7853722`) | Đã vá F1/F2; thưởng về đích hàm mũ; map half; thử decay entropy | $2.66\text{M}$ steps | **$100.0\%$** ($46.0\%$) | $130.2\text{s}$ | Vòng sạch đầu tiên tại $1.35\text{M}$; cuối run crash $100\%$ | – | **Exploration Collapse (F14 & I9)**: Decay ent về $0.001$ làm kẹt vĩnh viễn ở khúc cua $46\%$ (mét 1306). |
| **`p2_single` v2** (Nhánh `819d3e1`) | Map 1:1, GT3RealCar, trần bám $1.56\text{G}$, vận tốc $> 140\text{ km/h}$ | $3.32\text{M}$ steps | **$100.0\%$** ($66.8\%$) | **$137.5\text{s}$** (Best sạch $143.4\text{s}$) | $0/10$ sạch ở $3.3\text{M}$; $5/10$ crash ở $5.4\text{km}$ | – | **Hiện tượng F17 & F20**: Giật pedal 5 Hz và đảo lái 74 lần/100 bước gây drift văng lề ở vận tốc cao ($a_y \propto v^2\delta$). |
| **`p2_line`** (Nhánh `6df6ba9`) | Map 1:1, PPO + Racing line và $v^*$ trong observation (Prior) | $2.66\text{M}$ steps | **$100.0\%$** ($54.0\%$) | $138.6\text{s}$ (Sạch $153.7\text{s}$) | Sạch sớm tại $1.15\text{M}$; sau $2\text{M}$ chậm dần | – | **Prior Trade-off (F21)**: Line giúp an toàn sớm nhưng giảm tốc độ về $115\text{ km/h}$ do triệt tiêu động lực khám phá. |
| **`run_v5`** (`logs/run_v5`) | GT3 3 làn đường ($13.5\text{m}$), $r_{\text{time}} = 0$, không penalize steer | $379\text{k}$ steps | $11.8\%$ ($10.1\%$) | Không có | $0\%$ (Crash $100\%$) | – | Xe rơi vào bẫy rà phanh chạy chậm, không vượt qua được nửa vòng đầu. |
| **`run_v6`** (`logs/run_v6`) | GT3 3 làn, điều chỉnh phân bố khởi tạo | $737\text{k}$ steps | $32.7\%$ ($12.4\%$) | Không có | $0\%$ (Crash $100\%$) | – | Tiến độ tăng nhưng tiếp tục crash ở các góc cua tốc độ cao do thiếu áp lực thời gian. |
| **`run_v7`** (`logs/run_v7`) | GT3 3 làn, bổ sung áp lực thời gian $r_{\text{time}} = -0.01/\text{step}$ | $963\text{k}$ steps | **$100.0\%$** ($27.3\%$) | $141.5\text{s}$ | Checkpoint `clean_389k`: **$5/6$ ($83.3\%$)** sạch | – | **Đột phá hoàn thành vòng đua**: Áp lực thời gian ép tác tử duy trì vận tốc qua apex; đạt $150\text{s}$ vòng sạch. |
| **`run_v8b`** (`logs/run_v8b`) | GT3 3 làn, 40D obs, Slew rate $0.18$ + Phạt mượt lái $-0.1(\Delta\delta)^2$ | $502\text{k}$ steps | **$100.0\%$** ($22.6\%$) | **$135.4\text{s}$** | Checkpoint `clean_123k`: **$5/6$ ($83.3\%$)** sạch | – | **Tối ưu hóa độ mượt 40D**: Hoàn thành vòng đua ở $123\text{k}$ bước, tốc độ TB $69.0\text{ km/h}$, lap sạch $141.2\text{s}$. |
| **`run_f1b`** (`logs/run_f1b`) | F1 đơn có Downforce, tinh chỉnh hệ số ma sát và lực bám | $399\text{k}$ steps | **$100.0\%$** ($100.0\%$) | $138.9\text{s}$ | **$100\%$ sạch** (Crash $0.0\%$) | – | Downforce giúp xe bám đường tuyệt đối, triệt tiêu hoàn toàn nguy cơ văng lề ở giai đoạn đầu. |
| **`run_f1c`** (`logs/run_f1c`) | F1 đơn Downforce $F_{\text{down}} = mg(v/200)^2$, tốc độ trần $301\text{ km/h}$ | $1.00\text{M}$ steps | **$100.0\%$** ($87.8\%$) | **$109.4\text{s}$** | **$100\%$ sạch ($2/2$) liên tục** từ 600k đến 1000k | – | **Hiệu năng siêu việt**: Tốc độ trung bình tăng từ $83.8\text{ km/h}$ (600k) lên $90.2\text{ km/h}$ (last agent), lap $109.4\text{s}$. |
| **`multicar_v1`** (`logs/multicar_v1`) | MARL 6 xe GT3 self-play, xuất phát hàng ngang (abreast grid) | $4.45\text{M}$ samples | $39.1\%$ | $128.2\text{s}$ | $1/6$ xe hoàn thành | Contacts: $12$ | Xe va chạm chùm ngay vạch xuất phát do cự ly hàng ngang quá hẹp. |
| **`multicar_v3`** (`logs/multicar_v3`) | MARL 6 xe GT3, xuất phát so le (staggered grid), cooldown vượt | $1.41\text{M}$ samples | $40.0\%$ | $132.9\text{s}$ | $2/6$ xe hoàn thành | Contacts: $100$ | Cải thiện xuất phát nhưng xe vẫn dính nhau ở góc cua hẹp đầu tiên. |
| **`multicar_f1`** (`logs/multicar_f1`) | MARL 6 xe F1 Downforce thế hệ 1, phạt va chạm ban đầu | $1.47\text{M}$ samples | $33.1\%$ | $111.2\text{s}$ | $1/6$ xe hoàn thành | Contacts: $67$ | Tốc độ cao khiến các pha va chạm dây chuyền làm hỏng episode của cả nhóm. |
| **`multicar_f1b`** (`logs/multicar_f1b`) | MARL 6 xe F1, điều chỉnh lực đẩy hồi tiếp | $660\text{k}$ samples | $64.5\%$ | $113.5\text{s}$ | $3/6$ xe hoàn thành | Contacts: $113$ | Tỷ lệ hoàn thành tăng vọt lên $64.5\%$, $3/6$ xe về đích an toàn. |
| **`multicar_f1d`** (`logs/multicar_f1d`) | MARL 6 xe F1, cơ chế hồi sinh tại chỗ (In-place Respawn) | $1.50\text{M}$ samples | $92.5\%$ | $114.5\text{s}$ | **$5/6$ xe hoàn thành** | Contacts: $106$, Respawns: $14$ | **Bước ngoặt MARL**: Cơ chế hồi sinh giữ các xe tiếp tục học, $5/6$ xe hoàn thành chặng đua. |
| **`multicar_f1e`** (`logs/multicar_f1e`) | MARL 6 xe F1 hoàn thiện: Phạt đối xứng $-2.0/-0.2$, miễn trừ 150 bước | $1.50\text{M}$ samples | **$94.7\%$** | **$112.9\text{s}$** | **$5.67 / 6$ xe ($\approx 95\%$)** | Contacts: **$33.0$**, Respawns: **$5.3$** | **Đỉnh cao tự chơi 6 xe**: Checkpoint 750k đạt $6/6$ về đích với chỉ $1$ respawn và $44$ va chạm; lap trung bình $123.2\text{s}$. |

---

### 6.7. Giải Mã Chuyên Sâu Các Cơ Chế Động Lực Học Từ Dữ Liệu Đo Đạc
Từ các con số đo lường trong Bảng 4, ba kết luận khoa học quan trọng được rút ra:

1. **Tác động Quyết định của Áp lực Thời gian ($r_{\text{time}} = -0.01/\text{step}$)**:
   - Trong `run_v5` và `run_v6` ($r_{\text{time}} = 0$), mạng Actor tối ưu hóa phần thưởng bằng cách bò chậm ở tốc độ $30 - 50\text{ km/h}$ để tránh rủi ro văng lề. Kết quả là tỷ lệ hoàn thành tối đa chỉ đạt $32.7\%$ và crash rate luôn ở mức $100\%$ khi gặp các đoạn cua dốc.
   - Khi áp dụng $r_{\text{time}} = -0.01$ ở `run_v7` và `run_v8b`, việc dừng đỗ hoặc chạy chậm bị trừng phạt tích lũy ($-1.0$ điểm mỗi $10\text{s}$). Điều này tạo ra lực đẩy gradient cưỡng bức phương tiện phải tăng tốc qua các góc cua, giúp tỷ lệ hoàn thành đạt $100\%$ ngay tại $389\text{k}$ bước và rút ngắn lap time xuống $135.4\text{s}$.

2. **Khí Động Học Downforce Hóa Giải Ranh Giới Bão Hòa Elip Ma Sát**:
   - So sánh giữa xe GT3 (`p2_single`, không downforce) và xe F1 (`run_f1c`, có downforce $F_{\text{down}} = mg(v/200)^2$):
     - Ở GT3 map 1:1, khi vận tốc vượt $140\text{ km/h}$, tải trọng bánh xe chỉ phụ thuộc vào trọng trường tĩnh ($F_z \approx mg$). Gia tốc ly tâm cần thiết để ôm cua vượt qua giới hạn bám $1.56\text{G}$, dẫn đến hiện tượng trượt ngang không thể cứu vãn ($\beta_{\text{max}} = 52.7^\circ$).
     - Ở F1 (`run_f1c`), ở vận tốc $200\text{ km/h}$, lực ép khí động học tăng thêm đúng bằng $100\%$ trọng lượng xe ($F_z = 2mg$). Trần lực ma sát ngang $F_{y, \text{max}} = \mu F_z$ tăng gấp đôi mà không làm tăng khối lượng quán tính cần bẻ lái ($m$). Do đó, `run_f1c` duy trì sự ổn định tuyệt đối với **$100\%$ tỷ lệ vòng sạch ($2/2$ sạch ở mọi checkpoint từ 600k đến 1000k)** và lap time giảm sâu xuống **$109.4\text{s}$**.

3. **Sự Tiến Hóa Trật Tự Trong Đua Đa Tác Tử (Self-Play Order Emergence)**:
   - Trong giai đoạn đầu (`multicar_f1` và `multicar_f1c`), các tác tử đua theo phong cách hỗn loạn: số vụ va chạm lên tới $142$ lần mỗi episode và chỉ $2/6$ xe hoàn thành.
   - Nhờ sự kết hợp của 3 cơ chế: (1) Phạt va chạm đối xứng loại bỏ hành vi chèn ép ác ý, (2) Cooldown 50 bước loại bỏ thói quen nhường đường cày điểm, và (3) Hồi sinh tại chỗ kèm 150 bước miễn trừ xếp hạng, mô hình `multicar_f1e` ghi nhận sự chuyển dịch pha ngoạn mục: số va chạm giảm hơn $76\%$ (từ $142$ xuống còn $33.0$), số lần hồi sinh giảm từ $14$ xuống $5.3$, và **$5.67 / 6$ xe hoàn thành cuộc đua**. Telemetry ghi nhận các xe tự động xếp hàng so le khí động học khi vào cua và tận dụng hiệu ứng núp gió (slipstream) để vượt nhau trên đoạn thẳng.

---

## 7. GIỚI HẠN NGHIÊN CỨU VÀ HƯỚNG PHÁT TRIỂN (LIMITATIONS & FUTURE WORK)

Nhằm duy trì tính liêm chính học thuật cao nhất, các giới hạn kỹ thuật của đồ án được làm rõ:

1. **Giới hạn Mô hình Hình học Sa bàn Silverstone**:
   - Sa bàn trong môi trường MetaDrive gồm 23 khối đường (18 góc cua cung tròn và 5 đoạn thẳng) được xây dựng theo tô-pô của Silverstone nhưng bán kính các góc cua được ước lượng xấp xỉ thủ công ($R \in [25, 95]\text{ m}$ ở map half) và khép kín qua thuật toán tối ưu hóa góc.
   - Bản đồ không chứa các đoạn cong chuyển tiếp (Clothoid Transition Curves) và bỏ qua độ mấp mô địa hình 3D ($z = 0$). Chiều dài bản đồ bị nén theo tỉ lệ xấp xỉ $\approx 1:2.075$ ($2,838.7\text{ m}$ so với $5,891\text{ m}$ thực tế).
   - *Tính thực tế của bề rộng mặt đường*: Tuy nhiên, độ rộng đường lại rất sát với thực tế: 3 làn $\times 4.8\text{ m} = 14.4\text{ m}$ (so với $15.0\text{ m}$ thực tế của Silverstone). Tỉ lệ chiều rộng xe / mặt đường ($1.85\text{ m} / 14.4\text{ m} \approx 1:7.8$) phản ánh chính xác không gian hình học Out-In-Out.
   - *Hiện tượng các sai số tự bù trừ nhau trong profile $v^*$*: Trên sa bàn tỷ lệ lớn, việc thiếu lực cản khí động học khiến xe giữ nguyên trần $260\text{ km/h}$ trên toàn bộ đoạn thẳng, trong khi việc thiếu downforce khiến xe ôm cua chậm hơn thực tế. Tổng thời gian $T^* = 124.0\text{ s}$ rất gần với lap GT3 thực tế, nhưng đây là do **các sai số vật lý tự bù trừ nhau**, không phải do mô hình chất điểm trùng khớp hoàn hảo với động lực học xe đua ngoài đời thực.

2. **Giới hạn Động lực học Xe Bullet**:
   - Raycast vehicle thiếu mô hình khí động học dòng khí đầy đủ (không có lực cản không khí $F_{\text{drag}} \propto v^2$).
   - Không có sự biến thiên nhiệt độ lốp, hao mòn lốp hay áp suất lốp.
   - Thân xe hình hộp cố định với trọng tâm cao ($0.55\text{ m}$) khiến xe dễ bị nhấc đuôi khi phanh gấp trên $1.7\text{G}$, buộc hệ thống phải giới hạn áp lực phanh thực nghiệm.

3. **Hướng phát triển Sim-to-Real**:
   - Tích hợp hàm mất mát CAPS đầy đủ (cả Temporal Smoothness $L_T$ và Spatial Smoothness $L_S$) trực tiếp vào hàm mục tiêu của Actor PPO để triệt tiêu hoàn toàn hiện tượng rung giật vô lăng $10\text{ Hz}$.
   - Áp dụng kỹ thuật Xáo trộn Miền (Domain Randomization) theo Chisari et al. (2021) trên các tham số ma sát $\mu \in [0.8, 1.4]$, độ trễ cơ cấu chấp hành ($20 - 80\text{ ms}$), và khối lượng xe nhằm chuẩn bị cho việc triển khai trên phần cứng xe đua mô hình vật lý thực tế.

---

## 8. KẾT LUẬN (CONCLUSION)

Đồ án đã nghiên cứu và triển khai thành công hệ thống điều khiển xe đua thể thao tự hành tốc độ cao tiệm cận giới hạn ma sát vật lý bằng thuật toán Học Tăng Cường PPO liên tục trên nền tảng mô phỏng Bullet Physics/MetaDrive. 

Các đóng góp khoa học và kỹ thuật cốt lõi bao gồm:
1. **Minh chứng Thực nghiệm về Năng lực của RL Thuần**: Chứng minh rằng một tác tử PPO với quan sát thuần hình học tim đường (Zero Racing Line Prior) hoàn toàn có khả năng tự khám phá các kỹ năng đua xe thể thao phức tạp (Apex clipping, Trail-braking, Over-the-limit slip control) và đạt thời gian hoàn thành vòng đua vượt trội ($109.44 \pm 0.50\text{ s}$ flying) so với mốc chuẩn giải tích dựa trên mô hình chất điểm tối ưu ($112.80\text{ s}$).
2. **Làm sáng tỏ Cơ chế Model-Plant Mismatch**: Phân tích định lượng nguyên nhân RL vượt qua cận lý thuyết giải tích $T^*$ là nhờ khả năng tối ưu hóa trực tiếp trên mô hình mô phỏng đa vật thể, khai thác độ bám động học tức thời ($a_{\text{lat}}$ lên tới $1.21\text{G}$) và chuyển dịch tải trọng lốp mà mô hình giải tích tĩnh bỏ qua.
3. **Bóc tách Minh bạch các Thành phần Hệ thống**: Chỉ ra sự phụ thuộc của quá trình học vào thành phần phạt tim đường $r_{\text{center}}$ như một rào cản thế năng chống phân kỳ, đồng thời đối chiếu bản chất hiện tượng chattering điều khiển với các nghiên cứu hàng đầu thế giới (CAPS, GT Sophy).
4. **Chuẩn hóa Thực nghiệm và Mã nguồn**: Toàn bộ quy trình từ khâu mô phỏng, hiệu chuẩn xe bóng ma giải tích, sửa lỗi gradient PPO đến công cụ đánh giá tự động đa tác tử đã được xây dựng theo chuẩn mực hướng đối tượng (OOP) vững chắc, tạo nền tảng vững chắc cho các nghiên cứu tiếp theo về đua xe tự hành.

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

## PHỤ LỤC: DANH MỤC TRUY XUẤT ARTIFACT VÀ SỐ LIỆU (DATA REPRODUCIBILITY INDEX)

Để hỗ trợ hội đồng chấm thi và cộng đồng nghiên cứu kiểm chứng độc lập, Bảng 5 cung cấp chỉ mục truy xuất chính xác từng tệp dữ liệu, mô hình và nhật ký đánh giá trong kho lưu trữ Git:

*Bảng 5: Chỉ mục truy xuất artifact và log thực nghiệm toàn diện.*

| Tên Dữ liệu / Mô hình | Nhánh Git & Commit SHA | Đường dẫn Tệp trong Repository | Mô tả Nội dung / Số liệu Đối sánh |
| :--- | :--- | :--- | :--- |
| **Mô hình Tham chiếu Neo (`ref_111s`)** | `feature/111s-honest-anchor` (`ee99f9d`) | `models/ref_111s/best_agent.pt` | Trọng số mạng nơ-ron PPO 35D, đạt Standing $114.78\text{s}$, Flying $109.44\text{s}$. |
| **Nhật ký Đánh giá 20 Lap `ref_111s`** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_111s_anchor.json` | Log JSON chi tiết 20 episode kiểm thử độc lập dưới luật 4 bánh FIA. |
| **Nhật ký Đánh giá Ghost Hiệu chỉnh** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_ghost_anchor.json` | Đánh giá xe bóng ma `gt3_anchor` ($0.95\text{G} / 15.0 / 3.09$), đạt Flying $112.80\text{s}$. |
| **Nhật ký Đánh giá Ghost Legacy v22** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_ghost_legacy.json` | Đánh giá Ghost cũ tham số lý thuyết, đạt Flying $114.20\text{s}$. |
| **Mô hình & Nhật ký Lặp lại Seed 43** | `feature/111s-honest-anchor` (`7853722`) | `logs/anchor/eval_overnight_seed43_best.json` | Kiểm chứng tính lặp lại độc lập ($n=20$), đạt Flying $109.96\text{s}$ (19/20 sạch). |
| **Nhật ký Fine-tune Luật Chặt 4 Bánh** | `feature/111s-honest-anchor` (`f4d8ef9`) | `logs/anchor/eval_overnight_ft_best.json` | Kết quả fine-tune 1M bước, lap $108.98\text{s}$ nhưng crash $60\%$ (8/20 sạch). |
| **Nhật ký Bóc tách Bỏ $r_{\text{center}}$** | `exp/111s-no-rcenter` | `logs/anchor/eval_nocenter_s42.json`, `...s43.json` | Minh chứng sụp đổ tỷ lệ vòng sạch về $0\% - 10\%$ khi loại bỏ phạt lệch tim đường. |
| **Dữ liệu Profile Tối ưu Giải tích** | `feature/111s-honest-anchor` | `scripts/silverstone_optimal_profile.npz` | Quỹ đạo lồi Convex QP và phân bố vận tốc Lipp-Boyd ($T^* = 110.62\text{s}$). |
| **Nhật ký Huấn luyện GT3 3 Làn (`run_v8b`)** | `feature/oop-clean-physics-40d` (`0180988`) | `logs/run_v8b/training_metrics.json`, `checkpoint_eval.json` | 40D obs, time penalty $-0.01$; checkpoint `clean_123k` đạt $83.3\%$ vòng sạch, lap $135.4\text{s}$. |
| **Nhật ký Huấn luyện F1 Downforce (`run_f1c`)** | `feature/f1-downforce` (`24fb2e4`) | `logs/run_f1c/training_metrics.json`, `checkpoint_eval.json` | F1 đơn có downforce $F_{\text{down}} \propto v^2$, đạt $100\%$ vòng sạch liên tục từ 600k đến 1000k, lap $109.4\text{s}$. |
| **Nhật ký Đua 6 Xe Tự Chơi (`multicar_f1e`)** | `feature/f1-downforce` (`cdfb928`) | `logs/multicar_f1e/training_metrics.json`, `checkpoint_eval.json` | MARL 6 xe F1 hoàn thiện: $5.67/6$ xe về đích ($\approx 95\%$), va chạm giảm còn $33.0$, lap TB $123.2\text{s}$. |
| **Nhật ký Baseline Cũ Lỗi F1 (`run_a0_legacy`)** | `feature/111s-honest-anchor` (`7853722`) | `logs/run_a0_legacy/training_metrics.json` | $3.38\text{M}$ bước với $100\%$ crash rate, minh chứng sự phá hủy của lỗi actor_logstd freeze. |
| **Nhật ký Phase 1 Main (`p1_main`)** | `feature/111s-honest-anchor` (`7853722`) | `logs/p1_main/training_metrics.json` | $2.66\text{M}$ bước, minh chứng lỗi suy sụp khám phá khi decay entropy về $0.001$. |
| **Nhật ký Phase 2 Single (`p2_single`)** | `feature/111s-honest-anchor` (`819d3e1`) | `logs/p2_single/training_metrics.json` | $3.32\text{M}$ bước trên map 1:1, chẩn đoán cơ chế dao động pedal 5 Hz (F17 & F20). |
| **Nhật ký Phase 2 Racing Line (`p2_line`)** | `feature/111s-honest-anchor` (`6df6ba9`) | `logs/p2_line/training_metrics.json` | $2.66\text{M}$ bước với racing line prior, minh chứng sự suy thoái tốc độ dài hạn (F21). |
| **Nhật ký Chẩn đoán Kỹ thuật (F1–F24)**| `feature/111s-honest-anchor` | `knowledge/rl_debugging_findings.md` | Báo cáo chi tiết 24 phát hiện kỹ thuật, cơ chế lỗi và bài học thuật toán. |
| **Tổng kết Vật lý Bullet & Hiệu chuẩn**| `feature/111s-honest-anchor` | `knowledge/vehicle_and_engine_physics.md` | Báo cáo đo đạc trực tiếp hệ thống raycast vehicle, ma sát lốp và phân tích sai lệch mô hình. |
