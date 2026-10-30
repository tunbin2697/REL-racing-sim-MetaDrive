# 🏎️ METADRIVE 3D AUTONOMOUS RACING & HUMAN VS AI BENCHMARK
> **Đồ Án Môn Học Tăng Cường (Reinforcement Learning - REL)**  
> **Môi Trường Mô Phỏng:** MetaDrive Simulator (Panda3D Physics Engine - Bullet Physics)  
> **Thuật Toán Cốt Lõi:** PPO Continuous Action Space (Squashed Gaussian Policy, Stabilized Jacobian, Unclipped Critic MSE)

> [!IMPORTANT]
> **TÀI LIỆU ĐẶC TẢ YÊU CẦU ĐỀ TÀI CHÍNH THỨC:**  
> Xem chi tiết toàn bộ quy định học thuật, luật đua 4 bánh, cấm data leak và tiêu chuẩn benchmark tại:  
> 📄 **[`PROJECT_SPECIFICATION.md`](file:///c:/Users/tunbi/Project/school/REL%20Assignment/plan/requirements/PROJECT_SPECIFICATION.md)**

---

## 📖 1. Giới Thiệu & Triết Lý Đồ Án

Dự án nghiên cứu và phát triển hệ thống điều khiển xe đua tự hành trong không gian 3D vật lý thời gian thực bằng thuật toán **Học Tăng Cường (Deep Reinforcement Learning - PPO)** trên nền tảng **MetaDrive Simulator**.

Mục tiêu cốt lõi: Huấn luyện một Agent có khả năng **tự học cách đua xe thể thao tối ưu hóa thời gian hoàn thành 1 vòng (Lap Time Optimization)** ở giới hạn vật lý cực hạn của phương tiện, bảo toàn tính liêm chính học thuật môn học REL (nói không với mớm đường đua hay data leakage).

### 🎯 Nguyên Tắc Vận Hành Cốt Lõi Của Đề Tài:
1. **Setting môi trường mặc định (Default Engine Settings)**:
   * Giữ nguyên 100% các thông số vật lý mặc định của MetaDrive: gia tốc động cơ, độ trượt bánh (tire drift/skid), hệ số ma sát lốp $\mu$, tốc độ tối đa, khối lượng và trọng tâm xe. Tránh can thiệp vật lý làm sai lệch bài toán.
2. **Luật giới hạn đường đua chuẩn FIA 4 bánh (Track Limits & Lane Freedom)**:
   * Agent **được phép chạy ở bất kỳ đâu trên mặt đường**, kể cả làn ngược chiều để tối ưu hóa góc vào cua.
   * Vòng chạy **CHỈ terminate khi cả 4 bánh xe hoàn toàn vượt ra ngoài vạch 2 bên mép đường**.
3. **Đường đua không rào chắn sát mép (No Barriers)**:
   * Không đặt rào chắn hay tường chắn sát vạch biên. Nếu có rào chắn hoặc chướng ngại vật địa hình thì phải lùi xa hơn mép đường một khoảng $> 0.5$ thân xe (chạm vào sẽ terminate). Ưu tiên không dùng rào chắn để tránh lỗi va chạm ảo từ địa hình.
4. **Thước đo định tính chuẩn F1 (Qualitative F1 Evaluation)**:
   * Được dùng để người chấm điểm và người phát triển quan sát, đánh giá sự thông minh của Agent sau huấn luyện: Agent tự hình thành khả năng đạp lút sàn ($a_{\text{acc}} \to 1.0$) ở đoạn thẳng, rà phanh nhả dần khi bẻ lái vào cua (**Trail Braking**), tự khám phá quỹ đạo (**Out - In - Out**), và cắt sát đỉnh góc cua (**Apex Clipping**) mà không bị can thiệp hay mớm bất kỳ logic cứng nào.
5. **Tuyệt đối tuân thủ phạm vi môn học REL (No Data Leak)**:
   * **KHÔNG mớm racing line**, không mớm waypoint quỹ đạo mẫu, không mớm hành động mẫu (không dùng Imitation/Supervised Learning ép buộc trong RL).
   * **Loại bỏ bẫy hàm thưởng cục bộ**: Tuyệt đối không dùng hàm phạt lệch tim đường hay thưởng tốc độ mù quáng mọi thời điểm (những hàm này bẫy Agent vào cực tiểu cục bộ và không dám mở cua).
   * **Tự học thuần túy**: Thưởng tiến độ tịnh tiến dọc trục Frenet ($\Delta s$) kết hợp áp lực thời gian (step penalty) để Agent tự khám phá ra racing line tối ưu thông qua tương tác thử - sai.
6. **Mốc đánh giá đối chiếu độc lập (Reference Benchmark)**:
   * Xây dựng một mốc thời gian cơ sở độc lập (tính tay lý thuyết, tính toán bán tự động hoặc nguồn đối chiếu hợp lý / make sense).
   * **Dữ liệu đối chiếu được giữ kín, không leak vào quá trình train**. Outcome kỳ vọng là Agent tự học đạt thời gian lap time **tiệm cận, hợp lý (make sense) và có nguồn đối chiếu rõ ràng**.
7. **Định hướng mở rộng (Bonus Extensions - Khuyến khích nếu agent chính đạt kết quả tốt và còn thời gian trước ngày báo cáo)**:
   * Huấn luyện đua đối kháng đa tác tử ($\ge 2$ Agents).
   * Chế độ người chơi bàn phím đua trực tiếp với AI thời gian thực (`play_vs_ai.py`).
   * Tổng quát hóa trên các đường đua F1 danh tiếng như Monza, Silverstone, Spa-Francorchamps.

---

## 📂 2. Cấu Trúc Thư Mục Dự Án

```text
REL Assignment/
├── plan/requirements/PROJECT_SPECIFICATION.md # 📄 ĐẶC TẢ YÊU CẦU ĐỒ ÁN MÔN HỌC (Quy tắc, Luật đua, Benchmark)
├── README.md                  # Hướng dẫn chi tiết, cài đặt và thực thi hệ thống
│
├── agents/
│   ├── __init__.py
│   ├── networks.py            # Kiến trúc Squashed Gaussian Actor-Critic & Jacobian bù log-prob
│   ├── ppo_agent.py           # Thuật toán PPO Continuous (GAE-Lambda, Unclipped Critic MSE, Minibatch SGD)
│   └── optimal_agent.py       # Tác tử Đối chứng Cận trên Tối ưu (Pure Pursuit + Feedforward Anticipatory Braking)
│
├── game_env/
│   ├── __init__.py
│   ├── racing_env.py          # Wrapper MetaRacingEnv (Obs 40D, frenet s, track limits 4 bánh, Pure Motorsport)
│   ├── vehicle.py             # Thực thể Động lực học GT3RaceCar & CivilianSedan (OOP Vehicle Architecture)
│   ├── benchmark.py           # Engine Phân tích Cận trên (Convex QP Min Curvature + Lipp & Boyd 3-Pass)
│   ├── ghost_car.py           # Thực thể Xe Bóng Ma Đối Chứng 3D (Ghost Car Entity)
│   ├── multiagent_env.py      # Môi trường đua đối đầu 2 xe (Người vs AI / Ghost)
│   └── telemetry.py           # Hệ thống ghi nhận Telemetry (v, acc_lat, acc_long, Delta Time)
│
├── training/
│   ├── train_racing.py        # Pipeline huấn luyện PPO chính + Telemetry + In-Training Benchmark
│   ├── evaluate.py            # Benchmark đánh giá học thuật độc lập đa kịch bản
│   └── evaluator.py           # Module RacingEvaluator OOP đánh giá đa episode và trích xuất chỉ số benchmark
│
├── demo/
│   ├── play_vs_ai.py          # SCRIPT ĐINH: Người bấm phím W-A-S-D trực tiếp so tài với AI hoặc Optimal Ghost Car
│   └── spectate_ai.py         # Chế độ Cinematic theo dõi AI / Optimal Agent tự lái & Ghost Car thời gian thực
│
├── scripts/
│   ├── plot_learning_curves.py      # Xuất 4 biểu đồ học thuật hội tụ (Return, Completion %, Losses, Speed/Crash)
│   ├── plot_telemetry_diagnostic.py # Dashboard Telemetry 6 Panel chuyên sâu + Theoretical Benchmark Overlay
│   ├── plot_silverstone_preview.py  # Trực quan hóa sơ đồ 18 góc cua Silverstone F1
│   └── solve_silverstone_layout.py  # Tối ưu hóa khép góc hình học Closed-Loop Topology
│
├── models/
│   └── best_agent.pt          # Trọng số PPO đạt thành tích kỷ lục (100% clean lap, lap time ~111s)
│
├── plots/
│   └── silverstone_circuit_preview.png # Sơ đồ bố cục đường đua Silverstone F1
│
├── logs/
│   ├── training_metrics.json    # Dữ liệu telemetry chi tiết qua các lần đánh giá định kỳ
│   └── evaluation_results.json  # Dữ liệu đánh giá benchmark học thuật chuẩn mực
│
├── knowledge/                 # 8 tài liệu lý thuyết chuẩn môn Học Tăng Cường & Sim Racing
├── requirements.txt           # Danh sách các thư viện phụ thuộc
└── .gitignore                 # Cấu hình bỏ qua các file tạm, cache, model nặng
```

---

## ⚙️ 3. Hướng Dẫn Cài Đặt (Installation)

### Bước 1: Khởi tạo môi trường Conda (Python 3.10)
```bash
conda create -n rel_racing python=3.10 -y
conda activate rel_racing
```

### Bước 2: Cài đặt PyTorch
* **Máy có GPU NVIDIA (Khuyến nghị CUDA 12.1 / 12.4):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
* **Máy chỉ dùng CPU:**
  ```bash
  pip install torch torchvision
  ```

### Bước 3: Cài đặt MetaDrive & Thư viện phụ trợ
```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra cài đặt đồ họa 3D
```bash
python scripts/test_metadrive_install.py
```
*(Nếu cửa sổ 3D mở ra và bạn lái được xe bằng `W-A-S-D`, hệ thống đã sẵn sàng 100%)*.

---

## 🎮 4. Hướng Dẫn Chạy Các Chế Độ Thực Thi

### 🏆 Chế Độ 1: Đua Trực Tiếp Người vs AI / Ghost Car (Human vs AI Race - Demo Chính)
Mở cửa sổ đồ họa 3D Panda3D 60FPS để trực tiếp so tài tốc độ với AI Agent hoặc Xe Đối Chứng Lý Thuyết (Optimal Ghost Car):
```bash
# Đua với AI PPO đã huấn luyện:
python demo/play_vs_ai.py --model models/best_agent.pt --opponent ppo

# Đua với Tác tử Đối chứng Tối ưu Lý thuyết (Ghost Benchmark Car):
python demo/play_vs_ai.py --opponent optimal
```
* **Xe Người chơi (Player)**: 🟢 **Xanh Lá Cây**. Điều khiển bằng phím:
  * `W`: Đạp ga (Throttle)
  * `S`: Thắng / Lùi (Brake / Reverse)
  * `A` / `D`: Đánh lái Trái / Phải (Steer)
* **Xe Đối thủ (AI / Optimal Ghost Car)**: 🔴 **Đỏ Thể Thao**.
* **Phím chức năng**:
  * `C`: Đổi góc nhìn camera giữa Xe Bạn và Xe Đối thủ.
  * `B`: Góc nhìn toàn cảnh từ trên cao (Bird-eye view).
  * `R`: Reset lại vòng đua mới.
  * `Esc`: Thoát trận đấu.

---

### 🎬 Chế Độ 2: Theo Dõi AI & Ghost Car Tự Lái Góc Quay Điện Ảnh (Cinematic Spectator)
Cho phép quan sát AI PPO hoặc Tác tử Tối ưu Lý thuyết (`OptimalRacingAgent`) biểu diễn trên sa bàn Silverstone F1, hỗ trợ bật **Xe Bóng Ma (Ghost Car)** thời gian thực:
```bash
# 1. Theo dõi AI PPO tự lái:
python demo/spectate_ai.py --model models/best_agent.pt --agent ppo

# 2. Theo dõi AI PPO so tài trực tiếp với Xe Bóng Ma Đối Chứng (Ghost Car):
python demo/spectate_ai.py --model models/best_agent.pt --ghost

# 3. Xem Tác tử Tối ưu Lý thuyết (OptimalRacingAgent) chạy mẫu vòng đua hoàn hảo (100% Clean Lap, 118.2s):
python demo/spectate_ai.py --agent optimal

# 4. Tự lái xe bằng bàn phím (W-A-S-D) trên đường đua Silverstone so tài cùng Xe Bóng Ma:
python demo/spectate_ai.py --manual --ghost
```
* **Điều khiển camera & quan sát 360° tự do**:
  * **Lia chuột (Mouse Look)**: Xoay tự do 360° quanh xe (không bị giới hạn góc, không bị tự động reset về giữa).
  * **Chuột giữa (Middle Click)**: Reset nhanh góc nhìn về thẳng phía trước (0 deg).
  * `G`: **Đổi xe theo dõi (Spectate Target)**: Chuyển camera bám theo Xe bóng ma (Ghost Car) $\leftrightarrow$ Xe chính (Main Car).
  * `L`: Bật / Tắt hiển thị quỹ đạo đua tối ưu 3D (Racing Line & Apex Markers).
  * `C`: **Chụp ảnh màn hình (Screenshot)**: Lưu ảnh `.png` trực tiếp tại thư mục gốc dự án (`main_index_*.png`).
  * `B`: Góc nhìn toàn cảnh từ trên cao (Bird-eye View).
  * `R`: Reset xe về vạch xuất phát.
  * `H`: Bật / Tắt bảng menu trợ giúp phím tắt trên màn hình Panda3D.
  * `Esc`: Thoát chế độ xem.

---

### 🚀 Chế Độ 3: Huấn Luyện Học Tăng Cường (PPO Continuous Training)
Huấn luyện Agent tự khám phá và đột phá các góc cua từ đầu bằng thuật toán PPO thuần:
```bash
python training/train_racing.py --total-steps 4000000 --eval-freq 5 --eval-episodes 3 --milestone-freq 500000
```
* **Cờ cấu hình chính**:
  * `--total-steps 4000000`: Tổng số bước tương tác môi trường (mặc định 1.000.000).
  * `--lr-decay` / `--no-lr-decay`: Tự động suy giảm Learning Rate tuyến tính ($3\times 10^{-4} \to 5\times 10^{-5}$) (Mặc định bật).
  * `--ent-decay` / `--no-ent-decay`: Tự động suy giảm hệ số Entropy (Mặc định tắt, giữ cố định $0.005$).
  * `--milestone-freq 500000`: Lưu checkpoint snapshot tiến trình mỗi 500k bước (`models/agent_step_*k.pt`).
* **Cơ chế lưu mô hình thông minh**:
  * `models/best_agent.pt`: Tự động lưu khi Agent đạt **100% Zero-Crash (3/3 episode đánh giá không va chạm)** và lập kỷ lục thời gian hoàn thành vòng đua (`Best Lap Time` hoặc `Flying Lap Time`).
  * `models/last_agent.pt`: Tự động lưu liên tục sau mỗi lần đánh giá (10.240 bước) để hỗ trợ khôi phục tiến trình.

---

### 🔄 Chế Độ 4: Tiếp Tục Huấn Luyện (Resume Training)
Tiếp tục huấn luyện an toàn từ checkpoint đã có (tự động load lại số bước, metrics cũ và optimizer):
```bash
python training/train_racing.py --resume models/last_agent.pt --steps-to-run 1000000
```
*(Tự động đánh dấu resume mark vào `logs/training_metrics.json` để vẽ vạch nét đứt trên biểu đồ hội tụ, bảo đảm tính minh bạch học thuật)*.

---

### 📈 Chế Độ 5: Xuất Biểu Đồ Học Thuật Ngoại Tuyến (Learning Curves)
Đọc dữ liệu từ `logs/training_metrics.json` và xuất biểu đồ 4 panel chất lượng cao (300 DPI) độc lập mà không ảnh hưởng tới tiến trình train:
```bash
python scripts/plot_learning_curves.py --metrics logs/training_metrics.json --save-path plots/learning_curves.png
```
* **Panel 1**: Episodic Return Convergence (Raw + Đường xu hướng SMA-3).
* **Panel 2**: Track Completion Rate (%) kèm các mốc 50% (Mid-track), 75.4% (S-Turn), 100% (Finish).
* **Panel 3**: Critic MSE Loss (trục trái) song song Actor Policy Loss & Entropy (trục phải).
* **Panel 4**: Tốc độ trung bình (km/h) đối chiếu Tỷ lệ đâm xe (Crash Rate %).

---

### 🏎️ Chế Độ 6: Chẩn Đoán Sim Racing Telemetry Chuyên Sâu (6-Panel Dashboard)
Chạy 1 vòng mô phỏng độc lập với checkpoint model để xuất Dashboard phân tích kỹ thuật:
```bash
python scripts/plot_telemetry_diagnostic.py --model models/best_agent.pt --save-path plots/telemetry_diagnostic.png
```
* **Panel 1**: Speed Trace ($v$ vs $s$) định vị rõ tốc độ đáy cua (V-min) tại Chicane 1, Hairpin 180°, và S-Turn 1650m.
* **Panel 2**: Bản đồ 2D đường đua thực tế + Quỹ đạo xe chạy đổi màu theo dải tốc độ.
* **Panel 3**: Góc bẻ lái vô lăng kèm vùng rẽ Trái/Phải và lực ép ngang Lateral G.
* **Panel 4**: Biểu đồ đạp Ga/Phanh và vùng rà phanh Trail Braking.
* **Panel 5**: G-G Friction Circle Diagram (vòng tròn ma sát khai thác độ bám lốp 0.5G, 1.0G, 1.5G).
* **Panel 6**: Độ lệch ngang tim đường ($d_{\text{lat}}$) đối chiếu giới hạn đường đua FIA $\pm 4.0\text{m}$.

---

### 📊 Chế Độ 7: Đánh Giá Benchmark Đa Vòng (Academic Evaluation)
Đánh giá độ ổn định và xác suất hoàn thành chặng trên 10 vòng đua ngẫu nhiên từ vạch xuất phát:
```bash
python training/evaluate.py --model models/best_agent.pt --episodes 10
```

---

## 📐 5. Khung Lý Thuyết Môn Học Tăng Cường (REL Formulation)

Bài toán được mô hình hóa theo chuẩn **Markov Decision Process (MDP)**:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

### 1. Không Gian Quan Sát (Observation Space $\mathcal{S} \in \mathbb{R}^{40}$)
* **16 chiều Ego Kinematics, IMU & Controls**:
  $$[v_x/40, v_y/20, \dot{\psi}/\pi, \beta, a_{\text{long}}, a_{\text{lat}}, \phi_{\text{roll}}/0.35, \theta_{\text{pitch}}/0.35, \delta_{t-1}, \text{thr}_{t-1}, \text{brk}_{t-1}, \Delta\delta, \mathbb{I}_{\text{runoff}}, d_{\text{lat}}/4.5, \sin(\Delta\theta), \cos(\Delta\theta)]$$
  * $\beta = \arctan2(v_y, \max(0.1, v_x)) / (\pi/4)$: Góc trượt thân xe (Body Slip Angle), hỗ trợ cảm nhận giới hạn trượt và hiện tượng thừa/thiếu lái.
  * $a_{\text{long}}, a_{\text{lat}}$: Gia tốc trọng trường dọc và ngang (IMU G-force), giúp Agent cảm nhận vòng tròn ma sát Kamm's Friction Circle.
  * $\phi_{\text{roll}}, \theta_{\text{pitch}}$: Độ nghiêng chao đảo trọng tâm thân xe khi vào cua và phanh gấp.
  * $\Delta\delta$: Tốc độ xoay vô lăng giữa 2 bước liên tiếp.
* **24 chiều Lookahead Horizon Preview (6 Waypoints $\times$ 4)**:
  Tại các cự ly $d \in \{10\text{m}, 25\text{m}, 45\text{m}, 70\text{m}, 105\text{m}, 150\text{m}\}$ phía trước xe dọc trục Frenet $s$:
  $$[\Delta x_{\text{rel}} / 150, \Delta y_{\text{rel}} / 40, \sin(\Delta \theta_{\text{wpt}}), \text{clip}(\kappa_{\text{wpt}} \cdot 50, -2, 2)]$$
  *(Cung cấp tầm nhìn xa lên tới 150m ở dải tốc độ cao > 110 km/h, bảo đảm nguyên tắc Zero Data Leakage vì chỉ phản ánh hình học tim đường).*

### 2. Không Gian Hành Động 3 Trục Độc Lập (Decoupled Action Space $\mathcal{A} \in [-1, 1]^3$)
* $\mathbf{a}_t = [\delta_t, \text{thr}_t, \text{brk}_t] \in [-1, 1]^3$:
  * $\delta_t \in [-1, 1]$: Tỷ lệ góc bẻ lái vô lăng liên tục (Trái $\leftrightarrow$ Phải).
  * $\text{thr}_t \in [-1, 1]$: Chân ga độc lập (Chuẩn hóa từ $[0, 1]$).
  * $\text{brk}_t \in [-1, 1]$: Chân phanh độc lập (Chuẩn hóa từ $[0, 1]$).
* **Bộ trộn ECU liên tục $C^1$**:
  $$\text{net\_pedal} = \text{thr\_eff} \cdot (1 - \text{brk\_eff}) - \text{brk\_eff}$$
  Kết hợp **Slew Rate Limiter** ($1.0/\text{step}$) triệt tiêu 100% rung giật và tạo hàm gradient khả vi trơn tru cho PPO.

### 3. Phép Bù Jacobian Cho Chính Sách Squashed Gaussian
Khi chuyển từ biến ngẫu nhiên Gaussian $u \sim \mathcal{N}(\mu, \sigma^2)$ sang hành động $a = \tanh(u)$, xác suất log-policy được hiệu chỉnh chính xác bằng định lý chuyển đổi biến ngẫu nhiên:

$$\log \pi(a \vert s) = \log \mu(u \vert s) - \sum_{i=1}^d \log\left(1 - \tanh^2(u_i) + \epsilon\right)$$

Giúp gradient truyền chính xác mà không gặp lỗi số học hay bão hòa tại biên $[-1, 1]$.

### 4. Hàm Phần Thưởng Đua Xe Thể Thao Thuần Túy (Pure Motorsport Markovian Reward)
$$R_t = r_{\text{progress}} + r_{\text{steer\_smooth}} + r_{\text{idle}} + R_{\text{macro}}$$

* **Tiến độ cung đường Frenet**: $r_{\text{progress}} = 0.05 \cdot \max(0, \Delta s)$ (thưởng tịnh tiến dọc trục đường đua, bảo toàn tính Markov).
* **Phạt rung giật đánh lái**: $r_{\text{steer\_smooth}} = -0.01 \cdot (\Delta \text{steer})^2$ (giữ ổn định toán học, không cản trở vào cua gắt).
* **Không phạt tim đường ($r_{\text{center}} = 0$)**: Tuyệt đối giải phóng Agent khỏi tim đường để tự do mở góc cua Out - In - Out và chém Apex.
* **Phạt đứng yên / lề mề**: $r_{\text{idle}} = -0.15$ nếu $v < 10\text{ km/h}$.
* **Macro Events**:
  * Phạt va chạm / 4 bánh ra ngoài: $R_{\text{crash}} = -10.0 - 20.0 \cdot (v/120)^2$.
  * Thưởng về đích: $R_{\text{finish}} = +100.0 + \max(0, (130.0 - T_{\text{lap}}) \times 1.5)$.

### 5. Thuật Toán PPO Continuous & Critic Loss Mở Rộng
* **Clipped Surrogate Policy Loss**:
  $$L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
* **Unclipped Critic MSE Loss**:
  $$L^{\text{VF}}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left(V_\phi(s_t) - \hat{R}_t\right)^2 \right]$$

---

## 🏁 6. Kết Quả Huấn Luyện Thực Nghiệm & Đối Chiếu Benchmark

| Chỉ Số Đánh Giá | Baseline Cũ (LiDAR 259D + Single Pedal) | AI Agent PPO (`best_agent.pt`) | Tác Tử Đối Chứng Tối Ưu (`OptimalRacingAgent`) | Giới Hạn Cận Trên Vật Lý Lý Thuyết |
| :--- | :---: | :---: | :---: | :---: |
| **Vận tốc cao nhất ($V_{\max}$)** | $42.1\text{ km/h}$ | **$118.4\text{ km/h}$** | **$120.0\text{ km/h}$** | $120.0\text{ km/h}$ (Trần GT3) |
| **Vận tốc trung bình ($V_{\text{avg}}$)** | $33.5\text{ km/h}$ | **$108.2\text{ km/h}$** | **$102.1\text{ km/h}$** | $103.5\text{ km/h}$ |
| **Thời gian 1 Lap ($T_{\text{lap}}$)** | DNF (Kẹt cua) | **$111.30\text{s}$** | **$118.20\text{s}$** | **$116.99\text{s}$** (Cận trên lý thuyết) |
| **Tỷ lệ hoàn thành (Completion)** | $71.4\%$ | **$98.5\%$ (Hoàn thành)** | **$100.0\%$ (Clean Lap)** | $100.0\%$ |
| **Sai số Racing Line (MAE)** | Không đánh giá | $1.46\text{m}$ (Tự khám phá) | **$0.46\text{m}$** (Bám sát lý thuyết) | $0.00\text{m}$ |
| **Kỹ thuật phanh** | Liệt phanh (bão hòa) | **Trail-Braking tự sinh** | **Kamm Trail-Braking + Lift-and-Coast** | Giới hạn ma sát lốp $\mu = 1.35$ |

---

## 👥 7. Thông Tin Đồ Án
* **Môn học:** Học Tăng Cường (Reinforcement Learning - REL)
* **Mã nguồn:** Lưu trữ và quản lý phiên bản trên GitHub.
* **Tài liệu tham khảo chuyên sâu:** Xem thư mục [`knowledge/`](file:///c:/Users/tunbi/Project/school/REL%20Assignment/knowledge) để xem chi tiết các chứng minh toán học và phân tích động lực học xe đua.
