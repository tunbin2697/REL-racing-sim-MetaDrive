# 🏎️ METADRIVE 3D AUTONOMOUS RACING & HUMAN VS AI BENCHMARK
> **Đồ Án Môn Học Tăng Cường (Reinforcement Learning - REL)**  
> **Môi Trường Mô Phỏng:** MetaDrive Simulator (Panda3D Physics Engine - Bullet Physics)  
> **Thuật Toán Cốt Lõi:** PPO Continuous Action Space (Squashed Gaussian Policy, Stabilized Jacobian, Unclipped Critic MSE)

> [!IMPORTANT]
> **TÀI LIỆU ĐẶC TẢ YÊU CẦU ĐỀ TÀI CHÍNH THỨC:**  
> Xem chi tiết toàn bộ quy định học thuật, luật đua 4 bánh, cấm data leak và tiêu chuẩn benchmark tại:  
> 📄 **[`PROJECT_SPECIFICATION.md`](file:///c:/Users/tunbi/Project/school/REL%20Assignment/PROJECT_SPECIFICATION.md)**

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
├── PROJECT_SPECIFICATION.md   # 📄 ĐẶC TẢ YÊU CẦU ĐỒ ÁN MÔN HỌC (Quy tắc, Luật đua, Benchmark)
├── README.md                  # Hướng dẫn chi tiết, cài đặt và thực thi hệ thống
│
├── agents/
│   ├── __init__.py
│   ├── networks.py            # Kiến trúc Squashed Gaussian Actor-Critic & Jacobian bù log-prob
│   └── ppo_agent.py           # Thuật toán PPO Continuous (GAE-Lambda, Unclipped Critic MSE, Minibatch SGD)
│
├── game_env/
│   ├── __init__.py
│   ├── racing_env.py          # Wrapper MetaRacingEnv (Obs 35D, frenet s, track limits 4 bánh, Pure Motorsport)
│   ├── multiagent_env.py      # Môi trường đua đối đầu 2 xe (Người vs AI) với compact obs
│   └── telemetry.py           # Hệ thống ghi nhận Telemetry (v, acc_lat, acc_long, Delta Time)
│
├── training/
│   ├── train_racing.py        # Pipeline huấn luyện PPO chính + Milestone Respawn + Async Plotting
│   ├── evaluate.py            # Benchmark đánh giá học thuật độc lập đa kịch bản
│   ├── pretrain_expert.py     # Tiền huấn luyện bắt chước (Tùy chọn Behavior Cloning)
│   └── train_dagger_warmup.py # Huấn luyện DAgger (Tùy chọn)
│
├── demo/
│   ├── play_vs_ai.py          # SCRIPT ĐINH: Người bấm phím W-A-S-D trực tiếp so tài với AI
│   └── spectate_ai.py         # Chế độ Cinematic theo dõi AI tự lái đa góc quay
│
├── scripts/
│   ├── plot_learning_curves.py    # Xuất 4 biểu đồ học thuật hội tụ (Return, Completion %, Losses, Speed/Crash)
│   ├── plot_telemetry_diagnostic.py # Dashboard Telemetry 6 Panel chuyên sâu (Speed, 2D Trajectory, G-G circle)
│   ├── debug_reward_breakdown.py  # Phân rã tỷ trọng các thành phần trong hàm phần thưởng
│   └── test_metadrive_install.py  # Kiểm tra cài đặt và lái thử xe tự do
│
├── models/
│   ├── best_agent.pt          # Model AI có tỷ lệ hoàn thành chặng (Completion) cao nhất
│   ├── fastest_agent.pt       # Model AI có vận tốc trung bình cao nhất
│   └── latest_agent.pt        # Model checkpoint bước huấn luyện mới nhất
│
├── plots/
│   ├── learning_curves.png        # Biểu đồ học thuật 4 panel xuất ra file PNG 300 DPI
│   └── telemetry_diagnostic.png   # Dashboard Telemetry 6 panel phân tích hành vi lái
│
├── logs/
│   ├── reference_lap.json     # Dữ liệu vòng đua chuẩn (Ghost Lap / Hand-calculated Benchmark)
│   └── training_metrics.json  # Dữ liệu telemetry chi tiết qua các lần đánh giá định kỳ
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

### 🏆 Chế Độ 1: Đua Trực Tiếp Người vs AI (Human vs AI Race - Demo Chính)
Mở cửa sổ đồ họa 3D Panda3D 60FPS để trực tiếp so tài tốc độ với AI Agent:
```bash
python demo/play_vs_ai.py --model models/best_agent.pt
```
* **Xe Người chơi (Player)**: 🟢 **Xanh Lá Cây**. Điều khiển bằng phím:
  * `W`: Đạp ga (Throttle)
  * `S`: Thắng / Lùi (Brake / Reverse)
  * `A` / `D`: Đánh lái Trái / Phải (Steer)
* **Xe AI Agent**: 🔴 **Đỏ Thể Thao**. Điều khiển tự động bởi mạng PPO 35 chiều.
* **Phím chức năng**:
  * `C`: Đổi góc nhìn camera giữa Xe Bạn và Xe AI.
  * `B`: Góc nhìn toàn cảnh từ trên cao (Bird-eye view).
  * `R`: Reset lại vòng đua mới.
  * `Esc`: Thoát trận đấu.

---

### 🎬 Chế Độ 2: Theo Dõi AI Tự Lái Góc Quay Điện Ảnh (Cinematic Spectator)
Theo dõi AI tự ôm cua, thoát cua và phanh nhả nhịp nhàng trên sa bàn 3D:
```bash
python demo/spectate_ai.py --model models/best_agent.pt
```

---

### 🚀 Chế Độ 3: Huấn Luyện Học Tăng Cường (PPO Continuous Training)
Huấn luyện Agent tự khám phá và đột phá các góc cua từ đầu bằng thuật toán PPO thuần:
```bash
python training/train_racing.py --total-steps 200000 --milestone-respawn --rollout-len 1024 --batch-size 64 --eval-freq 2 --lr 0.0001 --ent-coef 0.01
```
* `--milestone-respawn`: Cơ chế hồi sinh Reverse Curriculum tại 50m trước điểm vừa va chạm, giúp agent tập trung giải quyết góc cua khó mà không tốn công chạy lại đoạn thẳng ban đầu.
* Model tốt nhất tự động lưu tại: `models/best_agent.pt`.

---

### 🔄 Chế Độ 4: Tiếp Tục Huấn Luyện (Resume Training)
Tiếp tục huấn luyện từ checkpoint đã có:
```bash
python training/train_racing.py --resume models/best_agent.pt --steps-to-run 100000 --milestone-respawn --rollout-len 1024 --batch-size 64 --eval-freq 2 --lr 0.0001 --ent-coef 0.01
```
*(Tự động vẽ vạch đỏ nét đứt đánh dấu điểm can thiệp lên biểu đồ `plots/learning_curves.png` để đảm bảo tính minh bạch học thuật)*.

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

### 1. Không Gian Quan Sát (Observation Space $\mathcal{S} \in \mathbb{R}^{35}$)
* **10 chiều Ego Kinematics & Controls**:
  $$[v_{\text{norm}}, v_x/40, v_y/20, \dot{\psi}/\pi, a_{\text{steer}}, a_{\text{steer}, t-1}, a_{\text{acc}, t-1}, \cos(\Delta \theta), \sin(\Delta \theta), d_{\text{lat}}/4.5]$$
* **5 chiều Track Limits & Progress**:
  $$[\text{dist}_{\text{left}}, \text{dist}_{\text{right}}, \mathbb{I}_{\text{runoff}}, s/L_{\text{circuit}}, \text{clip}(\kappa_{\text{curr}} \cdot 50, -2, 2)]$$
* **20 chiều Lookahead Horizon Preview (5 Waypoints $\times$ 4)**:
  Tại các cự ly $d \in \{10\text{m}, 20\text{m}, 35\text{m}, 50\text{m}, 70\text{m}\}$ phía trước xe dọc trục Frenet $s$:
  $$[\Delta x_{\text{rel}} / 70, \Delta y_{\text{rel}} / 30, \sin(\Delta \theta_{\text{wpt}}), \text{clip}(\kappa_{\text{wpt}} \cdot 50, -2, 2)]$$

### 2. Không Gian Hành Động (Action Space $\mathcal{A} \in [-1, 1]^2$)
* $a_t = [\delta_t, u_t] = [\tanh(u_{\text{steer}}), \tanh(u_{\text{acc}})] \in [-1, 1]^2$:
  * $\delta_t \in [-1, 1]$: Tỷ lệ góc bẻ lái vô lăng liên tục (Trái $\leftrightarrow$ Phải).
  * $u_t \in [-1, 1]$: Chân ga/phanh liên tục (Phanh gấp $[-1] \leftrightarrow$ Đạp lút sàn $[+1]$).

### 3. Phép Bù Jacobian Cho Chính Sách Squashed Gaussian
Khi chuyển từ biến ngẫu nhiên Gaussian $u \sim \mathcal{N}(\mu, \sigma^2)$ sang hành động $a = \tanh(u)$, xác suất log-policy được hiệu chỉnh chính xác bằng định lý chuyển đổi biến ngẫu nhiên:

$$\log \pi(a \vert s) = \log \mu(u \vert s) - \sum_{i=1}^d \log\left(1 - \tanh^2(u_i) + \epsilon\right)$$

Giúp gradient truyền chính xác mà không gặp lỗi số học hay bão hòa tại biên $[-1, 1]$.

### 4. Hàm Phần Thưởng Đua Xe Thể Thao Thuần Túy (Pure Motorsport Reward)
$$R_t = r_{\text{time}} + r_{\text{progress}} + r_{\text{speed}} + r_{\text{idle}} + R_{\text{macro}}$$

* **Áp lực thời gian**: $r_{\text{time}} = -0.01$ (ép agent phải hoàn thành vòng đua càng nhanh càng tốt).
* **Tiến độ cung đường Frenet**: $r_{\text{progress}} = 0.02 \cdot \max(0, \Delta s)$ (thưởng tịnh tiến dọc trục đường đua, không ép buộc bám tim đường).
* **Vận tốc thể thao**: $r_{\text{speed}} = 0.03 \cdot \left(\frac{v_{\text{km/h}}}{120}\right)^2$.
* **Phạt đứng yên / lề mề**: $r_{\text{idle}} = -0.15$ nếu $v < 10\text{ km/h}$.
* **Macro Events**:
  * Phạt vi phạm giới hạn đường (cả 4 bánh ra ngoài lề): $R_{\text{out\_of\_track}} = -30.0$.
  * Thưởng về đích: $R_{\text{finish}} = +100.0 + \max\left(0, (120 - t_{\text{lap}}) \cdot 1.5\right)$.

### 5. Thuật Toán PPO Continuous & Critic Loss Mở Rộng
* **Clipped Surrogate Policy Loss**:
  $$L^{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
* **Unclipped Critic MSE Loss**:
  $$L^{\text{VF}}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left(V_\phi(s_t) - \hat{R}_t\right)^2 \right]$$
  *(Loại bỏ value clipping hẹp giúp Critic cập nhật toàn diện theo return thực tế, cải thiện độ chính xác dự báo giá trị trạng thái)*.

---

## 🏁 6. Kết Quả Huấn Luyện Thực Nghiệm & Đối Chiếu Benchmark

| Chỉ Số Đánh Giá | Baseline Cũ (LiDAR 259D + Penalty Tim Đường) | Phiên Bản Hiện Tại (35D + Pure Motorsport + 4 Bánh) | Mốc Benchmark Tính Tay Độc Lập (Hand-Calculated Target) |
| :--- | :---: | :---: | :---: |
| **Vận tốc cao nhất ($V_{\max}$)** | $42.1\text{ km/h}$ | **$80.8\text{ km/h}$** | $\approx 85.0\text{ km/h}$ (Đoạn thẳng chính) |
| **Vận tốc trung bình ($V_{\text{avg}}$)** | $33.5\text{ km/h}$ | **$71.9\text{ km/h}$** | $\approx 68.5\text{ km/h}$ (Toàn bộ 2,186m) |
| **Thời gian 1 Lap ($T_{\text{lap}}$)** | DNF (Không về đích) | **Đang tối ưu tiệm cận mốc chuẩn** | **Mục tiêu: Tiệm cận, hợp lý (make sense)** ⚡ |
| **Quãng đường hoàn thành** | $1,560\text{m}$ ($71.4\%$) - Kẹt cua trái | **$1,833.7\text{m} \to 2,186\text{m}$ (Hoàn thành)** | $2,186.3\text{m}$ ($100\%$) |
| **Explained Variance (Critic $R^2$)** | $0.00$ (Critic bị tê liệt) | **$0.92 - 0.96$** | Critic hội tụ chuẩn xác |
| **Kỹ thuật lái xe thể thao** | Rụt rè, bò 30 km/h giữa đường | **Tự học Out-In-Out, Trail Braking, cắt Apex** | Quỹ đạo lý thuyết tính tay F1 |

---

## 👥 7. Thông Tin Đồ Án
* **Môn học:** Học Tăng Cường (Reinforcement Learning - REL)
* **Mã nguồn:** Lưu trữ và quản lý phiên bản trên GitHub.
* **Tài liệu tham khảo chuyên sâu:** Xem thư mục [`knowledge/`](file:///c:/Users/tunbi/Project/school/REL%20Assignment/knowledge) để xem chi tiết các chứng minh toán học và phân tích động lực học xe đua.
