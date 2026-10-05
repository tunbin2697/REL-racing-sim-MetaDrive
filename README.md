# 🏎️ METADRIVE 3D AUTONOMOUS RACING & HUMAN VS AI BENCHMARK
> **Đồ Án Môn Học Tăng Cường (Reinforcement Learning - REL)**  
> **Môi Trường Mô Phỏng:** MetaDrive Simulator (Panda3D Physics Engine)  
> **Thuật Toán Cốt Lõi:** PPO Continuous Action Space + Behavior Cloning Warm Start

---

## 📖 1. Giới Thiệu Đồ Án

Dự án phát triển một hệ thống xe tự hành đua xe trong môi trường không gian 3D tiêu chuẩn quốc tế **MetaDrive Simulator**, tuân thủ 100% khung lý thuyết môn **Học Tăng Cường (REL)**:
- **Khắc phục hoàn toàn lỗi "mớm đường đua" (Over-Reward Shaping)**: Agent hoàn toàn **không** được cung cấp trước tọa độ waypoints tương lai, góc cua sắp tới, hay đường kẻ apex định sẵn. Agent phải "tự nhìn tự lái" thông qua **240 tia LiDAR $360^\circ$** kết hợp cảm biến vận tốc và góc lái nội tại.
- **Tính năng đinh - Đua trực tiếp Người vs AI (`play_vs_ai.py`)**: Người chơi dùng bàn phím (`W-A-S-D`) trực tiếp so tài tốc độ với AI Agent trên đường đua 3D 60FPS. Cả hai xe cùng xuất phát cố định trên vạch xuất phát chuẩn F1, tương tác vật lý thời gian thực, có bảng đồng hồ HUD và tự động phân định thắng thua.
- **Kiến trúc học tăng cường hiện đại**: Kết hợp **Imitation Learning (Behavior Cloning)** khởi tạo tri thức bám cua/điều tiết ga-phanh cùng **PPO Continuous Actor-Critic (GAE-$\lambda$, Clipped Surrogate Objective)**.

---

## 📂 2. Cấu Trúc Mã Nguồn

```text
REL Assignment/
├── game_env/
│   ├── __init__.py
│   ├── racing_env.py          # Wrapper MetaRacingEnv (Chuẩn Gymnasium, hàm thưởng tự nhiên)
│   └── multiagent_env.py      # Môi trường đua đối đầu 2 xe (Người vs AI)
│
├── agents/
│   ├── __init__.py
│   ├── networks.py            # Mạng Actor-Critic liên tục (Gaussian Policy, Orthogonal Init)
│   └── ppo_agent.py           # Thuật toán PPO Continuous (GAE-Lambda, Minibatch SGD)
│
├── training/
│   ├── pretrain_expert.py     # Tiền huấn luyện bắt chước (Behavior Cloning Warm Start)
│   ├── train_racing.py        # Pipeline huấn luyện PPO + tự động xuất biểu đồ học thuật
│   ├── evaluate.py            # Benchmark đánh giá học thuật độc lập trên nhiều scenario
│   └── plot_telemetry.py      # Xuất bộ 4 biểu đồ Sim Racing Telemetry chuyên sâu
│
├── demo/
│   ├── play_vs_ai.py          # SCRIPT ĐINH: Người bấm phím W-A-S-D trực tiếp đua với AI
│   └── spectate_ai.py         # Chế độ Cinematic theo dõi AI tự lái đa góc quay
│
├── scripts/
│   └── test_metadrive_install.py # Lái thử xe tự do bằng bàn phím để kiểm tra đồ họa & vật lý
│
├── models/
│   ├── best_agent.pt          # Model AI tốt nhất đã huấn luyện (đã lưu sẵn để chạy ngay)
│   ├── warmup_racing_agent.pt # Model Warm-up trên 12 blocks circuit
│   └── latest_agent.pt        # Model checkpoint mới nhất
│
├── plots/
│   ├── learning_curves.png    # Biểu đồ hội tụ học thuật (Return, Loss, Completion, Speed)
│   └── telemetry_analysis.png # Bộ 4 biểu đồ Sim Racing: Speed trace, Pedals, G-G, Steering
│
├── logs/
│   ├── reference_lap.json     # Dữ liệu vòng đua chuẩn (Ghost Lap) tính Delta Bar thời gian thực
│   └── training_metrics.json  # Dữ liệu telemetry chi tiết qua các vòng lặp huấn luyện
│
├── archived_ver1/             # Toàn bộ mã nguồn và mô hình 2D cũ được đóng gói độc lập
├── knowledge/                 # 7 tài liệu lý thuyết chuẩn môn Học Tăng Cường
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── .gitignore                 # Cấu hình bỏ qua các file tạm
└── README.md                  # Hướng dẫn chi tiết dự án
```

---

## ⚙️ 3. Hướng Dẫn Cài Đặt (Setup Instructions)

### Bước 1: Clone dự án về máy
```bash
git clone <URL_REPO_CUA_BAN>
cd "REL Assignment"
```

### Bước 2: Khởi tạo môi trường Conda (Khuyến nghị Python 3.10)
```bash
conda create -n rel_racing python=3.10 -y
conda activate rel_racing
```

### Bước 3: Cài đặt PyTorch với CUDA (Nếu máy có card GPU NVIDIA)
*Dành cho máy có GPU (CUDA 12.4):*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*Dành cho máy chỉ dùng CPU:*
```bash
pip install torch torchvision
```

### Bước 4: Cài đặt các thư viện mô phỏng MetaDrive & phụ trợ
```bash
pip install -r requirements.txt
```

---

## 🎮 4. Hướng Dẫn Chạy Các Chế Độ (Execution Guide)

Sau khi kích hoạt môi trường `conda activate rel_racing`, bạn có thể chạy các tính năng:

### 🏆 Chế Độ 1: Đua Trực Tiếp Người vs AI (Human vs AI Race - Demo Chính)
Mở cửa sổ đồ họa 3D Panda3D 60FPS để trực tiếp so tài với AI đã được huấn luyện:
```bash
python demo/play_vs_ai.py --model models/best_agent.pt --map O
```
* **Xe của bạn (Player)**: 🟢 **Xanh Lá Cây (Pole Position)**. Điều khiển bằng cụm phím **`W` (Ga), `S` (Thắng/Lùi), `A` (Rẽ trái), `D` (Rẽ phải)**.
* **Xe AI Agent**: 🔴 **Đỏ Thể Thao (P2 Grid)**. Tự động đọc dữ liệu từ cảm biến LiDAR và lái xe theo mạng nơ-ron PPO.
* **Phím camera**:
  * Bấm **`C`**: Chuyển đổi góc nhìn camera giữa Xe Bạn và Xe AI.
  * Bấm **`B`**: Chuyển sang góc nhìn toàn cảnh từ trên cao (Bird-eye view).
* **Luật thi đấu**: Phân định thắng/thua thời gian thực trên bảng HUD (Xe nào về đích trước, hoặc xe nào bị đâm rào/lệch đường trước sẽ bị xử thua).

---

### 🚗 Chế Độ 2: Lái Thử Xe Tự Do (Kiểm tra đồ họa & vật lý)
```bash
python scripts/test_metadrive_install.py
```
Dùng để kiểm tra độ trễ, lực ma sát lốp xe và cảm giác lái tự do bằng bàn phím `W-A-S-D`.

---

### 🎬 Chế Độ 3: Xem AI Tự Lái Góc Quay Điện Ảnh (Cinematic Spectator)
```bash
python demo/spectate_ai.py --model models/best_agent.pt --map O
```
Theo dõi AI tự động xử lý các góc cua, phanh nhả nhịp nhàng và hiển thị bảng đo telemetry thời gian thực.

---

### 🧠 Chế Độ 4: Tiền Huấn Luyện Bắt Chước (Behavior Cloning Warm Start)
Thu thập dữ liệu bay mẫu của chuyên gia và khởi tạo tri thức ban đầu cho mạng Actor:
```bash
python training/pretrain_expert.py --steps 5000 --epochs 100 --map O
```

---

### 🚀 Chế Độ 5: Huấn Luyện Học Tăng Cường (PPO Continuous RL Training)
Huấn luyện tăng cường từ đầu hoặc tinh chỉnh mô hình đã có với tốc độ ~180 FPS trên GPU:
```bash
python training/train_racing.py --total-steps 20000 --rollout-len 512 --batch-size 64 --map O
```
* Checkpoint kỷ lục tự động lưu tại: `models/best_agent.pt`.
* Đồ thị học thuật được tự động vẽ tại: `plots/learning_curves.png`.

---

### 📊 Chế Độ 6: Đánh Giá Benchmark Học Thuật (Evaluation)
Chạy thử nghiệm kiểm chứng độ tin cậy trên 10 vòng đua ngẫu nhiên:
```bash
python training/evaluate.py --model models/best_agent.pt --episodes 10
```

---

## 📐 5. Khung Lý Thuyết Môn Học Tăng Cường (REL Formulation)

Bài toán được thiết kế chuẩn mực theo mô hình **Markov Decision Process (MDP)**:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

### 1. Không gian Trạng thái (Observation Space $\mathcal{S} \in \mathbb{R}^{259}$)
* **240 tia LiDAR 3D**: Quét $360^\circ$ xung quanh thân xe, đo khoảng cách chính xác đến mép đường, rào chắn, xe đối thủ và vật cản.
* **19 chiều Telemetry nội tại**: Vận tốc dài $v_x$, vận tốc ngang $v_y$, góc đánh lái $\delta$, gia tốc xe, vận tốc góc quay thân xe $\omega_z$, và vector điều hướng cơ bản.
* *Tuyệt đối không sử dụng tọa độ waypoint định sẵn để mớm đường chạy.*

### 2. Không gian Hành động (Action Space $\mathcal{A} \in [-1, 1]^2$)
* $a_t = [\text{steering}, \text{throttle/brake}] \in [-1, 1]^2$:
  * $a_t[0]$: Góc bẻ lái vô lăng liên tục (Trái $\leftrightarrow$ Phải).
  * $a_t[1]$: Chân ga và phanh (Thắng gấp $[-1] \leftrightarrow$ Đạp hết ga $[+1]$).

### 3. Hàm Phần thưởng Tự nhiên (Natural Reward Function $\mathcal{R}$)
$$R_t = R_{\text{driving}} \cdot \Delta d_{\text{longitudinal}} \cdot \lambda_{\text{lateral}} + R_{\text{speed}} \cdot \frac{v}{v_{\text{max}}} - c_{\text{step}} - c_{\text{crash}} \cdot \mathbb{I}_{\text{crash}} - c_{\text{out}} \cdot \mathbb{I}_{\text{out}} + R_{\text{success}} \cdot \mathbb{I}_{\text{arrive}}$$

* Trong đó: $\lambda_{\text{lateral}} = \text{clip}\left(1 - \frac{2|d_{\text{lateral}}|}{w_{\text{lane}}}, 0, 1\right)$ là hệ số giữ làn tự nhiên, triệt tiêu phần thưởng tiến tới nếu xe bị trượt văng ra lề đường.

### 4. Thuật toán PPO Continuous (Proximal Policy Optimization)
* **Clipped Surrogate Objective**:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
* **Generalized Advantage Estimation (GAE-$\lambda$)**:
  $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
  Với TD error: $\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)$.

---

## 📈 6. Kết Quả Huấn Luyện & Thực Nghiệm

* **Tốc độ mô phỏng**: Đạt ~180 FPS trên GPU NVIDIA.
* **Tỷ lệ hoàn thành chặng đua (Completion Rate)**: Đạt **$98.0\%$** trên bản đồ đua vòng lặp.
* **Vận tốc trung bình trong trận đua**: $35 - 42 \text{ km/h}$.
* **Hành vi thông minh học được**:
  * Đoạn thẳng: Tăng ga tối đa bứt tốc.
  * Trước khúc cua: Tự động hãm ga để giảm lực ly tâm.
  * Trong khúc cua: Bẻ góc lái $-0.30$ mượt mà ôm sát tim đường.
  * Thoát cua: Trả thẳng lái và bứt tốc về đích.

*(Toàn bộ biểu đồ học thuật chi tiết được lưu trong [plots/learning_curves.png](file:///c:/Users/tunbi/Project/school/REL%20Assignment/plots/learning_curves.png)).*
