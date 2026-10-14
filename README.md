# 🏎️ MetaDrive 3D Autonomous Racing — v2
> **Đồ án môn Học Tăng Cường (Reinforcement Learning – REL)**
> **Môi trường:** MetaDrive Simulator 0.4.3 (Panda3D / Bullet physics) — đường đua khép kín 2,186 m
> **Thuật toán:** PPO liên tục (Squashed Gaussian + Jacobian) · Policy luật Pure Pursuit · PPO v2 Residual khởi tạo bằng Behavior Cloning

---

## 🏁 1. Kết quả chính

Mỗi policy chạy **20 vòng deterministic từ vạch xuất phát** (`scripts/benchmark_policies.py`):

| Policy | Về đích | Completion TB | Tốc độ TB | Lap time TB |
| :--- | :---: | :---: | :---: | :---: |
| PPO v1 gốc (`models/best_agent.pt`) | 0 / 20 | 71.7 % | 73.9 km/h | — (luôn văng tại 1,556 m) |
| PPO v1 sau sửa lỗi + tune reward (`models/final_agent.pt`) | 10 / 20 | 91.7 % | 67.7 km/h | 115.1 s |
| Policy luật 0.8g (`rule`) | **20 / 20** | 98.5 % | 72.5 km/h | 105.6 s |
| Policy luật 0.85g (`rule-fast`, sát giới hạn văng) | **20 / 20** | 98.6 % | 73.1 km/h | 105.0 s |
| PPO v2 – chỉ Behavior Cloning (`models/v2/bc_warmup_residual.pt`) | **20 / 20** | 98.5 % | 72.4 km/h | 105.7 s |
| **PPO v2 – BC + RL tối ưu lap time** (`models/v2/laptime/latest_agent.pt`) | **20 / 20** | 98.5 % | 72.5 km/h | **105.1 s** |

Số liệu thô: `logs/v2/benchmark_*.json`.

> **Lưu ý:** bảng kết quả ở README cũ (83.9 % completion, 80.8 km/h) không tái hiện được — benchmark lại checkpoint gốc cho 0/20 vòng về đích và luôn văng tại 1,556 m.

---

## 🧭 2. Có gì mới trong v2

### 2.1 Sửa lỗi môi trường & pipeline (ảnh hưởng cả v1)
| Lỗi | Hậu quả | Sửa |
| :--- | :--- | :--- |
| `--milestone-respawn` không cập nhật chỉ số tìm kiếm tim đường sau khi dịch xe | Xe hồi sinh giữa đường bị tính là "văng" ngay step đầu (−15đ) → cơ chế reverse curriculum **chưa từng hoạt động**, critic học sai vùng trước cua khó | Cập nhật `current_track_idx` sau khi tìm kiếm toàn cục trong `reset()` |
| `dt = 0.02` nhưng MetaDrive gộp `decision_repeat = 5` bước vật lý | Lap time / gia tốc G sai 5 lần | `dt = 0.02 × 5 = 0.1 s` |
| Bộ đếm "FINISH" cộng mọi step có completion ≥ 98 % | Log báo hàng trăm lần về đích ảo | Chỉ đếm khi `termination_reason == "lap_completed"` |
| Dừng xe trước cua chỉ tốn ~−24đ, rẻ hơn văng (−30đ) | **Reward hacking**: agent học cách đứng yên trước khúc cua khó | Phạt −30 khi bị kết thúc vì đứng yên |

### 2.2 Hàm thưởng mới
* **Phạt vượt tốc theo vật lý** thay cho phạt cố định −0.03: tốc độ cho phép tính từ độ cong 70 m phía trước theo đường bao phanh
  $v_{allow} = \min_d \sqrt{a_{lat}/|\kappa_d| + 2 a_{brake} d}$, phạt $-\min(0.5,\ 0.05\,(\Delta v / 5)^2)$.
  Gia tốc ngang $a_{lat}$ là tham số `overspeed_a_lat` của env (mặc định 0.65g; v2 dùng 0.9g ≈ giới hạn văng thực tế).
* Phạt văng khỏi đường −15 → **−30**.

### 2.3 Policy luật (`agents/rule_policy.py`)
Không học, chỉ đọc vector quan sát 35 chiều:
* **Lái:** Pure Pursuit $\delta = \arctan(2L\sin\alpha / L_d)$, điểm ngắm $L_d = 0.6v + 8$ m nội suy giữa các waypoint nhìn trước.
* **Ga/phanh:** bám tốc độ mục tiêu $v_{allow}$ (cùng công thức với reward), hệ số tỷ lệ 0.15 / (km/h).
* Giới hạn bám đo thực nghiệm: 0.85g về đích 10/10, **0.9g văng 10/10** → mặc định 0.8g để có biên an toàn.

### 2.4 PPO v2 (`agents/networks_v2.py`, `agents/ppo_agent_v2.py`)
| Thành phần | v1 | v2 |
| :--- | :--- | :--- |
| Đầu vào | 35 chiều | 35 chiều + **4 đặc trưng vật lý** (v_allow, vượt tốc, lái & ga gợi ý của Policy luật) + LayerNorm |
| Đầu ra Actor | $\mu(s)$ | **Residual:** $u = \operatorname{atanh}(a_{rule}) + 0.5\tanh(f_\theta(s))$ — khởi đầu đúng bằng Policy luật |
| Độ lệch chuẩn | 1 tham số chung cho lái & ga | Bias riêng từng hành động + head phụ thuộc trạng thái; **σ_lái ≈ 0.03, σ_ga ≈ 0.30** |
| Khởi tạo | Ngẫu nhiên | Behavior Cloning từ Policy luật (DAgger-style relabel) + critic học return Monte Carlo |
| Ổn định fine-tune | — | Critic warmup (5 update đầu đóng băng Actor) + BC regularization giảm dần |

### 2.5 Bài học rút ra khi huấn luyện
1. **Xe cực nhạy với nhiễu lái:** cộng nhiễu σ = 0.1 vào tay lái của Policy luật → văng 3/3 vòng; nhiễu ga σ = 0.3 vẫn về đích 3/3. Dùng chung một σ cho cả hai hành động (như v1) khiến rollout gần như luôn văng.
2. **Fine-tune PPO trực tiếp từ BC sụp đổ** sau 4–20k step (lái lệch đều ~0.1 → văng ở cua dễ). Residual policy + σ tách riêng giải quyết triệt để: 109 lần về đích trong rollout, 0 lần văng khi đánh giá suốt 150k step.
3. **Đánh giá 3 episode không đủ tin cậy:** kết quả mỗi episode phụ thuộc thứ tự episode từ lúc khởi tạo engine, nên `best_agent.pt` do script tự chọn có lúc chỉ về đích 1/20. Luôn chọn checkpoint bằng benchmark ≥ 20 vòng.
4. **Reward phải có "lực kéo" về lap time:** khi giới hạn phạt vượt tốc (0.65g) thấp hơn nhịp chạy thật (0.8g), RL tăng return bằng cách *chạy chậm lại*; nâng lên 0.9g thì RL nhanh hơn Policy luật nền 0.5 s/vòng.

---

## 📂 3. Cấu trúc thư mục

```text
REL-racing-sim-MetaDrive/
├── agents/
│   ├── networks.py            # [v1] Squashed Gaussian Actor-Critic
│   ├── ppo_agent.py           # [v1] PPO (GAE, clipped surrogate, decoupled optimizer)
│   ├── racing_features.py     # [v2] Đặc trưng vật lý từ obs (numpy + torch): v_allow, Pure Pursuit
│   ├── rule_policy.py         # [v2] Policy luật Pure Pursuit + profile tốc độ
│   ├── networks_v2.py         # [v2] Actor-Critic residual, physics features, std theo trạng thái
│   ├── ppo_agent_v2.py        # [v2] PPO + critic warmup + BC regularization
│   └── policy_loader.py       # [v2] Nạp 'rule' / 'rule-fast' / checkpoint v1 / v2
├── game_env/
│   ├── racing_env.py          # Wrapper MetaRacingEnv (obs 35 chiều, reward, track limits)
│   ├── multiagent_env.py      # Môi trường 2 xe Người vs AI
│   └── telemetry.py           # Telemetry (tốc độ, G-G, delta time)
├── training/
│   ├── train_racing.py        # [v1] Pipeline PPO chính
│   ├── pretrain_bc_v2.py      # [v2] Behavior Cloning từ Policy luật
│   ├── train_racing_v2.py     # [v2] PPO v2 (dùng lại train_racing.py, thêm tham số v2)
│   ├── evaluate.py, pretrain_expert.py, train_dagger_warmup.py, plot_telemetry.py
├── demo/
│   ├── spectate_v2.py         # [v2] Xem mọi loại policy trong 3D, HUD tốc độ cho phép
│   ├── spectate_ai.py         # [v1] Xem PPO v1
│   └── play_vs_ai.py          # [v1] Người vs AI (chỉ PPO v1)
├── scripts/
│   ├── benchmark_policies.py  # [v2] So sánh nhiều policy trên N vòng
│   └── plot_learning_curves.py, plot_telemetry_diagnostic.py, debug_reward_breakdown.py, test_metadrive_install.py
├── models/
│   ├── best_agent.pt …        # Checkpoint PPO v1 gốc
│   ├── final_agent.pt         # PPO v1 tốt nhất sau sửa lỗi (10/20 vòng)
│   ├── step1/                 # Kết quả fine-tune bước 1 (trước khi đổi reward)
│   └── v2/
│       ├── bc_warmup_residual.pt     # PPO v2 sau Behavior Cloning
│       ├── residual/latest_agent.pt  # PPO v2 sau RL (reward 0.65g)
│       └── laptime/latest_agent.pt   # PPO v2 sau RL tối ưu lap time (khuyên dùng)
├── logs/                      # training_metrics.json từng bước, benchmark_*.json
├── knowledge/                 # Tài liệu lý thuyết môn REL & Sim Racing
└── archived_ver1/             # Phiên bản 2D đầu tiên
```

---

## ⚙️ 4. Cài đặt (không dùng Conda)

Yêu cầu **Python 3.10**. Nếu máy chưa có, có thể dùng [uv](https://docs.astral.sh/uv/) để tự tải Python 3.10.

```powershell
# 1. Tạo môi trường ảo (chọn một cách)
uv venv .venv --python 3.10 --seed       # có uv
py -3.10 -m venv .venv                   # đã cài sẵn Python 3.10

# 2. Kích hoạt (PowerShell) — mỗi lần mở terminal mới
.\.venv\Scripts\Activate.ps1             # cmd: .venv\Scripts\activate.bat | Linux/macOS: source .venv/bin/activate

# 3. PyTorch: GPU NVIDIA (CUDA 12.4) hoặc CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# pip install torch torchvision

# 4. Thư viện còn lại
pip install -r requirements.txt

# 5. Kiểm tra cửa sổ 3D (lái thử bằng W-A-S-D)
python scripts/test_metadrive_install.py
```

> Nếu gặp `ModuleNotFoundError: No module named 'gymnasium'`, terminal đang dùng Python hệ thống — hãy kích hoạt `.venv` hoặc gọi thẳng `.\.venv\Scripts\python.exe`.

---

## 🎮 5. Chạy thử

### 5.1 Xem policy chạy trong 3D
```powershell
python demo/spectate_v2.py                                    # PPO v2 (mặc định)
python demo/spectate_v2.py --policy rule                      # Policy luật
python demo/spectate_v2.py --policy models/final_agent.pt     # PPO v1
python demo/spectate_v2.py --realtime --laps 3                # Phát theo thời gian thực, 3 vòng
```
Phím: `C` đổi camera · `B` nhìn từ trên cao · `Esc` thoát.

### 5.2 Benchmark
```powershell
python scripts/benchmark_policies.py rule rule-fast models/final_agent.pt models/v2/laptime/latest_agent.pt --episodes 20
```

### 5.3 Người vs AI (PPO v1)
```powershell
python demo/play_vs_ai.py --model models/final_agent.pt
```
> `multiagent_env.py` tính quan sát hơi khác `racing_env.py` (tim đường lệch 2.25 m, độ cong theo bán kính làn), nên chưa dùng được cho Policy luật / PPO v2.

---

## 🚀 6. Huấn luyện

### 6.1 PPO v2 (khuyên dùng)
```powershell
# Bước 1 — Behavior Cloning từ Policy luật (~3 phút)
python training/pretrain_bc_v2.py --steps 40000 --noise 0.08 --perturb-prob 0.003 `
    --save-path models/v2/bc_warmup_residual.pt

# Bước 2 — PPO residual (~40 phút / 150k step trên CPU 1 luồng)
python training/train_racing_v2.py --warmup-model models/v2/bc_warmup_residual.pt `
    --total-steps 150000 --milestone-respawn --lr 0.00015 --ent-coef 0.0 `
    --eval-freq 4 --eval-episodes 3 --overspeed-a-lat 8.83 --bc-coef 1.0 `
    --log-dir logs/v2/rl_laptime --save-dir models/v2/laptime
```
Tham số riêng của v2: `--overspeed-a-lat` (m/s², giới hạn phạt vượt tốc), `--bc-coef`, `--bc-decay`, `--critic-warmup-updates`.
Không truyền `--reset-logstd` cho v2: σ riêng từng hành động đã nằm sẵn trong checkpoint BC.

### 6.2 PPO v1
```powershell
python training/train_racing.py --resume models/best_agent.pt --steps-to-run 150000 --milestone-respawn `
    --lr 0.00015 --reset-logstd -1.2 --ent-coef 0.005 --eval-freq 4 --eval-episodes 3 `
    --log-dir logs/run_v1 --save-dir models/run_v1
```
Chuỗi thực nghiệm dẫn tới `final_agent.pt` (step 1 → 5) được ghi trong `logs/step*/training_metrics.json`.

### 6.3 Theo dõi & biểu đồ
```powershell
tensorboard --logdir runs
python scripts/plot_learning_curves.py --metrics logs/v2/rl_laptime/training_metrics.json --save-path plots/learning_curves_v2.png
```

---

## 📐 7. Mô hình hoá MDP

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle,\quad \gamma = 0.99,\ \Delta t = 0.1\text{ s}$$

### 7.1 Không gian quan sát $\mathcal{S} \in \mathbb{R}^{35}$
* **10 chiều động học:** $[v/40,\ v_x/40,\ v_y/20,\ \dot\psi/\pi,\ \delta_t,\ \delta_{t-1},\ u_{t-1},\ \cos\Delta\theta,\ \sin\Delta\theta,\ d_{lat}/4.5]$
* **5 chiều giới hạn & tiến độ:** $[d_{left},\ d_{right},\ \mathbb{I}_{runoff},\ s/L,\ \text{clip}(50\kappa, \pm2)]$
* **20 chiều nhìn trước:** 5 waypoint tại $\{10, 20, 35, 50, 70\}$ m × $[\Delta x/70,\ \Delta y/30,\ \sin\Delta\theta_{wpt},\ \text{clip}(50\kappa_{wpt}, \pm2)]$

### 7.2 Không gian hành động $\mathcal{A} = [-1, 1]^2$
$a = [\tanh u_{steer},\ \tanh u_{acc}]$; env làm mượt lái $\delta_{applied} = 0.7\,a_{steer} + 0.3\,\delta_{t-1}$, góc lái tối đa 40°, xe giới hạn 80 km/h.

### 7.3 Hàm thưởng
$$R_t = r_{time} + r_{progress} + r_{speed} + r_{overspeed} + r_{sector} + r_{idle} + R_{macro}$$

| Thành phần | Công thức |
| :--- | :--- |
| Áp lực thời gian | $-0.01$ / step |
| Tiến độ Frenet | $0.02 \max(0, \Delta s)\,(0.5 + 0.5 \min(v/80, 1.5))$ |
| Tốc độ | $0.03\,(v/120)^2$ khi tiến về phía trước |
| Vượt tốc vật lý | $-\min(0.5,\ 0.05\,(\max(0, v - v_{allow})/5)^2)$, +0.02 nếu đang phanh khi vượt tốc |
| Qua sector (10 sector) | $+5$ |
| Lề mề | $-0.15$ khi $v < 10$ km/h |
| Văng khỏi đường / lật xe | $-30$ (kết thúc) |
| Đứng yên quá lâu | $-30$ (kết thúc) |
| Về đích | $+100 + \max(0,\ 1.5\,(120 - t_{lap}))$ |

### 7.4 Squashed Gaussian & hiệu chỉnh Jacobian
$$\log \pi(a \mid s) = \log \mathcal{N}(u \mid \mu, \sigma) - \sum_i \log\left(1 - \tanh^2 u_i + \epsilon\right)$$

### 7.5 PPO
* Clipped surrogate: $L^{CLIP} = -\hat{\mathbb{E}}\left[\min(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1 \pm \epsilon)\hat A_t)\right]$, $\epsilon = 0.15$, GAE $\lambda = 0.95$.
* Critic: MSE với value clipping rộng (±10) để chống spike khi va chạm; Actor/Critic dùng optimizer riêng (lr Actor = 0.2·lr, Critic = 1.5·lr).
* **v2 thêm:** $L_{actor} = L^{CLIP} - c_{ent} H + c_{bc}\,\|\tanh\mu(s) - a_{rule}(s)\|^2$, với $c_{bc}$ giảm dần theo hệ số 0.98 mỗi update.

---

## 👥 8. Thông tin đồ án
* **Môn học:** Học Tăng Cường (Reinforcement Learning – REL)
* **Tài liệu lý thuyết:** thư mục [`knowledge/`](knowledge/)
