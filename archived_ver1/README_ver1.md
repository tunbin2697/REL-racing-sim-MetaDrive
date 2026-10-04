# F1 AUTONOMOUS RACING WITH MULTI-POLICY BENCHMARK & FEW-SHOT ADAPTATION

Hệ thống Đua xe Tự hành F1 kết hợp Học Tăng Cường (Reinforcement Learning - Continuous Actor-Critic PPO), Quy hoạch Động Bellman (Bellman Optimal Velocity Profiler - OVP) và Tối ưu hóa Quỹ đạo Đua xe Ngoài - Trong - Ngoài (Out-In-Out Racing Line).

---

## 1. Cấu Trúc Dự Án (Modular Architecture)

```
Assignment/
├── core/                       # Nhân môi trường và mô hình vật lý
│   ├── dynamics.py             # Mô hình động lực học xe F1 (Bicycle Model, góc trượt, ma sát)
│   ├── track_generator.py      # Bộ nạp và sinh đường đua F1 (Monza, Spa, Suzuka, Silverstone, Random)
│   ├── racing_env.py           # Môi trường Gym 2D API (Cảm biến LiDAR 180 độ, va chạm, rewards)
│   ├── circuit_profiler.py     # Khảo sát hình học vi phân kappa(s), Racing line & Bellman OVP
│   └── ppo_agent.py            # Thuật toán Continuous Actor-Critic (Gaussian Policy, PPO-Clip)
│
├── policies/                   # Hệ thống các Chính sách Đua xe (Pluggable Policies)
│   ├── base_policy.py          # Lớp trừu tượng BasePolicy (select_action, reset)
│   ├── centerline_policy.py    # CenterlineSafePolicy & CenterlineSpeedPolicy
│   ├── racing_line_policy.py   # RacingLineOptimalPolicy (Out-In-Out Apex Optimizer)
│   └── ppo_policy.py           # PPOAgentPolicy (Inference mạng nơ-ron sâu f1_racing_agent.pt)
│
├── evaluation/                 # Công cụ Đánh giá & Xuất Dữ liệu
│   ├── compare_policies.py     # Module thực thi benchmark đối đầu A/B testing
│   ├── export_telemetry.py     # Xuất telemetry đa circuit ra JSON/JS cho web visualizer
│   └── generate_race_video.py  # Kết xuất video MP4 cuộc đua F1 kèm HUD telemetry
│
├── data/                       # Dữ liệu hình học & Telemetry
│   ├── silverstone_center.npy  # Tọa độ số hóa Silverstone (400 waypoints)
│   ├── telemetry_data.json     # Dữ liệu phục vụ Web Visualizer
│   └── telemetry_data.js       # Dữ liệu JavaScript cho Canvas 60FPS
│
├── models/                     # Checkpoints trọng số mạng nơ-ron
│   └── f1_racing_agent.pt      # Trọng số Actor-Critic PPO đã tiền huấn luyện
│
├── plots/                      # Toàn bộ biểu đồ phân tích telemetry & video mp4
│   ├── policy_comparison_silverstone.png
│   ├── policy_comparison_monza.png
│   ├── demo_silverstone_telemetry.png
│   ├── demo_monza_telemetry.png
│   ├── demo_spa_telemetry.png
│   ├── demo_suzuka_telemetry.png
│   ├── f1_adaptation_benchmark.png
│   └── demo_monza_race.mp4
│
├── knowledge/                  # Toàn bộ 7 tài liệu Knowledge Base môn REL (Slide 1.1 -> 3.12)
│
├── compare_policies.py         # Entry point: Chạy thử nghiệm đối đầu trực tiếp giữa các Policy
├── demo_racing.py              # Entry point: Chạy mô phỏng cuộc đua thích ứng 3 vòng
├── train.py                    # Entry point: Huấn luyện PPO Agent trên đường đua ngẫu nhiên
└── f1_racing_visualizer.html   # Web Visualizer 60FPS (Chế độ Đối đầu Đa Policy trực quan)
```

---

## 2. Hướng Dẫn Thực Thi Nhanh

### A. Chạy Thử Nghiệm Đối Đầu Đa Policy (A/B Testing: Centerline vs Racing Line)
So sánh trực tiếp `Centerline Safe`, `Centerline Speed (Bellman)`, `Racing Line (Out-In-Out)` và `Pure PPO`:
```bash
# Thử nghiệm trên Silverstone (mạch đua 18 góc cua):
python compare_policies.py --track Silverstone

# Thử nghiệm trên Monza (mạch đua tốc độ cao):
python compare_policies.py --track Monza

# Tùy chọn khác: --track Spa, --track Suzuka
```
*Kết quả biểu đồ đối đầu chất lượng cao sẽ tự động lưu vào thư mục `plots/`.*

### B. Chạy Demo Cuộc Đua Thích Ứng 3 Vòng (Few-Shot Adaptation Demo)
```bash
python demo_racing.py --track Silverstone
python demo_racing.py --track Monza
```

### C. Huấn Luyện Mạng Nơ-ron PPO Mới (Training)
```bash
python train.py --episodes 40 --weights_path models/f1_racing_agent.pt
```

### D. Trải Nghiệm Web Visualizer 60FPS
Mở trực tiếp tập tin `f1_racing_visualizer.html` bằng trình duyệt (Google Chrome, Edge,...).
- Bấm nút **`⚔️ ĐỐI ĐẦU: CENTERLINE vs RACING LINE`** để xem xe Centerline (đỏ) và xe Racing Line (xanh neon) đua song song.
- Dễ dàng chọn chuyển đổi giữa cả 4 mạch đua: **Monza**, **Spa**, **Suzuka**, và **Silverstone**.
