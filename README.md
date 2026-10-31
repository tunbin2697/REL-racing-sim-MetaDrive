# Đua xe tự hành bằng PPO trên MetaDrive

> Đồ án môn Học tăng cường (REL). Môi trường: MetaDrive (Panda3D + Bullet). Thuật toán: PPO với chính sách Gaussian nén bằng tanh.
> Đặc tả đề tài: [`plan/requirements/PROJECT_SPECIFICATION.md`](plan/requirements/PROJECT_SPECIFICATION.md) · Điểm lệch có chủ đích: [`SPECIFICATION_INTENDED_VIOLATION.md`](plan/requirements/SPECIFICATION_INTENDED_VIOLATION.md) · Trạng thái dự án: [`plan/plan.md`](plan/plan.md)

---

## 1. Kết quả chính

**Mô hình tham chiếu `ref_111s`** ([`models/ref_111s/best_agent.pt`](models/ref_111s/best_agent.pt), train ở commit `ee99f9d`; tên lấy từ lap 111.4s lúc train). Đánh giá dưới luật 4 bánh, không kerb ảo, cùng vật lý xe với ghost:

| | Vòng xuất phát đứng | Vòng flying | Vòng sạch |
| :-- | :-- | :-- | :-- |
| Ghost tối ưu đã hiệu chỉnh (T\* = 110.62s) | 118.2s | 112.8s | 3/3 |
| `ref_111s`, 20 episode × 2 vòng | **114.8 ± 0.4s** | **109.4 ± 0.5s** | 18/20 |
| Cùng công thức, seed 43 | 115.2 ± 0.4s | 110.0 ± 0.7s | 19/20 |

Policy RL nhanh hơn bộ điều khiển dựa trên mô hình khoảng 3%. Diễn giải và hạn chế: [`knowledge/rl_debugging_findings.md`](knowledge/rl_debugging_findings.md) F24.

## 2. Bài toán

- **Map**: Silverstone dựng từ các block MetaDrive, khép kín, khoảng 2.85 km, 2 làn × 4.5 m (`silverstone_half`). Có thêm bản tỉ lệ 1:1 (5.9 km, `silverstone_full`), đã đóng băng.
- **Xe**: `GT3RaceCar` là thân xe mặc định của MetaDrive với thông số đã chỉnh (lực động cơ 850 N/bánh, phanh 650 N/bánh, lái 20°, trần 120 km/h). Không phải mô hình GT3 thật. `gt3_anchor` có cùng vật lý, chỉ thêm giới hạn gia tốc đo được cho ghost. Chi tiết: [`knowledge/vehicle_and_engine_physics.md`](knowledge/vehicle_and_engine_physics.md).
- **Obs 35D**: động học xe, khoảng cách tới mép đường, tiến trình, độ cong, 5 điểm tim đường phía trước. Không có dữ liệu tối ưu.
- **Action 3D**: `[lái, ga, phanh]`, giới hạn tốc độ xoay vô lăng 0.18/bước.
- **Reward của `ref_111s`**: thưởng tiến độ dọc tim đường, phạt Δlái², phạt lệch tim đường `r_center`, phạt đứng yên, phạt ra khỏi đường theo tốc độ, thưởng về đích theo lap time. `r_center` lệch với khuyến nghị của đặc tả và được khai báo ở file violation.
- **Luật**: kết thúc khi cả 4 bánh rời mặt nhựa.
- **Mốc so sánh**: racing line cực tiểu độ cong, profile tốc độ tối ưu thời gian, ghost bám bằng MPC dọc và pure pursuit (`game_env/benchmark.py`, `agents/optimal_agent.py`). Chỉ dùng để đối chiếu, không vào obs hay reward.

## 3. Cài đặt

```bash
conda create -n rel_racing python=3.10 -y
conda activate rel_racing
pip install torch torchvision          # hoặc bản CUDA phù hợp máy
pip install -r requirements.txt
```

## 4. Lệnh thường dùng

```bash
# Xem ref_111s đua cùng ghost đã hiệu chỉnh (phím H: trợ giúp, G: đổi xe theo dõi, L: bật/tắt racing line)
python demo/spectate_ai.py --model models/ref_111s/best_agent.pt --map silverstone --vehicle gt3_anchor --ghost

# Chỉ xem ghost
python demo/spectate_ai.py --agent optimal --map silverstone --vehicle gt3_anchor

# Đánh giá (bảng ở mục 1)
python training/evaluate.py --model models/ref_111s/best_agent.pt --map silverstone --vehicle gt3_anchor --episodes 20 --laps 2
python training/evaluate.py --agent optimal --map silverstone --vehicle gt3_anchor --episodes 3 --laps 2

# Train hoặc fine-tune (xem --help; các cờ reward như --center-coef, --time-penalty đều mặc định giữ hành vi cũ)
python training/train_racing.py --help
```

`demo/play_vs_ai.py` là demo người chơi so với AI kiểu cũ (map mặc định là đường oval).

## 5. Cấu trúc thư mục

```text
agents/        ppo_agent.py (PPO), networks.py, optimal_agent.py (ghost: MPC dọc + pure pursuit)
game_env/      racing_env.py (env đơn xe), vehicle.py (các xe), benchmark.py (racing line + profile),
               ghost_car.py, track_presets.py (map half/full), silverstone_map.py, racing_line_obs.py,
               spawn_curriculum.py, telemetry.py, multiagent_env.py (demo người vs AI cũ)
training/      train_racing.py, evaluate.py, evaluator.py
demo/          spectate_ai.py, play_vs_ai.py
scripts/       calibrate_vehicle.py, drive_trace.py, plot_*.py, solve_silverstone_layout.py,
               tests/ (smoke test), overnight/ (lệnh chạy dài), *_optimal_profile.npz (cache profile)
models/        ref_111s/ (kết quả chính), ref_111s_seed43/, p1_main/, p2_*/, legacy_*/ (các run cũ)
logs/          anchor/ (số liệu so sánh chính), metrics của từng run
knowledge/     phân tích dự án (xem mục 6) và ghi chú môn học (01–08)
plan/          trạng thái, đặc tả, bàn giao; plan/archive/ là plan cũ
report/        báo cáo (đang viết)
```

## 6. Tài liệu

| File | Nội dung |
| :-- | :-- |
| [`plan/plan.md`](plan/plan.md) | Trạng thái dự án, việc còn lại, chiến lược viết báo cáo |
| [`plan/team_handoff_multicar.md`](plan/team_handoff_multicar.md) | Bàn giao demo nhiều xe |
| [`report/README.md`](report/README.md) | Quy chuẩn viết báo cáo |
| [`knowledge/rl_debugging_findings.md`](knowledge/rl_debugging_findings.md) | Các phát hiện F1–F24 |
| [`knowledge/vehicle_and_engine_physics.md`](knowledge/vehicle_and_engine_physics.md) | Vật lý xe, hiệu chỉnh, benchmark, sai số mô hình |
| [`knowledge/related_work_racing_rl.md`](knowledge/related_work_racing_rl.md) | GT Sophy, Fuchs, và các paper liên quan |
| [`knowledge/ablation_inventory.md`](knowledge/ablation_inventory.md) | Kiểm kê toàn bộ run, cách lấy lại artifact |
