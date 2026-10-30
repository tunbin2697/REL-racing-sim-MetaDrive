# Kiểm kê các run có thể dùng cho ablation

> **Tạo**: 30/10/2026, sau khi rà toàn bộ nhánh và commit (`git log --all`) và lịch sử trao đổi.
> **Mục đích**: liệt kê các run đã có kết quả, cấu hình của chúng và cách lấy lại artifact, để sau này viết phần ablation. Chưa phải bản phân tích ablation.
> **Nguyên tắc**: các run **không cùng cấu hình** (map, xe, obs, action, reward, luật track limit đều có thể khác). Mỗi so sánh phải ghi rõ những biến khác nhau. Một run vẫn dùng được nếu nó trả lời được một câu hỏi, dù không phải ablation "một biến".
> **Số liệu** lấy từ `training_metrics.json` / `evaluation_results.json` trong từng commit (đọc bằng `git show <commit>:<path>`), cộng với các phép đo của trợ lý trong phiên 29–30/10. "Vòng sạch" = eval không crash.

---

## 1. Bảng tổng hợp

| ID | Nhánh / commit | Map | Xe / vật lý | Obs / action | Step | Kết quả chính | Artifact |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| R0 | `feature/racing-circuit-and-laptime-reward` @ `d52d626` (cũng là `main` `92aa7f9`) | RacingMap có sẵn của MetaDrive, 2 làn | Xe mặc định | 35D? / **2D (lái, pedal gộp)** | 0.2M | Completion max 76%, chưa về đích; 75 km/h | `models/{best,fastest,latest,warmup}_agent.pt`, `logs/training_metrics.json` |
| R1 | `origin/v2` @ `d1a49da` | RacingMap, 2 làn | Xe mặc định, dt 0.1 s | 35D / 2D | ≤0.15M mỗi run | 20 vòng: v1 gốc 0/20 → v1 chỉnh 10/20 (115.1s) → **policy luật 20/20 (105.6s)** → **residual PPO 20/20 (105.1–105.7s)** | `logs/v2/benchmark_*.json`, `logs/v2/rl*/`, `models/v2/...` (xem commit) |
| R2 | `experiment/120kmh-braking-dynamics` @ `0d755a0` | **Silverstone half** (lần đầu), 2 làn × 4.5 m | GT3 ban đầu | 35D / 3D | 1.0M | Completion max 98.5%, 8 eval sạch, cuối run 34% | `models/best_agent.pt`, `logs/training_metrics.json` |
| **R3** | `feature/clean-111s-optimal-ghost` = `refactor/oop-gt3-vehicle` @ `ee99f9d` | Silverstone half, 2 làn × 4.5 m, **track limit theo 4 góc thân xe (kerb ảo)** | GT3RaceCar, **μ cầu trước thực tế 0.9 (bug F16)** | 35D / 3D | 4.0M | **111.4s** (89.3 km/h), 72 eval sạch. Trong env hiện tại vẫn chạy hết vòng: 115.1s xuất phát đứng | `models/ref_111s/best_agent.pt` (= `ee99f9d:models/best_agent.pt`), `ee99f9d:logs/training_metrics.json` |
| R4 | `single-agent-with-optimal-ghost-car` @ `65a7672`, `80bcd70` | như R3 | như R3 | như R3 | (không có run mới) | Ghost giải tích đầu tiên; profile cũ | `scripts/legacy_optimal/silverstone_profile_65a7672.npz` |
| R5 | `a79aee7` (nhánh phase hiện tại, trước `76e3a9e`) | Silverstone half | **ExtendedVehiclePhysics** (cản chà lốp…) | 35D / 3D, **2 vòng** | 1.0M | 7 eval sạch, lap min 120.3s | `a79aee7:logs/training_metrics.json`, `a79aee7:models/best_agent.pt` |
| R6 | `9654fdd` "add old run model" → đổi tên `legacy_35d` ở `2055469` | Silverstone half | ExtendedVehiclePhysics + MPC ghost | 35D / 3D, 2 vòng | 4.0M | 10 eval sạch, **lap min 108.6s** (nhiều khả năng là vòng flying), cuối run 38% | `models/legacy_35d/` (HEAD) |
| R7 | `feature/extended-physics-mpc-optimal` @ `2055469` | Silverstone half | ExtendedVehiclePhysics | **42D** / 3D | 4.0M | 0 eval sạch, completion max 86.6%; **σ cả 3 chiều kẹt trần** (F1) | `2055469:models/*.pt`, `2055469:logs/training_metrics.json` |
| R8 | `feature/oop-clean-physics-40d` @ `0180988` / `63feaf2`; HEAD gọi là `run_a0_legacy` | Silverstone half | **Bullet thuần** (bỏ vật lý mở rộng) | **40D** / 3D | 3.38M | 0 eval sạch, completion max 42.5% | `models/run_a0_legacy/`, `logs/run_a0_legacy/`, `models/legacy_40d_27-10/` |
| R9 | `origin/feature/multicar-selfplay` @ `7ab74a5`: `run_v5`…`run_v8b` | Silverstone half, **3 làn × 4.5 m = 13.5 m** | GT3RaceCar | 35D / 3D, đã có vá F1/F2 | ≤1M mỗi run | Best lap 135–146s, crash cao; σ cả 3 chiều sập về sàn (F8) | `7ab74a5:logs/run_v*/` (model trung gian không commit) |
| R10 | như R9: `multicar_v1`…`v3` | Silverstone half, 13.5 m | GT3RaceCar, **6 xe self-play** | 65D / 3D | 4.45M / 1.0M / 1.4M samples | Best lap **128.2s** (v1), về đích tối đa 5–6 xe/eval | `7ab74a5:logs/multicar_v*/`, `7ab74a5:models/multicar_best.pt` |
| **R11** | `feature/silverstone-bk-phase1` @ `6ec59ed` (P1-main) | Silverstone half, 2 làn, **track limit 4 bánh** | GT3RaceCar (F16) | 35D / 3D; vá F1, F2, F3, F13, F14 | 2.66M | 31 eval sạch, **best sạch 125.6s**; kẹt 46% cuối run (I9) | `models/p1_main/`, `logs/p1_main/` |
| **R12** | `e72ee80` (P2-single v1) | **Silverstone full** (×2.075, 3 làn × 4.8 m) | **GT3RealCar** (đã sửa F16) | 35D / 3D; kết thúc khi ra ngoài | 4.54M | 1 eval sạch (218s, 1.5M); lap min 157.6s (eval không sạch); pedal giật, ~150 km/h trên đường thẳng (F17) | `models/p2_single_v1/`, `logs/p2_single_v1/` |
| **R13** | `b783194` (P2-single v2) | Silverstone full | GT3RealCar | 35D / **2D pedal gộp**; **phạt ra ngoài, không kết thúc**; spawn từ lỗi (D35) | **3.32M** (dừng) | Eval sạch đầu tiên 2.15M; **best sạch 143.4s @2.56M**. Eval 10 ep @2.6M: 10/10 về đích, 4/10 sạch, 145.8 ± 0.6s. 3.0–3.3M nhanh hơn (137–142s) nhưng 0/10 sạch (F20) | `models/p2_single/` (`best_agent.pt` = 2.56M), `logs/p2_single/` |
| R14 | (chưa chạy) P2-line | Silverstone full | GT3RealCar | **47D (racing line + v\*)** / 2D; setup D35 | mục tiêu 6M | — | `models/p2_line/` |
| G-half | Ghost giải tích, map half | Silverstone half | GT3RaceCar | — | — | T\* 112.59s (profile v22); ghost chạy thật 119.5s | `scripts/silverstone_optimal_profile.npz` |
| G-full | Ghost giải tích, map full | Silverstone full | GT3RealCar | — | — | T\* 124.03s (v23); ghost 128.6s xuất phát đứng, 124.9s flying | `scripts/silverstone_full_gt3_realcar_optimal_profile.npz`; `python training/evaluate.py --agent optimal ...` |

Các nhánh không có run riêng: `feature/oop-clean-physics-40d` local `63feaf2` chỉ lưu trữ R8; `main` chỉ là R0 cộng tài liệu.

---

## 2. Các câu hỏi ablation có thể trả lời bằng run sẵn có

| # | Câu hỏi | Run so sánh | Biến khác nhau ngoài biến cần xét (phải nêu) | Mức tin cậy |
| :-- | :-- | :-- | :-- | :-- |
| A1 | Các bản vá PPO (F1 projection, F2 log_prob, F13) có tác dụng không? | R8 (A0, trước vá, 3.38M, 0 vòng sạch) vs R11 (sau vá, 2.66M, 31 eval sạch) | Obs 40D vs 35D, reward thưởng đích (η³ vs mũ), time penalty | Khá: cùng map/xe; nhiều biến nhưng chênh lệch rất lớn |
| A2 | σ kẹt trần vs sập sàn | R7 (42D, kẹt trần) – R9 (sập sàn) – R11 (projection theo chiều, D19) | Obs, map width, step | Định tính, có số σ từ checkpoint (F1, F8, F14) |
| A3 | Vật lý mở rộng (ExtendedVehiclePhysics) vs Bullet thuần | R5/R6 (mở rộng) vs R3/R11 (thuần) | Obs, số vòng, reward | Thấp; chỉ nên nêu như bối cảnh |
| A4 | Độ rộng đường 9 m vs 13.5 m | R3/R11 (2 làn) vs R9 (3 làn) | Vá PPO, step | Thấp–trung bình |
| A5 | Một xe vs self-play 6 xe | R9 (một xe, ≤1M, best 135–146s) vs R10 (6 xe, best 128.2s) | Obs 35D vs 65D, số sample | Trung bình: cùng map 13.5 m |
| A6 | RL thuần vs policy có cấu trúc (luật / residual) | R1: policy luật 105.6s vs residual PPO 105.1s vs PPO v1 115.1s | Map RacingMap, khác hẳn Silverstone | Tốt trong nội bộ R1 (20 vòng mỗi policy) |
| A7 | Pedal 2 trục vs pedal gộp, kết thúc vs phạt khi ra ngoài | R12 (v1) vs R13 (v2) | Spawn từ lỗi cũng đổi cùng lúc (3 biến gộp trong D35) | Trung bình; nêu rõ là "gói thay đổi theo bài báo" |
| A8 | Prior racing line trong obs | R13 (pure) vs R14 (line) | **Chỉ khác obs** | Cao nhất trong danh sách (1 seed) |
| A9 | RL vs tham chiếu giải tích | R3 (111.4s) vs G-half (T\* 112.59s, ghost 119.5s); R13/R14 vs G-full | Ghost là mô hình chất điểm (§7 file vật lý) | Tốt; diễn giải theo I12 |
| A10 | Đổi map/xe làm bài toán khó lên bao nhiêu | R11 (half, GT3RaceCar) vs R12 (full, GT3RealCar) với cùng vá PPO | Map ×2.075, xe nhanh gấp đôi, F16 đã sửa | Định tính |
| A11 | Luật track limit (4 góc thân xe/kerb ảo) vs 4 bánh | R3 vs R11 | Reward, thưởng đích, số vòng (F9) | Thấp; R3 vẫn chạy được trong env 4 bánh (115.1s) là một điểm dữ liệu |

---

## 3. Lưu ý khi dùng

- **Lấy artifact từ commit cũ** mà không đổi nhánh: `git show <commit>:<path> > <file>` (ví dụ `git show ee99f9d:models/best_agent.pt > tmp.pt`). Checkpoint cũ cần code của đúng commit đó để chạy lại (obs/action khác), nên khi cần eval lại thì dùng `git worktree add ../wt_<id> <commit>`.
- **Eval lại trong env hiện tại** chỉ làm được với checkpoint 35D/3D (R3, R11, R12). R3 đã được kiểm chứng chạy hết vòng.
- **Lap time trong metrics các run cũ** (`eval_best_lap_time`) có thể đến từ một eval chưa sạch hoàn toàn, hoặc là vòng flying (2 vòng/episode ở R5, R6). Khi trích, ghi rõ là lap sạch hay không.
- **R3 có bug F16** (μ cầu trước 0.9) và track limit theo thân xe; R12–R14 dùng GT3RealCar đã sửa. Không so lap time giữa hai nhóm như cùng một bài toán.
- **Sim không tất định** (I14): eval 2–5 episode chỉ là mẫu nhiễu.
- **Chuỗi R1** chạy trên RacingMap, không phải Silverstone. Chỉ dùng cho câu hỏi về cấu trúc policy (A6), không so số tuyệt đối với Silverstone.
- Các phân tích gốc giữa các nhánh (diff env/train/xe, so sánh reward, định nghĩa obs của từng nhánh) lưu ở `knowledge/raw_analysis/`. Diễn giải đã chốt ở `knowledge/rl_debugging_findings.md` (F1–F19).
