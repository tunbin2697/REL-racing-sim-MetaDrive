# Kiểm kê các run có thể dùng cho ablation

> **Tạo**: 30/10/2026, sau khi rà toàn bộ nhánh và commit (`git log --all`) và lịch sử trao đổi. **Cập nhật 31/10 (lần 2)**: thêm R15–R17, R-F1, G-anchor, §1c (các phép đo không phải run train); R3 có kết quả dưới luật 4 bánh. **Cập nhật 31/10 (lần 1)**: thêm chuỗi thử chống bang-bang bằng vật lý (R-S1, R5 chi tiết), dòng thời gian track limit/kerb (§1b), kết quả R14 và R14b, câu hỏi A12–A15.
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
| **R3 = `ref_111s`** (kết quả chính) | `feature/clean-111s-optimal-ghost` = `refactor/oop-gt3-vehicle` @ `ee99f9d`, seed 42 | Silverstone half, 2 làn × 4.5 m, **track limit theo 4 góc thân xe (kerb ảo)** | GT3RaceCar, **μ cầu trước thực tế 0.9 (bug F16)** | 35D / 3D; có `r_center` | 4.0M | Lúc train: **111.4s** (89.3 km/h), 72 eval sạch, eval sạch đầu tiên 0.73M. **Eval dưới luật 4 bánh (D41)**: 114.8 ± 0.4 / 109.4 ± 0.5s (đứng / flying), 18/20 sạch | `models/ref_111s/best_agent.pt` (= `ee99f9d:models/best_agent.pt`), `ee99f9d:logs/training_metrics.json` |
| R4 | `single-agent-with-optimal-ghost-car` @ `65a7672`, `80bcd70` | như R3 | như R3 | như R3 | (không có run mới) | Ghost giải tích đầu tiên; profile cũ | `scripts/legacy_optimal/silverstone_profile_65a7672.npz` |
| R5 | `a79aee7` (nhánh `feature/extended-physics-mpc-optimal`, 26/10) | Silverstone half | **ExtendedVehiclePhysics** (`game_env/extended_physics.py`): (1) cản chà lốp khi đánh lái, (2) lái nhạy theo tốc độ: góc lái tối đa giảm còn 50% và tốc độ xoay còn 75% từ 25 → 110 km/h, (3) quán tính ga/phanh bậc 1 (α 0.70 / 0.85) để chống chattering 10 Hz | 35D / 3D, **2 vòng** | 1.0M | 7 eval sạch, lap min 120.3s. **Theo người dùng: vẫn bang-bang** (chưa đo lại) | `a79aee7:logs/training_metrics.json`, `a79aee7:models/best_agent.pt` |
| R6 | `9654fdd` "add old run model" → đổi tên `legacy_35d` ở `2055469` | Silverstone half | ExtendedVehiclePhysics + MPC ghost | 35D / 3D, 2 vòng | 4.0M | 10 eval sạch, **lap min 108.6s** (nhiều khả năng là vòng flying), cuối run 38% | `models/legacy_35d/` (HEAD) |
| R7 | `feature/extended-physics-mpc-optimal` @ `2055469` | Silverstone half | ExtendedVehiclePhysics | **42D** / 3D | 4.0M | 0 eval sạch, completion max 86.6%; **σ cả 3 chiều kẹt trần** (F1) | `2055469:models/*.pt`, `2055469:logs/training_metrics.json` |
| R8 | `feature/oop-clean-physics-40d` @ `0180988` / `63feaf2`; HEAD gọi là `run_a0_legacy` | Silverstone half | **Bullet thuần** (bỏ vật lý mở rộng) | **40D** / 3D | 3.38M | 0 eval sạch, completion max 42.5% | `models/run_a0_legacy/`, `logs/run_a0_legacy/`, `models/legacy_40d_27-10/` |
| R9 | `origin/feature/multicar-selfplay` @ `7ab74a5`: `run_v5`…`run_v8b` | Silverstone half, **3 làn × 4.5 m = 13.5 m** | GT3RaceCar | 35D / 3D, đã có vá F1/F2 | ≤1M mỗi run | Best lap 135–146s, crash cao; σ cả 3 chiều sập về sàn (F8) | `7ab74a5:logs/run_v*/` (model trung gian không commit) |
| R10 | như R9: `multicar_v1`…`v3` | Silverstone half, 13.5 m | GT3RaceCar, **6 xe self-play** | 65D / 3D | 4.45M / 1.0M / 1.4M samples | Best lap **128.2s** (v1), về đích tối đa 5–6 xe/eval | `7ab74a5:logs/multicar_v*/`, `7ab74a5:models/multicar_best.pt` |
| **R11** | `feature/silverstone-bk-phase1` @ `6ec59ed` (P1-main) | Silverstone half, 2 làn, **track limit 4 bánh** | GT3RaceCar (F16) | 35D / 3D; vá F1, F2, F3, F13, F14 | 2.66M | 31 eval sạch, **best sạch 125.6s**; kẹt 46% cuối run (I9) | `models/p1_main/`, `logs/p1_main/` |
| **R12** | `e72ee80` (P2-single v1) | **Silverstone full** (×2.075, 3 làn × 4.8 m) | **GT3RealCar** (đã sửa F16) | 35D / 3D; kết thúc khi ra ngoài | 4.54M | 1 eval sạch (218s, 1.5M); lap min 157.6s (eval không sạch); pedal giật, ~150 km/h trên đường thẳng (F17) | `models/p2_single_v1/`, `logs/p2_single_v1/` |
| **R13** | `b783194` (P2-single v2) | Silverstone full | GT3RealCar | 35D / **2D pedal gộp**; **phạt ra ngoài, không kết thúc**; spawn từ lỗi (D35) | **3.32M** (dừng) | Eval sạch đầu tiên 2.15M; **best sạch 143.4s @2.56M**. Eval 10 ep @2.6M: 10/10 về đích, 4/10 sạch, 145.8 ± 0.6s. 3.0–3.3M nhanh hơn (137–142s) nhưng 0/10 sạch (F20) | `models/p2_single/` (`best_agent.pt` = 2.56M), `logs/p2_single/` |
| R14 | `819d3e1` + obs line (P2-line) | Silverstone full | GT3RealCar | **47D (racing line + v\*)** / 2D; setup D35 | **2.66M** (dừng) | Eval sạch đầu tiên **1.147M** (v2: 2.15M); **best sạch 153.7s @1.74M**; sau 2M sạch hơn nhưng chậm dần (173–181s) (F21) | `models/p2_line/`, `logs/p2_line/` (chưa commit) |
| R14b | Fine-tune tạm từ R14 @1.74M, **r_racingline** (D36) | như R14 | như R14 | 47D / 2D, `--line-reward-lat 0.03 --line-reward-v 0.02` | 0.2M | Về đích 10/20 so với 0/9 ở đoạn đối chứng; 0 eval sạch 2/2; lệch line 4.38 → 3.19 m; thành phần tốc độ bão hòa (F21) | Đã xóa artifact; số liệu ở F21 |
| R-S1 | Các commit 06/10 trên `experiment/120kmh-braking-dynamics` (thời RacingMap, trước R2): `c1904fa` lọc EMA tay lái + phạt mượt, `4a1dae1` sửa thiên lệch rẽ phải trong micro reward | RacingMap | Xe mặc định | 35D? / 2D | – | Không có metrics riêng; là một phần chuỗi thử chống giật lái mà người dùng mô tả (F23) | `git show c1904fa`, `git show 4a1dae1` |
| **R15** | `ee99f9d` lặp lại, **seed 43** (worktree `../REL_111s_seed43`) | như R3 | như R3 | như R3 | 4.0M | Học chậm hơn R3: eval sạch đầu tiên 0.90M, 26 eval sạch. Best, eval dưới luật 4 bánh: 115.2 ± 0.4 / 110.0 ± 0.7s, 19/20 sạch (F24) | `models/ref_111s_seed43/`, `logs/anchor/seed43_training_metrics.json`, `logs/anchor/eval_overnight_seed43_best.json` |
| **R16** | Fine-tune R3 trên env `feature/111s-honest-anchor` (luật 4 bánh), cờ reward khớp R3 trừ thưởng về đích (hàm mũ thay bậc thang) và phạt đứng yên | Silverstone half, luật 4 bánh | `gt3_anchor` (= vật lý GT3RaceCar) | 35D / 3D | 1.0M | Best (~0.5M): 113.6 / 109.0s nhưng 8/20 sạch; last 1/20. Không dùng (F24) | `models/ref_111s_ft_strict/` (local, không commit), `logs/anchor/ft_strict_training_metrics.json`, `logs/anchor/eval_overnight_ft_*.json` |
| **R17** | `ee99f9d` **bỏ `r_center`** (sửa 1 dòng), seed 42 và 43 (worktree `../REL_111s_nocenter`) | như R3 | như R3 | như R3, `r_center = 0` | 4.0M mỗi seed | **Đang chạy (31/10)**. Đối chứng: R3, R15 | `scripts/overnight/run_111s_no_rcenter.ps1`; kết quả sẽ ở `../REL_111s_nocenter/logs/nocenter_s{42,43}/` |
| R-F1 | `origin/feature/f1-downforce` (nhánh team) | Silverstone half | Xe F1 có downforce ∝ v², lò xo cứng; và một bản GT3 | – | – | Chỉ đo phong cách lái (F20): pedal đổi chiều 19.7 / 10.9 lần/100 bước, β p95 2.7° / 2.4°, chạy 87 / 67 km/h | Nhánh team |
| G-half | Ghost giải tích, map half | Silverstone half | GT3RaceCar | — | — | T\* 112.59s (profile v22); ghost chạy thật 119.5s | `scripts/silverstone_optimal_profile.npz` |
| **G-anchor** | Ghost map half hiệu chỉnh (`gt3_anchor`, D41) | Silverstone half | GT3RaceCar + envelope 0.95 G / 15.0 / 3.09 | – | – | T\* 110.62s; ghost 118.2 / 112.8s (đứng / flying), 3/3 sạch. Chọn từ quét 33 cấu hình | `scripts/silverstone_half_gt3_racecar_anchor_optimal_profile.npz`, `logs/anchor/eval_ghost_anchor.json` |
| G-full | Ghost giải tích, map full | Silverstone full | GT3RealCar | — | — | T\* 124.03s (v23); ghost 128.6s xuất phát đứng, 124.9s flying | `scripts/silverstone_full_gt3_realcar_optimal_profile.npz`; `python training/evaluate.py --agent optimal ...` |

### 1c. Các phép đo không phải run train (cũng là bằng chứng cho báo cáo)
| Phép đo | Ở đâu | Kết luận |
| :-- | :-- | :-- |
| Hiệu chuẩn xe trên Bullet (bám ngang, phanh, tăng tốc, ổn định) | `scripts/calibrate_vehicle.py`, `logs/calibration/` | `knowledge/vehicle_and_engine_physics.md` §4 |
| Quét envelope ghost map half (33 cấu hình) | `scripts/analysis/ghost_envelope_sweep.py`; kết quả ở vehicle physics §6.5 | G-anchor |
| Phong cách lái (đổi pedal, đảo lái, β, a_lat) cho R3, R13, R-F1, ghost | `scripts/analysis/drive_style.py` | F20, F23 |
| Cắt cua qua phần ngoài đường | `scripts/analysis/cut_check.py` | F22 |
| Drive trace Phase 1 (full ga, thả trôi, phanh) | `scripts/drive_trace.py` | F15 |
| σ từng chiều đọc từ checkpoint | – | F1, F8, F14 |

### 1b. Dòng thời gian luật track limit / kerb ảo
| Commit | Ngày | Thay đổi |
| :-- | :-- | :-- |
| `9147c6b` | 06/10 | RacingMap + vật lý runoff ban đầu |
| `01bc5b1` | 06/10 | Tắt mesh va chạm tường, dùng runoff mở theo khoảng cách tới mép |
| `afae820` | 06/10 | Bỏ phạt runoff tường minh, để tốc độ và lap time tự tối ưu việc chém apex |
| `ee99f9d` (R3, 111s) | 22/10 | Kết thúc khi cả 4 **góc thân xe** ra quá 4.5 m (gờ kerb ảo; ghi chú trong code nói runoff tới 5.5 m) |
| P1 (R11) | 28–29/10 | Kết thúc khi cả **4 bánh** rời mặt nhựa (bỏ kerb ảo) |
| D35 (R13, R14) | 30/10 | Cả 4 bánh ra ngoài thì **phạt và chạy tiếp**; chỉ kết thúc khi ra quá 25 m hoặc ở ngoài quá 50 bước |

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
| A8 | Prior racing line trong obs | R13 (pure) vs R14 (line) | **Chỉ khác obs** | Cao nhất trong danh sách (1 seed). **Kết quả (F21)**: line sạch sớm hơn ~1M step, nhưng lap sạch chậm hơn (153.7s so với 143.4s) và chậm dần sau 2M |
| **A9** | **RL vs tham chiếu giải tích (kết quả chính)** | **R3, R15 vs G-anchor dưới luật 4 bánh** (F24); phụ: R13/R14 vs G-full | R3/R15 train dưới kerb ảo, eval dưới luật 4 bánh | **Tốt**: 2 seed, 20 episode mỗi model, ghost đã làm mạnh nhất |
| A16 | Fine-tune trên luật chặt có giúp không | R16 vs R3 | Thưởng về đích, phạt đứng yên khác R3 | Trung bình: nhanh hơn nhưng mất ổn định |
| A10 | Đổi map/xe làm bài toán khó lên bao nhiêu | R11 (half, GT3RaceCar) vs R12 (full, GT3RealCar) với cùng vá PPO | Map ×2.075, xe nhanh gấp đôi, F16 đã sửa | Định tính |
| A11 | Luật track limit (4 góc thân xe/kerb ảo) vs 4 bánh | R3 vs R11 | Reward, thưởng đích, số vòng (F9) | Thấp; R3 vẫn chạy được trong env 4 bánh (115.1s) là một điểm dữ liệu |
| A12 | Ràng buộc vật lý (EMA, extended physics, slew rate) vs phạt trong reward để chống bang-bang | R-S1, R5/R6 vs R3 (có `r_steer`); đo phong cách lái ở F20/F23 | Map, obs, số vòng | Định tính, dựa trên quan sát của người dùng; lập luận cơ chế ở F23. Nếu cần số: chạy `drive_style` cho `models/legacy_35d/best_agent.pt` bằng worktree `a79aee7` |
| **A13** | `r_center` (phạt lệch tim đường kiểu 111s) | **R17 (bỏ `r_center`, 2 seed) vs R3, R15 (có, 2 seed)**: ablation một biến, đang chạy. Phụ: R3 vs R11 (6 biến đổi cùng lúc, F9) | Không (R17 chỉ khác 1 dòng) | Cao khi R17 xong. Giả thuyết cần kiểm: `r_center` là biên an toàn lúc đầu học (giữ xe xa mép nên ít episode bị cắt), không quyết định cách lái cuối (R3 vẫn có Line MAE 2.1 m) |
| A14 | Racing line trong reward (r_racingline) | R14b vs đoạn 1.74–1.94M của R14 | Optimizer và seed rollout khởi động lại khi resume | Trung bình (cùng checkpoint, 200k step, eval 2 episode) |
| A15 | Cắt cua qua phần ngoài đường ở chế độ penalty | R13 vs R14, đo trực tiếp bằng `cut_check.py` (F22) | Obs | Tốt cho câu hỏi định tính (3 vòng mỗi model) |

---

## 3. Lưu ý khi dùng

- **Lấy artifact từ commit cũ** mà không đổi nhánh: `git show <commit>:<path> > <file>` (ví dụ `git show ee99f9d:models/best_agent.pt > tmp.pt`). Checkpoint cũ cần code của đúng commit đó để chạy lại (obs/action khác), nên khi cần eval lại thì dùng `git worktree add ../wt_<id> <commit>`.
- **Eval lại trong env hiện tại** chỉ làm được với checkpoint 35D/3D (R3, R11, R12). R3 đã được kiểm chứng chạy hết vòng.
- **Lap time trong metrics các run cũ** (`eval_best_lap_time`) có thể đến từ một eval chưa sạch hoàn toàn, hoặc là vòng flying (2 vòng/episode ở R5, R6). Khi trích, ghi rõ là lap sạch hay không.
- **R3 có bug F16** (μ cầu trước 0.9) và track limit theo thân xe; R12–R14 dùng GT3RealCar đã sửa. Không so lap time giữa hai nhóm như cùng một bài toán.
- **Sim không tất định** (I14): eval 2–5 episode chỉ là mẫu nhiễu.
- **Chuỗi R1** chạy trên RacingMap, không phải Silverstone. Chỉ dùng cho câu hỏi về cấu trúc policy (A6), không so số tuyệt đối với Silverstone.
- Các phân tích gốc giữa các nhánh (diff env/train/xe, so sánh reward, định nghĩa obs của từng nhánh) lưu ở `knowledge/raw_analysis/`. Diễn giải đã chốt ở `knowledge/rl_debugging_findings.md` (F1–F23).
