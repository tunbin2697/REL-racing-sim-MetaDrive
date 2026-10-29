# KẾ HOẠCH THEO WORK ITEM

> **Vị trí**: `scratch/plan.md` · **Phiên bản**: v11.1 (30/10/2026, D2). Phase 1 đã dừng train ở 2.66M; Phase 2 sẵn sàng code.
> **v11.1 so với v11.0**: thêm chuẩn hóa obs đầy đủ theo preset (WI-10/12), sửa `evaluate.py` + `spectate_ai.py` (WI-12), kiểm tra dim khi load (WI-12), chốt danh sách cờ của `train_multicar.py` (WI-15/16), chốt reward lúc hồi sinh và cách xếp hạng cho thưởng vượt (WI-L5), làm rõ phạm vi ent 0.01 (D19).
> **Vấn đề còn mở**: `scratch/remain_problems_need_decision.md`
> **Phân tích / bằng chứng**: `knowledge/rl_debugging_findings.md` (F1–F15, bảng insight I1–I8)
> **Phân công**: một agent khác code các WI-10 → WI-16. Trợ lý review từng WI trước khi người dùng commit và chạy train.
> **Quy ước cho agent code**:
> - Tuân thủ `AGENTS.md`: không tự commit, không tự chạy train, không đưa optimal profile vào obs/reward.
> - **Mặc định của map nhỏ phải giữ nguyên hành vi Phase 1.** Mọi thay đổi chỉ có hiệu lực khi chọn `silverstone_full` / `gt3_real`.

## BỐI CẢNH ĐÃ CHỐT
- **Nội dung chính khi bảo vệ là bản 111s** (`ee99f9d`).
- **Phase 1** (map nhỏ, vá PPO): **xong về mục tiêu**. Người dùng **đã dừng P1-main ở 2,662,400 step**. Lúc dừng, eval kẹt ở 46% (crash tại 2624–2626m, đúng I9), Line MAE 2.44m, Pace Eff 93.1%. Best sạch là 125.6s, **không vượt** bản 111s (111.4s). Điều này đã được chấp nhận vì 111s vẫn là kết quả chính.
- **Phase 2**: Silverstone 1:1 + xe GT3 thật, gồm:
  - **P2-single**: một xe;
  - **P2-multi**: 6 xe, pool đối thủ, grid so le, phạt va chạm đối xứng, thưởng vượt sạch, hồi sinh khi vượt track limit;
  - không áp luật FIA, không dùng PID.
- **Ablation**: tạm hoãn. Giữ cờ `--no-logstd-projection`.

---

## A. SỔ QUYẾT ĐỊNH

| ID | Quyết định | Giá trị chốt |
| :-- | :--- | :--- |
| D1 | Siêu tham số | P2-single dùng bộ của 111s, **riêng entropy theo D19**. P2-multi dùng bộ của multicar (WI-L1) |
| D2 | `LOGSTD_MIN` / `LOGSTD_MAX` | −2.3 / 0.0 trên map nhỏ. Trên map lớn, `LOGSTD_MAX` lấy **theo từng chiều** (D19) |
| D3 | Time penalty | 0.01/bước cho cả hai map |
| D4 | Thưởng đích map nhỏ | $50 + \min(4e^{(180-T)/25}, 150)$ |
| D5 | Smoke test 1–2 bước | Được phép |
| D8/D9 | Xe Phase 2 | Class mới `GT3RealCar`; giữ nguyên `GT3RaceCar` |
| D11 | Model chính khi bảo vệ | 111s |
| D12 | Hướng Phase 2 | Map 1:1 + GT3 thật |
| D13 | Cách chọn map | `--map {silverstone, silverstone_half, silverstone_full}`, trong đó `silverstone` = `silverstone_half`. Mọi hằng số phụ thuộc map nằm trong `game_env/track_presets.py` |
| D14 | Thưởng đích map lớn | **v2 (30/10): $T_0 = 186$s, $\tau = 26$s**, $A = 4$, $R_{max} = 150$. Quy tắc cũ 172/24 bỏ vì giả định lap time tỉ lệ $v_{max}$, trong khi tốc độ vào cua bị chặn bởi trần bám. Quy tắc mới chỉ dùng tim đường + trần bám đo được + $v_{max}$ (không dùng profile); chi tiết ở `knowledge/vehicle_and_engine_physics.md` §8.3 (lần áp đầu ra 219/30 do dùng trần bám sai của bug F16) |
| D15 | Reward tiến độ map lớn | 0.0241/m (WI-L3) |
| D16 | PID | Không làm |
| D17 | Ablation | Tạm hoãn |
| D18 | Cơ chế đua nhiều xe | WI-L5. Người dùng đã chốt: **phạt va chạm đối xứng** và **thưởng vượt sạch +1.0** |
| **D19** | **Chống σ ga/phanh kẹt trần (F14)** | **`LOGSTD_MAX` theo từng chiều [lái 0.0, ga −0.7, phanh −0.7]**, tức σ ga/phanh tối đa 0.5. Trần này áp cho **cả P2-single và P2-multi** (gắn với preset `silverstone_full`); map nhỏ vẫn [0, 0, 0]. **Entropy**: P2-single dùng **ent 0.01 cố định, lr cố định, KHÔNG decay**. P2-multi giữ bộ của multicar (ent 0.001, lr 1e-4) theo D1. Người dùng chốt 30/10 |
| D23 | Chuẩn hóa obs ở map lớn | Mọi hằng chuẩn hóa obs đọc từ preset. Nguyên tắc: **giữ khoảng giá trị số của mỗi chiều giống map nhỏ** (cùng tư duy đồng dạng đã dùng cho `progress_coef`, T0/τ), để không chiều nào bị clip ở 250 km/h hay ở mốc lookahead 200m. Chi tiết ở WI-10 |
| D24 | Reward lúc hồi sinh (P2-multi) | **Giữ phạt crash phụ thuộc tốc độ** $-10-20(v/v_{max})^2$, y như bản một xe, cộng thêm việc bị lùi về sau. Người dùng chốt |
| D25 | Xếp hạng cho thưởng vượt | Dùng tiến độ cộng dồn `lap × L + s`. Xe vừa hồi sinh được miễn khỏi danh sách "bị vượt" trong 150 bước. Chi tiết ở WI-L5 |
| D27 | Thông số `GT3RealCar` (R3) | mass 1100 (Bullet cố định), engine 2400, **ma sát trước/sau 1.55/1.80** (μ bằng nhau thì xoay khi trail-braking), **phanh 160 N** (≥ 200 N chổng đuôi), trần 260. Bám 1.53–1.60 G, phanh 1.65 G. Lookahead giữ nguyên (R4). Xem `knowledge/vehicle_and_engine_physics.md`, F16 |
| D28 | Envelope benchmark map full | a_lat 13.8 (90% của 1.56 G), a_brake 15.5, a_engine 8.70 m/s², elip Kamm chặt; lưu trong `VehicleConfig`, chỉ cho benchmark/ghost. $T^*_{full}$ = **124.03s** (profile v23). Map half giữ v22 = 112.59s (đã kiểm tra hồi quy) |
| D29 | Bug ma sát cầu trước (F16) | `attach_agent` đọc bánh từ `system.getWheel`. `GT3RaceCar` đặt tường minh `front_wheel_friction = 0.90` để giữ vật lý Phase 1 / 111s (đã kiểm chứng 115.10s, ghost 119.5s) |
| D26 | Kiểm tra dim khi load | Mọi chỗ load checkpoint đều so `in_dim` của lớp đầu với `obs_dim` của env; lệch thì báo lỗi rõ ràng, không để lỗi shape của torch |
| D20 | Số episode eval của P2 | 5 (`--eval-episodes 5`), theo I7 |
| D21 | Không đưa `r_center` trở lại | Người dùng chốt: trong đua nhiều xe, `r_center` kéo mọi xe về tim đường nên các xe xếp hàng một, mâu thuẫn với việc vượt nhau |
| D22 | Lịch siêu tham số cho Phase 2 | Thay decay bằng **fine-tune theo đoạn do người điều khiển**: run nền (Đêm 2) → đọc kết quả → các đoạn khoảng 1M step từ best, **mỗi đoạn chỉ đổi 1 tham số**, ghi lại trong `resume_marks`. Cách này học theo cách nhánh multicar ra được racing line tốt |

---

## B. LÝ LUẬN DÙNG CHUNG

### WI-L1 · Siêu tham số (D1)
| Tham số | 111s `ee99f9d` | `0180988` | Multicar `train_racing.py` | Multicar `train_multicar.py` |
| :-- | :-- | :-- | :-- | :-- |
| lr | 3e-4 | 3e-4 | 3e-4 | **1e-4** |
| γ | 0.99 | 0.99 | 0.99 | **0.995** |
| λ / clip | 0.90 / 0.15 | như trên | như trên | như trên |
| ent | 0.01 | 0.010 | 0.010 | **0.001** |
| batch / rollout | 64 / 2048 | như trên | như trên | **128 / 512 × N** |

- Bộ siêu tham số một xe giống hệt nhau ở cả 3 nhánh. Bộ của multicar là bộ cho **fine-tune** sau khi đã warm-start.
- **Bài học từ Phase 1 (F14, I3–I4, I9)**:
  - ent 0.01: σ ga/phanh kẹt trần 1.0.
  - ent 0.001: σ lái sập về sàn trong khoảng 150k step, và **agent kẹt ở cùng một khúc cua (46%) chỉ sau khoảng 300k step**.
  - Decay lr/ent ở một nhánh cũ: giai đoạn sau gần như không cải thiện (kinh nghiệm của người dùng).
- **Áp dụng cho P2-single**:
  - Giữ **ent 0.01 cố định** (σ lái luôn khỏe ở mức này).
  - Chặn σ ga/phanh bằng trần theo từng chiều (D19). Với trần 0.5, việc σ ga/phanh nằm ở 0.5 là hành vi được thiết kế, không phải lỗi.
  - Không decay. Nếu cần giảm entropy thì làm bằng các đoạn fine-tune (D22).

### WI-L2 · PPO và reward
- **F1** (projection logstd), **F2/F13** (log_prob tính tại $u$ đã sample, lưu `raw_us` vào buffer), **F3** (bias phanh −1.6): đã nằm trong code (commit `958c11f`). **Không được đổi.**
- **D19, lý do chọn −0.7**:
  - σ_max = 0.5 vẫn đủ để khám phá vùng phanh: với mean ở ngưỡng deadband, ±1σ quét được brk khoảng 0–0.3.
  - Không cho phép nhiễu σ = 1 làm phanh trong rollout thành ngẫu nhiên.
  - Lái giữ trần 0.0, vì σ lái chưa bao giờ kẹt trần (luôn ở khoảng 0.18–0.25 khi ent = 0.01).
- **D3, D4**: xem F6 và F7. Không thêm delta time (F11); không bật sector bonus (ablation hoãn).

### WI-L3 · Hằng số reward cho map lớn (D14, D15)
- **Nguyên tắc**: hằng số chỉ suy ra từ hình học đường và thông số xe, được chốt **trước khi** tính $T^*_{full}$, và **không** chỉnh lại theo $T^*$ (zero leakage).
- **D14**:
  - Quy tắc $T_0 = L/(0.475\,v_{max})$ khớp với map nhỏ: 2838.7 / (0.475 × 33.3) = 180s.
  - Map lớn: 5891 / (0.475 × 72.2) ≈ **172s**; $\tau = 25 \times 172/180 \approx 24$s.
- **D15**: $0.05 \times 2838.7/5891$ = 0.0241/m, giữ tổng reward tiến độ mỗi vòng khoảng 142. Time penalty vẫn tính theo bước.

### WI-L4 · Cách optimal profile được tính hiện nay
Nguồn: `game_env/benchmark.py` (commit `0180988`), T* = 112.6s, `profile_version = 22`, cache `scripts/silverstone_optimal_profile.npz`.

1. **Racing line (minimum curvature tuyến tính hóa, theo Heilmeier et al. 2020)**:
   - Lấy mẫu khoảng 3m.
   - Hàm mục tiêu: $\min_\alpha \|D_2(x_0+n_x\alpha)\|^2 + \|D_2(y_0+n_y\alpha)\|^2 + w\|\alpha\|^2$, với $\sqrt w = 5\cdot10^{-5}$.
   - Ràng buộc: $|\alpha| \le$ min(4.05, W/2 − 0.45).
   - Giải bằng `lsq_linear` (TRF), sau đó periodic spline → Gaussian σ = 1.8, tính lại κ → Gaussian σ = 2.5.
2. **Speed profile (3-pass Lipp & Boyd 2014, có elip ma sát Kamm)**:
   - **Pass 1**: $v = \sqrt{a_{lat}(\kappa)/\kappa_{env}}$, với $a_{lat}$ = 8.85 / 9.10 / 10.20 / 11.80 m/s² tương ứng các ngưỡng κ = 0.017 / 0.009 / 0.0065 và ngưỡng dκ = 0.00055 / 0.00035.
   - Bao độ cong: cửa sổ −4/+22 mẫu, hệ số 0.88; ở chicane cộng thêm 17.5·dκ (nhận diện khi κ đổi dấu vượt ±0.004).
   - **Pass 2 (phanh)**: $a_{brake} = 6.20$. **Pass 3 (tăng tốc)**: $a_{engine} = 3.40$. Bám còn lại $r = \sqrt{\max(0.15, 1-\min(0.92,(a_{lat}/a_{lim})^2))}$. Mỗi pass lặp 4 lần.
   - Cuối cùng Gaussian σ = 2, clip [12 m/s, v_max].
3. **T\*** = $\sum \Delta s/v$.
- $a_{engine}$ và $a_{brake}$ là số **đo trên Bullet cho xe hiện tại** (khoảng 0.35g / 0.63g), không phải số của spec report.
- **Cần tổng quát hóa cho map lớn**:
  - ngưỡng κ ÷ scale; ngưỡng dκ ÷ scale²;
  - cửa sổ tính theo mét;
  - $\sqrt w$ ÷ scale²;
  - `max_lat_bound` = W/2 − 0.45;
  - gia tốc lấy từ `VehicleConfig`;
  - **cache đặt tên theo map + xe** (hiện tại hai map ghi đè lẫn nhau).

### WI-L5 · Cơ chế đua nhiều xe (D18)

| Thành phần | Chốt cho P2-multi | Lý do |
| :--- | :--- | :--- |
| Policy | Pool: 3 xe learner chạy policy hiện tại + 3 xe chạy snapshot đóng băng, bốc ngẫu nhiên mỗi episode; lưu snapshot mỗi 250k mẫu, pool tối đa 5. Chưa có snapshot thì dùng checkpoint warm-start | Chống dao động và tránh policy chỉ giỏi khi đấu với chính nó |
| Dữ liệu train | Chỉ transition của xe learner | Dữ liệu của xe snapshot là off-policy với PPO |
| Grid | `staggered` kiểu F1: mỗi vị trí lùi 8m, lệch ±3.5m quanh tim đường | Xếp ngang hàng dễ va nhau ngay khi xuất phát |
| Bất tử khi xuất phát | Không dùng | Hết thời gian bất tử thì các xe đang chồng lên nhau sẽ va ngay |
| Va chạm xe–xe | **Đối xứng**: cả hai xe chịu −2.0 lúc bắt đầu chạm, −0.2/bước khi còn dính; không terminal | Người dùng chốt |
| Thưởng vượt | **+1.0** khi xe A vượt xe B theo thứ hạng (xem cách tính bên dưới) và đồng thời: không va chạm, A không ở runoff, B không trong cửa sổ miễn trừ; cooldown 50 bước cho mỗi cặp (A, B) | Người dùng chốt (P4, D25) |
| Vượt track limit / lật | Hồi sinh tại chỗ, **chỉ lùi về sau**: chiếu vị trí lên tim đường, lùi từng 20m (tối đa 300m) đến khi phía trước ≥ 50m và phía sau ≥ 80m không có xe; vận tốc 15 m/s theo hướng tim đường | Xe đứng yên ở điểm ngẫu nhiên là chướng ngại cho xe tốc độ cao; chỉ lùi nên không lợi dụng crash để "đi tắt" được |
| Reward lúc hồi sinh | Bước gây hồi sinh nhận **phạt crash** $-10-20(v/v_{max})^2$ (với $v_{max}$ = `max_speed_km_h` của xe), giống bản một xe | D24. Nếu không phạt thì xe có thể "cắt cua" ra ngoài rồi hồi sinh |

**Cách tính thứ hạng cho thưởng vượt (D25)**:
- Mỗi xe có tiến độ cộng dồn `P = lap_count × L + s`, trong đó `s` là tọa độ Frenet trên tim đường. **Không dùng `s` thô**, vì qua vạch đích `s` nhảy từ L về 0 và sinh ra cú "vượt" giả.
- `lap_count` chỉ tăng khi xe thực sự cắt vạch đích theo chiều tiến (dùng logic đếm vòng sẵn có của env). Khi hồi sinh, `P` tính lại theo vị trí mới (nhỏ hơn).
- Mỗi bước, với mỗi cặp (A, B): nếu bước trước `P_A < P_B` và bước này `P_A > P_B` thì A vừa vượt B.
- Chỉ xét các cặp có `|P_A − P_B| < 30m` ở bước trước. Cặp ở xa nhau mà đổi thứ tự chỉ có thể do hồi sinh, không phải vượt thật.
- **Cửa sổ miễn trừ**: trong 150 bước (khoảng 15s) sau khi B hồi sinh, việc vượt B không được thưởng. Lý do: B bị đặt lùi ra sau và chạy chậm (15 m/s), nên các xe phía sau sẽ đi qua nó "miễn phí".
- Không thưởng nếu A đang `contact` với bất kỳ xe nào hoặc đang ở runoff tại bước vượt.
- Log `overtakes_per_lap` cho từng xe để theo dõi rủi ro cày điểm (xem file vấn đề còn mở).
| GAE khi hồi sinh | `done = True` rồi bắt đầu trajectory mới; vòng đang chạy bị hủy (không nhận bonus đích) | Không trộn GAE qua lần hồi sinh |
| Obs | 35D ego + 5 đối thủ × 5 đặc trưng = 60D; `opponent_range` lấy theo preset | Ego giữ nguyên để warm-start |
| Warm-start | `warm_start_from_single(base_dim=35)`: các cột đối thủ khởi tạo bằng 0 | Ở bước đầu, policy giống hệt bản một xe |
| Eval | Đua đầy đủ, không hồi sinh, có bảng xếp hạng | |
| Số xe / số vòng | 6 xe / 2 vòng | |

### WI-L6 · Kết quả Phase 1 (tóm tắt, chi tiết ở F13–F15)
- **F1 được xác nhận**: run P1-main có vòng sạch đầu tiên ở khoảng 1.35M. Run cũ chạy 3.38M không có vòng sạch nào.
- **F13** làm run nổ NaN ở 1.38M; đã sửa và resume từ `best_agent_1372k_preNaN.pt` (resume lần 1).
- **F14** làm σ ga/phanh kẹt trần. Đã resume lần 2 ở 2.31M với ent = 0.001 và reset σ ga/phanh = 0.37.
- **F15**: agent chậm vì thả trôi (48% thời gian), không phải vì lạng lách.
- **Dừng ở 2.66M**: sau resume lần 2, eval kẹt cố định ở 46% (2624m) từ khoảng 2.35M (I9).

---

## C. TRẠNG THÁI WORK ITEM

| WI | Tên | Trạng thái |
| :-- | :--- | :--- |
| 00–04 | Bảo toàn bằng chứng, vá PPO, reward, train script, smoke test | ✅ Xong (commit `e0738e8`, F13 fix `958c11f`) |
| 05 | P1-main | ⏹ Đã dừng ở 2,662,400 step (kẹt 46%) |
| P1C | Đóng Phase 1: eval, drive trace, biểu đồ | ⏳ Làm được ngay |
| 10–13 | Map preset, `GT3RealCar`, obs/reward theo preset, benchmark hai map + ghost, D19 | ✅ Xong. R8 đạt: ghost map full 128.6s / flying 124.9s (T* 124.03s) |
| 14 | P2-single | ▶ Sẵn sàng bấm (smoke 4096 step đã chạy thông) |
| 15 | Port multicar | ⏳ D3, agent code làm → trợ lý review |
| 16 | P2-multi | ⏳ Đêm 3 |
| 17 | Báo cáo | Làm liên tục |
| 18 | Eval cuối, demo, đóng băng | D4 |

---

### WI-P1C · Đóng Phase 1 (P1-main đã dừng ở 2.66M)
- [ ] Eval `models/p1_main/best_agent.pt` với 5 episode bằng `training/evaluate.py`. Ghi lại lap1, lap2, best, crash và các chỉ số benchmark so với ghost 112.6s.
- [ ] Drive trace (theo `scratch/drive_trace_reference.py`) cho best cuối và cho `scratch/b111.pt` (checkpoint của bản 111s), ra bảng F15 bản cuối.
- [ ] Vẽ 3 biểu đồ vào `plots/p1/`:
  1. σ lái/ga/phanh theo step, có vạch dọc tại các mốc `resume_marks` (1.39M ứng với F13, 2.31M ứng với F14);
  2. tỉ lệ eval sạch theo cửa sổ 100k step kèm best lap sạch;
  3. P1-main so với run cũ (`logs/run_a0_legacy/training_metrics.json`): completion và crash.
- [ ] Người dùng commit `logs/p1_main/`, `plots/p1/` và các checkpoint cần giữ.
- **Lý luận**: đây là toàn bộ bằng chứng cho chương "chẩn đoán và cải tiến". Phase 1 không cần train thêm.

### WI-10 · Preset map + rà hằng số hardcode
- [ ] Tạo `game_env/track_presets.py`: dataclass `TrackPreset` và dict `TRACK_PRESETS` gồm hai preset, cùng hàm `get_track_preset(name)` (trong đó `"silverstone"` → half).

  | Trường | `silverstone_half` | `silverstone_full` |
  | :--- | :--- | :--- |
  | `length_scale` (nhân `len` và `r` của block) | 1.0 | 2.075 |
  | `lane_num` × `lane_width` | 2 × 4.5 | 3 × 4.8 |
  | `lookahead_m` | [10, 20, 35, 50, 70] | [15, 40, 80, 130, 200] (có thể chỉnh sau R4) |
  | `lookahead_dx_norm` | 70 | 200 (= max lookahead) |
  | `lookahead_dy_norm` | 30 | 80 (≈ 0.4 × max lookahead, cùng tỉ lệ 30/70 của map nhỏ) |
  | `kappa_norm` (nhân κ, dùng cho cả κ tức thời và κ waypoint) | 50 | 104 (= 50 × 2.075) |
  | `v_norm` (obs vận tốc, dọc) | 40 | 80 |
  | `vy_norm` (vận tốc trượt ngang) | 20 | 40 (= v_norm / 2) |
  | `lat_norm` (lệch ngang, `dist_left/right`) | 4.5 | `half_width` = 7.2 |
  | `progress_coef` | 0.05 | 0.0241 |
  | `finish_T0`, `finish_tau` | 180, 25 | 172, 24 |
  | `time_scale` (horizon, cap step khi eval) | 1.0 | 1.45 |
  | `logstd_max` (D19) | [0, 0, 0] | [0, −0.7, −0.7] |
  | `opponent_range` | giá trị của nhánh multicar | × 2.075 |
- **Lý do chọn các hằng chuẩn hóa (D23)**:
  - Giữ dải giá trị của từng chiều obs giống map nhỏ. Ở map nhỏ, mọi chiều nằm gọn trong khoảng clip; map lớn phải giữ được điều đó.
  - **dx**: code hiện tại là `clip(dx/70, −1, 1.5)`. Nếu giữ 70 thì mốc 130m (1.86) và 200m (2.86) **luôn bị clip ở 1.5**, nghĩa là agent mất hai mốc nhìn xa nhất, vốn là lý do phải tăng lookahead (quãng phanh từ 260 km/h khoảng 180m). Dùng 200 thì mốc xa nhất bằng 1.0, như mốc 70m của map nhỏ.
  - **κ**: mọi bán kính nhân 2.075 nên κ chia 2.075. Giữ ×50 thì κ của cua nhanh co về gần 0, agent khó phân biệt cua. Nhân 104 thì một khúc cua cho cùng giá trị feature ở hai map.
  - **v**: 260 km/h = 72.2 m/s; chia 80 cho 0.90, dưới clip 1.5. Giữ 40 thì bị clip ở 216 km/h (F12).
  - **vy, lat**: tỉ lệ theo `v_norm` và bề rộng đường. Với yaw rate, giữ chia π: tốc độ quay thân xe ở cua lớn không đổi nhiều (v và R cùng tăng).
  - Với `silverstone_half`, mọi giá trị bằng đúng hằng số cũ, nên obs map nhỏ không đổi (đây là output bắt buộc bên dưới).
- [ ] `silverstone_map.py`: nhân `len` và `r` với `length_scale`. Port `lane_num`/`lane_width` → `center_lat`/`half_width` từ nhánh multicar (`git show origin/feature/multicar-selfplay:game_env/racing_env.py`).
- [ ] Thay các hằng số hardcode:

  | Hằng số | Vị trí (`racing_env.py` trừ khi ghi khác) | Thay bằng |
  | :--- | :--- | :--- |
  | `2.25`, `4.5` | dòng 230–246, 435; obs `lat/4.5`, `dist_left/right` | `center_lat`, `half_width` |
  | Chuẩn hóa obs `/40` (speed, vx), `/20` (vy), `κ×50` (2 chỗ), `dx/70`, `dy/30`, mảng lookahead | `_get_obs`, khoảng dòng 336–396 | `v_norm`, `vy_norm`, `kappa_norm`, `lookahead_dx_norm`, `lookahead_dy_norm`, `lookahead_m` |
  | Vạch đích `x = 60.0`, dải y | 246, 435, 517–534 | `half_width` |
  | `horizon` | 111 | × `time_scale` |
  | Phạt crash `/120.0` | 652 | `max_speed_km_h` của xe |
  | `obs[0]*40*3.6`, `v_c = 120`, `118.5/121.5`, `85` | `optimal_agent.py` 145, 222, 241, 333 | `v_norm`, `max_speed_km_h` |
- **Output cần nộp để review**:
  - `circuit_length` của map full ≈ 5,891m;
  - ảnh top-down so sánh hai map;
  - **obs của `silverstone_half` khớp từng giá trị với code trước khi sửa** (cùng seed, chạy 50 bước).
- **Lý luận**:
  - Nhân đều `len` và `r` là phép đồng dạng, vòng đua vẫn khép kín.
  - Gom mọi hằng số phụ thuộc map vào một chỗ (D13).
  - 14.4m khớp bề rộng thật 12–15m của Silverstone.

### WI-11 · Hiệu chỉnh `GT3RealCar`
- [ ] Viết `scripts/calibrate_vehicle.py` (mô phỏng ngắn, không train; người dùng chạy) gồm các test:
  - kiểm tra MetaDrive có áp `mass` sau khi tạo chassis không;
  - đạp hết ga: đo 0→100, 0→200 km/h và tốc độ trần trên đoạn thẳng khoảng 770m;
  - ổn định ở 230–260 km/h (đi thẳng và đổi làn nhẹ);
  - phanh hết từ 260 và từ 200 km/h: giảm tốc và quãng phanh;
  - chạy vòng tròn R = 30 / 80 / 130m: $a_y$ cực đại;
  - quét các tham số `max_engine_force`, `max_brake_force`, `wheel_friction`.
- [ ] Thêm class `GT3RealCar` vào `vehicle.py` và lựa chọn `--vehicle gt3_real`.
- **Output**: `knowledge/vehicle_and_engine_physics.md` (bảng tham số → số đo, kèm số đo của xe hiện tại để đối chiếu).
- **Quyết định**: R3, R4.
- **Lý luận**:
  - Mục tiêu tốc độ: 253–265 km/h trên Hangar Straight.
  - Mục tiêu phanh và bám ngang: khoảng 1.3–1.5g **cơ khí**, vì MetaDrive không có khí động học.
  - Lực tính bằng N phải được đo ra gia tốc, không suy luận trực tiếp. Hiện xe chỉ đạt 0.35g / 0.63g, nên có thể phải tăng lực khoảng 2–2.5 lần.

### WI-12 · Obs/reward theo preset + D19 + công cụ theo dõi
- [ ] Lookahead, chuẩn hóa `dx`/`dy`, `v/v_norm`, `progress_coef`, hằng số thưởng đích, `time_scale`: tất cả đọc từ preset.
- [ ] **D19**: `ActorCritic` nhận thêm `logstd_max: Sequence[float]` (mặc định [0, 0, 0]).
  - Lưu dưới dạng tensor **không persistent** (`register_buffer(..., persistent=False)`) để checkpoint cũ vẫn load được.
  - Clamp bằng `torch.minimum/maximum` theo tensor, áp ở **cả ba chỗ**: `actor_forward`, `get_action_and_value`, và projection trong `ppo_agent.py` (sau `step()` và trong `load()`).
  - `train_racing.py` truyền `logstd_max` từ preset vào agent.
- [ ] `scripts/drive_trace.py`: port từ `scratch/drive_trace_reference.py`, thêm `--models`, `--map`, `--vehicle`, `--laps`. Lấy lệch ngang và κ **từ env** (đừng tính ngược từ obs, vì cách chuẩn hóa obs đổi theo preset).
- [ ] **`training/evaluate.py`**:
  - Thêm `--map {silverstone, silverstone_half, silverstone_full}` (mặc định `silverstone`) và mở `--vehicle` thành `{gt3, gt3_real, sedan}`.
  - Thêm `--laps` và `--episodes`, nếu chưa có.
  - Đọc preset để lấy cap step (`max_eval_steps_per_lap × time_scale`) và file optimal profile tương ứng với map + xe (WI-13). Không có file profile thì bỏ qua phần benchmark, kèm cảnh báo, không crash.
- [ ] **`demo/spectate_ai.py`**: cùng các thay đổi `--map`/`--vehicle` như trên; ghost car (nếu bật) đọc profile theo map + xe. HUD tốc độ lấy `max_speed_km_h` của xe (xem WI-15, mục HUD).
- [ ] **Kiểm tra dim khi load (D26)**: viết một hàm dùng chung, ví dụ `check_obs_dim(state_dict, obs_dim, path)` trong `agents/ppo_agent.py`, gọi ở `PPOAgent.load()` (và mọi chỗ load trực tiếp `state_dict` trong `evaluate.py`, `spectate_ai.py`, `train_racing.py --resume/--warmup-model`, `train_multicar.py --init-model/--resume`).
  - Đọc `in_dim` từ shape của trọng số lớp Linear đầu tiên của actor.
  - Nếu khác `obs_dim` của env, `raise ValueError` với nội dung kiểu: `Checkpoint 'models/best_agent.pt' có in_dim=40 nhưng env (map=silverstone_full) có obs_dim=35. Có thể đây là model legacy 40D.`
  - Riêng warm-start multicar (35 → 60D) đi qua `warm_start_from_single(base_dim=35)`, nên kiểm tra `in_dim == base_dim` thay vì `obs_dim`.
- **Output**:
  - smoke test in ra obs ở 250 km/h trên preset full (không chiều nào bị clip, kể cả dx của mốc 200m);
  - logstd sau projection đúng bằng [≤0, ≤−0.7, ≤−0.7] với preset full;
  - load `models/best_agent.pt` (40D legacy) vào env 35D thì in đúng thông báo D26;
  - `evaluate.py --map silverstone --model scratch/b111.pt` chạy được 1 episode.
- **Lý luận**:
  - Nếu giữ `v/40` thì obs bị clip ở 216 km/h (F12).
  - Quãng phanh từ 260 km/h khoảng 180m, dài hơn nhiều so với lookahead 70m.
  - Obs vẫn 35D, nhưng ý nghĩa các chiều đã đổi, nên P2 phải train từ đầu.
  - D19: xem WI-L2.

### WI-13 · Tính lại optimal profile cho cả hai map + ghost + smoke P2
- [ ] Tổng quát hóa `benchmark.py` theo WI-L4. Cache đặt tên theo map + xe.
- [ ] **Kiểm tra hồi quy trên map nhỏ**: `GT3RaceCar` phải ra lại **112.6s**. Nếu lệch, sửa trước khi đi tiếp.
- [ ] **Map lớn**: `GT3RealCar`, gia tốc lấy từ WI-11 → $T^*_{full}$. Đối chiếu tốc độ vào cua với spec report (The Loop 81, Copse 183, Becketts 139 km/h).
- [ ] Ghost `OptimalRacingAgent` chạy 1 vòng trên map lớn → R8.
- [ ] Smoke test P2, bắt buộc gồm TEST 4D (F13) và kiểm tra D19.
- **Output**: `scripts/silverstone_optimal_profile_half_gt3.npz`, `..._full_gt3real.npz`, bảng T*, log ghost.
- **Lý luận**:
  - Kiểm tra hồi quy chứng minh việc tổng quát hóa không làm hỏng cách tính 112.6s đã được chấp nhận.
  - Profile chỉ dùng để đối chiếu và cho ghost car (zero leakage).

### WI-14 · P2-single (Đêm 2, khoảng 9h)
```powershell
python training/train_racing.py --total-steps 6000000 `
  --gamma 0.99 --gae-lambda 0.90 --clip-coef 0.15 --ent-coef 0.010 `
  --time-penalty 0.01 --finish-bonus exp --eval-episodes 5 `
  --map silverstone_full --vehicle gt3_real --max-laps 1 --seed 42 `
  --save-dir models/p2_single --log-dir logs/p2_single
```
- **Theo dõi** (dựa trên bài học P1):
  - σ từng chiều: σ ga/phanh nằm ở trần 0.5 là chấp nhận được; σ lái không được về sàn.
  - Có crash lặp lại ở cùng một vị trí qua nhiều lần eval hay không (dấu hiệu kẹt cực trị cục bộ như P1 ở 46%).
  - Không có `[NaN GUARD]` xuất hiện thường xuyên.
  - Chạy drive trace ở các mốc 2M và 4M: % đạp hết ga, % thả trôi, % có phanh.
- **Lý luận**:
  - Một vòng map lớn khoảng 1600 bước (F12). P1 cần khoảng 2.5M step để xuống khoảng 126s, nên 6M là vừa đủ, không dư (I8).
  - `--max-laps 1` vì giai đoạn đầu agent chạy chậm.
  - Best của run này là điểm warm-start cho P2-multi.
- **Fine-tune theo đoạn (D22, sáng D3, tùy chọn)**:
  ```powershell
  Copy-Item models/p2_single/best_agent.pt models/p2_single/ft1_src.pt
  python training/train_racing.py --resume models/p2_single/ft1_src.pt --steps-to-run 1000000 `
    --ent-coef 0.003 ... (giữ các cờ còn lại như run nền) --save-dir models/p2_single --log-dir logs/p2_single
  ```
  - Mỗi đoạn chỉ đổi 1 tham số so với đoạn trước.
  - Luôn copy checkpoint nguồn ra một tên riêng trước khi resume.
  - **Bắt buộc có `--save-dir`/`--log-dir`**, nếu không sẽ nạp nhầm metrics như sự cố ngày 29/10.

### WI-15 · Port multicar sang map lớn
- [ ] Port `multicar_env.py`, `train_multicar.py`, `demo/race_multicar.py`, `game_env/gamepad.py` từ `origin/feature/multicar-selfplay`, chuyển sang 35D ego cùng preset map. Kế thừa toàn bộ bản vá F1/F13/D19 (multicar dùng chung `PPOAgent`; buffer phải lưu `raw_us`).
- [ ] Triển khai đúng bảng WI-L5 (gồm D24, D25).
- [ ] **Cờ của `train_multicar.py`**: giữ ở mức tối thiểu. Các cờ chỉ để lệnh chạy hiện rõ cấu hình; **giá trị mặc định trong code chính là giá trị đã chốt**, nên chạy không cờ cũng ra đúng cấu hình P2-multi.
  - Giữ các cờ sẵn có của nhánh multicar: `--total-samples`, `--rollout-steps`, `--num-cars`, `--batch-size`, `--lr`, `--gamma`, `--gae-lambda`, `--clip-coef`, `--ent-coef`, `--time-penalty`, `--collision-penalty`, `--init-model`, `--resume`, `--eval-freq`, `--milestone-freq`, `--save-dir`, `--log-dir`, `--seed`.
  - Sửa: `--grid` mặc định thành `staggered`; bỏ `--lane-num` (lấy từ preset); `--init-model` mặc định `models/p2_single/best_agent.pt`, sửa help thành "checkpoint đơn tác tử 35D"; `--milestone-freq` dùng luôn làm chu kỳ lưu snapshot vào pool (250000).
  - Thêm đúng 6 cờ mới:

    | Cờ | Mặc định |
    | :-- | :-- |
    | `--map` | `silverstone_full` |
    | `--vehicle` | `gt3_real` |
    | `--max-laps` | 2 |
    | `--finish-bonus` | `exp` |
    | `--num-learners` | 3 (số xe còn lại chạy snapshot) |
    | `--pool-size` | 5 |
  - Không thêm cờ cho cooldown 50 bước, cửa sổ miễn trừ 150 bước, thưởng vượt +1.0, khoảng cách grid, tham số hồi sinh: để là hằng số có tên rõ ràng ở đầu `multicar_env.py`.
- [ ] **Sửa HUD**:
  - Gỡ giới hạn hiển thị 120 km/h. Nguồn của giới hạn này chưa xác định; nhiều khả năng là dashboard của MetaDrive đọc `max_speed_km_h` lúc tạo xe.
  - Sửa luôn các hardcode cùng loại trong HUD của demo nhiều xe.
- [ ] Smoke test:
  - 6 xe, không va nhau ở bước 0;
  - hồi sinh không bao giờ đặt xe lên **trước** vị trí cũ;
  - thưởng vượt không kích hoạt khi đang va chạm, đang trong cooldown, hoặc khi B đang trong cửa sổ miễn trừ 150 bước;
  - xe cắt vạch đích **không** sinh cú vượt giả (thứ hạng dùng `lap × L + s`);
  - bước hồi sinh nhận đúng phạt crash theo tốc độ (D24);
  - obs 60D; warm-start khớp shape.
- **Output**: `logs/smoke_multi.txt`, ảnh grid xuất phát.

### WI-16 · P2-multi (Đêm 3, khoảng 9h)
```powershell
python training/train_multicar.py --total-samples <R10> `
  --num-cars 6 --num-learners 3 --pool-size 5 --milestone-freq 250000 `
  --map silverstone_full --vehicle gt3_real --max-laps 2 --grid staggered `
  --lr 1e-4 --gamma 0.995 --gae-lambda 0.90 --clip-coef 0.15 --ent-coef 0.001 `
  --batch-size 128 --rollout-steps 512 --time-penalty 0.01 --collision-penalty 2.0 `
  --finish-bonus exp --init-model models/p2_single/best_agent.pt `
  --save-dir models/p2_multi --log-dir logs/p2_multi --seed 42
```
- Mọi cờ trong lệnh trên đều bằng giá trị mặc định (trừ `--total-samples`), chỉ ghi ra để lệnh tự mô tả cấu hình.
- `--total-samples` chốt theo FPS đo được lúc smoke test, sao cho vừa khoảng 9h (R10).
- **Lý luận**:
  - Bộ của multicar (ent 0.001, lr 1e-4) đã cho ra racing line tốt khi fine-tune trên đường rộng 13.5m. Map 1:1 rộng 14.4m, tương đương.
  - Rủi ro: σ lái về sàn và kẹt cực trị cục bộ (I4, như P1 ở 46%). Ở map 1:1 có 6 xe nên trải nghiệm đa dạng hơn, có thể giảm bớt rủi ro này.
  - Theo dõi crash lặp lại cùng vị trí. Nếu xảy ra, chạy đoạn tiếp theo với ent 0.003 (D22).
  - Trần σ theo từng chiều (D19) vẫn giữ.

### WI-17 · Báo cáo (làm liên tục)
- [ ] **Nội dung chính**: bản 111s.
- [ ] **Chẩn đoán và cải tiến**: F1, F2, F3, F6, F11, **F13, F14, F15**, cùng 3 biểu đồ của WI-P1C.
- [ ] **Silverstone 1:1**: F12, WI-L4, T* của hai map, bảng hiệu chỉnh GT3, kết quả P2-single và P2-multi. Giới hạn: không có khí động học.

### WI-18 · Eval cuối, demo, đóng băng (D4)
- [ ] Demo:
  - 111s cùng ghost;
  - P2-single trên map 1:1 cùng ghost (nếu R8 đạt);
  - P2-multi 6 xe (nếu chưa đạt thì dùng nhánh multicar).
- [ ] Người dùng commit bản cuối.

---

## D. NGOÀI PHẠM VI (HƯỚNG PHÁT TRIỂN)
- Ablation: A0 so với A1, η³ so với hàm mũ, sector bonus, các phương án D19 (a/b/c).
- Luật hành vi FIA; delta time so với best lap của chính agent (F11); MAPPO.
