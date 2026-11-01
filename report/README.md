# Báo cáo đồ án: quy chuẩn viết và làm việc chung

> **Người viết chính**: chủ dự án. Các bạn trong team viết phần được giao và gửi để duyệt.
> **Chiến lược và tiến độ**: `plan/plan.md` §5. **Quy ước nhánh**: `plan/plan.md` §4. **Nguồn số liệu và lập luận**: `knowledge/` (xem §6 dưới đây).

---

## 1. Nguyên tắc nội dung

1. **Đúng trọng tâm học thuật.** Mỗi đoạn trả lời một câu hỏi: làm gì, vì sao, kết quả ra sao, nghĩa là gì. Không kể lại quá trình nếu nó không phục vụ lập luận.
2. **Không quảng cáo.** Không dùng các từ như "triệt để", "hoàn hảo", "100%", "đột phá", "cực kỳ", "chuẩn mực quốc tế". Kết quả tự nói: viết số và đơn vị, kèm cách đo.
3. **Không dài dòng.** Một ý một câu. Câu nào xóa đi mà đoạn vẫn đủ nghĩa thì xóa.
4. **Liêm chính khi kết quả chưa tốt.** Điểm yếu được nêu thẳng, kèm lập luận và bằng chứng, không né, không làm nhẹ. Ví dụ đúng tinh thần này:
   - `ref_111s` có phạt lệch tim đường `r_center`, lệch với khuyến nghị của đặc tả; nêu rõ, kèm kết quả ablation bỏ `r_center`.
   - Mốc so sánh phải được làm **mạnh nhất có thể** trước khi so (ghost hiệu chỉnh, F24), không chọn mốc yếu để thắng.
   - RL thắng mô hình vì tối ưu trực tiếp simulator; cũng vì thế nó khai thác điểm lỏng của simulator. Viết cả hai mặt.
5. **Phân biệt mức độ chắc chắn.** "Đo được" (có log hoặc eval), "suy luận" (lập luận từ số liệu), "giả thuyết" (chưa kiểm). Không viết suy luận như thể đã đo.
6. **Mọi con số có nguồn.** Ghi chú nguồn trong bản nháp bằng `<!-- nguồn: logs/anchor/eval_111s_anchor.json -->` hoặc `<!-- nguồn: F24 -->`. Không lấy số từ trí nhớ hoặc từ tài liệu đã đánh dấu "lịch sử".

## 2. Ký hiệu thống nhất

Khai báo một lần ở chương mô hình hóa; mọi chương sau dùng đúng ký hiệu này. Muốn thêm ký hiệu thì thêm vào bảng này trước.

| Ký hiệu | Ý nghĩa | Đơn vị | Tên trong code |
| :-- | :-- | :-- | :-- |
| $t$, $\Delta t$ | Bước quyết định, độ dài bước (0.1 s) | –, s | `dt` |
| $s_t$ | Tiến trình dọc tim đường (tọa độ Frenet) | m | `track_s` |
| $L$ | Chiều dài vòng | m | `circuit_length` |
| $d_t$ | Độ lệch ngang khỏi tim đường | m | `lat_offset` |
| $w$ | Nửa bề rộng mặt đường (4.5 m ở map half) | m | `half_width` |
| $v_t$ | Tốc độ xe | m/s (bảng có thể ghi km/h) | `speed_km_h` |
| $\kappa(s)$ | Độ cong tim đường hoặc racing line | 1/m | `track_kappa` |
| $a_{lat}$, $a_{brake}$, $a_{eng}$ | Giới hạn gia tốc ngang, phanh, tăng tốc của mô hình | m/s² | `envelope_a_*` |
| $\mathbf{o}_t$ | Quan sát (35 chiều) | – | `obs` |
| $\mathbf{a}_t = (\delta_t, a^{thr}_t, a^{brk}_t)$ | Action: lái, ga, phanh | – | `action` |
| $r_t$ | Reward tại bước $t$ | – | `step_reward` |
| $\gamma$, $\lambda$, $\epsilon$ | Hệ số chiết khấu, GAE, clip của PPO | – | `gamma`, `gae_lambda`, `clip_coef` |
| $\pi_\theta$, $V_\phi$ | Policy, value function | – | `actor`, `critic` |
| $\sigma$ | Độ lệch chuẩn của Gaussian trước tanh | – | `exp(actor_logstd)` |
| $T_{lap}$ | Lap time đo được | s | `lap_time` |
| $T^*$ | Lap time lý thuyết của mô hình chất điểm (tích phân profile) | s | `theoretical_best_lap_time` |
| $\beta$ | Góc trượt thân xe | độ | – |

Quy ước thêm:
- Các thành phần reward viết $r^{prog}_t$, $r^{center}_t$, $r^{steer}_t$, $r^{idle}_t$, $R^{crash}$, $R^{finish}$, với hệ số $c_{prog}$, $c_{center}$, $c_{steer}$. Ví dụ: $r^{center}_t = -c_{center}\,(d_t / w)^2$, với $c_{center} = 0.04$.
- Tên mô hình: `ref_111s` (kết quả chính), "ghost hiệu chỉnh" (`gt3_anchor`), "ghost legacy". Không dùng "bản 111s", "best agent" hay tên khác cho cùng một thứ.
- Vòng xuất phát đứng ghi là "vòng đứng", vòng thứ hai là "vòng flying".

## 3. Công thức và trích dẫn

- **Lấy công thức từ paper gốc**, đổi ký hiệu theo bảng §2 nếu cần, và trích nguồn ngay cạnh. Không tự diễn giải lại thành dạng mới khi paper đã có dạng chuẩn.
- Chỉ đưa công thức nào được dùng trong lập luận. Không chép cả trang lý thuyết PPO.
- Công thức riêng của dự án (reward, luật track limit) viết đúng như code, kèm tên file.

Nguồn đã dùng trong dự án (kiểm lại thông tin xuất bản trước khi đưa vào danh mục tài liệu tham khảo):

| Chủ đề | Nguồn |
| :-- | :-- |
| PPO | Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 1707.06347, 2017 |
| GAE | Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 |
| Gaussian nén tanh, hiệu chỉnh log-prob | Haarnoja et al., *Soft Actor-Critic*, ICML 2018 (phụ lục) |
| Reward shaping theo thế năng | Ng, Harada, Russell, ICML 1999 |
| MetaDrive | Li et al., *MetaDrive*, IEEE TPAMI 2022 |
| GT Sophy | Wurman et al., *Nature* 602, 2022 |
| RL đua xe GT Sport | Fuchs et al., IEEE RA-L 2021 (arXiv 2008.07971) |
| Racing line cực tiểu độ cong | Heilmeier et al., *Vehicle System Dynamics*, 2020 |
| Profile tốc độ tối ưu thời gian | Lipp, Boyd, *Int. J. Control*, 2014 |
| Bang-bang trong RL | Seyde et al., NeurIPS 2021 (arXiv 2111.02552) |
| Điều khiển mượt (CAPS) | Mysore et al., ICRA 2021 (arXiv 2012.06644) |
| RL dùng racing line (TAL) | Evans et al., IEEE RA-L 2023 (arXiv 2306.07003) |

Tóm tắt nội dung và cách các nguồn này liên quan tới dự án: `knowledge/related_work_racing_rl.md`.

## 4. Hình thức

- Tiếng Việt. Thuật ngữ tiếng Anh để nguyên khi chưa có từ Việt thông dụng (PPO, racing line, apex, ghost), in nghiêng ở lần xuất hiện đầu.
- Số: dấu chấm thập phân (114.8 s), ghi kèm độ lệch chuẩn và số mẫu (114.8 ± 0.4 s, n = 19).
- Bảng có tiêu đề và nguồn số liệu. Hình có chú thích, lưu ở `report/figures/`, kèm tên script sinh ra hình.
- Không dùng emoji trong báo cáo.

## 5. Làm việc nhiều người

1. **Mỗi chương một file** trong `report/sections/` (ví dụ `03_moi_truong.md`). Đầu file ghi người phụ trách và trạng thái: `nháp`, `chờ duyệt`, `đã duyệt`.
2. **Làm trên nhánh riêng** (ví dụ `report/<tên>-<chương>`) tạo từ nhánh neo `feature/111s-honest-anchor`, gửi PR để chủ dự án duyệt trước khi gộp. Không commit thẳng lên nhánh neo, và không sửa code trên nhánh báo cáo; thí nghiệm mới đi nhánh `exp/<tên>` (`plan/plan.md` §4). Không sửa chương của người khác khi chưa trao đổi; góp ý bằng comment trong PR hoặc ghi chú `<!-- góp ý (tên): ... -->`.
3. **Chỗ chưa xong** đánh dấu `[TODO: ...]` hoặc `[CẦN SỐ: ...]` để dễ tìm. Bản nộp không còn đánh dấu nào.
4. **Thêm ký hiệu hay thuật ngữ mới** thì cập nhật §2 trong cùng commit.
5. **Số liệu mới** (ví dụ kết quả ablation bỏ `r_center`) chỉ đưa vào sau khi đã ghi vào `knowledge/` hoặc `logs/`, để báo cáo và tài liệu dự án không lệch nhau.
6. Trước khi nộp, một người đọc lại toàn bộ chỉ để kiểm tra §2 (ký hiệu) và §4 (hình thức).

## 6. Nguồn nội dung cho từng chương

| Chương gợi ý | Lấy từ |
| :-- | :-- |
| Bài toán và đặc tả | `plan/requirements/PROJECT_SPECIFICATION.md`, `SPECIFICATION_INTENDED_VIOLATION.md` |
| Môi trường, xe, MDP | `README.md` §2, `knowledge/vehicle_and_engine_physics.md` §1–5, `game_env/racing_env.py` |
| Thuật toán và các lỗi PPO đã sửa | `knowledge/rl_debugging_findings.md` F1, F2, F13, F14 |
| Mốc so sánh giải tích | `knowledge/vehicle_and_engine_physics.md` §6–7 |
| Kết quả chính | `knowledge/rl_debugging_findings.md` F24, `logs/anchor/` |
| Phong cách lái và độ mượt | F20, F23; `scripts/analysis/drive_style.py` |
| Ablation và các hướng đã thử | `knowledge/ablation_inventory.md`; Phase 2: F17–F22 |
| Liên hệ nghiên cứu | `knowledge/related_work_racing_rl.md` |
| Hạn chế | `knowledge/vehicle_and_engine_physics.md` §10, F24, file violation |
