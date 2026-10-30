# Trạng thái dự án và việc còn lại

> **Cập nhật**: 31/10/2026 · **Nhánh phát triển chính**: `feature/111s-honest-anchor`
> **Sổ quyết định đầy đủ (D1–D41) và toàn bộ work item cũ**: `plan/archive/plan_v12_full.md`. Các tài liệu trong `knowledge/` nhắc tới "plan/plan.md Dxx" là nói tới file đó.
> Kế hoạch theo work item **đã đóng**. File này chỉ còn ghi trạng thái cuối và các việc kiểm thử, demo còn lại.

---

## 1. Kết quả chính (đã chốt)

**Mô hình tham chiếu `ref_111s`** (`models/ref_111s/best_agent.pt`, commit gốc `ee99f9d` trên `feature/clean-111s-optimal-ghost`). Tên lấy từ lap 111.4s lúc train. PPO, obs 35D, action 3D (lái, ga, phanh), map Silverstone half.

So sánh liêm chính (D41, F24): cùng env, cùng luật **4 bánh** (không kerb ảo), cùng vật lý xe, ghost đã hiệu chỉnh bằng envelope đo trên Bullet (chọn tốt nhất trong 33 cấu hình).

| | Vòng đứng | Vòng flying | Sạch |
| :-- | :-- | :-- | :-- |
| Ghost hiệu chỉnh (`gt3_anchor`), T\* 110.62s | 118.2s | 112.8s | 3/3 |
| `ref_111s`, 20 episode × 2 vòng | **114.8 ± 0.4s** | **109.4 ± 0.5s** | 18/20 |

Luận điểm cho báo cáo: trong cùng simulator và cùng luật, policy RL nhanh hơn bộ điều khiển dựa trên mô hình đã hiệu chỉnh khoảng 3%, và vòng flying nhanh hơn cả T\* của mọi envelope mà ghost chạy được. RL thắng vì tối ưu trực tiếp simulator thay vì mô hình đơn giản hóa của nó; cũng vì thế nó khai thác được các điểm lỏng của simulator. Viết cả hai mặt (F23, F24, `knowledge/vehicle_and_engine_physics.md` §7).

Hạn chế phải nêu: xe còn bug F16 ở cả hai bên; `ref_111s` có `r_center` và được train dưới luật kerb ảo; 1 seed; ghost là một bộ điều khiển cụ thể (MPC dọc + pure pursuit).

## 2. Phase 2 (map 1:1) – đóng băng

Các run P2 (v1, v2, line, thử r_racingline) giữ làm chương "mở rộng đã thử": `knowledge/ablation_inventory.md` R12–R14b, `knowledge/rl_debugging_findings.md` F17–F23. Không train tiếp.

## 3. Việc còn lại

| Việc | Ai | Ghi chú |
| :-- | :-- | :-- |
| Demo 1 xe | Chủ dự án | `python demo/spectate_ai.py --model models/ref_111s/best_agent.pt --map silverstone --vehicle gt3_anchor --ghost` |
| Fine-tune `ref_111s` trên env chặt (tùy chọn) | Chủ dự án | Model gốc vẫn là neo; bản fine-tune chỉ dùng nếu tốt hơn rõ |
| Lặp lại `ref_111s` với seed khác (tùy chọn) | Chủ dự án | Kiểm tra kết quả có phụ thuộc seed không |
| **Demo nhiều xe** | Team | `plan/team_handoff_multicar.md` |
| Báo cáo | Chủ dự án | Khung nội dung: `plan/archive/plan_v12_full.md` WI-17 |

## 4. Bản đồ tài liệu

| File | Nội dung |
| :-- | :-- |
| `plan/team_handoff_multicar.md` | Bàn giao demo nhiều xe cho team |
| `plan/requirements/PROJECT_SPECIFICATION.md` | Đặc tả yêu cầu gốc của đề tài, cộng trạng thái thực hiện |
| `plan/requirements/SPECIFICATION_INTENDED_VIOLATION.md` | Các điểm lệch có chủ đích so với đặc tả, kèm lý do |
| `knowledge/rl_debugging_findings.md` | Các phát hiện F1–F24 và bảng insight I1–I20 |
| `knowledge/vehicle_and_engine_physics.md` | Vật lý xe, hiệu chỉnh, benchmark, sai số mô hình |
| `knowledge/related_work_racing_rl.md` | GT Sophy, Fuchs, và paper về bang-bang, CAPS, TAL |
| `knowledge/ablation_inventory.md` | Kiểm kê mọi run đã có, cách lấy lại artifact |
| `plan/archive/` | Plan đầy đủ v12, lưu ý triển khai, các vấn đề đã đóng |
