# Trạng thái dự án và việc còn lại

> **Cập nhật**: 31/10/2026 · **Nhánh phát triển chính**: `feature/111s-honest-anchor`
> Kế hoạch theo work item **đã đóng**. Quy ước nhánh: §4. Sổ quyết định D1–D41 và lý luận từng work item: `plan/archive/plan_v12_full.md` (các tài liệu trong `knowledge/` nhắc "plan/plan.md Dxx" là nói tới file đó).

---

## 1. Kết quả chính (đã chốt)

**Mô hình tham chiếu `ref_111s`** (`models/ref_111s/best_agent.pt`, train ở `ee99f9d` trên `feature/clean-111s-optimal-ghost`; tên lấy từ lap 111.4s lúc train). PPO, obs 35D, action 3D (lái, ga, phanh), map Silverstone half.

So sánh (D41, F24): cùng env, luật **4 bánh** (không kerb ảo), cùng vật lý xe; ghost hiệu chỉnh bằng envelope đo trên Bullet, chọn tốt nhất trong 33 cấu hình.

| | Vòng đứng | Vòng flying | Sạch |
| :-- | :-- | :-- | :-- |
| Ghost hiệu chỉnh (`gt3_anchor`), T\* 110.62s | 118.2s | 112.8s | 3/3 |
| `ref_111s`, 20 episode × 2 vòng | **114.8 ± 0.4s** | **109.4 ± 0.5s** | 18/20 |
| Cùng công thức, seed 43 (`models/ref_111s_seed43/`) | 115.2 ± 0.4s | 110.0 ± 0.7s | 19/20 |
| Fine-tune `ref_111s` trên luật 4 bánh (không dùng) | 113.6 ± 0.4s | 109.0 ± 0.2s | 8/20 |

**Luận điểm**: trong cùng simulator và cùng luật, policy RL nhanh hơn bộ điều khiển dựa trên mô hình đã hiệu chỉnh khoảng 3% ở cả hai seed, và vòng flying nhanh hơn cả T\* của mọi envelope mà ghost chạy được. RL thắng vì tối ưu trực tiếp simulator thay vì mô hình đơn giản hóa của nó; cũng vì thế nó khai thác được điểm lỏng của simulator. Viết cả hai mặt (F23, F24, `knowledge/vehicle_and_engine_physics.md` §7).

**Hạn chế**: bug F16 ở cả hai bên (công bằng nhưng xe chưa thật); `ref_111s` có `r_center` và được train dưới luật kerb ảo; 2 seed, seed 43 hội tụ chậm hơn; ghost là một bộ điều khiển cụ thể (MPC dọc + pure pursuit).

## 2. Đã đóng

- **Phase 1** (vá PPO trên map half): xong, dùng làm chương chẩn đoán (F1–F15).
- **Phase 2** (map 1:1, xe `GT3RealCar`): đóng băng, giữ làm chương "mở rộng đã thử" (`knowledge/ablation_inventory.md` R12–R14b, F17–F23).
- **Hướng phát triển, không làm trong đồ án** (chi tiết ở mục D của plan lưu trữ): SAC/QR-SAC; obs lốp và lookahead theo thời gian; late apex qua γ lớn hơn hoặc shaping theo thế năng (I20); phạt trượt lốp kiểu GT Sophy; obs apex hình học; port multicar lên map 1:1; dựng lại layout từ centerline thật; khí động học.

## 3. Việc còn lại

| Việc | Ai | Trạng thái |
| :-- | :-- | :-- |
| Ablation bỏ `r_center` (R17, 2 seed) | Chủ dự án chạy, trợ lý phân tích | ▶ Nhánh `exp/111s-no-rcenter`, worktree `../REL_111s_nocenter`; kết quả chép vào `logs/anchor/`, F24, A13 |
| Demo 1 xe | Chủ dự án | ✅ `python demo/spectate_ai.py --model models/ref_111s/best_agent.pt --map silverstone --vehicle gt3_anchor --ghost` |
| Demo nhiều xe | Team | ⏳ `plan/team_handoff_multicar.md` |
| Báo cáo | Chủ dự án (viết chính) + team | ⏳ §5 |

## 4. Quy ước nhánh

| Nhánh | Dùng cho | Ai commit |
| :-- | :-- | :-- |
| `feature/111s-honest-anchor` (**nhánh neo**) | Tài liệu, `report/`, kết quả đã chốt (`logs/anchor/`, model tham chiếu), công cụ phân tích chỉ đọc (`scripts/analysis/`) | Chỉ chủ dự án; team gửi PR |
| `exp/<tên>` | Thí nghiệm có sửa code, kèm launcher. Ví dụ `exp/111s-no-rcenter` | Người làm thí nghiệm |
| `report/<tên>-<chương>` | Viết từng chương báo cáo | Người được giao chương |
| Nhánh làm việc của team multicar | Demo nhiều xe (`plan/team_handoff_multicar.md`) | Team |
| `feature/clean-111s-optimal-ghost` (`ee99f9d`) | Nguồn gốc `ref_111s` | Chỉ đọc |

Thí nghiệm có kết quả dùng cho báo cáo: **chỉ chép log và số liệu** sang nhánh neo (`logs/anchor/`, cập nhật `knowledge/`), không gộp code thí nghiệm. Thí nghiệm không sửa code (chỉ đổi cờ hoặc seed) thì không cần nhánh; ghi lệnh tái lập vào `knowledge/ablation_inventory.md` §3.

## 5. Chiến lược viết báo cáo

**Nơi viết**: thư mục [`report/`](../report/README.md). Quy chuẩn viết, bảng ký hiệu, danh mục nguồn và cách làm nhiều người ở `report/README.md`.

**Câu chuyện chính** (một mạch, mọi chương phục vụ nó):
1. Bài toán: học lái xe đua tối ưu lap time trên MetaDrive bằng RL, so với nghiệm tối ưu tính từ mô hình.
2. Muốn PPO học được, phải sửa ba lỗi cài đặt (F1, F2, F13) và chống σ kẹt trần (F14). Đây là phần "làm cho RL chạy đúng".
3. Kết quả: `ref_111s` nhanh hơn ghost đã hiệu chỉnh khoảng 3% dưới cùng luật và vật lý, lặp lại được với seed thứ hai.
4. Vì sao: sai lệch giữa mô hình và simulator; RL tối ưu thẳng simulator (kể cả điểm lỏng của nó).
5. Liêm chính: các điểm lệch đặc tả (`r_center`, kerb ảo lúc train, bug F16) và ablation bỏ `r_center`.
6. Mở rộng đã thử: map 1:1, racing line trong obs và reward, phạt ra ngoài đường kiểu GT Sophy. Rút ra gì, vì sao dừng.
7. Hạn chế và hướng phát triển.

**Dàn chương gợi ý**:

| # | Chương | Nội dung chính | Nguồn |
| :-- | :-- | :-- | :-- |
| 1 | Giới thiệu | Bài toán, đóng góp, cấu trúc báo cáo | Đặc tả |
| 2 | Nghiên cứu liên quan | GT Sophy, Fuchs; bang-bang, CAPS; TAL | `related_work_racing_rl.md` |
| 3 | Môi trường và MDP | Map, xe, obs, action, reward, luật, điểm lệch đặc tả | README §2, file violation, vehicle physics §1–5 |
| 4 | Thuật toán | PPO, GAE, Gaussian nén tanh; các lỗi đã sửa | F1, F2, F13, F14 |
| 5 | Mốc so sánh giải tích | Racing line, profile tốc độ, ghost, hiệu chỉnh envelope | Vehicle physics §6–7 |
| 6 | Kết quả | Bảng §1, phong cách lái, tái lập với seed 43 | F23, F24, `logs/anchor/` |
| 7 | Phân tích | Ablation `r_center` (R17), fine-tune, Phase 1 A0 vs A5 | Ablation inventory, F24 |
| 8 | Mở rộng đã thử | Phase 2 | F17–F22 |
| 9 | Hạn chế, kết luận | – | Vehicle physics §10, file violation |

**Thứ tự viết**: chương 3 và 5 trước (mô tả, ít phụ thuộc kết quả), rồi 6, rồi 4 và 7 (chờ R17), 2 và 8, cuối cùng 1 và 9. Kết quả R17 về thì cập nhật chương 7 và mục liêm chính.

**Phân công gợi ý**: chủ dự án viết 1, 6, 7, 9 và duyệt toàn bộ; team nhận 2, 3, 8 sau khi xong demo nhiều xe; chương 4, 5 ai rảnh trước thì nhận.

## 6. Bản đồ tài liệu

| File | Nội dung |
| :-- | :-- |
| `README.md` | Tổng quan, cài đặt, lệnh thường dùng |
| `report/README.md` | Quy chuẩn viết báo cáo |
| `plan/team_handoff_multicar.md` | Bàn giao demo nhiều xe |
| `plan/requirements/PROJECT_SPECIFICATION.md` | Đặc tả yêu cầu gốc, cộng trạng thái thực hiện |
| `plan/requirements/SPECIFICATION_INTENDED_VIOLATION.md` | Các điểm lệch có chủ đích so với đặc tả |
| `knowledge/rl_debugging_findings.md` | Phát hiện F1–F24, insight I1–I20 |
| `knowledge/vehicle_and_engine_physics.md` | Vật lý xe, hiệu chỉnh, benchmark, sai số mô hình |
| `knowledge/related_work_racing_rl.md` | Các paper liên quan |
| `knowledge/ablation_inventory.md` | Kiểm kê mọi run và phép đo, cách lấy lại artifact |
| `scripts/analysis/` | Script đo phong cách lái, cắt cua, quét envelope ghost |
| `plan/archive/` | Plan đầy đủ v12, lưu ý triển khai, vấn đề đã đóng |
