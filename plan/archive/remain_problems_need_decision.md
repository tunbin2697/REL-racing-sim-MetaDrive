# Vấn đề còn mở cần quyết định

> Đi kèm `plan/plan.md` v12.0. **31/10**: P2 chuyển sang mục tiêu demo (D37–D39); P9, P5, P6, R9, R10, R13, R14 đóng hoặc hết hiệu lực (ghi ngay tại mục).
> Bản gốc đi kèm v11.2. Mỗi mục ghi rõ loại:
> - **[Chốt ngay]**: có thể quyết định ngay bây giờ.
> - **[Chờ kết quả]**: cần có output trước rồi mới quyết định.
>
> Khi chưa có quyết định, trợ lý sẽ làm theo mục "Mặc định nếu không chốt".

---

## [Chốt ngay]

### P7. D19: chống σ ga/phanh kẹt trần ở Phase 2 (F14) — [ĐÃ CHỐT 30/10]
- `LOGSTD_MAX` theo từng chiều [lái 0.0, ga −0.7, phanh −0.7], áp cho cả P2-single và P2-multi.
- **P2-single: ent 0.01 cố định, lr cố định, không decay.** P2-multi giữ bộ của multicar (ent 0.001, lr 1e-4) theo D1.
  - Lý do: decay từng làm giai đoạn sau gần như không cải thiện ở một nhánh cũ.
  - ent 0.001 làm P1 kẹt ở khúc cua 46%.
- Muốn giảm entropy thì làm bằng các đoạn fine-tune khoảng 1M step do người điều khiển (D22).
- Không đưa `r_center` trở lại (D21).

### P1. "Thêm 2 bản để dễ so sánh" là 2 bản nào? [Tạm thời chưa bàn]
Người dùng viết: "hiện tại đã có thể so sánh giữa bản train với optimal profile ghost car rồi. Cần thêm 2 bản để dễ so sánh thôi."

Có hai cách hiểu:
- **(a)** 2 bản là P1-main và P2-single, tức các run đã có trong plan. Mỗi map sẽ có cặp so sánh "agent đã train so với ghost". Không cần thêm run nào.
- **(b)** 2 run bổ sung chạy trên máy team, ví dụ P1-main seed 2 và P2-single seed 2, để có phương sai giữa các seed.

**Mặc định nếu không chốt**: (a). Nếu chọn (b), máy team có thể chạy song song vào Đêm 1 và Đêm 2 mà không ảnh hưởng lịch.

### P2. Lịch D2 và D3 rất dày [làm được]
- D2 gồm WI-10 đến WI-13: preset map, hiệu chỉnh xe, obs/reward, benchmark cho hai map, ghost, smoke test.
- D3 gồm WI-15: port đua nhiều xe cùng pool, hồi sinh, HUD.
- Nếu một trong hai ngày trễ, đêm train tương ứng sẽ bị lùi lại và **không còn đêm dự phòng**.

Phương án cắt giảm nếu trễ, theo thứ tự cắt:
1. Bỏ ghost trên map lớn (R8).
2. P2-multi bỏ pool, dùng self-play thuần như nhánh multicar.
3. P2-multi warm-start từ P2-single giữa chừng (checkpoint 3M) để khởi chạy sớm hơn.
4. Bỏ P2-multi. Demo nhiều xe dùng nhánh multicar sẵn có.

**Mặc định**: cắt theo thứ tự trên, trợ lý báo trước khi cắt.

### P3. Phân lỗi va chạm — [ĐÃ CHỐT]
- **Quyết định của người dùng**: Phạt **đối xứng cả hai xe** (như nhánh multicar: −2.0 lúc chạm đầu, −0.2/bước khi dính). Không phân lỗi phức tạp để tránh rủi ro heuristic phán đoán góc va chạm sai.

### P4. Thưởng vượt xe — [ĐÃ CHỐT]
- **Quyết định của người dùng**: **Thêm thưởng vượt xe sạch có kiểm soát (+1.0)**.
- **Quy tắc an toàn**: Mức thưởng vừa phải (+1.0), nhỏ hơn nhiều so với phạt va chạm (−2.0) và phạt crash (−10). Bắt buộc thỏa mãn:
  1. Không có va chạm (`contact == False`) và xe vượt ở trên đường nhựa (`on_runoff == False`).
  2. Xe bị vượt không ở trạng thái vừa crash/off-track.
  3. Có cơ chế cooldown 50 bước giữa 2 xe để chống cày điểm lùi–tiến (oscillating overtake).

### P8. Reward lúc hồi sinh, cách xếp hạng cho thưởng vượt, kiểm tra dim — [ĐÃ CHỐT 30/10]
- Hồi sinh: giữ phạt crash phụ thuộc tốc độ (D24).
- Xếp hạng dùng `lap × L + s`, chỉ xét cặp cách nhau < 30m, miễn trừ 150 bước sau hồi sinh (D25).
- Load checkpoint lệch dim thì báo lỗi rõ ràng (D26).
- Chuẩn hóa obs map lớn theo nguyên tắc giữ dải giá trị (D23).

### P9. Lịch máy cho D2 tối → D4 sau khi mở lại ablation (D30) — [HẾT HIỆU LỰC 31/10]
> Đã đóng: ablation dừng (D37). P2-single v2 dừng ở 3.32M, P2-line dừng ở 2.66M. Lịch máy còn lại: WI-25 (P2-demo-single, máy của người dùng) và WI-26 (multicar, máy của team).

Máy chỉ chạy thoải mái 1 run train mỗi lúc (khoảng 220 FPS ở map full). Các việc cần máy:

| Việc | Thời lượng ước tính | Phụ thuộc |
| :-- | :-- | :-- |
| P2-single chạy nốt 4.04M → 6M | ~2.5h | Đang chạy |
| WI-22 P2-line 6M | ~7.5h | WI-19 qua review và smoke |
| WI-23 fine-tune spawn mistake 1M | ~1.3h | P2-single xong 6M, WI-20 qua review |
| WI-16 P2-multi | ~9h | WI-15 (chưa port) |
| WI-21 eval mở rộng (20 ep × nhiều cấu hình × 3 nhánh) | ~1h | Checkpoint 6M của hai nhánh |

Tổng lớn hơn quỹ máy còn lại trước buổi bảo vệ.

- **Đề xuất của trợ lý**:
  - Tối D2: P2-line.
  - Sáng D3: WI-23, sau đó WI-21.
  - **P2-multi hạ thành tùy chọn.** Nếu không kịp thì demo đua nhiều xe bằng nhánh `feature/multicar-selfplay` sẵn có, và ghi P2-multi vào hướng phát triển.
  - Lý do: ablation 3 nhánh trả lời một câu hỏi rõ ràng, có bằng chứng cho báo cáo. P2-multi còn phụ thuộc port WI-15 chưa làm, rủi ro cao nhất.
- **Phương án khác**: chạy song song P2-line và P2-multi. Trước khi chọn, phải đo FPS khi chạy 2 tiến trình cùng lúc. Nếu FPS mỗi run giảm dưới ~150 thì không lợi.
- **Mặc định nếu không chốt**: theo đề xuất của trợ lý.

### P5. Pool đối thủ: tỉ lệ learner/snapshot và tần suất lưu — [HẾT HIỆU LỰC 31/10: P2-multi trên map 1:1 chuyển sang hướng phát triển, D39]
- **Đề xuất**: 3 learner + 3 snapshot; lưu snapshot mỗi 250k mẫu; pool tối đa 5.
- Trước khi pool có snapshot đầu tiên, các xe snapshot dùng chính checkpoint warm-start (P2-single best).

**Mặc định**: làm theo đề xuất.

### P6. Số vòng của episode — [P2-multi: theo cấu hình sẵn có của nhánh team, D39]
- P2-single: plan để 1 vòng vì lúc đầu agent còn chạy chậm.
- P2-multi: chưa chốt.
  - 1 vòng: cho mẫu nhanh hơn.
  - 2 vòng: có tình huống vượt trong flying lap.

**Mặc định**: P2-multi chạy 2 vòng, vì warm-start từ một xe đã chạy hết được 1 vòng.

---

## [Chờ kết quả]

| ID | Câu hỏi | Output cần có | Thời điểm |
| :-- | :--- | :--- | :--- |
| ~~R1~~ | ✅ **Đã chốt: qua gate.** F1 đã được khắc phục (có vòng sạch). F13 và F14 đã được xử lý bằng 2 lần resume. Insight I1–I8 đã đủ để chốt Phase 2 | `logs/p1_main`, F13–F15 | 30/10 |
| R3 | Thông số `GT3RealCar` | Bảng hiệu chỉnh WI-11 | Chiều D2 |
| R4 | Mốc lookahead của map lớn | Quãng phanh đo từ 260 km/h | Chiều D2 |
| R5 | Có chạy P2-single ở Đêm 2 không | WI-10 đến WI-13 qua smoke test, R1 khỏe, map nhỏ tính lại ra đúng 112.6s | Tối D2 |
| R8 | Ghost trên map lớn có dùng được không | `OptimalRacingAgent` chạy hết 1 vòng | D2–D3 |
| ~~R9~~ | P2-multi warm-start từ checkpoint nào | – | ✅ D39: `multicar_best.pt` của team |
| ~~R10~~ | `--total-samples` của P2-multi | – | ✅ D39/WI-26: khoảng 1M samples mỗi bộ hệ số |
| R11 | Model nào dùng để demo | Demo bằng mắt + WI-27 theo tiêu chí WI-25 | Sau WI-25 / WI-26. Dự phòng: v2 best và `multicar_best.pt` |
| R12 | P2-single có cần can thiệp giữa chừng không (σ kẹt trần 0.5 hoặc σ lái về sàn sớm; tỉ lệ phanh thấp kéo dài như I6) | σ từng chiều + drive trace ở mốc 2M | ✅ Đã kiểm tra ở 3.5M (F17): σ pedal kẹt trần, pedal giật. Người dùng chọn **chạy tiếp không đổi** tới 6M; can thiệp bằng WI-23 (spawn), không đổi trần σ |
| ~~R13~~ | Nhánh line có thật sự dùng feature line không | – | ✅ Có, một phần (F21): lệch v\* giảm 120 → 46 km/h đầu run rồi chững; lệch line chững ở 3.6–4.5 m |
| ~~R14~~ | Spawn mistake có làm hết crash lặp lại không | – | ➡ Hết hiệu lực: spawn mistake dùng từ đầu (D35), WI-23 bỏ |
| R15 | Hệ số phạt mượt D38 có làm xe rụt rè không | WI-25 bước 1 (200k): lap, vòng sạch, chỉ số WI-27 | Ngay sau WI-24 |

---

## Rủi ro kỹ thuật chưa kiểm chứng (sẽ biết khi làm)

- **Ổn định của Bullet ở 72 m/s**: nếu xe rung hoặc bay, phải hạ trần tốc độ hoặc chỉnh lại hệ treo (WI-11).
- **MetaDrive có áp `mass` sau khi tạo chassis không**: chưa xác minh.
- **Nguồn của giới hạn HUD 120 km/h**: không tìm thấy trong code của dự án. Nhiều khả năng là dashboard của MetaDrive.
- **Bậc $a_{lat}$ trong benchmark là hệ số khớp tay theo từng loại cua của map nhỏ**: khi sang map lớn với `GT3RealCar` có thể phải khớp lại.
  - Nếu khớp lại theo số đo hiệu chỉnh: hợp lệ.
  - Nếu khớp lại theo lap time của agent RL: **không** hợp lệ.
- **Các số mục tiêu của GT3** (0–100, 1.3–1.5g cơ khí) là ước lượng của trợ lý. Cần đối chiếu nguồn trước khi trích dẫn trong báo cáo.
- **Thưởng vượt +1.0 chỉ cộng cho xe đi vượt, xe bị vượt không bị trừ** (P4, người dùng chốt). Hai xe về lý thuyết có thể luân phiên vượt nhau để "cày" điểm. Cooldown 50 bước cho mỗi cặp đã giới hạn tốc độ cày: tối đa +1 mỗi 5s. Muốn nhường để bị vượt thì phải giảm tốc, tức mất reward tiến độ và chịu thêm time penalty, nên nhiều khả năng không có lợi. **Cần theo dõi** số lần vượt trên mỗi vòng ở P2-multi: nếu cao bất thường (ví dụ trên 20 lần/vòng giữa cùng một cặp xe) thì xem xét trừ −1.0 cho xe bị vượt.
- **σ lái ở P2-multi**: dùng ent 0.001 cố định có thể làm σ lái về sàn (I4). Ở giai đoạn fine-tune thì chấp nhận được.
- **Run P1-main sau 2 lần resume**: bộ đếm step lệch so với trọng số (lần 1 lệch khoảng 20k, lần 2 lệch khoảng 450k). `resume_marks` đã ghi lại; cần nêu rõ trong báo cáo.
