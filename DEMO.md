# Ghi chú chạy demo multicar

Demo **không cần train**. `ref_111s` đã có sẵn, MPC tính trực tiếp mỗi bước.

## Chạy demo

Mở terminal ở thư mục gốc repo:

```powershell
conda activate rel_racing
python tests\feature-multicar-demo-phat-bk-finetune_test_multicar_dev_plan\demo_compare.py --mode mpc_faults --laps 3
```

3 vòng mất khoảng 5–6 phút. Tất cả xe do AI lái, chế độ này không có xe người chơi.

| Chế độ | Có gì | Khi nào dùng |
|---|---|---|
| `--mode mpc_faults` | `ref_111s` + MPC + lỗi lái ở nhóm đầu | **Bản chính** |
| `--mode mpc` | `ref_111s` + MPC, không lỗi | Dự phòng nếu lỗi gây rối. Sạch nhất, gần như không có pha vượt sau xuất phát |
| `--mode ref` | Chỉ `ref_111s` | Để so sánh: xe va chạm rất nhiều |

Đổi số vòng bằng `--laps 5`. Nhiều vòng hơn thì có thêm cơ hội vượt.

## Phím

| Phím | Tác dụng |
|---|---|
| `B` | Nhìn từ trên cao |
| `Q` | Camera bám xe |
| `Tab` hoặc `N` / `M` | Xe kế tiếp / xe trước đó |
| `0` hoặc `L` | Theo xe dẫn đầu |
| Chuột | Xoay camera 360 độ, chuột giữa để về chính diện |
| `P` | Tạm dừng |
| `R` | Đua lại từ đầu |

Khi xe mắc lỗi, terminal và HUD hiện dòng kiểu `AI3 oversteer ở phút 2.4`. Lúc đó bấm `Tab` để chuyển sang xe đó.

## Nên chuẩn bị

- Chạy thử một lần trước giờ demo.
- Quay sẵn vài lượt và giữ một lượt có pha vượt đẹp làm dự phòng, vì mỗi lần chạy có thể khác nhau.

## Điều sẽ thấy khi chạy

- Xuất phát: các xe luồn lách tranh vị trí. Đây là đoạn đẹp nhất.
- Lỗi đầu tiên xuất hiện sau khoảng 60 giây, sau đoạn thẳng lớn thứ 2, và chỉ rơi vào top 3.
- Pha vượt không chắc có: trong 12 race 3 vòng, 5 race không có pha nào, số còn lại có 1–5 pha.
- Xe văng rồi hồi sinh là bình thường, khoảng 3–4 lần trong 3 vòng.

Lỗi quá dày thì đổi `interval_s=(20.0, 40.0)` thành `(40.0, 70.0)` trong `grip_faults.py`. Mức này chưa được đo, và pha vượt có thể giảm.

## Cách giải thích

> Phần RL là `ref_111s`, được train một xe, lap sạch và mượt. Cho 6 xe cùng chạy thì nó không biết né vì chưa từng thấy xe khác, nên va chạm rất nhiều. Em thử train tiếp multicar nhưng xe quên cách lái. Vì vậy em giữ nguyên model, thêm một lớp MPC xử lý tương tác: khi có xe gần, nó dự đoán 2.5 giây tới, chọn lệch line và mức phanh. Còn `ref_111s` vẫn tự lái. Các lỗi lockup, understeer, oversteer là do hệ thống chủ động tạo để cuộc đua có diễn biến, không phải model tự mắc.

## Số liệu

Đo trên vật lý hiện tại, số race ít nên chỉ đọc xu hướng.

| 6 xe | Pha vượt/race | Đợt va chạm/vòng | Xe văng/race |
|---|---|---|---|
| MPC + lỗi nhóm đầu, 3 vòng, 12 race | 1.0 (khoảng tin cậy 0.3–1.8) | 4.4 | 3.8 |
| MPC + lỗi nhóm đầu, 10 vòng, 6 race | 2.0 (khoảng tin cậy 0.5–3.7) | 3.0 | 7.2 |
| MPC không lỗi, 3 vòng, 12 race | 0 | 2.3 | 0.7 |
| `ref_111s` + lỗi lịch trước, 10 vòng, 6 race | 2.0 (khoảng tin cậy 1.0–2.8) | 19.4 | 6.2 |

- Dòng `ref_111s` dùng lịch lỗi khác, nên chỉ so sánh tương đối.
- Bản thử train tiếp 300k mẫu trước khi đổi vật lý: xe chạy một mình giảm từ 12/12 xuống 0/12. Bản này đổi nhiều tham số cùng lúc nên chưa biết tham số nào gây ra.

## Câu hỏi có thể gặp

| Câu hỏi | Trả lời |
|---|---|
| Multicar có phải RL không? | Không. RL chỉ ở phần lái, phần tương tác là planner. Đây là kết hợp RL với điều khiển dựa trên mô hình. |
| Vì sao không train multicar? | Đã thử nhiều lần nhưng chưa đạt. Các lần đầu còn dính luật hỗ trợ, lần sạch thì mất kỹ năng lái. Chưa kết luận RL không làm được, chỉ là chưa có cấu hình tốt trong thời gian này. |
| Xe có giống `ref_111s` gốc không? | Model giống hệt, nhưng xe đã chỉnh: μ cầu trước 0.95 (gốc 0.9) và lực phanh 250 (gốc 650) để bớt bốc đuôi, không train lại. Vì thế lap khác bản báo cáo gốc. |
| MPC có dùng thông tin gì đặc biệt không? | Có. Nó dùng hồ sơ tốc độ ghi từ một vòng của `ref_111s` và vị trí, tốc độ xe khác. Đây là thông tin dùng riêng cho demo, không dùng để train. |
| Xe có tự biết vượt không? | Chưa hẳn. Không có lỗi thì gần như không có pha vượt. Xe mới biết tránh và tận dụng khi xe trước mắc lỗi, chưa chủ động phanh muộn để tấn công. |

**Nên tránh nói:** "Agent tự học cách vượt", "Xe tự mắc lỗi", "Đây là multi-agent RL".

## Giới hạn đã biết

- MPC chưa nối vào chế độ tay cầm và bàn phím.
- Cấu hình này chưa được kiểm chứng ở 20 vòng, nên đừng chạy quá dài.
- Chưa xác nhận đoàn xe có co lại nhờ lỗi nhóm đầu hay không.
