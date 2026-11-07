# Ablation hàm thưởng

Mỗi lần chỉ đổi một thành phần thưởng/phạt. Chọn `best_agent` trong 3.082.240 step đầu, eval 20 vòng.

| Cấu hình | Thay đổi | Vòng sạch | Lap (s) | v TB (km/h) | So với gốc | Kết quả |
|:--|:--|:-:|:-:|:-:|:-:|:--|
| **Công thức gốc (3 run)** | Không đổi | **55/60** | **112.1** (111.6 – 112.5) | **89.0** | Mốc | Mốc so sánh |
| `no_finish` | Bỏ thưởng về đích | 10/20 | 109.9 | 90.1 | −2.2 s | Nhanh nhất, văng nhiều |
| `no_time_bonus` | Bỏ thưởng theo lap time | 20/20 | 112.2 | 89.7 | +0.1 s | Gần như không đổi |
| `no_idle` | Bỏ phạt dưới 10 km/h | 19/20 | 114.3 | 87.3 | +2.1 s | Chậm hơn |
| `time_penalty` | Thêm phạt thời gian | 12/20 | 116.1 | 86.4 | +4.0 s | Chậm hơn, văng nhiều hơn |
| `crash_flat` | Phạt va chạm cố định | 19/20 | 116.1 | 86.3 | +4.0 s | Chậm hơn |
| `no_center` | Bỏ phạt lệch tim đường | 9/20 | 117.8 | 87.2 | +5.7 s | Chậm hơn, văng nhiều hơn |
| `no_steer` | Bỏ phạt đánh lái | 19/20 | 118.3 | 84.6 | +6.2 s | Chậm nhất |
| Ghost hiệu chỉnh | Mốc tối ưu giải tích | 20/20 | 115.9 | 85.2 | +3.8 s | Mốc tham chiếu |

- **Công thức gốc:** 3 lần train độc lập (`ref_111s` 111.6 s, `baseline` 112.3 s, `seed43` 112.5 s) đều nhanh hơn ghost.
- **Ablation:** mỗi biến thể chỉ có 1 run. Lap chỉ tính các vòng sạch.
- **Môi trường đo:** env lúc train (`ee99f9d`), vòng kết thúc ở 98.5% quãng đường. Số này không so trực tiếp với số đo luật 4 bánh (`ref_111s` 114.8 s, ghost 118.2 s).
