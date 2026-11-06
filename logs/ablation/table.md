Log train cắt tại step 3082240; eval 3 episode mỗi 10.240 step. Eval cuối: best_agent.pt, 20 episode trên env ee99f9d.

| Run | Thay đổi | Eval sạch đầu tiên | Số eval sạch | Completion 0.5M cuối (%) | Tốc độ eval sạch cao nhất (km/h) | Eval cuối: sạch | Lap (s) | Tốc độ TB |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| ref_111s (log cắt tại mốc chung) | Công thức gốc | 737280 | 40/302 | 72.1 | 89.3 | 19/20 | 111.6 ± 0.4 (n=19) | 89.3 |
| no_center | Bỏ phạt lệch tim đường | 1177600 | 9/302 | 63.8 | 86.2 | 9/20 | 117.8 ± 0.2 (n=9) | 87.2 |
| no_steer | Bỏ phạt đánh lái | 993280 | 48/302 | 80.8 | 85.0 | 19/20 | 118.3 ± 0.8 (n=19) | 84.6 |
| time_penalty | Thêm phạt thời gian (ref_111s không có) | 1044480 | 15/302 | 51.9 | 86.6 | 12/20 | 116.1 ± 0.6 (n=12) | 86.4 |
| crash_flat | Phạt va chạm cố định -10, không theo tốc độ | 1218560 | 23/302 | 57.3 | 86.6 | 19/20 | 116.1 ± 0.6 (n=19) | 86.3 |
| no_finish | Bỏ toàn bộ thưởng về đích | 286720 | 93/302 | 75.1 | 91.1 | 10/20 | 109.9 ± 0.2 (n=10) | 90.1 |
| no_time_bonus | Bỏ thưởng bậc thang theo lap time | 962560 | 54/302 | 71.2 | 89.9 | 20/20 | 112.2 ± 0.2 (n=20) | 89.7 |
| no_idle | Bỏ phạt dưới 10 km/h | 993280 | 50/302 | 65.9 | 87.8 | 19/20 | 114.3 ± 0.6 (n=19) | 87.3 |
| ghost hiệu chỉnh | Mốc so sánh | – | – | – | – | 20/20 | 115.9 ± 0.0 (n=20) | 85.2 |

Lưu ý: best_agent.pt của ref_111s được xác định là bản lưu ở mốc cắt qua log tốc độ; log gốc không ghi lap time nên không loại trừ hoàn toàn một lần lưu muộn hơn theo luật lap time.
