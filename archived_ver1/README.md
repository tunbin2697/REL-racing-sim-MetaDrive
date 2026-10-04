# Archived Version 1: 2D F1 Autonomous Racing Simulator

Thư mục này lưu trữ toàn bộ mã nguồn, dữ liệu mô phỏng, mô hình đã huấn luyện và visualizer của phiên bản 1 (2D F1 Racing Environment).

## Cấu trúc lưu trữ
- `core/`: Mô hình động lực học xe (Kinematic Bicycle Model), môi trường 2D tự xây dựng (`racing_env.py`), tạo track Silverstone (`track_generator.py`), hồ sơ vận tốc cua (`circuit_profiler.py`), thuật toán PPO thuần (`ppo_agent.py`).
- `policies/`: Các chính sách baseline (Centerline Follower, Racing Line Apex Tracker, PPO Policy).
- `evaluation/`: Công cụ so sánh các chính sách, xuất telemetry JSON, xuất video mô phỏng HTML5 canvas.
- `data/`: Dữ liệu track Silverstone và telemetry mẫu.
- `models/`: Checkpoint mô hình PPO (`f1_racing_agent.pt`).
- `plots/`: Biểu đồ phân tích telemetry, friction circle, v-s profile và video mp4.
- `f1_racing_visualizer.html`: Web visualizer tương tác trực tiếp 2D canvas.
- `README_ver1.md`: Tài liệu hướng dẫn gốc của phiên bản 1.
