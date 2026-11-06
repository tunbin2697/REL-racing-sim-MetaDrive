# Demo đua nhiều xe trên MetaDrive

Đua 2–10 xe trên sa bàn Silverstone bán phần, huấn luyện bằng PPO self-play chia sẻ tham số, warm-start từ mô hình đơn xe `ref_111s`. Người chơi có thể đua trực tiếp với AI bằng tay cầm Xbox hoặc bàn phím.

Báo cáo học thuật, kết quả `ref_111s` và ablation nằm ở nhánh khác (xem [Nơi lưu phần còn lại](#nơi-lưu-phần-còn-lại)).

## Cấu trúc

```text
agents/
  networks.py            Actor-Critic Gaussian nén tanh
  ppo_agent.py           PPO (GAE, nhiều epoch, chiếu logstd)
game_env/
  racing_env.py          Lớp nền: engine, hình học đường đua, obs ego 35D của ref_111s
  multicar_env.py        Đua nhiều xe: spawn, va chạm, reward, obs 60D, respawn
  vehicle.py             GT3RaceCar (vật lý của ref_111s, μ cầu trước 0.9)
  track_presets.py       Sa bàn 2 làn (9.0m) và 3 làn (13.5m)
  silverstone_map.py     Dựng sa bàn từ silverstone_blocks.json
  gamepad.py             Điều khiển bằng tay cầm
training/train_multicar.py   Huấn luyện self-play
demo/race_multicar.py        Demo 3D: người chơi hoặc khán giả
tests/test_env_regression.py Kiểm thử hồi quy env
models/
  ref_111s/best_agent.pt            Mô hình tham chiếu đơn xe (35D)
  multicar_final_1M/agent_1000k.pt  Mô hình multicar của Phát (60D), học trong env có shield nhả ga
```

## Cài đặt

```powershell
conda create -n rel_racing python=3.10 -y
conda activate rel_racing
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Chạy

```powershell
# Đua với AI bằng tay cầm hoặc bàn phím
python demo/race_multicar.py --model models/multicar_final_1M/agent_1000k.pt

# Chỉ xem AI đua
python demo/race_multicar.py --spectate --model models/multicar_final_1M/agent_1000k.pt

# Huấn luyện từ ref_111s (35D -> 60D: 25 cột đối thủ khởi tạo bằng 0)
python training/train_multicar.py --init-model models/ref_111s/best_agent.pt --preset-smooth A --grid staggered --lr 5e-5 --total-samples 2000000 --save-dir models/multicar_run --log-dir logs/multicar_run

# Huấn luyện tiếp từ checkpoint 60D
python training/train_multicar.py --resume models/multicar_final_1M/agent_1000k.pt --preset-smooth A --grid staggered --lr 5e-5 --total-samples 2000000 --save-dir models/multicar_run --log-dir logs/multicar_run
```

Phím điều khiển trong demo được ghi ở đầu file `demo/race_multicar.py`.

## Kiểm thử

```powershell
python tests/test_env_regression.py            # so với vết đã ghi
python tests/test_env_regression.py --record   # ghi lại vết sau khi CỐ Ý đổi env, reward hoặc vật lý
```

Test chạy 6 xe trong 250 bước và so khớp tuyệt đối obs, reward, vị trí, cờ kết thúc. Vật lý chỉ lặp lại được trên cùng máy và cùng bản thư viện, nên khi đổi máy cần ghi lại vết trước khi sửa code.

## Mô hình bài toán

- **Quan sát 60D:** 35D ego của `ref_111s` cộng 5 đối thủ gần nhất, mỗi xe 5 chiều (dx, dy, dvx, dvy trong hệ trục xe, cờ có xe).
- **Hành động 3D:** lái, ga, phanh trong [-1, 1], giới hạn tốc độ quay vô lăng.
- **Kết thúc:** cả 4 bánh ra khỏi mặt nhựa, lật xe, hoặc dưới 10 km/h trong 120 bước.

Reward và các quy tắc hỗ trợ hiện tại nằm ở `MultiCarRacingEnv.step()` và `_apply_racing_etiquette()`.

## Nơi lưu phần còn lại

Nhánh này chỉ giữ phần cần cho demo. Mọi thứ đã bỏ vẫn còn trong lịch sử git:

| Nội dung | Vị trí |
|---|---|
| Báo cáo, slide, tài liệu `ref_111s` | `main`, `feature/111s-honest-anchor` |
| Ablation reward | `exp/111s-reward-ablation` |
| Multicar của Phát trước các lần finetune | `feature/multicar-demo-phat` |
| Phase 1/2, ghost, benchmark, script chẩn đoán, các run finetune cũ | commit `fdfdaeb` trên nhánh này |
