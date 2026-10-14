"""
Huấn luyện PPO v2 (Physics-informed policy) - dùng lại nguyên pipeline training/train_racing.py,
chỉ thay lớp agent bằng PPOAgentV2. Mọi tham số dòng lệnh của train_racing.py giữ nguyên, thêm:
  --overspeed-a-lat  Gia tốc ngang (m/s^2) cho phạt vượt tốc của env (mặc định 0.65g như v1;
                     0.9g ~ giới hạn văng thực tế -> phạt chỉ còn chặn sát giới hạn, RL được tự tối ưu lap time)
  --bc-coef / --bc-decay / --critic-warmup-updates  Điều chỉnh regularization về Policy luật của PPOAgentV2

Cách chạy (khởi tạo từ Behavior Cloning của Policy luật):
    python training/pretrain_bc_v2.py --save-path models/v2/bc_warmup_residual.pt
    python training/train_racing_v2.py --warmup-model models/v2/bc_warmup_residual.pt --total-steps 150000 \
        --milestone-respawn --lr 0.00015 --ent-coef 0.0 --eval-freq 4 --eval-episodes 3 \
        --overspeed-a-lat 8.83 --bc-coef 1.0 --log-dir logs/v2/rl_laptime --save-dir models/v2/laptime
"""

import os
import sys
import argparse
import functools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import train_racing
from agents.ppo_agent_v2 import PPOAgentV2


def parse_v2_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--overspeed-a-lat", type=float, default=0.65 * 9.81)
    parser.add_argument("--bc-coef", type=float, default=5.0)
    parser.add_argument("--bc-decay", type=float, default=0.98)
    parser.add_argument("--critic-warmup-updates", type=int, default=5)
    v2_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return v2_args


if __name__ == "__main__":
    v2_args = parse_v2_args()
    train_racing.PPOAgent = functools.partial(
        PPOAgentV2,
        bc_coef=v2_args.bc_coef,
        bc_decay=v2_args.bc_decay,
        critic_warmup_updates=v2_args.critic_warmup_updates,
    )
    train_racing.MetaRacingEnv = functools.partial(train_racing.MetaRacingEnv, overspeed_a_lat=v2_args.overspeed_a_lat)
    print(f">> PPO v2 | overspeed_a_lat={v2_args.overspeed_a_lat:.2f} m/s^2 | bc_coef={v2_args.bc_coef} | bc_decay={v2_args.bc_decay}")
    train_racing.main()
