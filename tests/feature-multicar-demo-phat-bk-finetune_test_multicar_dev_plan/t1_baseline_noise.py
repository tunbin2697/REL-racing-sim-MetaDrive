"""T1. Mốc sau khi bỏ luật cũ và mức nhiễu của phép đo.
Chạy 24 cuộc đua 6 xe trên env sạch, không train. Chạy thêm 12 cuộc đua với policy lấy mẫu ngẫu nhiên để xem nhiễu tự nhiên có tạo ra pha vượt không.
    python t1_baseline_noise.py <model> <races>
"""
import sys
from common import make_env, load_policy, run_race, save

model, races = sys.argv[1], int(sys.argv[2])
env = make_env()
agent = load_policy(env, model)
det = [run_race(env, agent, seed=k, deterministic=True) for k in range(races)]
sto = [run_race(env, agent, seed=k, deterministic=False) for k in range(races // 2)]
env.close()
save(f"t1_{model}", {"model": model, "deterministic": det, "stochastic": sto})
