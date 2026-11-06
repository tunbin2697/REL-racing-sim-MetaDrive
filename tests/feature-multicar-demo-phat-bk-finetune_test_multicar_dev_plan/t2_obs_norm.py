"""T2. ref_111s train trên đường 9m (lat_norm 4.5). Đường 3 làn chuẩn hóa theo 6.75 có làm policy lái kém đi không?
So sánh 3 cấu hình, mỗi cấu hình chạy một xe và 6 xe:
- l2: 2 làn, đúng môi trường gốc của ref_111s
- l3: 3 làn, lat_norm 6.75 như hiện tại
- l3_norm45: 3 làn nhưng chuẩn hóa obs theo 4.5
    python t2_obs_norm.py <case> <races>
"""
import sys
from common import make_env, load_policy, run_race, save

CASES = {"l2": (2, None), "l3": (3, None), "l3_norm45": (3, 4.5)}
case, races = sys.argv[1], int(sys.argv[2])
lane, lat_norm = CASES[case]

solo_env = make_env(num_cars=1, lane=lane, lat_norm=lat_norm)
agent = load_policy(solo_env, "ref_111s")
solo = [run_race(solo_env, agent, seed=k) for k in range(races)]
solo_env.close()

env = make_env(lane=lane, lat_norm=lat_norm)
agent = load_policy(env, "ref_111s")
pack = [run_race(env, agent, seed=k) for k in range(races)]
env.close()
save(f"t2_{case}", {"case": case, "solo": solo, "pack": pack})
