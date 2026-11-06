"""T7. Chẩn đoán một checkpoint: policy kém đi vì mất kỹ năng lái hay vì tín hiệu đối thủ?
- solo: chạy một mình. Kém ở đây nghĩa là mất kỹ năng lái, không liên quan đến xe khác.
- pack: 6 xe, policy thấy đối thủ như bình thường.
- pack_blind: 6 xe, che obs đối thủ trước khi đưa vào policy. So với pack để thấy tác động của tín hiệu đối thủ.
    python t7_checkpoint_diag.py <model> <races>
"""
import sys
import numpy as np
from common import make_env, load_policy, run_race, save, BASE_OBS_DIM
import common

model, races = sys.argv[1], int(sys.argv[2])

env = make_env(num_cars=1)
agent = load_policy(env, model)
solo = [run_race(env, agent, seed=k) for k in range(races)]
env.close()

env = make_env()
agent = load_policy(env, model)
pack = [run_race(env, agent, seed=k) for k in range(races)]

real_policy_act = common.policy_act


def blind_policy_act(ag, obs, deterministic):
    masked = np.array(obs, copy=True)
    masked[:, BASE_OBS_DIM:] = 0.0
    return real_policy_act(ag, masked, deterministic)


common.policy_act = blind_policy_act
blind = [run_race(env, agent, seed=k) for k in range(races)]
env.close()
save(f"t7_{model}", {"model": model, "solo": solo, "pack": pack, "pack_blind": blind})
