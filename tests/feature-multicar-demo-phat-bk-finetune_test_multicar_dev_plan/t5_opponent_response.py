"""T5. Policy có phản ứng với xe phía trước không?
Lấy các trạng thái của một vòng chạy một mình, chèn thêm xe ảo vào obs đối thủ và so action trung bình với lúc không có xe.
Trạng thái và vật lý giữ nguyên, nên mọi chênh lệch action chỉ đến từ thông tin đối thủ.
    python t5_opponent_response.py <model>
"""
import sys
import numpy as np
import torch
from common import make_env, load_policy, run_race, save, BASE_OBS_DIM

model = sys.argv[1]
env = make_env(num_cars=1)
agent = load_policy(env, model)
states = []


def collect(e, obs, step):
    if step % 10 == 0 and e.active_mask()[0]:
        states.append(obs[0].copy())


lap = run_race(env, agent, seed=0, on_step=collect)
rng_range = env.opponent_range
kappa_idx = 14
env.close()
S = np.array(states, dtype=np.float32)
speeds = S[:, 0] * 40.0
print("states", len(S), "lap finished", lap["finished"], flush=True)


def mean_action(x):
    with torch.no_grad():
        a, _, _, _ = agent.ac.get_action_and_value(torch.as_tensor(x, dtype=torch.float32), deterministic=True)
    a = a.numpy()
    thr, brk = 0.5 * (a[:, 1] + 1.0), 0.5 * (a[:, 2] + 1.0)
    pedal = np.where(brk > 0.05, thr * (1.0 - brk) - brk, thr)
    return a[:, 0], pedal


base_steer, base_pedal = mean_action(S)
rows = []
for dx in (6.0, 10.0, 20.0, 40.0):
    for dy in (0.0, 2.0, -2.0, 3.5):
        for closing in (0.0, 5.0, 10.0):
            X = S.copy()
            X[:, BASE_OBS_DIM:BASE_OBS_DIM + 5] = [dx / rng_range, dy / rng_range, -closing / 20.0, 0.0, 1.0]
            steer, pedal = mean_action(X)
            rows.append({"dx": dx, "dy": dy, "closing": closing,
                         "steer_mean_abs": round(float(np.mean(np.abs(steer - base_steer))), 4),
                         "steer_toward_clear": round(float(np.mean((steer - base_steer) * (-np.sign(dy) if dy else 0.0))), 4),
                         "pedal_mean": round(float(np.mean(pedal - base_pedal)), 4),
                         "pedal_p05": round(float(np.percentile(pedal - base_pedal, 5)), 4),
                         "steer_p95_abs": round(float(np.percentile(np.abs(steer - base_steer), 95)), 4)})
save(f"t5_{model}", {"model": model, "n_states": len(S), "base_steer_std": round(float(np.std(base_steer)), 4),
                     "logstd": [round(float(v), 3) for v in agent.ac.actor_logstd.detach().view(-1)], "rows": rows})
