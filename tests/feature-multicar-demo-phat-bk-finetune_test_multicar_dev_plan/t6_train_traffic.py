"""T6. Lúc train, xe có đủ đông để gặp tình huống bám và vượt không?
Mô phỏng đúng cách thu rollout: policy ngẫu nhiên, có hồi sinh ngẫu nhiên, reset khi hết 3000 bước. Không cập nhật mạng.
    python t6_train_traffic.py <model> <steps>
"""
import sys
import numpy as np
from common import make_env, load_policy, policy_act, save

model, total = sys.argv[1], int(sys.argv[2])
env = make_env(respawn=True)
agent = load_policy(env, model)
obs, _ = env.reset()
actions = np.zeros((env.num_cars, 3), dtype=np.float32)
ahead = {10: 0, 15: 0, 30: 0, 60: 0}
car_steps = episode_step = 0
dones_total = {"crash_or_track_limit": 0, "stalled": 0, "finished": 0}
early_steps = 0
by_phase = {"first_300": [0, 0], "after_300": [0, 0]}
for t in range(total):
    a, _, _, _ = policy_act(agent, obs, False)
    actions[:] = a
    reasons_before = [s.termination_reason for s in env.racers]
    obs, _, d, tr, _ = env.step(actions)
    episode_step += 1
    phase = "first_300" if episode_step <= 300 else "after_300"
    for i, st in enumerate(env.racers):
        if d[i]:
            dones_total[st.termination_reason] = dones_total.get(st.termination_reason, 0) + 1
        car_steps += 1
        dists = [env.race_distance(o) - env.race_distance(st) for o in env.racers if o is not st]
        dists = [x % env.circuit_length for x in dists]
        nearest = min(dists)
        for r in ahead:
            ahead[r] += int(nearest < r)
        by_phase[phase][0] += int(nearest < 15)
        by_phase[phase][1] += 1
    if tr:
        obs, _ = env.reset()
        episode_step = 0
env.close()
save(f"t6_{model}", {"model": model, "steps": total, "car_steps": car_steps, "dones": dones_total,
                     "share_ahead_within_m": {str(k): round(v / car_steps, 4) for k, v in ahead.items()},
                     "share_within15_by_phase": {k: round(v[0] / max(1, v[1]), 4) for k, v in by_phase.items()},
                     "phase_car_steps": {k: v[1] for k, v in by_phase.items()}})
