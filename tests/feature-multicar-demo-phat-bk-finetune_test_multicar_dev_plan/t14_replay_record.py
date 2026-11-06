"""T14. Ghi lại một chặng đua để phát lại từ trên cao, gồm vị trí từng xe và các sự kiện chạm, vượt, lỗi, văng.
    python t14_replay_record.py <ref|mpc|mpc_faults> <laps> <seed>
"""
import json
import os
import sys

import numpy as np

from common import RESULTS, make_env, load_policy, run_race, save
from mpc_adapter import MPCAdapter, MPCConfig, BRAVE_CONFIG
from grip_faults import TopFieldFaults

which, laps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = make_env(max_laps=laps, max_steps=2200 * laps)
agent = load_policy(env, "ref_111s")
env.reset()
ad = MPCAdapter(env, agent, dict(np.load(os.path.join(RESULTS, "ref_profile.npz"))), MPCConfig(**BRAVE_CONFIG)) if which.startswith("mpc") else None
mist = TopFieldFaults(env, seed=seed) if which.endswith("faults") else None
frames, events = [], []
in_contact = [False] * env.num_cars
was_out = [False] * env.num_cars
mistake_seen = 0


def fn(e, obs, active, step):
    if step == 0:
        ad.reset()
    return ad.act(e, obs, active)


def on_step(e, obs, step):
    global mistake_seen
    if step % 2 == 0:
        row = []
        for st in e.racers:
            x, y = st.car.position
            row += [round(float(x), 1), round(float(y), 1), round(float(st.speed_km_h)), int(st.in_contact), int(st.active)]
        frames.append(row)
    for i, st in enumerate(e.racers):
        if st.in_contact and not in_contact[i]:
            events.append({"t": step, "type": "contact", "car": i})
        in_contact[i] = st.in_contact
        if not st.active and not st.finished:
            events.append({"t": step, "type": "out", "car": i, "reason": st.termination_reason})
            e.respawn_car(i)
    if mist is not None and mist.triggered > mistake_seen:
        mistake_seen = mist.triggered
        events.append({"t": step, "type": "mistake", "car": mist.log[-1]["car"], "kind": mist.log[-1]["kind"]})


r = run_race(env, agent, seed=seed, mistakes=[mist] if mist else [], act_fn=fn if ad else None, on_step=on_step)
for h in r["held"]:
    events.append({"t": h["t"], "type": "overtake", "car": h["a"], "other": h["b"], "corner": h["corner"], "touched": h["touched"]})
th, xy = env.track_theta, env.track_xy
normal = np.stack([-np.sin(th), np.cos(th)], axis=1)
stride = 6
payload = {
    "which": which, "laps": laps, "seed": seed, "finished": r["finished"], "steps": r["steps"], "contacts": r["contact_episodes"], "best_lap": r["best_lap"],
    "track": {"center": np.round(xy[::stride], 1).tolist(), "left": np.round((xy + normal * env.half_width)[::stride], 1).tolist(), "right": np.round((xy - normal * env.half_width)[::stride], 1).tolist()},
    "frames": frames, "events": sorted(events, key=lambda ev: ev["t"]), "standings": env.standings(),
    "n_overtakes": len(r["held"]), "n_outs": sum(ev["type"] == "out" for ev in events), "n_mistakes": mistake_seen,
}
env.close()
save(f"t14_{which}_{seed}", payload)
