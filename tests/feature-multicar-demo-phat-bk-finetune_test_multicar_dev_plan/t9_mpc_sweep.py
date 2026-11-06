"""T9. Quét trọng số planner: tăng tính quyết liệt có đem lại pha vượt mà vẫn giữ được ít va chạm không?
    python t9_mpc_sweep.py <config> <races> <none|brake50>
    python t9_mpc_sweep.py report
"""
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

CONFIGS = {
    "base": {},
    "soft2": {"w_soft": 2.0},
    "soft05": {"w_soft": 0.5},
    "soft05_prog3": {"w_soft": 0.5, "w_progress": 3.0},
    "soft2_prog3_near30": {"w_soft": 2.0, "w_progress": 3.0, "trigger_ahead_m": 30.0},
    "soft05_prog3_near30": {"w_soft": 0.5, "w_progress": 3.0, "trigger_ahead_m": 30.0},
}

if sys.argv[1] == "report":
    print(f"{'config':24}{'fault':9}{'fin':>8}{'crash':>7}{'stall':>7}{'contact':>9}{'episode':>9}{'overtake':>10}{'touched':>9}{'lap':>7}{'steer':>7}{'ped':>6}{'ms':>6}")
    for path in sorted(glob.glob(os.path.join(RESULTS, "t9_*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        rr = d["races"]
        outs = [o for r in rr for o in r["outs"]]
        held = [h for r in rr for h in r["held"]]
        laps = [r["mean_lap"] for r in rr if r["mean_lap"] > 0]
        print(f"{d['config']:24}{d['fault']:9}{sum(r['finished'] for r in rr):>4}/{len(rr) * 6:<3}"
              f"{sum(o['reason'] == 'crash_or_track_limit' for o in outs):>7}{sum(o['reason'] == 'stalled' for o in outs):>7}"
              f"{np.mean([r['contacts'] for r in rr]):>9.1f}{np.mean([r['contact_episodes'] for r in rr]):>9.1f}"
              f"{len(held) / len(rr):>10.2f}{np.mean([h['touched'] for h in held]) if held else 0:>9.2f}{np.mean(laps) if laps else 0:>7.1f}"
              f"{np.mean([r['steer_rev'] for r in rr]):>7.1f}{np.mean([r['pedal_sw'] for r in rr]):>6.1f}{d['adapter']['plan_ms_per_step']:>6.1f}")
    sys.exit(0)

from common import make_env, load_policy, run_race, Mistake, save
from mpc_adapter import MPCAdapter, MPCConfig

name, races, fault = sys.argv[1], int(sys.argv[2]), sys.argv[3]
env = make_env()
agent = load_policy(env, "ref_111s")
env.reset()
L = env.circuit_length
ad = MPCAdapter(env, agent, dict(np.load(os.path.join(RESULTS, "ref_profile.npz"))), MPCConfig(**CONFIGS[name]))


def fn(e, obs, active, step):
    if step == 0:
        ad.reset()
    return ad.act(e, obs, active)


out = []
for seed in range(races):
    rng = np.random.default_rng(5000 + seed)
    ms = [] if fault == "none" else [Mistake(car, L * rng.uniform(0.08, 0.95), "brake50", 1.0) for car in range(env.num_cars) for _ in range(3)]
    out.append(run_race(env, agent, seed=seed, mistakes=ms, act_fn=fn))
env.close()
save(f"t9_{name}_{fault}", {"config": name, "params": CONFIGS[name], "fault": fault, "races": out,
                            "adapter": {"plan_ms_per_step": 1000.0 * ad.plan_time / max(1, ad.total_steps / env.num_cars), "engaged_share": ad.engaged_steps / max(1, ad.total_steps)}})
