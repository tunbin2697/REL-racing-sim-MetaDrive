"""T16. Bật dần các thay đổi để xe dám vượt hơn, trên vật lý và lỗi bám lốp mới. Đặt vật lý bằng biến môi trường MC_BRAKE, MC_MU_F.
    python t16_brave_mpc.py <config> pack <laps> <races> <none|grip>
    python t16_brave_mpc.py <config> duel
    python t16_brave_mpc.py report
"""
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

START = {"no_pedal_below_ms": 14.0}
BRAVE = {**START, "w_brake": 8.0, "offset_rate": 2.5}
NARROW = {**BRAVE, "soft_wid_scale": 1.0, "soft_len_scale": 1.2}
CONFIGS = {
    "d0_base": {},
    "d1_start": START,
    "d2_brave": BRAVE,
    "d3_narrow": NARROW,
    "d4_soft3": {**NARROW, "w_soft": 3.0},
    "d5_soft3_map": {**NARROW, "w_soft": 3.0, "corner_limit": 1.5},
    "d6_noprog": {**NARROW, "w_soft": 1.5},
    "ref": None,
}
LAP_S = 110.0

if sys.argv[1] == "report":
    print(f"{'config':15}{'laps':>5}{'fault':>6}{'fin':>8}{'văng':>6}{'đứng':>6}{'chạm/vòng':>10}{'vượt/race':>10}{'có chạm':>8}{'lap':>7}{'steer':>7}{'pedal':>7}{'beta':>6}{'lỗi':>6}")
    for p in sorted(glob.glob(os.path.join(RESULTS, "t16_pack_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        rr = d["races"]
        outs = [o for r in rr for o in r["outs"]]
        held = [h for r in rr for h in r["held"]]
        laps = [x for r in rr for x in r["all_laps"]]
        print(f"{d['config'] + '/' + d.get('engine', '?'):15}{d['laps']:>5}{d['fault']:>6}{sum(r['finished'] for r in rr):>4}/{len(rr) * 6:<3}{sum(o['reason'][:5] == 'crash' for o in outs) / len(rr):>6.1f}{sum(o['reason'] == 'stalled' for o in outs) / len(rr):>6.1f}"
              f"{np.mean([r['contact_episodes'] for r in rr]) / d['laps']:>10.1f}{len(held) / len(rr):>10.2f}{np.mean([h['touched'] for h in held]) if held else 0:>8.2f}{np.median(laps) if laps else 0:>7.1f}"
              f"{np.mean([r['steer_rev'] for r in rr]):>7.1f}{np.mean([r['pedal_sw'] for r in rr]):>7.1f}{np.mean([r['beta'] for r in rr]):>6.2f}{d['faults_per_race']:>6.1f}")
    print("\nduel (xe trước lỗi bám lốp, 12 lần)")
    for p in sorted(glob.glob(os.path.join(RESULTS, "t16_duel_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        rows = d["rows"]
        print(f"{d['config']:15} vượt {sum(bool(r['held']) for r in rows)}/{len(rows)}, chạm {sum(r['contacts'] > 0 for r in rows)}/{len(rows)}, văng {sum(any(o['reason'][:5] == 'crash' for o in r['outs']) for r in rows)}, gap {np.min([r['gap_m'] for r in rows]):.0f}-{np.max([r['gap_m'] for r in rows]):.0f}m")
    sys.exit(0)

from common import make_env, load_policy, run_race, save
from mpc_adapter import MPCAdapter, MPCConfig
from grip_faults import GripFault, SmartGripFaults, TopFieldFaults, KINDS

name, mode = sys.argv[1], sys.argv[2]
profile = dict(np.load(os.path.join(RESULTS, os.environ.get("MC_PROFILE", "ref_profile.npz"))))
if mode == "pack":
    laps, races, fault_mode = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    env = make_env(max_laps=laps, max_steps=2200 * laps + 800)
else:
    env = make_env(num_cars=2)
agent = load_policy(env, "ref_111s")
env.reset()
ad = MPCAdapter(env, agent, profile, MPCConfig(**CONFIGS[name])) if CONFIGS[name] is not None else None


def fn(e, obs, active, step):
    if step == 0:
        ad.reset()
    return ad.act(e, obs, active)


def on_step(e, obs, step):
    for i, st in enumerate(e.racers):
        if not st.active and not st.finished:
            e.respawn_car(i)


if mode == "pack":
    out, total = [], 0
    for seed in range(races):
        ms = []
        if fault_mode == "grip":
            ms = [SmartGripFaults(env, per_lap=float(os.environ.get("MC_FAULTS_PER_LAP", "0.5")), seed=seed)]
        elif fault_mode == "top":
            ms = [TopFieldFaults(env, seed=seed)]
        r = run_race(env, agent, seed=seed, mistakes=ms, act_fn=fn if ad else None, on_step=on_step)
        r["fault_log"] = ms[0].log if ms else []
        r["all_laps"] = [float(x) for s in env.racers for x in s.lap_times]
        r.pop("follow_dy")
        r.pop("follow_closing")
        r["final_gaps_m"] = np.diff(np.sort([env.race_distance(s) for s in env.racers])).tolist()
        out.append(r)
        total += ms[0].triggered if ms else 0
    env.close()
    save(f"t16_pack_{name}_{laps}_{fault_mode}_e{os.environ.get('MC_ENGINE', 'def')}", {"engine": os.environ.get("MC_ENGINE", "def"), "config": name, "laps": laps, "fault": fault_mode, "races": out, "faults_per_race": total / races})
else:
    L = env.circuit_length
    rows = []
    for k in range(4):
        at = L * (0.15 + 0.2 * k)
        info = {}

        def probe(e, obs, step):
            d = [e.race_distance(s) for s in e.racers]
            if "gap" not in info and max(d) >= at:
                info["gap"], info["leader"] = float(max(d) - min(d)), int(np.argmax(d))

        run_race(env, agent, seed=k, on_step=probe, act_fn=fn if ad else None)
        for kind in KINDS:
            fault = GripFault(info.get("leader", 0), at, kind, "ext")
            fault.start_step = None
            r = run_race(env, agent, seed=k, mistakes=[fault], act_fn=fn if ad else None)
            rows.append({"kind": kind, "at": round(at / L, 2), "gap_m": info.get("gap", 0.0), "held": r["held"], "contacts": r["contacts"], "outs": r["outs"]})
    env.close()
    save(f"t16_duel_{name}", {"config": name, "rows": rows})
