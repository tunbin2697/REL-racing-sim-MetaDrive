"""T10. Bật lần lượt các tùy chọn của planner để xem mỗi cái đóng góp gì.
    python t10_mpc_options.py <config> pack <races> <none|smart>
    python t10_mpc_options.py <config> duel
    python t10_mpc_options.py report
Lỗi "smart": chỉ phanh 50% trong 1s khi có xe bám trong 1.5s và phía trước 80m là đường thẳng, mỗi xe tối đa 2 lần mỗi vòng.
"""
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

CONFIGS = {
    "c0_base": {},
    "c1_grid": {"start_grace_steps": 100},
    "c2_grid_map": {"start_grace_steps": 100, "corner_limit": 1.5},
    "c3_attack": {"start_grace_steps": 100, "corner_limit": 1.5, "attack": True},
    "c4_attack_close": {"start_grace_steps": 100, "corner_limit": 1.5, "attack": True, "w_soft": 2.0, "soft_len_scale": 1.2},
    "c5_close_noattack": {"start_grace_steps": 100, "corner_limit": 1.5, "w_soft": 2.0, "soft_len_scale": 1.2},
    "c6_attack_nomap": {"start_grace_steps": 100, "attack": True, "w_soft": 2.0, "soft_len_scale": 1.2},
    "c7_late_attack": {"attack": True, "attack_after_steps": 200, "attack_range_m": 30.0, "attack_closing_ms": 3.0},
    "c8_gentle_start": {"start_grace_steps": 150, "start_max_mode": 2, "start_no_offset": True},
    "c10_base_map": {"corner_limit": 1.5},
    "c11_straight_attack": {"corner_limit": 1.5, "attack": True, "attack_after_steps": 200, "attack_range_m": 30.0, "attack_closing_ms": 3.0, "attack_straight_m": 60},
    "c9_gentle_late_attack": {"start_grace_steps": 150, "start_max_mode": 2, "start_no_offset": True, "attack": True, "attack_after_steps": 200, "attack_range_m": 30.0, "attack_closing_ms": 3.0},
}


from t10_lib import SmartMistakes

if sys.argv[1] == "report":
    print(f"{'config':20}{'fault':7}{'fin':>8}{'crash':>6}{'stall':>6}{'episode':>8}{'contact':>8}{'overtake':>9}{'touched':>8}{'lap':>7}{'steer':>7}{'ped':>6}{'beta':>6}{'ms':>6}{'att':>6}{'pass':>6}{'mist':>6}")
    for path in sorted(glob.glob(os.path.join(RESULTS, "t10_pack_*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        rr, a = d["races"], d["adapter"]
        outs = [o for r in rr for o in r["outs"]]
        held = [h for r in rr for h in r["held"]]
        laps = [r["mean_lap"] for r in rr if r["mean_lap"] > 0]
        print(f"{d['config']:20}{d['fault']:7}{sum(r['finished'] for r in rr):>4}/{len(rr) * 6:<3}{sum(o['reason'] == 'crash_or_track_limit' for o in outs):>6}{sum(o['reason'] == 'stalled' for o in outs):>6}"
              f"{np.mean([r['contact_episodes'] for r in rr]):>8.1f}{np.mean([r['contacts'] for r in rr]):>8.1f}{len(held) / len(rr):>9.2f}{np.mean([h['touched'] for h in held]) if held else 0:>8.2f}{np.mean(laps) if laps else 0:>7.1f}"
              f"{np.mean([r['steer_rev'] for r in rr]):>7.1f}{np.mean([r['pedal_sw'] for r in rr]):>6.1f}{np.mean([r['beta'] for r in rr]):>6.2f}{a['plan_ms_per_step']:>6.1f}{a['attacks_started'] / len(rr):>6.1f}{a['attacks_passed'] / len(rr):>6.1f}{d['mistakes_per_race']:>6.1f}")
    print("\nduel (xe trước phanh lỗi, 16 lần)")
    for path in sorted(glob.glob(os.path.join(RESULTS, "t10_duel_*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        rows = [r for r in d["rows"] if r["kind"]]
        print(f"{d['config']:20} pass {sum(bool(r['held']) for r in rows)}/16, contact {sum(r['contacts'] > 0 for r in rows)}/16, crash {sum(any(o['reason'][:5] == 'crash' for o in r['outs']) for r in rows)}, stall {sum(any(o['reason'] == 'stalled' for o in r['outs']) for r in rows)}, gaps {min(r['gap_m'] for r in rows)}-{max(r['gap_m'] for r in rows)}m")
    sys.exit(0)

from common import make_env, load_policy, run_race, Mistake, save
from mpc_adapter import MPCAdapter, MPCConfig

name, mode = sys.argv[1], sys.argv[2]
profile = dict(np.load(os.path.join(RESULTS, "ref_profile.npz")))
env = make_env(num_cars=6 if mode == "pack" else 2)
agent = load_policy(env, "ref_111s")
env.reset()
ad = MPCAdapter(env, agent, profile, MPCConfig(**CONFIGS[name]))


def fn(e, obs, active, step):
    if step == 0:
        ad.reset()
    return ad.act(e, obs, active)


def stats():
    return {"plan_ms_per_step": 1000.0 * ad.plan_time / max(1, ad.total_steps / env.num_cars), "engaged_share": ad.engaged_steps / max(1, ad.total_steps),
            "attacks_started": ad.attacks_started, "attacks_passed": ad.attacks_passed}


if mode == "pack":
    races, fault = int(sys.argv[3]), sys.argv[4]
    out, triggered = [], 0
    for seed in range(races):
        ms = [SmartMistakes(env)] if fault == "smart" else []
        out.append(run_race(env, agent, seed=seed, mistakes=ms, act_fn=fn))
        triggered += ms[0].triggered if ms else 0
    env.close()
    save(f"t10_pack_{name}_{fault}", {"config": name, "params": CONFIGS[name], "fault": fault, "races": out, "adapter": stats(), "mistakes_per_race": triggered / races})
else:
    L = env.circuit_length
    rows = []
    for k in range(8):
        at = L * (0.12 + 0.10 * k)
        info = {}

        def probe(e, obs, step):
            d = [e.race_distance(s) for s in e.racers]
            if "gap" not in info and max(d) >= at:
                info["gap"], info["leader"] = float(max(d) - min(d)), int(np.argmax(d))

        run_race(env, agent, seed=k, on_step=probe, act_fn=fn)
        for dur in (1.0, 1.5):
            r = run_race(env, agent, seed=k, mistakes=[Mistake(info.get("leader", 0), at, "brake50", dur)], stop_after_mistake=300, act_fn=fn)
            rows.append({"frac": round(at / L, 3), "kind": "brake50", "dur": dur, "gap_m": round(info.get("gap", -1.0), 1), "held": r["held"], "contacts": r["contacts"], "outs": r["outs"]})
    env.close()
    save(f"t10_duel_{name}", {"config": name, "rows": rows, "adapter": stats()})
