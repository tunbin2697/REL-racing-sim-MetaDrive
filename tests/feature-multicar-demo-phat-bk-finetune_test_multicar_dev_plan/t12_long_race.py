"""T12. Đua dài như một chặng đua thật: xe có bị rơi rụng dần, đoàn xe có giãn ra, và tay đua lên xuống bao nhiêu hạng?
Xe văng được hồi sinh tại chỗ như trong demo. Pha vượt qua xe vừa hồi sinh không được tính.
    python t12_long_race.py <ref|mpc|mpc_smart|ref_smart> <laps> <seed>
    python t12_long_race.py report
"""
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

if sys.argv[1] == "report":
    groups = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "t12_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        # Race bị kẹt vòng lặp hồi sinh (T13) không đại diện cho cách đua, tách riêng
        key = "stuck" if len(d["outs"]) > 200 else "ok"
        if key == "stuck":
            print(f"BỎ race kẹt hồi sinh: {d['which']} seed {d['seed']} ({len(d['outs'])} lần)")
            continue
        groups.setdefault((d["which"], d["laps"]), []).append(d)
    for (which, laps), runs in sorted(groups.items()):
        ov = [len(r["held"]) for r in runs]
        crash = [sum(o["reason"][:5] == "crash" for o in r["outs"]) for r in runs]
        stall = [sum(o["reason"] == "stalled" for o in r["outs"]) for r in runs]
        print(f"\n{which}, {laps} vòng, {len(runs)} race (thời gian mô phỏng TB {np.mean([r['steps'] for r in runs]) * 0.1 / 60:.1f} phút)")
        print(f"  pha vượt/race {np.mean(ov):.1f} ({np.mean(ov) / laps:.2f}/vòng), từng race {ov}; có chạm {np.mean([h['touched'] for r in runs for h in r['held']] or [0]):.0%}")
        print(f"  văng/race {np.mean(crash):.1f}, đứng yên/race {np.mean(stall):.1f}, đợt va chạm/vòng {np.mean([r['contact_episodes'] for r in runs]) / laps:.1f}, lỗi tạo ra/race {np.mean([r['mistakes'] for r in runs]):.1f}")
        print(f"  về đủ {laps} vòng: {sum(r['finished'] for r in runs)}/{len(runs) * 6}")
        print(f"  đổi hạng so với lưới: TB tuyệt đối {np.mean([abs(g - f) for r in runs for g, f in zip(r['grid_pos'], r['final_pos'])]):.2f}, lên nhiều nhất {max(g - f for r in runs for g, f in zip(r['grid_pos'], r['final_pos']))}")
        print(f"  lap trung vị {np.median([x for r in runs for x in r['all_laps']]):.1f}s; pedal_sw {np.mean([r['pedal_sw'] for r in runs]):.1f}, steer_rev {np.mean([r['steer_rev'] for r in runs]):.1f}, β {np.mean([r['beta'] for r in runs]):.2f}")
        bins = max(1, laps // 4)
        for b in range(0, laps, bins):
            ot = [sum(b <= h["leader_lap"] < b + bins for h in r["held"]) for r in runs]
            inc = [sum(b <= o["leader_lap"] < b + bins for o in r["outs"]) for r in runs]
            gaps = [g for r in runs for k, g in enumerate(r["gap_median_s"]) if b <= k < b + bins]
            print(f"    vòng {b + 1:>2}-{b + bins:<2}: vượt/race {np.mean(ot):.1f}, sự cố/race {np.mean(inc):.1f}, khoảng cách xe liền kề trung vị {np.mean(gaps) if gaps else 0:.1f}s")
    sys.exit(0)

from common import make_env, load_policy, run_race, save
from mpc_adapter import MPCAdapter, MPCConfig
from t10_lib import SmartMistakes

which, laps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = make_env(max_laps=laps, max_steps=2200 * laps)
agent = load_policy(env, "ref_111s")
env.reset()
ad = MPCAdapter(env, agent, dict(np.load(os.path.join(RESULTS, "ref_profile.npz"))), MPCConfig()) if which.startswith("mpc") else None
grid_order = {}
gap_samples = {}
track_lap = {"leader": 0}


def fn(e, obs, active, step):
    if step == 0:
        ad.reset()
    return ad.act(e, obs, active)


def on_step(e, obs, step):
    # Hồi sinh tại chỗ như demo; xe đã về đủ số vòng thì rời đường đua
    for i, st in enumerate(e.racers):
        if not st.active and not st.finished:
            e.respawn_car(i)
    d = np.array([e.race_distance(s) for s in e.racers])
    track_lap["leader"] = int(max(d.max(), 0.0) // e.circuit_length)
    if step % 50 == 0:
        act = [s for s in e.racers if s.active and not s.finished]
        if len(act) >= 2:
            vals = np.sort([e.race_distance(s) for s in act])
            v = max(np.mean([np.linalg.norm(s.car.velocity) for s in act]), 5.0)
            gap_samples.setdefault(track_lap["leader"], []).append(float(np.median(np.diff(vals))) / v)


mist = SmartMistakes(env, max_per_car=2 * laps) if which.endswith("smart") else None
r = run_race(env, agent, seed=seed, mistakes=[mist] if mist else [], act_fn=fn if ad else None, on_step=on_step)
order_start = sorted(range(env.num_cars), key=lambda k: -env.racers[k].start_s)
grid_pos = [order_start.index(k) for k in range(env.num_cars)]
final = env.standings()
final_pos = [final.index(k) for k in range(env.num_cars)]
for h in r["held"]:
    h["leader_lap"] = int(h["frac"])
for o in r["outs"]:
    o["leader_lap"] = int(o["frac"])
all_laps = [float(x) for s in env.racers for x in s.lap_times]
env.close()
r.pop("follow_dy")
r.pop("follow_closing")
r.update({"which": which, "laps": laps, "seed": seed, "grid_pos": grid_pos, "final_pos": final_pos, "all_laps": all_laps,
          "gap_median_s": [float(np.median(gap_samples.get(k, [0.0]))) for k in range(laps)], "mistakes": mist.triggered if mist else 0})
save(f"t12_{which}_{laps}_{seed}", r)
