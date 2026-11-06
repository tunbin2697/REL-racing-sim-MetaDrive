"""T15. Hai bài đo một xe, không train:
    python t15_physics_faults.py physics <mu_front> <brake_force>   # đổi vật lý: phanh có còn chúc đầu không, ref_111s còn lái được không
    python t15_physics_faults.py faults                             # mỗi loại lỗi mất bao nhiêu giây, xe có sống không
    python t15_physics_faults.py report
"""
import dataclasses
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

SEVS = ("light", "mid", "hard", "ext")
if sys.argv[1] == "report":
    print(f"{'mu_f':>5}{'brake':>7}{'solo':>7}{'lap':>7}{'beta':>6}{'steer':>7}{'pedal':>7}{'peak_g':>8}{'mean_g':>8}{'pitch':>7}{'rearZ':>7}")
    for p in sorted(glob.glob(os.path.join(RESULTS, "t15_physics_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        print(f"{d['mu_f']:>5}{d['brake']:>7}{d['finished']:>4}/{d['races']:<2}{d['lap']:>7.1f}{d['beta']:>6.2f}{d['steer_rev']:>7.1f}{d['pedal_sw']:>7.1f}{d['peak_decel_g']:>8.2f}{d.get('mean_decel_g', 0):>8.2f}{d['max_nose_down_deg']:>7.1f}{d['rear_lift_m']:>7.2f}")
    d = json.load(open(os.path.join(RESULTS, "t15_faults.json"), encoding="utf-8")) if os.path.exists(os.path.join(RESULTS, "t15_faults.json")) else None
    if d:
        print("\nclean lap", d["clean_lap"])
        for kind in d["kinds"]:
            for sev in SEVS:
                rows = [r for r in d["rows"] if r["kind"] == kind and r["severity"] == sev]
                loss = [r["loss"] for r in rows if r["loss"] is not None]
                fired = sum(r["fired"] for r in rows)
                print(f"  {kind:11}{sev:6} nổ {fired}/{len(rows)}, sống {len(loss)}/{len(rows)}, mất TB {np.mean(loss) if loss else 0:5.2f}s, khoảng {min(loss) if loss else 0:.1f}..{max(loss) if loss else 0:.1f}s")
    sys.exit(0)

import game_env.multicar_env as menv
from game_env.vehicle import GT3RaceCar
from common import make_env, load_policy, run_race, policy_act, save
from grip_faults import GripFault, KINDS, SEVERITY

mode = sys.argv[1]

if mode == "physics":
    mu_f, brake = float(sys.argv[2]), float(sys.argv[3])
    base = GT3RaceCar().cfg
    cfg = dataclasses.replace(base, front_wheel_friction=mu_f, max_brake_force=brake)
    menv.GT3RaceCar = lambda: GT3RaceCar(dataclasses.replace(cfg))
    env = make_env(num_cars=1)
    agent = load_policy(env, "ref_111s")
    races = [run_race(env, agent, seed=k) for k in range(6)]
    # Phanh gấp hết cỡ trên đoạn thẳng từ tốc độ cao
    obs, _ = env.reset()
    st = env.racers[0]
    actions = np.zeros((1, 3), dtype=np.float32)
    braking, peak_g, nose, rear_lift, prev_v = False, 0.0, 0.0, 0.0, 0.0
    v_start, brake_steps = 0.0, 0
    z0 = None
    for step in range(2500):
        a, _, _, _ = policy_act(agent, obs, True)
        actions[0] = a[0]
        idx = st.current_track_idx
        if not braking and step > 100 and st.speed_km_h > 95.0 and not np.any(np.abs(env.track_kappa[np.arange(idx, idx + 120) % env.num_track_points]) > 0.0):
            braking = True
            v_start = float(np.linalg.norm(st.car.velocity))
            z0 = float(st.raw_vehicle.origin.get_pos()[2])
        if braking:
            actions[0][1:] = [-1.0, 1.0]
        obs, _, d, tr, _ = env.step(actions)
        if d[0] or tr:
            break
        v = float(np.linalg.norm(st.car.velocity))
        if braking:
            brake_steps += 1
            peak_g = max(peak_g, (prev_v - v) / env.dt / 9.81)
            hpr = st.raw_vehicle.origin.get_hpr()
            nose = max(nose, abs(float(hpr[1])))
            rear_lift = max(rear_lift, float(st.raw_vehicle.origin.get_pos()[2]) - z0)
            if v < 5.0:
                break
        prev_v = v
    env.close()
    laps = [r["best_lap"] for r in races if r["finished"]]
    save(f"t15_physics_{mu_f}_{int(brake)}", {"mu_f": mu_f, "brake": brake, "races": len(races), "finished": sum(r["finished"] for r in races), "lap": float(np.mean(laps)) if laps else 0.0,
                                              "beta": float(np.mean([r["beta"] for r in races])), "steer_rev": float(np.mean([r["steer_rev"] for r in races])), "pedal_sw": float(np.mean([r["pedal_sw"] for r in races])),
                                              "peak_decel_g": peak_g, "mean_decel_g": (v_start - prev_v) / max(1, brake_steps) / env.dt / 9.81, "max_nose_down_deg": nose, "rear_lift_m": rear_lift, "outs": [o["frac"] for r in races for o in r["outs"]]})
else:
    env = make_env(num_cars=1)
    agent = load_policy(env, "ref_111s")
    env.reset()
    L = env.circuit_length
    clean = np.mean([run_race(env, agent, seed=k)["best_lap"] for k in range(3)])
    rows = []
    for kind in KINDS:
        for sev in SEVERITY[kind]:
            for k in range(8):
                fault = GripFault(0, L * (0.05 + 0.115 * k), kind, sev)
                r = run_race(env, agent, seed=k, mistakes=[fault])
                rows.append({"kind": kind, "severity": sev, "at": round(0.05 + 0.115 * k, 3), "fired": fault.begin is not None, "loss": round(r["best_lap"] - clean, 2) if r["finished"] else None, "outs": [o["frac"] for o in r["outs"]]})
    env.close()
    save("t15_faults", {"clean_lap": float(clean), "kinds": list(KINDS), "rows": rows})
