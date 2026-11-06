"""T8. Đo bản mẫu adapter MPC, không train.
    python t8_mpc.py profile              # ghi hồ sơ tốc độ và line của ref_111s
    python t8_mpc.py offset               # ref_111s có bám được line lệch khi dịch obs không?
    python t8_mpc.py solo <races>         # đường trống phải giống ref_111s
    python t8_mpc.py pack <ref|mpc> <races> [none|brake50]   # 6 xe, có thể thêm lỗi phanh 1s, 3 lần mỗi xe mỗi vòng
    python t8_mpc.py duel <ref|mpc>       # 2 xe, xe dẫn mắc lỗi (cùng kịch bản T4)
"""
import os
import sys

import numpy as np

from common import make_env, load_policy, run_race, Mistake, save, RESULTS
from mpc_adapter import MPCAdapter, record_profile, shifted_ego_obs
import common

PROFILE = os.path.join(RESULTS, "ref_profile.npz")
mode = sys.argv[1]


def adapter_fn(ad):
    def fn(env, obs, active, step):
        if step == 0:
            ad.reset()
        return ad.act(env, obs, active)
    return fn


def stats(ad):
    return {"plan_ms_per_step": 1000.0 * ad.plan_time / max(1, ad.total_steps / ad.env.num_cars),
            "engaged_share": ad.engaged_steps / max(1, ad.total_steps)}


if mode == "profile":
    env = make_env(num_cars=1)
    agent = load_policy(env, "ref_111s")
    prof, r = record_profile(env, agent)
    env.close()
    np.savez(PROFILE, **prof)
    print("profile saved, lap", r["best_lap"], "finished", r["finished"], flush=True)

elif mode == "offset":
    env = make_env(num_cars=1)
    agent = load_policy(env, "ref_111s")
    prof = dict(np.load(PROFILE))
    rows = []
    for off in (0.0, 1.5, -1.5, 3.0, -3.0, 4.5, -4.5, "swing2", "swing4"):
        for seed in range(3):
            lat_err = []

            def fn(e, obs, active, step):
                st = e.racers[0]
                o = float(off) if not isinstance(off, str) else float(off[5:]) * np.sin(step * 0.1 * 2 * np.pi / 10.0)
                shifted = np.array(obs, copy=True)
                if abs(o) > 1e-6:
                    shifted[0, :35] = shifted_ego_obs(e, st, o, obs[0])
                lat_err.append(float(st.lat_offset - prof["n"][st.current_track_idx] - o))
                a, _, _, _ = common.policy_act(agent, shifted[active], True)
                return a

            r = run_race(env, agent, seed=seed, act_fn=fn)
            rows.append({"offset": str(off), "seed": seed, "finished": r["finished"], "lap": r["best_lap"], "outs": r["outs"],
                         "steer_rev": r["steer_rev"], "beta": r["beta"],
                         "lat_err_mean": round(float(np.mean(lat_err[50:])), 3), "lat_err_rms": round(float(np.sqrt(np.mean(np.square(lat_err[50:])))), 3)})
    env.close()
    save("t8_offset", {"rows": rows})

elif mode == "solo":
    races = int(sys.argv[2])
    env = make_env(num_cars=1)
    agent = load_policy(env, "ref_111s")
    plain = [run_race(env, agent, seed=k) for k in range(races)]
    ad = MPCAdapter(env, agent, dict(np.load(PROFILE)))
    mpc = [run_race(env, agent, seed=k, act_fn=adapter_fn(ad)) for k in range(races)]
    env.close()
    save("t8_solo", {"plain": plain, "mpc": mpc})

elif mode == "pack":
    which, races, fault = sys.argv[2], int(sys.argv[3]), (sys.argv[4] if len(sys.argv) > 4 else "none")
    env = make_env()
    agent = load_policy(env, "ref_111s")
    env.reset()
    L = env.circuit_length
    ad = MPCAdapter(env, agent, dict(np.load(PROFILE))) if which == "mpc" else None
    out = []
    for seed in range(races):
        rng = np.random.default_rng(5000 + seed)
        ms = [] if fault == "none" else [Mistake(car, L * rng.uniform(0.08, 0.95), "brake50", 1.0) for car in range(env.num_cars) for _ in range(3)]
        out.append(run_race(env, agent, seed=seed, mistakes=ms, act_fn=adapter_fn(ad) if ad else None))
    env.close()
    save(f"t8_pack_{which}_{fault}", {"which": which, "fault": fault, "races": out, "adapter": stats(ad) if ad else None})

elif mode == "duel":
    which = sys.argv[2]
    env = make_env(num_cars=2)
    agent = load_policy(env, "ref_111s")
    env.reset()
    L = env.circuit_length
    ad = MPCAdapter(env, agent, dict(np.load(PROFILE))) if which == "mpc" else None
    fn = adapter_fn(ad) if ad else None
    rows = []
    for k in range(8):
        at = L * (0.12 + 0.10 * k)
        info = {}

        def probe(e, obs, step):
            d = [e.race_distance(s) for s in e.racers]
            if "gap" not in info and max(d) >= at:
                info["gap"], info["leader"] = float(max(d) - min(d)), int(np.argmax(d))

        run_race(env, agent, seed=k, on_step=probe, act_fn=fn)
        leader = info.get("leader", 0)
        for kind, dur in [(None, 0.0), ("brake50", 1.0), ("brake50", 1.5)]:
            ms = [] if kind is None else [Mistake(leader, at, kind, dur)]
            r = run_race(env, agent, seed=k, mistakes=ms, stop_after_mistake=300 if kind else None, act_fn=fn)
            rows.append({"frac": round(at / L, 3), "kind": kind, "dur": dur, "gap_m": round(info.get("gap", -1.0), 1), "leader": leader,
                         "held": r["held"], "contacts": r["contacts"], "contact_episodes": r["contact_episodes"], "outs": r["outs"], "finished": r["finished"]})
    env.close()
    save(f"t8_duel_{which}", {"which": which, "rows": rows, "adapter": stats(ad) if ad else None})
