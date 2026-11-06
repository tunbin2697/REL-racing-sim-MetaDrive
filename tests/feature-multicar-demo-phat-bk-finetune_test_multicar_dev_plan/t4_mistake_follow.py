"""T4. Giả thuyết của chủ dự án: xe giống hệt nhau, xe trước mắc lỗi chậm 1-2s thì xe sau có cơ hội vượt.
Với policy hiện có (chưa train cho tình huống này), xe sau vượt được, húc vào, hay văng?
- duel: 2 xe, xe đang dẫn mắc 1 lỗi. Có đối chứng không lỗi ở cùng vị trí.
- pack: 6 xe, mỗi xe mắc lỗi ngẫu nhiên khoảng 3 lần mỗi vòng. So với T1 (không lỗi).
    python t4_mistake_follow.py duel <model>
    python t4_mistake_follow.py pack <model> <races> <kind> <duration>
"""
import sys
import numpy as np
from common import make_env, load_policy, run_race, Mistake, save

mode, model = sys.argv[1], sys.argv[2]

if mode == "duel":
    env = make_env(num_cars=2)
    agent = load_policy(env, model)
    env.reset()
    L = env.circuit_length
    rows = []
    # Grid so le: xe 0 xuất phát phía trước; lỗi gắn với xe đang dẫn ở lần chạy đối chứng
    for k in range(8):
        at = L * (0.12 + 0.10 * k)
        leader_gap = {}

        def probe(e, obs, step):
            d = [e.race_distance(s) for s in e.racers]
            if "gap" not in leader_gap and max(d) >= at:
                leader_gap["gap"] = float(max(d) - min(d))
                leader_gap["leader"] = int(np.argmax(d))

        # Chạy thăm dò để biết xe dẫn và khoảng cách tại vị trí tạo lỗi
        run_race(env, agent, seed=k, on_step=probe)
        leader = leader_gap.get("leader", 0)
        for kind, dur in [(None, 0.0), ("coast", 2.0), ("brake50", 1.0), ("brake50", 1.5)]:
            ms = [] if kind is None else [Mistake(leader, at, kind, dur)]
            stop = 300 if kind is not None else None
            r = run_race(env, agent, seed=k, mistakes=ms, stop_after_mistake=stop)
            rows.append({"frac": round(at / L, 3), "kind": kind, "dur": dur, "gap_m": round(leader_gap.get("gap", -1.0), 1),
                         "leader": leader, "held": r["held"], "contacts": r["contacts"], "contact_types": r["contact_types"],
                         "outs": r["outs"], "finished": r["finished"]})
    env.close()
    save(f"t4_duel_{model}", {"model": model, "rows": rows})
else:
    races, kind, dur = int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
    env = make_env()
    agent = load_policy(env, model)
    env.reset()
    L = env.circuit_length
    out = []
    for seed in range(races):
        rng = np.random.default_rng(5000 + seed)
        ms = [Mistake(car, L * rng.uniform(0.08, 0.95), kind, dur) for car in range(env.num_cars) for _ in range(3)]
        out.append(run_race(env, agent, seed=seed, mistakes=ms))
    env.close()
    save(f"t4_pack_{model}_{kind}{dur}", {"model": model, "kind": kind, "dur": dur, "races": out})
