"""T3. Một lỗi thật sự làm mất bao nhiêu giây? Chạy một xe, tạo lỗi ở 8 vị trí, so lap time với vòng không lỗi.
Mục đích: chọn loại lỗi tương ứng với mức chậm 1-2s mà chủ dự án mô tả.
    python t3_mistake_cost.py <model>
"""
import sys
from common import make_env, load_policy, run_race, Mistake, save

KINDS = [("coast", 1.0), ("coast", 2.0), ("coast", 3.0), ("brake50", 0.5), ("brake50", 1.0), ("brake50", 1.5)]
model = sys.argv[1]
env = make_env(num_cars=1)
agent = load_policy(env, model)
env.reset()
L = env.circuit_length
clean = run_race(env, agent, seed=0)
rows = []
for kind, dur in KINDS:
    for k in range(8):
        at = L * (0.08 + 0.105 * k)
        r = run_race(env, agent, seed=0, mistakes=[Mistake(0, at, kind, dur)])
        rows.append({"kind": kind, "dur": dur, "frac": round(at / L, 3), "finished": r["finished"], "lap": r["best_lap"],
                     "loss": round(r["best_lap"] - clean["best_lap"], 2) if r["finished"] else None,
                     "outs": r["outs"]})
env.close()
save(f"t3_{model}", {"model": model, "clean_lap": clean["best_lap"], "clean_finished": clean["finished"], "rows": rows})
