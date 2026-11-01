"""Cách dùng: python scripts/analysis/ghost_envelope_sweep.py a_lat,a_brake,a_engine [...]   (biến môi trường: N_EP, OUT)
Cấu hình đã chọn cho gt3_anchor: 9.32,15.0,3.09 (knowledge/vehicle_and_engine_physics.md §6.5).
Lưu ý: mỗi cấu hình ghi đè scripts/silverstone_half_gt3_racecar_env_optimal_profile.npz; xóa file đó sau khi quét.
Quét envelope cho ghost map half trên đúng vật lý GT3RaceCar (xe của 111s).
Mỗi cấu hình: tính lại racing line + profile (v23), cho ghost chạy 2 vòng (standing + flying), ghi T*, lap, crash."""
import os, sys, json, time, dataclasses
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import GT3RaceCar
from agents.optimal_agent import OptimalRacingAgent

G = 9.81
configs = [tuple(map(float, c.split(","))) for c in sys.argv[1:]] or [(0.90 * 0.90 * G, 15.0, 3.09)]
n_ep = int(os.environ.get("N_EP", "2"))
out = []
for a_lat, a_brk, a_eng in configs:
    base = GT3RaceCar().cfg
    cfg = dataclasses.replace(base, name="GT3_RaceCar_env", envelope_a_lat=a_lat, envelope_a_brake=a_brk, envelope_a_engine=a_eng)
    car = GT3RaceCar(cfg)
    env = MetaRacingEnv(map_type="silverstone", vehicle=car, use_render=False, max_laps=2)
    t0 = time.time()
    obs, _ = env.reset()
    bench = env.get_benchmark_engine()
    tstar = bench.theoretical_best_lap_time
    res = dict(a_lat=a_lat, a_lat_g=a_lat / G, a_brake=a_brk, a_engine=a_eng, T_star=tstar, episodes=[])
    for ep in range(n_ep):
        if ep:
            obs, _ = env.reset()
        ag = OptimalRacingAgent.from_benchmark(bench, vehicle_cfg=car.cfg, v_norm=env.preset.v_norm); ag.reset()
        laps, end, steps, off = [], "horizon", 0, 0
        while steps < 8000:
            a, _, _ = ag.select_action(obs, vehicle=env.car)
            obs, r, te, tr, info = env.step(a); steps += 1
            if info.get("lap_completed") and info.get("completed_lap_time"):
                laps.append(round(float(info["completed_lap_time"]), 2))
            if te or tr:
                end = info.get("termination_reason", "?"); break
        res["episodes"].append(dict(laps=laps, end=end, crash_s=float(info.get("crash_distance", -1))))
    env.close()
    print(f"a_lat {a_lat:5.2f} ({a_lat/G:.2f}G) brake {a_brk:5.1f} eng {a_eng:4.2f} | T*={tstar:6.2f}s | "
          + " ; ".join(f"{e['end']} laps={e['laps']}" + (f" crash@{e['crash_s']:.0f}" if e['end'] != 'lap_completed' else "") for e in res["episodes"])
          + f" | {time.time()-t0:.0f}s", flush=True)
    out.append(res)
path = os.environ.get("OUT")
if path:
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
