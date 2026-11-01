"""R8: OptimalRacingAgent (ghost) tự lái 1 vòng trên map half và full qua MetaRacingEnv.step."""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import GT3RaceCar, GT3RealCar
from agents.optimal_agent import OptimalRacingAgent

maps = sys.argv[1:] or ["silverstone_half", "silverstone_full"]
for map_type in maps:
    car = GT3RealCar() if "full" in map_type else GT3RaceCar()
    env = MetaRacingEnv(map_type=map_type, vehicle=car, use_render=False, max_laps=1)
    obs, info = env.reset()
    bench = env.get_benchmark_engine()
    agent = OptimalRacingAgent.from_benchmark(bench, vehicle_cfg=car.cfg, v_norm=env.preset.v_norm)
    agent.reset()
    speeds, step, lap_time, end = [], 0, None, "horizon"
    max_steps = int(3000 * env.preset.time_scale)
    while step < max_steps:
        action, _, _ = agent.select_action(obs, vehicle=env.car)
        obs, r, term, trunc, info = env.step(action)
        speeds.append(info.get("speed_km_h", 0.0)); step += 1
        if info.get("lap_completed"):
            lap_time = info.get("lap_time"); end = "lap"; break
        if term or trunc:
            end = f"term s={getattr(env, 'last_s', 0):.0f}m ({100*getattr(env,'last_s',0)/env.circuit_length:.1f}%)"; break
    print(f"[{map_type}] T*={bench.theoretical_best_lap_time:.2f}s | end={end} | lap_time={lap_time} | steps={step} ({step*env.dt:.1f}s) | vmax={max(speeds):.1f} vmean={np.mean(speeds):.1f} km/h")
    env.close()
