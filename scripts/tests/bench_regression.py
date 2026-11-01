"""Kiểm tra hồi quy benchmark: map half phải ra lại T* = 112.6s; tính T* map full với envelope đo được."""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from game_env.racing_env import MetaRacingEnv
from game_env.benchmark import RacingBenchmarkEngine
from game_env.vehicle import GT3RaceCar, GT3RealCar

import shutil, tempfile
for map_type, car in [("silverstone_half", GT3RaceCar()), ("silverstone_full", GT3RealCar())]:
    env = MetaRacingEnv(map_type=map_type, vehicle=car, use_render=False)
    env.reset()
    cached = env.get_benchmark_engine()
    cache_files = [os.path.join('scripts', f) for f in os.listdir('scripts') if f.endswith('.npz')]
    keep = {f: open(f, 'rb').read() for f in cache_files}  # force_recompute ghi đè cache: giữ nguyên bản gốc
    fresh = RacingBenchmarkEngine(track_xy=env.track_xy, track_theta=env.track_theta, track_kappa=env.track_kappa,
                                  track_s=env.track_s, vehicle_cfg=car.cfg, track_width=2 * env.preset.half_width,
                                  preset_name=env.preset.name, force_recompute=True)
    v = fresh.optimal_speed_kmh
    print(f"[{map_type}] L={env.circuit_length:.1f}m  N={fresh.num_points}  version={fresh.profile_version}")
    print(f"   T* cached={cached.theoretical_best_lap_time:.2f}s  recomputed={fresh.theoretical_best_lap_time:.2f}s")
    print(f"   line diff vs cache max={np.max(np.linalg.norm(fresh.optimal_line_xy - cached.optimal_line_xy, axis=1)):.4f}m")
    print(f"   v: min={v.min():.1f} max={v.max():.1f} km/h | % at cap={100*np.mean(v >= car.cfg.max_speed_km_h - 1):.1f}%")
    print("   apexes:", [round(a["target_speed_kmh"]) for a in sorted(fresh.corner_apexes, key=lambda a: a["s"])])
    for f, b in keep.items():
        open(f, 'wb').write(b)
    env.close()
