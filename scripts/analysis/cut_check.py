"""Cách dùng: python scripts/analysis/cut_check.py <checkpoint P2> [số_episode]
(Viết cho checkpoint Phase 2 có metadata, map silverstone_full, chế độ offtrack penalty; xem F22.)
Kiểm tra agent có cắt cua qua phần ngoài đường (shortcut) ở chế độ offtrack penalty không.
Mỗi sự kiện off-track: vị trí s, số bước, phía trong/ngoài cua, cung tim đường bị bỏ qua so với quãng xe đi thật."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from agents.ppo_agent import PPOAgent

ckpt = sys.argv[1]
n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
meta = PPOAgent.peek_meta(ckpt)
env = MetaRacingEnv(use_render=False, map_config="silverstone_full", start_seed=42, vehicle=get_vehicle("gt3_real"),
                    max_laps=1, obs_mode=meta.get("obs_mode", "pure"), action_mode=meta.get("action_mode", "combined"),
                    offtrack_mode="penalty")
obs, _ = env.reset(seed=42, options={"start_mode": "standing"})
ag = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_dim, device="cpu"); ag.load(ckpt)
hw = float(env.preset.half_width)
all_ev = []
for ep in range(n_ep):
    if ep:
        obs, _ = env.reset(seed=42, options={"start_mode": "standing"})
    rec = []
    while True:
        a = np.asarray(ag.select_action(obs, deterministic=True)[0], dtype=np.float32)
        obs, r, te, tr, info = env.step(a)
        idx = env.current_track_idx
        p = np.asarray(env.car.position[:2], float); c = env.track_xy[idx]; th = float(env.track_theta[idx])
        lat = float(np.dot(p - c, [-np.sin(th), np.cos(th)]))
        # độ cong trung bình quanh điểm (±30 m) để biết hướng cua
        k = float(np.mean(env.track_kappa[max(0, idx - 15): idx + 15]))
        rec.append((info["track_s"], p[0], p[1], lat, k, bool(info.get("off_track", False)), info["speed_km_h"]))
        if te or tr:
            break
    end = info.get("termination_reason"); lap = info.get("lap_time", 0.0)
    s, x, y, lat, k, off, v = [np.array(z) for z in zip(*rec)]
    print(f"--- ep{ep}: {end} lap={lap:.1f}s off-steps={int(off.sum())}")
    i = 0
    while i < len(off):
        if not off[i]:
            i += 1; continue
        j = i
        while j < len(off) and off[j]:
            j += 1
        a0, b0 = max(0, i - 1), min(len(s) - 1, j)
        arc = float(s[b0] - s[a0])
        path = float(np.sum(np.hypot(np.diff(x[a0:b0 + 1]), np.diff(y[a0:b0 + 1]))))
        km = float(np.mean(k[i:j])); lm = float(np.mean(lat[i:j]))
        side = "TRONG" if np.sign(km) == np.sign(lm) and abs(km) > 1e-3 else ("NGOÀI" if abs(km) > 1e-3 else "thẳng")
        maxoff = float(np.max(np.abs(lat[i:j])) - hw)
        # thời gian lợi ước tính: (cung tim đường - quãng đi) / tốc độ TB
        vm = float(np.mean(v[i:j])) / 3.6
        gain = (arc - path) / max(vm, 1.0)
        ev = dict(s=float(s[i]), steps=j - i, side=side, arc=arc, path=path, maxoff=maxoff, R=1 / abs(km) if abs(km) > 1e-4 else 9999, v=vm * 3.6, gain=gain)
        all_ev.append(ev)
        print(f"   s={ev['s']:6.0f} {ev['steps']:3d} bước, phía {side:5s} (R≈{ev['R']:5.0f} m), ra xa mép {maxoff:4.1f} m, "
              f"cung tim {arc:5.1f} m vs đi thật {path:5.1f} m, v {ev['v']:.0f} km/h, lợi ≈ {gain:+.2f}s")
        i = j
env.close()
if all_ev:
    tr = [e for e in all_ev if e["side"] == "TRONG"]
    print(f"== Tổng {len(all_ev)} sự kiện; phía trong cua {len(tr)}; tổng lợi thời gian ước tính {sum(e['gain'] for e in all_ev):+.2f}s / {n_ep} ep")
