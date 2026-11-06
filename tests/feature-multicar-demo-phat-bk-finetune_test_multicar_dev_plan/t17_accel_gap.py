"""T17. Gia tốc và trần tốc độ có làm xe sau khó tận dụng khoảng trống nhỏ không? Một xe, không train.
Đổi công suất bằng MC_ENGINE, trần tốc độ bằng MC_VMAX.
- Gia tốc thật theo dải tốc độ và tỉ lệ thời gian chạm trần trong một vòng.
- ref_111s còn lái được không.
- Lỗi nhỏ: phanh 30% trong 0.2s hoặc 0.4s ngay trước khi vào cua, rồi so tốc độ theo vị trí với vòng sạch:
  xe mất bao nhiêu km/h lúc ra cua, chênh lệch kéo dài bao xa, mất bao nhiêu giây trước khi đoạn thẳng kết thúc.
    python t17_accel_gap.py run
    python t17_accel_gap.py report
"""
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

if sys.argv[1] == "report":
    print(f"{'engine':>7}{'vmax':>6}{'solo':>6}{'lap':>7}{'beta':>6}{'steer':>7}{'trần%':>7}{'a40-80':>8}{'a80-110':>8}{'a>110':>7}{'t0-100':>8} | không lỗi | lỗi 0.2s: Δv, xa, Δt | lỗi 0.4s")
    for p in sorted(glob.glob(os.path.join(RESULTS, "t17_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        f = {x["dur"]: x for x in d["small_fault"]}
        print(f"{d['engine']:>7.0f}{d['vmax']:>6.0f}{d['finished']:>4}/6{d['lap']:>7.1f}{d['beta']:>6.2f}{d['steer_rev']:>7.1f}{100 * d['at_cap_share']:>7.1f}{d['a_40_80']:>8.2f}{d['a_80_110']:>8.2f}{d['a_110up']:>7.2f}{d['t_0_100']:>8.1f} | "
              f"{f[0.0]['dv_kmh']:.1f}, {f[0.0]['dist_m']:.0f}m, {f[0.0]['dt_s']:+.2f}s | {f[0.2]['dv_kmh']:.1f} km/h, {f[0.2]['dist_m']:.0f}m, {f[0.2]['dt_s']:+.2f}s | {f[0.4]['dv_kmh']:.1f} km/h, {f[0.4]['dist_m']:.0f}m, {f[0.4]['dt_s']:+.2f}s")
    sys.exit(0)

from common import make_env, load_policy, policy_act, run_race, save
import dataclasses
import game_env.multicar_env as menv
from game_env.vehicle import GT3RaceCar

engine = float(os.environ.get("MC_ENGINE", 850.0))
vmax = float(os.environ.get("MC_VMAX", 120.0))
cfg = dataclasses.replace(GT3RaceCar().cfg, max_engine_force=engine, max_speed_km_h=vmax)
menv.GT3RaceCar = lambda: GT3RaceCar(dataclasses.replace(cfg))
env = make_env(num_cars=1)
agent = load_policy(env, "ref_111s")


class SmallFault:
    """Phanh nhẹ rất ngắn ngay trước khi vào cua."""
    start_step = None

    def __init__(self, at_m, dur):
        self.at_m, self.steps, self.begin = at_m, int(round(dur / 0.1)), None

    def apply(self, e, step, actions):
        st = e.racers[0]
        if self.begin is None:
            idx = st.current_track_idx
            n = e.num_track_points
            near = np.abs(e.track_kappa[np.arange(idx + 5, idx + 30) % n]) > 0.0
            if e.race_distance(st) < self.at_m or np.abs(e.track_kappa[idx]) > 0.0 or not near.any() or st.speed_km_h < 50.0:
                return
            self.begin = step
            self.where = float(e.race_distance(st))
        if self.steps > 0 and step - self.begin < self.steps:
            actions[0][1:] = [-1.0, -0.4]


def lap_trace(fault=None, seed=0):
    """Tốc độ và thời điểm theo từng điểm tim đường. Xe đi khoảng 3m mỗi bước nên phải nội suy các điểm ở giữa."""
    dist, vs_, ts = [], [], []

    def probe(e, obs, step):
        st = e.racers[0]
        if st.active:
            dist.append(e.race_distance(st))
            vs_.append(st.speed_km_h)
            ts.append(step * e.dt)

    r = run_race(env, agent, seed=seed, mistakes=[fault] if fault else [], on_step=probe)
    # Xe xuất phát sau vạch nên quãng đường ban đầu âm; nội suy theo quãng đường, không theo chỉ số điểm
    d = np.maximum.accumulate(np.array(dist))
    keep = np.concatenate([[True], d[1:] > d[:-1]])
    return np.interp(env.track_s, d[keep], np.array(vs_)[keep]), np.interp(env.track_s, d[keep], np.array(ts)[keep]), r


races = [run_race(env, agent, seed=k) for k in range(6)]
base_v, base_t, base = lap_trace()
at_cap = float(np.mean(base_v >= vmax - 2.0))

# Tăng tốc thẳng toàn ga, lái theo policy nhưng ép ga hết cỡ để đo gia tốc theo dải tốc độ
obs, _ = env.reset()
st = env.racers[0]
actions = np.zeros((1, 3), dtype=np.float32)
vs = []
for step in range(600):
    a, _, _, _ = policy_act(agent, obs, True)
    actions[0] = [a[0][0], 1.0, -1.0]
    obs, _, d, tr, _ = env.step(actions)
    if d[0] or tr:
        break
    vs.append(st.speed_km_h)
vs = np.array(vs)


def accel(lo, hi):
    i0 = np.argmax(vs >= lo) if (vs >= lo).any() else None
    i1 = np.argmax(vs >= hi) if (vs >= hi).any() else None
    if i0 is None or i1 is None or i1 <= i0:
        return 0.0
    return float((hi - lo) / 3.6 / ((i1 - i0) * env.dt))


rows = []
L = env.circuit_length
for dur in (0.0, 0.2, 0.4):  # 0.0 là mốc nhiễu: không có lỗi nhưng vẫn chạy lại vòng đua
    dvs, dists, dts = [], [], []
    for k in range(6):
        fault = SmallFault(L * (0.05 + 0.15 * k), dur)
        v, t, r = lap_trace(fault)
        if not r["finished"] or fault.begin is None:
            continue
        start = int(np.argmin(np.abs(env.track_s - (fault.where % L))))
        if start + 400 >= env.num_track_points:  # Cửa sổ đo phải nằm trọn trong vòng
            continue
        win = np.arange(start, start + 400)
        delta = base_v[win] - v[win]
        dvs.append(float(delta[:120].max()))
        over = np.flatnonzero(delta[:300] > 1.0)
        dists.append(float(over[-1]) if len(over) else 0.0)
        dts.append(float(t[win[150]] - base_t[win[150]]))
    rows.append({"dur": dur, "dv_kmh": float(np.mean(dvs)) if dvs else 0.0, "dist_m": float(np.mean(dists)) if dists else 0.0, "dt_s": float(np.mean(dts)) if dts else 0.0, "n": len(dvs)})
env.close()
laps = [r["best_lap"] for r in races if r["finished"]]
save(f"t17_{int(engine)}_{int(vmax)}", {"engine": engine, "vmax": vmax, "finished": sum(r["finished"] for r in races), "lap": float(np.mean(laps)) if laps else 0.0,
                                        "beta": float(np.mean([r["beta"] for r in races])), "steer_rev": float(np.mean([r["steer_rev"] for r in races])), "at_cap_share": at_cap,
                                        "a_40_80": accel(40, 80), "a_80_110": accel(80, 110), "a_110up": accel(110, min(vmax - 1.0, 118.0)) if vmax > 112 else 0.0,
                                        "t_0_100": float(np.argmax(vs >= 100) * env.dt) if (vs >= 100).any() else 0.0, "small_fault": rows})
