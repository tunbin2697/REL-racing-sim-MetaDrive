"""T18. Xe còn dư bao nhiêu để phanh muộn? Một xe, không train.
Khi ref_111s bắt đầu phanh trước cua: giữ ga thêm `delay` giây, rồi phanh `strength` cho tới khi về đúng tốc độ của vòng sạch tại vị trí đó, sau đó trả lái lại cho ref_111s.
    python t18_late_brake.py <delay_s> <strength>
    python t18_late_brake.py report
"""
import glob
import json
import os
import sys

import numpy as np

from common import RESULTS

if sys.argv[1] == "report":
    print(f"{'delay':>6}{'phanh':>7}{'về đích':>9}{'lap':>7}{'lợi s':>7}{'beta':>6}{'steer':>7}{'pedal':>7}{'cú/vòng':>8} nơi văng")
    rows = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(RESULTS, "t18_*.json"))]
    base = next((r["lap"] for r in rows if r["delay"] == 0.0), 0.0)
    for r in sorted(rows, key=lambda x: (x["strength"], x["delay"])):
        print(f"{r['delay']:>6.1f}{r['strength']:>7.1f}{r['finished']:>6}/{r['races']:<2}{r['lap']:>7.1f}{base - r['lap'] if r['lap'] else 0:>7.2f}{r['beta']:>6.2f}{r['steer_rev']:>7.1f}{r['pedal_sw']:>7.1f}{r['events_per_lap']:>8.1f} {r['outs']}")
    sys.exit(0)

from common import make_env, load_policy, run_race, save

delay, strength = float(sys.argv[1]), float(sys.argv[2])
env = make_env(num_cars=1)
agent = load_policy(env, "ref_111s")
prof = dict(np.load(os.path.join(RESULTS, "ref_profile.npz")))
DT = 0.1


class LateBrake:
    """Ghi đè pedal của xe 0 quanh các vùng phanh. Không đổi lái."""
    start_step = None

    def __init__(self):
        self.phase, self.count, self.calm, self.events = "idle", 0, 0, 0

    def apply(self, e, step, actions):
        st = e.racers[0]
        if not st.active:
            return
        a = actions[0]
        thr, brk = 0.5 * (a[1] + 1.0), 0.5 * (a[2] + 1.0)
        ref_braking = brk > 0.05 and st.speed_km_h > 60.0
        v = float(np.linalg.norm(st.car.velocity))
        target = float(prof["v"][st.current_track_idx])
        if self.phase == "idle":
            self.calm = self.calm + 1 if not ref_braking else 0
            if ref_braking and self.calm == 0 and self.count == 0:
                self.phase, self.count = "hold", 0
                self.events += 1
        if self.phase == "hold":
            self.count += 1
            a[1], a[2] = 1.0, -1.0
            if self.count >= int(round(delay / DT)):
                self.phase, self.count = "brake", 0
        elif self.phase == "brake":
            self.count += 1
            a[1], a[2] = -1.0, 2.0 * strength - 1.0
            if v <= target + 0.5 or self.count > 40:
                self.phase, self.count = "cool", 0
        elif self.phase == "cool":
            self.count += 1
            if not ref_braking and self.count > 20:
                self.phase, self.count = "idle", 0


races, events = [], []
for seed in range(6):
    lb = [LateBrake()] if delay > 0.0 else []
    races.append(run_race(env, agent, seed=seed, mistakes=lb))
    events.append(lb[0].events if lb else 0)
env.close()
laps = [r["best_lap"] for r in races if r["finished"]]
save(f"t18_{delay}_{strength}", {"delay": delay, "strength": strength, "races": len(races), "finished": sum(r["finished"] for r in races), "lap": float(np.mean(laps)) if laps else 0.0,
                                 "beta": float(np.mean([r["beta"] for r in races])), "steer_rev": float(np.mean([r["steer_rev"] for r in races])), "pedal_sw": float(np.mean([r["pedal_sw"] for r in races])),
                                 "events_per_lap": float(np.mean(events)), "outs": [o["frac"] for r in races for o in r["outs"]]})
