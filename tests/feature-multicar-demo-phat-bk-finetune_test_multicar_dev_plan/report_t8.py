"""Tổng hợp kết quả T8 (adapter MPC) so với ref_111s."""
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def ci(vals, n=2000):
    rng = np.random.default_rng(0)
    vals = np.asarray(vals, float)
    m = [rng.choice(vals, len(vals)).mean() for _ in range(n)]
    return f"{vals.mean():.2f} [{np.percentile(m, 2.5):.2f}, {np.percentile(m, 97.5):.2f}]"


def line(label, races, cars=6):
    crash = [o for r in races for o in r["outs"] if o["reason"] == "crash_or_track_limit"]
    stall = sum(1 for r in races for o in r["outs"] if o["reason"] == "stalled")
    laps = [r["mean_lap"] for r in races if r["mean_lap"] > 0]
    hist = np.histogram(np.clip([o["frac"] for o in crash], 0, 0.999), bins=10, range=(0, 1))[0].tolist() if crash else [0] * 10
    held = [h for r in races for h in r["held"]]
    print(f"{label}\n  finished {sum(r['finished'] for r in races)}/{len(races) * cars} per race {ci([r['finished'] for r in races])}; crash {len(crash)} stall {stall}; crash hist {hist}; contact within 3s {np.mean([o['contact_3s'] for o in crash]) if crash else 0:.2f}")
    print(f"  contacts/race {ci([r['contacts'] for r in races])}; episodes/race {ci([r['contact_episodes'] for r in races])}")
    print(f"  held overtakes/race {ci([len(r['held']) for r in races])}; corner share {np.mean([h['corner'] for h in held]) if held else 0:.2f}; touched share {np.mean([h['touched'] for h in held]) if held else 0:.2f}")
    print(f"  lap {np.mean(laps) if laps else 0:.1f} best {min([r['best_lap'] for r in races if r['best_lap'] > 0] or [0]):.1f}; pedal_sw {np.mean([r['pedal_sw'] for r in races]):.2f} steer_rev {np.mean([r['steer_rev'] for r in races]):.1f} beta {np.mean([r['beta'] for r in races]):.2f}")


d = load("t8_offset")
if d:
    print("=== line-offset tracking (3 laps each)")
    for off in dict.fromkeys(r["offset"] for r in d["rows"]):
        rows = [r for r in d["rows"] if r["offset"] == off]
        print(f"  {off}: finished {sum(r['finished'] for r in rows)}/3, laps {[round(r['lap'], 1) for r in rows]}, lat err rms {np.mean([r['lat_err_rms'] for r in rows]):.2f}")

d = load("t8_solo")
if d:
    print("\n=== solo")
    line("ref_111s", d["plain"], cars=1)
    line("mpc", d["mpc"], cars=1)
    same = sum(a["best_lap"] == b["best_lap"] and a["finished"] == b["finished"] for a, b in zip(d["plain"], d["mpc"]))
    print(f"  races identical to ref_111s: {same}/{len(d['plain'])}")

for fault in ("none", "brake50"):
    print(f"\n=== pack, fault={fault}")
    for which in ("ref", "mpc"):
        d = load(f"t8_pack_{which}_{fault}")
        if d:
            line(which, d["races"])
            if d.get("adapter"):
                print("  adapter:", {k: round(v, 3) for k, v in d["adapter"].items()})

print("\n=== duel")
for which in ("ref", "mpc"):
    d = load(f"t8_duel_{which}")
    if not d:
        continue
    print(which, "gaps", sorted({r["gap_m"] for r in d["rows"]}), d.get("adapter"))
    for kind, dur in [(None, 0.0), ("brake50", 1.0), ("brake50", 1.5)]:
        rows = [r for r in d["rows"] if r["kind"] == kind and r["dur"] == dur]
        print(f"  {kind} {dur}s n={len(rows)}: pass {sum(bool(r['held']) for r in rows)}, contact runs {sum(r['contacts'] > 0 for r in rows)}, episodes {sum(r['contact_episodes'] for r in rows)}, runs with crash {sum(any(o['reason'] == 'crash_or_track_limit' for o in r['outs']) for r in rows)}, runs with stall {sum(any(o['reason'] == 'stalled' for o in r['outs']) for r in rows)}")
