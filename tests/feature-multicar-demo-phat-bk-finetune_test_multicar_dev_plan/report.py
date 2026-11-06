"""Tổng hợp kết quả trong results/ thành bảng."""
import glob
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


def boot_ci(values, n=2000):
    rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summary(label, races, cars=6):
    if not races:
        return
    fin = [r["finished"] for r in races]
    held = [len(r["held"]) for r in races]
    laps = [r["mean_lap"] for r in races if r["mean_lap"] > 0]
    outs = [o for r in races for o in r["outs"]]
    crash = [o for o in outs if o["reason"] == "crash_or_track_limit"]
    hist = np.histogram(np.clip([o["frac"] for o in crash], 0, 0.999), bins=10, range=(0, 1))[0].tolist() if crash else [0] * 10
    lo, hi = boot_ci(fin)
    rear = sum(r["contact_types"]["rear"] for r in races)
    side = sum(r["contact_types"]["side"] for r in races)
    c3 = np.mean([o["contact_3s"] for o in crash]) if crash else 0
    print(f"{label:34} fin {sum(fin):3}/{len(races) * cars} (mean {np.mean(fin):.2f}, CI {lo:.2f}-{hi:.2f}) out {len(crash):2} stall {len(outs) - len(crash):2} "
          f"cont {np.mean([r['contacts'] for r in races]):6.1f} rear/side {rear}/{side} heldOT {np.mean(held):.2f} startOT {np.mean([r['start_passes'] for r in races]):.2f} "
          f"lap {np.mean(laps) if laps else 0:.1f} ped {np.mean([r['pedal_sw'] for r in races]):.1f} st {np.mean([r['steer_rev'] for r in races]):.1f} b {np.mean([r['beta'] for r in races]):.2f} "
          f"outHist {hist} c3s {c3:.2f}")


print("=== T1 baseline")
for m in ("ref_111s", "final_1000k"):
    d = load(f"t1_{m}")
    if d:
        det = d["deterministic"]
        summary(m + " det all", det)
        summary(m + " det first half", det[:len(det) // 2])
        summary(m + " det second half", det[len(det) // 2:])
        summary(m + " stochastic", d["stochastic"])
        dy = np.array([x for r in det for x in r["follow_dy"]])
        cl = np.array([x for r in det for x in r["follow_closing"]])
        if len(dy):
            print(f"   following <15m: samples {len(dy)}, |dy|<1.85 (same line) {np.mean(dy < 1.85):.2f}, |dy| median {np.median(dy):.2f}, closing median {np.median(cl):.2f} p90 {np.percentile(cl, 90):.2f}")
        held = [h for r in det for h in r["held"]]
        print(f"   held overtakes: {len(held)}, in corner {np.mean([h['corner'] for h in held]) if held else 0:.2f}, touched {np.mean([h['touched'] for h in held]) if held else 0:.2f}")

print("\n=== T2 obs norm (ref_111s)")
for c in ("l2", "l3", "l3_norm45"):
    d = load(f"t2_{c}")
    if d:
        summary(c + " solo", d["solo"], cars=1)
        summary(c + " pack", d["pack"])

print("\n=== T3 mistake cost (solo)")
for m in ("ref_111s", "final_1000k"):
    d = load(f"t3_{m}")
    if d:
        print(m, "clean lap", d["clean_lap"], "finished", d["clean_finished"])
        for kind, dur in sorted({(r["kind"], r["dur"]) for r in d["rows"]}):
            rows = [r for r in d["rows"] if r["kind"] == kind and r["dur"] == dur]
            loss = [r["loss"] for r in rows if r["loss"] is not None]
            print(f"  {kind} {dur}s: survive {len(loss)}/{len(rows)}, loss mean {np.mean(loss) if loss else 0:.2f}s range {min(loss) if loss else 0:.2f}..{max(loss) if loss else 0:.2f}, out at {[r['frac'] for r in rows if r['loss'] is None]}")

print("\n=== T4 duel")
for m in ("ref_111s", "final_1000k"):
    d = load(f"t4_duel_{m}")
    if d:
        print(m, "gap at mistake m:", sorted({r["gap_m"] for r in d["rows"]}))
        for kind in (None, "coast", "brake50"):
            for dur in sorted({r["dur"] for r in d["rows"] if r["kind"] == kind}):
                rows = [r for r in d["rows"] if r["kind"] == kind and r["dur"] == dur]
                print(f"  {kind} {dur}s n={len(rows)}: held pass {sum(bool(r['held']) for r in rows)}, any contact {sum(r['contacts'] > 0 for r in rows)}, rear {sum(r['contact_types']['rear'] for r in rows)}, side {sum(r['contact_types']['side'] for r in rows)}, runs with outs {sum(bool(r['outs']) for r in rows)}")

print("\n=== T4 pack with mistakes (3 per car per lap)")
for f in sorted(glob.glob(os.path.join(R, "t4_pack_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    summary(f"{d['model']} {d['kind']} {d['dur']}s", d["races"])

print("\n=== T5 opponent response")
for m in ("ref_111s", "final_1000k"):
    d = load(f"t5_{m}")
    if d:
        print(m, "logstd", d["logstd"], "base steer std", d["base_steer_std"])
        for r in d["rows"]:
            if r["closing"] in (0.0, 10.0) and r["dy"] in (0.0, 2.0):
                print(f"  dx {r['dx']:4} dy {r['dy']:3} closing {r['closing']:4}: |Δsteer| {r['steer_mean_abs']:.3f} p95 {r['steer_p95_abs']:.3f} toward-clear {r['steer_toward_clear']:+.3f} Δpedal {r['pedal_mean']:+.3f} p05 {r['pedal_p05']:+.3f}")

print("\n=== T6 train traffic")
d = load("t6_final_1000k")
if d:
    print(json.dumps({k: v for k, v in d.items() if k != "model"}))
