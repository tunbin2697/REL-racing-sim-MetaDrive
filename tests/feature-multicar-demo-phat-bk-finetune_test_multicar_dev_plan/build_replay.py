"""Gộp 3 chặng đua đã ghi bằng T14 thành một trang phát lại cạnh nhau.
    python build_replay.py <seed> <đường_dẫn_html_đầu_ra>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LABELS = {"ref": "Chỉ ref_111s", "mpc": "ref_111s + MPC", "mpc_faults": "ref_111s + MPC + lỗi bám lốp"}
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
seed, out = sys.argv[1], sys.argv[2]


def ranks(frames, center):
    """Xếp hạng từng khung hình theo quãng đường đã đi trên tim đường."""
    pts = np.asarray(center, dtype=np.float32)
    arr = np.asarray(frames, dtype=np.float32)
    n_pts = len(pts)
    idx = np.zeros((len(arr), 6), dtype=np.int64)
    for c in range(6):
        d = ((arr[:, c * 5, None] - pts[:, 0]) ** 2 + (arr[:, c * 5 + 1, None] - pts[:, 1]) ** 2)
        idx[:, c] = d.argmin(axis=1)
    delta = np.diff(idx, axis=0)
    delta = np.where(delta < -n_pts / 2, delta + n_pts, delta)
    delta = np.where(delta > n_pts / 2, delta - n_pts, delta)
    start = np.where(idx[0] > n_pts / 2, idx[0] - n_pts, idx[0])
    progress = np.vstack([start, start + np.cumsum(delta, axis=0)])
    return np.argsort(-progress, axis=1, kind="stable").tolist()


modes = []
for which, label in LABELS.items():
    d = json.load(open(os.path.join(RESULTS, f"t14_{which}_{seed}.json"), encoding="utf-8"))
    frames = [[int(v) if float(v).is_integer() else v for v in row] for row in d["frames"]]
    modes.append({"label": label, "frames": frames, "ranks": ranks(d["frames"], d["track"]["center"]), "track": d["track"], "events": d["events"],
                  "n_overtakes": d["n_overtakes"], "n_outs": d["n_outs"], "n_mistakes": d["n_mistakes"], "contacts": d["contacts"], "best_lap": d["best_lap"]})
html = open(os.path.join(HERE, "replay_template.html"), encoding="utf-8").read()
assert html.count("/*DATA*/null") == 1
html = html.replace("/*DATA*/null", json.dumps({"modes": modes}, ensure_ascii=False, separators=(",", ":")))
open(out, "w", encoding="utf-8").write(html)
print("written", out, os.path.getsize(out) // 1024, "KB", [(m["label"], len(m["frames"]), m["n_overtakes"]) for m in modes])
