"""Chạy toàn bộ test song song, tối đa 5 tiến trình để chừa CPU cho việc train khác."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = [
    "t1_baseline_noise.py ref_111s 24",
    "t2_obs_norm.py l2 12",
    "t2_obs_norm.py l3 12",
    "t2_obs_norm.py l3_norm45 12",
    "t3_mistake_cost.py ref_111s",
    "t4_mistake_follow.py duel ref_111s",
    "t4_mistake_follow.py pack ref_111s 12 coast 2.0",
    "t5_opponent_response.py ref_111s",
]
os.makedirs(os.path.join(HERE, "results", "console"), exist_ok=True)
pending, running = list(JOBS), []
while pending or running:
    while pending and len(running) < 5:
        job = pending.pop(0)
        log = open(os.path.join(HERE, "results", "console", job.replace(".py", "").replace(" ", "_") + ".txt"), "w", encoding="utf-8")
        running.append((job, subprocess.Popen([sys.executable, "-u"] + job.split(), cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT), time.time()))
    for item in list(running):
        job, proc, t0 = item
        if proc.poll() is not None:
            running.remove(item)
            print(f"{job}: exit {proc.returncode}, {time.time() - t0:.0f}s", flush=True)
    time.sleep(2)
print("ALL_DONE", flush=True)
