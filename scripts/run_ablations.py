"""Lập lịch ablation hàm thưởng trên env ee99f9d, có resume.

    python scripts/run_ablations.py run [--parallel 3] [--runs no_steer,no_center] [--dry-run]
    python scripts/run_ablations.py status
    python scripts/run_ablations.py eval [--episodes 20]
    python scripts/run_ablations.py table

Mỗi run chỉ đổi hệ số reward ghi trong RUNS; mọi tham số train khác để mặc định của train_racing.py.
Ngắt giữa chừng (Ctrl+C, tắt máy) rồi chạy lại đúng lệnh: run đã xong được bỏ qua, run dở tiếp từ mốc eval gần nhất.
Sau resume không tái lập từng bit vì trạng thái ngẫu nhiên không được lưu; mốc resume ghi ở resume_marks.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_COMMIT = "ee99f9d"
CROP_STEP = 3082240  # Bằng --total-steps mặc định: mốc ghi kỷ lục tốc độ cuối cùng của ref_111s (2048 x 1505)
MAX_RETRIES = 2
POLL_SECONDS = 10


@dataclass(frozen=True)
class AblationRun:
    name: str
    flags: Dict[str, float]
    question: str

    @property
    def save_dir(self) -> str:
        return os.path.join("models", f"abl_{self.name}")

    @property
    def log_dir(self) -> str:
        return os.path.join("logs", f"abl_{self.name}")

    def command(self) -> List[str]:
        cmd = [sys.executable, "-u", "training/train_racing.py", "--auto-resume",
               "--save-dir", self.save_dir, "--log-dir", self.log_dir,
               "--tb-logdir", os.path.join(self.log_dir, "tb")]
        for flag, value in self.flags.items():
            cmd += [f"--{flag}", str(value)]
        return cmd

    def state(self) -> Optional[dict]:
        path = os.path.join(ROOT, self.save_dir, "train_state.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def finished(self) -> bool:
        state = self.state()
        return bool(state and state.get("finished"))


# Thứ tự là thứ tự ưu tiên chạy
RUNS = [
    AblationRun("no_center", {"center-coef": 0.0}, "Bỏ phạt lệch tim đường"),
    AblationRun("no_steer", {"steer-coef": 0.0}, "Bỏ phạt đánh lái"),
    AblationRun("time_penalty", {"time-penalty": 0.01}, "Thêm phạt thời gian (ref_111s không có)"),
    AblationRun("crash_flat", {"crash-speed": 0.0}, "Phạt va chạm cố định -10, không theo tốc độ"),
    AblationRun("no_finish", {"finish-base": 0.0, "finish-time-scale": 0.0}, "Bỏ toàn bộ thưởng về đích"),
    AblationRun("no_time_bonus", {"finish-time-scale": 0.0}, "Bỏ thưởng bậc thang theo lap time"),
    AblationRun("no_idle", {"idle-penalty": 0.0}, "Bỏ phạt dưới 10 km/h"),
]
BASELINE = AblationRun("baseline", {}, "Đối chứng: công thức gốc, cùng số step")


def select(names: Optional[str], include_baseline: bool = False) -> List[AblationRun]:
    pool = RUNS + [BASELINE] if include_baseline or (names and "baseline" in names) else RUNS
    if not names:
        return pool
    wanted = [n.strip() for n in names.split(",")]
    unknown = set(wanted) - {r.name for r in pool}
    if unknown:
        raise SystemExit(f"Không có run: {sorted(unknown)}")
    return [r for r in pool if r.name in wanted]


def cmd_run(args) -> None:
    os.chdir(ROOT)
    queue = [r for r in select(args.runs, args.include_baseline) if not r.finished]
    if args.dry_run:
        for r in queue:
            print(" ".join(r.command()))
        return
    retries = {r.name: 0 for r in queue}
    active = {}
    try:
        while queue or active:
            while queue and len(active) < args.parallel:
                run = queue.pop(0)
                os.makedirs(run.log_dir, exist_ok=True)
                log = open(os.path.join(run.log_dir, "console.txt"), "a", encoding="utf-8")
                env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                active[run.name] = (run, subprocess.Popen(run.command(), stdout=log, stderr=subprocess.STDOUT, env=env), log)
                print(f"[{time.strftime('%H:%M')}] Bắt đầu {run.name} (resume từ {(run.state() or {}).get('global_step', 0)})", flush=True)
            time.sleep(POLL_SECONDS)
            for name, (run, proc, log) in list(active.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del active[name]
                if proc.returncode == 0 and run.finished:
                    print(f"[{time.strftime('%H:%M')}] Xong {name}", flush=True)
                elif retries[name] < MAX_RETRIES:
                    retries[name] += 1
                    queue.insert(0, run)
                    print(f"[{time.strftime('%H:%M')}] {name} dừng (mã {proc.returncode}), thử lại lần {retries[name]}", flush=True)
                else:
                    print(f"[{time.strftime('%H:%M')}] {name} lỗi {MAX_RETRIES + 1} lần, bỏ qua. Xem {run.log_dir}/console.txt", flush=True)
    except KeyboardInterrupt:
        print("Đang dừng; chạy lại đúng lệnh này để tiếp tục.")
        for _, proc, _ in active.values():
            proc.terminate()
    finally:
        for _, proc, log in active.values():
            proc.wait()
            log.close()


def cmd_status(args) -> None:
    for r in RUNS + [BASELINE]:
        st = r.state()
        step = st["global_step"] if st else 0
        print(f"{r.name:15s} {'XONG' if r.finished else 'dở' if st else 'chưa chạy':10s} step {step:>8d}  {r.question}")


def eval_path(name: str) -> str:
    return os.path.join(ROOT, "logs", "ablation", f"eval_{name}.json")


def cmd_eval(args) -> None:
    os.chdir(ROOT)
    os.makedirs(os.path.dirname(eval_path("x")), exist_ok=True)
    jobs = [("ref_111s", ["--model", "models/best_agent.pt"]),
            ("ghost", ["--agent", "optimal", "--vehicle", "gt3_anchor"])]
    jobs += [(r.name, ["--model", os.path.join(r.save_dir, "best_agent.pt")])
             for r in RUNS + [BASELINE] if os.path.exists(os.path.join(ROOT, r.save_dir, "best_agent.pt"))]
    for name, extra in jobs:
        if os.path.exists(eval_path(name)) and not args.force:
            continue
        print(f">> eval {name}", flush=True)
        subprocess.run([sys.executable, "training/evaluate.py", "--episodes", str(args.episodes),
                        "--out", eval_path(name)] + extra, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def curve_stats(metrics: dict, crop: int) -> dict:
    """Chỉ số từ log train, chỉ tính các eval có step <= crop (mỗi eval 3 episode)"""
    idx = [i for i, s in enumerate(metrics["steps"]) if s <= crop]
    comp = [metrics["eval_completion"][i] for i in idx]
    crash = [metrics["crash_rate"][i] for i in idx]
    speed = [metrics["eval_speed"][i] for i in idx]
    steps = [metrics["steps"][i] for i in idx]
    clean = [i for i in range(len(idx)) if crash[i] == 0 and comp[i] >= 98.0]
    tail = [i for i in range(len(idx)) if steps[i] > crop - 500000]
    return {
        "first_clean": steps[clean[0]] if clean else None,
        "clean_evals": len(clean),
        "n_evals": len(idx),
        "tail_completion": sum(comp[i] for i in tail) / max(1, len(tail)),
        "best_clean_speed": max((speed[i] for i in clean), default=None),
    }


def fmt_eval(name: str) -> List[str]:
    if not os.path.exists(eval_path(name)):
        return ["–", "–", "–"]
    with open(eval_path(name), "r", encoding="utf-8") as f:
        d = json.load(f)
    lap = f"{d['lap_time_mean']:.1f} ± {d['lap_time_std']:.1f} (n={len(d['lap_times'])})" if d.get("lap_time_std") is not None else "–"
    return [f"{round(d['success_rate'] * d['num_episodes'])}/{d['num_episodes']}", lap, f"{d['mean_speed_kmh']:.1f}"]


def cmd_table(args) -> None:
    os.chdir(ROOT)
    logs = {}
    for r in RUNS + [BASELINE]:
        path = os.path.join(r.log_dir, "training_metrics.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                logs[r.name] = json.load(f)
    if not logs:
        raise SystemExit("Chưa có run nào.")
    crop = CROP_STEP
    for name in [n for n, m in logs.items() if m["steps"][-1] < crop]:
        print(f"Bỏ {name}: chưa tới step {crop} (đang ở {logs.pop(name)['steps'][-1]})")
    ref = json.loads(subprocess.check_output(["git", "show", f"{REF_COMMIT}:logs/training_metrics.json"]))
    rows = [("ref_111s (log cắt tại mốc chung)", "Công thức gốc", curve_stats(ref, crop), "ref_111s")]
    for r in RUNS + [BASELINE]:
        if r.name in logs:
            rows.append((r.name, r.question, curve_stats(logs[r.name], crop), r.name))
    lines = [f"Log train cắt tại step {crop}; eval 3 episode mỗi 10.240 step. Eval cuối: best_agent.pt, 20 episode trên env ee99f9d.", "",
             "| Run | Thay đổi | Eval sạch đầu tiên | Số eval sạch | Completion 0.5M cuối (%) | Tốc độ eval sạch cao nhất (km/h) | Eval cuối: sạch | Lap (s) | Tốc độ TB |",
             "| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |"]
    for name, q, c, key in rows:
        sp = f"{c['best_clean_speed']:.1f}" if c["best_clean_speed"] else "–"
        lines.append(f"| {name} | {q} | {c['first_clean'] or '–'} | {c['clean_evals']}/{c['n_evals']} | {c['tail_completion']:.1f} | {sp} | " + " | ".join(fmt_eval(key)) + " |")
    lines += [f"| ghost hiệu chỉnh | Mốc so sánh | – | – | – | – | " + " | ".join(fmt_eval("ghost")) + " |", "",
              "Lưu ý: best_agent.pt của ref_111s được xác định là bản lưu ở mốc cắt qua log tốc độ; log gốc không ghi lap time nên không loại trừ hoàn toàn một lần lưu muộn hơn theo luật lap time."]
    os.makedirs("logs/ablation", exist_ok=True)
    with open("logs/ablation/table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--runs", type=str, default=None, help="Tên run, cách nhau bởi dấu phẩy")
    p.add_argument("--include-baseline", action="store_true", help="Chạy thêm đối chứng công thức gốc ở cuối hàng đợi (tùy chọn: kiểm tra độ lặp lại)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("eval")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_eval)
    sub.add_parser("table").set_defaults(func=cmd_table)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
