"""
Hiệu chuẩn động lực học xe trên Bullet (MetaDrive) - WI-11
=========================================================
Đo trực tiếp vật lý của xe, KHÔNG đi qua MetaRacingEnv.step():
- MetaRacingEnv.step() kết thúc episode khi xe ra khỏi đường đua, làm cắt ngang các bài đo
  (phanh, chạy vòng tròn). Ở đây gọi thẳng env.env.step([steer, pedal]) của MetaDrive;
  địa hình ngoài đường đua là mặt phẳng có va chạm nên xe chạy tự do được.
- Không có slew limiter của BaseRaceVehicle (bài đo tự điều khiển góc lái chậm).

Các bài đo:
1. Tham số thực sự được áp vào Bullet (mass, friction slip, engine, brake, max speed).
2. Tăng tốc hết ga từ đứng yên: mốc 0-100/0-200/0-250, quãng đường tới 250 km/h, a_x theo dải tốc độ.
3. Phanh hết cỡ từ 100/200/250 km/h: quãng phanh, giảm tốc trung bình.
4. Bám ngang (constant speed, ramp steer): giữ tốc độ không đổi, tăng dần góc lái;
   a_y = v * yaw_rate. Đỉnh a_y ổn định = giới hạn bám ngang.
5. Ổn định ở tốc độ trần: giữ 250+ km/h, lái hình sin nhẹ; đo dao động z, roll, pitch.

Chạy:  python scripts/calibrate_vehicle.py [--quick]
Xuất:  logs/calibration/calibration_results.json (số đo thô).
Kết luận viết tay trong knowledge/vehicle_and_engine_physics.md (script KHÔNG tự viết kết luận).
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import GT3RaceCar, GT3RealCar, BaseRaceVehicle, VehicleConfig

G = 9.81


class FreeDriveRig:
    """Bộ đo: spawn xe qua MetaRacingEnv, sau đó điều khiển trực tiếp xe Bullet (bỏ qua luật track limit)."""

    def __init__(self, car: BaseRaceVehicle, map_type: str):
        self.env = MetaRacingEnv(map_type=map_type, vehicle=car, use_render=False)
        self.dt = float(self.env.dt)

    def reset(self):
        self.env.reset(options={"start_mode": "standing"})
        self.raw = self.env.env.agent
        self.prev_heading = float(self.raw.heading_theta)

    def step(self, steer: float, pedal: float) -> Dict[str, float]:
        self.env.env.step([float(np.clip(steer, -1, 1)), float(np.clip(pedal, -1, 1))])
        vel = np.array(self.raw.velocity[:2], dtype=np.float64)
        v = float(np.linalg.norm(vel))
        h = float(self.raw.heading_theta)
        dh = (h - self.prev_heading + np.pi) % (2 * np.pi) - np.pi
        self.prev_heading = h
        fwd = np.array([np.cos(h), np.sin(h)])
        beta = float(np.arctan2(fwd[0] * vel[1] - fwd[1] * vel[0], max(1e-3, float(np.dot(fwd, vel)))))
        hpr = self.raw.origin.getHpr()
        return {
            "v": v,
            "yaw_rate": dh / self.dt,
            "beta": beta,
            "x": float(self.raw.position[0]),
            "y": float(self.raw.position[1]),
            "z": float(self.raw.origin.getZ()),
            "pitch": float(hpr[1]),
            "roll": float(hpr[2]),
        }

    def accelerate_to(self, v_target_kmh: float, max_steps: int = 600) -> Dict[str, float]:
        s = {"v": 0.0}
        for _ in range(max_steps):
            s = self.step(0.0, 1.0)
            if s["v"] * 3.6 >= v_target_kmh:
                break
        return s

    def close(self):
        self.env.close()


def read_applied_params(rig: FreeDriveRig, car: BaseRaceVehicle) -> Dict[str, Any]:
    raw = rig.raw
    return {
        "cfg_mass": car.cfg.mass,
        "bullet_mass": float(raw.body.getMass()),
        "cfg_friction": car.cfg.wheel_friction,
        "bullet_friction_slip_FL_FR_RL_RR": [round(float(raw.system.getWheel(k).getFrictionSlip()), 4) for k in range(raw.system.getNumWheels())],
        "engine_force_per_wheel": float(raw.config["max_engine_force"]),
        "brake_force_per_wheel": float(raw.config["max_brake_force"]),
        "max_speed_km_h": float(raw.max_speed_km_h),
        "max_steering_deg": float(raw.max_steering),
    }


def test_acceleration(rig: FreeDriveRig, max_steps: int = 500) -> Dict[str, Any]:
    rig.reset()
    marks = {100: None, 200: None, 250: None}
    dist_250 = None
    x0, y0 = float(rig.raw.position[0]), float(rig.raw.position[1])
    vs: List[float] = []
    top = 0.0
    for i in range(max_steps):
        s = rig.step(0.0, 1.0)
        vs.append(s["v"])
        kmh = s["v"] * 3.6
        top = max(top, kmh)
        for m in marks:
            if marks[m] is None and kmh >= m:
                marks[m] = round((i + 1) * rig.dt, 2)
                if m == 250:
                    dist_250 = float(np.hypot(s["x"] - x0, s["y"] - y0))
    vs_arr = np.array(vs)
    ax = np.diff(np.concatenate([[0.0], vs_arr])) / rig.dt
    bands = {}
    for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250)]:
        mask = (vs_arr * 3.6 >= lo) & (vs_arr * 3.6 < hi)
        if mask.any():
            bands[f"{lo}-{hi}"] = round(float(ax[mask].mean()), 3)
    return {
        "t_0_100_s": marks[100],
        "t_0_200_s": marks[200],
        "t_0_250_s": marks[250],
        "dist_0_250_m": None if dist_250 is None else round(dist_250, 1),
        "top_speed_kmh": round(top, 1),
        "ax_mean_by_band_ms2": bands,
    }


def test_braking(rig: FreeDriveRig, entry_kmh: float) -> Dict[str, Any]:
    rig.reset()
    s = rig.accelerate_to(entry_kmh)
    v0 = s["v"]
    x0, y0 = s["x"], s["y"]
    steps = 0
    for _ in range(400):
        s = rig.step(0.0, -1.0)
        steps += 1
        if s["v"] < 0.5:
            break
    dist = float(np.hypot(s["x"] - x0, s["y"] - y0))
    t = steps * rig.dt
    return {
        "entry_kmh": round(v0 * 3.6, 1),
        "stop_dist_m": round(dist, 1),
        "stop_time_s": round(t, 2),
        "decel_from_dist_g": round(v0 ** 2 / (2 * max(dist, 1e-3)) / G, 3),
    }


def test_lateral(rig: FreeDriveRig, speed_kmh: float, ramp_steps: int = 250) -> Dict[str, Any]:
    """Constant-speed ramp-steer: giữ tốc độ, tăng góc lái tuyến tính 0 -> 1 trong ramp_steps bước."""
    rig.reset()
    rig.accelerate_to(speed_kmh)
    v_ref = speed_kmh / 3.6
    hist = []
    v_now = float(np.linalg.norm(rig.raw.velocity[:2]))
    for i in range(ramp_steps):
        steer = (i + 1) / ramp_steps
        # P controller giữ tốc độ theo vận tốc bước trước
        pedal = float(np.clip(0.8 * (v_ref - v_now), -1.0, 1.0))
        s = rig.step(steer, pedal)
        v_now = s["v"]
        a_y = s["v"] * s["yaw_rate"]
        hist.append((steer, s["v"], a_y, s["beta"]))
        if abs(s["beta"]) > 0.35:  # xoay đuôi / mất lái
            break
    arr = np.array(hist)
    a_y_s = np.convolve(arr[:, 2], np.ones(5) / 5, mode="valid") if len(arr) >= 5 else arr[:, 2]
    k = int(np.argmax(np.abs(a_y_s)))
    k_raw = min(k + 2, len(arr) - 1)
    return {
        "speed_kmh": speed_kmh,
        "peak_ay_g": round(float(abs(a_y_s[k])) / G, 3),
        "steer_at_peak": round(float(arr[k_raw, 0]), 3),
        "v_at_peak_kmh": round(float(arr[k_raw, 1]) * 3.6, 1),
        "beta_at_peak_deg": round(float(np.degrees(arr[k_raw, 3])), 2),
        "spun": bool(abs(arr[-1, 3]) > 0.35),
        "radius_at_peak_m": round(float(arr[k_raw, 1] ** 2 / max(abs(a_y_s[k]), 1e-3)), 1),
    }


def test_stability(rig: FreeDriveRig, speed_kmh: float = 250.0, steps: int = 150) -> Dict[str, Any]:
    rig.reset()
    s = rig.accelerate_to(speed_kmh)
    z0 = s["z"]
    zs, rolls, pitches, vs = [], [], [], []
    for i in range(steps):
        steer = 0.04 * np.sin(2 * np.pi * i / 30.0)
        s = rig.step(steer, 1.0)
        zs.append(abs(s["z"] - z0))
        rolls.append(abs(s["roll"]))
        pitches.append(abs(s["pitch"]))
        vs.append(s["v"] * 3.6)
    return {
        "speed_reached_kmh": round(float(np.max(vs)), 1),
        "max_dz_m": round(float(np.max(zs)), 4),
        "max_roll_deg": round(float(np.max(rolls)), 3),
        "max_pitch_deg": round(float(np.max(pitches)), 3),
    }


def run_car(name: str, car: BaseRaceVehicle, map_type: str, lateral_speeds: List[float], quick: bool) -> Dict[str, Any]:
    print(f"\n>>> {name} | map={map_type}")
    rig = FreeDriveRig(car, map_type)
    try:
        rig.reset()
        res: Dict[str, Any] = {"name": name, "map": map_type, "cfg": car.cfg.__dict__.copy()}
        res["applied"] = read_applied_params(rig, car)
        print("    applied:", res["applied"])
        res["accel"] = test_acceleration(rig)
        print("    accel:", res["accel"])
        v_cap = car.cfg.max_speed_km_h
        entries = [e for e in (100.0, 200.0, 250.0) if e < v_cap - 2]
        res["braking"] = [test_braking(rig, e) for e in entries]
        for b in res["braking"]:
            print("    brake:", b)
        res["lateral"] = [test_lateral(rig, v) for v in lateral_speeds if v < v_cap - 2]
        for l in res["lateral"]:
            print("    lateral:", l)
        if not quick:
            res["stability"] = test_stability(rig, speed_kmh=min(250.0, v_cap - 5))
            print("    stability:", res["stability"])
        return res
    finally:
        rig.close()


def real_variant(**overrides) -> GT3RealCar:
    base = GT3RealCar().cfg.__dict__.copy()
    base.update(overrides)
    return GT3RealCar(VehicleConfig(**base))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Bỏ bài ổn định và bớt cấu hình quét")
    parser.add_argument("--out", type=str, default="logs/calibration/calibration_results.json")
    args = parser.parse_args()

    speeds_small = [40.0, 70.0, 100.0]
    speeds_full = [60.0, 100.0, 150.0, 200.0]

    runs = [
        ("GT3RaceCar (map half, tham chiếu Phase 1)", GT3RaceCar(), "silverstone_half", speeds_small),
        ("GT3RealCar baseline", GT3RealCar(), "silverstone_full", speeds_full),
    ]
    if not args.quick:
        runs += [
            ("GT3RealCar friction 1.35", real_variant(wheel_friction=1.35), "silverstone_full", speeds_full),
            ("GT3RealCar friction 1.95", real_variant(wheel_friction=1.95), "silverstone_full", speeds_full),
            ("GT3RealCar brake x0.5", real_variant(max_brake_force=GT3RealCar().cfg.max_brake_force * 0.5), "silverstone_full", [100.0]),
            ("GT3RealCar engine 2000", real_variant(max_engine_force=2000.0), "silverstone_full", [100.0]),
        ]

    results = [run_car(n, c, m, s, args.quick) for n, c, m, s in runs]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n>> Đã lưu số đo thô: {args.out}")


if __name__ == "__main__":
    main()
