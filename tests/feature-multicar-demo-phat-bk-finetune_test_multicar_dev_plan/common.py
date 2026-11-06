"""Công cụ đo dùng chung cho các test định hướng plan multicar. Không sửa env, chỉ quan sát và can thiệp action từ ngoài."""

import dataclasses
import json
import os
import sys

import numpy as np
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from game_env.multicar_env import MultiCarRacingEnv, BASE_OBS_DIM
from agents.ppo_agent import PPOAgent
from training.train_multicar import warm_start_from_single, policy_act

MODELS = {
    "ref_111s": "models/ref_111s/best_agent.pt",
    "pilot_100k": "models/multicar_clean_pilot/agent_100k.pt",
    "pilot_200k": "models/multicar_clean_pilot/agent_200k.pt",
    "pilot_300k": "models/multicar_clean_pilot/agent_300k.pt",
}
HOLD_STEPS = 30       # Xe vượt phải giữ được vị trí 3s mới tính là vượt thật
START_PHASE_STEPS = 150  # 15s đầu là tranh vị trí sau xuất phát, tính riêng


def make_env(num_cars=6, lane=3, grid="staggered", respawn=False, lat_norm=None, max_laps=1, max_steps=3000):
    torch.set_num_threads(1)
    if os.environ.get("MC_BRAKE") or os.environ.get("MC_MU_F") or os.environ.get("MC_ENGINE"):
        # Thử vật lý khác từ biến môi trường mà không sửa code xe
        import game_env.multicar_env as menv
        from game_env.vehicle import GT3RaceCar
        base = GT3RaceCar().cfg
        cfg = dataclasses.replace(base, max_brake_force=float(os.environ.get("MC_BRAKE", base.max_brake_force)),
                                  front_wheel_friction=float(os.environ.get("MC_MU_F", base.front_wheel_friction)),
                                  max_engine_force=float(os.environ.get("MC_ENGINE", base.max_engine_force)))
        menv.GT3RaceCar = lambda: GT3RaceCar(dataclasses.replace(cfg))
    env = MultiCarRacingEnv(num_cars=num_cars, lane_num=lane, grid=grid, respawn=respawn, start_seed=42, max_laps=max_laps, max_steps=max_steps)
    if lat_norm is not None:
        env.preset = dataclasses.replace(env.preset, lat_norm=float(lat_norm))
    return env


def load_policy(env, name):
    agent = PPOAgent(state_dim=env.obs_dim, action_dim=3, logstd_max=[0.0, -0.7, -0.7], device="cpu")
    path = MODELS[name]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["model_state_dict"]["actor_mean.0.weight"].shape[1] == env.obs_dim:
        agent.load(path)
    else:
        warm_start_from_single(agent, path, base_dim=BASE_OBS_DIM)
    return agent


class Mistake:
    """Lỗi của một xe: khi xe đạt quãng đường `at_m`, ghi đè action trong `duration` giây."""

    def __init__(self, car, at_m, kind, duration):
        self.car, self.at_m, self.kind, self.duration = car, float(at_m), kind, float(duration)
        self.start_step = None
        self.done = False

    def apply(self, env, step, actions):
        if self.done:
            return
        st = env.racers[self.car]
        if not st.active:
            self.done = True
            return
        if self.start_step is None:
            if env.race_distance(st) < self.at_m:
                return
            self.start_step = step
        if step - self.start_step >= int(round(self.duration / env.dt)):
            self.done = True
            return
        if self.kind == "coast":
            actions[self.car][1:] = [-1.0, -1.0]      # Nhả hết ga, chưa phanh
        elif self.kind == "brake50":
            actions[self.car][1:] = [-1.0, 0.0]       # Phanh 50%
        else:
            raise ValueError(self.kind)


def run_race(env, agent, seed=0, deterministic=True, mistakes=(), stop_after_mistake=None, on_step=None, act_fn=None):
    """Chạy 1 cuộc đua, trả về số đo. Nhiễu lái nhỏ 10 bước đầu theo seed để các cuộc đua khác nhau."""
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(seed)
    obs, _ = env.reset()
    n = env.num_cars
    actions = np.zeros((n, 3), dtype=np.float32)
    held, pending, outs = [], [], []
    start_passes = 0
    last_contact = [-999] * n
    contact_types = {"rear": 0, "side": 0}
    contact_episodes = 0  # Gộp các cú chạm cách nhau dưới 1s của cùng một xe thành một lần
    prev_in_contact = [False] * n
    follow_dy, follow_closing = [], []
    step = 0
    first_mistake_step = None
    while True:
        active = env.active_mask()
        if active.any():
            if act_fn is not None:
                act = act_fn(env, obs, active, step)
            else:
                act, _, _, _ = policy_act(agent, obs[active], deterministic)
            if step < 10:
                act[:, 0] += rng.normal(0.0, 0.02, size=act.shape[0])
            actions[active] = act
        for m in mistakes:
            m.apply(env, step, actions)
            if m.start_step is not None and first_mistake_step is None:
                first_mistake_step = m.start_step
        before = [env.race_distance(s) for s in env.racers]
        was_active = [s.active for s in env.racers]
        obs, _, dones, truncated, _ = env.step(actions)
        step += 1
        after = [env.race_distance(s) for s in env.racers]

        for i, st in enumerate(env.racers):
            if st.in_contact:
                if not prev_in_contact[i]:
                    if step - last_contact[i] > 10:
                        contact_episodes += 1
                    # Xếp loại cú chạm theo xe gần nhất: lệch dọc lớn là đâm đuôi, nhỏ là cọ sườn
                    h = st.car.heading_theta
                    fwd = np.array([np.cos(h), np.sin(h)])
                    near = min((o for o in env.racers if o is not st and o.active),
                               key=lambda o: np.linalg.norm(o.car.position - st.car.position), default=None)
                    if near is not None:
                        dx = float(np.dot(near.car.position - st.car.position, fwd))
                        contact_types["rear" if abs(dx) > 3.0 else "side"] += 1
                last_contact[i] = step
            prev_in_contact[i] = st.in_contact
            if was_active[i] and dones[i] and st.termination_reason != "finished":
                outs.append({"car": i, "reason": st.termination_reason, "frac": round(before[i] / env.circuit_length, 3),
                             "t": step, "contact_3s": step - last_contact[i] <= 30, "kmh": round(st.speed_km_h, 1)})
            # Hình học khi bám đuôi: xe trước trong 15m
            if st.active:
                h = st.car.heading_theta
                fwd = np.array([np.cos(h), np.sin(h)])
                lat = np.array([-np.sin(h), np.cos(h)])
                for o in env.racers:
                    if o is st or not o.active:
                        continue
                    rel = o.car.position - st.car.position
                    dx = float(rel @ fwd)
                    if 3.0 < dx < 15.0:
                        follow_dy.append(abs(float(rel @ lat)))
                        follow_closing.append(-float((o.car.velocity - st.car.velocity) @ fwd))

        # Phát hiện pha vượt: A đang sau B thì lên trước, xác nhận sau HOLD_STEPS
        for a in range(n):
            for b in range(n):
                if a == b or not (was_active[a] and was_active[b]):
                    continue
                if before[a] < before[b] and before[b] - before[a] < 30.0 and after[a] > after[b]:
                    pending.append((a, b, step, bool(env.racers[a].in_contact)))
        for p in list(pending):
            a, b, t0, touched = p
            if step - t0 < HOLD_STEPS:
                continue
            pending.remove(p)
            sa, sb = env.racers[a], env.racers[b]
            if not (sa.active or sa.finished) or not (sb.active or sb.finished):
                continue
            if t0 - sb.respawn_step <= 300:  # Không tính pha vượt qua xe vừa hồi sinh trong 30s
                continue
            if env.race_distance(sa) > env.race_distance(sb):
                if t0 <= START_PHASE_STEPS:
                    start_passes += 1
                else:
                    idx = sa.current_track_idx
                    held.append({"a": a, "b": b, "t": t0, "frac": round(after[a] / env.circuit_length, 3),
                                 "corner": bool(abs(env.track_kappa[idx]) > 0.0), "touched": touched})
        if on_step is not None:
            on_step(env, obs, step)
        stop = truncated or not env.active_mask().any()
        if stop_after_mistake is not None and first_mistake_step is not None and step - first_mistake_step >= stop_after_mistake:
            stop = True
        if stop:
            break
    laps = [s.lap_times[-1] for s in env.racers if s.finished]
    style = env.compute_driving_style_metrics()
    return {
        "steps": step,
        "finished": int(sum(s.finished for s in env.racers)),
        "outs": outs,
        "contacts": int(sum(s.contacts for s in env.racers)),
        "contact_types": contact_types,
        "contact_episodes": contact_episodes,
        "start_passes": start_passes,
        "held": held,
        "env_overtakes": int(sum(s.overtakes for s in env.racers)),
        "mean_lap": float(np.mean(laps)) if laps else 0.0,
        "best_lap": float(np.min(laps)) if laps else 0.0,
        "pedal_sw": style["pedal_sw_per_100"],
        "steer_rev": style["steer_rev_per_100"],
        "beta": style["beta_p95_deg"],
        "follow_dy": [round(x, 2) for x in follow_dy[::5]],
        "follow_closing": [round(x, 2) for x in follow_closing[::5]],
    }


def save(name, payload):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"saved {name}", flush=True)
