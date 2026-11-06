"""Mô hình lỗi có kiểm soát dùng chung cho T10 và T11."""
import numpy as np


class SmartMistakes:
    """Tạo lỗi có kiểm soát: chỉ khi có xe sau đủ gần để tận dụng và xe trước đang ở đoạn thẳng."""
    start_step = None

    def __init__(self, env, gap_s=1.5, duration=1.0, max_per_car=2, cooldown=200, ahead_m=80):
        self.gap_s, self.steps, self.max_per_car, self.cooldown, self.ahead_m = gap_s, int(duration / env.dt), max_per_car, cooldown, ahead_m
        self.used = np.zeros(env.num_cars, dtype=int)
        self.last = -np.ones(env.num_cars, dtype=int) * 10000
        self.until = np.zeros(env.num_cars, dtype=int)
        self.triggered = 0

    def apply(self, env, step, actions):
        for i, st in enumerate(env.racers):
            if not st.active:
                continue
            if step < self.until[i]:
                actions[i][1:] = [-1.0, 0.0]
                continue
            if step < 150 or self.used[i] >= self.max_per_car or step - self.last[i] < self.cooldown:
                continue
            idx = st.current_track_idx
            if np.any(np.abs(env.track_kappa[np.arange(idx, idx + self.ahead_m) % env.num_track_points]) > 0.0):
                continue
            s_i = env.race_distance(st)
            v = max(float(np.linalg.norm(st.car.velocity)), 1.0)
            followers = [o for o in env.racers if o is not st and o.active and 0.0 < (s_i - env.race_distance(o)) % env.circuit_length < self.gap_s * v]
            if not followers:
                continue
            self.used[i] += 1
            self.last[i] = step
            self.until[i] = step + self.steps
            self.triggered += 1
            actions[i][1:] = [-1.0, 0.0]
