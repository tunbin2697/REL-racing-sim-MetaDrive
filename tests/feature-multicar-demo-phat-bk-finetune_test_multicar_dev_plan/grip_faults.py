"""Lỗi lái thật hơn: tạm giảm độ bám lốp thay vì ghi đè action, để xe tự trượt và ref_111s phải tự cứu.
Cả 3 lỗi đều xảy ra quanh điểm vào cua, nơi tay đua thật hay mắc lỗi:
- lockup: trong vùng phanh trước cua, cầu trước mất bám và phanh gắt thêm, xe trôi thẳng và lỡ điểm vào cua
- understeer: sát điểm vào cua, cầu trước mất bám, xe đi rộng
- oversteer: vừa vào cua, cầu sau mất bám, đuôi xe văng
"""
import numpy as np

KINDS = ("lockup", "understeer", "oversteer")
# Mức độ: (hệ số nhân độ bám, thời gian trượt giây). Mức "ext" có thêm 1 pha tài xế nhả ga và rà phanh để cứu xe.
RECOVERY_S = {"ext": 1.2}
SEVERITY = {
    "lockup": {"light": (0.60, 0.6), "mid": (0.45, 0.8), "hard": (0.30, 1.0), "ext": (0.30, 1.0)},
    "understeer": {"light": (0.60, 0.6), "mid": (0.45, 0.8), "hard": (0.30, 1.0), "ext": (0.30, 1.0)},
    "oversteer": {"light": (0.75, 0.4), "mid": (0.60, 0.6), "hard": (0.45, 0.8), "ext": (0.60, 0.6)},
}


def spot_ok(env, idx, kind):
    """Vị trí trên đường có phù hợp với loại lỗi này không."""
    n = env.num_track_points
    corner = np.abs(env.track_kappa) > 0.0
    here = corner[idx]
    if kind == "lockup":
        return (not here) and corner[np.arange(idx + 20, idx + 60) % n].any()
    if kind == "understeer":
        return (not here) and corner[np.arange(idx + 3, idx + 20) % n].any()
    return here and not corner[np.arange(idx - 15, idx) % n].all()


def set_grip(st, scale_front=1.0, scale_rear=1.0):
    system = st.raw_vehicle.system
    cfg = st.car.cfg
    mu_front = cfg.front_wheel_friction if cfg.front_wheel_friction is not None else cfg.wheel_friction
    for k in range(system.getNumWheels()):
        wheel = system.getWheel(k)
        wheel.setFrictionSlip(float(mu_front * scale_front if wheel.isFrontWheel() else cfg.wheel_friction * scale_rear))


class GripFault:
    """Một lỗi đặt trước cho một xe, xảy ra ở vị trí phù hợp đầu tiên sau quãng đường at_m."""
    start_step = None

    def __init__(self, car, at_m, kind, severity="mid"):
        self.car, self.at_m, self.kind = car, float(at_m), kind
        self.scale, self.duration = SEVERITY[kind][severity]
        self.recovery = RECOVERY_S.get(severity, 0.0)
        self.begin, self.finished = None, False

    def apply(self, env, step, actions):
        if self.finished:
            return
        st = env.racers[self.car]
        if not st.active:
            # Xe văng giữa lúc mất bám: phải trả lại độ bám, nếu không xe hồi sinh sẽ chạy với lốp yếu
            set_grip(st)
            self.finished = True
            return
        if self.begin is None:
            if env.race_distance(st) < self.at_m or st.speed_km_h < 50.0 or not spot_ok(env, st.current_track_idx, self.kind):
                return
            self.begin = step
            set_grip(st, self.scale if self.kind != "oversteer" else 1.0, self.scale if self.kind == "oversteer" else 1.0)
        slide_steps, rec_steps = int(self.duration / env.dt), int(self.recovery / env.dt)
        if step - self.begin == slide_steps:
            set_grip(st)
        if step - self.begin >= slide_steps + rec_steps:
            self.finished = True
            return
        if step - self.begin >= slide_steps:
            actions[self.car][1:] = [-1.0, -0.4]  # Nhả ga, rà phanh 30% để lấy lại xe
            return
        if self.kind == "lockup":
            actions[self.car][1:] = [-1.0, 0.6]  # Phanh 80%


class SmartGripFaults:
    """Tạo lỗi khi có xe sau đủ gần để tận dụng, ở vị trí phù hợp, luân phiên 3 loại lỗi."""
    start_step = None

    def __init__(self, env, gap_s=5.0, severity="ext", per_lap=0.5, cooldown=300, seed=0, kinds=KINDS):
        # Ngân sách lỗi tăng dần theo vòng, để lỗi trải đều cả chặng thay vì dùng hết ở vòng đầu
        self.env, self.gap_s, self.severity, self.per_lap, self.cooldown = env, gap_s, severity, per_lap, cooldown
        self.rng = np.random.default_rng(seed)
        self.kinds = kinds
        n = env.num_cars
        self.used, self.last = np.zeros(n, dtype=int), -np.ones(n, dtype=int) * 10000
        self.running = [None] * n
        self.pick = [self.rng.choice(kinds) for _ in range(n)]
        self.log = []

    @property
    def triggered(self):
        return len(self.log)

    def apply(self, env, step, actions):
        for i, st in enumerate(env.racers):
            fault = self.running[i]
            if fault is not None:
                fault.apply(env, step, actions)
                if fault.finished:
                    self.running[i] = None
                continue
            lap = int(max(env.race_distance(st), 0.0) // env.circuit_length)
            if not st.active or step < 150 or self.used[i] >= (lap + 1) * self.per_lap or step - self.last[i] < self.cooldown or st.speed_km_h < 50.0:
                continue
            kind = self.pick[i]
            if not spot_ok(env, st.current_track_idx, kind):
                continue
            s_i = env.race_distance(st)
            v = max(float(np.linalg.norm(st.car.velocity)), 1.0)
            if not any(o is not st and o.active and 0.0 < (s_i - env.race_distance(o)) % env.circuit_length < self.gap_s * v for o in env.racers):
                continue
            fault = GripFault(i, 0.0, kind, self.severity)
            fault.apply(env, step, actions)
            self.running[i] = fault
            self.used[i] += 1
            self.last[i] = step
            self.pick[i] = self.rng.choice(self.kinds)
            self.log.append({"t": step, "car": i, "kind": str(kind)})


def big_straights(env, min_len_m=150.0):
    """Các đoạn thẳng dài theo thứ tự trên vòng đua, dạng (s_đầu, s_cuối) tính bằng mét từ vạch xuất phát."""
    straight = np.abs(env.track_kappa) == 0.0
    n = env.num_track_points
    first_corner = int(np.argmax(~straight))
    runs, start = [], None
    for k in range(first_corner, first_corner + n + 1):
        if straight[k % n]:
            if start is None:
                start = k
        elif start is not None:
            length = float(k - start)
            if length >= min_len_m:
                runs.append((float(env.track_s[start % n]), float(env.track_s[start % n]) + length))
            start = None
    return sorted(runs)


class TopFieldFaults:
    """Lỗi tập trung vào nhóm đầu để đoàn xe co lại: chưa có lỗi cho tới hết đoạn thẳng lớn thứ 2,
    mỗi lúc một lỗi, cách nhau ngẫu nhiên, xe hạng càng cao càng dễ mắc."""
    start_step = None
    TOP_WEIGHTS = (0.5, 0.3, 0.2)

    def __init__(self, env, severity="ext", interval_s=(20.0, 40.0), seed=0, kinds=KINDS):
        self.severity, self.kinds = severity, kinds
        self.interval = (int(interval_s[0] / env.dt), int(interval_s[1] / env.dt))
        self.rng = np.random.default_rng(seed)
        straights = big_straights(env)
        self.start_after_m = straights[1][1] if len(straights) >= 2 else env.circuit_length * 0.3
        self.next_step, self.running, self.pending = 0, None, None
        self.log = []

    @property
    def triggered(self):
        return len(self.log)

    def _order(self, env):
        cars = [i for i, st in enumerate(env.racers) if st.active]
        return sorted(cars, key=lambda i: -env.race_distance(env.racers[i]))

    def apply(self, env, step, actions):
        if self.running is not None:
            self.running.apply(env, step, actions)
            if self.running.finished:
                self.running = None
                self.next_step = step + int(self.rng.integers(*self.interval))
            return
        if step < self.next_step:
            return
        order = self._order(env)
        top = order[:len(self.TOP_WEIGHTS)]
        if not top or env.race_distance(env.racers[top[0]]) < self.start_after_m:
            return
        if self.pending is None or self.pending[0] not in top:
            w = np.array(self.TOP_WEIGHTS[:len(top)])
            self.pending = (int(self.rng.choice(top, p=w / w.sum())), str(self.rng.choice(self.kinds)))
        car, kind = self.pending
        st = env.racers[car]
        if st.speed_km_h < 50.0 or env.race_distance(st) < self.start_after_m or not spot_ok(env, st.current_track_idx, kind):
            return
        fault = GripFault(car, 0.0, kind, self.severity)
        fault.apply(env, step, actions)
        self.running, self.pending = fault, None
        self.log.append({"t": step, "car": car, "kind": kind, "rank": order.index(car), "dist": float(env.race_distance(st))})
