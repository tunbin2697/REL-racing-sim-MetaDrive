"""Bản mẫu adapter MPC: ref_111s vẫn lái, planner chỉ chọn lệch line và mức phanh khi có xe gần.

Mỗi bước, với mỗi xe có đối thủ trong tầm:
1. Dự đoán 2.5s tới trên hệ tọa độ dọc đường (s, n). Tốc độ và line danh nghĩa lấy từ một vòng chạy một mình của ref_111s.
   Đối thủ được giả định giữ tỉ lệ tốc độ so với hồ sơ và giữ độ lệch line hiện tại.
2. Thử các tổ hợp (độ lệch line đích, mức phanh), tính chi phí, chọn tổ hợp rẻ nhất, chỉ thực hiện bước đầu rồi lặp lại.
3. ref_111s nhận obs đã dịch ngang đúng bằng độ lệch, nên tự lái theo line lệch và không giằng co với planner.
Không có xe trong tầm thì độ lệch về 0 và xe lái đúng như ref_111s.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict

import numpy as np

from common import policy_act


@dataclass
class MPCConfig:
    horizon: int = 25            # 2.5s
    dt: float = 0.1
    trigger_ahead_m: float = 60.0
    trigger_behind_m: float = 12.0
    offsets: tuple = (-3.0, -1.5, 0.0, 1.5, 3.0)  # T8 offset: ref_111s bám tốt tới ±3m, văng ở ±4.5m
    # (mức giảm tốc m/s^2, action ga, action phanh); hàng đầu là không can thiệp pedal
    pedal_modes: tuple = ((0.0, None, None), (0.5, -1.0, -1.0), (4.5, -1.0, -0.5), (9.0, -1.0, 0.0), (18.0, -1.0, 1.0))
    pedal_hold_s: float = 1.0    # Giả định áp dụng mức phanh trong 1s đầu của horizon
    offset_rate: float = 1.5     # Tốc độ đổi độ lệch line (m/s)
    max_abs_line: float = 5.0    # Giới hạn |n| của tâm xe, nửa mặt đường 6.75m trừ thân xe và biên an toàn
    car_len: float = 4.6
    car_wid: float = 2.05
    w_collision: float = 200.0
    w_soft: float = 6.0
    behind_weight: float = 0.25  # Xe sau chịu trách nhiệm tránh nhiều hơn xe trước
    w_edge: float = 60.0
    w_progress: float = 1.0
    w_offset: float = 0.02
    w_switch: float = 3.0
    hold_steps: int = 10         # Giữ quyết định tối thiểu 1s để xe không lắc
    soft_len_scale: float = 1.8  # Tầm dọc của chi phí giữ khoảng cách mềm, tính theo chiều dài xe
    soft_wid_scale: float = 1.4  # Tầm ngang của chi phí đó, nhỏ hơn thì đã lệch sang line khác là hết bị phạt
    no_pedal_below_ms: float = 0.0  # Xe chậm hơn mức này ở đầu cuộc đua thì planner không được nhả ga, tránh đứng chờ trên lưới
    w_brake: float = 0.0         # Thêm chi phí khi planner phanh, để lách được ưu tiên hơn phanh theo
    # Các tùy chọn dưới đây tắt ở cấu hình gốc, T10 bật lần lượt
    start_grace_steps: int = 0   # Giai đoạn xuất phát kéo dài N bước, dùng 2 giới hạn dưới đây
    start_max_mode: int = 0      # Mức pedal mạnh nhất planner được dùng khi xuất phát (0 = không can thiệp pedal)
    start_no_offset: bool = False  # Không lệch line khi xuất phát
    corner_limit: float = 0.0    # > 0: ở gần cua chỉ được lệch line tối đa từng này mét
    corner_behind_m: float = 10.0
    corner_ahead_m: float = 50.0
    attack: bool = False         # Lớp quyết định vượt: chọn một bên và cam kết
    attack_after_steps: int = 0  # Chỉ bắt đầu vượt sau N bước
    attack_straight_m: int = 0   # > 0: chỉ bắt đầu vượt khi phía trước từng này mét là đường thẳng
    attack_range_m: float = 40.0
    attack_closing_ms: float = 1.5
    attack_hold_steps: int = 40
    w_attack: float = 1.0


# Cấu hình đã chọn sau T16: không đứng chờ khi xuất phát, ưu tiên lách hơn phanh, đi sát xe bên cạnh
BRAVE_CONFIG = dict(no_pedal_below_ms=14.0, w_brake=8.0, offset_rate=2.5, soft_wid_scale=1.0, soft_len_scale=1.2)


class MPCAdapter:
    def __init__(self, env, agent, profile: Dict[str, np.ndarray], cfg: MPCConfig = None):
        self.env, self.agent, self.cfg = env, agent, cfg or MPCConfig()
        self.v_prof, self.n_prof = profile["v"], profile["n"]
        self.reset()
        c = self.cfg
        grid = [(o, m) for o in c.offsets for m in range(len(c.pedal_modes))]
        self.cand_off = np.array([g[0] for g in grid], dtype=np.float64)
        self.cand_mode = np.array([g[1] for g in grid])
        self.cand_dec = np.array([c.pedal_modes[m][0] for m in self.cand_mode], dtype=np.float64)
        self.plan_time, self.plans, self.engaged_steps, self.total_steps = 0.0, 0, 0, 0
        self.attacks_started, self.attacks_passed = 0, 0
        self.off_limit = self._build_offset_limit()

    def _build_offset_limit(self):
        """Giới hạn độ lệch line theo từng điểm tim đường, từ độ cong ở gần điểm đó."""
        c, env = self.cfg, self.env
        limit = np.full(env.num_track_points, max(c.offsets))
        if c.corner_limit > 0.0:
            env_kappa = np.abs(env.track_kappa) > 0.0
            behind, ahead = int(c.corner_behind_m), int(c.corner_ahead_m)
            for k in np.flatnonzero(env_kappa):
                limit[np.arange(k - ahead, k + behind + 1) % env.num_track_points] = c.corner_limit
        return limit

    def reset(self):
        n = self.env.num_cars
        self.offset = np.zeros(n)
        self.target = np.zeros(n)
        self.mode = np.zeros(n, dtype=int)
        self.hold = np.zeros(n, dtype=int)
        self.t = 0
        self.attack_target = -np.ones(n, dtype=int)
        self.attack_side = np.zeros(n)
        self.attack_until = np.zeros(n, dtype=int)

    def _update_attack(self, i, s_now, v_now, n_now, active):
        """Quản lý trạng thái vượt của xe i: bắt đầu khi đang áp sát xe trước, kết thúc khi đã qua hoặc hết thời gian."""
        c, env = self.cfg, self.env
        j = self.attack_target[i]
        if j >= 0:
            ds = s_now[j] - s_now[i]
            if ds < -6.0:
                self.attacks_passed += 1
            if ds < -6.0 or not active[j] or self.t >= self.attack_until[i]:
                self.attack_target[i] = -1
            return
        if c.attack_straight_m > 0:
            idx = env.racers[i].current_track_idx
            if np.any(np.abs(env.track_kappa[np.arange(idx, idx + c.attack_straight_m) % env.num_track_points]) > 0.0):
                return
        ahead = [k for k in range(env.num_cars) if k != i and active[k] and 3.0 < s_now[k] - s_now[i] < c.attack_range_m
                 and v_now[i] - v_now[k] > c.attack_closing_ms]
        if not ahead:
            return
        j = min(ahead, key=lambda k: s_now[k] - s_now[i])
        room = {}
        for side in (-1.0, 1.0):
            line = n_now[j] + side * (c.car_wid + 0.4)
            blocked = any(k not in (i, j) and active[k] and abs(s_now[k] - s_now[j]) < 12.0 and abs(n_now[k] - line) < c.car_wid
                          for k in range(env.num_cars))
            room[side] = -1.0 if blocked or abs(line) > c.max_abs_line else c.max_abs_line - abs(line)
        side = max(room, key=lambda x: (room[x], -abs(n_now[j] + x * (c.car_wid + 0.4) - n_now[i])))
        if room[side] < 0.0:
            return
        self.attack_target[i], self.attack_side[i] = j, side
        self.attack_until[i] = self.t + c.attack_hold_steps
        self.attacks_started += 1
        self.hold[i] = 0

    def _idx(self, s):
        env = self.env
        return np.searchsorted(env.track_s, np.mod(s, env.circuit_length)) % env.num_track_points

    def _plan(self, i, s_now, v_now, n_off_now, active):
        """Trả về (độ lệch đích, mức phanh), hoặc None nếu không có đối thủ trong tầm."""
        c, env = self.cfg, self.env
        # So theo vị trí trên đường để thấy cả xe bị bắt vòng, không phải theo tổng quãng đường
        L = env.circuit_length
        s_now = s_now[i] + np.mod(s_now - s_now[i] + 0.5 * L, L) - 0.5 * L
        ids = [j for j in range(env.num_cars) if j != i and active[j]
               and -c.trigger_behind_m < s_now[j] - s_now[i] < c.trigger_ahead_m]
        if not ids:
            return None
        K = len(self.cand_off)
        s_e = np.full(K, s_now[i])
        v_e = np.full(K, v_now[i])
        off = np.full(K, self.offset[i])
        s_o = s_now[ids].astype(np.float64)
        ratio = np.clip(v_now[ids] / np.maximum(self.v_prof[self._idx(s_o)], 5.0), 0.2, 1.2)
        off_o = n_off_now[ids]
        hold_n = int(c.pedal_hold_s / c.dt)
        rate = c.offset_rate * c.dt
        cost = c.w_switch * ((self.cand_off != self.target[i]) | (self.cand_mode != self.mode[i])).astype(np.float64)
        if self.t < c.start_grace_steps:
            cost = cost + 1e6 * (self.cand_mode > c.start_max_mode)
            if c.start_no_offset:
                cost = cost + 1e6 * (self.cand_off != 0.0)
        if self.t < 300 and v_now[i] < c.no_pedal_below_ms:
            cost = cost + 1e6 * (self.cand_mode != 0)
        cost = cost + c.w_brake * self.cand_dec / 9.0
        attack_j = int(self.attack_target[i])
        soft_gain = np.array([0.0 if j == attack_j else 1.0 for j in ids])
        if attack_j in ids:
            k_j = ids.index(attack_j)
            n_goal = self.n_prof[self._idx(np.array([s_now[attack_j]]))][0] + off_o[k_j] + self.attack_side[i] * (c.car_wid + 0.4)
        s0 = s_e.copy()
        for t in range(c.horizon):
            prof_e = self.v_prof[self._idx(s_e)]
            if t < hold_n:
                braking = self.cand_dec > 0
                v_next = np.where(braking, v_e - self.cand_dec * c.dt, np.minimum(prof_e, v_e + 3.1 * c.dt))
                v_e = np.maximum(np.minimum(v_next, prof_e + 1.0), 0.0)
            else:
                v_e = np.minimum(prof_e, v_e + 3.1 * c.dt)
            s_e = s_e + v_e * c.dt
            off = off + np.clip(self.cand_off - off, -rate, rate)
            idx_e = self._idx(s_e)
            n_e = self.n_prof[idx_e] + off
            s_o = s_o + ratio * self.v_prof[self._idx(s_o)] * c.dt
            n_o = self.n_prof[self._idx(s_o)] + off_o
            ds = s_o[None, :] - s_e[:, None]
            dn = n_o[None, :] - n_e[:, None]
            weight = np.where(ds >= 0.0, 1.0, c.behind_weight)
            hit = (np.abs(ds) < c.car_len) & (np.abs(dn) < c.car_wid)
            soft = np.exp(-(ds / (c.soft_len_scale * c.car_len)) ** 2 - (dn / (c.soft_wid_scale * c.car_wid)) ** 2)
            cost += (weight * (c.w_collision * hit + c.w_soft * soft * soft_gain)).sum(axis=1)
            cost += c.w_edge * np.maximum(np.abs(n_e) - c.max_abs_line, 0.0) ** 2 + c.w_offset * off ** 2
            cost += c.w_edge * np.maximum(np.abs(off) - self.off_limit[idx_e], 0.0) ** 2
            if attack_j in ids:
                cost += c.w_attack * (n_e - n_goal) ** 2 / c.horizon
        cost -= c.w_progress * (s_e - s0)
        best = int(np.argmin(cost))
        return float(self.cand_off[best]), int(self.cand_mode[best])

    def act(self, env, obs, active):
        """Trả về action cho các xe đang active, theo đúng thứ tự trong mask."""
        import time
        c = self.cfg
        t0 = time.perf_counter()
        racers = env.racers
        s_now = np.array([env.race_distance(st) for st in racers])
        v_now = np.array([np.linalg.norm(st.car.velocity) for st in racers])
        idxs = np.array([st.current_track_idx for st in racers])
        n_now = np.array([st.lat_offset for st in racers])
        n_off_now = n_now - self.n_prof[idxs]
        shifted = np.array(obs, copy=True)
        self.cur_obs = obs
        rows = np.flatnonzero(active)
        for i in rows:
            self.total_steps += 1
            self.hold[i] = max(0, self.hold[i] - 1)
            if c.attack and self.t >= c.attack_after_steps:
                self._update_attack(i, s_now, v_now, n_now, active)
            plan = self._plan(i, s_now, v_now, n_off_now, active)
            if plan is None:
                self.target[i], self.mode[i] = 0.0, 0
            else:
                self.engaged_steps += 1
                self.plans += 1
                if self.hold[i] == 0 and (plan[0] != self.target[i] or plan[1] != self.mode[i]):
                    self.target[i], self.mode[i] = plan
                    self.hold[i] = c.hold_steps
            limit = float(self.off_limit[idxs[i]])
            goal = float(np.clip(self.target[i], -limit, limit))
            self.offset[i] += float(np.clip(goal - self.offset[i], -c.offset_rate * c.dt, c.offset_rate * c.dt))
            if abs(self.offset[i]) > 1e-6:
                shifted[i, :35] = shifted_ego_obs(env, racers[i], self.offset[i], obs[i])
        self.t += 1
        self.plan_time += time.perf_counter() - t0
        act, _, _, _ = policy_act(self.agent, shifted[rows], True)
        for k, i in enumerate(rows):
            _, thr, brk = c.pedal_modes[self.mode[i]]
            if thr is not None:
                act[k, 1], act[k, 2] = thr, brk
        return act


def shifted_ego_obs(env, st, offset, obs_row):
    """Tính lại obs 35D như thể tâm xe nằm lệch `offset` mét so với thực tế, để ref_111s lái theo line lệch."""
    th = float(env.track_theta[st.current_track_idx])
    normal = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
    raw = st.raw_vehicle
    proxy = SimpleNamespace(
        position=np.array(raw.position[:2], dtype=np.float32) - float(offset) * normal,
        heading_theta=raw.heading_theta,
        velocity=raw.velocity,
        yaw_rate=getattr(raw, "yaw_rate", 0.0),
    )
    env.current_track_idx = st.current_track_idx
    env.last_action = st.prev_action
    return env.get_compact_racing_obs(proxy, action_applied=st.last_applied, on_runoff=bool(obs_row[12] > 0.5))


def record_profile(env, agent, seed=0):
    """Ghi tốc độ và độ lệch line theo vị trí từ 1 vòng chạy một mình của ref_111s (env một xe)."""
    n_pts = None
    from common import run_race
    samples = []

    def probe(e, obs, step):
        st = e.racers[0]
        samples.append((st.current_track_idx, float(np.linalg.norm(st.car.velocity)), float(st.lat_offset)))

    result = run_race(env, agent, seed=seed, on_step=probe)
    n_pts = env.num_track_points
    idx = np.array([s[0] for s in samples])
    grid = np.arange(n_pts)
    order = np.argsort(idx)
    v = np.interp(grid, idx[order], np.array([s[1] for s in samples])[order], period=n_pts)
    n = np.interp(grid, idx[order], np.array([s[2] for s in samples])[order], period=n_pts)
    return {"v": v, "n": n}, result
