"""
MultiCarRacingEnv: Đua 6 xe trên Silverstone mở rộng (3 làn x 4.5m = 13.5m)
- Xe 0 là agent MetaDrive (camera bám theo, dùng cho người chơi tay cầm), xe 1..N-1 spawn thêm vào engine.
- Va chạm vật lý Bullet thật giữa các xe (khác GhostCar xuyên thấu); va chạm chỉ bị phạt nhẹ, không loại xe.
- Lưới xuất phát phía sau vạch x = 60m: mặc định xếp ngang hàng (mỗi làn 1 xe, hàng cách 10m), tùy chọn so le kiểu F1.
- Mỗi xe giữ trạng thái riêng (RacerState) và dùng lại đúng quan sát 40D / luật 4 bánh FIA / hàm thưởng của MetaRacingEnv.
- Quan sát 40D (52D với xe F1) + K đối thủ gần nhất (dx, dy, dvx, dvy trong hệ trục xe + cờ có xe) -> 40 + 5K chiều. Zero Data Leakage:
  chỉ chứa trạng thái vật lý quan sát được, không có racing line tối ưu.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces
from panda3d.core import Vec3

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from game_env.racing_env import MetaRacingEnv
from game_env.telemetry import TelemetryTracker
from game_env.vehicle import BaseRaceVehicle, get_vehicle

OPPONENT_FEATURES = 5

# Màu xe: xe 0 (người chơi) xanh lá, các xe AI màu khác nhau để dễ phân biệt
CAR_COLORS = [
    (0.10, 0.90, 0.20),
    (0.95, 0.10, 0.10),
    (0.10, 0.35, 0.95),
    (0.95, 0.75, 0.05),
    (0.85, 0.20, 0.90),
    (0.05, 0.85, 0.85),
    (0.95, 0.45, 0.05),
    (0.90, 0.90, 0.90),
]


class RacerState:
    """Trạng thái riêng của từng xe; tên thuộc tính trùng với MetaRacingEnv để dùng chung hàm quan sát / track limits"""

    def __init__(self, car_id: int, car: BaseRaceVehicle, raw_vehicle):
        self.car_id = car_id
        self.car = car
        self.raw_vehicle = raw_vehicle
        self.telemetry = TelemetryTracker()
        self.current_track_idx = 0
        self.last_s = 0.0
        self.progress = 0.0        # Quãng đường tiến tới tích lũy kể từ lúc xuất phát (m)
        self.start_s = 0.0         # Tọa độ Frenet ô xuất phát (âm khi xếp lưới sau vạch)
        self.lap_target = 0.0      # Mốc progress để hoàn thành vòng kế tiếp
        self.current_lap = 1
        self.lap_start_step = 0
        self.lap_times: List[float] = []
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_acc_long_g = 0.0
        self.last_acc_lat_g = 0.0
        self.stagnant_steps = 0
        self.contacts = 0          # Số lần va chạm (tính theo sự kiện bắt đầu chạm)
        self.in_contact = False
        self.active = True
        self.finished = False
        self.termination_reason = ""
        self.speed_km_h = 0.0
        self.on_runoff = False
        self.crashes = 0           # Số lần văng (tính cả các lần đã hồi sinh)
        self.stalls = 0
        self.respawns = 0
        self.respawn_step = -1     # >= 0: đang chờ hồi sinh tại chỗ ở bước mô phỏng này


class MultiCarRacingEnv(MetaRacingEnv):
    def __init__(
        self,
        num_cars: int = 6,
        use_render: bool = False,
        lane_num: int = 3,
        max_laps: int = 1,
        time_penalty: float = 0.01,
        collision_penalty: float = 2.0,
        opponent_slots: int = 5,
        opponent_range: float = 60.0,
        max_steps: int = 3000,
        start_seed: int = 42,
        window_size=(1280, 720),
        respawn=False,
        respawn_clearance: float = 40.0,
        respawn_delay: float = 3.0,
        max_respawns: int = 5,
        grid: str = "abreast",
        vehicle: str = "gt3",
        stall_penalty: float = 0.0,
    ):
        super().__init__(
            use_render=use_render,
            window_size=window_size,
            start_seed=start_seed,
            vehicle=get_vehicle(vehicle),
            max_laps=max_laps,
            time_penalty=time_penalty,
            lane_num=lane_num,
            horizon=max_steps + 100,
            stall_penalty=stall_penalty,
        )
        self.num_cars = int(num_cars)
        self.vehicle_name = vehicle
        self.collision_penalty = float(collision_penalty)
        self.opponent_slots = int(opponent_slots)
        self.opponent_range = float(opponent_range)
        self.max_steps = int(max_steps)
        # Xe bị loại (văng / đứng yên):
        # - False: rời cuộc đua
        # - True / "random": đặt lại ngay ở vị trí trống ngẫu nhiên (episode mới, kiểu cũ)
        # - "inplace": đứng ngoài respawn_delay giây rồi quay lại đúng chỗ văng, giữ nguyên vòng / quãng đường
        self.respawn = respawn
        self.respawn_clearance = float(respawn_clearance)
        self.respawn_delay = float(respawn_delay)
        # Quá số lần hồi sinh tại chỗ -> bỏ cuộc (tránh xe kẹt vô hạn, vd. policy tất định đứng yên trước cua khó)
        self.max_respawns = int(max_respawns)
        if grid not in ("abreast", "staggered"):
            raise ValueError(f"grid phải là 'abreast' hoặc 'staggered', nhận được {grid!r}")
        self.grid = grid
        # Phần quan sát của 1 xe: 40D (GT3) hoặc 52D (F1, tầm nhìn mở rộng)
        self.base_obs_dim = int(self.obs_dim)
        self.obs_dim = self.base_obs_dim + OPPONENT_FEATURES * self.opponent_slots
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.racers: List[RacerState] = []
        self.extra_vehicles = []

    # ------------------------------------------------------------------ spawn / reset
    def grid_slots(self) -> List[Tuple[float, float]]:
        """
        Vị trí xuất phát, hàng đầu ngay sau vạch (x = 56m):
        - "abreast": mỗi hàng 1 xe / làn (3 xe ngang hàng trên đường 3 làn), các hàng cách nhau 10m
        - "staggered": lưới F1 so le, mỗi xe lùi 8m, lệch trái/phải 3m quanh tim đường
        """
        if self.grid == "abreast":
            per_row = self.lane_num
            return [(56.0 - 10.0 * (k // per_row), (k % per_row) * self.lane_width) for k in range(self.num_cars)]
        offset = min(3.0, self.half_width - 1.5)
        return [(56.0 - 8.0 * k, self.center_lat + (offset if k % 2 == 0 else -offset)) for k in range(self.num_cars)]

    def _spawn_extra_vehicle(self, car: BaseRaceVehicle, pos, heading: float):
        engine = self.env.engine
        cfg = {
            "use_special_color": False,
            "max_speed_km_h": float(car.cfg.max_speed_km_h),
            "max_engine_force": float(car.cfg.max_engine_force),
            "max_brake_force": float(car.cfg.max_brake_force),
            "wheel_friction": float(car.cfg.wheel_friction),
            "max_steering": float(car.cfg.max_steering_deg),
            "mass": float(car.cfg.mass),
            "show_navi_mark": False,
            "show_dest_mark": False,
            "show_line_to_dest": False,
        }
        raw = engine.spawn_object(DefaultVehicle, vehicle_config=cfg, position=list(pos), heading=float(heading))
        return raw

    def _set_color(self, raw, car_id: int):
        try:
            from panda3d.core import Vec4
            r, g, b = CAR_COLORS[car_id % len(CAR_COLORS)]
            if getattr(raw, "origin", None) is not None:
                raw.origin.setColorScale(Vec4(r, g, b, 1.0))
        except Exception:
            pass

    def reset(self, *, seed=None, options=None):
        # SingleRacingMetaDrive.reset tự dọn các xe spawn ngoài agent manager (xe AI của vòng trước)
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
        self.extra_vehicles = []
        if not self.circuit_initialized:
            self._init_circuit_polyline()

        self.current_step = 0
        self.racers = []
        slots = self.grid_slots()
        for k, (x, y) in enumerate(slots):
            if k == 0:
                raw = self.env.agent
                car = self.car
            else:
                car = get_vehicle(self.vehicle_name)
                raw = self._spawn_extra_vehicle(car, (x, y), 0.0)
                self.extra_vehicles.append(raw)
            car.attach_agent(raw)
            raw.set_position([x, y])
            raw.set_heading_theta(0.0)
            raw.set_velocity([0.0, 0.0])
            self._set_color(raw, k)
            self.racers.append(RacerState(k, car, raw))

        for st in self.racers:
            idx, s = self._get_track_s(st.car.position)
            if s > 0.5 * self.circuit_length:
                s -= self.circuit_length  # Xe xếp lưới phía sau vạch xuất phát
            st.current_track_idx = idx
            st.last_s = s
            st.start_s = s
            st.lap_target = self.circuit_length - s
            st.telemetry.reset()

        obs = self._all_observations(np.zeros(self.num_cars), np.zeros(self.num_cars, dtype=bool))
        return obs, {"active": self.active_mask()}

    # ------------------------------------------------------------------ observation
    def active_mask(self) -> np.ndarray:
        return np.array([st.active for st in self.racers], dtype=bool)

    def _opponent_features(self, st: RacerState) -> List[float]:
        pos = st.car.position
        vel = st.car.velocity
        heading = st.car.heading_theta
        fwd = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        lat = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        others = []
        for ot in self.racers:
            if ot is st or not ot.active:
                continue
            rel = ot.car.position - pos
            dist = float(np.linalg.norm(rel))
            if dist <= self.opponent_range:
                rel_v = ot.car.velocity - vel
                others.append((dist, [
                    float(np.dot(rel, fwd) / self.opponent_range),
                    float(np.dot(rel, lat) / self.opponent_range),
                    float(np.clip(np.dot(rel_v, fwd) / 20.0, -2.0, 2.0)),
                    float(np.clip(np.dot(rel_v, lat) / 20.0, -2.0, 2.0)),
                    1.0,
                ]))
        others.sort(key=lambda o: o[0])
        feats: List[float] = []
        for k in range(self.opponent_slots):
            feats.extend(others[k][1] if k < len(others) else [0.0] * OPPONENT_FEATURES)
        return feats

    def _all_observations(self, delta_steers, runoffs) -> np.ndarray:
        obs = np.zeros((self.num_cars, self.obs_dim), dtype=np.float32)
        for i, st in enumerate(self.racers):
            if not st.active:
                continue
            base = self.get_compact_racing_obs(
                st.raw_vehicle,
                action_applied=st.last_action,
                delta_steer=float(delta_steers[i]),
                on_runoff=bool(runoffs[i]),
                state=st,
            )
            obs[i, :self.base_obs_dim] = base
            obs[i, self.base_obs_dim:] = self._opponent_features(st)
        return obs

    # ------------------------------------------------------------------ step
    def _retire(self, st: RacerState):
        """Đưa xe đã dừng đua ra khỏi đường đua để không thành chướng ngại vật cho xe khác"""
        st.active = False
        try:
            st.raw_vehicle.set_velocity([0.0, 0.0])
            st.raw_vehicle.set_position([-2000.0 - 30.0 * st.car_id, -2000.0])
            st.raw_vehicle.set_static(True)
        except Exception:
            pass

    @property
    def respawn(self):
        return self._respawn_mode

    @respawn.setter
    def respawn(self, mode):
        if mode is True:
            mode = "random"
        if mode not in (False, None, "random", "inplace"):
            raise ValueError(f"respawn phải là False / 'random' / 'inplace', nhận được {mode!r}")
        self._respawn_mode = mode or False

    def race_over(self) -> bool:
        """Cuộc đua kết thúc khi không còn xe nào đang chạy hoặc đang chờ hồi sinh"""
        return not any(st.active or st.respawn_step >= 0 for st in self.racers)

    def _handle_out(self, st: RacerState) -> bool:
        """Xử lý xe vừa kết thúc ở bước này. Trả về True nếu xe được đặt lại ngay (obs là trạng thái mới)"""
        if st.finished or not self.respawn:
            self._retire(st)
            return False
        if self.respawn == "random":
            self._respawn(st)
            return True
        # inplace: tạm rời đường đua, chờ phạt thời gian rồi quay lại chỗ văng
        self._retire(st)
        if st.respawns >= self.max_respawns:
            return False
        st.respawn_step = self.current_step + max(1, int(round(self.respawn_delay / self.dt)))
        st.respawns += 1
        return False

    def _try_rejoin(self, st: RacerState) -> bool:
        """Đặt xe đang chờ trở lại tim đường gần chỗ văng (ưu tiên lùi về sau), cách các xe khác >= 20m"""
        others = [ot.car.position for ot in self.racers if ot is not st and ot.active]
        spacing = self.circuit_length / self.num_track_points
        chosen = None
        for off_m in (0, -10, -20, -30, -40, -50, -60, 10, 20, 30):
            idx = int((st.current_track_idx + int(round(off_m / spacing))) % self.num_track_points)
            if all(np.linalg.norm(self.track_xy[idx] - p) >= 20.0 for p in others):
                chosen = idx
                break
        if chosen is None:
            return False
        pos = self.track_xy[chosen]
        raw = st.raw_vehicle
        raw.set_static(False)
        raw.set_position([float(pos[0]), float(pos[1])])
        raw.set_heading_theta(float(self.track_theta[chosen]))
        raw.set_velocity([0.0, 0.0])
        try:
            raw.set_pitch(0.0)
            raw.set_roll(0.0)
            raw.body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        except Exception:
            pass
        # Giữ tiến độ đua: cộng phần chênh lệch Frenet giữa chỗ văng và chỗ quay lại (thường <= vài chục mét)
        new_s = float(self.track_s[chosen])
        delta_s = new_s - st.last_s
        if delta_s < -0.5 * self.circuit_length:
            delta_s += self.circuit_length
        elif delta_s > 0.5 * self.circuit_length:
            delta_s -= self.circuit_length
        st.progress += delta_s
        st.last_s = new_s
        st.current_track_idx = chosen
        st.car.reset_state()
        st.telemetry.reset()
        st.last_action = np.zeros(3, dtype=np.float32)
        st.last_acc_long_g = 0.0
        st.last_acc_lat_g = 0.0
        st.stagnant_steps = 0
        st.in_contact = False
        st.speed_km_h = 0.0
        st.on_runoff = False
        st.respawn_step = -1
        st.active = True
        st.termination_reason = ""
        return True

    def _respawn(self, st: RacerState):
        """Đặt lại xe đứng yên trên tim đường tại điểm ngẫu nhiên cách mọi xe khác >= respawn_clearance (episode mới của xe đó)"""
        others = [ot.car.position for ot in self.racers if ot is not st and ot.active]
        for _ in range(50):
            idx = int(self.np_random.integers(0, self.num_track_points))
            if all(np.linalg.norm(self.track_xy[idx] - p) >= self.respawn_clearance for p in others):
                break
        pos = self.track_xy[idx]
        st.raw_vehicle.set_position([float(pos[0]), float(pos[1])])
        st.raw_vehicle.set_heading_theta(float(self.track_theta[idx]))
        st.raw_vehicle.set_velocity([0.0, 0.0])
        try:
            # Xóa quay/lật còn sót lại (kể cả trạng thái NaN sau va chạm cực mạnh)
            st.raw_vehicle.set_pitch(0.0)
            st.raw_vehicle.set_roll(0.0)
            st.raw_vehicle.body.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        except Exception:
            pass
        st.car.reset_state()
        st.telemetry.reset()
        st.current_track_idx = idx
        st.last_s = float(self.track_s[idx])
        st.lap_target = st.progress + self.circuit_length
        st.current_lap = 1
        st.lap_start_step = self.current_step
        st.last_action = np.zeros(3, dtype=np.float32)
        st.last_acc_long_g = 0.0
        st.last_acc_lat_g = 0.0
        st.stagnant_steps = 0
        st.in_contact = False
        st.finished = False

    def step(self, actions: np.ndarray):
        """
        actions: (num_cars, 3) [steer, throttle, brake]; hành động của xe không còn active bị bỏ qua.
        Trả về obs (N, D), rewards (N,), dones (N,) [xe kết thúc ở bước này], truncated (bool), infos (list)
        """
        self.current_step += 1
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_cars, 3)
        was_active = self.active_mask()
        delta_steers = np.zeros(self.num_cars, dtype=np.float32)
        net_pedals = np.zeros(self.num_cars, dtype=np.float32)
        applied = [np.zeros(2, dtype=np.float32) for _ in range(self.num_cars)]

        # 1. Cơ cấu chấp hành (slew rate limiter + bộ trộn bàn đạp) cho từng xe
        for i, st in enumerate(self.racers):
            if not st.active:
                continue
            applied[i], delta_steers[i], net_pedals[i] = st.car.process_action(actions[i])
            if i > 0:
                st.raw_vehicle.before_step(applied[i])

        # 2. Một bước vật lý chung cho tất cả các xe (xe 0 nhận action qua MetaDrive agent)
        self.env.step(applied[0] if was_active[0] else np.array([0.0, 0.0], dtype=np.float32))
        for i in range(1, self.num_cars):
            if self.racers[i].active:
                self.racers[i].raw_vehicle._state_check()

        rewards = np.zeros(self.num_cars, dtype=np.float32)
        dones = np.zeros(self.num_cars, dtype=bool)
        infos: List[Dict] = [{} for _ in range(self.num_cars)]
        runoffs = np.zeros(self.num_cars, dtype=bool)
        current_time = float(self.current_step * self.dt)

        for i, st in enumerate(self.racers):
            if not st.active:
                continue
            on_runoff, is_full_out = self.check_track_limits_and_apply_runoff(state=st, vehicle_contact_is_out=False)
            runoffs[i] = on_runoff
            idx, current_s = self._get_track_s(st.car.position, st.current_track_idx)
            st.current_track_idx = idx
            sample = st.telemetry.log_step(
                current_time=current_time,
                position=st.car.position,
                velocity=st.car.velocity,
                steering=float(applied[i][0]),
                throttle_brake=float(net_pedals[i]),
                dt=self.dt,
                on_runoff=on_runoff,
            )
            st.last_acc_long_g = float(sample.get("acc_long_g", 0.0))
            st.last_acc_lat_g = float(sample.get("acc_lat_g", 0.0))
            speed_km_h = float(sample["speed_km_h"])
            st.speed_km_h = speed_km_h
            st.on_runoff = on_runoff

            # Tiến độ Frenet có xử lý quấn vòng qua vạch đích theo cả 2 chiều
            delta_s = current_s - st.last_s
            if delta_s < -0.5 * self.circuit_length:
                delta_s += self.circuit_length
            elif delta_s > 0.5 * self.circuit_length:
                delta_s -= self.circuit_length
            st.last_s = current_s
            st.progress += float(delta_s)

            if speed_km_h < 10.0:
                st.stagnant_steps += 1
            else:
                st.stagnant_steps = 0
            is_stalled = st.stagnant_steps >= 120

            # Hàm thưởng giống MetaRacingEnv + phạt va chạm xe
            r_progress = 0.05 * max(0.0, float(delta_s))
            r_steer_smooth = -0.10 * float(delta_steers[i] ** 2)
            r_idle = -0.15 if speed_km_h < 10.0 else 0.0
            r_time = -self.time_penalty
            # Phạt va chạm theo sự kiện: -collision_penalty khi bắt đầu chạm, -10% mỗi bước còn dính nhau
            contact = bool(st.car.crash_vehicle)
            new_contact = contact and not st.in_contact
            r_contact = -self.collision_penalty if new_contact else (-0.1 * self.collision_penalty if contact else 0.0)
            st.contacts += int(new_contact)
            st.in_contact = contact
            macro = 0.0
            st.last_action = np.array(actions[i], dtype=np.float32)

            if is_full_out:
                speed_ratio = float(np.clip(speed_km_h / 120.0, 0.0, 1.0))
                macro = -10.0 - 20.0 * (speed_ratio ** 2)
                dones[i] = True
                st.termination_reason = "crash_or_track_limit"
                st.crashes += 1
            elif is_stalled:
                macro = -self.stall_penalty
                dones[i] = True
                st.termination_reason = "stalled"
                st.stalls += 1
            elif st.progress >= st.lap_target:
                lap_time = float((self.current_step - st.lap_start_step) * self.dt)
                st.lap_times.append(lap_time)
                eta = float(np.clip((180.0 - lap_time) / 90.0, 0.0, 1.0))
                macro = 50.0 + 100.0 * float(eta ** 3)
                infos[i]["completed_lap_time"] = lap_time
                if st.current_lap >= self.max_laps:
                    dones[i] = True
                    st.finished = True
                    st.termination_reason = "finished"
                else:
                    st.current_lap += 1
                    st.lap_start_step = self.current_step
                    st.lap_target += self.circuit_length

            rewards[i] = r_progress + r_steer_smooth + r_idle + r_time + r_contact + macro
            infos[i].update({
                "speed_km_h": speed_km_h,
                "track_s": current_s,
                "progress": st.progress,
                "contact": contact,
                "on_runoff": on_runoff,
                "lap": st.current_lap,
                "lap_time": float((self.current_step - st.lap_start_step) * self.dt),
                "termination_reason": st.termination_reason,
            })

        for i, st in enumerate(self.racers):
            if dones[i] and self._handle_out(st):
                runoffs[i] = False
                delta_steers[i] = 0.0
        # Hồi sinh tại chỗ: xe đã hết thời gian phạt quay lại đường đua (bắt đầu chuỗi mới ở bước sau)
        for i, st in enumerate(self.racers):
            if st.respawn_step >= 0 and self.current_step >= st.respawn_step and self._try_rejoin(st):
                runoffs[i] = False
                delta_steers[i] = 0.0
        # Với respawn: obs của xe vừa done là trạng thái đầu của episode mới (trainer đã cắt chuỗi bằng dones)
        obs = self._all_observations(delta_steers, runoffs)
        # Lưới an toàn: vật lý Bullet có thể nổ (NaN/inf) khi va chạm cực mạnh -> coi như văng, đặt lại xe
        bad = [i for i in range(self.num_cars) if not (np.isfinite(obs[i]).all() and np.isfinite(rewards[i]))]
        if bad:
            for i in bad:
                st = self.racers[i]
                rewards[i] = -30.0
                if not dones[i]:
                    dones[i] = True
                    st.termination_reason = "crash_or_track_limit"
                    st.crashes += 1
                    if self._handle_out(st):
                        runoffs[i] = False
                        delta_steers[i] = 0.0
            obs = np.nan_to_num(self._all_observations(delta_steers, runoffs), nan=0.0, posinf=0.0, neginf=0.0)
        truncated = bool(self.current_step >= self.max_steps)
        return obs, rewards, dones, truncated, infos

    # ------------------------------------------------------------------ race info
    def race_distance(self, st: RacerState) -> float:
        """Vị trí trên đường đua tính từ vạch xuất phát (m), dùng để xếp hạng và tính khoảng cách giữa các xe"""
        return float(st.start_s + st.progress)

    def standings(self) -> List[int]:
        """Thứ hạng hiện tại: xe về đích trước xếp trên, còn lại theo vị trí trên đường đua"""
        def key(st: RacerState):
            total_time = sum(st.lap_times) if st.finished else float("inf")
            return (0 if st.finished else 1, total_time, -self.race_distance(st))
        return [st.car_id for st in sorted(self.racers, key=key)]
