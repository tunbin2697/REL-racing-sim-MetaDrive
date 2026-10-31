"""
MultiCarRacingEnv: Đua 4-6 xe trên Silverstone half (mặc định 3 làn x 4.5m = 13.5m)
- Xây dựng trực tiếp trên nền nhánh feature/111s-honest-anchor (obs ego 35D của ref_111s + xe gt3_anchor).
- Xe 0 là agent MetaDrive (camera bám theo, dùng cho người chơi tay cầm hoặc AI-0), xe 1..N-1 spawn thêm vào engine.
- Va chạm vật lý Bullet thật giữa các xe; va chạm bị phạt đối xứng + vùng đệm an toàn (proximity) + thưởng vượt sạch (WI-L5).
- Lưới xuất phát phía sau vạch x = 60m: mặc định xếp ngang hàng (abreast) hoặc so le kiểu F1 (staggered).
- Quan sát: 35D (ego chuẩn ref_111s nguyên bản) + 5K đối thủ gần nhất (dx, dy, dvx, dvy trong hệ trục xe + cờ có xe) = 60D.
- Hỗ trợ phạt mượt lái (steer_smooth_coef = c1) và mượt bàn đạp (pedal_smooth_coef = c2) theo CAPS (§4.2).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces

from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from game_env.racing_env import MetaRacingEnv
from game_env.telemetry import TelemetryTracker
from game_env.vehicle import BaseRaceVehicle, get_vehicle

BASE_OBS_DIM = 35
OPPONENT_FEATURES = 5

# Hằng số WI-L5 (§4.3): thưởng vượt sạch & vùng đệm né va chạm
OVERTAKE_BONUS = 1.0
OVERTAKE_COOLDOWN_STEPS = 50
RESPAWN_IMMUNITY_STEPS = 150
PROXIMITY_LONG_M = 5.4     # Dài thân xe 4.5m + đệm dọc 0.9m
PROXIMITY_LAT_M = 2.5      # Rộng thân xe 1.85m + đệm ngang 0.65m
PROXIMITY_PENALTY_COEF = 0.50

# Màu xe: xe 0 xanh lá, các xe AI màu khác nhau để dễ phân biệt
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
    """Trạng thái riêng của từng xe trên đường đua"""

    def __init__(self, car_id: int, car: BaseRaceVehicle, raw_vehicle, spawn_lat: float = 0.0):
        self.car_id = car_id
        self.car = car
        self.raw_vehicle = raw_vehicle
        self.telemetry = TelemetryTracker()
        self.current_track_idx = 0
        self.last_s = 0.0
        self.lat_offset = float(spawn_lat)
        self.steer_nudge = 0.0
        self.progress = 0.0        # Quãng đường tiến tới tích lũy kể từ lúc xuất phát (m)
        self.start_s = 0.0         # Tọa độ Frenet ô xuất phát (âm khi xếp lưới sau vạch)
        self.prev_race_dist = 0.0  # Vị trí đường đua ở bước trước (dùng xét vượt xe WI-L5)
        self.lap_target = 0.0      # Mốc progress để hoàn thành vòng kế tiếp
        self.current_lap = 1
        self.lap_start_step = 0
        self.respawn_step = -999   # Bước hồi sinh gần nhất (cửa sổ miễn trừ WI-L5)
        self.lap_times: List[float] = []
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_applied = np.zeros(2, dtype=np.float32)
        self.last_acc_long_g = 0.0
        self.last_acc_lat_g = 0.0
        self.stagnant_steps = 0
        self.contacts = 0          # Số lần va chạm (tính theo sự kiện bắt đầu chạm)
        self.overtakes = 0         # Số lần vượt sạch (WI-L5)
        self.in_contact = False
        self.active = True
        self.finished = False
        self.termination_reason = ""
        self.speed_km_h = 0.0
        self.on_runoff = False
        # Lịch sử điều khiển & góc trượt phục vụ đo chỉ số §6 (F20)
        self.steer_cmd_history: List[float] = []
        self.pedal_history: List[float] = []
        self.beta_deg_history: List[float] = []


class MultiCarRacingEnv(MetaRacingEnv):
    def __init__(
        self,
        num_cars: int = 6,
        use_render: bool = False,
        lane_num: int = 3,
        map_type: Optional[str] = None,
        vehicle_name: str = "gt3_anchor",
        max_laps: int = 1,
        time_penalty: float = 0.01,
        collision_penalty: float = 2.0,
        steer_smooth_coef: float = 0.10,
        pedal_smooth_coef: float = 0.0,
        center_coef: float = 0.0,
        opponent_slots: int = 5,
        opponent_range: float = 60.0,
        max_steps: int = 3000,
        start_seed: int = 42,
        window_size=(1280, 720),
        respawn: bool = False,
        respawn_clearance: float = 20.0,
        grid: str = "abreast",
    ):
        if map_type is None:
            map_type = "silverstone_half_3lane" if int(lane_num) == 3 else "silverstone_half"
        self.vehicle_name = str(vehicle_name)
        super().__init__(
            use_render=use_render,
            window_size=window_size,
            start_seed=start_seed,
            map_type=map_type,
            vehicle=get_vehicle(self.vehicle_name),
            max_laps=max_laps,
            time_penalty=time_penalty,
            center_coef=center_coef,
            horizon=max_steps + 100,
        )
        self.num_cars = int(num_cars)
        self.collision_penalty = float(collision_penalty)
        self.steer_smooth_coef = float(steer_smooth_coef)
        self.pedal_smooth_coef = float(pedal_smooth_coef)
        self.opponent_slots = int(opponent_slots)
        self.opponent_range = float(opponent_range)
        self.max_steps = int(max_steps)
        # Chỉ dùng khi train: xe bị loại được đặt lại ở vị trí trống ngẫu nhiên thay vì rời cuộc đua
        self.respawn = bool(respawn)
        self.respawn_clearance = float(respawn_clearance)
        if grid not in ("abreast", "staggered"):
            raise ValueError(f"grid phải là 'abreast' hoặc 'staggered', nhận được {grid!r}")
        self.grid = grid
        self.obs_dim = BASE_OBS_DIM + OPPONENT_FEATURES * self.opponent_slots
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.racers: List[RacerState] = []
        self.extra_vehicles = []
        self.overtake_cooldown: Dict[Tuple[int, int], int] = {}

    @property
    def lane_num(self) -> int:
        return int(self.preset.lane_num)

    @property
    def lane_width(self) -> float:
        return float(self.preset.lane_width)

    @property
    def half_width(self) -> float:
        return float(self.preset.half_width)

    @property
    def center_lat(self) -> float:
        return float(self.preset.center_lat)

    # ------------------------------------------------------------------ spawn / reset
    def grid_slots(self) -> List[Tuple[float, float]]:
        """
        Vị trí xuất phát, hàng đầu ngay sau vạch (x = 56m):
        - "abreast": mỗi hàng 3 xe (lệch nhẹ 2.5m dọc giữa các cột trong cùng hàng kiểu F1/GT3 để tránh 3 xe chụm mũi cùng lúc vào T1)
        - "staggered": lưới F1 so le, mỗi xe lùi 8m, lệch trái/phải 3.0m quanh tim đường
        """
        if self.grid == "abreast":
            per_row = self.lane_num
            slots = []
            for k in range(self.num_cars):
                row = k // per_row
                col = k % per_row
                x = 56.0 - 12.0 * row - 3.0 * col
                y = col * self.lane_width
                slots.append((x, y))
            return slots
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

    def _compute_car_lat_offset(self, pos: np.ndarray, track_idx: int) -> float:
        track_pt = self.track_xy[track_idx]
        th = float(self.track_theta[track_idx])
        lat_normal = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
        return float(np.dot(pos[:2] - track_pt, lat_normal))

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
        self.overtake_cooldown = {}
        self.racers = []
        slots = self.grid_slots()
        for k, (x, y) in enumerate(slots):
            if k == 0:
                raw = self.env.agent
                car = self.car
                car.reset_state()
            else:
                car = get_vehicle(self.vehicle_name)
                raw = self._spawn_extra_vehicle(car, (x, y), 0.0)
                self.extra_vehicles.append(raw)
            car.attach_agent(raw)
            raw.set_position([x, y])
            raw.set_heading_theta(0.0)
            raw.set_velocity([0.0, 0.0])
            self._set_color(raw, k)
            spawn_lat = float(y - self.center_lat)
            self.racers.append(RacerState(k, car, raw, spawn_lat=spawn_lat))

        for st in self.racers:
            idx, s = self._get_track_s(st.car.position)
            if s > 0.5 * self.circuit_length:
                s -= self.circuit_length  # Xe xếp lưới phía sau vạch xuất phát
            st.current_track_idx = idx
            st.last_s = s
            st.start_s = s
            st.prev_race_dist = s
            st.lap_target = self.circuit_length - s
            st.lat_offset = self._compute_car_lat_offset(st.car.position, idx)
            st.telemetry.reset()

        obs = self._all_observations(np.zeros(self.num_cars), np.zeros(self.num_cars, dtype=bool))
        return obs, {"active": self.active_mask()}

    # ------------------------------------------------------------------ observation & track limits
    def active_mask(self) -> np.ndarray:
        return np.array([st.active for st in self.racers], dtype=bool)

    def _apply_racing_etiquette(self, st: RacerState, action: np.ndarray) -> np.ndarray:
        """
        Bộ đệm an toàn cục bộ (Racing Etiquette Shield):
        - Giữ nguyên 100% quan sát 35D chuẩn của ref_111s để xe ôm cua chuẩn xác, không bao giờ văng lề.
        - Khi có xe khác áp sát trong vùng nguy hiểm (dx in [-2.5m, +9.5m], |dy| < 2.4m):
          1. Đánh lái nhẹ (steer_nudge mượt, có giới hạn biên đường |lat_offset| < 3.8m) để tách ngang.
          2. Xe đi sau (dx > 0.2m) nhả bớt ga mượt (giữ net_pedal >= +0.10 để không gây đổi dấu pedal_sw)
             nhằm nhường đỉnh cua (apex) hoặc đợi khoảng trống vượt sạch thay vì cọ hông 50 bước liên tiếp.
        """
        act = np.array(action, dtype=np.float32, copy=True)
        pos = st.car.position
        vel = st.car.velocity
        heading = st.car.heading_theta
        fwd = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        lat = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)

        raw_nudge = 0.0
        max_yield_urg = 0.0
        min_front_dx = 999.0
        front_closing_v = 0.0

        for ot in self.racers:
            if ot is st or not ot.active:
                continue
            rel = ot.car.position - pos
            dx = float(np.dot(rel, fwd))
            dy = float(np.dot(rel, lat))
            if -2.5 <= dx <= 11.5 and abs(dy) < 2.65:
                rel_v = ot.car.velocity - vel
                dvx = float(np.dot(rel_v, fwd))  # < 0 nghĩa là đang lao lại gần ot
                # 1. Đánh lái tách ngang rất nhẹ CHỈ khi đang chạy thẳng (|act[0]| < 0.12) và xa mép cỏ (|lat_offset| < 3.2m)
                if -2.0 <= dx <= 6.5 and abs(float(act[0])) < 0.12:
                    lat_margin = max(0.0, 1.0 - abs(dy) / 2.50)
                    dir_away = -1.0 if dy >= 0.0 else 1.0
                    if (dir_away > 0 and st.lat_offset < 3.2) or (dir_away < 0 and st.lat_offset > -3.2):
                        raw_nudge += dir_away * 0.04 * lat_margin

                # 2. Nhường nhịp dọc (Throttle lift) cho xe đi sau khi 2 xe chụm vào cùng 1 làn/đỉnh cua
                is_behind = (dx > 0.20) or (abs(dx) <= 0.20 and st.car_id > ot.car_id)
                if is_behind and dx >= -0.5 and abs(dy) < 2.40:
                    closing = max(0.0, -dvx)
                    urg = float(np.clip((9.5 - max(0.0, dx)) / 4.8 + 0.12 * closing, 0.0, 1.0)) * (1.0 - 0.25 * (abs(dy) / 2.40))
                    if urg > max_yield_urg:
                        max_yield_urg = urg
                    if abs(dy) < 1.95 and dx > 0.0 and dx < min_front_dx:
                        min_front_dx = dx
                        front_closing_v = closing

        # Lọc thông thấp cho steer_nudge để không tăng steer_rev và không làm lệch cua
        st.steer_nudge = float(0.80 * st.steer_nudge + 0.20 * np.clip(raw_nudge, -0.05, 0.05))
        act[0] = float(np.clip(act[0] + st.steer_nudge, -1.0, 1.0))

        # Nhả ga mượt cho xe đi sau (giữ throttle dương >= 0.12 để pedal không đổi dấu sang phanh)
        if max_yield_urg > 0.05:
            max_thr = float(np.clip(1.0 - 0.86 * max_yield_urg, 0.12, 1.0))
            cur_thr = 0.5 * (float(act[1]) + 1.0)
            if cur_thr > max_thr:
                act[1] = 2.0 * max_thr - 1.0
            if min_front_dx < 4.5 and front_closing_v > 2.2:
                # Chỉ phanh nhẹ khi thật sự sắp đâm đuôi ở khoảng cách < 4.5m
                brk = float(np.clip(0.28 * (4.5 - min_front_dx) / 1.8, 0.0, 0.35))
                act[1] = -1.0
                act[2] = max(float(act[2]), 2.0 * brk - 1.0)
        return act

    def _check_car_track_limits(self, st: RacerState) -> Tuple[bool, bool]:
        """Kiểm tra luật 4 bánh FIA cho từng xe; va chạm giữa các xe chỉ phạt ở reward, không loại xe"""
        wheels = st.car.get_wheel_positions()
        if not wheels:
            return False, False
        tire_half_width = 0.15
        asphalt_limit = self.preset.half_width
        wheels_on_track = 0
        for w_pos in wheels:
            idx, _ = self._get_track_s(w_pos, st.current_track_idx)
            dist_to_center = float(np.linalg.norm(w_pos - self.track_xy[idx]))
            dist_inner_edge = max(0.0, dist_to_center - tire_half_width)
            if dist_inner_edge <= asphalt_limit:
                wheels_on_track += 1
        is_airborne_or_flipped = st.car.is_airborne_or_flipped()
        is_full_out = bool(
            wheels_on_track == 0
            or is_airborne_or_flipped
            or st.car.crash_object
            or st.car.crash_sidewalk
        )
        on_runoff = bool(not is_full_out and wheels_on_track < 4)
        return on_runoff, is_full_out

    def _proximity_penalty(self, st: RacerState) -> float:
        """Phạt vùng đệm elip khi tâm 2 xe tiến quá sát nhau (< 5.4m dọc và < 2.5m ngang) để chủ động tách làn"""
        pos = st.car.position
        heading = st.car.heading_theta
        fwd = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        lat = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        max_intrusion = 0.0
        for ot in self.racers:
            if ot is st or not ot.active:
                continue
            rel = ot.car.position - pos
            dx = float(np.dot(rel, fwd))
            dy = float(np.dot(rel, lat))
            if abs(dx) < PROXIMITY_LONG_M and abs(dy) < PROXIMITY_LAT_M:
                d_ell = float(np.sqrt((dx / PROXIMITY_LONG_M) ** 2 + (dy / PROXIMITY_LAT_M) ** 2))
                if d_ell < 1.0:
                    intrusion = (1.0 - d_ell) ** 2
                    if intrusion > max_intrusion:
                        max_intrusion = intrusion
        return -PROXIMITY_PENALTY_COEF * max_intrusion

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
            # Đồng bộ trạng thái tạm thời để tái sử dụng nguyên bản 100% get_compact_racing_obs (35D) của MetaRacingEnv
            self.current_track_idx = st.current_track_idx
            self.last_action = st.prev_action
            base = self.get_compact_racing_obs(
                st.raw_vehicle,
                action_applied=st.last_applied,
                delta_steer=float(delta_steers[i]),
                on_runoff=bool(runoffs[i]),
            )
            st.current_track_idx = self.current_track_idx
            st.lat_offset = self._compute_car_lat_offset(st.car.position, st.current_track_idx)
            obs[i, :BASE_OBS_DIM] = base[:BASE_OBS_DIM]
            obs[i, BASE_OBS_DIM:] = self._opponent_features(st)
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

    def _respawn(self, st: RacerState):
        """Đặt lại xe ở làn ngẫu nhiên tại điểm cách các xe khác >= respawn_clearance"""
        others = [ot.car.position for ot in self.racers if ot is not st and ot.active]
        max_lat = min(2.5, self.half_width - 4.0)
        lane_choices = [-max_lat, 0.0, max_lat] if max_lat > 0.5 else [0.0]
        chosen_lat = float(self.np_random.choice(lane_choices))
        idx = 0
        for _ in range(50):
            idx = int(self.np_random.integers(0, self.num_track_points))
            th = float(self.track_theta[idx])
            lat_vec = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
            cand_pos = self.track_xy[idx] + chosen_lat * lat_vec
            if all(np.linalg.norm(cand_pos - p) >= self.respawn_clearance for p in others):
                break
        th = float(self.track_theta[idx])
        lat_vec = np.array([-np.sin(th), np.cos(th)], dtype=np.float32)
        pos = self.track_xy[idx] + chosen_lat * lat_vec
        st.raw_vehicle.set_position([float(pos[0]), float(pos[1])])
        st.raw_vehicle.set_heading_theta(th)
        st.raw_vehicle.set_velocity([0.0, 0.0])
        st.car.reset_state()
        st.telemetry.reset()
        st.current_track_idx = idx
        st.last_s = float(self.track_s[idx])
        st.lat_offset = chosen_lat
        st.steer_nudge = 0.0
        st.prev_race_dist = self.race_distance(st)
        st.lap_target = st.progress + self.circuit_length
        st.current_lap = 1
        st.lap_start_step = self.current_step
        st.respawn_step = self.current_step
        st.prev_action = np.zeros(3, dtype=np.float32)
        st.last_action = np.zeros(3, dtype=np.float32)
        st.last_applied = np.zeros(2, dtype=np.float32)
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
        delta_pedals = np.zeros(self.num_cars, dtype=np.float32)
        net_pedals = np.zeros(self.num_cars, dtype=np.float32)
        applied = [np.zeros(2, dtype=np.float32) for _ in range(self.num_cars)]
        effective_actions = np.copy(actions)

        # 1. Cơ cấu chấp hành (kèm bộ đệm an toàn cục bộ + slew rate limiter + bộ trộn bàn đạp)
        for i, st in enumerate(self.racers):
            if not st.active:
                continue
            st.prev_race_dist = self.race_distance(st)
            prev_pedal = float(st.car.last_pedal)
            effective_actions[i] = self._apply_racing_etiquette(st, actions[i])
            applied[i], delta_steers[i], net_pedals[i] = st.car.process_action(effective_actions[i])
            delta_pedals[i] = float(net_pedals[i] - prev_pedal)
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
            on_runoff, is_full_out = self._check_car_track_limits(st)
            runoffs[i] = on_runoff
            idx, current_s = self._get_track_s(st.car.position, st.current_track_idx)
            st.current_track_idx = idx
            st.lat_offset = self._compute_car_lat_offset(st.car.position, idx)
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

            # Ghi nhận lịch sử phục vụ đo chỉ số §6 (F20): steer, pedal, góc trượt beta khi v > 40 km/h
            st.steer_cmd_history.append(float(effective_actions[i][0]))
            st.pedal_history.append(float(net_pedals[i]))
            if speed_km_h > 40.0:
                vel = st.car.velocity
                h = st.car.heading_theta
                vx = float(vel[0] * np.cos(h) + vel[1] * np.sin(h))
                vy = float(-vel[0] * np.sin(h) + vel[1] * np.cos(h))
                beta_deg = float(np.degrees(np.arctan2(abs(vy), max(0.1, abs(vx)))))
                st.beta_deg_history.append(beta_deg)

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

            # Hàm thưởng giống MetaRacingEnv + phạt mượt lái/pedal (§4.2) + phạt va chạm & vùng đệm an toàn (§4.3)
            r_progress = self.preset.progress_coef * max(0.0, float(delta_s))
            r_steer_smooth = -self.steer_smooth_coef * float(delta_steers[i] ** 2)
            r_pedal_smooth = -self.pedal_smooth_coef * float(delta_pedals[i] ** 2)
            r_center = 0.0
            if self.center_coef > 0.0:
                r_center = -self.center_coef * float((st.lat_offset / self.preset.lat_norm) ** 2)
            r_idle = -0.15 if speed_km_h < 10.0 else 0.0
            r_time = -self.time_penalty
            # Phạt va chạm theo sự kiện: -collision_penalty khi bắt đầu chạm, -25% mỗi bước còn cọ sát nhau
            contact = bool(st.car.crash_vehicle)
            new_contact = contact and not st.in_contact
            r_contact = -self.collision_penalty if new_contact else (-0.25 * self.collision_penalty if contact else 0.0)
            r_proximity = self._proximity_penalty(st)
            st.contacts += int(new_contact)
            st.in_contact = contact
            macro = 0.0

            st.prev_action = st.last_action.copy()
            st.last_action = np.array(effective_actions[i], dtype=np.float32)
            st.last_applied = applied[i].copy()

            if is_full_out:
                speed_ratio = float(np.clip(speed_km_h / 120.0, 0.0, 1.0))
                macro = -10.0 - 20.0 * (speed_ratio ** 2)
                dones[i] = True
                st.termination_reason = "crash_or_track_limit"
            elif is_stalled:
                dones[i] = True
                st.termination_reason = "stalled"
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

            rewards[i] = r_progress + r_steer_smooth + r_pedal_smooth + r_center + r_idle + r_time + r_contact + r_proximity + macro
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

        # 3. Thưởng vượt sạch theo WI-L5 (D25): A vượt B mà không chạm xe, không lấn lề, có cooldown 50 bước
        for i, st_a in enumerate(self.racers):
            if not st_a.active or st_a.in_contact or st_a.on_runoff:
                continue
            cur_a = self.race_distance(st_a)
            prev_a = st_a.prev_race_dist
            for j, st_b in enumerate(self.racers):
                if i == j or not st_b.active:
                    continue
                if self.current_step - st_b.respawn_step <= RESPAWN_IMMUNITY_STEPS:
                    continue
                cur_b = self.race_distance(st_b)
                prev_b = st_b.prev_race_dist
                if prev_a < prev_b and (prev_b - prev_a) < 30.0 and cur_a > cur_b:
                    last_ov = self.overtake_cooldown.get((i, j), -999)
                    if self.current_step - last_ov >= OVERTAKE_COOLDOWN_STEPS:
                        rewards[i] += OVERTAKE_BONUS
                        st_a.overtakes += 1
                        self.overtake_cooldown[(i, j)] = self.current_step

        for i, st in enumerate(self.racers):
            if dones[i]:
                if self.respawn:
                    self._respawn(st)
                    runoffs[i] = False
                    delta_steers[i] = 0.0
                else:
                    self._retire(st)

        # Với respawn: obs của xe vừa done là trạng thái đầu của episode mới (trainer đã cắt chuỗi bằng dones)
        obs = self._all_observations(delta_steers, runoffs)
        truncated = bool(self.current_step >= self.max_steps)
        return obs, rewards, dones, truncated, infos

    # ------------------------------------------------------------------ race info & §6 metrics
    def race_distance(self, st: RacerState) -> float:
        """Vị trí trên đường đua tính từ vạch xuất phát (m), dùng để xếp hạng và tính khoảng cách giữa các xe"""
        return float(st.start_s + st.progress)

    def standings(self) -> List[int]:
        """Thứ hạng hiện tại: xe về đích trước xếp trên, còn lại theo vị trí trên đường đua"""
        def key(st: RacerState):
            total_time = sum(st.lap_times) if st.finished else float("inf")
            return (0 if st.finished else 1, total_time, -self.race_distance(st))
        return [st.car_id for st in sorted(self.racers, key=key)]

    def compute_driving_style_metrics(self) -> Dict[str, float]:
        """
        Đo các chỉ số lái theo mục §6 (F20 trong knowledge/rl_debugging_findings.md):
        - pedal_sw_per_100: số lần đổi dấu pedal (bỏ vùng |pedal| < 0.05) trên 100 bước (mục tiêu <= 5)
        - steer_rev_per_100: số lần đổi chiều của Δlái (|Δlái| > 0.02) trên 100 bước (mục tiêu <= 40)
        - beta_p95_deg: phân vị 95 của góc trượt β = atan2(|v_y|, |v_x|) khi tốc độ > 40 km/h (mục tiêu <= 4°)
        """
        pedal_sw_rates = []
        steer_rev_rates = []
        all_betas = []
        for st in self.racers:
            n_steps = len(st.pedal_history)
            if n_steps >= 10:
                pedals = np.array(st.pedal_history, dtype=np.float32)
                active_pedals = pedals[np.abs(pedals) >= 0.05]
                if len(active_pedals) >= 2:
                    signs = np.sign(active_pedals)
                    sw = int(np.sum(signs[1:] != signs[:-1]))
                else:
                    sw = 0
                pedal_sw_rates.append((sw / n_steps) * 100.0)

                steers = np.array(st.steer_cmd_history, dtype=np.float32)
                ds = np.diff(steers)
                if len(ds) >= 2:
                    revs = int(np.sum((np.sign(ds[1:]) != np.sign(ds[:-1])) & (np.abs(ds[1:]) > 0.02)))
                else:
                    revs = 0
                steer_rev_rates.append((revs / n_steps) * 100.0)
            if st.beta_deg_history:
                all_betas.extend(st.beta_deg_history)

        return {
            "pedal_sw_per_100": float(np.mean(pedal_sw_rates)) if pedal_sw_rates else 0.0,
            "steer_rev_per_100": float(np.mean(steer_rev_rates)) if steer_rev_rates else 0.0,
            "beta_p95_deg": float(np.percentile(all_betas, 95)) if all_betas else 0.0,
        }
