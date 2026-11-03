"""
ĐUA 2-10 XE TRÊN SILVERSTONE HALF: NGƯỜI CHƠI (TAY CẦM XBOX HOẶC BÀN PHÍM) VS AI PPO SELF-PLAY
Tích hợp Hệ Thống Camera & HUD Sim Racing Thể Thao từ nhánh exp/play-vs-ai-gamepad & 111s-honest-anchor:
  - Mở khóa lia chuột tự do 360 độ (360 Free Mouse Look quanh xe, không bị giới hạn góc, không bị giật về tâm).
  - Chuột giữa (Middle Click) hoặc [0]: Reset nhanh góc nhìn camera về chính diện (0 deg).
  - Phím [C], [Tab], [N]: Chuyển camera bám theo xe tiếp theo (YOU <-> AI-1 <-> AI-2 ...).
  - Phím [M]: Chuyển camera về xe phía trước.
  - Phím [1] đến [9]: Nhảy trực tiếp đến góc nhìn của xe tương ứng.
  - Phím [B]: Góc nhìn toàn cảnh từ trên cao (Bird-eye View) để quan sát toàn bộ đường đua và các pha vượt xe.
  - Phím [Q]: Khôi phục góc nhìn người thứ ba (Third-person Chase Cam).
  - Phím [Space]: Tạm dừng / Tiếp tục cuộc đua; [R]: Reset vòng đua mới.
  - HUD Telemetry chuyên nghiệp: Đồng hồ tốc độ km/h, Lap time, Vòng đua hiện tại, Bảng xếp hạng P1-P10 và khoảng cách Gap to Leader.
"""

import os
import sys
import time
import types
import argparse
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metadrive.policy.manual_control_policy import KeyboardController
from game_env.multicar_env import MultiCarRacingEnv, BASE_OBS_DIM
from game_env.gamepad import XboxGamepadController
from agents.ppo_agent import PPOAgent
from training.train_multicar import warm_start_from_single


class MultiCarCameraController:
    """
    Bộ điều khiển Camera Thể Thao Đa Xe (360 Free Mouse Look & Multi-car Tracker)
    Kế thừa kiến trúc từ RacingCameraController trên nhánh 111s-honest-anchor.
    """
    def __init__(self, engine: Any, num_cars: int, mouse_sensitivity: float = 3.2):
        self.engine = engine
        self.num_cars = num_cars
        self.mouse_sensitivity = mouse_sensitivity
        self.tracked_car_idx = 0
        self.is_bird_eye = False
        self.follow_leader = False

        self.main_camera = getattr(self.engine, "main_camera", None)
        self._orig_update_mouse_info = None

        self._setup_free_mouse_look()

    def _setup_free_mouse_look(self):
        """Mở khóa xoay chuột 360 độ tự do, không bị khóa biên hay tự reset về tâm"""
        if self.main_camera is None:
            return

        self._orig_update_mouse_info = getattr(self.main_camera, "update_mouse_info", None)

        def free_update_mouse_info(cam_self):
            if not getattr(cam_self, "has_mouse", False):
                return
            mwn = getattr(cam_self.engine, "mouseWatcherNode", None)
            if mwn is None or not mwn.hasMouse():
                return

            cur_x = mwn.getMouseX()
            if not hasattr(cam_self, "_last_free_mouse_x") or cam_self._last_free_mouse_x is None:
                cam_self._last_free_mouse_x = cur_x
                return

            dx = cur_x - cam_self._last_free_mouse_x
            if abs(cur_x) > 0.80:
                cam_self.set_mouse_to_center()
                cam_self._last_free_mouse_x = 0.0
            else:
                cam_self._last_free_mouse_x = cur_x

            if abs(dx) > 0.001:
                cam_self.mouse_rotate += -dx * self.mouse_sensitivity
                cam_self.mouse_rotate = (cam_self.mouse_rotate + np.pi) % (2.0 * np.pi) - np.pi

        self.main_camera.update_mouse_info = types.MethodType(free_update_mouse_info, self.main_camera)

    def reset_to_front(self):
        """Reset góc nhìn về thẳng sau xe (0 deg)"""
        if self.main_camera is not None:
            self.main_camera.mouse_rotate = 0.0
            print(">> [CAMERA] Đã reset góc nhìn về thẳng phía trước (0 deg).")

    def toggle_bird_eye(self):
        """Bật / Tắt chế độ nhìn toàn cảnh từ trên không (Bird-eye)"""
        self.is_bird_eye = not self.is_bird_eye
        if self.is_bird_eye:
            if self.main_camera is not None and hasattr(self.main_camera, "set_bird_view_pos"):
                self.main_camera.set_bird_view_pos((0.0, 0.0))
            print(">> [CAMERA] Góc nhìn toàn cảnh (Bird-eye View): BẬT")
        else:
            print(">> [CAMERA] Góc nhìn toàn cảnh (Bird-eye View): TẮT -> Trở về bám sau xe")

    def track_car(self, car_idx: int, racers: list, car_names: list):
        """Chuyển camera bám theo xe chỉ định"""
        self.is_bird_eye = False
        self.follow_leader = False
        self.tracked_car_idx = max(0, min(car_idx, self.num_cars - 1))
        st = racers[self.tracked_car_idx]
        if self.main_camera is not None and st.raw_vehicle is not None:
            self.main_camera.track(st.raw_vehicle)
        status_tag = "ĐANG ĐUA" if st.active else "ĐÃ DỪNG"
        print(f">> [CAMERA] Đang theo dõi xe: {car_names[self.tracked_car_idx]} ({status_tag})")

    def cycle_car(self, step: int, racers: list, car_names: list):
        """Chuyển tới xe kế tiếp hoặc xe trước đó"""
        self.is_bird_eye = False
        self.follow_leader = False
        nxt = (self.tracked_car_idx + step) % self.num_cars
        self.track_car(nxt, racers, car_names)

    def track_leader(self, standings: list, racers: list, car_names: list):
        """Chuyển camera bám theo xe đang dẫn đầu cuộc đua"""
        self.is_bird_eye = False
        self.follow_leader = True
        active_order = [c for c in standings if racers[c].active]
        leader_idx = active_order[0] if active_order else standings[0]
        self.tracked_car_idx = leader_idx
        if self.main_camera is not None and racers[leader_idx].raw_vehicle is not None:
            self.main_camera.track(racers[leader_idx].raw_vehicle)
        print(f">> [CAMERA] Đang theo dõi XE DẪN ĐẦU: {car_names[leader_idx]}")

    def update_view(self, standings: list, racers: list):
        """Cập nhật bám theo xe ở mỗi frame"""
        if self.is_bird_eye:
            if self.main_camera is not None and hasattr(self.main_camera, "set_bird_view_pos"):
                self.main_camera.set_bird_view_pos((0.0, 0.0))
            return self.tracked_car_idx

        target = self.tracked_car_idx
        if self.follow_leader or not racers[target].active:
            active_order = [c for c in standings if racers[c].active]
            if active_order:
                target = active_order[0]
                self.tracked_car_idx = target

        raw = racers[target].raw_vehicle
        if self.main_camera is not None and getattr(self.main_camera, "current_track_agent", None) is not raw:
            self.main_camera.track(raw)
        return target

    def get_hud_camera_status(self) -> str:
        """Chuỗi hiển thị trạng thái camera trên HUD (dùng deg tránh lỗi ký tự font)"""
        if self.is_bird_eye:
            return "[TOAN CANH TREN KHONG (Bird-eye)] | [B] De tro lai xe"

        cur_rot_deg = 0.0
        if self.main_camera is not None:
            cur_rot_deg = float(np.rad2deg(getattr(self.main_camera, "mouse_rotate", 0.0)))

        if abs(abs(cur_rot_deg) - 180.0) < 35.0:
            view_str = "NHIN SAU (180 deg)"
        elif abs(cur_rot_deg) < 35.0:
            view_str = "NHIN TRUOC (0 deg)"
        elif cur_rot_deg > 0:
            view_str = f"NHIN TRAI ({cur_rot_deg:+.0f} deg)"
        else:
            view_str = f"NHIN PHAI ({cur_rot_deg:+.0f} deg)"

        leader_tag = " (DAN DAU)" if self.follow_leader else ""
        return f"[{view_str}]{leader_tag} | Chuot 360 | Chuot giua reset | [C]/[Tab] Doi xe"

    def destroy(self):
        if self.main_camera is not None and self._orig_update_mouse_info is not None:
            self.main_camera.update_mouse_info = self._orig_update_mouse_info


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-car racing demo: Human (Gamepad/Keyboard) vs PPO AI")
    parser.add_argument("--model", type=str, default="models/ref_111s/best_agent.pt",
                        help="Weights PPO (60D multicar hoặc 35D ref_111s)")
    parser.add_argument("--num-cars", type=int, default=6, choices=range(2, 11),
                        help="Tổng số xe trên đường đua (2 đến 10 xe, mặc định 6)")
    parser.add_argument("--lane-num", type=int, default=3, choices=[2, 3],
                        help="Số làn đường (3 làn = 13.5m, 2 làn = 9.0m)")
    parser.add_argument("--vehicle", type=str, default="gt3_anchor", choices=["gt3_anchor", "gt3"])
    parser.add_argument("--laps", type=int, default=1, help="Số vòng đua")
    parser.add_argument("--spectate", action="store_true",
                        help="Chế độ khán giả (tất cả các xe do AI tự động điều khiển)")
    parser.add_argument("--grid", choices=["abreast", "staggered"], default="staggered",
                        help="Xếp xe: 'staggered' (so le F1 cách 8m) hoặc 'abreast' (ngang hàng)")
    parser.add_argument("--no-respawn", dest="respawn", action="store_false",
                        help="Tắt cơ chế hồi sinh xe người chơi khi tai nạn/văng ra cỏ (mặc định luôn BẬT hồi sinh)")
    parser.set_defaults(respawn=True)
    return parser.parse_args()


def load_multicar_or_single_agent(agent: PPOAgent, model_path: str):
    ckpt = torch.load(model_path, map_location=agent.device)
    sd = ckpt.get("model_state_dict", ckpt)
    w_shape = sd["actor_mean.0.weight"].shape
    if w_shape[1] == agent.ac.state_dict()["actor_mean.0.weight"].shape[1]:
        agent.load(model_path)
        print(f">> Nạp AI đa tác tử ({w_shape[1]}D) từ: {model_path}")
    else:
        warm_start_from_single(agent, model_path, base_dim=BASE_OBS_DIM)
        print(f">> Nạp AI đơn tác tử ({w_shape[1]}D -> {agent.ac.state_dict()['actor_mean.0.weight'].shape[1]}D warm-start) từ: {model_path}")


def main():
    args = parse_args()
    has_human_player = not args.spectate

    # 1. Khởi tạo môi trường đua đa xe 3D
    env = MultiCarRacingEnv(
        num_cars=args.num_cars,
        lane_num=args.lane_num,
        vehicle_name=args.vehicle,
        use_render=True,
        max_laps=args.laps,
        max_steps=3000 * args.laps,
        grid=args.grid,
    )

    # 2. Bộ điều khiển người chơi & Camera (khởi tạo trễ sau env.reset() để ShowBase/messenger sẵn sàng)
    gamepad: Optional[XboxGamepadController] = None
    keyboard_controller: Optional[KeyboardController] = None
    cam_controller: Optional[MultiCarCameraController] = None
    engine: Optional[Any] = None
    initialized_controllers = False

    # 3. Nạp mô hình AI
    agent = PPOAgent(state_dim=env.obs_dim, action_dim=3)
    if os.path.exists(args.model):
        load_multicar_or_single_agent(agent, args.model)
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy file {args.model}, AI sẽ chạy với trọng số ngẫu nhiên!")

    # 4. Tên hiển thị các xe
    car_names = [("YOU (Player)" if has_human_player else "AI-1")] + [f"AI-{k + 1}" for k in range(1, args.num_cars)]

    # Trạng thái game
    game_state = {
        "paused": False,
        "reset_requested": False,
    }

    print("=" * 75)
    print(f">> GIẢI ĐUA ĐA XE 3D: {args.num_cars} XE | {args.laps} VÒNG | {args.lane_num} LÀN ({env.half_width * 2:.1f}m)")
    if has_human_player:
        print(f">> BẠN ĐIỀU KHIỂN XE #1 (Tay cầm Xbox hoặc Bàn phím [W-A-S-D]) VS {args.num_cars - 1} XE AI PPO")
    else:
        print(f">> CHẾ ĐỘ KHÁN GIẢ: TOÀN BỘ {args.num_cars} XE DO AI PPO ĐIỀU KHIỂN")
    print("=" * 75)
    print("HƯỚNG DẪN ĐIỀU KHIỂN & CAMERA:")
    if has_human_player:
        print("  - [Tay cầm Xbox]: Cần analog trái = Lái | Cò RT = Ga | Cò LT = Phanh | Nút Y = Hồi sinh xe")
        print("  - [Bàn phím]: [W] / [Mũi tên Lên] = Ga | [S] / [Mũi tên Xuống] = Phanh | [A]/[D] = Lái")
        print("  - [Hồi sinh xe]: Phím [Backspace] hoặc [X] = Đặt lại xe về tim đường ngay lập tức")
    print("  - [Camera 360]: Lia chuột tự do nhìn quanh xe 360 độ | [Chuột giữa]: Reset về chính diện (0 deg)")
    print("  - [Đổi xe theo dõi]: Phím [Tab] hoặc [N]: Xe kế tiếp | [M]: Xe trước | [0] hoặc [L]: Xe dẫn đầu")
    print("  - [Góc nhìn]: [B]: Toàn cảnh Bird-eye | [Q]: Chase Cam")
    print("  - [Hệ thống MetaDrive]: [F]: Bật/Tắt khóa FPS (Unlimited/Realtime) | [C]: Chụp ảnh | [P]: Tạm dừng | [R]: Reset")
    print("=" * 75)

    try:
        while True:
            obs, info = env.reset()
            game_state["reset_requested"] = False
            step_idx = 0

            # Khởi tạo bộ điều khiển & camera sau reset() lần đầu khi Panda3D messenger đã sẵn sàng
            if not initialized_controllers:
                engine = getattr(env.env, "engine", None)
                if has_human_player:
                    if XboxGamepadController.is_available():
                        gamepad = XboxGamepadController()
                    keyboard_controller = KeyboardController(pygame_control=False)

                if engine is not None:
                    cam = MultiCarCameraController(engine=engine, num_cars=args.num_cars)
                    cam_controller = cam

                    # Đăng ký phím bấm tương tác hệ thống
                    if hasattr(engine, "accept"):
                        # Phím đổi xe: [Tab] hoặc [N]: xe kế tiếp | [M]: xe trước | [0] hoặc [L]: xe dẫn đầu
                        engine.accept("tab", lambda: cam.cycle_car(1, env.racers, car_names))
                        engine.accept("n", lambda: cam.cycle_car(1, env.racers, car_names))
                        engine.accept("N", lambda: cam.cycle_car(1, env.racers, car_names))
                        engine.accept("m", lambda: cam.cycle_car(-1, env.racers, car_names))
                        engine.accept("M", lambda: cam.cycle_car(-1, env.racers, car_names))
                        engine.accept("0", lambda: cam.track_leader(env.standings(), env.racers, car_names))
                        engine.accept("l", lambda: cam.track_leader(env.standings(), env.racers, car_names))
                        engine.accept("L", lambda: cam.track_leader(env.standings(), env.racers, car_names))

                        # Chuột giữa (mouse2): Reset góc nhìn về 0 deg
                        engine.accept("mouse2", cam.reset_to_front)

                        # Phím [B]: Toàn cảnh Bird-eye | [Q]: Chase Cam
                        engine.accept("b", cam.toggle_bird_eye)
                        engine.accept("B", cam.toggle_bird_eye)
                        engine.accept("q", lambda: cam.track_car(cam.tracked_car_idx, env.racers, car_names))
                        engine.accept("Q", lambda: cam.track_car(cam.tracked_car_idx, env.racers, car_names))

                        # Phím hồi sinh xe: [Backspace] hoặc [X]
                        # (TUYỆT ĐỐI KHÔNG gán [F] để giữ [F] nguyên bản cho MetaDrive bật/tắt giới hạn FPS force_fps.toggle)
                        engine.accept("backspace", lambda: env.respawn_car(0))
                        engine.accept("x", lambda: env.respawn_car(0))
                        engine.accept("X", lambda: env.respawn_car(0))

                        # Phím [P]: Tạm dừng | [R]: Reset vòng đua
                        def toggle_pause():
                            game_state["paused"] = not game_state["paused"]
                            print(f">> [TRẬN ĐẤU] {'TẠM DỪNG' if game_state['paused'] else 'TIẾP TỤC'}")
                        engine.accept("p", toggle_pause)
                        engine.accept("P", toggle_pause)

                        def request_reset():
                            game_state["reset_requested"] = True
                            print(">> [TRẬN ĐẤU] Đã yêu cầu khởi động lại vòng đua mới...")
                        engine.accept("r", request_reset)
                        engine.accept("R", request_reset)

                initialized_controllers = True

            actions = np.zeros((args.num_cars, 3), dtype=np.float32)

            while True:
                step_idx += 1
                if game_state["reset_requested"]:
                    break

                if game_state["paused"]:
                    env.render(text={"TRẠNG THÁI": "ĐANG TẠM DỪNG (Nhấn [Space] để tiếp tục đua)"})
                    time.sleep(0.05)
                    continue

                active = env.active_mask()

                # 1. Đọc hành động AI PPO cho các xe AI
                ai_start_idx = 0 if not has_human_player else 1
                ai_indices = [i for i in np.flatnonzero(active) if i >= ai_start_idx]
                if ai_indices:
                    for i in ai_indices:
                        actions[i], _, _ = agent.select_action(obs[i], deterministic=True)

                # 2. Đọc hành động Người chơi (Xe 0)
                if has_human_player and active[0]:
                    player_raw = env.racers[0].raw_vehicle
                    action_applied = False

                    if gamepad is not None and gamepad.connected:
                        if hasattr(gamepad, "is_respawn_pressed") and gamepad.is_respawn_pressed():
                            env.respawn_car(0)

                        gp_act = gamepad.process_input(player_raw)
                        if abs(gp_act[0]) > 0.04 or gp_act[1] > -0.90 or gp_act[2] > -0.90:
                            actions[0] = gp_act
                            action_applied = True

                    if not action_applied and keyboard_controller is not None:
                        kb_act = keyboard_controller.process_input(player_raw)
                        steer = float(kb_act[0])
                        pedal = float(kb_act[1])
                        if pedal >= 0.0:
                            thr = pedal
                            brk = 0.0
                        else:
                            thr = 0.0
                            brk = -pedal
                        actions[0] = np.array([steer, 2.0 * thr - 1.0, 2.0 * brk - 1.0], dtype=np.float32)

                # 3. Bước mô phỏng môi trường
                obs, rewards, dones, truncated, infos = env.step(actions)

                # Tự động hồi sinh cho tất cả các xe (Người chơi & AI) nếu gặp tai nạn / văng ra ngoài cỏ
                if args.respawn:
                    for i, st in enumerate(env.racers):
                        if dones[i] and not st.finished:
                            env.respawn_car(i)
                            dones[i] = False

                # 4. Cập nhật Leaderboard & HUD
                order = env.standings()
                leader_st = env.racers[order[0]]
                leader_dist = env.race_distance(leader_st)

                board_p1 = []
                board_p2 = []
                for pos, car_idx in enumerate(order):
                    st = env.racers[car_idx]
                    short_name = "YOU" if (has_human_player and car_idx == 0) else f"AI{car_idx + 1}"
                    if st.finished:
                        status = f"FIN {sum(st.lap_times):5.1f}s"
                    elif not st.active:
                        status = "OUT" if st.termination_reason == "crash_or_track_limit" else "DNF"
                    else:
                        gap = leader_dist - env.race_distance(st)
                        status = f"-{gap:4.0f}m" if pos > 0 else "LEAD"
                    item = f"P{pos + 1}:{short_name} ({status})"
                    if pos < 5:
                        board_p1.append(item)
                    else:
                        board_p2.append(item)

                # 5. Cập nhật Camera và lấy xe đang quan sát
                cam_car = cam_controller.update_view(order, env.racers) if cam_controller is not None else 0
                tracked_racer = env.racers[cam_car]
                active_count = int(np.sum(active))
                cam_status_str = cam_controller.get_hud_camera_status() if cam_controller is not None else "Normal Cam"

                # Trích xuất telemetry của xe đang quan sát
                tracked_act = actions[cam_car]
                steer_val = tracked_act[0]
                thr_val = 0.5 * (float(tracked_act[1]) + 1.0)
                brk_val = 0.5 * (float(tracked_act[2]) + 1.0)
                act_str = f"Lái: {steer_val:+.2f} | Ga: {thr_val:.2f} | Phanh: {brk_val:.2f}"

                lap_time_str = f"{infos[cam_car].get('lap_time', 0.0):.2f}s" if tracked_racer.active else "DỪNG"

                # Hiển thị khoảng cách Delta tới xe dẫn đầu
                gap_to_lead_m = leader_dist - env.race_distance(tracked_racer)
                gap_str = "DẪN ĐẦU" if gap_to_lead_m <= 1.0 else f"-{gap_to_lead_m:4.0f} mét"

                # 6. Render Text HUD Sim Racing chuẩn nhánh anchor
                hud = {
                    "SIM RACING MULTICAR": f"Lap: {min(tracked_racer.current_lap, args.laps)}/{args.laps} | Step: {step_idx} | Xe: {car_names[cam_car]}",
                    "Góc nhìn Camera": cam_status_str,
                    "Telemetry": f"Vận tốc: {tracked_racer.speed_km_h:5.1f} km/h | Lap Time: {lap_time_str} | Gap: {gap_str}",
                    "Cơ cấu chấp hành": act_str,
                    "Vị trí thi đấu": f"Hạng P{order.index(cam_car) + 1}/{args.num_cars} | Còn lại: {active_count}/{args.num_cars} xe | Va chạm: {tracked_racer.contacts}",
                    "Phím nóng": "[Backspace]/[X] Hồi sinh | [Tab]/[N] Đổi xe | [0] Dẫn đầu | [F] Bật/Tắt khóa FPS | [P] Tạm dừng",
                    "BẢNG XẾP HẠNG TOP": " | ".join(board_p1),
                }
                if board_p2:
                    hud["BẢNG XẾP HẠNG SAU"] = " | ".join(board_p2)

                env.render(text=hud)

                if truncated or not active.any():
                    break

            if game_state["reset_requested"]:
                continue

            # In kết quả sau khi kết thúc chặng
            style = env.compute_driving_style_metrics()
            print("\n" + "=" * 70)
            print("KẾT QUẢ CHẶNG ĐUA:")
            for pos, car_idx in enumerate(env.standings()):
                st = env.racers[car_idx]
                res_str = f"{sum(st.lap_times):.2f}s" if st.finished else (st.termination_reason or "Chưa hoàn thành")
                print(f"  Hạng {pos + 1:2d}: {car_names[car_idx]:20s} | Kết quả: {res_str:15s} | Va chạm: {st.contacts} lần")
            print("-" * 70)
            print(
                f"Chỉ số phong cách lái (§6): Đổi ga-phanh = {style['pedal_sw_per_100']:.1f}/100 | "
                f"Đổi hướng lái = {style['steer_rev_per_100']:.1f}/100 | "
                f"Góc trượt beta p95 = {style['beta_p95_deg']:.1f} deg"
            )
            print("=" * 70)
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n>> Dừng cuộc đua từ bàn phím.")
    finally:
        if cam_controller is not None:
            cam_controller.destroy()
        if gamepad is not None:
            gamepad.close()
        env.close()


if __name__ == "__main__":
    main()
