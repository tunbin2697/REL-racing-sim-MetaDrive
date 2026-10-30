"""
CHẾ ĐỘ XEM AI BIỂU DIỄN (CINEMATIC AI SPECTATOR)
Cho phép quan sát AI tự động điều khiển xe trên các bản đồ 3D phức tạp
với nhiều góc quay camera điện ảnh và hiển thị tia quét LiDAR thời gian thực.

Cách chạy:
    conda activate rel_racing
    python demo/spectate_ai.py --model models/best_agent.pt --ghost
"""

import os
import sys
import time
import argparse
import numpy as np

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from game_env.ghost_car import GhostCar
from game_env.visualizer import RacingLineVisualizer
from game_env.camera import RacingCameraController
from agents.ppo_agent import PPOAgent
from agents.optimal_agent import OptimalRacingAgent

def parse_args():
    default_model = "models/p1_main/best_agent.pt" if os.path.exists("models/p1_main/best_agent.pt") else "models/best_agent.pt"
    parser = argparse.ArgumentParser(description="Cinematic AI Spectator & Ghost Car Racing in MetaDrive 3D")
    parser.add_argument("--model", type=str, default=default_model, help="Đường dẫn weights của PPO AI")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "optimal"], help="Loại tác tử lái chính: 'ppo' (mạng neural PPO) hoặc 'optimal' (tác tử giải tích tối ưu lý thuyết)")
    parser.add_argument("--ghost", action="store_true", help="Bật xe bóng ma đối chứng (Ghost Car) điều khiển bởi OptimalRacingAgent")
    parser.add_argument("--manual", action="store_true", help="Chế độ tự lái bằng bàn phím (W-A-S-D) để so tài trực tiếp với Ghost Car")
    parser.add_argument("--vehicle", type=str, default=None, choices=["gt3", "gt3_real", "sedan"], help="Loại xe biểu diễn (mặc định: tự đọc từ checkpoint metadata hoặc 'gt3')")
    parser.add_argument("--map", type=str, default=None, help="Bản đồ đua (mặc định: tự đọc từ checkpoint metadata hoặc 'silverstone')")
    parser.add_argument("--obs-mode", type=str, default=None, choices=["pure", "line"], help="Chế độ quan sát (mặc định: tự đọc từ checkpoint metadata hoặc 'pure')")
    parser.add_argument("--action-mode", type=str, default=None, choices=["split", "combined"], help="Action 3 trục hoặc pedal gộp (mặc định: đọc từ checkpoint metadata hoặc 'split')")
    parser.add_argument("--offtrack-mode", type=str, default=None, choices=["terminate", "penalty"], help="Xử lý ra ngoài đường (mặc định: đọc từ checkpoint metadata hoặc 'terminate')")
    parser.add_argument("--traffic", type=float, default=0.0, help="Mật độ chướng ngại vật/xe khác (0.0 - 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed tạo bản đồ")
    parser.add_argument("--laps", type=int, default=2, help="Số vòng đua liên tục mỗi lượt chạy (mặc định 2: Lap 1 Standing Start + Lap 2 Flying Lap)")
    parser.add_argument("--stochastic", action="store_true", help="Chạy PPO ở chế độ Stochastic (hành vi thực chiến)")
    parser.add_argument("--hide-line", action="store_true", help="Bắt đầu mô phỏng với Racing Line 3D ẩn (bật lại bằng phím [L])")
    return parser.parse_args()


def _resolve_spectate_config(args):
    """
    Thứ tự ưu tiên (D34): CLI flag > checkpoint meta > legacy default ('silverstone', 'gt3', 'pure').
    """
    use_ckpt = (not args.manual) and (args.agent == "ppo") and os.path.exists(args.model)
    meta = PPOAgent.peek_meta(args.model) if use_ckpt else {}

    def resolve_field(cli_val, meta_key, legacy_default, label):
        meta_val = meta.get(meta_key, None)
        if cli_val is not None:
            if meta_val is not None and str(cli_val).lower() != str(meta_val).lower():
                print(
                    f"⚠️ [CẢNH BÁO METADATA D34] Cờ CLI --{label}='{cli_val}' KHÁC với checkpoint meta "
                    f"'{meta_key}'='{meta_val}' trong '{args.model}'! Ưu tiên dùng giá trị CLI '{cli_val}'."
                )
            return cli_val
        if meta_val is not None:
            return meta_val
        return legacy_default

    resolved_map = resolve_field(args.map, "map", "silverstone", "map")
    resolved_vehicle = resolve_field(args.vehicle, "vehicle", "gt3", "vehicle")
    resolved_obs_mode = resolve_field(args.obs_mode, "obs_mode", "pure", "obs-mode")
    args.action_mode = resolve_field(args.action_mode, "action_mode", "split", "action-mode")
    args.offtrack_mode = resolve_field(args.offtrack_mode, "offtrack_mode", "terminate", "offtrack-mode")
    return resolved_map, resolved_vehicle, resolved_obs_mode, meta


def main():
    args = parse_args()
    resolved_map, resolved_vehicle, resolved_obs_mode, ckpt_meta = _resolve_spectate_config(args)
    race_car = get_vehicle(resolved_vehicle)
    
    print("=" * 75)
    print("🎬 CHẾ ĐỘ XEM BIỂU DIỄN 3D & GHOST CAR RACING (CINEMATIC SPECTATOR)")
    print("=" * 75)
    if ckpt_meta:
        print(f"Checkpoint Meta: {ckpt_meta}")
    print(f"Bản đồ: {resolved_map} | Chế độ quan sát: {resolved_obs_mode}")
    print(f"Xe đua: {race_car.cfg.name} (Brake: {race_car.cfg.max_brake_force}N, Grip: {race_car.cfg.wheel_friction})")
    print(f"Tác tử điều khiển xe chính: {'TỰ LÁI BÀN PHÍM (MANUAL W-A-S-D)' if args.manual else args.agent.upper()}")
    print(f"Xe bóng ma đối chứng (Ghost Car): {'BẬT (OPTIMAL AGENT)' if args.ghost else 'TẮT'}")
    print("Điều khiển camera & quan sát 360 độ tự do:")
    print("  - [Lia chuột (Mouse Look)]: Xoay tự do 360° quanh xe (không giới hạn góc, không tự reset)")
    print("  - [Chuột giữa (Middle Click)]: Reset nhanh góc nhìn về thẳng phía trước (0 deg)")
    print("  - [G]: Đổi xe theo dõi (Main Car <-> Ghost Car)")
    print("  - [L]: Bật / Tắt hiển thị Racing Line 3D (Đường đua tối ưu)")
    print("  - [C]: Chụp ảnh màn hình (lưu file .png tại thư mục gốc dự án)")
    print("  - [B]: Góc nhìn toàn cảnh từ trên cao (Bird-eye View)")
    print("  - [R]: Reset xe về vạch xuất phát")
    print("  - [H]: Bật / Tắt menu trợ giúp phím tắt trên màn hình")
    if args.manual:
        print("  - [W/S/A/D]: Tăng ga / Thắng / Đánh lái")
    print("  - [Esc]: Thoát")
    print("=" * 75)

    # Khởi tạo môi trường có render 3D
    env = MetaRacingEnv(
        use_render=True,
        manual_control=args.manual,
        traffic_density=args.traffic,
        map_config=resolved_map,
        start_seed=args.seed,
        vehicle=race_car,
        max_laps=args.laps,
        obs_mode=resolved_obs_mode,
        action_mode=args.action_mode,
        offtrack_mode=args.offtrack_mode,
    )

    # Khởi tạo Tác tử chính
    ppo_agent = None
    opt_agent = None
    if not args.manual:
        if args.agent == "optimal":
            print(">> Khởi tạo Theoretical Optimal Benchmark Agent (Convex QP + Lipp & Boyd)...")
            env.reset()
            opt_agent = OptimalRacingAgent.from_benchmark(env.get_benchmark_engine(), vehicle_cfg=race_car.cfg, v_norm=float(env.preset.v_norm))
        else:
            obs_shape = env.observation_space.shape
            state_dim = int(obs_shape[0]) if obs_shape is not None else 35
            act_shape = env.action_space.shape
            action_dim = int(act_shape[0]) if act_shape is not None else 3
            ppo_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, logstd_max=env.preset.logstd_max)
            if os.path.exists(args.model):
                print(f">> Đang nạp weights từ: {args.model}")
                ppo_agent.load(args.model)
                print(">> Nạp mô hình AI thành công!")
            else:
                print(f">> [CẢNH BÁO] Không tìm thấy {args.model}. AI sẽ chạy thử nghiệm ngẫu nhiên.")

    # Khởi tạo Ghost Car nếu được yêu cầu
    ghost_car = None
    if args.ghost:
        print(">> Kích hoạt Ghost Car (Bóng ma đối chứng) trên đường đua 3D...")
        ghost_car = GhostCar(vehicle=race_car)

    try:
        obs, info = env.reset(seed=args.seed, options={"start_mode": "standing"})
        if opt_agent is not None:
            opt_agent.reset()
            
        # Khởi tạo 3D Racing Line Visualizer
        racing_line_visualizer = None
        bench_engine = env.get_benchmark_engine()
        if bench_engine is not None and getattr(bench_engine, "optimal_line_xy", None) is not None:
            engine = getattr(env.env, "engine", None)
            if engine is not None:
                racing_line_visualizer = RacingLineVisualizer(
                    engine=engine,
                    points_xy=bench_engine.optimal_line_xy,
                    z=0.15,
                    thickness=3.5,
                    visible=not args.hide_line,
                )
                print(">> [3D VISUALIZER] Khởi tạo Racing Line 3D thành công! Nhấn phím [L] để Bật/Tắt.")

        # Spawn Ghost Car cùng vị trí vạch xuất phát và cùng vận tốc 0 km/h với xe chính
        if ghost_car is not None:
            ghost_car.spawn_from_env(env)

        # Khởi tạo Bộ điều khiển Camera Thể thao Đa góc nhìn (Rear-view / Ghost Cam)
        camera_controller = None
        engine = getattr(env.env, "engine", None)
        if engine is not None:
            camera_controller = RacingCameraController(
                engine=engine,
                main_vehicle=env.car,
                ghost_car=ghost_car,
            )
            print(">> [CAMERA] Đã khởi tạo hệ thống Camera đa góc nhìn: Lia chuột 360 | Chuột giữa reset | [G] Ghost Cam.")

        step = 0
        lap_count = 1
        
        while True:
            step += 1
            
            # 1. Tác tử chính quyết định hành động
            if args.manual:
                # MetaDrive tự lấy input bàn phím khi manual_control=True
                action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            elif opt_agent is not None:
                action, _, _ = opt_agent.select_action(obs, vehicle=env.car)
            elif ppo_agent is not None:
                is_deterministic = not args.stochastic
                res_act = ppo_agent.select_action(obs, deterministic=is_deterministic)
                action = res_act[0]
            else:
                action = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            # 2. Ghost car thực hiện tính toán và ra lệnh
            if ghost_car is not None and ghost_car.active:
                ghost_car.step(dt=0.10)

            # 3. Bước mô phỏng môi trường
            obs, reward, terminated, truncated, info = env.step(action)
            
            speed_kmh = info.get("speed_km_h", info.get("speed_kmh", 0.0))
            # route_completion của env là tiến độ cả lượt đua (chia cho max_laps); HUD hiện thêm tiến độ vòng hiện tại
            comp = info.get("route_completion", 0.0) * 100.0
            n_laps = max(1, int(info.get("max_laps", args.laps)))
            cur_lap_idx = max(1, int(info.get("current_lap", 1)))
            lap_comp = float(np.clip(comp * n_laps - 100.0 * (cur_lap_idx - 1), 0.0, 100.0))
            # Lựa chọn action/tốc độ hiển thị dựa trên xe camera đang theo dõi
            g_act = ghost_car.last_action if ghost_car is not None else None
            viewing_ghost = bool(
                camera_controller is not None
                and getattr(camera_controller, "target_name", "main") == "ghost"
                and ghost_car is not None and ghost_car.active
            )
            if viewing_ghost and g_act is not None:
                g_steer = g_act[0]
                g_thr = 0.5 * (float(g_act[1]) + 1.0)
                g_brk = 0.5 * (float(g_act[2]) + 1.0)
                action_str = f"[GHOST] Steer: {g_steer:+.2f} | Thr: {g_thr:.2f} | Brk: {g_brk:.2f}"
            else:
                steer = action[0]
                prefix = "[MAIN]" if ghost_car is not None else ""
                if len(action) >= 3:
                    throttle = 0.5 * (float(action[1]) + 1.0)
                    brake = 0.5 * (float(action[2]) + 1.0)
                    action_str = f"{prefix} Steer: {steer:+.2f} | Thr: {throttle:.2f} | Brk: {brake:.2f}".strip()
                else:
                    throttle = action[1]
                    action_str = f"{prefix} Steer: {steer:+.2f} | Pedal: {throttle:+.2f}".strip()
            
            delta_t = info.get("delta_time", 0.0)
            delta_str = f"+{delta_t:.2f}s" if delta_t >= 0 else f"{delta_t:.2f}s"
            delta_color = info.get("delta_color", "gray")
            
            if args.manual:
                mode_str = "MANUAL HUMAN"
            elif opt_agent is not None:
                mode_str = "OPTIMAL QP BENCHMARK"
            else:
                mode_str = "DETERMINISTIC" if not args.stochastic else "STOCHASTIC"

            # Thông báo khi xe hoàn thành từng vòng đua
            if info.get("lap_completed", False):
                comp_lap_num = int(info.get("completed_lap_number", 1))
                comp_lap_t = float(info.get("completed_lap_time", info.get("lap_time", 0.0)))
                lap_tag = "🚀 FLYING LAP" if comp_lap_num >= 2 else "🏁 STANDING LAP"
                print(f"🏁 [HOÀN THÀNH LAP {comp_lap_num}] {lap_tag} | Thời gian: {comp_lap_t:.2f}s | Tốc độ TB: {info.get('mean_speed_km_h', 0.0):.1f} km/h")

            # 4. Tạo Text HUD với thông số Sim Racing chuyên nghiệp
            cur_lap_display = info.get("current_lap", 1)
            max_laps_display = info.get("max_laps", args.laps)
            hud_text = {
                "RACING SPECTATOR": f"Lap: {cur_lap_display}/{max_laps_display} | Step: {step} [{mode_str}]",
                "Delta Bar": f"[{delta_color.upper()}] {delta_str}",
                "Current Lap": f"{info.get('lap_time', 0.0):.2f}s",
                "Best Lap": f"{info.get('best_lap_time', 0.0):.2f}s",
                "Speed": f"[GHOST] {ghost_car.speed_km_h:.1f} km/h" if viewing_ghost and ghost_car is not None else f"[MAIN] {speed_kmh:.1f} km/h",
                "Completion (Main)": f"Lap {lap_comp:.1f}% | Race {comp:.1f}%" if n_laps > 1 else f"{comp:.1f}%",
                "Action": action_str,
            }
            
            # Thêm trạng thái Racing Line 3D vào HUD
            if racing_line_visualizer is not None:
                hud_text["3D Racing Line"] = f"[{'ON' if racing_line_visualizer.visible else 'OFF'}] (Press [L])"

            # Thêm trạng thái Camera vào HUD
            if camera_controller is not None:
                hud_text["Camera View"] = camera_controller.get_hud_status()

            # Thêm telemetry của Ghost Car vào HUD nếu đang kích hoạt
            if ghost_car is not None and ghost_car.active:
                g_tel = ghost_car.get_telemetry(main_pos=env.car.position, main_heading=env.car.heading_theta)
                gap_m = g_tel.get("gap_distance_m", 0.0)
                is_behind = g_tel.get("is_behind", False)
                pos_str = f"PHÍA SAU {-gap_m:4.1f}m" if is_behind else f"PHÍA TRƯỚC {+gap_m:4.1f}m"
                hint_str = "(Lia chuột nhìn sau)" if is_behind else "(Bấm [G] Ghost Cam)"
                hud_text["[GHOST CAR]"] = f"{pos_str} | Speed: {ghost_car.speed_km_h:5.1f} km/h | {hint_str}"

            env.render(text=hud_text)
            
            if terminated or truncated:
                reason = info.get("termination_reason", "unknown")
                is_win = reason == "lap_completed" or info.get("arrive_dest", False)
                if is_win:
                    print(f"🎉 [HOÀN THÀNH LƯỢT ĐUA {lap_count}] Về đích trọn vẹn {max_laps_display} vòng tại step {step}! (Best Lap: {info.get('best_lap_time', 0.0):.2f}s)")
                    bench = env.evaluate_current_lap_benchmark()
                    if bench:
                        print(f"   >> Benchmark: MAE Racing Line = {bench.get('racing_line_mae_m', 0.0):.2f}m | Apex Acc = {bench.get('apex_cutting_accuracy_pct', 0.0):.1f}% | Best Lap Diff = {bench.get('lap_time_diff_s', 0.0):+.2f}s")
                else:
                    print(f"💥 [KẾT THÚC] Lượt đua {lap_count} dừng tại step {step} (Lap {cur_lap_display}/{max_laps_display}): Lý do = {reason} (Lap: {lap_comp:.1f}% | Race: {comp:.1f}%)")
                    
                lap_count += 1
                step = 0
                time.sleep(1.0)
                if ghost_car is not None:
                    ghost_car.destroy(getattr(env.env, "engine", None))
                obs, info = env.reset(seed=args.seed, options={"start_mode": "standing"})
                if opt_agent is not None:
                    opt_agent.reset()
                if ghost_car is not None:
                    ghost_car.spawn_from_env(env)
                if camera_controller is not None:
                    camera_controller.on_reset(main_vehicle=env.car, ghost_car=ghost_car)

    except KeyboardInterrupt:
        print("\n>> Dừng chế độ xem biểu diễn.")
    finally:
        if racing_line_visualizer is not None:
            racing_line_visualizer.destroy()
        if camera_controller is not None:
            camera_controller.destroy()
        if ghost_car is not None:
            ghost_car.destroy(getattr(getattr(env, "env", env), "engine", None))
        env.close()
        print(">> Đã đóng cửa sổ mô phỏng an toàn.")

if __name__ == "__main__":
    main()
