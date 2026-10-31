"""
ĐUA 4-6 XE TRÊN SILVERSTONE HALF: KHÁN GIẢ (SPECTATE) HOẶC TAY CẦM XBOX VS AI PPO SELF-PLAY
- Hỗ trợ nạp trực tiếp cả model đa tác tử 60D (sau finetune) lẫn model đơn tác tử 35D (models/ref_111s/best_agent.pt).
- Va chạm vật lý thật giữa các xe. Bị loại khi cả 4 bánh rời mặt nhựa hoặc lật xe.
- HUD hiển thị bảng xếp hạng trực tiếp P1-P6 và khoảng cách tới xe dẫn đầu.
- Cuối mỗi cuộc đua in bảng chỉ số lái theo mục §6 (pedal_sw/100, steer_rev/100, beta_p95).

Cách chạy:
    python demo/race_multicar.py --spectate --model models/ref_111s/best_agent.pt
    python demo/race_multicar.py --spectate --model models/multicar_A/best_agent.pt
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.multicar_env import MultiCarRacingEnv, BASE_OBS_DIM
from game_env.gamepad import XboxGamepad
from agents.ppo_agent import PPOAgent
from training.train_multicar import warm_start_from_single


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-car racing demo: Spectate or Xbox gamepad vs PPO AI")
    parser.add_argument("--model", type=str, default="models/ref_111s/best_agent.pt", help="Weights PPO (60D multicar hoặc 35D ref_111s)")
    parser.add_argument("--num-cars", type=int, default=6)
    parser.add_argument("--lane-num", type=int, default=3, help="Số làn đường (3 làn = 13.5m, 2 làn = 9.0m)")
    parser.add_argument("--vehicle", type=str, default="gt3_anchor", choices=["gt3_anchor", "gt3"])
    parser.add_argument("--laps", type=int, default=1, help="Số vòng đua")
    parser.add_argument("--spectate", action="store_true", help="Xe 0 cũng do AI lái (chế độ khán giả)")
    parser.add_argument("--steer-gamma", type=float, default=1.0, help="Đường cong lái tay cầm")
    parser.add_argument("--grid", choices=["abreast", "staggered"], default="abreast",
                        help="Xếp xe ngang hàng (mỗi làn 1 xe) hoặc so le kiểu F1")
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
    gamepad = None if args.spectate else XboxGamepad(steer_gamma=args.steer_gamma)

    env = MultiCarRacingEnv(
        num_cars=args.num_cars,
        lane_num=args.lane_num,
        vehicle_name=args.vehicle,
        use_render=True,
        max_laps=args.laps,
        max_steps=3000 * args.laps,
        grid=args.grid,
    )
    agent = PPOAgent(state_dim=env.obs_dim, action_dim=3)
    if os.path.exists(args.model):
        load_multicar_or_single_agent(agent, args.model)
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy {args.model}, AI chạy trọng số ngẫu nhiên!")

    names = ["YOU" if gamepad is not None else "AI-0"] + [f"AI-{k}" for k in range(1, args.num_cars)]
    print("=" * 70)
    print(f">> ĐUA {args.num_cars} XE ({args.vehicle}) | {args.laps} vòng | đường {env.half_width * 2:.1f}m ({args.lane_num} làn)")
    if gamepad is not None:
        print(f">> Tay cầm: {gamepad.name} | [Joystick trái] lái | [RT] ga | [LT] phanh")
    print(f">> Camera: [1]-[{args.num_cars}] theo xe tương ứng | [N]/[Tab] xe kế tiếp | [M] xe trước | [0] xe dẫn đầu | [Esc] thoát")
    print("=" * 70)

    # Camera: mặc định theo xe 0; spectate hoặc khi xe đang theo dõi bị loại -> tự chuyển sang xe dẫn đầu
    follow = {"car": 0, "leader": gamepad is None}
    engine = None

    def follow_car(k: int):
        if k < 0:
            follow["leader"] = True
            print(">> [CAMERA] Theo xe dẫn đầu")
            return
        follow["leader"] = False
        follow["car"] = k
        if env.racers[k].active:
            print(f">> [CAMERA] Theo {names[k]}")
        else:
            print(f">> [CAMERA] {names[k]} đã bị loại -> tạm theo xe dẫn đầu")

    def cycle_car(step: int):
        active = [c for c in range(args.num_cars) if env.racers[c].active]
        if not active:
            return
        cur = follow.get("shown", follow["car"])
        nxt = min(active, key=lambda c: (c - cur - step) % args.num_cars)
        follow_car(nxt)

    def bind_camera_keys():
        for k in range(min(args.num_cars, 9)):
            engine.accept(str(k + 1), follow_car, [k])
        engine.accept("0", follow_car, [-1])
        engine.accept("n", cycle_car, [1])
        engine.accept("tab", cycle_car, [1])
        engine.accept("m", cycle_car, [-1])

    def update_camera():
        target = follow["car"]
        if follow["leader"] or not env.racers[target].active:
            active_order = [c for c in env.standings() if env.racers[c].active]
            if active_order:
                target = active_order[0]
        raw = env.racers[target].raw_vehicle
        if engine.main_camera is not None and engine.main_camera.current_track_agent is not raw:
            engine.main_camera.track(raw)
        follow["shown"] = target
        return target

    try:
        while True:
            obs, info = env.reset()
            if engine is None:
                engine = env.env.engine
                bind_camera_keys()
            actions = np.zeros((args.num_cars, 3), dtype=np.float32)
            while True:
                active = env.active_mask()
                ai_cars = [i for i in np.flatnonzero(active) if i != 0 or gamepad is None]
                if ai_cars:
                    for i in ai_cars:
                        actions[i], _, _ = agent.select_action(obs[i], deterministic=True)
                if gamepad is not None and active[0]:
                    actions[0] = gamepad.read()
                obs, rewards, dones, truncated, infos = env.step(actions)

                order = env.standings()
                leader_d = env.race_distance(env.racers[order[0]])
                board = []
                for pos, car in enumerate(order):
                    st = env.racers[car]
                    if st.finished:
                        status = f"FIN {sum(st.lap_times):6.1f}s"
                    elif not st.active:
                        status = "OUT" if st.termination_reason == "crash_or_track_limit" else "DNF"
                    else:
                        status = f"-{leader_d - env.race_distance(st):5.0f}m" if pos > 0 else "LEADER"
                    board.append(f"P{pos + 1} {names[car]:5s} {status}")
                cam_car = update_camera()
                me = env.racers[cam_car]
                out_cars = [names[st.car_id] for st in env.racers if not st.active and not st.finished]
                hud = {
                    "Camera": f"{names[cam_car]}{' (leader)' if follow['leader'] else ''} | [1]-[{args.num_cars}] / [N] [M] doi xe, [0] dan dau",
                    "Out": ", ".join(out_cars) if out_cars else "-",
                    "RACE": f"Lap {min(me.current_lap, args.laps)}/{args.laps} | {infos[cam_car].get('lap_time', 0.0):.1f}s" if me.active else "RACE: xe đã dừng",
                    "Speed": f"{me.speed_km_h:.0f} km/h | Contacts: {me.contacts}",
                    "Position": f"P{order.index(cam_car) + 1}/{args.num_cars}",
                    "Standings": " | ".join(board[:3]),
                    "": " | ".join(board[3:]),
                }
                env.render(text=hud)

                if truncated or not env.active_mask().any():
                    break

            style = env.compute_driving_style_metrics()
            print("\n" + "=" * 60)
            print("KẾT QUẢ CUỘC ĐUA:")
            for pos, car in enumerate(env.standings()):
                st = env.racers[car]
                result = f"{sum(st.lap_times):.2f}s" if st.finished else st.termination_reason or "chưa về đích"
                print(f"  P{pos + 1} {names[car]:5s} {result} | va chạm {st.contacts}")
            print("-" * 60)
            print(
                f"CHỈ SỐ LÁI (§6): Pedal đổi ga↔phanh = {style['pedal_sw_per_100']:.1f}/100 bước (mục tiêu <= 5) | "
                f"Lái đổi chiều = {style['steer_rev_per_100']:.1f}/100 bước (mục tiêu <= 40) | "
                f"Góc trượt β p95 = {style['beta_p95_deg']:.1f}° (mục tiêu <= 4°)"
            )
            print("=" * 60)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n>> Dừng cuộc đua.")
    finally:
        if gamepad is not None:
            gamepad.close()
        env.close()


if __name__ == "__main__":
    main()

