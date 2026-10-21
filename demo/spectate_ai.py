"""
CHẾ ĐỘ XEM AI BIỂU DIỄN (CINEMATIC AI SPECTATOR)
Cho phép quan sát AI tự động điều khiển xe trên các bản đồ 3D phức tạp
với nhiều góc quay camera điện ảnh và hiển thị tia quét LiDAR thời gian thực.

Cách chạy:
    conda activate rel_racing
    python demo/spectate_ai.py --model models/best_agent.pt --map 3
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
from agents.ppo_agent import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="AI Spectator Mode in MetaDrive 3D")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="Đường dẫn weights của AI")
    parser.add_argument("--map", type=str, default="silverstone", help="Bản đồ đua ('silverstone' hoặc 'racing')")
    parser.add_argument("--traffic", type=float, default=0.0, help="Mật độ chướng ngại vật/xe khác (0.0 - 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed tạo bản đồ")
    parser.add_argument("--stochastic", action="store_true", help="Chạy ở chế độ Stochastic (hành vi thực chiến lúc huấn luyện hoàn thành vòng đua)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🎬 CHẾ ĐỘ XEM AI BIỂU DIỄN 3D (CINEMATIC SPECTATOR)")
    print("=" * 70)
    print("Phím điều khiển camera & quan sát:")
    print("  - [C]: Chuyển góc nhìn camera (Chase cam / Top-down / Cockpit)")
    print("  - [B]: Góc nhìn toàn cảnh từ trên cao (Bird-eye View)")
    print("  - [R]: Reset xe về vạch xuất phát")
    print("  - [H]: Bật menu hướng dẫn của engine")
    print("  - [Esc]: Thoát")
    print("=" * 70)

    # Khởi tạo môi trường có render 3D
    env = MetaRacingEnv(
        use_render=True,
        traffic_density=args.traffic,
        map_config=args.map,
        start_seed=args.seed,
    )

    # Nạp Agent tương thích với không gian quan sát
    obs_shape = env.observation_space.shape
    state_dim = int(obs_shape[0]) if obs_shape is not None else 35
    act_shape = env.action_space.shape
    action_dim = int(act_shape[0]) if act_shape is not None else 3
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    if os.path.exists(args.model):
        print(f">> Đang nạp weights từ: {args.model}")
        agent.load(args.model)
        print(">> Nạp mô hình AI thành công!")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy {args.model}. AI sẽ chạy thử nghiệm ngẫu nhiên.")

    try:
        obs, info = env.reset()
        step = 0
        lap_count = 1
        
        while True:
            step += 1
            # AI quyết định hành động (Deterministic hoặc Stochastic)
            is_deterministic = not args.stochastic
            action, _, _ = agent.select_action(obs, deterministic=is_deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            speed_kmh = info.get("speed_km_h", info.get("speed_kmh", 0.0))
            comp = info.get("route_completion", 0.0) * 100.0
            steer = action[0]
            if len(action) >= 3:
                throttle = 0.5 * (float(action[1]) + 1.0)
                brake = 0.5 * (float(action[2]) + 1.0)
                action_str = f"Steer: {steer:+.2f} | Thr: {throttle:.2f} | Brk: {brake:.2f}"
            else:
                throttle = action[1]
                action_str = f"Steer: {steer:+.2f} | Pedal: {throttle:+.2f}"
            
            delta_t = info.get("delta_time", 0.0)
            delta_str = f"+{delta_t:.2f}s" if delta_t >= 0 else f"{delta_t:.2f}s"
            delta_color = info.get("delta_color", "gray")
            
            mode_str = "DETERMINISTIC" if is_deterministic else "STOCHASTIC (THỰC CHIẾN)"
            # Text HUD
            hud_text = {
                "RL AGENT SPECTATOR": f"Lap: {lap_count} | Step: {step} [{mode_str}]",
                "Delta Bar": f"[{delta_color.upper()}] {delta_str}",
                "Current Lap": f"{info.get('lap_time', 0.0):.2f}s",
                "Best Lap": f"{info.get('best_lap_time', 0.0):.2f}s",
                "Speed": f"{speed_kmh:.1f} km/h",
                "Completion": f"{comp:.1f}%",
                "Action": action_str,
                "Reward": f"{reward:+.2f}",
            }
            env.render(text=hud_text)
            
            if terminated or truncated:
                reason = info.get("termination_reason", "unknown")
                is_win = reason == "lap_completed" or info.get("arrive_dest", False)
                if is_win:
                    print(f"🎉 [VỀ ĐÍCH THÀNH CÔNG] Vòng {lap_count} hoàn thành tại step {step}! (Completion: {comp:.1f}%, Lap Time: {info.get('lap_time', 0.0):.2f}s, Macro Reward: {info.get('macro_reward', 0.0):+.1f}đ)")
                else:
                    print(f"💥 [KẾT THÚC] Vòng {lap_count} dừng tại step {step}: Lý do = {reason} (Completion: {comp:.1f}%)")
                lap_count += 1
                step = 0
                time.sleep(1.0)
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n>> Dừng chế độ xem biểu diễn.")
    finally:
        env.close()
        print(">> Đã đóng cửa sổ mô phỏng.")

if __name__ == "__main__":
    main()
