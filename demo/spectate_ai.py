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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from agents.ppo_agent import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="AI Spectator Mode in MetaDrive 3D")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="Đường dẫn weights của AI")
    parser.add_argument("--map", type=int, default=3, help="Độ dài bản đồ đua (số block cua)")
    parser.add_argument("--traffic", type=float, default=0.05, help="Mật độ chướng ngại vật/xe khác (0.0 - 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed tạo bản đồ")
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

    # Nạp Agent
    agent = PPOAgent(state_dim=259, action_dim=2)
    if os.path.exists(args.model):
        print(f">> Đang nạp weights từ: {args.model}")
        agent.load(args.model)
        print(">> Nạp mô hình AI thành công!")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy {args.model}. AI sẽ chạy thử nghiệm ngẫu nhiên.")

    # Khởi tạo môi trường có render 3D
    env = MetaRacingEnv(
        use_render=True,
        traffic_density=args.traffic,
        map_config=args.map,
        start_seed=args.seed,
    )

    try:
        obs, info = env.reset()
        step = 0
        lap_count = 1
        
        while True:
            step += 1
            # AI quyết định hành động tối ưu
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            speed_kmh = info.get("speed_kmh", 0.0)
            comp = info.get("route_completion", 0.0) * 100.0
            steer = action[0]
            throttle = action[1]
            
            # Text HUD
            hud_text = {
                "RL AGENT SPECTATOR": f"Lap: {lap_count} | Step: {step}",
                "Speed": f"{speed_kmh:.1f} km/h",
                "Completion": f"{comp:.1f}%",
                "Action": f"Steer: {steer:+.2f} | Throttle: {throttle:+.2f}",
                "Reward": f"{reward:+.2f}",
            }
            env.render()
            
            if terminated or truncated:
                print(f">> Vòng {lap_count} kết thúc tại step {step}! (Completion: {comp:.1f}%, Arrive: {info.get('arrive_dest', False)})")
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
