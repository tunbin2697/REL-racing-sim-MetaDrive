"""
CHẾ ĐỘ XEM POLICY 3D (bản v2) - chạy được mọi loại policy qua agents/policy_loader:
  - "rule" / "rule-fast"            : Policy luật (Pure Pursuit + profile tốc độ theo bán kính cua)
  - models/final_agent.pt           : PPO v1
  - models/v2/laptime/latest_agent.pt : PPO v2 residual (mặc định, về đích 100%, ~105.1s)
HUD hiển thị thêm tốc độ cho phép theo vật lý, lap time hiện tại / tốt nhất và số vòng hoàn thành.

Cách chạy:
    python demo/spectate_v2.py
    python demo/spectate_v2.py --policy rule
    python demo/spectate_v2.py --policy models/final_agent.pt
Phím: [C] đổi camera, [B] nhìn từ trên cao, [Esc] thoát.
"""

import os
import sys
import time
import argparse

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from agents.policy_loader import load_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Xem policy (rule / PPO v1 / PPO v2) chạy trong MetaDrive 3D")
    parser.add_argument("--policy", type=str, default="models/v2/laptime/latest_agent.pt",
                        help="'rule', 'rule-fast' hoặc đường dẫn checkpoint .pt (v1 hoặc v2)")
    parser.add_argument("--laps", type=int, default=0, help="Số vòng muốn xem (0 = chạy liên tục)")
    parser.add_argument("--realtime", action="store_true", help="Giữ tốc độ phát bằng thời gian thực (0.1s/step)")
    parser.add_argument("--max-steps", type=int, default=0, help="Dừng sau số step này (0 = không giới hạn, dùng để test)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"🎬 SPECTATE V2 | Policy: {args.policy}")
    print("   [C] đổi camera | [B] nhìn từ trên cao | [Esc] thoát")
    print("=" * 70)

    policy = load_policy(args.policy)
    policy_name = type(policy).__name__
    env = MetaRacingEnv(use_render=True)

    lap_idx, finishes, best_lap, total_steps = 1, 0, None, 0
    try:
        obs, info = env.reset(options={"force_start": True})
        while True:
            step_start = time.time()
            action, _, _ = policy.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            speed = info.get("speed_km_h", 0.0)
            v_allow = info.get("v_allow_km_h", 0.0)
            over = speed - v_allow
            env.render(text={
                "Policy": f"{policy_name} ({os.path.basename(args.policy)})",
                "Lap": f"{lap_idx} | Finishes: {finishes} | Best: {best_lap:.1f}s" if best_lap else f"{lap_idx} | Finishes: {finishes}",
                "Lap Time": f"{info.get('lap_time', 0.0):.1f}s",
                "Speed": f"{speed:5.1f} km/h",
                "Allowed": f"{v_allow:5.1f} km/h ({'OVER ' + format(over, '.1f') if over > 0 else 'OK'})",
                "Progress": f"{info.get('route_completion', 0.0) * 100:.1f}%",
                "Action": f"Steer {action[0]:+.2f} | Throttle {action[1]:+.2f}",
            })

            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                lap_time = info.get("lap_time", 0.0)
                if reason == "lap_completed":
                    finishes += 1
                    best_lap = lap_time if best_lap is None else min(best_lap, lap_time)
                    print(f"🏁 Vòng {lap_idx}: VỀ ĐÍCH {lap_time:.1f}s")
                else:
                    print(f"💥 Vòng {lap_idx}: {reason} tại {info.get('track_s', 0.0):.0f}m ({lap_time:.1f}s)")
                if args.laps and lap_idx >= args.laps:
                    break
                lap_idx += 1
                time.sleep(1.0)
                obs, info = env.reset(options={"force_start": True})

            if args.max_steps and total_steps >= args.max_steps:
                break
            if args.realtime:
                time.sleep(max(0.0, env.dt - (time.time() - step_start)))
    except KeyboardInterrupt:
        print("\n>> Dừng xem.")
    finally:
        env.close()
        print(f">> Tổng kết: {finishes}/{lap_idx} vòng về đích" + (f", best lap {best_lap:.1f}s" if best_lap else ""))


if __name__ == "__main__":
    main()
