"""
CHẾ ĐỘ ĐUA TRỰC TIẾP: NGƯỜI CHƠI BẤM PHÍM VS RL AI AGENT
Mở cửa sổ đồ hoạ 3D Panda3D 60FPS:
  - Người chơi (Player): Điều khiển xe bằng phím [W - A - S - D] hoặc [Mũi tên].
  - Tác tử RL (AI Agent): Tự động lái xe bằng mạng PPO Continuous Control đã huấn luyện.
  - Phân định thắng thua trực tiếp theo thời gian thực.

Cách chạy:
    conda activate rel_racing
    python demo/play_vs_ai.py --model models/best_agent.pt
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

from game_env.multiagent_env import MultiAgentRacingEnv
from game_env.vehicle import get_vehicle
from agents.ppo_agent import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Human vs AI 3D Racing Benchmark")
    parser.add_argument("--model", type=str, default="models/best_agent.pt", help="Đường dẫn file weights của AI")
    parser.add_argument("--map", type=str, default="O", help="Bản đồ đua ('O' cho oval track hoặc 'S', 'C', '3')")
    parser.add_argument("--vehicle", type=str, default="gt3", choices=["gt3", "sedan"], help="Loại xe đua ('gt3' hoặc 'sedan')")
    parser.add_argument("--seed", type=int, default=42, help="Bản đồ ngẫu nhiên")
    return parser.parse_args()

def main():
    args = parse_args()
    race_car = get_vehicle(args.vehicle)
    
    print("=" * 70)
    print(">> KHOI DONG GIAI DUA 3D: HUMAN VS AI (REINFORCEMENT LEARNING)")
    print("=" * 70)
    print("Quy tắc điều khiển:")
    print("  - [W] / [Mũi tên Lên]: Tăng ga (Throttle)")
    print("  - [S] / [Mũi tên Xuống]: Thắng / Lùi (Brake / Reverse)")
    print("  - [A] / [Mũi tên Trái]: Rẽ trái (Steer Left)")
    print("  - [D] / [Mũi tên Phải]: Rẽ phải (Steer Right)")
    print("  - [C]: Chuyển góc nhìn camera theo dõi Xe Người hoặc Xe AI")
    print("  - [B]: Góc nhìn toàn cảnh từ trên cao (Bird-eye View)")
    print("  - [R]: Reset lại vòng đua mới")
    print("  - [Esc]: Thoát trận đấu")
    print("=" * 70)

    # Tự động phát hiện state_dim từ checkpoint
    ckpt_dim = 35
    if os.path.exists(args.model):
        try:
            import torch
            ckpt = torch.load(args.model, map_location="cpu")
            ckpt_dim = int(ckpt.get("state_dim", 35))
        except Exception:
            ckpt_dim = 35

    ai_agent = PPOAgent(state_dim=ckpt_dim, action_dim=2)
    if os.path.exists(args.model):
        print(f">> Đang nạp mô hình AI từ: {args.model} (State Dim: {ckpt_dim})")
        ai_agent.load(args.model)
        print(">> Nạp trọng số AI thành công!")
    else:
        print(f">> [CẢNH BÁO] Không tìm thấy file {args.model}. AI sẽ chạy với trọng số ngẫu nhiên!")

    # Khởi tạo môi trường 2 xe
    racing_env = MultiAgentRacingEnv(
        use_render=True,
        map_config=args.map,
        start_seed=args.seed,
        vehicle=race_car,
    )
    racing_env.compact_obs = (ckpt_dim == 35)

    try:
        obs_dict, info_dict = racing_env.reset(seed=args.seed)
        agent_names = list(obs_dict.keys())
        player_id = agent_names[0]
        ai_id = agent_names[1]
        racing_env.set_vehicle_colors(player_id, ai_id)
        
        print(f">> Gán vai trò xe: [Player (Xanh): {player_id}] vs [AI Agent (Đỏ): {ai_id}]")
        print(">> Hãy click vào cửa sổ game 3D để bắt đầu đua!")

        winner = None
        step_idx = 0
        start_time = time.time()

        while True:
            step_idx += 1
            actions = {}
            
            # 1. Cung cấp hành động cho tất cả agent đang hiện diện trên đường đua
            active_agent_ids = list(racing_env.agents.keys())
            for aid in active_agent_ids:
                if aid == player_id:
                    player_vehicle = racing_env.vehicles.get(player_id)
                    if player_vehicle is not None:
                        actions[aid] = racing_env.get_player_action(player_vehicle)
                    else:
                        actions[aid] = [0.0, 0.0]
                elif aid == ai_id:
                    if aid in obs_dict:
                        ai_action, _, _ = ai_agent.select_action(obs_dict[aid], deterministic=True)
                        actions[aid] = ai_action
                    else:
                        actions[aid] = [0.0, 0.0]
                else:
                    actions[aid] = [0.0, 0.0]
                
            # 2. Bước mô phỏng vật lý 3D
            obs_dict, rewards, terminations, truncations, infos = racing_env.step(actions)
            
            # Trích xuất telemetry hiển thị HUD & Delta Bar
            p_speed = infos.get(player_id, {}).get("speed_km_h", infos.get(player_id, {}).get("velocity", 0.0) * 3.6)
            ai_speed = infos.get(ai_id, {}).get("speed_km_h", infos.get(ai_id, {}).get("velocity", 0.0) * 3.6)
            p_comp = infos.get(player_id, {}).get("route_completion", 0.0) * 100.0
            ai_comp = infos.get(ai_id, {}).get("route_completion", 0.0) * 100.0
            
            p_delta = infos.get(player_id, {}).get("delta_time", 0.0)
            p_color = infos.get(player_id, {}).get("delta_color", "gray").upper()
            ai_delta = infos.get(ai_id, {}).get("delta_time", 0.0)
            ai_color = infos.get(ai_id, {}).get("delta_color", "gray").upper()
            
            p_delta_str = f"+{p_delta:.2f}s" if p_delta > 0 else f"{p_delta:.2f}s"
            ai_delta_str = f"+{ai_delta:.2f}s" if ai_delta > 0 else f"{ai_delta:.2f}s"
            
            # Kiểm tra trạng thái hoàn thành / va chạm / lệch đường
            p_crash = infos.get(player_id, {}).get("crash", False) or infos.get(player_id, {}).get("crash_sidewalk", False)
            ai_crash = infos.get(ai_id, {}).get("crash", False) or infos.get(ai_id, {}).get("crash_sidewalk", False)
            p_out = infos.get(player_id, {}).get("out_of_road", False)
            ai_out = infos.get(ai_id, {}).get("out_of_road", False)
            p_arrive = infos.get(player_id, {}).get("arrive_dest", False)
            ai_arrive = infos.get(ai_id, {}).get("arrive_dest", False)

            # Phân định kết quả trận đấu ngay khi có biến cố
            if winner is None:
                if p_arrive and not ai_arrive:
                    winner = "[VICTORY] BAN (PLAYER) DA VE DICH TRUOC!"
                elif ai_arrive and not p_arrive:
                    winner = "[DEFEAT] AI AGENT DA VE DICH TRUOC!"
                elif (p_crash or p_out) and not (ai_crash or ai_out):
                    winner = "[DEFEAT] AI CHIEN THANG (Nguoi choi bi tai nan / lech duong)!"
                elif (ai_crash or ai_out) and not (p_crash or p_out):
                    winner = "[VICTORY] BAN CHIEN THANG (AI bi tai nan / lech duong)!"
                elif (p_crash or p_out) and (ai_crash or ai_out):
                    winner = "[DRAW] CA HAI XE DEU BI TAI NAN!"

            # Render text HUD lên màn hình 3D kèm Delta Bar Sim Racing
            hud_text = {
                "SIM RACING HUD": f"Lap Time: {step_idx * 0.02:5.2f}s",
                "Player (You)": f"Speed: {p_speed:5.1f} km/h | Progress: {p_comp:4.1f}% | Delta: {p_delta_str} [{p_color}]",
                "AI Agent": f"Speed: {ai_speed:5.1f} km/h | Progress: {ai_comp:4.1f}% | Delta: {ai_delta_str} [{ai_color}]",
                "Status": winner if winner else "RACING IN PROGRESS...",
            }
            racing_env.render(text=hud_text)

            # In console định kỳ mỗi 50 steps
            if step_idx % 50 == 0:
                print(f"[Race {step_idx * 0.02:5.2f}s] You: {p_speed:5.1f} km/h (Δ {p_delta_str} {p_color}) | AI: {ai_speed:5.1f} km/h (Δ {ai_delta_str} {ai_color}) | {winner or 'Dang dua...'}")

            # Kết thúc trận khi có xe thắng/thua hoặc hết thời gian
            round_ended = (
                winner is not None
                or terminations.get("__all__", False)
                or truncations.get("__all__", False)
                or (player_id in terminations and terminations[player_id])
                or (ai_id in terminations and terminations[ai_id])
            )

            if round_ended:
                print("\n" + "=" * 60)
                print(f"KET QUA VONG DUA: {winner or 'Ket thuc vong dua'}")
                print(f"  - Ban (Player):  Tien do {p_comp:.1f}% | Toc do cuoi: {p_speed:.1f} km/h")
                print(f"  - AI Agent:      Tien do {ai_comp:.1f}% | Toc do cuoi: {ai_speed:.1f} km/h")
                print("=" * 60)
                
                # Giữ màn hình hiển thị kết quả 2 giây
                for _ in range(30):
                    racing_env.render(text=hud_text)
                    time.sleep(0.05)
                    
                print(">> Dang khoi dong vong dua moi...")
                obs_dict, info_dict = racing_env.reset(seed=args.seed)
                agent_names = list(obs_dict.keys())
                player_id = agent_names[0]
                ai_id = agent_names[1]
                racing_env.set_vehicle_colors(player_id, ai_id)
                winner = None
                step_idx = 0

    except KeyboardInterrupt:
        print("\n>> Người dùng đã thoát cuộc đua.")
    finally:
        racing_env.close()
        print(">> Đã đóng cửa sổ đua 3D an toàn.")

if __name__ == "__main__":
    main()
