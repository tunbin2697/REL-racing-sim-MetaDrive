"""
SCRIPT BENCHMARK & DEBUG CHUYÊN SÂU TÁC TỬ TỐI ƯU (OPTIMAL AGENT DIAGNOSTIC TOOL)
Dùng để đối soát chi tiết từng step/phân đoạn:
- So sánh giá trị thực tế (v_curr, steer_applied, pedal) với giá trị lý thuyết (v_opt, target_line).
- Phát hiện các vị trí bị lố ga (overspeed), thả trôi (coasting), thiếu góc lái do kẹp khóa lái (lock clamped), hoặc lệch racing line.
- Báo cáo bảng thống kê top các phân đoạn lệch nhiều nhất để debug.

Cách chạy:
python scripts/benchmark_optimal_agent.py [--verbose] [--step-interval 50]
"""

import os
import sys
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from agents.optimal_agent import OptimalRacingAgent

# Danh sách các góc cua trọng điểm trên Silverstone để đánh dấu báo cáo
CORNER_SECTORS = [
    (0, 200, "Hamilton Straight (Start/Finish)"),
    (200, 380, "Turn 1-2 Abbey & Farm (Fast Right/Sweeper)"),
    (380, 520, "Turn 3-4 Village & The Loop (Hairpin)"),
    (520, 700, "Turn 5 Aintree & Wellington Straight"),
    (700, 950, "Turn 6-7 Brooklands & Luffield (Carousel)"),
    (950, 1200, "Woodcote & National Pit Straight"),
    (1200, 1400, "Turn 9 Copse (Fast 90° Right)"),
    (1400, 1650, "Turn 10-13 Maggots & Becketts (S-Bends)"),
    (1650, 1950, "Turn 14 Chapel & Hangar Straight (V-Max)"),
    (1950, 2200, "Turn 15 Stowe (Heavy Braking)"),
    (2200, 2500, "Turn 16-17 Vale & Club Chicane"),
    (2500, 2852, "Turn 18 Club Exit & Finish Straight"),
]

def get_sector_name(s_dist):
    for s_start, s_end, name in CORNER_SECTORS:
        if s_start <= s_dist < s_end:
            return name
    return "Track Sector"

def parse_args():
    parser = argparse.ArgumentParser(description="Optimal Agent Step-by-Step Benchmark")
    parser.add_argument("--step-interval", type=int, default=30, help="Khoảng bước in chi tiết bảng telemetry (mặc định 30 bước = 3.0s)")
    parser.add_argument("--verbose", action="store_true", help="In cảnh báo mỗi khi xe bị lố ga hoặc lệch line")
    parser.add_argument("--seed", type=int, default=42, help="Seed mô phỏng")
    return parser.parse_args()

def run_benchmark():
    args = parse_args()
    print("=" * 95)
    print("🏎️  BENCHMARK & KIỂM CHẨN TỪNG STEP: TÁC TỬ TỐI ƯU (OPTIMAL AGENT)")
    print("=" * 95)
    
    race_car = get_vehicle("gt3")
    env = MetaRacingEnv(use_render=False, vehicle=race_car)
    agent = OptimalRacingAgent()
    agent.reset()

    obs, info = env.reset(seed=args.seed)
    bench_engine = env.get_benchmark_engine()
    opt_line_xy = bench_engine.optimal_line_xy
    opt_speed_kmh = bench_engine.optimal_speed_kmh
    circuit_length = env.circuit_length

    logs = []
    step = 0
    terminated = False
    truncated = False
    
    print(f"\n{'Step':^6} | {'Dist(m)':^8} | {'V_curr':^7} | {'V_exp':^7} | {'ΔV':^7} | {'Steer_Act':^9} | {'LockLim':^7} | {'Thr':^5} | {'Brk':^5} | {'LineErr':^7} | {'Trạng thái':^18}")
    print("-" * 95)

    while not (terminated or truncated) and step < 3000:
        raw_action, steer_norm, net_pedal = agent.select_action(vehicle=env.car.raw_vehicle)
        
        # Trích xuất góc lái trước và sau khi qua extended_physics limiter
        curr_speed = float(env.car.speed_km_h)
        curr_pos = np.array(env.car.position[:2], dtype=np.float32)
        idx_near, current_s = env._get_track_s(curr_pos)
        
        # Expected optimal speed tại điểm gần nhất
        v_exp = float(opt_speed_kmh[min(idx_near, len(opt_speed_kmh)-1)])
        delta_v = curr_speed - v_exp
        
        # Sai số vị trí so với optimal racing line
        line_pt = opt_line_xy[min(idx_near, len(opt_line_xy)-1)]
        line_err = float(np.linalg.norm(curr_pos - line_pt))
        
        # Kiểm tra giới hạn khóa lái speed-sensitive hiện hành
        lock_scale, max_rate = env.car.ext_physics.calculate_steering_limits(curr_speed, env.car.cfg.max_steer_rate)
        steer_clamped = bool(abs(float(raw_action[0])) > lock_scale + 1e-3)
        
        # Step môi trường
        obs, reward, terminated, truncated, step_info = env.step(raw_action)
        step += 1
        
        steer_applied = float(env.car.last_steer)
        thr = float(step_info.get("throttle", 0.0))
        brk = float(step_info.get("brake", 0.0))
        
        # Xác định tag trạng thái
        status_tags = []
        if steer_clamped:
            status_tags.append("LOCK_CLAMP")
        if delta_v > 5.0:
            status_tags.append("OVERSPEED")
        elif delta_v < -15.0:
            status_tags.append("UNDERSPEED")
        if thr == 0.0 and brk == 0.0 and curr_speed > 30.0:
            status_tags.append("COASTING")
        if line_err > 2.0:
            status_tags.append("WIDE_LINE")
        if step_info.get("is_oversteering", False):
            status_tags.append("OVERSTEER")
            
        status_str = ", ".join(status_tags) if status_tags else "NORMAL"

        record = {
            "step": step,
            "dist": current_s,
            "v_curr": curr_speed,
            "v_exp": v_exp,
            "delta_v": delta_v,
            "steer_raw": float(raw_action[0]),
            "steer_applied": steer_applied,
            "lock_scale": lock_scale,
            "steer_clamped": steer_clamped,
            "throttle": thr,
            "brake": brk,
            "line_err": line_err,
            "status": status_str,
            "sector": get_sector_name(current_s),
        }
        logs.append(record)
        
        # In định kỳ theo interval hoặc in khi có lỗi bất thường nếu verbose
        should_print = (step % args.step_interval == 0) or (args.verbose and len(status_tags) >= 2)
        if should_print:
            print(f"{step:6d} | {current_s:7.1f}m | {curr_speed:6.1f} | {v_exp:6.1f} | {delta_v:+6.1f} | {steer_applied:+8.3f} | {lock_scale:6.2f} | {thr:4.2f} | {brk:4.2f} | {line_err:6.2f}m | {status_str:18}")

    env.close()

    # --- BÁO CÁO PHÂN TÍCH TỔNG HỢP ---
    print("\n" + "=" * 95)
    print("📊 BÁO CÁO TỔNG KẾT & PHÂN TÍCH KỸ THUẬT:")
    print("=" * 95)
    
    total_steps = len(logs)
    lap_time = total_steps * 0.1
    max_line_err = max(l["line_err"] for l in logs)
    mean_line_err = np.mean([l["line_err"] for l in logs])
    clamped_steps = sum(1 for l in logs if l["steer_clamped"])
    overspeed_steps = sum(1 for l in logs if l["delta_v"] > 5.0)
    coast_steps = sum(1 for l in logs if "COASTING" in l["status"])
    
    print(f"  • Thời gian hoàn thành vòng: {lap_time:.2f}s (Cận dưới lý thuyết: {bench_engine.theoretical_best_lap_time:.2f}s)")
    print(f"  • Sai số Racing Line trung bình: {mean_line_err:.2f}m (Lớn nhất: {max_line_err:.2f}m)")
    print(f"  • Số bước bị KẸP GÓC LÁI (Lock Scale Clamped): {clamped_steps}/{total_steps} bước ({(clamped_steps/total_steps)*100:.1f}%)")
    print(f"  • Số bước bị LỐ TỐC ĐỘ (>5 km/h): {overspeed_steps}/{total_steps} bước ({(overspeed_steps/total_steps)*100:.1f}%)")
    print(f"  • Số bước THẢ TRÔI (Coasting ga=0, phanh=0): {coast_steps}/{total_steps} bước ({(coast_steps/total_steps)*100:.1f}%)")

    # Phân nhóm theo góc cua để tìm điểm nghẽn
    print("\n" + "-" * 95)
    print("📍 CHI TIẾT TỪNG PHÂN ĐOẠN / GÓC CUA:")
    print(f"{'Phân đoạn đường đua':<45} | {'V_avg':^7} | {'ΔV_avg':^7} | {'LineErr_Max':^11} | {'Kẹp Lái':^9}")
    print("-" * 95)
    
    for s_start, s_end, name in CORNER_SECTORS:
        sec_logs = [l for l in logs if s_start <= l["dist"] < s_end]
        if not sec_logs:
            continue
        sec_v = np.mean([l["v_curr"] for l in sec_logs])
        sec_dv = np.mean([l["delta_v"] for l in sec_logs])
        sec_max_err = max([l["line_err"] for l in sec_logs])
        sec_clamp = sum(1 for l in sec_logs if l["steer_clamped"])
        clamp_str = f"{sec_clamp} steps" if sec_clamp > 0 else "-"
        print(f"{name:<45} | {sec_v:6.1f} | {sec_dv:+6.1f} | {sec_max_err:9.2f}m | {clamp_str:^9}")

    print("=" * 95)

if __name__ == "__main__":
    run_benchmark()
