"""
Script Đánh Giá Toàn Diện (Comprehensive Benchmark & Evaluation)
Thực hiện đánh giá mô hình PPO trên nhiều vòng đua với các seed và địa hình ngẫu nhiên.
Xuất số liệu thống kê chuẩn mực để đưa vào báo cáo và slide bảo vệ đồ án REL.
"""

import os
import sys
import argparse

# UTF-8 encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from agents.ppo_agent import PPOAgent
from agents.optimal_agent import OptimalRacingAgent
from training.evaluator import RacingEvaluator


class OptimalPolicyAdapter:
    """
    Bọc OptimalRacingAgent (ghost giải tích) theo giao diện select_action(obs, deterministic) mà RacingEvaluator dùng,
    để nhánh 1 của ablation (D30) được đánh giá bằng đúng cùng evaluator và cùng metric với hai nhánh PPO.
    Agent giải tích có trạng thái nội (bộ lọc lái, chỉ số điểm gần nhất), nên reset mỗi khi env bắt đầu episode mới.
    """

    def __init__(self, env: MetaRacingEnv, opt_agent: OptimalRacingAgent):
        self.env = env
        self.opt_agent = opt_agent

    def select_action(self, obs, deterministic: bool = True):
        if int(getattr(self.env, "current_step", 0)) == 0:
            self.opt_agent.reset()
        return self.opt_agent.select_action(obs, vehicle=self.env.car)

def parse_args():
    default_model = "models/p1_main/best_agent.pt" if os.path.exists("models/p1_main/best_agent.pt") else "models/best_agent.pt"
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Agent")
    parser.add_argument("--model", type=str, default=default_model, help="File weights cần đánh giá")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "optimal"], help="'ppo' (checkpoint) hoặc 'optimal' (ghost giải tích, nhánh 1 của ablation D30; cần --map và --vehicle)")
    parser.add_argument("--episodes", type=int, default=20, help="Số episode thử nghiệm (mặc định 20)")
    parser.add_argument("--laps", type=int, default=2, help="Số vòng đua mỗi episode (mặc định 2: Lap 1 Standing + Lap 2 Flying)")
    parser.add_argument("--vehicle", type=str, default=None, choices=["gt3", "gt3_real", "sedan"], help="Loại xe đánh giá (mặc định: tự đọc từ checkpoint metadata hoặc 'gt3')")
    parser.add_argument("--map", type=str, default=None, help="Bản đồ đua (mặc định: tự đọc từ checkpoint metadata hoặc 'silverstone')")
    parser.add_argument("--obs-mode", type=str, default=None, choices=["pure", "line"], help="Chế độ quan sát (mặc định: tự đọc từ checkpoint metadata hoặc 'pure')")
    parser.add_argument("--action-mode", type=str, default=None, choices=["split", "combined"], help="Action 3 trục hoặc pedal gộp (mặc định: đọc từ checkpoint metadata hoặc 'split')")
    parser.add_argument("--offtrack-mode", type=str, default=None, choices=["terminate", "penalty"], help="Xử lý ra ngoài đường (mặc định: đọc từ checkpoint metadata hoặc 'terminate')")
    parser.add_argument("--obs-noise", type=float, default=0.0, help="Độ lệch chuẩn nhiễu nhân trên quan sát sigma (VD: 0.01, 0.02, 0.05)")
    parser.add_argument("--action-delay", type=int, default=0, help="Số bước trễ hành động k (VD: 0, 1, 2)")
    parser.add_argument("--out", type=str, default=None, help="File lưu kết quả JSON (tương đương --save-json)")
    parser.add_argument("--save-json", type=str, default=None, help="File lưu kết quả JSON")
    return parser.parse_args()


def _resolve_config_with_meta(args):
    """
    Thứ tự ưu tiên (D34): CLI flag > checkpoint meta > legacy default ('silverstone', 'gt3', 'pure').
    Nếu người dùng truyền cả flag lẫn checkpoint có meta mà hai bên khác nhau -> in cảnh báo rõ ràng.
    """
    meta = PPOAgent.peek_meta(args.model) if (args.agent == "ppo" and os.path.exists(args.model)) else {}

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
    out_path = args.save_json or args.out or "logs/evaluation_results.json"
    return resolved_map, resolved_vehicle, resolved_obs_mode, out_path, meta


def main():
    args = parse_args()
    resolved_map, resolved_vehicle, resolved_obs_mode, out_path, ckpt_meta = _resolve_config_with_meta(args)
    race_car = get_vehicle(resolved_vehicle)
    
    print("=" * 70)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ CHỈ SỐ HỌC THUẬT (ACADEMIC BENCHMARK EVALUATION)")
    print("=" * 70)
    print(f"  - Agent: {'OPTIMAL GHOST (analytical reference)' if args.agent == 'optimal' else 'PPO'}")
    if args.agent == "ppo":
        print(f"  - Model file: {args.model}")
    if ckpt_meta:
        print(f"  - Checkpoint meta: {ckpt_meta}")
    print(f"  - Map layout: {resolved_map}")
    print(f"  - Vehicle Spec: {race_car.cfg.name} ({resolved_vehicle})")
    print(f"  - Obs Mode: {resolved_obs_mode} {'[PRIVILEGED RACING LINE PRIOR]' if resolved_obs_mode == 'line' else '[PURE SENSOR]'}")
    print(f"  - Số episode: {args.episodes} (Laps mỗi episode: {args.laps})")
    if args.obs_noise > 0.0 or args.action_delay > 0:
        print(f"  - Perturbation: obs_noise={args.obs_noise:.3f}, action_delay={args.action_delay} step(s)")
    print("=" * 70)

    env = MetaRacingEnv(
        map_type=resolved_map,
        use_render=False,
        num_scenarios=args.episodes,
        start_seed=100,
        vehicle=race_car,
        max_laps=args.laps,
        obs_mode=resolved_obs_mode,
        action_mode=args.action_mode,
        offtrack_mode=args.offtrack_mode,
    )
    obs_shape = getattr(env.observation_space, "shape", None)
    act_shape = getattr(env.action_space, "shape", None)
    obs_dim = int(obs_shape[0]) if obs_shape is not None else 35
    act_dim = int(act_shape[0]) if act_shape is not None else 3

    if args.agent == "optimal":
        env.reset()
        opt_agent = OptimalRacingAgent.from_benchmark(env.get_benchmark_engine(), vehicle_cfg=race_car.cfg, v_norm=float(env.preset.v_norm))
        agent = OptimalPolicyAdapter(env, opt_agent)
        print(">> Đánh giá ghost giải tích OptimalRacingAgent (không có trọng số mạng)")
    else:
        agent = PPOAgent(state_dim=obs_dim, action_dim=act_dim, logstd_max=env.preset.logstd_max)
        if os.path.exists(args.model):
            agent.load(args.model)
            print(f">> Nạp thành công weights từ {args.model} (State Dim: {obs_dim})")
        else:
            print(f">> [CẢNH BÁO] Không tìm thấy file {args.model}! Đánh giá agent ngẫu nhiên (chưa huấn luyện)...")

    evaluator = RacingEvaluator(env=env, agent=agent)
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        deterministic=True,
        verbose=True,
        obs_noise=args.obs_noise,
        action_delay=args.action_delay,
    )
    results["agent"] = args.agent
    results["model_path"] = args.model if args.agent == "ppo" else None
    results["map"] = resolved_map
    results["vehicle"] = resolved_vehicle
    results["obs_mode"] = resolved_obs_mode
    results["privileged_prior"] = bool(resolved_obs_mode == "line")
    results["checkpoint_meta"] = ckpt_meta
    env.close()

    evaluator.print_summary(results)
    evaluator.save_json(results, out_path)
    print(f">> Dữ liệu chi tiết đã lưu vào: {out_path}\n")

if __name__ == "__main__":
    main()
