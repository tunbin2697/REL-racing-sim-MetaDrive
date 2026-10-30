"""
RacingEvaluator: Module Đánh Giá Chuẩn OOP Dùng Chung (Academic Benchmark Evaluator)
Tuân thủ chuẩn Clean Code & DRY:
- Đóng gói toàn bộ logic đánh giá tác tử (Agent Evaluation Loop) cho cả quá trình huấn luyện và kiểm thử.
- Tự động trích xuất các chỉ số học thuật chuẩn rubric REL (Return, Lap Completion, Lap Times, Crash Locations).
- Tích hợp động cơ Benchmark Động lực học (Racing Line MAE/RMSE, Apex Accuracy, Friction Circle, Lap Efficiency).
"""

from typing import Dict, Any, List, Optional
from collections import deque
import os
import json
import numpy as np


class EvalPerturbation:
    """
    Wrapper nhiễu kiểm thử độ bền (Robustness Perturbation) tại thời điểm đánh giá (D33 / WI-21):
    - obs_noise: nhân từng chiều quan sát với (1 + N(0, sigma^2)) trước khi đưa vào agent.
    - action_delay: hàng đợi FIFO độ dài k bước (khởi tạo bằng k lệnh trung tính [0, -1, -1]).
    Không sửa MetaRacingEnv hay trọng số mạng.
    """

    def __init__(self, obs_noise: float = 0.0, action_delay: int = 0, seed: int = 12345):
        self.obs_noise = max(0.0, float(obs_noise))
        self.action_delay = max(0, int(action_delay))
        self.rng = np.random.default_rng(seed)
        self._queue: deque = deque()
        self.reset()

    def reset(self):
        self._queue.clear()
        neutral = np.array([0.0, -1.0, -1.0], dtype=np.float32)
        for _ in range(self.action_delay):
            self._queue.append(neutral.copy())

    def perturb_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_noise <= 0.0:
            return obs
        noise_factor = 1.0 + self.rng.normal(0.0, self.obs_noise, size=obs.shape)
        return (obs * noise_factor).astype(np.float32)

    def delay_action(self, action: np.ndarray) -> np.ndarray:
        act_arr = np.asarray(action, dtype=np.float32)
        if self.action_delay <= 0:
            return act_arr
        self._queue.append(act_arr.copy())
        return self._queue.popleft()


class RacingEvaluator:
    """
    Bộ đánh giá hiệu năng tác tử đua xe tự động.
    Được sử dụng chung bởi cả train_racing.py (định kỳ) và evaluate.py (kiểm thử độc lập).
    """

    def __init__(self, env: Any, agent: Any):
        self.env = env
        self.agent = agent

    def evaluate(
        self,
        num_episodes: int = 3,
        deterministic: bool = True,
        max_eval_steps_per_lap: int = 3000,
        verbose: bool = False,
        obs_noise: float = 0.0,
        action_delay: int = 0,
    ) -> Dict[str, Any]:
        """
        Thực hiện đánh giá tác tử qua num_episodes vòng đua độc lập.
        """
        eval_returns: List[float] = []
        eval_lengths: List[int] = []
        eval_completions: List[float] = []
        eval_times: List[float] = []
        eval_speeds: List[float] = []
        eval_max_speeds: List[float] = []
        eval_progress_rew: List[float] = []
        eval_steer_smooth_rew: List[float] = []
        eval_idle_pen: List[float] = []
        eval_macros_rew: List[float] = []
        eval_lap1_times: List[float] = []
        eval_lap2_times: List[float] = []
        eval_best_lap_times: List[float] = []
        clean_lap_times: List[float] = []
        crash_locations: List[str] = []
        crash_distances_m: List[float] = []
        crash_bins_50m: Dict[str, int] = {}
        crash_count: int = 0
        success_count: int = 0
        clean_ep_count: int = 0

        eval_line_mae: List[float] = []
        eval_line_rmse: List[float] = []
        eval_apex_acc: List[float] = []
        eval_corner_delta_v: List[float] = []
        eval_friction_util: List[float] = []
        eval_lap_efficiency: List[float] = []
        line_obs_lat_abs_m: List[float] = []
        line_obs_v_abs_kmh: List[float] = []
        line_obs_v_signed_kmh: List[float] = []
        episode_records: List[Dict[str, Any]] = []
        theoretical_best_lap_time: float = float(getattr(self.env, "theoretical_best_lap_time", 116.99))

        preset = getattr(self.env, "preset", None)
        time_scale = float(getattr(preset, "time_scale", 1.0))
        v_norm = float(getattr(preset, "v_norm", 33.33))
        lat_norm = float(getattr(preset, "lat_norm", 10.0))
        circuit_len = float(getattr(self.env, "circuit_length", 3417.8))
        max_laps = max(1, int(getattr(self.env, "max_laps", 1)))
        max_eval_steps = int(max_eval_steps_per_lap * time_scale * max_laps)

        perturbation = EvalPerturbation(obs_noise=obs_noise, action_delay=action_delay, seed=1000)

        for ep in range(num_episodes):
            num_scenarios = max(1, getattr(self.env, "num_scenarios", 1))
            target_seed = self.env.start_seed + (ep % num_scenarios)
            # 100% đánh giá từ xuất phát đứng (Lap 1: Standing Start -> Lap 2: Flying Lap tự nhiên)
            obs, info = self.env.reset(seed=target_seed, options={"force_start": True, "start_mode": "standing"})
            perturbation.reset()

            ep_return = 0.0
            ep_len = 0
            ep_progress = 0.0
            ep_smooth = 0.0
            ep_idle = 0.0
            ep_macro = 0.0
            speeds: List[float] = []
            ep_lap1_time: Optional[float] = None
            ep_lap2_time: Optional[float] = None
            ep_best_lap: Optional[float] = None
            is_crash = False
            is_arrived = False

            while True:
                if obs.shape[0] >= 47:
                    line_obs_lat_abs_m.append(abs(float(obs[35])) * lat_norm)
                    v_err_kmh = float(obs[36]) * v_norm * 3.6
                    line_obs_v_abs_kmh.append(abs(v_err_kmh))
                    line_obs_v_signed_kmh.append(v_err_kmh)

                noisy_obs = perturbation.perturb_obs(obs)
                raw_action, _, _ = self.agent.select_action(noisy_obs, deterministic=deterministic)
                exec_action = perturbation.delay_action(raw_action)
                obs, reward, terminated, truncated, step_info = self.env.step(exec_action)
                ep_return += reward
                ep_len += 1
                ep_progress += float(step_info.get("r_progress", 0.0))
                ep_smooth += float(step_info.get("r_steer_smooth", 0.0))
                ep_idle += float(step_info.get("r_idle", 0.0))
                ep_macro += float(step_info.get("macro_reward", 0.0))

                speed = float(step_info.get("speed_km_h", 0.0))
                speeds.append(speed)

                if step_info.get("lap_completed", False):
                    comp_lap_t = float(step_info.get("completed_lap_time", step_info.get("lap_time", 0.0)))
                    completed_lap_num = int(step_info.get("completed_lap_number", 1))
                    if completed_lap_num == 1 and ep_lap1_time is None:
                        ep_lap1_time = comp_lap_t
                    elif completed_lap_num >= 2 and ep_lap2_time is None:
                        ep_lap2_time = comp_lap_t
                    if ep_best_lap is None or comp_lap_t < ep_best_lap:
                        ep_best_lap = comp_lap_t

                if terminated or truncated or ep_len >= max_eval_steps:
                    is_crash = bool(
                        step_info.get("crash", False)
                        or step_info.get("crash_sidewalk", False)
                        or step_info.get("crash_vehicle", False)
                        or step_info.get("crash_object", False)
                        or (step_info.get("termination_reason") in ["crash_or_track_limit", "stalled"])
                    )
                    is_arrived = bool(step_info.get("arrive_dest", False))

                    if is_crash:
                        crash_count += 1
                        crash_dist = float(
                            step_info.get(
                                "crash_distance",
                                float(step_info.get("route_completion", 0.0) * circuit_len),
                            )
                        )
                        crash_distances_m.append(crash_dist)
                        crash_locations.append(f"{crash_dist:.0f}m")
                        bin_low = int(crash_dist // 50) * 50
                        bin_key = f"{bin_low}-{bin_low + 50}m"
                        crash_bins_50m[bin_key] = crash_bins_50m.get(bin_key, 0) + 1
                    if is_arrived:
                        success_count += 1
                    if is_arrived and not is_crash:
                        clean_ep_count += 1
                        if ep_lap1_time is not None:
                            clean_lap_times.append(ep_lap1_time)
                        if ep_lap2_time is not None:
                            clean_lap_times.append(ep_lap2_time)
                        elif ep_best_lap is not None and ep_lap1_time is None:
                            clean_lap_times.append(ep_best_lap)
                    break

            eval_returns.append(ep_return)
            eval_lengths.append(ep_len)
            cur_comp = float(step_info.get("route_completion", 0.0) * 100.0)
            eval_completions.append(cur_comp)
            eval_speeds.append(float(np.mean(speeds)) if speeds else 0.0)
            eval_max_speeds.append(float(np.max(speeds)) if speeds else 0.0)
            cur_time = float(step_info.get("lap_time", ep_len * getattr(self.env, "dt", 0.10)))
            eval_times.append(cur_time)
            eval_progress_rew.append(ep_progress)
            eval_steer_smooth_rew.append(ep_smooth)
            eval_idle_pen.append(ep_idle)
            eval_macros_rew.append(ep_macro)

            if ep_lap1_time is not None:
                eval_lap1_times.append(ep_lap1_time)
            if ep_lap2_time is not None:
                eval_lap2_times.append(ep_lap2_time)
            if ep_best_lap is not None:
                eval_best_lap_times.append(ep_best_lap)

            if hasattr(self.env, "evaluate_current_lap_benchmark"):
                bench = self.env.evaluate_current_lap_benchmark()
                eval_line_mae.append(bench.get("racing_line_mae_m", 0.0))
                eval_line_rmse.append(bench.get("racing_line_rmse_m", 0.0))
                eval_apex_acc.append(bench.get("apex_cutting_accuracy_pct", 0.0))
                eval_corner_delta_v.append(bench.get("mean_corner_delta_v_kmh", 0.0))
                eval_friction_util.append(bench.get("friction_circle_utilization_pct", 0.0))
                eval_lap_efficiency.append(bench.get("lap_efficiency_pct", 0.0))
                if "theoretical_best_lap_time" in bench and bench["theoretical_best_lap_time"] > 0:
                    theoretical_best_lap_time = bench["theoretical_best_lap_time"]

            status_str = "ARRIVED" if is_arrived else ("CRASH" if is_crash else "OUT_OF_TIME")
            episode_records.append({
                "episode": ep + 1,
                "seed": int(target_seed),
                "return": float(ep_return),
                "steps": int(ep_len),
                "completion_pct": float(cur_comp),
                "mean_speed_kmh": float(eval_speeds[-1]),
                "max_speed_kmh": float(eval_max_speeds[-1]),
                "status": status_str,
                "clean_finish": bool(is_arrived and not is_crash),
                "lap1_time": float(ep_lap1_time) if ep_lap1_time is not None else None,
                "lap2_time": float(ep_lap2_time) if ep_lap2_time is not None else None,
                "best_lap_time": float(ep_best_lap) if ep_best_lap is not None else None,
                "racing_line_mae_m": float(eval_line_mae[-1]) if eval_line_mae else 0.0,
            })

            if verbose:
                line_m = eval_line_mae[-1] if eval_line_mae else 0.0
                apex_p = eval_apex_acc[-1] if eval_apex_acc else 0.0
                print(
                    f"  - Round {ep+1:02d}: Return: {ep_return:+6.2f} | "
                    f"Progress: {cur_comp:5.1f}% | "
                    f"Avg Speed: {eval_speeds[-1]:4.1f} km/h | "
                    f"Status: {status_str} | "
                    f"Line MAE: {line_m:.2f}m | Apex: {apex_p:.0f}%"
                )

        mean_lap_t = float(np.mean(eval_best_lap_times)) if eval_best_lap_times else float(np.mean(eval_times))
        clean_lap_rate = float(clean_ep_count / max(1, num_episodes))

        return {
            "num_episodes": num_episodes,
            "obs_noise": float(obs_noise),
            "action_delay": int(action_delay),
            "mean_return": float(np.mean(eval_returns)),
            "std_return": float(np.std(eval_returns)),
            "mean_length": float(np.mean(eval_lengths)),
            "mean_time": mean_lap_t,
            "mean_completion": float(np.mean(eval_completions)),
            "mean_completion_rate": float(np.mean(eval_completions)),
            "success_rate": float(success_count / max(1, num_episodes)),
            "clean_lap_rate": clean_lap_rate,
            "crash_rate": float(crash_count / max(1, num_episodes)),
            "crash_locations": crash_locations,
            "crash_distances_m": crash_distances_m,
            "crash_bins_50m": crash_bins_50m,
            "mean_speed": float(np.mean(eval_speeds)),
            "mean_speed_kmh": float(np.mean(eval_speeds)),
            "max_speed_kmh": float(np.max(eval_max_speeds)) if eval_max_speeds else 0.0,
            "mean_lap1_time": float(np.mean(eval_lap1_times)) if eval_lap1_times else 0.0,
            "mean_lap2_time": float(np.mean(eval_lap2_times)) if eval_lap2_times else 0.0,
            "best_lap_time": float(np.min(eval_best_lap_times)) if eval_best_lap_times else 0.0,
            "clean_lap_times": clean_lap_times,
            "clean_lap_mean": float(np.mean(clean_lap_times)) if clean_lap_times else 0.0,
            "clean_lap_std": float(np.std(clean_lap_times)) if clean_lap_times else 0.0,
            "clean_lap_min": float(np.min(clean_lap_times)) if clean_lap_times else 0.0,
            "clean_lap_max": float(np.max(clean_lap_times)) if clean_lap_times else 0.0,
            "mean_progress_reward": float(np.mean(eval_progress_rew)),
            "mean_steer_smooth_reward": float(np.mean(eval_steer_smooth_rew)),
            "mean_idle_penalty": float(np.mean(eval_idle_pen)),
            "mean_macro_reward": float(np.mean(eval_macros_rew)),
            "racing_line_mae_m": float(np.mean(eval_line_mae)) if eval_line_mae else 0.0,
            "racing_line_rmse_m": float(np.mean(eval_line_rmse)) if eval_line_rmse else 0.0,
            "apex_accuracy_pct": float(np.mean(eval_apex_acc)) if eval_apex_acc else 0.0,
            "corner_delta_v_kmh": float(np.mean(eval_corner_delta_v)) if eval_corner_delta_v else 0.0,
            "friction_utilization_pct": float(np.mean(eval_friction_util)) if eval_friction_util else 0.0,
            "lap_efficiency_pct": float(np.mean(eval_lap_efficiency)) if eval_lap_efficiency else 0.0,
            "mean_abs_line_lat_err_m": float(np.mean(line_obs_lat_abs_m)) if line_obs_lat_abs_m else 0.0,
            "mean_abs_line_v_err_kmh": float(np.mean(line_obs_v_abs_kmh)) if line_obs_v_abs_kmh else 0.0,
            "mean_signed_line_v_err_kmh": float(np.mean(line_obs_v_signed_kmh)) if line_obs_v_signed_kmh else 0.0,
            "theoretical_best_lap_time": float(theoretical_best_lap_time),
            "max_speed_cap": float(getattr(self.env, "max_speed_km_h", 120.0)),
            "episode_records": episode_records,
        }

    @staticmethod
    def print_summary(results: Dict[str, Any]):
        """In bảng tóm tắt chỉ số học thuật đẹp mắt theo chuẩn rubric REL"""
        print("\n" + "=" * 70)
        print("🏆 KẾT QUẢ ĐÁNH GIÁ CHUẨN MỰC (FINAL ACADEMIC BENCHMARK SUMMARY)")
        print("=" * 70)
        if results.get("obs_noise", 0.0) > 0 or results.get("action_delay", 0) > 0:
            print(f"  - Perturbation:                obs_noise={results.get('obs_noise', 0.0):.3f}, action_delay={results.get('action_delay', 0)} step(s)")
        print(f"  - Mean Episodic Return:        {results['mean_return']:+.2f} ± {results.get('std_return', 0.0):.2f}")
        print(f"  - Average Track Completion:    {results['mean_completion']:.1f}%")
        print(f"  - Clean Lap Rate:              {results.get('clean_lap_rate', results['success_rate'])*100:.1f}% (Success: {results['success_rate']*100:.1f}%)")
        print(f"  - Crash Rate:                  {results['crash_rate']*100:.1f}%")
        if results.get("crash_locations"):
            print(f"  - Crash Locations:             {', '.join(results['crash_locations'])}")
        if results.get("crash_bins_50m"):
            bins_str = ", ".join(f"{k}: {v}" for k, v in sorted(results["crash_bins_50m"].items()))
            print(f"  - Crash Bins (50m):            {bins_str}")
        print(f"  - Average Velocity:            {results['mean_speed']:.1f} km/h (Max: {results.get('max_speed_kmh', 0.0):.1f} km/h)")
        if results.get("clean_lap_mean", 0.0) > 0:
            print(
                f"  - Clean Lap Time (mean±std):   {results['clean_lap_mean']:.2f}s ± {results.get('clean_lap_std', 0.0):.2f}s "
                f"[min: {results.get('clean_lap_min', 0.0):.2f}s, max: {results.get('clean_lap_max', 0.0):.2f}s]"
            )
        if results.get("best_lap_time", 0.0) > 0:
            print(f"  - Best Lap Time:               {results['best_lap_time']:.2f}s (Lap 1: {results.get('mean_lap1_time', 0.0):.2f}s, Lap 2: {results.get('mean_lap2_time', 0.0):.2f}s)")
        print(f"  - Racing Line Cross-Track MAE: {results['racing_line_mae_m']:.2f} m (RMSE: {results['racing_line_rmse_m']:.2f} m)")
        if results.get("mean_abs_line_lat_err_m", 0.0) > 0:
            print(
                f"  - Line Obs Tracking Error:     |lat_err|={results['mean_abs_line_lat_err_m']:.2f} m | "
                f"|v-v*|={results.get('mean_abs_line_v_err_kmh', 0.0):.1f} km/h (signed: {results.get('mean_signed_line_v_err_kmh', 0.0):+.1f} km/h)"
            )
        print(f"  - Apex Cutting Accuracy:       {results['apex_accuracy_pct']:.1f}%")
        print(f"  - Corner Speed Delta (ΔV):     {results['corner_delta_v_kmh']:+.1f} km/h")
        print(f"  - Friction Grip Utilization:   {results['friction_utilization_pct']:.1f}%")
        theo_t = results.get("theoretical_best_lap_time", 116.99)
        print(f"  - Lap Time Efficiency:         {results['lap_efficiency_pct']:.1f}% (vs Theoretical {theo_t:.2f}s)")
        print(f"  - Mean Steps per Episode:      {results['mean_length']:.1f}")
        print("=" * 70)

    @staticmethod
    def save_json(results: Dict[str, Any], filepath: str):
        """Lưu kết quả ra file JSON định dạng chuẩn"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
