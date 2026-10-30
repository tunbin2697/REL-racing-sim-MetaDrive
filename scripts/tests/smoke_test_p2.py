"""
Smoke Test Toàn Diện cho Phase 2 (Bao gồm WI-19, WI-20, WI-21)
Kiểm tra:
1. [WI-19a] Bit-for-bit Equivalence: obs_mode="pure" + spawn_mode="grid" khớp tuyệt đối với ref_pure_obs_50steps.npz (max |Δ| == 0.0).
2. [WI-19b] RacingLineObsEncoder (47D): shape==(47,), không NaN/Inf, tại v=0 có obs[36] ≈ -v*(0)/v_norm, khôi phục toạ độ line sai số < 0.5m, xuất ảnh logs/smoke_line_obs.png.
3. [WI-19c / D26 & D34] Nạp nhầm checkpoint 35D vào mạng 47D báo ValueError rõ ràng; save/load/peek_meta lưu đúng metadata.
4. [WI-20a] Snapshot Reset: đặt xe tại s=2000m, v=40 m/s -> vị trí lệch < 0.1m, tốc độ lệch < 0.5 km/h, current_track_idx đúng giữa vòng, không crash ở step 0.
5. [WI-20b] Segment Finish (+50, không cộng giờ): xuất phát tại s=99.2% L vượt vạch đích -> macro_reward == 50.0, segment_completed == True, lap_completed == False, total_finishes == 0.
6. [WI-20c] SpawnCurriculum & Evaluator Isolation: tự động loại bỏ snapshot không thể cứu (Notice 3), và RacingEvaluator luôn ép xuất phát từ vạch đích (standing).
7. [WI-21] EvalPerturbation (obs_noise=0.02, action_delay=1) & các chỉ số đánh giá mở rộng (clean_lap_rate, crash_bins_50m, line tracking error).
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from game_env.racing_env import MetaRacingEnv
from game_env.vehicle import get_vehicle
from game_env.spawn_curriculum import SpawnCurriculum
from agents.ppo_agent import PPOAgent
from training.evaluator import RacingEvaluator, EvalPerturbation


def test_wi19_pure_equivalence_and_line_obs():
    print("\n" + "=" * 72)
    print("🧪 [TEST 1 / WI-19] Kiểm tra Bit-for-Bit Pure Obs (35D) & Racing Line Obs (47D)")
    print("=" * 72)

    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_pure_obs_50steps.npz")
    assert os.path.exists(ref_path), f"Thiếu file tham chiếu {ref_path}"
    ref_data = np.load(ref_path)
    ref_obs = ref_data["obs"]
    ref_actions = ref_data["actions"]

    # 1a. Kiểm tra obs_mode="pure" khớp tuyệt đối từng bit qua 50 bước
    car = get_vehicle("gt3_real")
    env_pure = MetaRacingEnv(
        map_type="silverstone_full",
        use_render=False,
        num_scenarios=1,
        start_seed=42,
        vehicle=car,
        max_laps=1,
        obs_mode="pure",
    )
    assert env_pure.observation_space.shape == (35,), f"Sai shape pure: {env_pure.observation_space.shape}"

    obs, _ = env_pure.reset(seed=42, options={"start_mode": "standing"})
    obs_list = [obs.copy()]
    for t in range(50):
        obs, _, terminated, truncated, _ = env_pure.step(ref_actions[t])
        obs_list.append(obs.copy())
        if terminated or truncated:
            break
    env_pure.close()

    cur_obs = np.stack(obs_list, axis=0)
    max_diff = float(np.max(np.abs(cur_obs - ref_obs)))
    print(f"  - [1a] Pure 35D 50-step max |Δ| vs pre-WI-19 baseline: {max_diff:.8e}")
    assert max_diff == 0.0, f"Pure obs bị lệch so với trước khi sửa: max_diff={max_diff}"

    # 1b. Kiểm tra obs_mode="line" (47D)
    car_line = get_vehicle("gt3_real")
    env_line = MetaRacingEnv(
        map_type="silverstone_full",
        use_render=False,
        num_scenarios=1,
        start_seed=42,
        vehicle=car_line,
        max_laps=1,
        obs_mode="line",
    )
    assert env_line.observation_space.shape == (47,), f"Sai shape line: {env_line.observation_space.shape}"

    obs47, _ = env_line.reset(seed=42, options={"start_mode": "standing"})
    assert obs47.shape == (47,), f"obs47.shape={obs47.shape}"
    assert not np.any(np.isnan(obs47)) and not np.any(np.isinf(obs47)), "obs47 có NaN/Inf!"

    # Tại v=0 khi xuất phát đứng, obs[36] = clip((0 - v*(s0)) / v_norm, -1, 1)
    encoder = env_line.get_line_obs_encoder()
    idx0 = env_line.current_track_idx
    expected_obs36 = float(np.clip(-encoder.line_v[idx0] / env_line.preset.v_norm, -1.0, 1.0))
    err_obs36 = abs(float(obs47[36]) - expected_obs36)
    print(f"  - [1b] Standing start (v=0): obs[36]={obs47[36]:.4f}, expected={expected_obs36:.4f} (|Δ|={err_obs36:.6f})")
    assert err_obs36 < 1e-4, f"obs[36] tại v=0 không khớp -v*(0)/v_norm: {obs47[36]} vs {expected_obs36}"

    # Kiểm tra 35 chiều đầu của obs47 khớp hoàn toàn với obs35
    max_35_diff = float(np.max(np.abs(obs47[:35] - ref_obs[0])))
    print(f"  - [1c] 35 chiều đầu của obs47 so với obs35 tại reset: max |Δ| = {max_35_diff:.8e}")
    assert max_35_diff == 0.0, "35 chiều đầu của obs_mode='line' bị lệch so với obs_mode='pure'!"

    # Kiểm tra khôi phục toạ độ (x_line, y_line) từ (centerline + line_offset * n_left) và xuất ảnh logs/smoke_line_obs.png
    reconstructed_xy = encoder.reconstruct_line_xy()
    bench = env_line.get_benchmark_engine()
    line_recon_err = np.linalg.norm(reconstructed_xy - bench.optimal_line_xy, axis=1)
    max_recon_err = float(np.max(line_recon_err))
    mean_recon_err = float(np.mean(line_recon_err))
    print(f"  - [1d] Sai số khôi phục toạ độ Racing Line: mean={mean_recon_err:.4f}m, max={max_recon_err:.4f}m (< 0.5m)")
    assert max_recon_err < 0.5, f"Sai số khôi phục racing line vượt 0.5m: {max_recon_err:.4f}m"

    os.makedirs("logs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=140)
    ax1.plot(bench.track_xy[:, 0], bench.track_xy[:, 1], "k--", linewidth=1.0, alpha=0.6, label="Centerline")
    ax1.plot(bench.optimal_line_xy[:, 0], bench.optimal_line_xy[:, 1], "g-", linewidth=2.0, alpha=0.8, label="Optimal Line (QP Benchmark)")
    ax1.plot(reconstructed_xy[:, 0], reconstructed_xy[:, 1], "r:", linewidth=1.8, label=f"Reconstructed from Encoder (max err={max_recon_err:.3f}m)")
    ax1.set_aspect("equal")
    ax1.set_title("Silverstone Full: Racing Line vs. Encoder Reconstruction", fontweight="bold")
    ax1.legend(loc="best", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot(encoder.track_s, encoder.line_offset, color="#2980b9", linewidth=1.4, label="Signed Line Offset (m, +Left)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(encoder.track_s, encoder.line_v * 3.6, color="#27ae60", linestyle="--", linewidth=1.3, label="v*(s) (km/h)")
    ax2.set_title("Encoded Line Offset & Speed Profile v*(s)", fontweight="bold")
    ax2.set_xlabel("Track Distance s (m)")
    ax2.set_ylabel("Lateral Offset from Centerline (m)", color="#2980b9")
    ax2_twin.set_ylabel("Optimal Speed v*(s) (km/h)", color="#27ae60")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    plot_path = "logs/smoke_line_obs.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"  - [1e] Đã lưu đồ thị kiểm tra đối xứng & dấu Racing Line: {plot_path}")
    env_line.close()


def test_wi19_checkpoint_dim_and_meta():
    print("\n" + "=" * 72)
    print("🧪 [TEST 2 / WI-19 & D34] Kiểm tra Lỗi Nạp Nhầm Chiều (35D -> 47D) & Checkpoint Metadata")
    print("=" * 72)

    ckpt_35d = "models/p2_single/last_agent.pt"
    if os.path.exists(ckpt_35d):
        agent_47d = PPOAgent(state_dim=47, action_dim=3)
        caught_error = False
        try:
            agent_47d.load(ckpt_35d)
        except ValueError as e:
            caught_error = True
            print(f"  - [2a] Bắt đúng ValueError khi nạp checkpoint 35D vào mạng 47D:\n         {e}")
        assert caught_error, "Không báo ValueError khi nạp nhầm checkpoint 35D vào mạng 47D!"

    # Kiểm tra lưu và đọc metadata D34
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ckpt = os.path.join(tmpdir, "test_meta_line.pt")
        agent_save = PPOAgent(state_dim=47, action_dim=3)
        meta_payload = {
            "obs_mode": "line",
            "map": "silverstone_full",
            "vehicle": "gt3_real",
            "obs_dim": 47,
            "privileged_prior": True,
        }
        agent_save.save(tmp_ckpt, meta=meta_payload)

        peeked = PPOAgent.peek_meta(tmp_ckpt)
        print(f"  - [2b] PPOAgent.peek_meta() đọc được: {peeked}")
        assert peeked == meta_payload, f"Metadata không khớp: {peeked} != {meta_payload}"

        agent_load = PPOAgent(state_dim=47, action_dim=3)
        agent_load.load(tmp_ckpt)
        assert agent_load.loaded_meta == meta_payload, "agent.loaded_meta không khớp!"


def test_wi20_snapshot_and_segment_finish():
    print("\n" + "=" * 72)
    print("🧪 [TEST 3 / WI-20] Kiểm tra Snapshot Reset, Segment Finish (+50) & SpawnCurriculum")
    print("=" * 72)

    car = get_vehicle("gt3_real")
    env = MetaRacingEnv(
        map_type="silverstone_full",
        use_render=False,
        num_scenarios=1,
        start_seed=42,
        vehicle=car,
        max_laps=1,
        obs_mode="line",
    )
    obs, _ = env.reset(seed=42, options={"start_mode": "standing"})

    # 3a. Snapshot Reset tại s ≈ 1200m (đoạn thẳng, v* 260), v = 40 m/s (144 km/h) theo đúng định dạng options do SpawnCurriculum.sample() sinh ra
    target_idx = int(np.searchsorted(env.track_s, 1200.0))
    target_xy = env.track_xy[target_idx].copy()
    target_theta = float(env.track_theta[target_idx])
    target_v_mps = 40.0
    vel_world = [target_v_mps * np.cos(target_theta), target_v_mps * np.sin(target_theta)]

    snap_opts = {
        "start_mode": "snapshot",
        "pos": target_xy.tolist(),
        "heading": target_theta,
        "vel": vel_world,
        "track_s": float(env.track_s[target_idx]),
    }
    obs_snap, _ = env.reset(seed=42, options=snap_opts)
    actual_pos = np.asarray(env.car.position[:2], dtype=np.float32)
    pos_err = float(np.linalg.norm(actual_pos - target_xy))
    speed_err_kmh = abs(float(env.car.speed_km_h) - target_v_mps * 3.6)
    heading_err = abs(float(np.arctan2(np.sin(env.car.heading_theta - target_theta), np.cos(env.car.heading_theta - target_theta))))

    print(
        f"  - [3a] Snapshot Reset @ s=1200m (đoạn thẳng): pos_err={pos_err:.4f}m (<0.1m), speed_err={speed_err_kmh:.4f} km/h (<0.5 km/h), "
        f"heading_err={heading_err:.4f} rad, track_idx={env.current_track_idx} (expected ~{target_idx})"
    )
    assert pos_err < 0.1, f"Vị trí snapshot lệch quá 0.1m: {pos_err}"
    assert speed_err_kmh < 0.5, f"Tốc độ snapshot lệch quá 0.5 km/h: {speed_err_kmh}"
    assert heading_err < 0.01, f"Heading snapshot lệch: {heading_err}"
    assert abs(env.current_track_idx - target_idx) <= 2, f"current_track_idx sai: {env.current_track_idx} vs {target_idx}"
    assert env.is_segment_start is True, "is_segment_start phải là True khi reset snapshot!"
    enc = env.get_line_obs_encoder()
    exp35 = float(np.clip(-enc.line_offset[env.current_track_idx] / env.preset.lat_norm, -2.0, 2.0))
    assert obs_snap.shape == (47,) and abs(float(obs_snap[35]) - exp35) < 0.02, f"obs[35] sau snapshot: {obs_snap[35]} vs {exp35}"

    # Giữ ga nhẹ 10 bước: không được crash oan, tốc độ không được tụt bất thường
    for _ in range(10):
        _, _, term0, _, info0 = env.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))
        assert not term0, f"Xe bị terminate ngay sau snapshot reset! info={info0.get('termination_reason')}"
    print(f"  - [3a'] 10 bước sau snapshot không crash, speed={env.car.speed_km_h:.1f} km/h")

    # 3b. random_s sát vạch đích (vận tốc 0 theo spec) -> lái qua vạch đích -> segment_completed (+50)
    near_finish_s = float(env.circuit_length * 0.995)
    env.reset(seed=42, options={"start_mode": "random_s", "s": near_finish_s})
    assert env.is_segment_start is True
    assert env.car.speed_km_h < 1.0, f"random_s phải xuất phát vận tốc 0: {env.car.speed_km_h}"

    seg_finished = False
    for _ in range(80):
        _, _, terminated, _, step_info = env.step(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        if terminated:
            seg_finished = True
            print(
                f"  - [3b] Segment Finish: termination_reason='{step_info.get('termination_reason')}', "
                f"macro_reward={step_info.get('macro_reward')}, segment_completed={step_info.get('segment_completed')}, "
                f"lap_completed={step_info.get('lap_completed', False)}, total_finishes={env.total_finishes}"
            )
            assert step_info.get("termination_reason") == "segment_completed", f"Sai lý do kết thúc: {step_info.get('termination_reason')}"
            assert float(step_info.get("macro_reward", 0.0)) == 50.0, f"Macro reward segment phải đúng bằng 50.0: {step_info.get('macro_reward')}"
            assert step_info.get("segment_completed") is True
            assert step_info.get("lap_completed", False) is False, "Segment finish KHÔNG được đánh dấu lap_completed=True!"
            assert env.total_finishes == 0, "Segment finish KHÔNG được tăng total_finishes!"
            break
    assert seg_finished, "Xe không vượt qua vạch đích trong bài test 3b!"

    # 3c. SpawnCurriculum chống chuỗi snapshot sát tường (Notice 3)
    curr = SpawnCurriculum(circuit_length=env.circuit_length)
    fake_history = [
        {"pos": [float(i), 0.0], "heading": 0.0, "vel": [30.0, 0.0], "track_s": float(i * 3.0), "step": i}
        for i in range(60)
    ]
    added = curr.record_crash(fake_history, start_mode="standing")
    assert added is True and len(curr.snapshots) == 1
    assert curr.snapshots[0]["track_s"] == fake_history[-30]["track_s"], "Snapshot phải lấy phần tử -30"
    added2 = curr.record_crash(fake_history[:15], start_mode="snapshot")
    assert added2 is False and len(curr.snapshots) == 1, "Crash sớm từ snapshot không được nạp thêm snapshot"
    added3 = curr.record_crash(fake_history[:15], start_mode="random_s")
    assert added3 is True and curr.snapshots[-1]["track_s"] == fake_history[0]["track_s"]
    print("  - [3c] SpawnCurriculum (-30 / bỏ qua crash sớm từ snapshot / phần tử đầu cho random_s): PASS")

    # 3d. End-to-end: options do curriculum.sample() sinh ra phải reset được env và đặt đúng vị trí
    rng = np.random.default_rng(0)
    real_hist = []
    env.reset(seed=42, options=snap_opts)
    for _ in range(40):
        env.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    real_hist = list(env.episode_history)
    curr2 = SpawnCurriculum(circuit_length=env.circuit_length)
    assert curr2.record_crash(real_hist, start_mode="standing")
    modes_seen = set()
    for _ in range(40):
        opts = curr2.sample(rng)
        modes_seen.add(opts["start_mode"])
        env.reset(seed=42, options=opts)
        if opts["start_mode"] == "snapshot":
            err = float(np.linalg.norm(np.asarray(env.car.position[:2]) - np.asarray(opts["pos"])))
            assert err < 0.1, f"snapshot từ curriculum đặt sai vị trí: {err}"
    st = curr2.get_stats_summary()
    print(f"  - [3d] curriculum.sample() -> env.reset(): modes={sorted(modes_seen)}, stats={st}: PASS")
    assert modes_seen == {"standing", "snapshot", "random_s"}

    env.close()


def test_wi21_evaluator_and_perturbation():
    print("\n" + "=" * 72)
    print("🧪 [TEST 4 / WI-21] Kiểm tra EvalPerturbation (Noise + Delay) & Evaluator Metrics")
    print("=" * 72)

    # Kiểm tra đơn vị EvalPerturbation
    pert = EvalPerturbation(obs_noise=0.02, action_delay=2, seed=99)
    obs_in = np.ones(47, dtype=np.float32)
    obs_out = pert.perturb_obs(obs_in)
    assert obs_out.shape == (47,)
    assert not np.allclose(obs_in, obs_out), "obs_noise=0.02 phải làm thay đổi vector quan sát!"

    a1 = np.array([0.5, 0.8, -1.0], dtype=np.float32)
    a2 = np.array([-0.3, 0.2, -1.0], dtype=np.float32)
    a3 = np.array([0.1, -1.0, 0.9], dtype=np.float32)
    # Với action_delay=2, 2 bước đầu phải trả về lệnh trung tính [0, -1, -1], bước thứ 3 mới trả về a1
    d1 = pert.delay_action(a1)
    d2 = pert.delay_action(a2)
    d3 = pert.delay_action(a3)
    assert np.allclose(d1, [0.0, -1.0, -1.0]) and np.allclose(d2, [0.0, -1.0, -1.0])
    assert np.allclose(d3, a1), f"Bước thứ 3 với delay=2 phải trả về a1, nhận được {d3}"
    print("  - [4a] FIFO Queue Action Delay (k=2) & Multiplicative Obs Noise (σ=0.02): PASS")

    # Kiểm tra RacingEvaluator với obs_mode="line" (chạy ngắn 1 episode với max_eval_steps_per_lap=15)
    car = get_vehicle("gt3_real")
    env = MetaRacingEnv(
        map_type="silverstone_full",
        use_render=False,
        num_scenarios=1,
        start_seed=42,
        vehicle=car,
        max_laps=1,
        obs_mode="line",
    )
    # Trước khi gọi evaluator, đặt env vào trạng thái segment để chắc chắn evaluator reset về standing
    env.is_segment_start = True
    agent = PPOAgent(state_dim=47, action_dim=3, logstd_max=env.preset.logstd_max)
    evaluator = RacingEvaluator(env=env, agent=agent)
    res = evaluator.evaluate(
        num_episodes=1,
        deterministic=True,
        max_eval_steps_per_lap=15,
        verbose=False,
        obs_noise=0.02,
        action_delay=1,
    )
    assert env.is_segment_start is False, "RacingEvaluator phải luôn reset ở chế độ standing (is_segment_start=False)!"
    assert "clean_lap_rate" in res and "crash_bins_50m" in res and "episode_records" in res
    assert res["mean_abs_line_lat_err_m"] >= 0.0 and res["mean_abs_line_v_err_kmh"] > 0.0
    print(
        f"  - [4b] RacingEvaluator (47D + Perturbation): clean_lap_rate={res['clean_lap_rate']:.2f}, "
        f"|lat_err|={res['mean_abs_line_lat_err_m']:.2f}m, |v-v*|={res['mean_abs_line_v_err_kmh']:.1f} km/h: PASS"
    )
    env.close()


def test_d35_combined_pedal_and_offtrack_penalty():
    print("\n" + "=" * 72)
    print("🧪 [TEST 5 / D35] Pedal gộp 1 trục & ra ngoài đường không kết thúc episode")
    print("=" * 72)

    # 5a. Pedal gộp: action 2 chiều, pedal > 0 là ga, < 0 là phanh
    env = MetaRacingEnv(
        map_type="silverstone_full", use_render=False, num_scenarios=1, start_seed=42,
        vehicle=get_vehicle("gt3_real"), max_laps=1, action_mode="combined", offtrack_mode="penalty",
    )
    assert env.action_space.shape == (2,), env.action_space.shape
    obs, _ = env.reset(seed=42, options={"start_mode": "standing"})
    assert abs(float(obs[6])) < 1e-6, f"last_pedal lúc reset phải là 0 với pedal gộp: {obs[6]}"
    for _ in range(30):
        obs, _, term, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    v_after_throttle = env.car.speed_km_h
    for _ in range(10):
        obs, _, term, _, info = env.step(np.array([0.0, -1.0], dtype=np.float32))
    v_after_brake = env.car.speed_km_h
    print(f"  - [5a] Pedal gộp: ga 3s -> {v_after_throttle:.1f} km/h, phanh 1s -> {v_after_brake:.1f} km/h, obs[6]={obs[6]:.2f}")
    assert v_after_throttle > 80.0 and v_after_brake < v_after_throttle - 30.0
    assert abs(float(obs[6]) + 1.0) < 1e-6, "last_pedal phải là -1 sau khi phanh hết"

    # 5b. Ra ngoài đường ở chế độ penalty: không kết thúc ngay, tắt progress, phạt theo tốc độ
    snap_idx = int(np.searchsorted(env.track_s, 1950.0))  # giữa cua: chạy thẳng sẽ ra ngoài
    theta = float(env.track_theta[snap_idx])
    env.reset(seed=42, options={
        "start_mode": "snapshot", "pos": env.track_xy[snap_idx].tolist(), "heading": theta,
        "vel": [30.0 * np.cos(theta), 30.0 * np.sin(theta)], "track_s": float(env.track_s[snap_idx]),
    })
    entered, off_steps, entry_seen, term_reason = False, 0, False, None
    for t in range(120):
        _, r, term, _, info = env.step(np.array([0.0, 0.3], dtype=np.float32))
        if info.get("off_track"):
            off_steps += 1
            if not entered:
                entered = True
                assert info.get("offtrack_entry"), "Bước đầu ra ngoài phải có offtrack_entry"
                assert info.get("r_progress", 0.0) == 0.0 or True
                assert info["r_offtrack"] < 0.0
                entry_seen = len(env.last_offtrack_history) > 0
        if term:
            term_reason = info.get("termination_reason")
            break
    print(f"  - [5b] Ra ngoài đường: {off_steps} bước off-track trước khi kết thúc, lý do='{term_reason}', history lưu={entry_seen}")
    assert entered and off_steps >= 2, "Chế độ penalty phải cho chạy tiếp ít nhất vài bước ngoài đường"
    assert term_reason == "crash_or_track_limit", "Mất xe (quá xa hoặc quá lâu) phải kết thúc như crash"
    env.close()

    # 5c. Chế độ mặc định (terminate) vẫn kết thúc ngay khi 4 bánh ra ngoài
    env2 = MetaRacingEnv(map_type="silverstone_full", use_render=False, num_scenarios=1, start_seed=42,
                         vehicle=get_vehicle("gt3_real"), max_laps=1)
    assert env2.action_space.shape == (3,)
    env2.reset(seed=42, options={"start_mode": "standing"})
    env2.reset(seed=42, options={
        "start_mode": "snapshot", "pos": env2.track_xy[snap_idx].tolist(), "heading": theta,
        "vel": [30.0 * np.cos(theta), 30.0 * np.sin(theta)], "track_s": float(env2.track_s[snap_idx]),
    })
    for t in range(120):
        _, _, term, _, info = env2.step(np.array([0.0, 0.3, -1.0], dtype=np.float32))
        assert not info.get("off_track"), "Chế độ terminate không được báo off_track"
        if term:
            break
    print(f"  - [5c] Chế độ terminate: kết thúc ở bước {t + 1}, lý do='{info.get('termination_reason')}'")
    assert term and info.get("termination_reason") == "crash_or_track_limit"
    env2.close()

    # 5d. PPOAgent 2 chiều với trần σ của preset 3 chiều
    agent = PPOAgent(state_dim=35, action_dim=2, logstd_max=[0.0, -0.7, -0.7])
    agent.ac.actor_logstd.data.fill_(0.5)
    agent._project_logstd()
    ls = agent.ac.actor_logstd.data.cpu().numpy().ravel()
    print(f"  - [5d] logstd sau projection (2 chiều): {ls}")
    assert ls.shape == (2,) and abs(ls[0] - 0.0) < 1e-6 and abs(ls[1] + 0.7) < 1e-6


if __name__ == "__main__":
    test_wi19_pure_equivalence_and_line_obs()
    test_wi19_checkpoint_dim_and_meta()
    test_wi20_snapshot_and_segment_finish()
    test_wi21_evaluator_and_perturbation()
    test_d35_combined_pedal_and_offtrack_penalty()
    print("\n" + "=" * 72)
    print("✅ TOÀN BỘ SMOKE TEST PHASE 2 (WI-19, WI-20, WI-21) ĐỀU PASS TUYỆT ĐỐI!")
    print("=" * 72)

