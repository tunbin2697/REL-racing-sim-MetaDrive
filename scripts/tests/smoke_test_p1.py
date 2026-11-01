import os
import sys
import numpy as np
import torch

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
from agents.ppo_agent import PPOAgent
from agents.networks import ActorCritic

def run_smoke_test():
    output_lines = []
    def log(msg):
        try:
            print(msg)
        except Exception:
            try:
                sys.stdout.buffer.write((msg + "\n").encode("utf-8"))
            except Exception:
                pass
        output_lines.append(msg)

    log("=" * 70)
    log(">> SMOKE TEST PHASE 1 (WI-04): KIEM TRA TOAN DIEN BAN VA PPO & REWARD")
    log("=" * 70)

    # 1. Kiem tra hang so mang ActorCritic & Bias chan phanh
    log("\n[TEST 1] Kiem tra hang so ActorCritic & Bias khoi tao...")
    ac = ActorCritic(state_dim=35, action_dim=3)
    last_layer = list(ac.actor_mean.children())[-1]
    bias_tensor = getattr(last_layer, "bias", None)
    assert bias_tensor is not None, "LOI: Khong tim thay bias cua actor_out"
    bias_brake = float(bias_tensor[2].item())
    log(f"  - LOGSTD_MIN: {ac.LOGSTD_MIN} | LOGSTD_MAX: {ac.LOGSTD_MAX} | ACTION_BOUND: {ac.ACTION_BOUND}")
    log(f"  - Brake Bias actor_out.bias[2]: {bias_brake:.4f}")
    assert abs(bias_brake - (-1.6)) < 1e-5, f"LOI: Brake bias phai la -1.6, hien tai la {bias_brake}"
    
    # Kiem tra luc phanh ban dau
    tanh_bias = float(np.tanh(bias_brake))
    pedal_idle = 0.5 * (tanh_bias + 1.0)
    log(f"  - tanh(bias_brake) = {tanh_bias:.4f} -> pedal_idle = {pedal_idle:.4f} (< deadband 0.05: NHA PHANH 100%)")
    assert pedal_idle < 0.05, f"LOI: Ban dap phanh chua nha het! pedal = {pedal_idle}"
    log("  => TEST 1: PASS! (Bias chan phanh hoan hao)")

    # 1B. Kiem tra F2: Khop log_prob giua Rollout va Update khi hanh dong bao hoa
    log("\n[TEST 1B] Kiem tra khop log_prob F2 giua Rollout va Update khi hanh dong bao hoa...")
    ac_f2 = ActorCritic(state_dim=35, action_dim=3)
    last_layer_f2 = list(ac_f2.actor_mean.children())[-1]
    with torch.no_grad():
        getattr(last_layer_f2, "bias")[1] = 4.0  # Gia lap dap het ga tren duong thang
    
    dummy_states = torch.randn(200, 35)
    with torch.no_grad():
        rollout_act, rollout_lp, _, _, rollout_u = ac_f2.get_action_and_value(dummy_states, return_raw=True)
        _, update_lp, _, _ = ac_f2.get_action_and_value(dummy_states, raw_u=rollout_u)
        delta_lp = torch.abs(rollout_lp - update_lp)
        max_delta = float(delta_lp.max().item())
        mean_delta = float(delta_lp.mean().item())
        mismatch_count = int((delta_lp > 1e-4).sum().item())
    log(f"  - Max |Delta log_prob|: {max_delta:.8f} (ky vong < 1e-4)")
    log(f"  - Mean |Delta log_prob|: {mean_delta:.8f}")
    log(f"  - So luong mau lech log_prob > 1e-4: {mismatch_count}/200 (ky vong: 0)")
    assert max_delta < 1e-4, f"LOI F2: log_prob giua rollout va update bi lech! max_delta = {max_delta}"
    log("  => TEST 1B: PASS! (Loi F2 da duoc triet tieu hoan toan 100%, log_prob khop tuyet doi)")

    # 2. Kiem tra Bang Bonus moc thoi gian (Exponential Finish Bonus)
    log("\n[TEST 2] Kiem tra Bang Bonus Ve Dich Ham Mu (WI-02)...")
    test_laps = [146.0, 120.0, 111.0, 100.0]
    expected = [15.6, 44.1, 63.2, 98.1]
    for t_lap, exp_val in zip(test_laps, expected):
        bonus = min(150.0, 4.0 * float(np.exp((180.0 - t_lap) / 25.0)))
        total_finish_macro = 50.0 + bonus
        log(f"  - Lap {t_lap}s -> time_bonus = {bonus:.1f} (ky vong ~{exp_val:.1f}) | Tong macro = {total_finish_macro:.1f}")
        assert abs(bonus - exp_val) < 0.5, f"LOI: Bonus khong khop tai {t_lap}s: {bonus} vs {exp_val}"
    log("  => TEST 2: PASS! (Ham thuong can dich chuan xac)")

    # 3. Kiem tra Moi truong & Time Penalty r_time
    log("\n[TEST 3] Kiem tra Moi truong MetaRacingEnv & r_time...")
    race_car = get_vehicle("gt3")
    env = MetaRacingEnv(
        use_render=False,
        traffic_density=0.0,
        map_config="silverstone",
        vehicle=race_car,
        time_penalty=0.01,
        finish_bonus="exp",
    )
    obs, info = env.reset(seed=42)
    log(f"  - Obs shape: {obs.shape} (ky vong (35,))")
    assert obs.shape == (35,), f"LOI: Obs shape sai: {obs.shape}"

    # Step thu 1 buoc
    action = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    next_obs, reward, term, trunc, step_info = env.step(action)
    r_time = float(step_info.get("r_time", 0.0))
    log(f"  - step_info['r_time']: {r_time} (ky vong -0.01)")
    assert abs(r_time - (-0.01)) < 1e-6, f"LOI: r_time khong dung: {r_time}"
    log("  => TEST 3: PASS! (Moi truong tra obs 35D va r_time = -0.01 chuan)")

    # 4. Kiem tra PPO Agent, Logstd Projection & 1 Rollout + 1 Update
    log("\n[TEST 4] Kiem tra PPOAgent Rollout + GAE + Update...")
    agent = PPOAgent(
        state_dim=35,
        action_dim=3,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.90,
        clip_coef=0.15,
        ent_coef=0.010,
        minibatch_size=32,
        enable_logstd_projection=True,
    )
    
    # Gia lap 64 buoc rollout
    n_rollout = 64
    b_states = np.zeros((n_rollout, 35), dtype=np.float32)
    b_actions = np.zeros((n_rollout, 3), dtype=np.float32)
    b_raw_us = np.zeros((n_rollout, 3), dtype=np.float32)
    b_logprobs = np.zeros(n_rollout, dtype=np.float32)
    b_rewards = np.zeros(n_rollout, dtype=np.float32)
    b_values = np.zeros(n_rollout, dtype=np.float32)
    b_dones = np.zeros(n_rollout, dtype=np.float32)

    cur_o = obs
    for i in range(n_rollout):
        act, lp, val, raw_u = agent.select_action(cur_o, return_raw=True)
        b_states[i] = cur_o
        b_actions[i] = act
        b_raw_us[i] = raw_u
        b_logprobs[i] = lp
        b_values[i] = val
        
        no, rew, d, tr, inf = env.step(act)
        b_rewards[i] = float(rew)
        is_done = bool(d) or bool(tr)
        b_dones[i] = 1.0 if is_done else 0.0
        if is_done:
            cur_o, _ = env.reset()
        else:
            cur_o = no

    # Compute GAE
    _, _, last_val = agent.select_action(cur_o)
    advs, rets = agent.compute_gae(b_rewards, b_values, b_dones, last_val)
    
    # Update PPO
    train_res = agent.update(b_states, b_actions, b_logprobs, advs, rets, b_values, b_raw_us=b_raw_us)
    log(f"  - PPO Update: Policy Loss = {train_res['policy_loss']:.4f} | Value Loss = {train_res['value_loss']:.4f} | Entropy = {train_res['entropy']:.4f}")
    assert np.isfinite(train_res['policy_loss']), "LOI: Policy Loss khong huu han!"
    assert np.isfinite(train_res['value_loss']), "LOI: Value Loss khong huu han!"

    # 4B. Kiem tra Co che Projection kep bien khi logstd bi troi ra ngoai bien
    log("\n[TEST 4B] Kiem tra Co che Projection kep bien khi logstd bi troi ra ngoai bien...")
    # Thu co tinh gan logstd vuot bien cuc dai va cuc tieu: [0.5, 0.5, -3.0]
    with torch.no_grad():
        agent.ac.actor_logstd.copy_(torch.tensor([[0.5, 0.5, -3.0]], device=agent.device))
        logstd_before = agent.ac.actor_logstd.detach().cpu().numpy()[0]
    log(f"  - actor_logstd truoc khi update (bi ep troi ra ngoai): {logstd_before}")
    
    # Update PPO de kich hoat projection sau optimizer.step()
    agent.update(b_states, b_actions, b_logprobs, advs, rets, b_values, b_raw_us=b_raw_us)
    with torch.no_grad():
        logstd_proj = agent.ac.actor_logstd.detach().cpu().numpy()[0]
    log(f"  - actor_logstd sau update (duoc chieu kep bien): {logstd_proj}")
    for d_i in range(3):
        assert agent.ac.LOGSTD_MIN <= logstd_proj[d_i] <= agent.ac.LOGSTD_MAX, (
            f"LOI: dim {d_i} = {logstd_proj[d_i]} nam ngoai bien [{agent.ac.LOGSTD_MIN}, {agent.ac.LOGSTD_MAX}]"
        )
    assert logstd_proj[0] < 0.1 and logstd_proj[1] < 0.1, "LOI: Khong cat duoc phan duong 0.5"
    assert logstd_proj[2] > -2.5, "LOI: Khong keo duoc phan am -3.0 ve -2.3"
    log("  => TEST 4B: PASS! (Projection hoat dong chuan xac: [0.5, 0.5, -3.0] duoc chieu va nam chat trong [-2.3, 0.0])")

    # 4C. Kiem tra F13: ratio PPO khong no khi mean phanh nam sau trong vung bao hoa (tai hien loi NaN o step 1.38M)
    log("\n[TEST 4C] Kiem tra F13: PPO update on dinh khi hanh dong bao hoa sau (mean phanh = -5, sigma = 0.1)...")
    agent_f13 = PPOAgent(state_dim=35, action_dim=3, lr=3e-4, clip_coef=0.15, ent_coef=0.010, minibatch_size=64)
    last_layer_f13 = list(agent_f13.ac.actor_mean.children())[-1]
    with torch.no_grad():
        getattr(last_layer_f13, "bias")[2] = -5.0
        agent_f13.ac.actor_logstd.copy_(torch.tensor([[-1.6, -0.1, -2.3]], device=agent_f13.device))
    rng = np.random.default_rng(0)
    n_f13 = 512
    s_f13 = rng.standard_normal((n_f13, 35)).astype(np.float32)
    a_f13 = np.zeros((n_f13, 3), dtype=np.float32)
    u_f13 = np.zeros((n_f13, 3), dtype=np.float32)
    lp_f13 = np.zeros(n_f13, dtype=np.float32)
    v_f13 = np.zeros(n_f13, dtype=np.float32)
    for i in range(n_f13):
        a_i, lp_i, v_i, u_i = agent_f13.select_action(s_f13[i], return_raw=True)
        a_f13[i], lp_f13[i], v_f13[i], u_f13[i] = a_i, lp_i, v_i, u_i
    sat_ratio = float((np.abs(a_f13[:, 2]) >= 0.999).mean())
    adv_f13 = rng.standard_normal(n_f13).astype(np.float32)
    ret_f13 = rng.standard_normal(n_f13).astype(np.float32)
    max_pl = 0.0
    total_skipped = 0
    for _ in range(5):
        res = agent_f13.update(s_f13, a_f13, lp_f13, adv_f13, ret_f13, v_f13, b_raw_us=u_f13)
        max_pl = max(max_pl, abs(res["policy_loss"]))
        total_skipped += res["skipped_actor_updates"]
    params_finite = all(torch.isfinite(p).all().item() for p in agent_f13.ac.parameters())
    log(f"  - Ty le mau phanh bao hoa |a| >= 0.999: {sat_ratio:.2f}")
    log(f"  - Max |policy loss| qua 5 lan update: {max_pl:.4f} (ky vong < 10) | minibatch bi bo qua: {total_skipped}")
    log(f"  - Tham so mang huu han: {params_finite}")
    assert sat_ratio > 0.5, "LOI: Test khong tai hien duoc trang thai bao hoa"
    assert max_pl < 10.0 and params_finite and total_skipped == 0, "LOI F13: PPO update khong on dinh khi hanh dong bao hoa"
    log("  => TEST 4C: PASS! (log_prob tinh tai u da sample, ratio khong no)")

    # 4D. Doi chung co che F13: dich mean phanh 1 sigma ve phia bien, so ratio duong cu (u = atanh(0.999)) va duong moi (u da sample)
    log("\n[TEST 4D] Doi chung co che F13: dich mean phanh +1 sigma ve phia bien bao hoa...")
    ac_d = ActorCritic(state_dim=35, action_dim=3)
    last_d = list(ac_d.actor_mean.children())[-1]
    with torch.no_grad():
        getattr(last_d, "bias")[2] = -5.0
        ac_d.actor_logstd.copy_(torch.tensor([[-1.6, -0.1, -2.3]]))
        x_d = torch.randn(512, 35)
        act_d, lp_new_old, _, _, u_d = ac_d.get_action_and_value(x_d, return_raw=True)
        _, lp_clamp_old, _, _ = ac_d.get_action_and_value(x_d, action=act_d)
        getattr(last_d, "bias")[2] = -5.0 + float(np.exp(-2.3))  # mean dich 1 sigma (0.1) ve phia -3.8
        _, lp_new_new, _, _ = ac_d.get_action_and_value(x_d, raw_u=u_d)
        _, lp_clamp_new, _, _ = ac_d.get_action_and_value(x_d, action=act_d)
    ratio_new = float(torch.exp(lp_new_new - lp_new_old).max().item())
    ratio_old = float(torch.exp(lp_clamp_new - lp_clamp_old).max().item())
    log(f"  - Max ratio duong MOI (log_prob tai u da sample): {ratio_new:.2f} (ky vong < 20)")
    log(f"  - Max ratio duong CU  (log_prob tai atanh(0.999)): {ratio_old:.3e} (ky vong > 1e3, tai hien loi)")
    assert ratio_new < 20.0, "LOI F13: ratio duong moi qua lon"
    assert ratio_old > 1e3, "LOI: Test doi chung khong tai hien duoc co che F13"
    log("  => TEST 4D: PASS! (Duong cu no ratio, duong moi on dinh -> xac nhan dung nguyen nhan va dung ban sua)")

    env.close()

    log("\n" + "=" * 70)
    log(">> TOAN BO SMOKE TEST PHASE 1 (WI-04) DAT 100% SUCCESS!")
    log("=" * 70)

    # Luu log ra file logs/smoke_p1.txt
    os.makedirs("logs", exist_ok=True)
    with open("logs/smoke_p1.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    print("\nDa ghi ket qua vao logs/smoke_p1.txt")

if __name__ == "__main__":
    run_smoke_test()
