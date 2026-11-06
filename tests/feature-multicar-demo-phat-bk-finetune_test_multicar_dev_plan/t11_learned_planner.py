"""T11. Mạng có học được quyết định của planner MPC không? Học có giám sát, không dùng reward.
Mạng nhận obs 60D và độ lệch hiện tại, dự đoán độ lệch đích và mức pedal thay cho planner.
So sánh MLP với attention trên 5 xe đối thủ, có một vòng DAgger: chạy bằng mạng, planner gán nhãn trên các trạng thái mạng đã đi qua.
    python t11_learned_planner.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import make_env, load_policy, run_race, save, RESULTS
from mpc_adapter import MPCAdapter, MPCConfig
from t10_lib import SmartMistakes

BASE = 35
RACES_COLLECT, RACES_EVAL = 12, 12


class MLP(nn.Module):
    def __init__(self, n_off, n_mode):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(61, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.off, self.mode = nn.Linear(256, n_off), nn.Linear(256, n_mode)

    def forward(self, x):
        h = self.body(x)
        return self.off(h), self.mode(h)


class Attn(nn.Module):
    """Ego làm truy vấn, 5 xe đối thủ làm token; xe không có được che bằng cờ có xe."""

    def __init__(self, n_off, n_mode, d=64):
        super().__init__()
        self.ego = nn.Sequential(nn.Linear(36, 128), nn.Tanh(), nn.Linear(128, d))
        self.token = nn.Sequential(nn.Linear(5, d), nn.Tanh(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.Tanh())
        self.off, self.mode = nn.Linear(128, n_off), nn.Linear(128, n_mode)

    def forward(self, x):
        ego = self.ego(torch.cat([x[:, :BASE], x[:, -1:]], 1))
        tok = x[:, BASE:BASE + 25].view(-1, 5, 5)
        pad = tok[:, :, 4] < 0.5
        out, _ = self.attn(ego[:, None], self.token(tok), self.token(tok), key_padding_mask=pad)
        h = self.head(torch.cat([ego, torch.nan_to_num(out[:, 0])], 1))
        return self.off(h), self.mode(h)


class LearnedAdapter(MPCAdapter):
    """Giống MPCAdapter nhưng lấy quyết định từ mạng. Có thể ghi lại nhãn của planner tại trạng thái đang đi qua."""

    def __init__(self, env, agent, profile, net=None, cfg=None):
        super().__init__(env, agent, profile, cfg)
        self.net, self.X, self.y_off, self.y_mode = net, [], [], []
        self.net_time, self.net_calls = 0.0, 0

    def _plan(self, i, s_now, v_now, n_off_now, active):
        plan = super()._plan(i, s_now, v_now, n_off_now, active)
        if plan is None:
            return None
        x = np.concatenate([self.cur_obs[i], [self.offset[i] / 3.0]], dtype=np.float32)
        self.X.append(x)
        self.y_off.append(list(self.cfg.offsets).index(plan[0]))
        self.y_mode.append(plan[1])
        if self.net is None:
            return plan
        t0 = time.perf_counter()
        with torch.no_grad():
            lo, lm = self.net(torch.as_tensor(x[None]))
        self.net_time += time.perf_counter() - t0
        self.net_calls += 1
        return float(self.cfg.offsets[int(lo.argmax())]), int(lm.argmax())


def run_set(env, agent, ad, races, seed0=0):
    def fn(e, obs, active, step):
        if step == 0:
            ad.reset()
        return ad.act(e, obs, active)

    out, trig = [], 0
    for k in range(races):
        ms = [SmartMistakes(env)] if k % 2 else []
        out.append(run_race(env, agent, seed=seed0 + k, mistakes=ms, act_fn=fn))
    return out


def train(net, X, yo, ym, epochs=40):
    torch.set_num_threads(4)
    n = len(X)
    perm = np.random.default_rng(0).permutation(n)
    val, tr = perm[: n // 5], perm[n // 5:]
    X_t = torch.as_tensor(X)
    yo_t, ym_t = torch.as_tensor(yo).long(), torch.as_tensor(ym).long()
    w_mode = torch.as_tensor(1.0 / np.maximum(np.bincount(ym, minlength=5), 1) ** 0.5, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for b in torch.as_tensor(tr).split(256):
            lo, lm = net(X_t[b])
            loss = nn.functional.cross_entropy(lo, yo_t[b]) + nn.functional.cross_entropy(lm, ym_t[b], weight=w_mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        lo, lm = net(X_t[val])
        return {"val_off_acc": float((lo.argmax(1) == yo_t[val]).float().mean()), "val_mode_acc": float((lm.argmax(1) == ym_t[val]).float().mean()),
                "val_exact": float(((lo.argmax(1) == yo_t[val]) & (lm.argmax(1) == ym_t[val])).float().mean()),
                "majority_off": float(np.bincount(yo[val]).max() / len(val)), "majority_mode": float(np.bincount(ym[val]).max() / len(val)), "n": n}


def main():
    profile = dict(np.load(os.path.join(RESULTS, "ref_profile.npz")))
    env = make_env()
    agent = load_policy(env, "ref_111s")
    env.reset()
    n_off, n_mode = len(MPCConfig().offsets), len(MPCConfig().pedal_modes)
    teacher = LearnedAdapter(env, agent, profile)
    run_set(env, agent, teacher, RACES_COLLECT)
    X, yo, ym = np.array(teacher.X, dtype=np.float32), np.array(teacher.y_off), np.array(teacher.y_mode)
    print("collected", len(X), flush=True)
    result = {}
    torch.manual_seed(0)
    for name, net in (("mlp", MLP(n_off, n_mode)), ("attn", Attn(n_off, n_mode))):
        info = {"round0": train(net, X, yo, ym)}
        print(name, info["round0"], flush=True)
        torch.set_num_threads(1)
        learner = LearnedAdapter(env, agent, profile, net=net)
        run_set(env, agent, learner, RACES_COLLECT // 2, seed0=100)
        X2 = np.concatenate([X, np.array(learner.X, dtype=np.float32)])
        yo2 = np.concatenate([yo, np.array(learner.y_off)])
        ym2 = np.concatenate([ym, np.array(learner.y_mode)])
        info["round1"] = train(net, X2, yo2, ym2)
        torch.set_num_threads(1)
        ev = LearnedAdapter(env, agent, profile, net=net)
        races = run_set(env, agent, ev, RACES_EVAL, seed0=200)
        info["races"] = races
        info["net_ms_per_call"] = 1000.0 * ev.net_time / max(1, ev.net_calls)
        result[name] = info
        print(name, "done", flush=True)
    env.close()
    save("t11_learned", result)


if __name__ == "__main__":
    main()
