"""Xem bằng mắt các chế độ trong demo 3D, không train. Toàn bộ xe do AI lái (chế độ khán giả).
    python demo_compare.py --mode ref        # chỉ ref_111s
    python demo_compare.py --mode mpc        # ref_111s + adapter MPC
    python demo_compare.py --mode mpc_faults # thêm lỗi lockup, understeer, oversteer
Thêm --laps N để đổi số vòng (mặc định 3). Các phím camera giống demo gốc: [B] nhìn từ trên cao, [Tab] đổi xe, [0] xe dẫn đầu.
"""
import argparse
import os
import sys

import numpy as np

from common import RESULTS
from mpc_adapter import MPCAdapter, MPCConfig, BRAVE_CONFIG
from grip_faults import TopFieldFaults
import demo.race_multicar as demo

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ref", "mpc", "mpc_faults"], default="mpc_faults")
parser.add_argument("--laps", type=int, default=3)
args = parser.parse_args()
USE_MPC = args.mode.startswith("mpc")
USE_MISTAKES = args.mode.endswith("faults")
LABEL = {"ref": "ref_111s", "mpc": "ref_111s + MPC", "mpc_faults": "ref_111s + MPC + lỗi bám lốp"}[args.mode]
BaseEnv, BaseAgent = demo.MultiCarRacingEnv, demo.PPOAgent
state = {"env": None}


class DemoEnv(BaseEnv):
    """Giữ obs mới nhất cho adapter, tạo lỗi có kiểm soát và hiện chế độ đang xem trên HUD."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        state["env"] = self
        self.mistakes = None
        self.latest_obs = None
        self.notice = ""

    def reset(self, **k):
        obs, info = super().reset(**k)
        self.latest_obs = obs
        self.mistakes = TopFieldFaults(self) if USE_MISTAKES else None
        self.notice = ""
        if state.get("adapter") is not None:
            state["adapter"].reset()
        return obs, info

    def step(self, actions):
        if self.mistakes is not None:
            before = self.mistakes.triggered
            self.mistakes.apply(self, self.current_step, actions)
            if self.mistakes.triggered > before:
                ev = self.mistakes.log[-1]
                self.notice = f"AI{ev['car'] + 1} {ev['kind']} ở phút {self.current_step * self.dt / 60:.1f}"
                print(">> [LỖI]", self.notice, flush=True)
        out = super().step(actions)
        self.latest_obs = out[0]
        return out

    def render(self, *a, **k):
        text = dict(k.pop("text", {}) or {})
        text["Chế độ"] = LABEL
        if self.notice:
            text["Lỗi gần nhất"] = self.notice
        return super().render(*a, text=text, **k)


class DemoAgent(BaseAgent):
    """Demo gọi policy từng xe một; adapter tính cho cả đoàn trong lần gọi đầu mỗi bước rồi trả lại từng xe."""

    cache_step, cache = -1, {}

    def select_action(self, state_obs, deterministic=False):
        env = state["env"]
        if state.get("adapter") is None:
            if env is None or not USE_MPC:
                return super().select_action(state_obs, deterministic)
            state["adapter"] = MPCAdapter(env, self, dict(np.load(os.path.join(RESULTS, "ref_profile.npz"))), MPCConfig(**BRAVE_CONFIG))
        if self.cache_step != env.current_step:
            active = env.active_mask()
            acts = state["adapter"].act(env, env.latest_obs, active)
            self.cache = dict(zip(np.flatnonzero(active), acts))
            self.cache_step = env.current_step
            self.order = iter(sorted(self.cache))
        return self.cache[next(self.order)], 0.0, 0.0


demo.MultiCarRacingEnv, demo.PPOAgent = DemoEnv, DemoAgent
sys.argv = ["race_multicar.py", "--spectate", "--laps", str(args.laps), "--model", "models/ref_111s/best_agent.pt"]
print(f">> ĐANG XEM: {LABEL}", flush=True)
demo.main()
