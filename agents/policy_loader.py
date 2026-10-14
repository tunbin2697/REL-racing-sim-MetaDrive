"""
Nạp policy bất kỳ theo đường dẫn để các script đánh giá / demo dùng chung:
  - "rule"            -> RuleRacingPolicy (tham số mặc định)
  - "rule-fast"       -> RuleRacingPolicy với a_lat = 0.85g
  - file .pt có "arch": "v2" -> PPOAgentV2, còn lại -> PPOAgent v1
"""

import torch
from .ppo_agent import PPOAgent
from .ppo_agent_v2 import PPOAgentV2
from .rule_policy import RuleRacingPolicy


def load_policy(spec: str):
    if spec == "rule":
        return RuleRacingPolicy()
    if spec == "rule-fast":
        return RuleRacingPolicy(a_lat=0.85 * 9.81)
    ckpt = torch.load(spec, map_location="cpu")
    state_dim = int(ckpt.get("state_dim", 35))
    action_dim = int(ckpt.get("action_dim", 2))
    if ckpt.get("arch") == PPOAgentV2.ARCH:
        agent = PPOAgentV2(state_dim=state_dim, action_dim=action_dim, residual=bool(ckpt.get("residual", False)))
    else:
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(spec)
    return agent
