"""
Đặc trưng vật lý trích xuất trực tiếp từ vector quan sát 35 chiều của MetaRacingEnv.
Dùng chung cho Policy luật (numpy) và mạng PPO v2 (torch) để cả hai "nhìn" cùng một tín hiệu:
  - Tốc độ cho phép theo đường bao phanh: v_allow = min_d sqrt(a_lat / |kappa_d| + 2 * a_brake * d)
  - Điểm ngắm Pure Pursuit nội suy theo khoảng cách nhìn trước.

Bố cục quan sát (xem game_env/racing_env.py):
  obs[0]  = speed / 40 (m/s)          obs[14] = kappa_now * 50 (clip ±2)
  obs[15 + 4i : 19 + 4i] = [dx/70, dy/30, sin(dtheta), kappa*50] tại d_i ∈ {10, 20, 35, 50, 70} m
"""

import numpy as np
import torch

LOOKAHEAD_DISTANCES = np.array([10.0, 20.0, 35.0, 50.0, 70.0], dtype=np.float32)
KAPPA_DISTANCES = np.array([0.0, 10.0, 20.0, 35.0, 50.0, 70.0], dtype=np.float32)
KAPPA_IDX = [14, 18, 22, 26, 30, 34]
DX_IDX = [15, 19, 23, 27, 31]
DY_IDX = [16, 20, 24, 28, 32]
V_ALLOW_CAP = 120.0  # km/h, trần khi đường thẳng


def allowed_speed_kmh(obs, a_lat=0.65 * 9.81, a_brake=5.0):
    """Tốc độ cho phép (km/h) tính từ độ cong nhìn trước trong obs (numpy, 1 mẫu)."""
    kappa = np.abs(np.asarray(obs, dtype=np.float32)[KAPPA_IDX]) / 50.0
    v_corner_sq = np.where(kappa > 1e-4, a_lat / np.maximum(kappa, 1e-4), np.inf)
    v_allow = np.sqrt(np.min(v_corner_sq + 2.0 * a_brake * KAPPA_DISTANCES))
    return float(min(v_allow * 3.6, V_ALLOW_CAP))


def allowed_speed_kmh_torch(obs: torch.Tensor, a_lat=0.65 * 9.81, a_brake=5.0) -> torch.Tensor:
    """Phiên bản torch theo batch: obs (B, 35) -> v_allow (B, 1) km/h."""
    kappa = obs[:, KAPPA_IDX].abs() / 50.0
    dist = torch.as_tensor(KAPPA_DISTANCES, device=obs.device)
    v_corner_sq = torch.where(kappa > 1e-4, a_lat / kappa.clamp(min=1e-4), torch.full_like(kappa, 1e6))
    v_allow = torch.sqrt((v_corner_sq + 2.0 * a_brake * dist).min(dim=1, keepdim=True).values)
    return (v_allow * 3.6).clamp(max=V_ALLOW_CAP)


def physics_features_torch(
    obs: torch.Tensor,
    a_lat=0.8 * 9.81,
    a_brake=5.0,
    wheelbase=2.47,
    max_steering_rad=float(np.deg2rad(40.0)),
    ld_gain=0.6,
    ld_min=8.0,
) -> torch.Tensor:
    """
    4 đặc trưng vật lý theo batch (B, 4), cùng công thức với RuleRacingPolicy:
      [v_allow/100, (v - v_allow)/20, steer Pure Pursuit gợi ý, ga/phanh gợi ý]
    """
    speed_ms = obs[:, 0:1] * 40.0
    v_allow = allowed_speed_kmh_torch(obs, a_lat, a_brake)
    overspeed = speed_ms * 3.6 - v_allow

    # Nội suy tuyến tính điểm ngắm tại khoảng cách Ld giữa 5 waypoint (10..70m)
    dists = torch.as_tensor(LOOKAHEAD_DISTANCES, device=obs.device)
    ld = (ld_gain * speed_ms + ld_min).clamp(float(dists[0]), float(dists[-1]))
    seg = torch.searchsorted(dists, ld.contiguous(), right=True).clamp(1, len(LOOKAHEAD_DISTANCES) - 1)
    d0, d1 = dists[seg - 1], dists[seg]
    w = (ld - d0) / (d1 - d0)
    dx_all, dy_all = obs[:, DX_IDX] * 70.0, obs[:, DY_IDX] * 30.0
    dx = torch.lerp(dx_all.gather(1, seg - 1), dx_all.gather(1, seg), w)
    dy = torch.lerp(dy_all.gather(1, seg - 1), dy_all.gather(1, seg), w)

    alpha = torch.atan2(dy, dx.clamp(min=1e-3))
    dist = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-3)
    steer = (torch.atan(2.0 * wheelbase * torch.sin(alpha) / dist) / max_steering_rad).clamp(-1.0, 1.0)
    throttle = (-0.15 * overspeed).clamp(-1.0, 1.0)
    return torch.cat([v_allow / 100.0, (overspeed / 20.0).clamp(-3.0, 3.0), steer, throttle], dim=1)


def lookahead_point(obs, lookahead_m):
    """Điểm ngắm (dx, dy) trong hệ trục xe, nội suy tuyến tính giữa các waypoint theo khoảng cách."""
    obs = np.asarray(obs, dtype=np.float32)
    dx = obs[DX_IDX] * 70.0
    dy = obs[DY_IDX] * 30.0
    d = float(np.clip(lookahead_m, LOOKAHEAD_DISTANCES[0], LOOKAHEAD_DISTANCES[-1]))
    return float(np.interp(d, LOOKAHEAD_DISTANCES, dx)), float(np.interp(d, LOOKAHEAD_DISTANCES, dy))
