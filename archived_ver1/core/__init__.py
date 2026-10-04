"""
core package initialization.
Xuất khẩu các lớp cốt lõi: CarDynamics, TrackGenerator, RacingEnv, CircuitProfiler, PPOAgent.
"""

from .dynamics import CarDynamics
from .track_generator import TrackGenerator
from .racing_env import RacingEnv
from .circuit_profiler import CircuitProfiler
from .ppo_agent import PPOAgent

__all__ = [
    "CarDynamics", 
    "TrackGenerator", 
    "RacingEnv", 
    "CircuitProfiler",
    "PPOAgent"
]
