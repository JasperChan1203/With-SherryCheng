"""
RL Agents module for RLQAS.

This module provides reinforcement learning agents compatible with the
RLQAS framework. The primary interface is the RLAgent abstract base class,
with concrete implementations for various algorithms.

Classes:
    RLAgent: Abstract base class for RL agents
    PPOAgent: Proximal Policy Optimization agent using Stable-Baselines3

Helper Functions:
    get_device: Get appropriate device (GPU/CPU) for PyTorch/SB3
    set_seed: Set all random seeds for reproducibility
"""

from .base_agent import RLAgent
from .ppo_agent import PPOAgent, get_device

__all__ = ["RLAgent", "PPOAgent", "get_device"]