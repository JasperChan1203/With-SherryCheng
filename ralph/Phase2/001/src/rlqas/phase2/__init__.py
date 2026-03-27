"""
RLQAS Phase 2 - Multi-RL Algorithm Support.

This package extends RLQAS Phase 1 with support for multiple RL algorithms,
starting with DQN in addition to the existing PPO implementation.
"""

from .rl import (
    RLAgent,
    PPOAgent,
    AgentConfig,
    DQNAgent,
    DQNConfig,
    AgentFactory,
    create_agent,
)

__version__ = "2.0.0"
__all__ = [
    "RLAgent",
    "PPOAgent",
    "AgentConfig",
    "DQNAgent",
    "DQNConfig",
    "AgentFactory",
    "create_agent",
]
