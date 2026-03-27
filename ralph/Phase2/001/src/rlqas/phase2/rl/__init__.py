"""
Reinforcement learning module for RLQAS Phase 2.

This module extends Phase 1 RL capabilities with multi-algorithm support,
including DQN in addition to the existing PPO implementation.

Phase 2 adds:
- DQNAgent: Deep Q-Network agent for discrete action spaces
- AgentFactory: Unified factory for creating PPO and DQN agents
- DQNConfig: Configuration management for DQN hyperparameters
"""

# Import Phase 1 components (requires Phase 1 to be in Python path)
from rlqas.phase1.rl import RLAgent, PPOAgent, AgentConfig

from .dqn_agent import DQNAgent, DQNConfig
from .agent_factory import AgentFactory, create_agent

__all__ = [
    # Phase 1 imports (re-exported for convenience)
    "RLAgent",
    "PPOAgent",
    "AgentConfig",
    # Phase 2 additions
    "DQNAgent",
    "DQNConfig",
    "AgentFactory",
    "create_agent",
]
