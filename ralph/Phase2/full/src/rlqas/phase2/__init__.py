"""
RLQAS Phase 2 - Multi-RL Algorithm Support.

This package extends RLQAS Phase 1 with support for multiple RL algorithms,
starting with DQN in addition to the existing PPO implementation.
"""

__version__ = "2.0.0"

# Lazy imports to allow module initialization before Phase 1 is available
def __getattr__(name):
    """Lazy imports to allow module initialization before Phase 1 is available."""
    if name in [
        "RLAgent", "PPOAgent", "AgentConfig", "DQNAgent",
        "DQNConfig", "AgentFactory", "create_agent"
    ]:
        from .rl import (
            RLAgent, PPOAgent, AgentConfig, DQNAgent,
            DQNConfig, AgentFactory, create_agent,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RLAgent",
    "PPOAgent",
    "AgentConfig",
    "DQNAgent",
    "DQNConfig",
    "AgentFactory",
    "create_agent",
]
