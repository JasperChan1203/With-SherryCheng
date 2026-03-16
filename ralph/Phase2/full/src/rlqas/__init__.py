"""
RLQAS Phase 2 - Multi-RL Algorithm Support and Advanced Features.

This package extends RLQAS Phase 1 with:
- Multiple RL algorithms (PPO, DQN)
- Sequential testing framework for algorithm comparison
- HEA (Hardware Efficient Ansatz) search module
- Experiment management system
- Autonomous RL exploration framework
"""

__version__ = "2.0.0"

# Note: Phase 1 imports are handled at runtime via sys.path configuration
# Import Phase 2 components (Phase 1 dependencies resolved at runtime)
def __getattr__(name):
    """Lazy imports to allow module initialization before Phase 1 is available."""
    if name in [
        "RLAgent", "PPOAgent", "DQNAgent", "AgentConfig",
        "DQNConfig", "AgentFactory", "create_agent"
    ]:
        from rlqas.phase2.rl import (
            RLAgent, PPOAgent, DQNAgent, AgentConfig,
            DQNConfig, AgentFactory, create_agent,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # RL module
    "RLAgent",
    "PPOAgent",
    "DQNAgent",
    "AgentConfig",
    "DQNConfig",
    "AgentFactory",
    "create_agent",
]
