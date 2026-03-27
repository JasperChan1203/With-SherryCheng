"""
Reinforcement learning module for RLQAS Phase 2.

This module extends Phase 1 RL capabilities with multi-algorithm support,
including DQN and A2C in addition to the existing PPO implementation.

Phase 2 adds:
- DQNAgent: Deep Q-Network agent for discrete action spaces
- A2CAgent: Advantage Actor-Critic agent for discrete action spaces
- AgentFactory: Unified factory for creating PPO, DQN, and A2C agents
- DQNConfig: Configuration management for DQN hyperparameters
- A2CConfig: Configuration management for A2C hyperparameters
"""

# Lazy imports to handle Phase 1 dependency
def __getattr__(name):
    """Lazy imports to allow module initialization before Phase 1 is available."""
    if name in ["RLAgent", "PPOAgent", "AgentConfig"]:
        from rlqas.phase1.rl import RLAgent, PPOAgent, AgentConfig
        return locals()[name]
    elif name == "DQNAgent":
        from .dqn_agent import DQNAgent
        return DQNAgent
    elif name == "DQNConfig":
        from .dqn_agent import DQNConfig
        return DQNConfig
    elif name == "A2CAgent":
        from .a2c_agent import A2CAgent
        return A2CAgent
    elif name == "A2CConfig":
        from .a2c_agent import A2CConfig
        return A2CConfig
    elif name == "SACDiscreteAgent":
        from .sac_discrete_agent import SACDiscreteAgent
        return SACDiscreteAgent
    elif name == "SACDiscreteConfig":
        from .sac_discrete_agent import SACDiscreteConfig
        return SACDiscreteConfig
    elif name == "AgentFactory":
        from .agent_factory import AgentFactory
        return AgentFactory
    elif name == "create_agent":
        from .agent_factory import create_agent
        return create_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Phase 1 imports (re-exported for convenience)
    "RLAgent",
    "PPOAgent",
    "AgentConfig",
    # Phase 2 additions
    "DQNAgent",
    "DQNConfig",
    "A2CAgent",
    "A2CConfig",
    "SACDiscreteAgent",
    "SACDiscreteConfig",
    "AgentFactory",
    "create_agent",
]
