"""
AgentFactory: Factory for creating RL agents.

This module provides a unified factory interface for creating different types
of RL agents (PPO, DQN) in the RLQAS framework. It extends Phase 1 factory
patterns to support multi-algorithm selection.
"""

from typing import Dict, Optional, Any, Type
import gymnasium as gym

from rlqas_chem.rl.base_agent import RLAgent
from rlqas_chem.rl.ppo_agent import PPOAgent
from rlqas_chem.rl.dqn_agent import DQNAgent
from rlqas_chem.rl.a2c_agent import A2CAgent
from rlqas_chem.rl.sac_discrete_agent import SACDiscreteAgent
from rlqas_chem.rl.grpo_agent import GRPOAgent


class AgentFactory:
    """Factory for creating RL agents.

    This class provides a unified interface for instantiating different types
    of RL agents based on configuration. It supports PPO, DQN, and A2C agents
    and maintains backward compatibility with Phase 1 usage patterns.

    Example usage:
        # Create PPO agent
        ppo_agent = AgentFactory.create_agent("ppo", config, env)

        # Create DQN agent
        dqn_agent = AgentFactory.create_agent("dqn", config, env)

        # Create A2C agent
        a2c_agent = AgentFactory.create_agent("a2c", config, env)

        # Get available agent types
        available = AgentFactory.get_available_agents()
    """

    # Registry of agent types
    _AGENT_REGISTRY: Dict[str, Type[RLAgent]] = {
        "ppo": PPOAgent,
        "dqn": DQNAgent,
        "a2c": A2CAgent,
        "sac_discrete": SACDiscreteAgent,
        "grpo": GRPOAgent,
    }

    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        env: Optional[gym.Env] = None,
    ) -> RLAgent:
        """Create an RL agent of the specified type.

        Args:
            agent_type: Type of agent to create ("ppo", "dqn", or "a2c").
            config: Configuration dictionary for the agent.
            env: Gym environment (or callable that returns an environment).

        Returns:
            An instance of the specified agent type.

        Raises:
            ValueError: If agent_type is not registered.
            RuntimeError: If agent creation fails.
        """
        if agent_type not in cls._AGENT_REGISTRY:
            available = ", ".join(cls._AGENT_REGISTRY.keys())
            raise ValueError(
                f"Unknown agent type: '{agent_type}'. "
                f"Available agent types: {available}"
            )

        agent_class = cls._AGENT_REGISTRY[agent_type]

        try:
            return agent_class(config=config, env=env)
        except Exception as e:
            raise RuntimeError(f"Failed to create {agent_type} agent: {e}") from e

    @classmethod
    def register_agent(cls, name: str, agent_class: Type[RLAgent]) -> None:
        """Register a new agent type with the factory.

        This allows extending the factory with custom agent implementations.

        Args:
            name: Name to register the agent type under.
            agent_class: The agent class (must inherit from RLAgent).

        Raises:
            TypeError: If agent_class does not inherit from RLAgent.
        """
        if not issubclass(agent_class, RLAgent):
            raise TypeError(
                f"Agent class must inherit from RLAgent, "
                f"got {agent_class.__name__}"
            )

        cls._AGENT_REGISTRY[name] = agent_class

    @classmethod
    def unregister_agent(cls, name: str) -> bool:
        """Unregister an agent type from the factory.

        Args:
            name: Name of the agent type to unregister.

        Returns:
            True if agent was unregistered, False if it wasn't registered.
        """
        if name in cls._AGENT_REGISTRY:
            del cls._AGENT_REGISTRY[name]
            return True
        return False

    @classmethod
    def get_available_agents(cls) -> Dict[str, str]:
        """Get dictionary of available agent types.

        Returns:
            Dictionary mapping agent type names to their class names.
        """
        return {
            name: agent_class.__name__
            for name, agent_class in cls._AGENT_REGISTRY.items()
        }

    @classmethod
    def supported_agents(cls):
        """Return list of supported agent type names."""
        return list(cls._AGENT_REGISTRY.keys())

    @classmethod
    def is_agent_registered(cls, name: str) -> bool:
        """Check if an agent type is registered.

        Args:
            name: Name of the agent type.

        Returns:
            True if agent type is registered, False otherwise.
        """
        return name in cls._AGENT_REGISTRY

    @classmethod
    def get_agent_class(cls, name: str) -> Type[RLAgent]:
        """Get the agent class for a registered agent type.

        Args:
            name: Name of the agent type.

        Returns:
            The agent class.

        Raises:
            ValueError: If agent type is not registered.
        """
        if name not in cls._AGENT_REGISTRY:
            available = ", ".join(cls._AGENT_REGISTRY.keys())
            raise ValueError(
                f"Unknown agent type: '{name}'. "
                f"Available agent types: {available}"
            )
        return cls._AGENT_REGISTRY[name]


def create_agent(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
    env: Optional[gym.Env] = None,
) -> RLAgent:
    """Convenience function for creating agents.

    This is a shortcut for AgentFactory.create_agent().

    Args:
        agent_type: Type of agent to create ("ppo", "dqn", or "a2c").
        config: Configuration dictionary for the agent.
        env: Gym environment (or callable that returns an environment).

    Returns:
        An instance of the specified agent type.
    """
    return AgentFactory.create_agent(agent_type, config=config, env=env)
