"""
Tests for AgentFactory implementation.

These tests verify the agent factory can create both PPO and DQN agents
and supports extension with custom agent types.
"""

import os
import sys
import pytest
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Add Phase 1 and Phase 2 src to path (absolute paths for pytest compatibility)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
# Phase 1 is at ../../Phase1/006/src from project root
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

from rlqas.phase2.rl.agent_factory import AgentFactory, create_agent
from rlqas.phase1.rl.base_agent import RLAgent
from rlqas.phase1.rl.ppo_agent import PPOAgent
from rlqas.phase2.rl.dqn_agent import DQNAgent


class SimpleEnv(gym.Env):
    """Simple environment for testing."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.np_random.uniform(-1.0, 1.0, size=5).astype(np.float32), {}

    def step(self, action):
        reward = float(action)
        obs = self.np_random.uniform(-1.0, 1.0, size=5).astype(np.float32)
        return obs, reward, False, False, {}


class TestAgentFactory:
    """Tests for AgentFactory class."""

    def test_create_ppo_agent(self):
        """Test creating PPO agent."""
        env = SimpleEnv()
        agent = AgentFactory.create_agent("ppo", env=env, config={"verbose": 0})
        assert isinstance(agent, PPOAgent)
        assert isinstance(agent, RLAgent)

    def test_create_dqn_agent(self):
        """Test creating DQN agent."""
        env = SimpleEnv()
        agent = AgentFactory.create_agent("dqn", env=env, config={"verbose": 0})
        assert isinstance(agent, DQNAgent)
        assert isinstance(agent, RLAgent)

    def test_create_agent_unknown_type(self):
        """Test creating agent with unknown type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create_agent("unknown_agent")

    def test_create_agent_without_env(self):
        """Test creating agent without environment."""
        agent = AgentFactory.create_agent("dqn", config={"verbose": 0})
        assert isinstance(agent, DQNAgent)
        assert agent.model is None

    def test_create_agent_with_config(self):
        """Test creating agent with custom config."""
        custom_config = {"learning_rate": 0.01, "verbose": 0}
        agent = AgentFactory.create_agent(
            "dqn", config=custom_config, env=SimpleEnv()
        )
        assert agent.config["learning_rate"] == 0.01

    def test_get_available_agents(self):
        """Test getting available agent types."""
        available = AgentFactory.get_available_agents()
        assert isinstance(available, dict)
        assert "ppo" in available
        assert "dqn" in available
        assert available["ppo"] == "PPOAgent"
        assert available["dqn"] == "DQNAgent"

    def test_is_agent_registered(self):
        """Test checking if agent is registered."""
        assert AgentFactory.is_agent_registered("ppo")
        assert AgentFactory.is_agent_registered("dqn")
        assert not AgentFactory.is_agent_registered("unknown")

    def test_get_agent_class(self):
        """Test getting agent class."""
        ppo_class = AgentFactory.get_agent_class("ppo")
        assert ppo_class == PPOAgent
        dqn_class = AgentFactory.get_agent_class("dqn")
        assert dqn_class == DQNAgent

    def test_get_agent_class_unknown(self):
        """Test getting unknown agent class raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.get_agent_class("unknown")

    def test_register_agent(self):
        """Test registering custom agent."""

        class CustomAgent(RLAgent):
            def act(self, state):
                return 0, {}

            def learn(self, experience):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

        AgentFactory.register_agent("custom", CustomAgent)
        assert AgentFactory.is_agent_registered("custom")
        assert AgentFactory.get_agent_class("custom") == CustomAgent

        # Clean up
        AgentFactory.unregister_agent("custom")

    def test_register_agent_invalid_type(self):
        """Test registering non-RLAgent class raises error."""

        class NotAnAgent:
            pass

        with pytest.raises(TypeError, match="must inherit from RLAgent"):
            AgentFactory.register_agent("not_an_agent", NotAnAgent)

    def test_unregister_agent(self):
        """Test unregistering agent."""
        # Register first
        class TempAgent(RLAgent):
            def act(self, state):
                return 0, {}

            def learn(self, experience):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

        AgentFactory.register_agent("temp", TempAgent)
        assert AgentFactory.is_agent_registered("temp")

        # Unregister
        result = AgentFactory.unregister_agent("temp")
        assert result is True
        assert not AgentFactory.is_agent_registered("temp")

    def test_unregister_unknown_agent(self):
        """Test unregistering unknown agent returns False."""
        result = AgentFactory.unregister_agent("nonexistent")
        assert result is False

    def test_create_agent_function(self):
        """Test convenience create_agent function."""
        env = SimpleEnv()
        agent = create_agent("dqn", env=env, config={"verbose": 0})
        assert isinstance(agent, DQNAgent)


class TestAgentFactoryIntegration:
    """Integration tests for AgentFactory."""

    def test_ppo_agent_learns(self):
        """Test PPO agent can learn."""
        env = SimpleEnv()
        agent = AgentFactory.create_agent(
            "ppo", env=env, config={"verbose": 0, "n_steps": 128}
        )
        metrics = agent.learn(total_timesteps=64)
        assert isinstance(metrics, dict)
        assert "total_timesteps" in metrics

    def test_dqn_agent_learns(self):
        """Test DQN agent can learn."""
        env = SimpleEnv()
        agent = AgentFactory.create_agent(
            "dqn", env=env, config={"verbose": 0, "buffer_size": 100}
        )
        metrics = agent.learn(total_timesteps=64)
        assert isinstance(metrics, dict)
        assert "total_timesteps" in metrics

    def test_agent_switching(self):
        """Test switching between agent types."""
        env = SimpleEnv()

        # Create PPO agent
        ppo_agent = AgentFactory.create_agent("ppo", env=env, config={"verbose": 0})
        assert isinstance(ppo_agent, PPOAgent)

        # Create DQN agent
        dqn_agent = AgentFactory.create_agent("dqn", env=env, config={"verbose": 0})
        assert isinstance(dqn_agent, DQNAgent)

        # Both should work independently
        state, _ = env.reset()
        ppo_action, _ = ppo_agent.act(state)
        dqn_action, _ = dqn_agent.act(state)

        assert isinstance(ppo_action, int)
        assert isinstance(dqn_action, int)

    def test_save_load_roundtrip(self):
        """Test save/load roundtrip for both agent types."""
        import tempfile

        env = SimpleEnv()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test DQN
            dqn_path = os.path.join(tmpdir, "dqn.zip")
            dqn_agent = AgentFactory.create_agent(
                "dqn", env=env, config={"verbose": 0}
            )
            dqn_agent.learn(total_timesteps=50)
            dqn_agent.save(dqn_path)

            dqn_loaded = AgentFactory.create_agent("dqn", config={"verbose": 0})
            dqn_loaded.load(dqn_path)
            assert dqn_loaded.model is not None

            # Test PPO
            ppo_path = os.path.join(tmpdir, "ppo.zip")
            ppo_agent = AgentFactory.create_agent(
                "ppo", env=env, config={"verbose": 0, "n_steps": 128}
            )
            ppo_agent.learn(total_timesteps=64)
            ppo_agent.save(ppo_path)

            ppo_loaded = AgentFactory.create_agent("ppo", config={"verbose": 0})
            ppo_loaded.load(ppo_path)
            assert ppo_loaded.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
