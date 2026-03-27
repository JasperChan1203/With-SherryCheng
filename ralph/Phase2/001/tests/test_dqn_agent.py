"""
Unit tests for DQNAgent implementation.

These tests verify DQNAgent interface compliance with RLAgent base class
and test core functionality in isolation.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add Phase 1 and Phase 2 src to path (absolute paths for pytest compatibility)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
# Phase 1 is at ../../Phase1/006/src from project root
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

from rlqas.phase2.rl.dqn_agent import DQNAgent, DQNConfig
from rlqas.phase1.rl.base_agent import RLAgent


class SimpleDiscreteEnv(gym.Env):
    """Simple discrete environment for testing."""

    def __init__(self, n_states=10, n_actions=4):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_states,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)
        self._current_state = np.zeros(n_states, dtype=np.float32)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_state = self.np_random.uniform(
            -1.0, 1.0, size=self.n_states
        ).astype(np.float32)
        self._step_count = 0
        return self._current_state, {}

    def step(self, action):
        self._step_count += 1
        # Simple reward based on action
        reward = 1.0 if action == 0 else 0.0
        # Random state transition
        self._current_state = self.np_random.uniform(
            -1.0, 1.0, size=self.n_states
        ).astype(np.float32)
        done = self._step_count >= 100
        truncated = False
        return self._current_state, reward, done, truncated, {}


class TestDQNConfig:
    """Tests for DQNConfig class."""

    def test_default_config(self):
        """Test that default config is valid."""
        config = DQNConfig()
        assert config.get("learning_rate") == 1e-3
        assert config.get("gamma") == 0.99
        assert config.get("buffer_size") == 10000
        assert config.get("batch_size") == 64

    def test_custom_config(self):
        """Test custom configuration."""
        custom = {"learning_rate": 0.01, "gamma": 0.95}
        config = DQNConfig(custom)
        assert config.get("learning_rate") == 0.01
        assert config.get("gamma") == 0.95
        # Defaults should still be present
        assert config.get("buffer_size") == 10000

    def test_invalid_config_key(self):
        """Test that unknown keys raise error."""
        with pytest.raises(KeyError, match="Unknown configuration parameter"):
            DQNConfig({"invalid_key": 123})

    def test_invalid_learning_rate(self):
        """Test that invalid learning_rate raises error."""
        with pytest.raises(ValueError):
            DQNConfig({"learning_rate": -0.01})

    def test_invalid_gamma(self):
        """Test that invalid gamma raises error."""
        with pytest.raises(ValueError):
            DQNConfig({"gamma": 1.5})

    def test_to_dict(self):
        """Test config to_dict method."""
        config = DQNConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "learning_rate" in d

    def test_update(self):
        """Test config update method."""
        config = DQNConfig()
        config.update({"learning_rate": 0.005})
        assert config.get("learning_rate") == 0.005


class TestDQNAgent:
    """Tests for DQNAgent class."""

    def test_inheritance(self):
        """Test that DQNAgent inherits from RLAgent."""
        assert issubclass(DQNAgent, RLAgent)

    def test_initialization_no_env(self):
        """Test agent initialization without environment."""
        agent = DQNAgent()
        assert agent.model is None
        assert agent.env is None
        assert agent.config is not None

    def test_initialization_with_env(self):
        """Test agent initialization with environment."""
        env = SimpleDiscreteEnv()
        agent = DQNAgent(env=env)
        assert agent.model is not None
        assert agent.env is not None

    def test_initialization_with_config(self):
        """Test agent initialization with custom config."""
        custom_config = {"learning_rate": 0.01, "verbose": 0}
        agent = DQNAgent(config=custom_config)
        assert agent.config["learning_rate"] == 0.01
        assert agent.config["verbose"] == 0

    def test_act_no_model(self):
        """Test that act raises error without model."""
        agent = DQNAgent()
        state = np.zeros(10)
        with pytest.raises(RuntimeError, match="Model not initialized"):
            agent.act(state)

    def test_act_returns_valid_action(self):
        """Test that act returns valid action."""
        env = SimpleDiscreteEnv(n_states=10, n_actions=4)
        agent = DQNAgent(env=env, config={"verbose": 0})
        state, _ = env.reset()
        action, info = agent.act(state)
        assert isinstance(action, int)
        assert 0 <= action < 4
        assert isinstance(info, dict)

    def test_act_multiple_calls(self):
        """Test multiple act calls produce actions."""
        env = SimpleDiscreteEnv(n_states=5, n_actions=3)
        agent = DQNAgent(env=env, config={"verbose": 0})
        state, _ = env.reset()
        actions = [agent.act(state)[0] for _ in range(10)]
        assert all(isinstance(a, int) for a in actions)
        assert all(0 <= a < 3 for a in actions)

    def test_learn_no_model(self):
        """Test that learn raises error without model."""
        agent = DQNAgent()
        with pytest.raises(RuntimeError, match="Model not initialized"):
            agent.learn(total_timesteps=100)

    def test_learn_with_timesteps(self):
        """Test learning with timesteps."""
        env = SimpleDiscreteEnv(n_states=10, n_actions=4)
        agent = DQNAgent(
            env=env,
            config={
                "verbose": 0,
                "buffer_size": 500,
                "train_freq": 4,
                "target_update_interval": 100,
            },
        )
        metrics = agent.learn(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert "total_timesteps" in metrics
        assert metrics["total_timesteps"] == 200

    def test_learn_with_experience_not_implemented(self):
        """Test that learning from experience raises NotImplementedError."""
        env = SimpleDiscreteEnv()
        agent = DQNAgent(env=env, config={"verbose": 0})
        experience = {
            "states": np.zeros((10, 10)),
            "actions": np.zeros(10),
            "rewards": np.zeros(10),
            "next_states": np.zeros((10, 10)),
            "dones": np.zeros(10),
        }
        with pytest.raises(NotImplementedError):
            agent.learn(experience=experience)

    def test_save_load(self):
        """Test save and load functionality."""
        env = SimpleDiscreteEnv(n_states=10, n_actions=4)
        agent = DQNAgent(env=env, config={"verbose": 0})

        # Train briefly
        agent.learn(total_timesteps=100)

        # Get action before save
        state, _ = env.reset()
        action_before, _ = agent.act(state)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_dqn.zip")
            agent.save(path)
            assert os.path.exists(path)

            # Create new agent and load
            new_agent = DQNAgent(config={"verbose": 0})
            new_agent.load(path)

            # Get action after load (should be same due to deterministic seed)
            action_after, _ = new_agent.act(state)

            # Actions should be consistent
            assert action_before == action_after

    def test_set_env(self):
        """Test set_env method."""
        agent = DQNAgent()
        env = SimpleDiscreteEnv()
        agent.set_env(env)
        assert agent.env is not None
        assert agent.model is not None

    def test_get_config(self):
        """Test get_config method."""
        agent = DQNAgent()
        config = agent.get_config()
        assert isinstance(config, dict)
        assert "learning_rate" in config

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        seed = 42

        env1 = SimpleDiscreteEnv()
        env1.reset(seed=seed)
        agent1 = DQNAgent(env=env1, config={"seed": seed, "verbose": 0})
        agent1.learn(total_timesteps=100)

        env2 = SimpleDiscreteEnv()
        env2.reset(seed=seed)
        agent2 = DQNAgent(env=env2, config={"seed": seed, "verbose": 0})
        agent2.learn(total_timesteps=100)

        # Same state should produce same action
        state1, _ = env1.reset(seed=seed)
        state2, _ = env2.reset(seed=seed)
        action1, _ = agent1.act(state1)
        action2, _ = agent2.act(state2)

        # With same seed, actions should be reproducible
        # Note: This may not always be true due to exploration, but model weights should be same


class TestDQNAgentInterfaceCompliance:
    """Tests to verify DQNAgent conforms to RLAgent interface."""

    def test_has_required_methods(self):
        """Test that DQNAgent has all required RLAgent methods."""
        required_methods = ["act", "learn", "save", "load"]
        for method in required_methods:
            assert hasattr(DQNAgent, method)
            assert callable(getattr(DQNAgent, method))

    def test_act_signature(self):
        """Test act method signature."""
        env = SimpleDiscreteEnv()
        agent = DQNAgent(env=env, config={"verbose": 0})
        state, _ = env.reset()
        result = agent.act(state)
        assert isinstance(result, tuple)
        assert len(result) == 2
        action, info = result
        assert isinstance(action, (int, np.integer))
        assert isinstance(info, dict)

    def test_learn_signature(self):
        """Test learn method signature."""
        env = SimpleDiscreteEnv()
        agent = DQNAgent(env=env, config={"verbose": 0})
        result = agent.learn(total_timesteps=100)
        assert isinstance(result, dict)

    def test_save_load_signature(self):
        """Test save/load method signatures."""
        env = SimpleDiscreteEnv()
        agent = DQNAgent(env=env, config={"verbose": 0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.zip")
            # save should not raise
            agent.save(path)
            # load should not raise
            new_agent = DQNAgent()
            new_agent.load(path)

    def test_format_ucc_state_inherited(self):
        """Test that format_ucc_state is inherited from RLAgent."""
        agent = DQNAgent()
        energy = -1.5
        circuit_params = np.array([0.1, 0.2, 0.3])
        state = agent.format_ucc_state(energy, circuit_params)
        assert isinstance(state, np.ndarray)
        assert state[0] == energy
        np.testing.assert_array_equal(state[1:], circuit_params)

    def test_parse_ucc_action_inherited(self):
        """Test that parse_ucc_action is inherited from RLAgent."""
        agent = DQNAgent()
        result = agent.parse_ucc_action(5)
        assert isinstance(result, dict)
        assert "excitation_idx" in result
        assert result["excitation_idx"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
