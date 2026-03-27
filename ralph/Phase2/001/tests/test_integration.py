"""
Integration tests for Phase 2 with Phase 1 components.

These tests verify that DQN agents work correctly with Phase 1
environments and components.
"""

import os
import sys
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

from rlqas.phase2.rl import (
    DQNAgent,
    DQNConfig,
    AgentFactory,
    create_agent,
    RLAgent,
    PPOAgent,
)
from rlqas.phase1.rl.base_agent import RLAgent as Phase1RLAgent


class MockUCCEnv(gym.Env):
    """Mock UCC search environment for integration testing.

    This simulates the Phase 1 UCCSearchEnv interface for testing
    compatibility without requiring full quantum chemistry setup.
    """

    def __init__(self, state_dim=20, n_actions=10):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)
        self._step_count = 0
        self._max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        # Simulate UCC state: [energy] + [circuit_params]
        obs = self.np_random.uniform(-1.0, 1.0, size=self.state_dim).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        # Sparse reward simulation (like quantum architecture search)
        reward = 1.0 if action == self.np_random.integers(0, 2) else 0.0
        obs = self.np_random.uniform(-1.0, 1.0, size=self.state_dim).astype(np.float32)
        done = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, done, truncated, {"excitation_idx": action}


class TestPhase1Compatibility:
    """Tests for Phase 1 compatibility."""

    def test_dqn_inherits_from_phase1_rlagent(self):
        """Test DQNAgent inherits from Phase 1 RLAgent."""
        assert issubclass(DQNAgent, Phase1RLAgent)
        assert issubclass(DQNAgent, RLAgent)

    def test_dqn_implements_required_methods(self):
        """Test DQNAgent implements all RLAgent required methods."""
        required = ["act", "learn", "save", "load"]
        for method in required:
            assert hasattr(DQNAgent, method)
            assert callable(getattr(DQNAgent, method))

    def test_dqn_ucc_helpers_inherited(self):
        """Test UCC helper methods are inherited."""
        agent = DQNAgent()
        # format_ucc_state
        energy = -1.5
        circuit_params = np.array([0.1, 0.2, 0.3])
        state = agent.format_ucc_state(energy, circuit_params)
        assert len(state) == 4
        assert state[0] == energy

        # parse_ucc_action
        result = agent.parse_ucc_action(5)
        assert "excitation_idx" in result
        assert result["excitation_idx"] == 5

    def test_agent_factory_supports_both_types(self):
        """Test AgentFactory supports both PPO and DQN."""
        available = AgentFactory.get_available_agents()
        assert "ppo" in available
        assert "dqn" in available

    def test_create_agent_function(self):
        """Test create_agent convenience function."""
        env = MockUCCEnv()
        agent = create_agent("dqn", env=env, config={"verbose": 0})
        assert isinstance(agent, DQNAgent)


class TestIntegrationWithMockEnvironment:
    """Integration tests with mock UCC-like environment."""

    def test_dqn_training_loop(self):
        """Test complete DQN training loop."""
        env = MockUCCEnv(state_dim=20, n_actions=10)
        agent = DQNAgent(
            env=env,
            config={
                "verbose": 0,
                "buffer_size": 500,
                "train_freq": 4,
                "learning_rate": 0.01,
            },
        )

        # Training loop
        total_timesteps = 200
        metrics = agent.learn(total_timesteps=total_timesteps)

        assert metrics["total_timesteps"] == total_timesteps
        assert agent.model is not None

    def test_dqn_act_after_training(self):
        """Test DQN action selection after training."""
        env = MockUCCEnv(state_dim=15, n_actions=8)
        agent = DQNAgent(
            env=env, config={"verbose": 0, "buffer_size": 500, "train_freq": 4}
        )

        # Train
        agent.learn(total_timesteps=100)

        # Act
        state, _ = env.reset()
        action, info = agent.act(state)

        assert 0 <= action < 8
        assert isinstance(info, dict)

    def test_dqn_save_load_integration(self):
        """Test DQN save/load in integration context."""
        import tempfile

        env = MockUCCEnv(state_dim=10, n_actions=5)
        agent = DQNAgent(env=env, config={"verbose": 0, "buffer_size": 200})

        # Train briefly
        agent.learn(total_timesteps=50)

        # Get pre-save action
        state, _ = env.reset(seed=42)
        pre_action, _ = agent.act(state)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "integration_test.zip")
            agent.save(path)

            new_agent = DQNAgent(config={"verbose": 0})
            new_agent.load(path)

            # Get post-load action
            post_action, _ = new_agent.act(state)

            # Should produce same action with same weights
            assert pre_action == post_action

    def test_ppo_dqn_comparison_same_interface(self):
        """Test PPO and DQN have same interface."""
        env = MockUCCEnv(state_dim=10, n_actions=5)

        ppo_agent = AgentFactory.create_agent(
            "ppo", env=env, config={"verbose": 0, "n_steps": 128}
        )
        dqn_agent = AgentFactory.create_agent(
            "dqn", env=env, config={"verbose": 0, "buffer_size": 200}
        )

        state, _ = env.reset()

        # Both should have act method returning (action, info)
        ppo_action, ppo_info = ppo_agent.act(state)
        dqn_action, dqn_info = dqn_agent.act(state)

        assert isinstance(ppo_action, int)
        assert isinstance(dqn_action, int)
        assert isinstance(ppo_info, dict)
        assert isinstance(dqn_info, dict)

        # Both should have learn method returning dict
        ppo_metrics = ppo_agent.learn(total_timesteps=64)
        dqn_metrics = dqn_agent.learn(total_timesteps=64)

        assert isinstance(ppo_metrics, dict)
        assert isinstance(dqn_metrics, dict)

    def test_ucc_state_formatting_with_agent(self):
        """Test UCC state formatting works with agent."""
        agent = DQNAgent()

        # Simulate UCC state from Phase 1
        energy = -7.5  # Hartree
        circuit_params = np.array([0.1, 0.2, -0.15, 0.3])

        state = agent.format_ucc_state(energy, circuit_params)

        # Should be 1D array with energy first
        assert state.ndim == 1
        assert len(state) == 5
        assert state[0] == energy
        np.testing.assert_array_equal(state[1:], circuit_params)

    def test_ucc_action_parsing_with_agent(self):
        """Test UCC action parsing works with agent."""
        agent = DQNAgent()

        # Parse action index
        action_idx = 7
        result = agent.parse_ucc_action(action_idx)

        assert isinstance(result, dict)
        assert "excitation_idx" in result
        assert result["excitation_idx"] == action_idx


class TestSparseRewardEnvironment:
    """Test DQN in sparse reward environment (typical for quantum search)."""

    def test_dqn_handles_sparse_rewards(self):
        """Test DQN can handle sparse reward signals."""

        class SparseRewardEnv(gym.Env):
            """Environment with very sparse rewards."""

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(4)
                self._target_action = 2
                self._step_count = 0

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._step_count = 0
                return self.np_random.uniform(-1.0, 1.0, size=10).astype(np.float32), {}

            def step(self, action):
                self._step_count += 1
                # Only reward specific action
                reward = 1.0 if action == self._target_action else 0.0
                done = self._step_count >= 20
                obs = self.np_random.uniform(-1.0, 1.0, size=10).astype(np.float32)
                return obs, reward, done, False, {}

        env = SparseRewardEnv()
        agent = DQNAgent(
            env=env,
            config={
                "verbose": 0,
                "buffer_size": 1000,
                "exploration_fraction": 0.3,  # More exploration for sparse rewards
                "learning_rate": 0.001,
            },
        )

        # Should be able to train without errors
        metrics = agent.learn(total_timesteps=500)
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
