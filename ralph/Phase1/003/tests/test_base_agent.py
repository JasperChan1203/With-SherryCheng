"""
Unit tests for RLAgent abstract base class.
"""

import pytest
import numpy as np
from src.modules.rl_agents.base_agent import RLAgent


class ConcreteAgent(RLAgent):
    """Concrete implementation for testing abstract methods."""

    def act(self, state):
        return 0, {}

    def learn(self, experience):
        return {}

    def save(self, path):
        pass

    def load(self, path):
        pass


class TestRLAgent:
    """Test RLAgent abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that RLAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RLAgent()

    def test_concrete_subclass_can_be_instantiated(self):
        """Test that concrete subclass can be instantiated."""
        agent = ConcreteAgent()
        assert isinstance(agent, RLAgent)
        assert isinstance(agent, ConcreteAgent)

    def test_abstract_methods_must_be_implemented(self):
        """Test that subclass must implement all abstract methods."""
        class IncompleteAgent(RLAgent):
            # Missing act, learn, save, load
            pass

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_format_ucc_state_default_implementation(self):
        """Test default implementation of format_ucc_state."""
        agent = ConcreteAgent()
        energy = -1.5
        circuit_params = np.array([0.1, 0.2, 0.3])
        state = agent.format_ucc_state(energy, circuit_params)

        # Check shape and values
        expected = np.concatenate([[energy], circuit_params])
        assert np.array_equal(state, expected)
        assert state.shape == (4,)
        assert state[0] == energy
        assert np.array_equal(state[1:], circuit_params)

    def test_format_ucc_state_with_empty_params(self):
        """Test format_ucc_state with empty circuit parameters."""
        agent = ConcreteAgent()
        energy = 0.0
        circuit_params = np.array([])
        state = agent.format_ucc_state(energy, circuit_params)

        expected = np.array([energy])
        assert np.array_equal(state, expected)
        assert state.shape == (1,)

    def test_parse_ucc_action_default_implementation(self):
        """Test default implementation of parse_ucc_action."""
        agent = ConcreteAgent()
        action_idx = 42
        result = agent.parse_ucc_action(action_idx)

        assert isinstance(result, dict)
        assert result["excitation_idx"] == action_idx
        assert len(result) == 1

    def test_parse_ucc_action_with_different_indices(self):
        """Test parse_ucc_action with various action indices."""
        agent = ConcreteAgent()
        for idx in [0, 1, 10, -5]:
            result = agent.parse_ucc_action(idx)
            assert result["excitation_idx"] == idx

    def test_act_signature(self):
        """Test that act() returns tuple of (int, dict)."""
        agent = ConcreteAgent()
        state = np.array([1.0, 2.0, 3.0])
        action, info = agent.act(state)

        assert isinstance(action, int)
        assert isinstance(info, dict)

    def test_learn_signature(self):
        """Test that learn() returns dict."""
        agent = ConcreteAgent()
        experience = {
            "states": np.array([[1.0, 2.0]]),
            "actions": np.array([0]),
            "rewards": np.array([1.0]),
            "next_states": np.array([[1.0, 2.0]]),
            "dones": np.array([False]),
        }
        metrics = agent.learn(experience)

        assert isinstance(metrics, dict)