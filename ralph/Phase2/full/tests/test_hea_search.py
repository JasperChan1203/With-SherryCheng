"""
Unit tests for HEA Search Module.

These tests verify HEASearchEnv, HEACircuitBuilder, and HEASearchController
functionality.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

from rlqas.phase2.hea_search import (
    HEASearchEnv,
    HEACircuitBuilder,
    create_hea_circuit,
    HEASearchController,
    run_hea_search,
    HEAConfig,
    get_default_config,
)


class TestHEASearchEnv:
    """Tests for HEASearchEnv class."""

    def test_initialization(self):
        """Test environment initialization."""
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        assert env.n_qubits == 4
        assert env.max_layers == 3
        assert env._current_layer == 0

    def test_reset(self):
        """Test environment reset."""
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert "layer" in info
        assert info["layer"] == 0

    def test_step(self):
        """Test environment step."""
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        obs, _ = env.reset(seed=42)

        # Take a step
        action = 0
        next_obs, reward, done, truncated, info = env.step(action)
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert "energy" in info
        assert "entanglement" in info
        assert "rotation" in info

    def test_full_episode(self):
        """Test running a full episode."""
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        obs, _ = env.reset()

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        assert steps == 3  # max_layers

    def test_get_circuit_config(self):
        """Test getting circuit configuration."""
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        env.reset()

        # Take some actions
        for _ in range(2):
            action = 0
            env.step(action)

        config = env.get_circuit_config()
        assert config["n_qubits"] == 4
        assert config["max_layers"] == 3
        assert "entanglement_history" in config
        assert "rotation_history" in config

    def test_action_space(self):
        """Test action space size."""
        env = HEASearchEnv(
            n_qubits=4,
            max_layers=3,
            entanglement_patterns=["linear", "circular"],
            rotation_gates=["rx", "ry"],
        )
        # Should be 2 patterns * 2 gates = 4 actions
        assert env.action_space.n == 4

    def test_observation_space(self):
        """Test observation space."""
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape


class TestHEACircuitBuilder:
    """Tests for HEACircuitBuilder class."""

    def test_initialization(self):
        """Test circuit builder initialization."""
        builder = HEACircuitBuilder(n_qubits=4, n_layers=3)
        assert builder.n_qubits == 4
        assert builder.n_layers == 3
        assert builder.entanglement_pattern == "linear"

    def test_build_circuit(self):
        """Test building a circuit."""
        builder = HEACircuitBuilder(n_qubits=4, n_layers=3)
        circuit = builder.build()

        assert circuit["n_qubits"] == 4
        assert circuit["n_layers"] == 3
        assert len(circuit["layers"]) == 3
        assert "entanglement_pairs" in circuit

    def test_entanglement_linear(self):
        """Test linear entanglement pattern."""
        builder = HEACircuitBuilder(n_qubits=4, n_layers=2, entanglement_pattern="linear")
        circuit = builder.build()

        pairs = circuit["entanglement_pairs"]
        # Linear: (0,1), (1,2), (2,3)
        assert len(pairs) == 3
        assert (0, 1) in pairs
        assert (1, 2) in pairs
        assert (2, 3) in pairs

    def test_entanglement_circular(self):
        """Test circular entanglement pattern."""
        builder = HEACircuitBuilder(n_qubits=4, n_layers=2, entanglement_pattern="circular")
        circuit = builder.build()

        pairs = circuit["entanglement_pairs"]
        # Circular: (0,1), (1,2), (2,3), (3,0)
        assert len(pairs) == 4
        assert (3, 0) in pairs

    def test_entanglement_fully_connected(self):
        """Test fully connected entanglement pattern."""
        builder = HEACircuitBuilder(n_qubits=4, n_layers=2, entanglement_pattern="fully_connected")
        circuit = builder.build()

        pairs = circuit["entanglement_pairs"]
        # Full: all pairs = 4*(4-1)/2 = 6
        assert len(pairs) == 6

    def test_parameter_sharing_global(self):
        """Test global parameter sharing."""
        builder = HEACircuitBuilder(
            n_qubits=4, n_layers=3,
            parameter_sharing="global",
            rotation_gates=["rx", "ry", "rz"]
        )
        circuit = builder.build()

        # Global: 3 parameters (one per gate type)
        assert circuit["total_parameters"] == 3

    def test_parameter_sharing_layer_wise(self):
        """Test layer-wise parameter sharing."""
        builder = HEACircuitBuilder(
            n_qubits=4, n_layers=3,
            parameter_sharing="layer_wise",
            rotation_gates=["rx", "ry", "rz"]
        )
        circuit = builder.build()

        # Layer-wise: 3 layers * 3 gate types = 9 parameters
        assert circuit["total_parameters"] == 9

    def test_parameter_sharing_none(self):
        """Test no parameter sharing."""
        builder = HEACircuitBuilder(
            n_qubits=4, n_layers=3,
            parameter_sharing="none",
            rotation_gates=["rx", "ry", "rz"]
        )
        circuit = builder.build()

        # None: 3 layers * 4 qubits * 3 gate types = 36 parameters
        assert circuit["total_parameters"] == 36

    def test_invalid_entanglement_pattern(self):
        """Test invalid entanglement pattern raises error."""
        with pytest.raises(ValueError):
            HEACircuitBuilder(n_qubits=4, n_layers=2, entanglement_pattern="invalid")

    def test_invalid_rotation_gate(self):
        """Test invalid rotation gate raises error."""
        with pytest.raises(ValueError):
            HEACircuitBuilder(n_qubits=4, n_layers=2, rotation_gates=["rx", "invalid"])

    def test_set_parameters(self):
        """Test setting circuit parameters."""
        builder = HEACircuitBuilder(n_qubits=4, n_layers=2, parameter_sharing="layer_wise")
        builder.build()

        new_params = np.zeros(6)  # 2 layers * 3 gates
        builder.set_parameters(new_params)

        params = builder.get_parameters()
        np.testing.assert_array_equal(params, new_params)

    def test_create_hea_circuit_function(self):
        """Test convenience function."""
        circuit = create_hea_circuit(n_qubits=4, n_layers=2)
        assert circuit["n_qubits"] == 4
        assert circuit["n_layers"] == 2


class TestHEAConfig:
    """Tests for HEAConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = HEAConfig()
        assert config.n_qubits == 4
        assert config.max_layers == 4
        assert config.agent_type == "ppo"

    def test_custom_config(self):
        """Test custom configuration."""
        config = HEAConfig(n_qubits=6, max_layers=5, agent_type="dqn")
        assert config.n_qubits == 6
        assert config.max_layers == 5
        assert config.agent_type == "dqn"

    def test_invalid_entanglement_pattern(self):
        """Test invalid entanglement pattern raises error."""
        with pytest.raises(ValueError):
            HEAConfig(entanglement_patterns=["invalid"])

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = HEAConfig(n_qubits=6)
        config_dict = config.to_dict()
        assert config_dict["n_qubits"] == 6
        assert "max_layers" in config_dict

    def test_from_dict(self):
        """Test creating from dictionary."""
        config_dict = {"n_qubits": 8, "max_layers": 6}
        config = HEAConfig.from_dict(config_dict)
        assert config.n_qubits == 8
        assert config.max_layers == 6

    def test_update(self):
        """Test updating configuration."""
        config = HEAConfig()
        config.update({"n_qubits": 10})
        assert config.n_qubits == 10

    def test_get_agent_config(self):
        """Test getting agent configuration."""
        config = HEAConfig(agent_type="ppo", agent_config={"learning_rate": 0.01})
        agent_config = config.get_agent_config()
        assert "learning_rate" in agent_config
        assert agent_config["learning_rate"] == 0.01

    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config(n_qubits=6, max_layers=5)
        assert config.n_qubits == 6
        assert config.max_layers == 5


class TestHEASearchController:
    """Tests for HEASearchController class."""

    def test_initialization(self):
        """Test controller initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(n_qubits=4, output_dir=tmpdir)
            assert controller.n_qubits == 4
            assert controller.output_dir == tmpdir
            assert controller._best_energy == float("inf")

    def test_setup_environment(self):
        """Test setting up environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(n_qubits=4, output_dir=tmpdir)
            env = controller.setup_environment()
            assert env is not None
            assert env.n_qubits == 4

    def test_setup_agent(self):
        """Test setting up agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(n_qubits=4, output_dir=tmpdir)
            controller.setup_environment()
            agent = controller.setup_agent(agent_type="dqn", config={"verbose": 0, "buffer_size": 100})
            assert agent is not None

    def test_search(self):
        """Test running HEA search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(
                n_qubits=4,
                max_layers=2,
                output_dir=tmpdir,
                verbose=0,
            )
            results = controller.search(
                agent_type="dqn",
                agent_config={"verbose": 0, "buffer_size": 100},
                total_timesteps=100,
            )
            assert "best_energy" in results
            assert "training_history" in results
            assert results["n_qubits"] == 4

    def test_get_best_circuit(self):
        """Test getting best circuit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(n_qubits=4, output_dir=tmpdir, verbose=0)
            controller.setup_environment()

            # Before search, should be None
            assert controller.get_best_circuit() is None

    def test_run_episode(self):
        """Test running a single episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(n_qubits=4, max_layers=2, output_dir=tmpdir, verbose=0)
            controller.setup_environment()
            controller.setup_agent(agent_type="dqn", config={"verbose": 0, "buffer_size": 100})

            episode_result = controller.run_episode(episode_idx=0)
            assert "episode" in episode_result
            assert "reward" in episode_result
            assert "final_energy" in episode_result

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = HEASearchController(n_qubits=4, output_dir=tmpdir, verbose=0)
            controller.setup_environment()

            checkpoint_path = controller.save_checkpoint()
            assert os.path.exists(checkpoint_path)

    def test_run_hea_search_function(self):
        """Test convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_hea_search(
                n_qubits=4,
                max_layers=2,
                agent_type="dqn",
                total_timesteps=50,
                output_dir=tmpdir,
                verbose=0,
            )
            assert "best_energy" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
