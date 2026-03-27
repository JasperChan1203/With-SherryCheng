#!/usr/bin/env python3
"""Unit tests for UCCSearchEnv."""

import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
import gym

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.molecule_processor import MoleculeData
from src.modules.ucc_search.environment import UCCSearchEnv


class TestUCCSearchEnv(unittest.TestCase):
    """Test UCCSearchEnv class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock MoleculeData object
        self.molecule_data = Mock(spec=MoleculeData)
        self.molecule_data.n_qubits = 2
        self.molecule_data.fci_energy = -1.1372838344885023
        self.molecule_data.hamiltonian = Mock()  # placeholder
        self.molecule_data.reference_state = np.array([1, 0, 0, 0], dtype=complex)
        self.molecule_data.molecular_info = {
            "hf_energy": -1.1167593073964255,
            "formula": "H2",
            "bond_length_angstrom": 0.74,
            "basis_set": "sto-3g",
            "transform": "parity",
            "ansatz_type": "UCC"
        }
        # Mock the circuit builder to avoid actual quantum simulation
        self.mock_builder = Mock()
        self.mock_builder.n_params = 2
        self.mock_builder.get_available_excitations.return_value = [
            (3, 2), (1, 0), (1, 3, 2, 0)
        ]
        self.mock_builder.get_parameter_indices_for_excitation.return_value = [0]
        self.mock_builder.initialize_parameters.return_value = np.array([0.1, 0.2])
        self.mock_builder.build_circuit.return_value = Mock()
        self.mock_builder.evaluate_energy.return_value = -1.1
        self.mock_builder.ucc = Mock()
        self.mock_builder.ucc.hamiltonian = Mock()
        # Mock reward function
        self.mock_reward = Mock()
        self.mock_reward.compute_reward.return_value = 0.0
        self.mock_reward.update_baseline = Mock()
        # Mock simulator
        self.mock_simulator = Mock()
        self.mock_simulator.compute_energy.return_value = -1.1

    def test_environment_initialization(self):
        """Test environment initialization with molecule data."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            self.assertIsInstance(env, gym.Env)
            self.assertEqual(env.n_actions, 3)
            self.assertEqual(env.n_params, 2)
            self.assertIsNotNone(env.action_space)
            self.assertIsNotNone(env.observation_space)
            # Check that reset initializes state
            obs = env.reset()
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(env.step_count, 0)
            self.assertEqual(len(env.current_excitations), 0)
            self.assertIsNotNone(env.current_energy)

    def test_reset(self):
        """Test environment reset."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            obs = env.reset()
            # Should be within observation space bounds
            self.assertTrue(np.all(obs >= env.observation_space.low))
            self.assertTrue(np.all(obs <= env.observation_space.high))
            self.assertEqual(env.current_energy, env._get_hf_energy())
            self.assertEqual(env.best_energy, env._get_hf_energy())
            self.assertEqual(env.step_count, 0)
            self.assertFalse(env.done)

    def test_step_valid_action(self):
        """Test step with valid action."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            env.reset()
            # Mock the energy evaluation to return a specific value
            self.mock_builder.evaluate_energy.return_value = -1.12
            obs, reward, done, info = env.step(0)
            self.assertFalse(done)
            self.assertIn('energy', info)
            self.assertIn('best_energy', info)
            self.assertIn('excitations', info)
            self.assertEqual(len(env.current_excitations), 1)
            self.assertEqual(env.step_count, 1)

    def test_step_duplicate_excitation(self):
        """Test step with duplicate excitation (should terminate)."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            env.reset()
            # First action valid
            env.step(0)
            # Second same action (duplicate)
            obs, reward, done, info = env.step(0)
            self.assertTrue(done)
            self.assertEqual(reward, -10.0)
            self.assertEqual(info['termination_reason'], 'invalid_action')

    def test_step_max_depth(self):
        """Test step exceeding max depth."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            # Configure max_depth = 1
            env = UCCSearchEnv(self.molecule_data, config={"environment": {"max_depth": 1}})
            env.reset()
            obs, reward, done, info = env.step(0)
            self.assertTrue(done)
            self.assertEqual(reward, 0.0)
            self.assertEqual(info['termination_reason'], 'max_depth_reached')

    def test_observation_space(self):
        """Test observation space dimensions."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            obs_space = env.observation_space
            self.assertIsInstance(obs_space, gym.spaces.Box)
            # Default config: max_depth=10, max_excitations=20
            # Observation components: energy(1) + params(max_depth) + arch(max_excitations) + step(1)
            expected_dim = 1 + 10 + 20 + 1  # 32
            self.assertEqual(obs_space.shape, (expected_dim,))
            # Check bounds
            self.assertTrue(np.all(obs_space.low[:1] >= -10.0))
            self.assertTrue(np.all(obs_space.high[:1] <= 10.0))
            # Parameters bounded between -pi and pi
            self.assertTrue(np.all(obs_space.low[1:11] == -np.pi))
            self.assertTrue(np.all(obs_space.high[1:11] == np.pi))
            # Architecture encoding bounded between 0 and 1
            self.assertTrue(np.all(obs_space.low[11:31] == 0.0))
            self.assertTrue(np.all(obs_space.high[11:31] == 1.0))
            # Step normalized between 0 and 1
            self.assertEqual(obs_space.low[-1], 0.0)
            self.assertEqual(obs_space.high[-1], 1.0)

    def test_action_space(self):
        """Test action space."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            action_space = env.action_space
            self.assertIsInstance(action_space, gym.spaces.Discrete)
            self.assertEqual(action_space.n, 3)

    def test_render(self):
        """Test render method (should not crash)."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            env.reset()
            # Should not raise exception
            env.render(mode='human')
            env.render(mode='ansi')

    def test_close(self):
        """Test close method."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            env.close()  # Should not raise exception

    def test_termination_conditions(self):
        """Test termination condition checking."""
        with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
                   return_value=self.mock_builder), \
             patch('src.modules.ucc_search.environment.UCCRewardFunction',
                   return_value=self.mock_reward), \
             patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
                   return_value=self.mock_simulator):
            env = UCCSearchEnv(self.molecule_data)
            env.reset()
            # Initially not terminated
            self.assertFalse(env._check_termination())
            # Simulate max depth reached
            env.current_excitations = [(3, 2)] * 10  # length = max_depth default
            self.assertTrue(env._check_termination())
            # Reset
            env.current_excitations = []
            # Simulate convergence (energy close to FCI)
            env.current_energy = env.molecule_data.fci_energy + 1e-4
            self.assertTrue(env._check_termination())


if __name__ == '__main__':
    unittest.main()