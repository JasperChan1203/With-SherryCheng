#!/usr/bin/env python3
"""Integration tests for UCC search module with H2 molecule."""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.molecule_processor import MoleculeData, process_molecule
from src.modules.quantum_simulator import SimulatorFactory
from src.modules.rl_agents import PPOAgent
from src.modules.ucc_search import (
    UCCSearchEnv, UCCCircuitBuilder, UCCRewardFunction,
    UCCSearchController, UCCSearchConfig
)


class TestUCCSearchIntegration(unittest.TestCase):
    """Integration tests for UCC search module."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock molecule data for H2
        self.molecule_data = Mock(spec=MoleculeData)
        self.molecule_data.n_qubits = 2
        self.molecule_data.fci_energy = -1.1372838344885023
        self.molecule_data.hamiltonian = Mock()
        self.molecule_data.reference_state = np.array([1, 0, 0, 0], dtype=complex)
        self.molecule_data.molecular_info = {
            "hf_energy": -1.1167593073964255,
            "formula": "H2",
            "bond_length_angstrom": 0.74,
            "basis_set": "sto-3g",
            "transform": "parity",
            "ansatz_type": "UCC"
        }

    @patch('src.modules.ucc_search.environment.UCCCircuitBuilder')
    @patch('src.modules.ucc_search.environment.UCCRewardFunction')
    @patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator')
    def test_environment_integration(self, mock_simulator_factory, mock_reward_class, mock_builder_class):
        """Test environment integration with mocked dependencies."""
        # Mock dependencies
        mock_builder = Mock()
        mock_builder.n_params = 2
        mock_builder.get_available_excitations.return_value = [(3, 2), (1, 0)]
        mock_builder.get_parameter_indices_for_excitation.return_value = [0]
        mock_builder.initialize_parameters.return_value = np.array([0.1, 0.2])
        mock_builder.build_circuit.return_value = Mock()
        mock_builder.evaluate_energy.return_value = -1.12
        mock_builder_class.return_value = mock_builder

        mock_reward = Mock()
        mock_reward.compute_reward.return_value = 0.05
        mock_reward.update_baseline = Mock()
        mock_reward_class.return_value = mock_reward

        mock_simulator = Mock()
        mock_simulator.compute_energy.return_value = -1.12
        mock_simulator_factory.return_value = mock_simulator

        # Create environment
        env = UCCSearchEnv(self.molecule_data)

        # Test basic functionality
        self.assertIsNotNone(env)
        self.assertEqual(env.n_actions, 2)

        # Test reset
        obs = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)

        # Test step
        obs, reward, done, info = env.step(0)
        self.assertFalse(done)
        self.assertEqual(reward, 0.05)
        self.assertIn('energy', info)

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_circuit_builder_integration(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test circuit builder integration with mocked tencirchem."""
        # Mock tencirchem.UCCSD
        mock_ucc = Mock()
        mock_ucc.ex_ops = [(3, 2), (1, 0)]
        mock_ucc.param_ids = [0, 1]
        mock_ucc.param_to_ex_ops = {0: [(3, 2)], 1: [(1, 0)]}
        mock_ucc.n_params = 2
        mock_ucc.get_circuit.return_value = Mock()
        mock_ucc.energy.return_value = -1.12
        mock_ucc_class.return_value = mock_ucc

        # Mock pyscf
        mock_mol = Mock()
        mock_hf = Mock()
        mock_hf.kernel.return_value = -1.116759
        mock_hf.converged = True
        mock_gto.return_value = mock_mol
        mock_rhf.return_value = mock_hf

        # Create circuit builder
        builder = UCCCircuitBuilder(self.molecule_data)

        # Test available excitations
        excitations = builder.get_available_excitations()
        self.assertEqual(excitations, [(3, 2), (1, 0)])

        # Test building circuit
        circuit = builder.build_circuit([(3, 2)])
        self.assertIsNotNone(circuit)

        # Test energy evaluation
        params = np.array([0.1, 0.2])
        energy = builder.evaluate_energy(circuit, params)
        self.assertEqual(energy, -1.12)

    def test_reward_function_integration(self):
        """Test reward function with different configurations."""
        # Test default configuration
        rf = UCCRewardFunction()
        self.assertEqual(rf.baseline_type, "hartree_fock")
        self.assertEqual(rf.energy_weight, 1.0)

        # Test with custom config
        config = {
            "reward_function": {
                "baseline_type": "current_best",
                "energy_weight": 2.0,
                "complexity_penalty": 0.05
            }
        }
        rf2 = UCCRewardFunction(config)
        self.assertEqual(rf2.baseline_type, "current_best")
        self.assertEqual(rf2.energy_weight, 2.0)
        self.assertEqual(rf2.complexity_penalty, 0.05)

        # Test reward computation
        rf2.compute_reward(-1.0, 1)  # First evaluation
        reward = rf2.compute_reward(-1.1, 2)  # Improvement
        self.assertGreater(reward, 0.0)

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_controller_integration(self, mock_agent_class, mock_env_class):
        """Test controller integration with mocked components."""
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = np.array([0.0, 0.1, 0.2])
        mock_env.step.return_value = (
            np.array([0.0, 0.1, 0.2]), 0.05, False,
            {"energy": -1.12, "excitations": [(3, 2)], "params": [0.1]}
        )
        mock_env.current_energy = -1.116759
        mock_env.global_best_energy = -1.12
        mock_env.global_best_excitations = [(3, 2)]
        mock_env.global_best_params = np.array([0.1])
        mock_env_class.return_value = mock_env

        # Mock agent
        mock_agent = Mock()
        mock_agent.select_action.return_value = 0
        mock_agent.store_experience = Mock()
        mock_agent.train = Mock()
        mock_agent.save = Mock()
        mock_agent_class.return_value = mock_agent

        # Create controller
        controller = UCCSearchController(self.molecule_data, agent_type='ppo')

        # Run short search
        controller.config = {"n_episodes": 1, "log_frequency": 1000, "train_frequency": 1000}
        mock_env.step.side_effect = [
            (np.array([0.0, 0.1, 0.2]), 0.05, True,
             {"energy": -1.12, "excitations": [(3, 2)], "params": [0.1]})
        ]

        results = controller.search(n_episodes=1)

        # Verify results
        self.assertIn('best_energy', results)
        self.assertIn('training_history', results)
        self.assertEqual(len(results['training_history']), 1)

    def test_config_management(self):
        """Test configuration management integration."""
        # Test default config
        config = UCCSearchConfig()
        env_config = config.get_section("environment")
        self.assertEqual(env_config.get("max_depth"), 10)
        self.assertEqual(env_config.get("max_excitations"), 20)

        # Test custom config
        custom_config = {
            "environment": {
                "max_depth": 5,
                "max_excitations": 10
            }
        }
        config2 = UCCSearchConfig(custom_config)
        env_config2 = config2.get_section("environment")
        self.assertEqual(env_config2.get("max_depth"), 5)
        self.assertEqual(env_config2.get("max_excitations"), 10)

        # Test validation
        self.assertTrue(config2.validate())

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    @patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_end_to_end_mocked(self, mock_agent_class, mock_simulator_factory,
                               mock_ucc_class, mock_rhf, mock_gto):
        """Test end-to-end pipeline with all dependencies mocked."""
        # Mock tencirchem
        mock_ucc = Mock()
        mock_ucc.ex_ops = [(3, 2), (1, 0)]
        mock_ucc.param_ids = [0, 1]
        mock_ucc.param_to_ex_ops = {0: [(3, 2)], 1: [(1, 0)]}
        mock_ucc.n_params = 2
        mock_ucc.get_circuit.return_value = Mock()
        mock_ucc.energy.return_value = -1.12
        mock_ucc_class.return_value = mock_ucc

        # Mock pyscf
        mock_mol = Mock()
        mock_hf = Mock()
        mock_hf.kernel.return_value = -1.116759
        mock_hf.converged = True
        mock_gto.return_value = mock_mol
        mock_rhf.return_value = mock_hf

        # Mock simulator
        mock_simulator = Mock()
        mock_simulator.compute_energy.return_value = -1.12
        mock_simulator_factory.return_value = mock_simulator

        # Mock agent
        mock_agent = Mock()
        mock_agent.select_action.return_value = 0
        mock_agent.store_experience = Mock()
        mock_agent.train = Mock()
        mock_agent.save = Mock()
        mock_agent_class.return_value = mock_agent

        # Create controller
        controller = UCCSearchController(self.molecule_data, agent_type='ppo')
        controller.config = {"n_episodes": 1, "log_frequency": 1000, "train_frequency": 1000}

        # Run search
        results = controller.search(n_episodes=1)

        # Verify basic results
        self.assertIn('best_energy', results)
        self.assertIn('training_history', results)
        self.assertIsInstance(results['episode_rewards'], list)
        self.assertIsInstance(results['episode_energies'], list)


if __name__ == '__main__':
    unittest.main()