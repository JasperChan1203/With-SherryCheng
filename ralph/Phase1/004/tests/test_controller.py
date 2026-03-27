#!/usr/bin/env python3
"""Unit tests for UCCSearchController."""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import tempfile
import os

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.molecule_processor import MoleculeData
from src.modules.ucc_search.controller import UCCSearchController, UCCPPOAgent


class TestUCCSearchController(unittest.TestCase):
    """Test UCCSearchController class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock MoleculeData
        self.molecule_data = Mock(spec=MoleculeData)
        self.molecule_data.n_qubits = 2
        self.molecule_data.fci_energy = -1.1372838344885023
        self.molecule_data.molecular_info = {
            "hf_energy": -1.1167593073964255,
            "formula": "H2",
            "bond_length_angstrom": 0.74,
            "basis_set": "sto-3g",
            "transform": "parity",
            "ansatz_type": "UCC"
        }

        # Mock environment
        self.mock_env = Mock()
        self.mock_env.reset.return_value = np.array([0.0, 0.1, 0.2])
        self.mock_env.step.return_value = (
            np.array([0.0, 0.1, 0.2]),  # next obs
            0.05,  # reward
            False,  # done
            {"energy": -1.12, "excitations": [(3, 2)], "params": [0.1]}
        )
        self.mock_env.current_energy = -1.116759
        self.mock_env.global_best_energy = -1.12
        self.mock_env.global_best_excitations = [(3, 2)]
        self.mock_env.global_best_params = np.array([0.1])

        # Mock agent
        self.mock_agent = Mock()
        self.mock_agent.select_action.return_value = 0
        self.mock_agent.store_experience = Mock()
        self.mock_agent.train = Mock()
        self.mock_agent.save = Mock()

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_initialization(self, mock_agent_class, mock_env_class):
        """Test controller initialization."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        controller = UCCSearchController(self.molecule_data, agent_type='ppo')

        # Check that environment was created with molecule_data and config
        mock_env_class.assert_called_once_with(self.molecule_data, None)
        # Check that agent was created with config and environment
        mock_agent_class.assert_called_once()
        # Check that controller attributes are set
        self.assertEqual(controller.molecule_data, self.molecule_data)
        self.assertEqual(controller.env, self.mock_env)
        self.assertEqual(controller.agent, self.mock_agent)
        self.assertIsNotNone(controller.results)
        self.assertFalse(controller.results['convergence_reached'])

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_search_single_episode(self, mock_agent_class, mock_env_class):
        """Test search with a single episode."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        controller = UCCSearchController(self.molecule_data)
        # Override config to force single episode, no training
        controller.config = {"n_episodes": 1, "log_frequency": 1000, "train_frequency": 0, "checkpoint_frequency": 0}

        # Mock environment step to terminate after one step
        self.mock_env.step.side_effect = [
            (np.array([0.0, 0.1, 0.2]), 0.05, True, {"energy": -1.12, "excitations": [(3, 2)], "params": [0.1]})
        ]

        results = controller.search(n_episodes=1)

        # Verify environment reset called
        self.mock_env.reset.assert_called_once()
        # Verify agent select_action called at least once
        self.mock_agent.select_action.assert_called()
        # Verify environment step called
        self.mock_env.step.assert_called_once()
        # Verify store_experience called (maybe)
        self.mock_agent.store_experience.assert_called()
        # Verify train not called (since train_frequency large)
        self.mock_agent.train.assert_not_called()
        # Verify results contain expected keys
        self.assertIn('best_energy', results)
        self.assertIn('training_history', results)
        self.assertEqual(len(results['training_history']), 1)
        # Episode energy should be recorded
        self.assertEqual(results['episode_energies'][0], -1.12)

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_search_early_stopping(self, mock_agent_class, mock_env_class):
        """Test early stopping due to convergence."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        controller = UCCSearchController(self.molecule_data)
        controller.config = {"n_episodes": 1000, "log_frequency": 1000, "train_frequency": 0, "checkpoint_frequency": 0}

        # Make convergence check return True after first episode
        original_check = controller._check_convergence
        controller._check_convergence = Mock(return_value=True)

        results = controller.search(n_episodes=1000)

        # Should have broken after first episode
        self.assertTrue(results['convergence_reached'])
        self.assertEqual(len(results['training_history']), 1)
        # Reset mock
        controller._check_convergence = original_check

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_check_convergence_with_fci(self, mock_agent_class, mock_env_class):
        """Test convergence check when FCI energy is available."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        controller = UCCSearchController(self.molecule_data)
        controller.best_overall_energy = -1.137  # close to FCI
        # Threshold 1.6e-3
        self.assertTrue(controller._check_convergence(1.6e-3))
        # Energy error ~0.0003 < 0.0016
        controller.best_overall_energy = -1.135  # error 0.0023 > 0.0016
        self.assertFalse(controller._check_convergence(1.6e-3))

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_check_convergence_without_fci(self, mock_agent_class, mock_env_class):
        """Test convergence check when FCI energy not available."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        # Molecule data without FCI energy
        molecule_data_no_fci = Mock(spec=MoleculeData)
        molecule_data_no_fci.fci_energy = None
        molecule_data_no_fci.molecular_info = {}

        controller = UCCSearchController(molecule_data_no_fci)
        controller.results['episode_energies'] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        # Low std deviation
        self.assertTrue(controller._check_convergence(0.01))
        # High std deviation
        controller.results['episode_energies'] = [-1.0, -0.9, -1.1, -0.8, -1.2, -0.7, -1.3, -0.6, -1.4, -0.5]
        self.assertFalse(controller._check_convergence(0.01))

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_save_and_load_results(self, mock_agent_class, mock_env_class):
        """Test saving and loading results."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        controller = UCCSearchController(self.molecule_data)
        # Populate some results
        controller.results = {
            'best_energy': -1.12,
            'best_excitations': [(3, 2)],
            'best_params': np.array([0.1]),
            'training_history': [{'episode': 0, 'reward': 0.05, 'energy': -1.12, 'depth': 1}],
            'convergence_reached': False,
            'episode_rewards': [0.05],
            'episode_energies': [-1.12],
            'episode_depths': [1],
        }
        controller.best_overall_energy = -1.12
        controller.best_overall_excitations = [(3, 2)]
        controller.best_overall_params = np.array([0.1])

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            controller.save_results(temp_path)
            # Verify agent.save called
            self.mock_agent.save.assert_called_once()
            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))
            # Load results into a new controller
            controller2 = UCCSearchController(self.molecule_data)
            controller2.load_results(temp_path)
            # Verify results match
            self.assertEqual(controller2.results['best_energy'], -1.12)
            self.assertEqual(controller2.results['best_excitations'], [(3, 2)])
            self.assertEqual(controller2.results['training_history'][0]['episode'], 0)
            self.assertIsInstance(controller2.results['best_params'], np.ndarray)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            agent_path = temp_path.replace('.json', '_agent.pkl')
            if os.path.exists(agent_path):
                os.unlink(agent_path)

    @patch('src.modules.ucc_search.controller.UCCSearchEnv')
    @patch('src.modules.ucc_search.controller.UCCPPOAgent')
    def test_global_best_tracking(self, mock_agent_class, mock_env_class):
        """Test that controller updates best overall from environment global best."""
        mock_env_class.return_value = self.mock_env
        mock_agent_class.return_value = self.mock_agent

        controller = UCCSearchController(self.molecule_data)
        controller.config = {"n_episodes": 1, "log_frequency": 1000, "train_frequency": 0, "checkpoint_frequency": 0}
        # Simulate environment global best better than current best_overall
        controller.best_overall_energy = -1.11  # worse than global_best_energy -1.12
        self.mock_env.global_best_energy = -1.12
        self.mock_env.global_best_excitations = [(3, 2)]
        self.mock_env.global_best_params = np.array([0.1])

        # Mock search to just run one episode and call the update logic
        # We'll directly call the update block from search method
        # Instead, we can run search with mocked env step that returns early
        self.mock_env.step.side_effect = [
            (np.array([0.0, 0.1, 0.2]), 0.05, True, {"energy": -1.12, "excitations": [(3, 2)], "params": [0.1]})
        ]
        results = controller.search(n_episodes=1)

        # Verify that best_overall_energy updated to global best
        self.assertEqual(controller.best_overall_energy, -1.12)
        self.assertEqual(results['best_energy'], -1.12)
        self.assertEqual(results['best_excitations'], [(3, 2)])


if __name__ == '__main__':
    unittest.main()