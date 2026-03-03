#!/usr/bin/env python3
"""End-to-end test with H2 molecule for UCC search module."""

import sys
import unittest
import numpy as np
import tempfile
import os

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search import UCCSearchEnv, UCCSearchController, UCCSearchConfig


class TestH2EndToEnd(unittest.TestCase):
    """End-to-end test with real H2 molecule data."""

    def setUp(self):
        """Set up H2 molecule data."""
        # Process H2 molecule using Task 001
        self.molecule_data = process_molecule("H2", 0.74, "UCC")
        # Ensure we have necessary attributes
        self.assertIsNotNone(self.molecule_data)
        self.assertIsNotNone(self.molecule_data.hamiltonian)
        self.assertIsNotNone(self.molecule_data.fci_energy)
        self.assertIsNotNone(self.molecule_data.molecular_info)
        print(f"H2 molecule processed: HF energy = {self.molecule_data.molecular_info['hf_energy']}")
        print(f"FCI energy = {self.molecule_data.fci_energy}")

    def test_environment_with_h2(self):
        """Test UCCSearchEnv with real H2 molecule."""
        # Create environment
        env = UCCSearchEnv(self.molecule_data, config={"environment": {"max_depth": 3}})

        # Check environment properties
        self.assertGreater(env.n_actions, 0)
        self.assertEqual(env.action_space.n, env.n_actions)
        self.assertIsNotNone(env.observation_space)

        # Reset environment
        obs = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIsNotNone(env.current_energy)
        print(f"Initial energy (Hartree-Fock): {env.current_energy}")

        # Take a few random steps (valid actions)
        for i in range(3):
            # Choose random action (within valid range)
            action = np.random.randint(0, env.n_actions)
            obs, reward, done, info = env.step(action)

            # Check step results
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            self.assertIn('energy', info)
            self.assertIn('excitations', info)
            self.assertIn('params', info)
            print(f"Step {i}: action={action}, energy={info['energy']}, reward={reward}")

            # If done (max depth reached), break
            if done:
                break

        # Close environment
        env.close()

    def test_controller_integration_with_h2(self):
        """Test UCCSearchController integration with H2 molecule (short run)."""
        # Use a very small configuration for quick test
        config = {
            "environment": {
                "max_depth": 2,
                "max_excitations": 5
            },
            "controller": {
                "n_episodes": 2,
                "log_frequency": 1,
                "train_frequency": 0,  # No training for quick test
                "checkpoint_frequency": 0,
                "agent_type": "ppo",
                "use_gpu": False,
                "seed": 42
            }
        }

        # Create controller
        controller = UCCSearchController(self.molecule_data, agent_type='ppo', config=config)

        # Run short search (2 episodes)
        results = controller.search(n_episodes=2)

        # Verify results structure
        required_keys = ['best_energy', 'best_circuit', 'best_excitations', 'best_params',
                        'training_history', 'convergence_reached', 'episode_rewards',
                        'episode_energies', 'episode_depths']
        for key in required_keys:
            self.assertIn(key, results)

        # Verify training history length matches number of episodes
        self.assertEqual(len(results['training_history']), 2)
        self.assertEqual(len(results['episode_rewards']), 2)
        self.assertEqual(len(results['episode_energies']), 2)
        self.assertEqual(len(results['episode_depths']), 2)

        # Verify that energies are valid (not None)
        for energy in results['episode_energies']:
            self.assertIsInstance(energy, float)
            self.assertLess(energy, 0)  # Energy should be negative for H2

        print(f"Controller test completed. Best energy: {results['best_energy']}")

        # Test saving and loading results
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            controller.save_results(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            # Load results into new controller
            controller2 = UCCSearchController(self.molecule_data, agent_type='ppo', config=config)
            controller2.load_results(temp_path)

            # Verify loaded results match saved
            self.assertEqual(controller2.results['best_energy'], results['best_energy'])
            self.assertEqual(len(controller2.results['training_history']), 2)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            agent_path = temp_path.replace('.json', '_agent.pkl')
            if os.path.exists(agent_path):
                os.unlink(agent_path)

    def test_energy_improvement(self):
        """Test that UCC search can improve energy from Hartree-Fock."""
        # This test runs a longer search to verify energy improvement
        # We'll limit to a small number of episodes for speed
        config = {
            "environment": {
                "max_depth": 3,
                "max_excitations": 10,
                "param_init_strategy": "zeros"  # Start with zero parameters for consistency
            },
            "controller": {
                "n_episodes": 5,
                "log_frequency": 5,
                "train_frequency": 0,  # No training for deterministic test
                "checkpoint_frequency": 0,
                "agent_type": "ppo",
                "use_gpu": False,
                "seed": 12345
            }
        }

        controller = UCCSearchController(self.molecule_data, agent_type='ppo', config=config)
        results = controller.search(n_episodes=5)

        # Check that we have at least some energy values
        self.assertGreater(len(results['episode_energies']), 0)

        # Get initial Hartree-Fock energy
        hf_energy = self.molecule_data.molecular_info['hf_energy']
        print(f"Hartree-Fock energy: {hf_energy}")
        print(f"Best energy found: {results['best_energy']}")

        # Verify that best energy is not worse than HF (allowing small numerical tolerance)
        if results['best_energy'] is not None:
            # Energy should be <= HF energy (lower is better)
            self.assertLessEqual(results['best_energy'], hf_energy + 1e-6)

        # Check that at least one episode completed without error
        self.assertTrue(all(isinstance(e, float) for e in results['episode_energies']))

    def test_configuration_defaults(self):
        """Test that default configuration matches RLQAS spec."""
        config = UCCSearchConfig()

        # Check environment defaults
        env_config = config.get_section("environment")
        self.assertEqual(env_config.get("max_depth"), 10)
        self.assertEqual(env_config.get("max_excitations"), 20)
        self.assertTrue(env_config.get("use_sqeb", False))

        # Check reward function defaults
        reward_config = config.get_section("reward_function")
        self.assertEqual(reward_config.get("energy_weight"), 1.0)
        self.assertEqual(reward_config.get("complexity_penalty"), 0.01)
        self.assertEqual(reward_config.get("baseline_type"), "hartree_fock")

        # Check controller defaults
        controller_config = config.get_section("controller")
        self.assertEqual(controller_config.get("agent_type"), "ppo")
        self.assertEqual(controller_config.get("n_episodes"), 1000)
        self.assertEqual(controller_config.get("early_stop_threshold"), 1.6e-3)


if __name__ == '__main__':
    unittest.main()