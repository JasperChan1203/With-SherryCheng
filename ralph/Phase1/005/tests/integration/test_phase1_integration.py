"""
Phase 1 integration tests.

Tests the complete Phase 1 pipeline integration.
"""

import sys
import os
import pytest
import tempfile
import json
import numpy as np

# Add task directories to Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(test_dir)))  # Phase1 directory
TASK_DIRS = [os.path.join(project_root, f"{i:03d}") for i in range(1, 5)]
for dir_path in TASK_DIRS:
    if os.path.exists(dir_path):
        sys.path.append(dir_path)

# Import Phase 1 modules
from src.modules.molecule_processor import process_molecule
from src.modules.quantum_simulator import SimulatorFactory
from src.modules.rl_agents import PPOAgent
from src.modules.ucc_search.controller import UCCSearchController
from src.modules.ucc_search.environment import UCCSearchEnv
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder
from src.modules.ucc_search.reward_function import UCCRewardFunction


class TestPhase1Integration:
    """Integration tests for complete Phase 1 pipeline."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='phase1_test_')
        print(f"Test output directory: {self.temp_dir}")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_data_flow_between_modules(self):
        """Test data flow between all Phase 1 modules."""
        # 1. Process molecule (Task 001)
        print("1. Processing molecule...")
        molecule_data = process_molecule(
            molecule='LiH',
            bond_length=1.6,
            ansatz_type='UCC',
            active_space=(2, 2),
            basis_set='sto-3g',
            transform='parity'
        )
        assert molecule_data.n_qubits == 2
        fci_energy = molecule_data.fci_energy
        assert fci_energy is not None

        # 2. Create simulator (Task 002)
        print("2. Creating simulator...")
        simulator = SimulatorFactory.create_simulator(
            molecule_data.n_qubits,
            config={'max_memory_gb': 8}
        )
        assert simulator is not None

        # 3. Create UCC circuit builder (Task 004 component)
        print("3. Creating UCC circuit builder...")
        circuit_builder = UCCCircuitBuilder(molecule_data)
        assert circuit_builder is not None
        assert hasattr(circuit_builder, 'build_circuit')

        # 4. Create reward function (Task 004 component)
        print("4. Creating reward function...")
        reward_function = UCCRewardFunction(config={'max_depth': 5, 'max_excitations': 8})
        assert reward_function is not None
        assert hasattr(reward_function, 'compute_reward')

        # 5. Create UCC search environment (Task 004)
        print("5. Creating UCC search environment...")
        env = UCCSearchEnv(molecule_data, config={'max_depth': 5, 'max_excitations': 8})
        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'action_space')

        # 6. Create PPO agent (Task 003)
        print("6. Creating PPO agent...")
        agent = PPOAgent(config={
            'seed': 42,
            'use_gpu': False,
            'policy_type': 'MlpPolicy',
            'learning_rate': 3e-4,
            'verbose': 0
        })
        assert agent is not None

        # 7. Create UCC search controller (Task 004)
        print("7. Creating UCC search controller...")
        controller = UCCSearchController(
            molecule_data,
            agent_type='ppo',
            config={
                'max_depth': 5,
                'max_excitations': 8,
                'n_episodes': 3,
                'early_stop_threshold': 0.1,
                'checkpoint_frequency': 0,
                'log_frequency': 1,
                'train_frequency': 1
            }
        )
        assert controller is not None

        print("✓ All Phase 1 modules created successfully with correct data flow")

    def test_module_compatibility(self):
        """Test that modules are compatible with each other."""
        # Process molecule
        molecule_data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,2))

        # Check that simulator can be created with molecule's qubit count
        simulator = SimulatorFactory.create_simulator(molecule_data.n_qubits)
        assert simulator.get_max_qubits() >= molecule_data.n_qubits

        # Check that environment can be created with molecule data
        env = UCCSearchEnv(molecule_data, config={'max_depth': 5})
        assert env.molecule_data == molecule_data

        # Check that controller can be created with molecule data and environment
        controller = UCCSearchController(molecule_data, agent_type='ppo')
        assert controller.molecule_data == molecule_data
        assert controller.env is not None

        print("✓ All modules are compatible with each other")

    def test_reproducibility_with_random_seeds(self):
        """Test that random seeds ensure reproducible results."""
        import random
        import torch

        # Set seed and process molecule
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        molecule_data1 = process_molecule('LiH', 1.6, 'UCC', active_space=(2,2))
        fci1 = molecule_data1.fci_energy

        # Reset and set same seed
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        molecule_data2 = process_molecule('LiH', 1.6, 'UCC', active_space=(2,2))
        fci2 = molecule_data2.fci_energy

        # FCI energy should be identical (deterministic calculation)
        assert fci1 == fci2, f"FCI energies differ: {fci1} vs {fci2}"

        print(f"✓ Reproducibility confirmed: FCI energy = {fci1:.6f} Hartree")

    def test_error_handling_across_modules(self):
        """Test error handling across module boundaries."""
        # Test with invalid molecule
        with pytest.raises(ValueError):
            process_molecule('InvalidMolecule', 1.0, 'UCC')

        # Test with invalid bond length
        with pytest.raises(ValueError):
            process_molecule('LiH', -1.0, 'UCC')

        # Test with invalid active space
        with pytest.raises(Exception):  # May raise various exceptions
            process_molecule('LiH', 1.6, 'UCC', active_space=(100, 100))

        print("✓ Error handling works across modules")

    def test_configuration_consistency(self):
        """Test configuration consistency across all modules."""
        # Test that all modules accept configuration dictionaries
        molecule_data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,2))

        # Simulator config
        simulator_config = {'max_memory_gb': 16, 'engine': 'statevector'}
        simulator = SimulatorFactory.create_simulator(2, simulator_config)
        assert simulator is not None

        # Agent config
        agent_config = {'seed': 42, 'learning_rate': 1e-3, 'verbose': 0}
        agent = PPOAgent(config=agent_config)
        assert agent is not None

        # Controller config
        controller_config = {
            'max_depth': 10,
            'max_excitations': 12,
            'n_episodes': 100,
            'early_stop_threshold': 1e-3,
            'seed': 42
        }
        controller = UCCSearchController(molecule_data, config=controller_config)
        assert controller is not None

        print("✓ Configuration consistency verified")

    @pytest.mark.slow
    def test_minimal_end_to_end_run(self):
        """Test minimal end-to-end run of complete pipeline."""
        molecule_data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,2))

        # Create controller with minimal settings for quick test
        controller = UCCSearchController(
            molecule_data,
            agent_type='ppo',
            config={
                'max_depth': 3,
                'max_excitations': 4,
                'n_episodes': 2,  # Very small
                'early_stop_threshold': 1.0,  # Very loose
                'checkpoint_frequency': 0,
                'log_frequency': 1,
                'train_frequency': 1,
                'seed': 42
            }
        )

        # Run search
        results = controller.search(n_episodes=2, early_stop_threshold=1.0)

        # Verify results structure
        assert 'best_energy' in results
        assert 'episode_rewards' in results
        assert 'episode_energies' in results
        assert 'training_history' in results

        # Should have completed exactly 2 episodes
        assert len(results['episode_rewards']) == 2
        assert len(results['episode_energies']) == 2

        print(f"✓ Minimal end-to-end run completed: {len(results['episode_energies'])} episodes")


if __name__ == "__main__":
    # Run tests manually if needed
    test = TestPhase1Integration()
    test.setup_method()
    try:
        test.test_data_flow_between_modules()
        print("✓ Data flow test passed")
        test.test_module_compatibility()
        print("✓ Module compatibility test passed")
        test.test_reproducibility_with_random_seeds()
        print("✓ Reproducibility test passed")
        test.test_error_handling_across_modules()
        print("✓ Error handling test passed")
        test.test_configuration_consistency()
        print("✓ Configuration consistency test passed")
    finally:
        test.teardown_method()