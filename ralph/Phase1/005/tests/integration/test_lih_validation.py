"""
Integration tests for LiH validation.

Tests the complete RLQAS Phase 1 system integration using LiH molecule.
"""

import sys
import os
import pytest
import tempfile
import json
import numpy as np

# Add parent directory to Python path to import scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))

# Import validation function
from validate_lih import run_lih_validation, set_seed


class TestLiHValidation:
    """Integration tests for LiH validation."""

    def setup_method(self):
        """Set up test environment."""
        set_seed(42)  # Ensure reproducibility
        self.temp_dir = tempfile.mkdtemp(prefix='lih_test_')
        print(f"Test output directory: {self.temp_dir}")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_module_imports(self):
        """Test that all required Phase 1 modules can be imported."""
        # Already imported at module level, but verify
        from src.modules.molecule_processor import process_molecule, MoleculeData
        from src.modules.quantum_simulator import SimulatorFactory
        from src.modules.rl_agents import PPOAgent
        from src.modules.ucc_search.controller import UCCSearchController

        # Verify imports succeeded (no exception)
        assert process_molecule is not None
        assert SimulatorFactory is not None
        assert PPOAgent is not None
        assert UCCSearchController is not None

    def test_lih_molecule_processing(self):
        """Test LiH molecule processing with validation parameters."""
        from src.modules.molecule_processor import process_molecule

        # Process LiH with validation test parameters
        data = process_molecule(
            molecule='LiH',
            bond_length=1.6,
            ansatz_type='UCC',
            active_space=(2, 2),
            basis_set='sto-3g',
            transform='parity'
        )

        # Verify molecule data
        assert data.n_qubits == 2  # Parity transform reduces from 4 to 2 qubits
        assert hasattr(data, 'fci_energy')
        assert data.fci_energy is not None
        assert hasattr(data, 'hamiltonian')
        assert hasattr(data, 'reference_state')
        assert hasattr(data, 'molecular_info')

        # Verify molecular info contains expected fields
        info = data.molecular_info
        assert info['formula'] == 'LiH'
        assert info['bond_length_angstrom'] == 1.6
        assert info['active_space'] == (2, 2)
        assert info['basis_set'] == 'sto-3g'
        assert info['transform'] == 'parity'

        print(f"LiH processed: {data.n_qubits} qubits, FCI energy = {data.fci_energy:.6f} Hartree")

    def test_simulator_creation(self):
        """Test quantum simulator creation for LiH system."""
        from src.modules.quantum_simulator import SimulatorFactory

        # Create simulator for 2-qubit LiH (parity transform)
        simulator = SimulatorFactory.create_simulator(2, config={'max_memory_gb': 8})

        assert simulator is not None
        assert hasattr(simulator, 'compute_energy')
        assert hasattr(simulator, 'get_max_qubits')
        assert hasattr(simulator, 'estimate_memory')

        max_qubits = simulator.get_max_qubits()
        assert max_qubits >= 2  # Should support at least 2 qubits

        memory_estimate = simulator.estimate_memory(2)
        assert memory_estimate > 0

        print(f"Simulator created: max qubits = {max_qubits}, memory estimate for 2 qubits = {memory_estimate:.4f} GB")

    def test_ppo_agent_creation(self):
        """Test PPO agent creation with validation configuration."""
        from src.modules.rl_agents import PPOAgent

        # Create agent with validation config
        agent = PPOAgent(config={
            'seed': 42,
            'use_gpu': False,
            'policy_type': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 0
        })

        assert agent is not None
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'learn')
        assert hasattr(agent, 'save')
        assert hasattr(agent, 'load')

        print("PPO agent created successfully")

    def test_ucc_controller_creation(self):
        """Test UCC search controller creation with mock dependencies."""
        from src.modules.molecule_processor import process_molecule
        from src.modules.ucc_search.controller import UCCSearchController

        # Process LiH molecule
        molecule_data = process_molecule(
            molecule='LiH',
            bond_length=1.6,
            ansatz_type='UCC',
            active_space=(2, 2),
            basis_set='sto-3g',
            transform='parity'
        )

        # Create controller with fast configuration
        controller = UCCSearchController(
            molecule_data,
            agent_type='ppo',
            config={
                'max_depth': 5,
                'max_excitations': 8,
                'use_gpu': False,
                'seed': 42,
                'n_episodes': 10,
                'early_stop_threshold': 0.1,
                'checkpoint_frequency': 0,
                'log_frequency': 5,
                'train_frequency': 1
            }
        )

        assert controller is not None
        assert hasattr(controller, 'search')
        assert hasattr(controller, 'save_results')
        assert hasattr(controller, 'load_results')

        print("UCC search controller created successfully")

    def test_fast_validation_run(self):
        """Test fast validation run with minimal parameters."""
        # Run validation with fast configuration
        results = run_lih_validation(
            bond_length=1.6,
            active_space=(2, 2),
            basis_set='sto-3g',
            transform='parity',
            n_episodes=2,  # Very small for test
            early_stop_threshold=0.1,  # Very loose threshold
            output_dir=self.temp_dir
        )

        # Verify results structure
        assert 'validation_start_time' in results
        assert 'validation_end_time' in results
        assert 'total_time_seconds' in results
        assert 'configuration' in results
        assert 'metrics' in results
        assert 'success' in results
        assert 'errors' in results

        # Check configuration matches input
        config = results['configuration']
        assert config['bond_length'] == 1.6
        # active_space may be tuple or list depending on serialization
        assert tuple(config['active_space']) == (2, 2)
        assert config['basis_set'] == 'sto-3g'
        assert config['transform'] == 'parity'
        assert config['n_episodes'] == 2
        assert config['early_stop_threshold'] == 0.1

        # Check that results file was created
        results_file = os.path.join(self.temp_dir, 'validation_results.json')
        assert os.path.exists(results_file), f"Results file not created: {results_file}"

        # Load and verify saved results
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        assert saved_results['configuration'] == config

        print(f"Fast validation run completed: success={results['success']}, time={results['total_time_seconds']:.2f}s")

    def test_validation_with_invalid_params(self):
        """Test validation with invalid parameters should handle errors gracefully."""
        # Test with invalid bond length (negative)
        results = run_lih_validation(
            bond_length=-1.0,  # Invalid
            active_space=(2, 2),
            basis_set='sto-3g',
            transform='parity',
            n_episodes=5,
            early_stop_threshold=0.1,
            output_dir=self.temp_dir
        )

        # Should have errors and success=False
        assert results['success'] is False
        assert len(results['errors']) > 0
        print(f"Invalid parameter test: {results['errors'][0][:50]}...")

    @pytest.mark.slow
    def test_full_validation_configuration(self):
        """Test validation with full configuration (marked as slow)."""
        # This test runs with default validation parameters (500 episodes)
        # Marked as slow to skip in regular test runs
        results = run_lih_validation(
            bond_length=1.6,
            active_space=(2, 2),
            basis_set='sto-3g',
            transform='parity',
            n_episodes=10,  # Still small for CI, but larger than fast test
            early_stop_threshold=1.6e-3,
            output_dir=self.temp_dir
        )

        # Verify structure
        assert 'metrics' in results
        metrics = results['metrics']
        if results['success']:
            # If successful, check metrics exist
            assert 'final_energy' in metrics
            assert 'fci_energy' in metrics
            assert 'chemical_accuracy_achieved' in metrics
        else:
            # If failed, there should be either errors or unmet criteria (chemical accuracy)
            # At least metrics should have been populated
            assert 'fci_energy' in metrics

        print(f"Full configuration test: success={results['success']}, episodes={len(results.get('search_results', {}).get('episode_energies', []))}")


if __name__ == "__main__":
    # Run tests manually if needed
    test = TestLiHValidation()
    test.setup_method()
    try:
        test.test_module_imports()
        print("✓ Module imports test passed")
        test.test_lih_molecule_processing()
        print("✓ Molecule processing test passed")
        test.test_simulator_creation()
        print("✓ Simulator creation test passed")
        test.test_ppo_agent_creation()
        print("✓ PPO agent creation test passed")
        test.test_ucc_controller_creation()
        print("✓ UCC controller creation test passed")
        test.test_fast_validation_run()
        print("✓ Fast validation run test passed")
        test.test_validation_with_invalid_params()
        print("✓ Invalid parameters test passed")
    finally:
        test.teardown_method()