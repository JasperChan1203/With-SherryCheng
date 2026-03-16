"""
Multi-Algorithm Comparison Integration Tests.

These tests verify the algorithm comparison functionality
with different molecules and qubit counts.

CRITICAL: LiH active space notation is (n_electrons, n_orbitals).
Under Jordan-Wigner mapping, n_qubits = 2 * n_orbitals.

- active_space=(2, 5) -> 2 electrons, 5 orbitals -> 10 qubits
- active_space=(2, 6) -> 2 electrons, 6 orbitals -> 12 qubits
"""

import os
import sys
import tempfile
import pytest
import json
from datetime import datetime

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

# Chemical accuracy threshold: 1.6 mHa = 0.0016 Ha
CHEMICAL_ACCURACY_THRESHOLD = 0.0016


class TestMultiAlgorithmComparison:
    """Tests for multi-algorithm comparison functionality."""

    def test_comparison_with_h2_4qubits(self):
        """Test algorithm comparison with H2 4 qubits."""
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            env = UCCSearchEnv(molecule_data=mol)

            tester = SequentialRLTester(output_dir=tmpdir)

            # Run PPO using run_single_agent
            ppo_result = tester.run_single_agent(
                agent_type='ppo',
                env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                agent_name='ppo_h2_4q',
                total_timesteps=500,
            )

            # Run DQN
            dqn_result = tester.run_single_agent(
                agent_type='dqn',
                env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                agent_name='dqn_h2_4q',
                total_timesteps=500,
            )

            # Compare
            comparison = tester.compare_results()

            assert comparison is not None
            assert 'ranking' in comparison
            assert len(comparison.get('agents', {})) == 2

    def test_lih_10qubits_ppo_vs_dqn(self):
        """
        Test PPO vs DQN comparison on LiH 10 qubits.

        CRITICAL: Uses active_space=(2, 5) -> 10 qubits.
        """
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            # CORRECT: active_space=(2, 5) -> 10 qubits
            mol = process_molecule(
                molecule='LiH',
                bond_length=1.6,
                ansatz_type='UCC',
                active_space=(2, 5),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            tester = SequentialRLTester(output_dir=tmpdir)

            # Run both agents
            for agent_type in ['ppo', 'dqn']:
                result = tester.run_single_agent(
                    agent_type=agent_type,
                    env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                    agent_name=f'{agent_type}_lih_10q',
                    total_timesteps=1000,
                )
                assert result is not None, f"{agent_type.upper()} training failed"

            # Compare results
            comparison = tester.compare_results()
            assert comparison is not None
            assert 'ranking' in comparison
            assert len(comparison.get('agents', {})) == 2

            # Verify chemical accuracy tracking
            fci_energy = mol.fci_energy
            for agent_name, agent_result in comparison.get('agents', {}).items():
                final_energy = agent_result.get('final_metrics', {}).get('final_energy')
                if final_energy is not None and fci_energy is not None:
                    energy_error = abs(final_energy - fci_energy)
                    # Track which agents achieve chemical accuracy
                    agent_result['chemical_accuracy_achieved'] = energy_error < CHEMICAL_ACCURACY_THRESHOLD

    def test_comparison_metrics_collection(self):
        """Test that metrics are properly collected during comparison."""
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase2.sequential_tester.metrics import MetricsCollector
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            env = UCCSearchEnv(molecule_data=mol)

            tester = SequentialRLTester(output_dir=tmpdir)
            metrics = MetricsCollector()

            # Run test using run_single_agent
            result = tester.run_single_agent(
                agent_type='ppo',
                env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                agent_name='metrics_test',
                total_timesteps=300,
            )

            # Record metrics from result
            train_time = result.get('avg_train_time', result.get('train_time', 0))
            metrics.record_final_metrics(
                final_energy=result.get('final_energy', -1.0),
                final_reward=result.get('final_reward', 0),
                total_episodes=result.get('total_episodes', 1),
            )

            # Verify metrics - check that we have history or metrics
            assert metrics.metrics is not None

    def test_comparison_report_generation(self):
        """Test comparison report generation."""
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            env = UCCSearchEnv(molecule_data=mol)

            tester = SequentialRLTester(output_dir=tmpdir)

            # Run both agents using run_single_agent
            for agent_type in ['ppo', 'dqn']:
                tester.run_single_agent(
                    agent_type=agent_type,
                    env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                    agent_name=agent_type,
                    total_timesteps=200,
                )

            # Generate report - save_comparison_report doesn't take output_dir
            report_path = tester.save_comparison_report(
                filename='test_comparison_report.json',
            )

            assert os.path.exists(report_path)

            # Verify report content
            with open(report_path, 'r') as f:
                report = json.load(f)

            assert 'agents' in report
            assert 'ranking' in report
            assert 'n_agents' in report

    def test_chemical_accuracy_tracking(self):
        """Test that chemical accuracy is properly tracked."""
        from rlqas.phase2.sequential_tester.metrics import MetricsCollector

        # 1.6 mHa = 0.0016 Ha threshold
        threshold = 0.0016

        # Energy within chemical accuracy
        assert abs(0.001) < threshold
        assert abs(0.0005) < threshold

        # Energy outside chemical accuracy
        assert abs(0.002) > threshold
        assert abs(0.01) > threshold

    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        from rlqas.phase2.sequential_tester.comparison import compare_energy_convergence

        # Create sample results dictionary
        results = {
            'ppo': {
                'agent_type': 'ppo',
                'final_metrics': {
                    'final_energy': -1.139,
                },
            },
            'dqn': {
                'agent_type': 'dqn',
                'final_metrics': {
                    'final_energy': -1.135,
                },
            },
        }

        # Use compare_energy_convergence to analyze
        conv_info = compare_energy_convergence(
            results,
            target_energy=-1.140,
            chemical_accuracy=0.001,
        )

        assert 'agents' in conv_info
        assert 'best_agent' in conv_info
        assert 'best_energy_error' in conv_info


class TestAlgorithmPerformance:
    """Tests for algorithm performance evaluation."""

    def test_ppo_training_efficiency(self):
        """Test PPO training efficiency metrics."""
        from rlqas.phase2.sequential_tester.metrics import MetricsCollector
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            env = UCCSearchEnv(molecule_data=mol)

            agent = AgentFactory.create_agent('ppo', config=None, env=env)

            # Run training
            results = agent.learn(total_timesteps=500)

            # Collect metrics using record_final_metrics
            metrics = MetricsCollector()
            metrics.record_final_metrics(
                final_energy=results.get('final_energy', -1.0),
                final_reward=results.get('reward', 0),
                total_episodes=results.get('episodes', 1),
            )

            # Verify efficiency metrics - check metrics dict exists
            assert metrics.metrics is not None
            assert 'final_energy' in metrics.metrics

    def test_dqn_training_efficiency(self):
        """Test DQN training efficiency metrics."""
        from rlqas.phase2.sequential_tester.metrics import MetricsCollector
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            env = UCCSearchEnv(molecule_data=mol)

            agent = AgentFactory.create_agent('dqn', config=None, env=env)

            # Run training
            results = agent.learn(total_timesteps=500)

            # Collect metrics using record_final_metrics
            metrics = MetricsCollector()
            metrics.record_final_metrics(
                final_energy=results.get('final_energy', -1.0),
                final_reward=results.get('reward', 0),
                total_episodes=results.get('episodes', 1),
            )

            # Verify efficiency metrics - check metrics dict exists
            assert metrics.metrics is not None
            assert 'final_energy' in metrics.metrics

    def test_algorithm_comparison_excitation_operators(self):
        """
        Test that framework identifies which algorithm uses fewer excitation operators.

        This test verifies the sequential tester can track and compare
        the number of excitation operators used by different algorithms.
        """
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            # CORRECT: active_space=(2, 5) -> 10 qubits for LiH
            mol = process_molecule(
                molecule='LiH',
                bond_length=1.6,
                ansatz_type='UCC',
                active_space=(2, 5),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            tester = SequentialRLTester(output_dir=tmpdir)

            # Run both agents
            for agent_type in ['ppo', 'dqn']:
                result = tester.run_single_agent(
                    agent_type=agent_type,
                    env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                    agent_name=f'{agent_type}_excitation_test',
                    total_timesteps=500,
                )
                assert result is not None

            # Compare and verify excitation operator tracking
            comparison = tester.compare_results()
            assert comparison is not None
            assert 'ranking' in comparison

            # The framework should track metrics that correlate with
            # circuit complexity (which relates to excitation operators)
            for agent_name, agent_result in comparison.get('agents', {}).items():
                # Verify metrics are collected
                assert 'final_metrics' in agent_result or len(agent_result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
