"""
Integration tests for Phase 2 components.

These tests verify that all Phase 2 components work together correctly.
"""

import os
import sys
import tempfile
import pytest

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_all_phase2_modules_importable(self):
        """Test that all Phase 2 modules are importable."""
        from rlqas.phase2 import rl
        from rlqas.phase2 import sequential_tester
        from rlqas.phase2 import hea_search
        from rlqas.phase2 import experiment
        from rlqas.phase2 import adaptation

        # Verify exports
        assert hasattr(rl, 'DQNAgent')
        assert hasattr(rl, 'AgentFactory')
        assert hasattr(sequential_tester, 'SequentialRLTester')
        assert hasattr(hea_search, 'HEASearchEnv')
        assert hasattr(experiment, 'ExperimentManager')
        assert hasattr(adaptation, 'CapabilityDetector')

    def test_rl_agents_with_environment(self):
        """Test RL agents can interact with UCC environment."""
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create small test environment
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            env = UCCSearchEnv(molecule_data=mol)

            # Test PPO agent
            ppo_agent = AgentFactory.create_agent('ppo', config=None, env=env)
            assert ppo_agent is not None

            # Test DQN agent
            dqn_agent = AgentFactory.create_agent('dqn', config=None, env=env)

            # Test environment interaction
            obs, info = env.reset(seed=42)
            assert obs is not None

            # Test action sampling
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert reward is not None

    def test_sequential_tester_with_agents(self):
        """Test sequential tester can run multiple agents."""
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

            # Run PPO test using run_single_agent
            ppo_result = tester.run_single_agent(
                agent_type='ppo',
                env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                agent_name='ppo_integration',
                total_timesteps=200,
            )

            assert ppo_result is not None

            # Run DQN test
            dqn_result = tester.run_single_agent(
                agent_type='dqn',
                env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                agent_name='dqn_integration',
                total_timesteps=200,
            )

            assert dqn_result is not None

            # Compare results
            comparison = tester.compare_results()
            assert comparison is not None

    def test_hea_search_pipeline(self):
        """Test complete HEA search pipeline."""
        from rlqas.phase2.hea_search import (
            HEASearchEnv,
            HEACircuitBuilder,
            HEASearchController,
        )

        # Test circuit builder with different patterns
        builder = HEACircuitBuilder(n_qubits=4, n_layers=2)

        for pattern in ['linear', 'circular']:
            circuit = builder.build()
            assert circuit is not None

        # Test environment
        env = HEASearchEnv(n_qubits=4, max_layers=2)
        obs, info = env.reset(seed=42)

        # Run a few steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        # Test controller
        controller = HEASearchController(
            n_qubits=4,
            max_layers=2,
            output_dir='/tmp/hea_pipeline_test',
            verbose=0,
        )
        assert controller is not None

    def test_experiment_management_pipeline(self):
        """Test experiment management pipeline."""
        from rlqas.phase2.experiment import (
            ExperimentManager,
            ResultsDatabase,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test config loading
            config_dict = {
                'name': 'integration_test',
                'type': 'sequential_test',
                'description': 'Test experiment',
                'environment': {
                    'molecule': 'h2',
                    'basis': 'sto-3g',
                },
                'training': {
                    'max_episodes': 2,
                },
            }

            # Test manager
            manager = ExperimentManager(output_dir=tmpdir)
            assert manager is not None

            # Test database
            db = ResultsDatabase(db_path=os.path.join(tmpdir, 'test.db'))

            # Store and retrieve
            db.store_experiment(
                experiment_id='int_test_001',
                name='integration_test',
                experiment_type='sequential_test',
                config=config_dict,
                results={'energy': -1.0, 'episodes': 2},
            )

            results = db.get_metrics('int_test_001')
            assert results is not None

    def test_adaptation_framework_integration(self):
        """Test adaptation framework components work together."""
        from rlqas.phase2.adaptation import (
            ExplorationFramework,
            CapabilityDetector,
            FeatureImplementer,
            AdaptiveExecutor,
            CapabilityRegistry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all components
            framework = ExplorationFramework(output_dir=tmpdir)
            detector = CapabilityDetector(cache_results=True)
            implementer = FeatureImplementer(output_dir=tmpdir)
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)
            registry = CapabilityRegistry()

            # Test detection
            status = detector.detect_module('json')
            assert status.available

            # Test implementation generation
            adapter_path = implementer.generate_parity_adapter()
            assert os.path.exists(adapter_path)

            # Test execution with adaptation
            def simple_op(x, y):
                return x + y

            result = executor.execute_with_adaptation(
                operation=simple_op,
                required_capabilities=['json'],
                x=1,
                y=2,
            )
            assert result == 3

            # Test registry
            registry.register_capability('test_cap', 'test', '1.0', validated=True)
            assert registry.is_registered('test_cap')
            assert registry.is_validated('test_cap')

    def test_cross_component_compatibility(self):
        """Test that components from different phases work together."""
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase2.adaptation import AdaptiveExecutor
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase1.molecule.processor import process_molecule

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create environment
            mol = process_molecule(
                molecule='H2',
                bond_length=0.74,
                ansatz_type='UCC',
                active_space=(2, 2),
                basis_set='sto-3g',
                transform='jordan_wigner',
            )

            # Create tester
            tester = SequentialRLTester(output_dir=tmpdir)

            # Create executor
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            # Run agent test with adaptation using run_single_agent
            result = executor.execute_with_adaptation(
                operation=tester.run_single_agent,
                required_capabilities=['os'],
                agent_type='ppo',
                env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                agent_name='cross_component',
                total_timesteps=100,
            )

            assert result is not None

    def test_full_rl_search_cycle(self):
        """Test complete RL search cycle with both agents."""
        from rlqas.phase2.sequential_tester import SequentialRLTester
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

            tester = SequentialRLTester(output_dir=tmpdir)

            # Test both agents using run_single_agent
            for agent_type in ['ppo', 'dqn']:
                result = tester.run_single_agent(
                    agent_type=agent_type,
                    env_fn=lambda seed=None: UCCSearchEnv(molecule_data=mol),
                    agent_name=f'e2e_{agent_type}',
                    total_timesteps=200,
                )

                assert result is not None

            # Generate comparison
            comparison = tester.compare_results()
            assert comparison is not None

    def test_hea_full_pipeline(self):
        """Test complete HEA pipeline from config to search."""
        from rlqas.phase2.hea_search import (
            HEAConfig,
            HEASearchController,
        )

        # Create configuration
        config = HEAConfig(
            n_qubits=4,
            max_layers=2,
            n_episodes=2,
        )

        # Create controller
        controller = HEASearchController(
            n_qubits=4,
            max_layers=2,
            output_dir='/tmp/hea_e2e_test',
            verbose=0,
        )

        # Run search (limited iterations for testing)
        result = controller.search(
            agent_type='ppo',
            n_episodes=2,
            total_timesteps=100,
        )

        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
