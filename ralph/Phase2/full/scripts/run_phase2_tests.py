#!/usr/bin/env python3
"""
Phase 2 Integration Test Runner.

This script runs comprehensive integration tests for all Phase 2 components:
1. Multi-algorithm comparison (PPO, DQN)
2. LiH molecule tests with Jordan-Wigner transformation
3. HEA search validation
4. Runtime adaptation testing

Usage:
    python run_phase2_tests.py [--molecule MOL] [--qubits N] [--verbose]
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)


def run_unit_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run all Phase 2 unit tests.

    Args:
        verbose: Whether to show detailed output

    Returns:
        Test results dictionary
    """
    import subprocess

    test_dir = os.path.join(PROJECT_ROOT, 'tests')
    cmd = [sys.executable, '-m', 'pytest', test_dir, '-v', '--tb=short']

    if not verbose:
        cmd.append('-q')

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time

    # Parse results
    passed = result.stdout.count(' PASSED') + result.stderr.count(' PASSED')
    failed = result.returncode != 0

    return {
        'test_type': 'unit_tests',
        'passed': passed,
        'failed': failed,
        'elapsed_seconds': elapsed,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }


def run_integration_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run Phase 2 integration tests.

    Args:
        verbose: Whether to show detailed output

    Returns:
        Test results dictionary
    """
    import subprocess

    test_dir = os.path.join(PROJECT_ROOT, 'tests', 'integration')
    if not os.path.exists(test_dir):
        return {
            'test_type': 'integration_tests',
            'skipped': True,
            'reason': 'Integration test directory not found',
        }

    cmd = [sys.executable, '-m', 'pytest', test_dir, '-v', '--tb=short']

    if not verbose:
        cmd.append('-q')

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time

    return {
        'test_type': 'integration_tests',
        'passed': result.returncode == 0,
        'return_code': result.returncode,
        'elapsed_seconds': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }


def test_algorithm_comparison(verbose: bool = False) -> Dict[str, Any]:
    """Test algorithm comparison framework.

    Args:
        verbose: Whether to show detailed output

    Returns:
        Test results dictionary
    """
    try:
        from rlqas.phase2.sequential_tester import SequentialRLTester
        from rlqas.phase2.rl import AgentFactory
        from rlqas.phase1.rl.ucc_env import UCCSearchEnv

        # Create test environment
        env = UCCSearchEnv(
            molecule='lih',
            basis='sto-3g',
            n_qubits=4,  # Small test
            transformation='jordan_wigner',
        )

        # Create tester
        tester = SequentialRLTester(output_dir='/tmp/phase2_tests')

        # Test with PPO
        ppo_agent = AgentFactory.create_agent('ppo', env)
        ppo_result = tester.run_agent_test(
            agent=ppo_agent,
            agent_id='ppo_test',
            env=env,
            max_episodes=2,
        )

        # Test with DQN
        dqn_agent = AgentFactory.create_agent('dqn', env)
        dqn_result = tester.run_agent_test(
            agent=dqn_agent,
            agent_id='dqn_test',
            env=env,
            max_episodes=2,
        )

        # Compare results
        comparison = tester.compare_results()

        return {
            'test_type': 'algorithm_comparison',
            'passed': True,
            'ppo_episodes': len(ppo_result.get('episodes', [])),
            'dqn_episodes': len(dqn_result.get('episodes', [])),
            'comparison_available': comparison is not None,
        }

    except Exception as e:
        return {
            'test_type': 'algorithm_comparison',
            'passed': False,
            'error': str(e),
        }


def test_hea_search(verbose: bool = False) -> Dict[str, Any]:
    """Test HEA search functionality.

    Args:
        verbose: Whether to show detailed output

    Returns:
        Test results dictionary
    """
    try:
        from rlqas.phase2.hea_search import (
            HEASearchEnv,
            HEACircuitBuilder,
            HEASearchController,
            HEAConfig,
        )

        # Test circuit builder
        builder = HEACircuitBuilder(n_qubits=4, n_layers=2)
        circuit = builder.build_circuit(
            entanglement_pattern='linear',
            rotation_type='ry',
        )

        # Test environment
        env = HEASearchEnv(n_qubits=4, n_layers=2)
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Test controller
        config = HEAConfig(n_qubits=4, n_layers=2, max_iterations=2)
        controller = HEASearchController(config=config)

        return {
            'test_type': 'hea_search',
            'passed': True,
            'circuit_built': circuit is not None,
            'env_step_successful': reward is not None,
            'controller_created': controller is not None,
        }

    except Exception as e:
        return {
            'test_type': 'hea_search',
            'passed': False,
            'error': str(e),
        }


def test_experiment_management(verbose: bool = False) -> Dict[str, Any]:
    """Test experiment management system.

    Args:
        verbose: Whether to show detailed output

    Returns:
        Test results dictionary
    """
    try:
        from rlqas.phase2.experiment import (
            ExperimentManager,
            ConfigLoader,
            ResultsDatabase,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test config loader
            config_dict = {
                'experiment_name': 'test_exp',
                'molecule': 'h2',
                'basis': 'sto-3g',
                'algorithm': 'ppo',
                'max_episodes': 2,
            }

            config = ConfigLoader.validate_config(config_dict)

            # Test experiment manager
            manager = ExperimentManager(output_dir=tmpdir)

            # Test results database
            db = ResultsDatabase(db_path=os.path.join(tmpdir, 'test.db'))
            db.initialize()

            # Store test result
            db.store_experiment(
                experiment_id='test_001',
                config=config_dict,
                metrics={'energy': -1.0, 'episodes': 2},
            )

            # Query results
            results = db.get_metrics('test_001')

            return {
                'test_type': 'experiment_management',
                'passed': True,
                'config_validated': config is not None,
                'db_initialized': True,
                'results_stored': results is not None,
            }

    except Exception as e:
        return {
            'test_type': 'experiment_management',
            'passed': False,
            'error': str(e),
        }


def test_adaptation_framework(verbose: bool = False) -> Dict[str, Any]:
    """Test runtime adaptation framework.

    Args:
        verbose: Whether to show detailed output

    Returns:
        Test results dictionary
    """
    try:
        from rlqas.phase2.adaptation import (
            ExplorationFramework,
            CapabilityDetector,
            FeatureImplementer,
            AdaptiveExecutor,
            CapabilityRegistry,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test exploration framework
            framework = ExplorationFramework(output_dir=tmpdir)
            framework.evaluate_compatibility('ppo')

            # Test capability detector
            detector = CapabilityDetector()
            status = detector.detect_module('os')

            # Test feature implementer
            implementer = FeatureImplementer(output_dir=tmpdir)
            adapter_path = implementer.generate_parity_adapter()

            # Test adaptive executor
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)
            executor.check_capability('os')

            # Test registry
            registry = CapabilityRegistry()
            registry.register_capability('test_cap', 'test', '1.0')

            return {
                'test_type': 'adaptation_framework',
                'passed': True,
                'framework_works': True,
                'detector_works': status.available,
                'implementer_works': os.path.exists(adapter_path),
                'executor_works': True,
                'registry_works': registry.is_registered('test_cap'),
            }

    except Exception as e:
        return {
            'test_type': 'adaptation_framework',
            'passed': False,
            'error': str(e),
        }


def generate_report(results: List[Dict[str, Any]], output_path: str) -> str:
    """Generate test report.

    Args:
        results: List of test results
        output_path: Path to save report

    Returns:
        Path to saved report
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_tests': len(results),
        'passed_tests': sum(1 for r in results if r.get('passed', False)),
        'failed_tests': sum(1 for r in results if not r.get('passed', False)),
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 2 Integration Test Runner')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', type=str, default='phase2_test_results.json',
                       help='Output file for results')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 Integration Test Runner")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    results = []

    # Run unit tests
    print("Running unit tests...")
    unit_results = run_unit_tests(verbose=args.verbose)
    results.append(unit_results)
    print(f"  Unit tests completed in {unit_results['elapsed_seconds']:.2f}s")
    print()

    # Run integration tests
    print("Running integration tests...")
    integration_results = run_integration_tests(verbose=args.verbose)
    results.append(integration_results)
    print(f"  Integration tests completed in {integration_results['elapsed_seconds']:.2f}s")
    print()

    # Test algorithm comparison
    print("Testing algorithm comparison...")
    algo_results = test_algorithm_comparison(verbose=args.verbose)
    results.append(algo_results)
    print(f"  Algorithm comparison: {'PASSED' if algo_results['passed'] else 'FAILED'}")
    print()

    # Test HEA search
    print("Testing HEA search...")
    hea_results = test_hea_search(verbose=args.verbose)
    results.append(hea_results)
    print(f"  HEA search: {'PASSED' if hea_results['passed'] else 'FAILED'}")
    print()

    # Test experiment management
    print("Testing experiment management...")
    exp_results = test_experiment_management(verbose=args.verbose)
    results.append(exp_results)
    print(f"  Experiment management: {'PASSED' if exp_results['passed'] else 'FAILED'}")
    print()

    # Test adaptation framework
    print("Testing adaptation framework...")
    adapt_results = test_adaptation_framework(verbose=args.verbose)
    results.append(adapt_results)
    print(f"  Adaptation framework: {'PASSED' if adapt_results['passed'] else 'FAILED'}")
    print()

    # Generate report
    report_path = generate_report(results, args.output)
    print(f"Report saved to: {report_path}")
    print()

    # Summary
    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)
    print("=" * 60)
    print(f"Summary: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
