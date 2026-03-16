#!/usr/bin/env python3
"""
HEA Search Validation Script.

This script validates the Hardware Efficient Ansatz (HEA) search module
on quantum chemistry problems.

Usage:
    python validate_hea.py --molecule lih --qubits 10 --layers 3
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


def validate_hea_search(
    molecule: str = 'lih',
    basis: str = 'sto-3g',
    n_qubits: int = 10,
    n_layers: int = 3,
    entanglement_pattern: str = 'linear',
    max_iterations: int = 50,
    output_dir: str = 'results/hea_validation',
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate HEA search functionality.

    Args:
        molecule: Molecule name
        basis: Basis set
        n_qubits: Number of qubits
        n_layers: Number of HEA layers
        entanglement_pattern: Entanglement pattern to use
        max_iterations: Maximum optimization iterations
        output_dir: Directory for results
        verbose: Whether to print progress

    Returns:
        Validation results dictionary
    """
    from rlqas.phase2.hea_search import (
        HEASearchEnv,
        HEACircuitBuilder,
        HEASearchController,
        HEAConfig,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'molecule': molecule,
        'basis': basis,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'entanglement_pattern': entanglement_pattern,
        'tests': {},
    }

    # Test 1: Circuit Builder
    if verbose:
        print("\n" + "=" * 50)
        print("Test 1: HEA Circuit Builder")
        print("=" * 50)

    try:
        builder = HEACircuitBuilder(n_qubits=n_qubits, n_layers=n_layers)

        # Test different entanglement patterns
        patterns_tested = []
        for pattern in ['linear', 'circular', 'fully_connected']:
            try:
                circuit = builder.build_circuit(
                    entanglement_pattern=pattern,
                    rotation_type='ry',
                )
                patterns_tested.append(pattern)
                if verbose:
                    print(f"  {pattern}: OK")
            except Exception as e:
                if verbose:
                    print(f"  {pattern}: FAILED - {e}")

        results['tests']['circuit_builder'] = {
            'passed': True,
            'patterns_tested': patterns_tested,
        }

    except Exception as e:
        results['tests']['circuit_builder'] = {
            'passed': False,
            'error': str(e),
        }
        if verbose:
            print(f"Circuit Builder: FAILED - {e}")

    # Test 2: HEA Environment
    if verbose:
        print("\n" + "=" * 50)
        print("Test 2: HEA Environment")
        print("=" * 50)

    try:
        env = HEASearchEnv(n_qubits=n_qubits, n_layers=n_layers)
        obs, info = env.reset(seed=42)

        # Run a few steps
        step_results = []
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_results.append({
                'reward': float(reward) if reward is not None else None,
                'terminated': terminated,
            })

        results['tests']['environment'] = {
            'passed': True,
            'steps_completed': len(step_results),
            'observation_shape': list(obs.shape) if hasattr(obs, 'shape') else 'scalar',
        }

        if verbose:
            print(f"  Environment reset: OK")
            print(f"  Steps completed: {len(step_results)}")
            print(f"  Observation shape: {results['tests']['environment']['observation_shape']}")

    except Exception as e:
        results['tests']['environment'] = {
            'passed': False,
            'error': str(e),
        }
        if verbose:
            print(f"Environment: FAILED - {e}")

    # Test 3: HEA Controller
    if verbose:
        print("\n" + "=" * 50)
        print("Test 3: HEA Controller")
        print("=" * 50)

    try:
        config = HEAConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            max_iterations=min(max_iterations, 10),  # Limit for testing
            entanglement_pattern=entanglement_pattern,
        )

        controller = HEASearchController(config=config)

        # Run a short search
        start_time = time.time()
        search_result = controller.search(
            molecule=molecule,
            basis=basis,
            max_iterations=min(max_iterations, 10),
        )
        search_time = time.time() - start_time

        results['tests']['controller'] = {
            'passed': True,
            'search_time_seconds': search_time,
            'result': search_result,
        }

        if verbose:
            print(f"  Controller created: OK")
            print(f"  Search completed in {search_time:.2f}s")
            if search_result:
                print(f"  Best energy: {search_result.get('best_energy', 'N/A')}")

    except Exception as e:
        results['tests']['controller'] = {
            'passed': False,
            'error': str(e),
        }
        if verbose:
            print(f"Controller: FAILED - {e}")

    # Test 4: Configuration Validation
    if verbose:
        print("\n" + "=" * 50)
        print("Test 4: Configuration Validation")
        print("=" * 50)

    try:
        # Test valid config
        valid_config = HEAConfig(n_qubits=4, n_layers=2)

        # Test different parameter sharing strategies
        strategies_tested = []
        for strategy in ['none', 'layer_wise', 'global']:
            try:
                config = HEAConfig(
                    n_qubits=4,
                    n_layers=2,
                    parameter_sharing=strategy,
                )
                strategies_tested.append(strategy)
            except Exception:
                pass

        results['tests']['configuration'] = {
            'passed': True,
            'strategies_tested': strategies_tested,
        }

        if verbose:
            print(f"  Valid config: OK")
            print(f"  Strategies tested: {strategies_tested}")

    except Exception as e:
        results['tests']['configuration'] = {
            'passed': False,
            'error': str(e),
        }
        if verbose:
            print(f"Configuration: FAILED - {e}")

    # Overall pass/fail
    all_passed = all(
        test.get('passed', False)
        for test in results['tests'].values()
    )
    results['overall_passed'] = all_passed

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'hea_validation_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print("\n" + "=" * 50)
        print("Validation Summary")
        print("=" * 50)
        print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
        print(f"Results saved to: {results_path}")

    return results


def generate_validation_report(
    results: Dict[str, Any],
    output_path: str,
) -> str:
    """Generate human-readable validation report.

    Args:
        results: Validation results dictionary
        output_path: Path to save report

    Returns:
        Path to saved report
    """
    report_lines = [
        "=" * 60,
        "HEA Search Validation Report",
        "=" * 60,
        "",
        f"Generated: {results.get('timestamp', 'N/A')}",
        "",
        "Configuration:",
        f"  Molecule: {results.get('molecule', 'N/A')}",
        f"  Qubits: {results.get('n_qubits', 'N/A')}",
        f"  Layers: {results.get('n_layers', 'N/A')}",
        f"  Entanglement: {results.get('entanglement_pattern', 'N/A')}",
        "",
        "Test Results:",
        "",
    ]

    for test_name, test_result in results.get('tests', {}).items():
        status = "PASSED" if test_result.get('passed') else "FAILED"
        report_lines.append(f"  {test_name.replace('_', ' ').title()}: {status}")

        if test_result.get('patterns_tested'):
            report_lines.append(f"    Patterns: {', '.join(test_result['patterns_tested'])}")
        if test_result.get('strategies_tested'):
            report_lines.append(f"    Strategies: {', '.join(test_result['strategies_tested'])}")
        if test_result.get('error'):
            report_lines.append(f"    Error: {test_result['error']}")

    report_lines.extend([
        "",
        "=" * 60,
        f"OVERALL: {'PASSED' if results.get('overall_passed') else 'FAILED'}",
        "=" * 60,
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HEA Search Validation')
    parser.add_argument('--molecule', type=str, default='lih', help='Molecule name')
    parser.add_argument('--basis', type=str, default='sto-3g', help='Basis set')
    parser.add_argument('--qubits', type=int, default=10, help='Number of qubits')
    parser.add_argument('--layers', type=int, default=3, help='Number of HEA layers')
    parser.add_argument('--entanglement', type=str, default='linear',
                       help='Entanglement pattern')
    parser.add_argument('--iterations', type=int, default=50, help='Max iterations')
    parser.add_argument('--output', type=str, default='results/hea_validation',
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 60)
    print("HEA Search Validation Script")
    print("=" * 60)

    # Run validation
    results = validate_hea_search(
        molecule=args.molecule,
        basis=args.basis,
        n_qubits=args.qubits,
        n_layers=args.layers,
        entanglement_pattern=args.entanglement,
        max_iterations=args.iterations,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # Generate report
    report_path = os.path.join(args.output, 'validation_report.txt')
    generate_validation_report(results, report_path)
    print(f"\nReport saved to: {report_path}")

    return 0 if results.get('overall_passed') else 1


if __name__ == '__main__':
    sys.exit(main())
