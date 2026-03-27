#!/usr/bin/env python3
"""
Multi-Algorithm Comparison Script.

This script compares PPO and DQN algorithms on quantum architecture search
problems and generates detailed comparison reports.

Usage:
    python compare_algorithms.py --molecule lih --qubits 10 --episodes 50
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


def compare_algorithms(
    molecule: str = 'lih',
    basis: str = 'sto-3g',
    n_qubits: int = 10,
    transformation: str = 'jordan_wigner',
    max_episodes: int = 50,
    output_dir: str = 'results/algorithm_comparison',
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare PPO and DQN algorithms.

    Args:
        molecule: Molecule name
        basis: Basis set
        n_qubits: Number of qubits
        transformation: Qubit transformation type
        max_episodes: Maximum training episodes
        output_dir: Directory for results
        verbose: Whether to print progress

    Returns:
        Comparison results dictionary
    """
    from rlqas.phase2.sequential_tester import SequentialRLTester
    from rlqas.phase2.rl import AgentFactory
    from rlqas.phase1.rl.ucc_env import UCCSearchEnv

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Creating environment: {molecule} ({n_qubits} qubits, {transformation})")

    # Create environment
    env = UCCSearchEnv(
        molecule=molecule,
        basis=basis,
        n_qubits=n_qubits,
        transformation=transformation,
    )

    # Create tester
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tester = SequentialRLTester(output_dir=output_dir)

    results = {}

    # Test PPO
    if verbose:
        print("\n" + "=" * 50)
        print("Testing PPO Algorithm")
        print("=" * 50)

    start_time = time.time()
    ppo_agent = AgentFactory.create_agent('ppo', env)
    ppo_result = tester.run_agent_test(
        agent=ppo_agent,
        agent_id='ppo',
        env=env,
        max_episodes=max_episodes,
    )
    ppo_time = time.time() - start_time

    results['ppo'] = {
        'algorithm': 'PPO',
        'result': ppo_result,
        'training_time_seconds': ppo_time,
    }

    if verbose:
        print(f"PPO completed in {ppo_time:.2f}s")
        if 'best_energy' in ppo_result:
            print(f"Best energy: {ppo_result['best_energy']:.6f}")

    # Test DQN
    if verbose:
        print("\n" + "=" * 50)
        print("Testing DQN Algorithm")
        print("=" * 50)

    start_time = time.time()
    dqn_agent = AgentFactory.create_agent('dqn', env)
    dqn_result = tester.run_agent_test(
        agent=dqn_agent,
        agent_id='dqn',
        env=env,
        max_episodes=max_episodes,
    )
    dqn_time = time.time() - start_time

    results['dqn'] = {
        'algorithm': 'DQN',
        'result': dqn_result,
        'training_time_seconds': dqn_time,
    }

    if verbose:
        print(f"DQN completed in {dqn_time:.2f}s")
        if 'best_energy' in dqn_result:
            print(f"Best energy: {dqn_result['best_energy']:.6f}")

    # Generate comparison
    comparison = tester.compare_results()

    # Determine winner
    ppo_energy = ppo_result.get('best_energy', float('inf'))
    dqn_energy = dqn_result.get('best_energy', float('inf'))

    if ppo_energy < dqn_energy:
        winner = 'ppo'
        winner_energy = ppo_energy
    else:
        winner = 'dqn'
        winner_energy = dqn_energy

    # Check chemical accuracy (1.6 mHa = 0.0016 Ha)
    chemical_accuracy_threshold = 0.0016
    ppo_chemical_accuracy = abs(ppo_energy) < chemical_accuracy_threshold if ppo_energy != float('inf') else False
    dqn_chemical_accuracy = abs(dqn_energy) < chemical_accuracy_threshold if dqn_energy != float('inf') else False

    comparison_summary = {
        'timestamp': datetime.now().isoformat(),
        'molecule': molecule,
        'basis': basis,
        'n_qubits': n_qubits,
        'transformation': transformation,
        'max_episodes': max_episodes,
        'ppo': {
            'best_energy': ppo_energy,
            'training_time': ppo_time,
            'chemical_accuracy': ppo_chemical_accuracy,
        },
        'dqn': {
            'best_energy': dqn_energy,
            'training_time': dqn_time,
            'chemical_accuracy': dqn_chemical_accuracy,
        },
        'winner': winner,
        'winner_energy': winner_energy,
        'comparison': comparison,
    }

    # Save results
    results_path = os.path.join(output_dir, f'comparison_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)

    if verbose:
        print("\n" + "=" * 50)
        print("Comparison Summary")
        print("=" * 50)
        print(f"Winner: {winner.upper()}")
        print(f"Best energy: {winner_energy:.6f}")
        print(f"PPO chemical accuracy: {ppo_chemical_accuracy}")
        print(f"DQN chemical accuracy: {dqn_chemical_accuracy}")
        print(f"Results saved to: {results_path}")

    return comparison_summary


def generate_comparison_report(
    results: Dict[str, Any],
    output_path: str,
) -> str:
    """Generate human-readable comparison report.

    Args:
        results: Comparison results dictionary
        output_path: Path to save report

    Returns:
        Path to saved report
    """
    report_lines = [
        "=" * 60,
        "Multi-Algorithm Comparison Report",
        "=" * 60,
        "",
        f"Generated: {results.get('timestamp', 'N/A')}",
        "",
        "Experiment Configuration:",
        f"  Molecule: {results.get('molecule', 'N/A')}",
        f"  Basis: {results.get('basis', 'N/A')}",
        f"  Qubits: {results.get('n_qubits', 'N/A')}",
        f"  Transformation: {results.get('transformation', 'N/A')}",
        f"  Max Episodes: {results.get('max_episodes', 'N/A')}",
        "",
        "Results:",
        "",
        "PPO Algorithm:",
        f"  Best Energy: {results.get('ppo', {}).get('best_energy', 'N/A'):.6f}",
        f"  Training Time: {results.get('ppo', {}).get('training_time', 0):.2f}s",
        f"  Chemical Accuracy: {results.get('ppo', {}).get('chemical_accuracy', False)}",
        "",
        "DQN Algorithm:",
        f"  Best Energy: {results.get('dqn', {}).get('best_energy', 'N/A'):.6f}",
        f"  Training Time: {results.get('dqn', {}).get('training_time', 0):.2f}s",
        f"  Chemical Accuracy: {results.get('dqn', {}).get('chemical_accuracy', False)}",
        "",
        "=" * 60,
        f"WINNER: {results.get('winner', 'N/A').upper()}",
        f"Best Energy: {results.get('winner_energy', 'N/A'):.6f}",
        "=" * 60,
    ]

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multi-Algorithm Comparison')
    parser.add_argument('--molecule', type=str, default='lih', help='Molecule name')
    parser.add_argument('--basis', type=str, default='sto-3g', help='Basis set')
    parser.add_argument('--qubits', type=int, default=10, help='Number of qubits')
    parser.add_argument('--transformation', type=str, default='jordan_wigner',
                       help='Qubit transformation')
    parser.add_argument('--episodes', type=int, default=50, help='Max episodes')
    parser.add_argument('--output', type=str, default='results/algorithm_comparison',
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Algorithm Comparison Script")
    print("=" * 60)

    # Run comparison
    results = compare_algorithms(
        molecule=args.molecule,
        basis=args.basis,
        n_qubits=args.qubits,
        transformation=args.transformation,
        max_episodes=args.episodes,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # Generate report
    report_path = os.path.join(args.output, 'comparison_report.txt')
    generate_comparison_report(results, report_path)
    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
