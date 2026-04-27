#!/usr/bin/env python3
"""E3: Pareto frontier — energy vs circuit complexity."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import rlqas_chem

ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]
MOLECULE = 'LiH'
BOND_LENGTH = 1.6
N_EPISODES = int(os.environ.get('RLQAS_N_EPISODES', '300'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_e3():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for alpha in ALPHAS:
        print(f"\nRunning E3: LiH @ {BOND_LENGTH} Å, alpha={alpha}")
        try:
            r = rlqas_chem.search(
                molecule=MOLECULE,
                bond_length=BOND_LENGTH,
                ansatz_type='UCC',
                agent_type='ppo',
                operator_pool='fop',
                alpha=alpha,
                n_episodes=N_EPISODES,
            )
            entry = {
                'alpha': alpha,
                'best_energy': r['best_energy'],
                'energy_error_mha': r['energy_error_mha'],
                'chemical_accuracy': r['chemical_accuracy'],
                'n_operators': r.get('n_operators'),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            entry = {
                'alpha': alpha,
                'error': str(e),
                'best_energy': None,
                'energy_error_mha': None,
                'chemical_accuracy': False,
                'n_operators': None,
            }
        results.append(entry)
        err = entry.get('energy_error_mha')
        ops = entry.get('n_operators')
        if err is not None:
            print(f"  alpha={alpha}: error={err:.3f} mHa, n_ops={ops}")
        else:
            print(f"  alpha={alpha}: ERROR - {entry.get('error', 'unknown')}")

    output_path = os.path.join(RESULTS_DIR, 'e3_pareto_frontier.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print Pareto table
    print("\n=== E3 Pareto Frontier: Energy vs Circuit Complexity ===")
    print(f"{'Alpha':<8} {'Error(mHa)':<14} {'N_Operators'}")
    print("-" * 35)
    for entry in results:
        err = f"{entry['energy_error_mha']:.3f}" if entry['energy_error_mha'] is not None else "ERROR"
        ops = str(entry['n_operators']) if entry['n_operators'] is not None else "N/A"
        print(f"{entry['alpha']:<8.1f} {err:<14} {ops}")

    return results


if __name__ == '__main__':
    run_e3()
    sys.exit(0)
