#!/usr/bin/env python3
"""E1: FOP vs QOP operator pool comparison."""
import sys
import os
import json

# Ensure package is findable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import rlqas_chem

MOLECULES = {
    'H2': 0.74,
    'LiH': 1.6,
}
POOLS = ['fop', 'qop']
N_EPISODES = int(os.environ.get('RLQAS_N_EPISODES', '300'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_e1():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for molecule, bond_length in MOLECULES.items():
        for pool in POOLS:
            print(f"\nRunning E1: {molecule} @ {bond_length} Å, pool={pool}, n_episodes={N_EPISODES}")
            try:
                r = rlqas_chem.search(
                    molecule=molecule,
                    bond_length=bond_length,
                    ansatz_type='UCC',
                    agent_type='ppo',
                    operator_pool=pool,
                    n_episodes=N_EPISODES,
                )
                entry = {
                    'molecule': molecule,
                    'bond_length': bond_length,
                    'pool': pool,
                    'best_energy': r['best_energy'],
                    'energy_error_mha': r['energy_error_mha'],
                    'chemical_accuracy': r['chemical_accuracy'],
                    'n_operators': r.get('n_operators'),
                    'n_episodes_run': r.get('n_episodes_run', N_EPISODES),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                entry = {
                    'molecule': molecule,
                    'bond_length': bond_length,
                    'pool': pool,
                    'error': str(e),
                    'best_energy': None,
                    'energy_error_mha': None,
                    'chemical_accuracy': False,
                    'n_operators': None,
                    'n_episodes_run': 0,
                }
            results.append(entry)
            be = entry.get('best_energy')
            err = entry.get('energy_error_mha')
            acc = entry.get('chemical_accuracy')
            if be is not None and err is not None:
                print(f"  Result: energy={be:.6f} Ha, error={err:.3f} mHa, chem_acc={acc}")
            else:
                print(f"  Result: ERROR - {entry.get('error', 'unknown')}")

    # Save JSON
    output_path = os.path.join(RESULTS_DIR, 'e1_pool_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== E1 Summary: FOP vs QOP ===")
    print(f"{'Molecule':<10} {'Pool':<6} {'Error(mHa)':<12} {'ChemAcc':<10} {'N_Ops':<8}")
    print("-" * 50)
    for entry in results:
        err = f"{entry['energy_error_mha']:.3f}" if entry['energy_error_mha'] is not None else "ERROR"
        acc = "YES" if entry['chemical_accuracy'] else "NO"
        ops = str(entry['n_operators']) if entry['n_operators'] is not None else "N/A"
        print(f"{entry['molecule']:<10} {entry['pool']:<6} {err:<12} {acc:<10} {ops:<8}")

    return results


if __name__ == '__main__':
    run_e1()
    sys.exit(0)
