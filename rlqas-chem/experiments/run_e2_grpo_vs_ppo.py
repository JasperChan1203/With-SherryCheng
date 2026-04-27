#!/usr/bin/env python3
"""E2: GRPO vs PPO agent comparison."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import rlqas_chem

MOLECULES = {
    'LiH': 1.6,
    'BeH2': 1.3,
}
AGENTS = ['ppo', 'grpo']
N_EPISODES = int(os.environ.get('RLQAS_N_EPISODES', '300'))
N_SEEDS = int(os.environ.get('RLQAS_N_SEEDS', '3'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_e2():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for agent in AGENTS:
        for molecule, bond_length in MOLECULES.items():
            print(f"\nRunning E2: {molecule} @ {bond_length} Å, agent={agent}, {N_SEEDS} seeds")
            seed_results = []
            for seed in range(N_SEEDS):
                print(f"  Seed {seed}...")
                try:
                    r = rlqas_chem.search(
                        molecule=molecule,
                        bond_length=bond_length,
                        ansatz_type='UCC',
                        agent_type=agent,
                        operator_pool='fop',
                        n_episodes=N_EPISODES,
                    )
                    seed_results.append({
                        'best_energy': r['best_energy'],
                        'chemical_accuracy': r['chemical_accuracy'],
                        'n_episodes_run': r.get('n_episodes_run', N_EPISODES),
                    })
                except Exception as e:
                    print(f"    ERROR: {e}")
                    seed_results.append({
                        'best_energy': None,
                        'chemical_accuracy': False,
                        'n_episodes_run': N_EPISODES,
                    })

            valid_energies = [s['best_energy'] for s in seed_results if s['best_energy'] is not None]
            valid_episodes = [s['n_episodes_run'] for s in seed_results]
            chem_acc_rate = sum(1 for s in seed_results if s['chemical_accuracy']) / N_SEEDS

            entry = {
                'agent': agent,
                'molecule': molecule,
                'bond_length': bond_length,
                'mean_energy': float(np.mean(valid_energies)) if valid_energies else None,
                'std_energy': float(np.std(valid_energies)) if len(valid_energies) > 1 else 0.0,
                'mean_vqe_calls': float(np.mean(valid_episodes)),
                'std_vqe_calls': float(np.std(valid_episodes)),
                'chem_acc_rate': chem_acc_rate,
                'seed_results': seed_results,
            }
            results.append(entry)
            me = entry['mean_energy']
            if me is not None:
                print(f"  mean_energy={me:.6f} Ha, chem_acc_rate={chem_acc_rate:.2f}")
            else:
                print(f"  mean_energy=None, chem_acc_rate={chem_acc_rate:.2f}")

    output_path = os.path.join(RESULTS_DIR, 'e2_grpo_vs_ppo.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== E2 Summary: GRPO vs PPO ===")
    print(f"{'Agent':<6} {'Molecule':<8} {'Mean VQE Calls':<16} {'Std':<8} {'ChemAcc Rate'}")
    print("-" * 55)
    for entry in results:
        calls_str = f"{entry['mean_vqe_calls']:.1f}"
        std_str = f"{entry['std_vqe_calls']:.1f}"
        print(f"{entry['agent']:<6} {entry['molecule']:<8} {calls_str:<16} {std_str:<8} {entry['chem_acc_rate']:.2f}")

    return results


if __name__ == '__main__':
    run_e2()
    sys.exit(0)
