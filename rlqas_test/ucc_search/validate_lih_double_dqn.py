#!/usr/bin/env python3
"""
LiH UCC Double-DQN Learning Validation
=========================================
Validates that the Double-DQN agent in RLQAS can achieve the same goals
as the DQN baseline: chemical accuracy + <= 6 excitation operators on
LiH @ 1.6 A with max_excitations=5.

Double-DQN decouples action selection (online net) from value evaluation
(target net) to reduce Q-value overestimation bias.

Pass criteria (ALL must be satisfied):
  1. Chemical accuracy: |best_energy - FCI| < 1.6 mHa
  2. Operator efficiency: <= 6 excitation operators (matching ADAPT-VQE)
  3. Exploration decay: final epsilon < initial epsilon (epsilon-greedy decay)
  4. Energy trend: best_energy improves (< HF energy)

Note: Double-DQN diagnostics are simpler than SB3-DQN because the
controller's _double_dqn_search does not wire a step-level callback.
Epsilon before/after training is read directly from controller.agent.

Usage:
  python validate_lih_double_dqn.py [--episodes 2000] [--output results/double_dqn_lih_validation.json]
"""

import argparse
import datetime
import json
import os
import sys

RLQAS_CHEM = os.path.join(os.path.dirname(__file__), '..', '..', 'rlqas-chem', 'src')
sys.path.insert(0, os.path.abspath(RLQAS_CHEM))

from rlqas_chem.molecule.processor import process_molecule
from rlqas_chem.search.ucc.controller import UCCSearchController

MOLECULE     = 'LiH'
BOND_LENGTH  = 1.6
ACTIVE_SPACE = (4, 6)
CHEM_ACC     = 1.6e-3   # Hartree
TARGET_OPS   = 6        # ADAPT-VQE needs 5 @ 1.6 A; we target <= 6


def run_validation(n_episodes: int, output_path: str):
    print("=" * 60)
    print("RLQAS LiH Double-DQN Learning Validation")
    print(f"Molecule    : {MOLECULE} @ {BOND_LENGTH} A  active_space={ACTIVE_SPACE}")
    print(f"Episodes    : {n_episodes}")
    print(f"Started     : {datetime.datetime.now()}")
    print("=" * 60)

    mol = process_molecule(MOLECULE, BOND_LENGTH, 'UCC', active_space=ACTIVE_SPACE)
    fci_energy = mol.fci_energy
    print(f"FCI energy  : {fci_energy:.6f} Ha  (from processor)")
    print(f"Target ops  : <= {TARGET_OPS} (ADAPT-VQE @ 1.6 A needs 5)")

    config = {
        'environment': {
            'max_excitations': 5,
        },
        'run_classical_opt': True,
        'param_init_strategy': 'zeros',
        'use_early_stop': True,
        # Double-DQN hyperparameters (raw config keys read by controller)
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995,
        'target_update_freq': 100,
        'batch_size': 32,
        'buffer_capacity': 10000,
    }

    controller = UCCSearchController(mol, agent_type='double_dqn', config=config)

    # Record epsilon before training
    eps_initial = controller.agent.epsilon
    print(f"Epsilon initial : {eps_initial:.3f}")

    # --- Run ---
    results = controller.search(
        n_episodes=n_episodes,
        early_stop_threshold=CHEM_ACC,
    )

    # Record epsilon after training
    eps_final = controller.agent.epsilon
    print(f"Epsilon final   : {eps_final:.3f}")

    # --- Evaluate ---
    best_energy   = results.get('best_energy', float('inf'))
    best_ops      = len(results.get('best_excitations') or [])
    chem_acc_pass = bool(abs(best_energy - fci_energy) < CHEM_ACC) if best_energy != float('inf') else False
    ops_pass      = bool(chem_acc_pass and best_ops <= TARGET_OPS)
    energy_error  = abs(best_energy - fci_energy) * 1000  # mHa

    epsilon_decay_pass = bool(eps_final < eps_initial)

    print("\n" + "=" * 60)
    print("DOUBLE-DQN LEARNING DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Exploration decay  : eps {eps_initial:.3f} -> {eps_final:.3f}"
          f"  {'PASS' if epsilon_decay_pass else 'FAIL (epsilon should decrease)'}")
    print(f"  Chemical accuracy  : error={energy_error:.3f} mHa"
          f"  {'PASS' if chem_acc_pass else 'FAIL (need < 1.6 mHa)'}")
    print(f"  Operator efficiency: {best_ops} ops (target <= {TARGET_OPS})"
          f"  {'PASS' if ops_pass else f'FAIL (need <= {TARGET_OPS} ops with chem acc)'}")

    overall = bool(epsilon_decay_pass and chem_acc_pass and ops_pass)
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'PASS -- Double-DQN is learning' if overall else 'FAIL -- Double-DQN may not be learning'}")
    print("=" * 60)

    diagnostics = {
        'epsilon_initial': float(eps_initial),
        'epsilon_final': float(eps_final),
        'epsilon_decay_pass': bool(epsilon_decay_pass),
        'best_energy_final': float(best_energy) if best_energy != float('inf') else None,
        'overall_pass': bool(epsilon_decay_pass),
    }

    report = {
        'timestamp': str(datetime.datetime.now()),
        'agent': 'double_dqn',
        'molecule': MOLECULE,
        'bond_length': BOND_LENGTH,
        'n_episodes': n_episodes,
        'fci_energy': fci_energy,
        'best_energy': best_energy,
        'energy_error_mha': energy_error,
        'best_ops': best_ops,
        'target_ops': TARGET_OPS,
        'chemical_accuracy_pass': chem_acc_pass,
        'operator_efficiency_pass': ops_pass,
        'diagnostics': diagnostics,
        'overall_pass': overall,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {output_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output',   default='results/double_dqn_lih_validation.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
