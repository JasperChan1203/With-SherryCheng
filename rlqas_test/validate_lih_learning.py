#!/usr/bin/env python3
"""
LiH UCC PPO Learning Validation
================================
Validates that the PPO agent in RLQAS is genuinely learning —
not performing sophisticated random search.

Pass criteria (ALL must be satisfied):
  1. explained_variance (last 20% of updates) > 0.1
  2. Policy entropy has a decreasing trend over training
  3. Best energy improves over training
  4. Chemical accuracy (1.6 mHa) reached within 2000 episodes

Usage:
  python validate_lih_learning.py [--episodes 2000] [--output results/lih_validation.json]
"""

import argparse
import json
import os
import sys
import datetime

# Ensure rlqas-chem is on path
RLQAS_CHEM = os.path.join(os.path.dirname(__file__), '..', 'rlqas-chem', 'src')
sys.path.insert(0, os.path.abspath(RLQAS_CHEM))

import numpy as np
from rlqas_chem.molecule.processor import process_molecule
from rlqas_chem.search.ucc.controller import UCCSearchController
from rlqas_chem.rl.diagnostics_callback import DiagnosticsCallback

MOLECULE       = 'LiH'
BOND_LENGTH    = 1.6
ACTIVE_SPACE   = (4, 6)
FCI_ENERGY     = None        # obtained dynamically from processor
CHEM_ACC       = 1.6e-3      # Ha
TARGET_OPS     = 6           # ADAPT-VQE needs 5 @ 1.6 Å; we target ≤ 6


def run_validation(n_episodes: int, output_path: str, diag_path: str):
    print("=" * 60)
    print(f"RLQAS LiH Learning Validation")
    print(f"Molecule    : {MOLECULE} @ {BOND_LENGTH} Å  active_space={ACTIVE_SPACE}")
    print(f"Episodes    : {n_episodes}")
    print(f"FCI energy  : {FCI_ENERGY} Ha")
    print(f"Started     : {datetime.datetime.now()}")
    print("=" * 60)

    # --- Setup ---
    mol = process_molecule(MOLECULE, BOND_LENGTH, 'UCC', active_space=ACTIVE_SPACE)
    fci_energy = mol.fci_energy
    print(f"FCI energy  : {fci_energy:.6f} Ha  (from processor)")
    print(f"Target ops  : ≤ {TARGET_OPS} (ADAPT-VQE @ 1.6 Å needs 5)")

    config = {
        # UCCSearchConfig reads 'environment' section for env settings.
        'environment': {
            'max_excitations': 5,  # hard cap: episode ends at 5 operators
        },
        'run_classical_opt': True,
        'param_init_strategy': 'zeros',
        'use_early_stop': True,
        'ent_coef': 0.05,          # higher entropy bonus to encourage exploration
        'n_steps': 512,            # shorter rollout → more frequent updates → faster feedback
        'batch_size': 64,
        'n_epochs': 10,
        'learning_rate': 3e-4,
        'verbose': 1,
    }

    controller = UCCSearchController(mol, agent_type='ppo', config=config)

    callback = DiagnosticsCallback(
        output_path=diag_path,
        checkpoint_freq=5,
        verbose=1,
    )

    # --- Run ---
    results = controller.search(
        n_episodes=n_episodes,
        early_stop_threshold=CHEM_ACC,
        callbacks=callback,
    )

    # --- Evaluate ---
    summary = callback.summary()

    best_energy   = results.get('best_energy', float('inf'))
    best_ops      = len(results.get('best_excitations') or [])
    chem_acc_pass = abs(best_energy - fci_energy) < CHEM_ACC if best_energy != float('inf') else False
    ops_pass      = chem_acc_pass and best_ops <= TARGET_OPS
    energy_error  = abs(best_energy - fci_energy) * 1000  # mHa

    print("\n" + "=" * 60)
    print("LEARNING DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Updates collected      : {summary['n_updates']}")
    print(f"  explained_variance     : {summary['explained_variance_final']:.4f}"
          f"  {'✓ PASS' if summary['ev_pass'] else '✗ FAIL (need > 0.1)'}")
    print(f"  Entropy slope          : {summary['entropy_slope']:.5f}"
          f"  {'✓ PASS' if summary['entropy_pass'] else '✗ FAIL (entropy_loss should rise toward 0)'}")
    print(f"  Energy trend           : {summary['best_energy_first']:.6f} → {summary['best_energy_last']:.6f} Ha"
          f"  {'✓ PASS' if summary['energy_trend_pass'] else '✗ FAIL'}")
    print(f"  Chemical accuracy      : error={energy_error:.3f} mHa"
          f"  {'✓ PASS' if chem_acc_pass else '✗ FAIL (need < 1.6 mHa)'}")
    print(f"  Operator efficiency    : {best_ops} ops (target ≤ {TARGET_OPS})"
          f"  {'✓ PASS' if ops_pass else f'✗ FAIL (need ≤ {TARGET_OPS} ops with chem acc)'}")

    overall = summary['overall_pass'] and chem_acc_pass and ops_pass
    print("\n" + ("=" * 60))
    print(f"  OVERALL: {'✓ PASS — RL is learning' if overall else '✗ FAIL — RL may not be learning'}")
    print("=" * 60)

    # --- Save full report ---
    report = {
        'timestamp': str(datetime.datetime.now()),
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
        'diagnostics': summary,
        'overall_pass': overall,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {output_path}")
    print(f"Learning curves → {diag_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output',   default='results/lih_validation.json')
    parser.add_argument('--diag',     default='results/lih_diagnostics.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output, args.diag)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
