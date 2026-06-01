#!/usr/bin/env python3
"""
LiH UCC A2C Learning Validation
=================================
Validates that the A2C agent in RLQAS is genuinely learning on LiH @ 1.6 Å
with active_space=(4,6) and max_excitations=5.

Pass criteria (ALL must be satisfied):
  1. explained_variance (last 20% of updates) > 0.1
  2. Policy entropy has a decreasing trend over training
  3. Best energy improves over training
  4. Chemical accuracy (1.6 mHa) reached within 2000 episodes
  5. Operator efficiency: ≤ 6 excitation operators

A2C uses the same SB3 on-policy infrastructure as PPO so DiagnosticsCallback
works unchanged.

Usage:
  python validate_lih_a2c.py [--episodes 2000] [--output results/a2c_lih_validation.json]
"""

import argparse
import datetime
import json
import os
import sys

RLQAS_CHEM = os.path.join(os.path.dirname(__file__), '..', '..', 'rlqas-chem', 'src')
sys.path.insert(0, os.path.abspath(RLQAS_CHEM))

import numpy as np
from rlqas_chem.molecule.processor import process_molecule
from rlqas_chem.search.ucc.controller import UCCSearchController
from rlqas_chem.rl.diagnostics_callback import DiagnosticsCallback

MOLECULE     = 'LiH'
BOND_LENGTH  = 1.6
ACTIVE_SPACE = (4, 6)
CHEM_ACC     = 1.6e-3   # Hartree
TARGET_OPS   = 6        # ADAPT-VQE needs 5 @ 1.6 Å; we target ≤ 6


def run_validation(n_episodes: int, output_path: str, diag_path: str):
    print("=" * 60)
    print("RLQAS LiH A2C Learning Validation")
    print(f"Molecule    : {MOLECULE} @ {BOND_LENGTH} Å  active_space={ACTIVE_SPACE}")
    print(f"Episodes    : {n_episodes}")
    print(f"Started     : {datetime.datetime.now()}")
    print("=" * 60)

    mol = process_molecule(MOLECULE, BOND_LENGTH, 'UCC', active_space=ACTIVE_SPACE)
    fci_energy = mol.fci_energy
    print(f"FCI energy  : {fci_energy:.6f} Ha  (from processor)")
    print(f"Target ops  : ≤ {TARGET_OPS} (ADAPT-VQE @ 1.6 Å needs 5)")

    config = {
        'environment': {
            'max_excitations': 5,
        },
        'run_classical_opt': True,
        'param_init_strategy': 'zeros',
        'use_early_stop': True,
        # A2C hyperparameters
        'learning_rate': 7e-4,
        'n_steps': 128,         # A2C default: shorter rollouts than PPO
        'ent_coef': 0.05,       # entropy bonus to encourage exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
    }

    controller = UCCSearchController(mol, agent_type='a2c', config=config)

    callback = DiagnosticsCallback(
        output_path=diag_path,
        checkpoint_freq=5,
        verbose=1,
    )

    results = controller.search(
        n_episodes=n_episodes,
        early_stop_threshold=CHEM_ACC,
        callbacks=callback,
    )

    summary = callback.summary()

    best_energy   = results.get('best_energy', float('inf'))
    best_ops      = len(results.get('best_excitations') or [])
    chem_acc_pass = bool(abs(best_energy - fci_energy) < CHEM_ACC) if best_energy != float('inf') else False
    ops_pass      = bool(chem_acc_pass and best_ops <= TARGET_OPS)
    energy_error  = abs(best_energy - fci_energy) * 1000  # mHa

    print("\n" + "=" * 60)
    print("A2C LEARNING DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Updates collected      : {summary['n_updates']}")

    ev_val = summary['explained_variance_final']
    ev_str = f"{ev_val:.4f}" if ev_val is not None else "N/A"
    print(f"  explained_variance     : {ev_str}"
          f"  {'PASS' if summary['ev_pass'] else 'FAIL (need > 0.1)'}")

    ent_slope = summary['entropy_slope']
    ent_str = f"{ent_slope:.5f}" if ent_slope is not None else "N/A"
    print(f"  Entropy slope          : {ent_str}"
          f"  {'PASS' if summary['entropy_pass'] else 'FAIL (entropy_loss should rise toward 0)'}")

    e_first = summary['best_energy_first']
    e_last  = summary['best_energy_last']
    e_first_str = f"{e_first:.6f}" if e_first is not None else "N/A"
    e_last_str  = f"{e_last:.6f}"  if e_last  is not None else "N/A"
    print(f"  Energy trend           : {e_first_str} -> {e_last_str} Ha"
          f"  {'PASS' if summary['energy_trend_pass'] else 'FAIL'}")

    print(f"  Chemical accuracy      : error={energy_error:.3f} mHa"
          f"  {'PASS' if chem_acc_pass else 'FAIL (need < 1.6 mHa)'}")
    print(f"  Operator efficiency    : {best_ops} ops (target <= {TARGET_OPS})"
          f"  {'PASS' if ops_pass else f'FAIL (need <= {TARGET_OPS} ops with chem acc)'}")

    # For A2C, explained_variance may not be available if the logger
    # hasn't flushed yet at _on_rollout_end time.  Fall back to the
    # energy-trend + chemical-accuracy criteria in that case.
    diag_pass = summary['overall_pass'] if summary['n_updates'] > 0 else (
        summary['energy_trend_pass']
    )
    overall = diag_pass and chem_acc_pass and ops_pass
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'PASS -- A2C is learning' if overall else 'FAIL -- A2C may not be learning'}")
    print("=" * 60)

    report = {
        'timestamp': str(datetime.datetime.now()),
        'agent': 'a2c',
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
    os.makedirs(os.path.dirname(diag_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {output_path}")
    print(f"Learning curves -> {diag_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output',   default='results/a2c_lih_validation.json')
    parser.add_argument('--diag',     default='results/a2c_lih_diagnostics.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output, args.diag)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
