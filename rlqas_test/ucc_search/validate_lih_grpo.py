#!/usr/bin/env python3
"""
LiH UCC GRPO Learning Validation
==================================
Validates that the GRPO agent in RLQAS is genuinely learning on
LiH @ 1.6 Å with active_space=(4,6) and max_excitations=5.

Pass criteria (ALL must be satisfied):
  1. Chemical accuracy: |best_energy - FCI| < 1.6 mHa
  2. Operator efficiency: <= 6 excitation operators
  3. Energy trend: global_best_energy improved over training
  4. Policy loss trend: last 20% mean < first 20% mean

GRPO runs in groups of episodes (not individual timesteps), so diagnostics
are collected per-group via GRPODiagnosticsTracker.

Usage:
  python validate_lih_grpo.py [--episodes 2000] [--output results/grpo_lih_validation.json]
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
from rlqas_chem.rl.grpo_diagnostics_callback import GRPODiagnosticsTracker

MOLECULE     = 'LiH'
BOND_LENGTH  = 1.6
ACTIVE_SPACE = (4, 6)
CHEM_ACC     = 1.6e-3   # Hartree
TARGET_OPS   = 6        # ADAPT-VQE needs 5 @ 1.6 A; we target <= 6


def run_validation(n_episodes: int, output_path: str, diag_path: str):
    print("=" * 60)
    print("RLQAS LiH GRPO Learning Validation")
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
        # GRPO hyperparameters
        'learning_rate': 3e-4,   # maps to 'lr' in GRPOAgent
        'clip_range': 0.2,
        'group_size': 8,          # larger groups -> better variance estimate
        'ent_coef': 0.05,         # maps to 'entropy_coef' in GRPOAgent
        'gamma': 0.99,
        'verbose': 0,
    }

    tracker = GRPODiagnosticsTracker(
        output_path=diag_path,
        checkpoint_freq=10,
        verbose=1,
    )

    controller = UCCSearchController(mol, agent_type='grpo', config=config)

    results = controller.search(
        n_episodes=n_episodes,
        early_stop_threshold=CHEM_ACC,
        callbacks=tracker,
    )

    tracker.finish()
    summary = tracker.summary()

    best_energy   = results.get('best_energy', float('inf'))
    best_ops      = len(results.get('best_excitations') or [])
    chem_acc_pass = bool(abs(best_energy - fci_energy) < CHEM_ACC) if best_energy != float('inf') else False
    ops_pass      = bool(chem_acc_pass and best_ops <= TARGET_OPS)
    energy_error  = abs(best_energy - fci_energy) * 1000  # mHa

    print("\n" + "=" * 60)
    print("GRPO LEARNING DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Groups completed       : {summary['n_groups']}")
    print(f"  Policy loss trend      : {summary['loss_first']:.4f} -> {summary['loss_last']:.4f}"
          f"  {'PASS' if summary['loss_trend_pass'] else 'FAIL (loss should decrease)'}")
    print(f"  Energy trend           : {summary['best_energy_first']:.6f} -> {summary['best_energy_last']:.6f} Ha"
          f"  {'PASS' if summary['energy_trend_pass'] else 'FAIL'}")
    print(f"  Chemical accuracy      : error={energy_error:.3f} mHa"
          f"  {'PASS' if chem_acc_pass else 'FAIL (need < 1.6 mHa)'}")
    print(f"  Operator efficiency    : {best_ops} ops (target <= {TARGET_OPS})"
          f"  {'PASS' if ops_pass else f'FAIL (need <= {TARGET_OPS} ops with chem acc)'}")

    overall = bool(summary['overall_pass'] and chem_acc_pass and ops_pass)
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'PASS -- GRPO is learning' if overall else 'FAIL -- GRPO may not be learning'}")
    print("=" * 60)

    report = {
        'timestamp': str(datetime.datetime.now()),
        'agent': 'grpo',
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
    print(f"GRPO diagnostics -> {diag_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output',   default='results/grpo_lih_validation.json')
    parser.add_argument('--diag',     default='results/grpo_lih_diagnostics.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output, args.diag)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
