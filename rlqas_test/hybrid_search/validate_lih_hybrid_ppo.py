#!/usr/bin/env python3
"""
LiH Hybrid (HEA+UCC) PPO Validation
======================================
Validates that the PPO agent with the Hybrid search controller achieves
chemical accuracy on LiH @ 1.6 Å, full space, Jordan-Wigner transform.

Pass criteria (ALL must be satisfied):
  1. Chemical accuracy: |best_energy - FCI| < 1.6 mHa
  2. Energy improves over HF (best_energy < HF energy)
  3. Circuit structure valid: at least 1 UCC excitation selected

Usage:
  python validate_lih_hybrid_ppo.py [--episodes 500] [--output results/hybrid_ppo_lih_validation.json]
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
from rlqas_chem.search.hybrid.controller import HybridSearchController

MOLECULE    = 'LiH'
BOND_LENGTH = 1.6
TRANSFORM   = 'jordan_wigner'
CHEM_ACC    = 1.6e-3  # Ha


def run_validation(n_episodes: int, output_path: str) -> bool:
    print("=" * 60)
    print("RLQAS LiH Hybrid (HEA+UCC) PPO Validation")
    print(f"Molecule    : {MOLECULE} @ {BOND_LENGTH} Å  full space")
    print(f"Transform   : {TRANSFORM}")
    print(f"Episodes    : {n_episodes}")
    print(f"Started     : {datetime.datetime.now()}")
    print("=" * 60)

    mol = process_molecule(
        MOLECULE, BOND_LENGTH, 'HYBRID',
        active_space=None,
        transform=TRANSFORM,
    )
    fci_energy = mol.fci_energy
    hf_energy  = mol.molecular_info['hf_energy']
    n_qubits   = mol.n_qubits
    print(f"FCI energy  : {fci_energy:.6f} Ha")
    print(f"HF energy   : {hf_energy:.6f} Ha")
    print(f"HF error    : {abs(hf_energy - fci_energy) * 1000:.3f} mHa")
    print(f"Qubits      : {n_qubits}")

    config = {
        'n_episodes': n_episodes,
        'max_blocks': 8,
        'max_depth': 20,
        'run_classical_opt': True,
        'log_frequency': 50,
        # PPO agent settings
        'n_steps': 64,
        'batch_size': 32,
        'n_epochs': 4,
        'learning_rate': 3e-4,
        'verbose': 0,
    }

    controller = HybridSearchController(
        molecule_data=mol,
        agent_type='ppo',
        config=config,
    )

    result = controller.search(
        n_episodes=n_episodes,
        early_stop_threshold=CHEM_ACC,
    )

    best_energy      = result.best_energy
    best_excitations = result.performance_metrics.get('best_excitations', [])
    convergence      = result.convergence_reached

    if best_energy is None:
        best_energy = float('inf')

    energy_error     = abs(best_energy - fci_energy) * 1000  # mHa
    chem_acc_pass    = best_energy != float('inf') and abs(best_energy - fci_energy) < CHEM_ACC
    energy_imp_pass  = best_energy < hf_energy + 1e-6
    circuit_pass     = len(best_excitations) >= 1

    overall = bool(chem_acc_pass and energy_imp_pass and circuit_pass)

    print()
    print("=" * 60)
    print("HYBRID PPO VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Best energy        : {best_energy:.6f} Ha")
    print(f"  FCI energy         : {fci_energy:.6f} Ha")
    print(f"  Energy error       : {energy_error:.3f} mHa  "
          f"{'PASS' if chem_acc_pass else 'FAIL (need < 1.6 mHa)'}")
    print(f"  Better than HF     : {best_energy:.6f} < {hf_energy:.6f}  "
          f"{'PASS' if energy_imp_pass else 'FAIL'}")
    print(f"  Circuit (UCC ops)  : {len(best_excitations)}  "
          f"{'PASS' if circuit_pass else 'FAIL (need >= 1 excitation)'}")
    print(f"  Best excitations   : {best_excitations}")
    print(f"  Converged          : {convergence}")
    print()
    print("=" * 60)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    report = {
        'timestamp': str(datetime.datetime.now()),
        'agent': 'ppo',
        'search_type': 'hybrid',
        'molecule': MOLECULE,
        'bond_length': BOND_LENGTH,
        'active_space': None,
        'transform': TRANSFORM,
        'n_qubits': n_qubits,
        'n_episodes': n_episodes,
        'fci_energy': fci_energy,
        'hf_energy': hf_energy,
        'best_energy': best_energy if best_energy != float('inf') else None,
        'energy_error_mha': energy_error,
        'chemical_accuracy_pass': bool(chem_acc_pass),
        'energy_improvement_pass': bool(energy_imp_pass),
        'circuit_structure_pass': bool(circuit_pass),
        'best_excitations': best_excitations,
        'convergence_reached': convergence,
        'overall_pass': overall,
    }

    def _to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(_to_serializable(report), f, indent=2)
    print(f"\nReport saved -> {output_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--output', default='results/hybrid_ppo_lih_validation.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
