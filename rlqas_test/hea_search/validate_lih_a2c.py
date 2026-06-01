#!/usr/bin/env python3
"""
LiH HEA A2C Learning Validation
=================================
Validates that the A2C agent in RLQAS can learn an HEA architecture that
achieves chemical accuracy on LiH @ 1.6 A with active_space=(2,3).

Pass criteria (ALL must be satisfied):
  1. Chemical accuracy: |best_energy - FCI| < 1.6 mHa
  2. Energy improves over training (best_energy < HF energy)
  3. Best circuit is structurally valid (max_layers=3 layers decided)

Usage:
  python validate_lih_a2c.py [--episodes 500] [--output results/a2c_lih_hea_validation.json]
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
from rlqas_chem.search.hea.controller import HEASearchController

MOLECULE     = 'LiH'
BOND_LENGTH  = 1.6
ACTIVE_SPACE = (2, 3)
TRANSFORM    = 'jordan_wigner'
MAX_LAYERS   = 3
CHEM_ACC     = 1.6e-3  # Ha


def run_validation(n_episodes: int, output_path: str):
    print("=" * 60)
    print("RLQAS LiH HEA A2C Learning Validation")
    print(f"Molecule    : {MOLECULE} @ {BOND_LENGTH} A  active_space={ACTIVE_SPACE}")
    print(f"Transform   : {TRANSFORM}  max_layers={MAX_LAYERS}")
    print(f"Episodes    : {n_episodes}")
    print(f"Started     : {datetime.datetime.now()}")
    print("=" * 60)

    mol = process_molecule(MOLECULE, BOND_LENGTH, 'HEA', active_space=ACTIVE_SPACE, aslst=[1, 2, 5],
                           transform=TRANSFORM)
    fci_energy = mol.fci_energy
    hf_energy  = mol.molecular_info['hf_energy']
    print(f"FCI energy  : {fci_energy:.6f} Ha")
    print(f"HF energy   : {hf_energy:.6f} Ha")
    print(f"HF error    : {abs(hf_energy - fci_energy) * 1000:.3f} mHa")

    total_timesteps = n_episodes * MAX_LAYERS
    # n_steps must be <= total_timesteps for A2C
    n_steps = min(64, total_timesteps)

    controller = HEASearchController(
        n_qubits=mol.n_qubits,
        max_layers=MAX_LAYERS,
        output_dir=os.path.join(os.path.dirname(output_path), 'checkpoints'),
        verbose=1,
    )

    results = controller.search(
        agent_type='a2c',
        agent_config={
            'n_steps': n_steps,
            'ent_coef': 0.05,
            'learning_rate': 7e-4,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
        },
        n_episodes=n_episodes,
        total_timesteps=total_timesteps,
        target_energy=fci_energy,
        molecule_data=mol,
        early_stop_threshold=CHEM_ACC,
        run_classical_opt=True,
    )

    best_energy  = results.get('best_energy', float('inf'))
    best_circuit = results.get('best_circuit')
    energy_error = abs(best_energy - fci_energy) * 1000  # mHa

    chem_acc_pass   = bool(best_energy != float('inf') and abs(best_energy - fci_energy) < CHEM_ACC)
    energy_imp_pass = bool(best_energy < hf_energy + 1e-6)  # 1 µHa tolerance for floating-point noise
    circuit_pass    = False
    circuit_info    = {}
    if best_circuit is not None:
        ent_len = len(best_circuit.get('entanglement_history', []))
        rot_len = len(best_circuit.get('rotation_history', []))
        circuit_pass = (ent_len == MAX_LAYERS and rot_len == MAX_LAYERS)
        circuit_info = {
            'entanglement_history': best_circuit.get('entanglement_history'),
            'rotation_history': best_circuit.get('rotation_history'),
            'n_parameters': best_circuit.get('n_parameters'),
        }

    overall = bool(chem_acc_pass and energy_imp_pass and circuit_pass)

    print("\n" + "=" * 60)
    print("A2C HEA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Best energy        : {best_energy:.6f} Ha")
    print(f"  FCI energy         : {fci_energy:.6f} Ha")
    print(f"  Energy error       : {energy_error:.3f} mHa"
          f"  {'PASS' if chem_acc_pass else 'FAIL (need < 1.6 mHa)'}")
    print(f"  Better than HF     : {best_energy:.6f} < {hf_energy:.6f}"
          f"  {'PASS' if energy_imp_pass else 'FAIL'}")
    print(f"  Circuit structure  : layers={MAX_LAYERS}"
          f"  {'PASS' if circuit_pass else 'FAIL (circuit missing or wrong length)'}")
    if circuit_info:
        print(f"  Entanglement       : {circuit_info['entanglement_history']}")
        print(f"  Rotations          : {circuit_info['rotation_history']}")
        print(f"  Parameters         : {circuit_info['n_parameters']}")
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'PASS -- A2C is learning' if overall else 'FAIL -- A2C may not be learning'}")
    print("=" * 60)

    report = {
        'timestamp': str(datetime.datetime.now()),
        'agent': 'a2c',
        'molecule': MOLECULE,
        'bond_length': BOND_LENGTH,
        'active_space': list(ACTIVE_SPACE),
        'transform': TRANSFORM,
        'max_layers': MAX_LAYERS,
        'n_episodes': n_episodes,
        'total_timesteps': total_timesteps,
        'fci_energy': fci_energy,
        'hf_energy': hf_energy,
        'best_energy': best_energy,
        'energy_error_mha': energy_error,
        'chemical_accuracy_pass': chem_acc_pass,
        'energy_improvement_pass': energy_imp_pass,
        'circuit_structure_pass': circuit_pass,
        'best_circuit': circuit_info,
        'overall_pass': overall,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {output_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--output', default='results/a2c_lih_hea_validation.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
