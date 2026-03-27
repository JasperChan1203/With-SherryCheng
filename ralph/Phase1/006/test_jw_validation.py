#!/usr/bin/env python3
"""Test LiH with Jordan-Wigner transformation."""

import sys
import os
sys.path.insert(0, 'src')
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.validation.validator import run_lih_validation

print("Running LiH validation with Jordan-Wigner transform")
print("active_space=(2,3), n_episodes=10, early_stop_threshold=1e-6")
results = run_lih_validation(
    bond_length=1.6,
    active_space=(2,3),
    basis_set='sto-3g',
    transform='jordan_wigner',
    n_episodes=10,
    early_stop_threshold=1e-6,  # very small to avoid early stop
    output_dir='./temp_jw_results',
    generate_report=False
)

print("\n=== Results ===")
print(f"Success: {results.get('success')}")
print(f"Errors: {results.get('errors')}")
if 'molecule_info' in results:
    print(f"Qubits: {results['molecule_info'].get('n_qubits')}")
    print(f"FCI energy: {results['fci_energy']}")
if 'search_results' in results:
    sr = results['search_results']
    print(f"Best energy: {sr.get('best_energy')}")
    print(f"Final energy: {sr.get('final_energy')}")
    print(f"Converged: {sr.get('converged')}")
    print(f"Episodes completed: {sr.get('episodes_completed')}")
    if sr.get('best_energy') and results.get('fci_energy'):
        error = (sr['best_energy'] - results['fci_energy']) * 1000
        print(f"Error: {error:.3f} mHa")
        print(f"Chemical accuracy (<1.6 mHa): {abs(error) < 1.6}")

# Clean up
import shutil
if os.path.exists('./temp_jw_results'):
    shutil.rmtree('./temp_jw_results')
    print("Cleaned up temp directory.")