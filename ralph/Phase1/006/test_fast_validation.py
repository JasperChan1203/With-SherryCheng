#!/usr/bin/env python3
"""Fast validation test with tuned hyperparameters."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'

from rlqas.phase1.validation.validator import run_lih_validation

print("Running fast LiH validation with tuned hyperparameters")
print("n_episodes=50, early_stop_threshold=0.01 (10 mHa)")
results = run_lih_validation(
    bond_length=1.6,
    active_space=(2,3),
    basis_set='sto-3g',
    transform='jordan_wigner',
    n_episodes=50,
    early_stop_threshold=0.01,  # 10 mHa for fast convergence
    output_dir='./temp_fast_val',
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
    print(f"Converged: {sr.get('convergence_reached')}")
    if sr.get('best_energy') and results.get('fci_energy'):
        error = (sr['best_energy'] - results['fci_energy']) * 1000
        print(f"Error: {error:.3f} mHa")
        print(f"Chemical accuracy (<1.6 mHa): {abs(error) < 1.6}")
    print(f"Episodes completed: {len(sr.get('episode_energies', []))}")
    # Print energy progression
    energies = sr.get('episode_energies', [])
    if len(energies) > 0:
        print(f"Energy progression (first 10): {[f'{e:.6f}' for e in energies[:10]]}")
        # Best energy per episode
        best = energies[0]
        for i, e in enumerate(energies):
            if e < best:
                best = e
                print(f"  Episode {i}: new best {e:.6f}")

# Clean up temp directory
import shutil
if os.path.exists('./temp_fast_val'):
    shutil.rmtree('./temp_fast_val')
    print("Cleaned up temp directory.")

print("\nFast validation complete.")