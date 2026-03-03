#!/usr/bin/env python3
"""Test process_molecule with BeH2."""
import sys
sys.path.insert(0, '.')

from src.modules.molecule_processor import process_molecule

print("Testing BeH2 molecule at 1.3 Å bond length")
try:
    result = process_molecule(
        molecule="BeH2",
        bond_length=1.3,
        ansatz_type="UCC",
        active_space=None,
        basis_set="sto-3g",
        transform="parity"
    )
    print("Success!")
    print(f"n_qubits: {result.n_qubits}")
    print(f"FCI energy: {result.fci_energy}")
    print(f"Reference state shape: {result.reference_state.shape}")
    print(f"Number of Hamiltonian terms: {len(result.hamiltonian.terms)}")
    # Print first few terms
    count = 0
    for term, coeff in result.hamiltonian.terms.items():
        print(f"  {term}: {coeff}")
        count += 1
        if count >= 5:
            break
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)