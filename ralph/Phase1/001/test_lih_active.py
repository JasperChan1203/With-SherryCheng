#!/usr/bin/env python3
"""Test process_molecule with LiH active space."""
import sys
sys.path.insert(0, '.')

from src.modules.molecule_processor import process_molecule

print("Testing LiH molecule at 1.6 Å bond length with active_space=(2,2)")
try:
    result = process_molecule(
        molecule="LiH",
        bond_length=1.6,
        ansatz_type="UCC",
        active_space=(2, 2),
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