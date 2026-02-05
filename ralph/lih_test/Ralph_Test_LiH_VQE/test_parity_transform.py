#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC, parity
from openfermion import FermionOperator

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
fermion_op = ucc.h_fermion_op
print(f"Fermion operator length: {len(fermion_op.terms)}")
print(f"Fermion operator constant term: {fermion_op.terms.get((), 0)}")

# Apply parity transformation
n_modes = 2 * int1e.shape[0]  # spin orbitals
n_elec = 2
print(f"n_modes = {n_modes}, n_elec = {n_elec}")
parity_op = parity(fermion_op, n_modes=n_modes, n_elec=n_elec)
print(f"Parity operator length: {len(parity_op.terms)}")
print(f"Parity operator constant term: {parity_op.terms.get((), 0)}")

# Determine number of qubits from parity operator
# Find max qubit index
max_idx = 0
for term in parity_op.terms:
    for idx, _ in term:
        if idx > max_idx:
            max_idx = idx
n_qubits = max_idx + 1
print(f"Number of qubits from parity operator: {n_qubits}")

# Compare with original qubit operator
print(f"\nOriginal qubit operator n_qubits: {ucc.n_qubits}")
print(f"Original qubit operator constant term: {ucc.h_qubit_op.terms.get((), 0)}")

# Let's also try to see if parity operator has fewer qubits
# Print first few terms
count = 0
for term, coeff in parity_op.terms.items():
    print(f"  {term}: {coeff}")
    count += 1
    if count > 5:
        break

# Check if there is a built-in method to get parity mapped Hamiltonian
# Look for parity method in UCC
if hasattr(ucc, 'parity'):
    print(f"\nUCC.parity method: {ucc.parity}")

# Also check if we can set mapping in UCC.from_integral via engine parameter?
print("\nChecking engine parameter:")
import inspect
sig = inspect.signature(UCC.from_integral)
print(sig)