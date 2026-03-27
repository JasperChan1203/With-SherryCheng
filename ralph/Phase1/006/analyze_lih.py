#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from rlqas.phase1.molecule.processor import process_molecule

print("Processing LiH with active_space=(2,3), Jordan-Wigner transform")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), basis_set='sto-3g', transform='jordan_wigner')
print(f"n_qubits = {data.n_qubits}")
print(f"FCI energy = {data.fci_energy}")
print(f"HF energy = {data.molecular_info.get('hf_energy')}")
print(f"Transform = {data.molecular_info.get('transform')}")
print(f"Active space = {data.molecular_info.get('active_space')}")
print(f"Hamiltonian terms count = {len(data.hamiltonian.terms)}")
print(f"Reference state shape = {data.reference_state.shape}")
print(f"Reference state index = {list(data.reference_state).index(1.0)}")
print(f"UCCSD object present = {data.ucc_sd_object is not None}")
if data.ucc_sd_object:
    print(f"UCCSD ex_ops count = {len(data.ucc_sd_object.ex_ops)}")
    print(f"UCCSD n_params = {data.ucc_sd_object.n_params}")

# Compute Hartree-Fock energy using reference state and Hamiltonian
import numpy as np
from openfermion import expectation
# Convert reference state to vector
psi = data.reference_state
# Compute expectation value
# For simplicity, compute diagonal energy
energy = 0.0
for term, coeff in data.hamiltonian.terms.items():
    diag = True
    sign = 1.0
    for idx, pauli in term:
        if pauli == 'Z':
            # Determine eigenvalue +1 or -1 based on reference state bit
            # reference state is one-hot at index best_state_idx
            best_state_idx = list(psi).index(1.0)
            bit = (best_state_idx >> idx) & 1
            sign *= (1 - 2*bit)
        elif pauli == 'I':
            continue
        else:
            diag = False
            break
    if diag:
        energy += coeff * sign
print(f"Diagonal energy of reference state = {energy}")