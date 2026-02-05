#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import patch_openfermion  # patches QubitOperator before importing tencirchem

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
from openfermion import QubitOperator
import tensorcircuit as tc

bond_length = 2.0
mol = gto.M(
    atom=[["H", 0, 0, 0], ["Li", bond_length, 0, 0]],
    basis='sto-3g',
    symmetry=True,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
print(f"HF energy: {hf.e_tot:.8f}")

# Select active orbitals via sort_mo
cas = mcscf.CASCI(hf, 3, 2)
mo_coeff = cas.sort_mo([2, 3, 6])
cas.kernel(mo_coeff)
active_orbitals = [1, 2, 5]
print(f"Active orbitals (0-based): {active_orbitals}")

# Get integrals
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
n_orb = int1e.shape[0]
int2e = ao2mo.restore(1, int2e, n_orb)

# Build Hamiltonian
ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core, hcb=False)
fermion_op = ucc.h_fermion_op
n_modes = 2 * int1e.shape[0]
h_qubit_op = parity(fermion_op, n_modes=n_modes, n_elec=2)
print(f"Number of qubits: {max(idx for term in h_qubit_op.terms for idx,_ in term) + 1}")
print(f"Number of terms: {len(h_qubit_op.terms)}")
print("First 10 terms:")
count = 0
for term, coeff in h_qubit_op.terms.items():
    print(f"  {term}: {coeff}")
    count += 1
    if count >= 10:
        break

# Compute HF state energy by brute force
n_qubits = 4
min_energy = float('inf')
best_state = None
for i in range(2**n_qubits):
    c = tc.Circuit(n_qubits)
    for q in range(n_qubits):
        if (i >> q) & 1:
            c.x(q)
    energy = 0.0
    for term, coeff in h_qubit_op.terms.items():
        x_list = []
        y_list = []
        z_list = []
        for idx, pauli in term:
            if pauli == 'X':
                x_list.append(idx)
            elif pauli == 'Y':
                y_list.append(idx)
            elif pauli == 'Z':
                z_list.append(idx)
        exp_val = c.expectation_ps(x=x_list, y=y_list, z=z_list)
        energy += coeff * exp_val
    energy = energy.real
    if energy < min_energy:
        min_energy = energy
        best_state = i
print(f"HF state energy (active space): {min_energy:.8f}")
print(f"HF state (binary): {format(best_state, '0'+str(n_qubits)+'b')}")
print(f"Total HF energy (core + active): {e_core + min_energy:.8f}")
print(f"Expected total HF energy: {hf.e_tot:.8f}")