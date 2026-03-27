#!/usr/bin/env python3
"""Explore excitation operators in tencirchem."""

import sys
sys.path.append("../001")
sys.path.append("../002")

from tencirchem import UCCSD
from pyscf import gto, scf

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
hf = scf.RHF(mol).run()
ucc = UCCSD(mol)

print(f"n_qubits: {ucc.n_qubits}")
print(f"n_elec: {ucc.n_elec}")
print(f"n_params: {ucc.n_params}")
print(f"ex_ops: {ucc.ex_ops}")
print(f"type ex_ops: {type(ucc.ex_ops)}")

if hasattr(ucc, 'get_ex1_ops'):
    ex1 = ucc.get_ex1_ops()
    print(f"get_ex1_ops: {ex1}")
if hasattr(ucc, 'get_ex2_ops'):
    ex2 = ucc.get_ex2_ops()
    print(f"get_ex2_ops: {ex2}")
if hasattr(ucc, 'get_ex_ops'):
    ex_all = ucc.get_ex_ops()
    print(f"get_ex_ops: {ex_all}")

# Check mapping between ex_ops and excitation tuples
# According to tencirchem documentation, ex_ops are tuples of indices
# For single excitation: (i, j) where i occupied, j virtual
# For double excitation: (i, j, k, l) where i,k occupied, j,l virtual
# Let's verify by checking length
print("\nExcitation operator lengths:")
for op in ucc.ex_ops:
    print(f"  {op} -> length {len(op)}")

# Get occupied and virtual orbital counts
if hasattr(ucc, 'no'):
    print(f"no (occupied): {ucc.no}")
if hasattr(ucc, 'nv'):
    print(f"nv (virtual): {ucc.nv}")

# Try to create UCC with subset
subset = ucc.ex_ops[:2]
print(f"\nSubset: {subset}")
ucc2 = UCCSD(mol, ex_ops=subset)
print(f"UCC2 n_params: {ucc2.n_params}")
print(f"UCC2 ex_ops: {ucc2.ex_ops}")

# Build circuit and evaluate energy
import numpy as np
params = np.zeros(ucc2.n_params)
energy = ucc2.energy(params)
print(f"Energy zero params: {energy}")
params_rand = np.random.randn(ucc2.n_params) * 0.1
energy_rand = ucc2.energy(params_rand)
print(f"Energy random params: {energy_rand}")

# Check circuit
if hasattr(ucc2, 'get_circuit'):
    circuit = ucc2.get_circuit()
    print(f"Circuit type: {type(circuit)}")
    # Try to get Hamiltonian
    if hasattr(ucc2, 'h_qubit_op'):
        ham = ucc2.h_qubit_op
        print(f"Ham type: {type(ham)}")