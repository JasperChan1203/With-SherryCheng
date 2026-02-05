#!/usr/bin/env python3
from pyscf import gto, scf
import tencirchem
from tencirchem import UCC

# Simple H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
hf = scf.RHF(mol).run()
ucc = UCC(hf, active_space=(2, 2))  # H2 active space
print("UCC attributes:")
for attr in dir(ucc):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Check for Hamiltonian
if hasattr(ucc, 'h'):
    print(f"\nh shape: {ucc.h.shape}")
if hasattr(ucc, 'h_qubit'):
    print(f"h_qubit: {ucc.h_qubit}")
if hasattr(ucc, 'int1e'):
    print(f"int1e shape: {ucc.int1e.shape}")
if hasattr(ucc, 'int2e'):
    print(f"int2e shape: {ucc.int2e.shape}")
if hasattr(ucc, 'n_qubits'):
    print(f"n_qubits: {ucc.n_qubits}")
if hasattr(ucc, 'mapping'):
    print(f"mapping: {ucc.mapping}")

# Try to get Hamiltonian as qubit operator
# Look for get_hamiltonian method
if hasattr(ucc, 'get_hamiltonian'):
    print(f"get_hamiltonian: {ucc.get_hamiltonian()}")