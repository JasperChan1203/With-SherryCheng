#!/usr/bin/env python3
import sys
sys.path.append("../001")
import numpy as np
from pyscf import gto, scf
from tencirchem import UCC

# Create H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
hf = scf.RHF(mol).run()
ucc = UCC(mol)

print("UCC object:", ucc)
print("n_qubits:", ucc.n_qubits)
print("h_fermion_op type:", type(ucc.h_fermion_op))

# Test civector method
print("\n--- Testing civector ---")
if hasattr(ucc, 'civector'):
    print("civector signature:", ucc.civector.__doc__)
    # Call with default parameters
    try:
        civ = ucc.civector()
        print("civector returned:", type(civ), "shape:", civ.shape if hasattr(civ, 'shape') else None)
    except Exception as e:
        print("civector error:", e)
else:
    print("civector not found")

# Test statevector method
print("\n--- Testing statevector ---")
if hasattr(ucc, 'statevector'):
    print("statevector signature:", ucc.statevector.__doc__)
    try:
        sv = ucc.statevector()
        print("statevector returned:", type(sv), "shape:", sv.shape if hasattr(sv, 'shape') else None)
    except Exception as e:
        print("statevector error:", e)

# Test energy method with parameters
print("\n--- Testing energy method ---")
if hasattr(ucc, 'energy'):
    print("energy signature:", ucc.energy.__doc__)
    # Get initial parameters
    params = ucc.params
    print("params shape:", params.shape if hasattr(params, 'shape') else params)
    # Compute energy with default parameters
    try:
        e = ucc.energy(params)
        print("energy with params:", e)
    except Exception as e:
        print("energy error:", e)
    # Compute energy with zero parameters
    try:
        zero_params = np.zeros_like(params)
        e0 = ucc.energy(zero_params)
        print("energy with zero params:", e0)
    except Exception as e:
        print("energy zero error:", e)

# Test engine parameter
print("\n--- Testing engine parameter ---")
try:
    e_ci = ucc.energy(params, engine='ci_vector')
    print("energy with engine='ci_vector':", e_ci)
except Exception as e:
    print("engine='ci_vector' error:", e)

try:
    e_sv = ucc.energy(params, engine='statevector')
    print("energy with engine='statevector':", e_sv)
except Exception as e:
    print("engine='statevector' error:", e)