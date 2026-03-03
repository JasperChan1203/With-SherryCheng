#!/usr/bin/env python3
"""
Explore Tencirchem API for quantum simulator integration.
"""
import sys
sys.path.append("../001")
import tencirchem
from tencirchem import UCC
import inspect

print("Tencirchem version:", tencirchem.__version__)

# List public classes
print("\n--- Classes in tencirchem ---")
for name in dir(tencirchem):
    if not name.startswith('_'):
        obj = getattr(tencirchem, name)
        if inspect.isclass(obj):
            print(f"  {name}")

# Examine UCC class
print("\n--- UCC class methods ---")
ucc_methods = [m for m in dir(UCC) if not m.startswith('_')]
for method in sorted(ucc_methods):
    print(f"  {method}")

# Check if there's a circuit module
try:
    import tencirchem.circuit
    print("\n--- Circuit module contents ---")
    for name in dir(tencirchem.circuit):
        if not name.startswith('_'):
            print(f"  {name}")
except ImportError:
    print("\nNo circuit module")

# Look for energy computation methods
print("\n--- Looking for energy computation ---")
ucc = None
try:
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    hf = scf.RHF(mol).run()
    ucc = UCC(mol)
    print(f"UCC created: {ucc}")
    print(f"n_qubits: {ucc.n_qubits}")
    print(f"h_fermion_op type: {type(ucc.h_fermion_op)}")
except Exception as e:
    print(f"Error creating UCC: {e}")

if ucc is not None:
    # Check for energy method
    if hasattr(ucc, 'energy'):
        print(f"ucc.energy signature: {inspect.signature(ucc.energy)}")
    if hasattr(ucc, 'kernel'):
        print(f"ucc.kernel signature: {inspect.signature(ucc.kernel)}")
    if hasattr(ucc, 'get_energy'):
        print(f"ucc.get_energy signature: {inspect.signature(ucc.get_energy)}")

# Check for CI vector engine
print("\n--- Searching for CI vector engine ---")
for name in dir(tencirchem):
    if 'ci' in name.lower() or 'CI' in name:
        obj = getattr(tencirchem, name)
        print(f"{name}: {type(obj)}")