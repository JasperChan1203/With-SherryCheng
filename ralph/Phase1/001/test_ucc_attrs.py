#!/usr/bin/env python3
"""Test script to inspect UCC attributes."""
import sys
sys.path.insert(0, '.')

try:
    from pyscf import gto, scf
    import tencirchem
    from tencirchem import UCC
    print("Imports successful")
    print(f"Tencirchem version: {tencirchem.__version__}")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Create H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
hf = scf.RHF(mol).run()
ucc = UCC(hf)
print("\nUCC object created")
print(f"Type: {type(ucc)}")

# List non-private attributes
print("\nAttributes:")
for attr in dir(ucc):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Check for important attributes
for attr in ['h_fermion_op', 'h_qubit', 'n_qubits', 'init_state', 'hf_state', 'get_hf_state']:
    if hasattr(ucc, attr):
        val = getattr(ucc, attr)
        print(f"\n{attr}: {val}")
        if callable(val):
            try:
                result = val()
                print(f"  Callable result: {result}")
            except Exception as e:
                print(f"  Callable error: {e}")

# Check mapping
if hasattr(ucc, 'mapping'):
    print(f"\nmapping: {ucc.mapping}")

# Check if we can get Hamiltonian
if hasattr(ucc, 'h'):
    print(f"\nh shape: {ucc.h.shape}")
if hasattr(ucc, 'h_fermion_op'):
    ferm_op = ucc.h_fermion_op
    print(f"\nFermion operator type: {type(ferm_op)}")
    print(f"Number of terms: {len(ferm_op.terms)}")
    # Try parity transformation
    from tencirchem import parity
    n_modes = 2 * ucc.int1e.shape[0] if hasattr(ucc, 'int1e') else None
    n_elec = mol.nelectron
    print(f"n_modes: {n_modes}, n_elec: {n_elec}")
    if n_modes:
        qubit_op = parity(ferm_op, n_modes=n_modes, n_elec=n_elec)
        print(f"Qubit operator type: {type(qubit_op)}")
        print(f"Number of terms: {len(qubit_op.terms)}")
        # Determine n_qubits
        max_idx = 0
        for term in qubit_op.terms:
            for idx, _ in term:
                if idx > max_idx:
                    max_idx = idx
        print(f"Max qubit index: {max_idx}, n_qubits: {max_idx + 1}")

print("\nDone.")