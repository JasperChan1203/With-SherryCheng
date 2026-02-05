#!/usr/bin/env python3
"""
Explore Tencirchem API for manual circuit construction.
"""
import tencirchem
print("Tencirchem version:", tencirchem.__version__)

# Check what's available
import inspect
for name in dir(tencirchem):
    if not name.startswith('_'):
        obj = getattr(tencirchem, name)
        if inspect.isclass(obj):
            print(f"Class: {name}")

# Look for VQE classes
print("\n--- Looking for VQE ---")
from tencirchem import VQE
print(VQE)

# Check if there's a circuit representation
# Let's see example from documentation (maybe online)
# We'll try to create a simple custom circuit.
# First, need Hamiltonian.
# Let's create a simple H2 molecule to test.
from tencirchem import UCC
import numpy as np

# H2 molecule at 0.74 Å
from pyscf import gto, scf
mol = gto.M(
    atom='H 0 0 0; H 0 0 0.74',
    basis='sto-3g',
)
hf = scf.RHF(mol).run()
# Get Hamiltonian using Tencirchem
ucc = UCC(mol)
print("\nUCC object:", ucc)
print("Hamiltonian shape:", ucc.h.shape if hasattr(ucc, 'h') else 'no h')

# Let's see if we can manually construct a circuit with tencirchem's circuit module
try:
    from tencirchem.circuit import Gate, Circuit
    print("Found Gate and Circuit")
except ImportError:
    print("No circuit module")

# Search for circuit submodules
import tencirchem.circuit as tc_circuit
print(dir(tc_circuit))