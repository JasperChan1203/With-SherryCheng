#!/usr/bin/env python3
"""Explore tencirchem UCC for circuit building."""

import sys
sys.path.append("../001")
sys.path.append("../002")

import tencirchem
from tencirchem import UCC
from pyscf import gto, scf
import numpy as np

print("Tencirchem version:", tencirchem.__version__)

# Create H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
hf = scf.RHF(mol).run()

# Create UCC instance
ucc = UCC(mol)
print(f"UCC created: {ucc}")
print(f"n_qubits: {ucc.n_qubits}")
print(f"n_elec: {ucc.n_elec}")
print(f"hf_energy: {ucc.hf_energy}")
print(f"Parameters: {ucc.init_guess}")

# Check attributes
print("\nAttributes containing 'exc':")
for attr in dir(ucc):
    if 'exc' in attr.lower():
        print(f"  {attr}: {getattr(ucc, attr)}")

print("\nMethods containing 'exc':")
for attr in dir(ucc):
    if 'exc' in attr.lower() and callable(getattr(ucc, attr)):
        print(f"  {attr}")

# Check if there is excitation list
if hasattr(ucc, 'exc_pairs'):
    print(f"\nucc.exc_pairs: {ucc.exc_pairs}")
    print(f"Type: {type(ucc.exc_pairs)}")
    if ucc.exc_pairs is not None:
        print(f"Length: {len(ucc.exc_pairs)}")
        print(f"First few: {ucc.exc_pairs[:5]}")

if hasattr(ucc, 'get_excitation_ops'):
    ops = ucc.get_excitation_ops()
    print(f"\nExcitation ops: {ops}")

# Try to build circuit with specific excitations
print("\n--- Testing circuit building ---")
# Get default excitations (maybe all singles and doubles)
# UCCSD includes all possible excitations within active space
# We want to select a subset.
# Let's see if we can set ex_ops attribute
if hasattr(ucc, 'ex_ops'):
    print(f"ex_ops: {ucc.ex_ops}")
    print(f"Type: {type(ucc.ex_ops)}")

# Try to create a new UCC with custom ex_ops
# According to tencirchem documentation, UCC accepts ex_ops parameter
# Let's try to pass a subset of excitation pairs
if hasattr(ucc, 'exc_pairs') and ucc.exc_pairs is not None:
    # Take first two excitation pairs
    subset = ucc.exc_pairs[:2]
    print(f"\nCreating UCC with subset of excitations: {subset}")
    try:
        ucc2 = UCC(mol, ex_ops=subset)
        print(f"Success: n_params = {ucc2.n_params}")
    except Exception as e:
        print(f"Error: {e}")

# Check circuit property
if hasattr(ucc, 'circuit'):
    print(f"\nCircuit type: {type(ucc.circuit)}")
    # Try to evaluate energy
    if hasattr(ucc, 'energy'):
        energy = ucc.energy(ucc.init_guess)
        print(f"Energy with init guess: {energy}")

print("\n--- Done ---")