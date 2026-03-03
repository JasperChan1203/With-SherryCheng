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
print(f"n_params: {ucc.n_params}")

# Get excitation operators
print("\n--- Excitation operators ---")
if hasattr(ucc, 'get_ex1_ops'):
    ex1 = ucc.get_ex1_ops()
    print(f"Single excitations: {ex1}")
    print(f"Type: {type(ex1)}")
    if isinstance(ex1, list) and len(ex1) > 0:
        print(f"First single excitation: {ex1[0]}")

if hasattr(ucc, 'get_ex2_ops'):
    ex2 = ucc.get_ex2_ops()
    print(f"Double excitations: {ex2}")
    print(f"Type: {type(ex2)}")
    if isinstance(ex2, list) and len(ex2) > 0:
        print(f"First double excitation: {ex2[0]}")

if hasattr(ucc, 'get_ex_ops'):
    ex_ops = ucc.get_ex_ops()
    print(f"All excitation ops: {ex_ops}")
    print(f"Type: {type(ex_ops)}")
    if isinstance(ex_ops, list) and len(ex_ops) > 0:
        print(f"First ex_op: {ex_ops[0]}")

if hasattr(ucc, 'exc_pairs'):
    print(f"exc_pairs: {ucc.exc_pairs}")
    if ucc.exc_pairs is not None:
        print(f"Number of exc_pairs: {len(ucc.exc_pairs)}")

# Check param_to_ex_ops
if hasattr(ucc, 'param_to_ex_ops'):
    print(f"\nparam_to_ex_ops: {ucc.param_to_ex_ops}")
    if ucc.param_to_ex_ops is not None:
        print(f"Mapping shape: {len(ucc.param_to_ex_ops)}")
        print(f"First mapping: {ucc.param_to_ex_ops[0]}")

# Test building circuit with subset of excitations
print("\n--- Building circuit with subset ---")
if hasattr(ucc, 'get_ex_ops'):
    all_ops = ucc.get_ex_ops()
    if isinstance(all_ops, list) and len(all_ops) >= 2:
        subset = all_ops[:2]
        print(f"Using subset: {subset}")
        # Create new UCC with custom ex_ops
        try:
            ucc_subset = UCC(mol, ex_ops=subset)
            print(f"Created UCC with n_params = {ucc_subset.n_params}")
            # Get circuit
            if hasattr(ucc_subset, 'get_circuit'):
                circuit = ucc_subset.get_circuit()
                print(f"Circuit type: {type(circuit)}")
                # Try to evaluate energy with random parameters
                params = np.random.randn(ucc_subset.n_params) * 0.1
                energy = ucc_subset.energy(params)
                print(f"Energy with random params: {energy}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

# Check if we can build circuit incrementally (add excitation operators)
print("\n--- Incremental circuit building ---")
# Idea: create UCC with empty ex_ops, then add?
try:
    ucc_empty = UCC(mol, ex_ops=[])
    print(f"UCC empty n_params: {ucc_empty.n_params}")
except Exception as e:
    print(f"Cannot create empty ex_ops: {e}")

# Check if we can update ex_ops after creation
if hasattr(ucc, 'ex_ops'):
    print(f"Original ex_ops attribute: {ucc.ex_ops}")
    # Try to modify
    try:
        ucc.ex_ops = subset
        print(f"Modified ex_ops to subset")
    except Exception as e:
        print(f"Cannot modify ex_ops: {e}")

print("\n--- Done ---")