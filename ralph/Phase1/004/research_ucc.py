#!/usr/bin/env python3
"""Research tencirchem UCC API for circuit building."""

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
print(f"HF energy: {hf.e_tot}")

# Create UCC instance without ex_ops
print("\n1. Creating UCC without ex_ops")
ucc = UCC(mol)
print(f"UCC object: {ucc}")

# Check if kernel can be called
print("\n2. Calling kernel()")
try:
    result = ucc.kernel()
    print(f"Kernel result: {result}")
except Exception as e:
    print(f"Kernel error: {e}")
    import traceback
    traceback.print_exc()

# Now check attributes
print("\n3. After kernel attempt")
for attr in ['n_params', 'exc_pairs', 'ex_ops', 'param_ids', 'get_ex_ops']:
    if hasattr(ucc, attr):
        try:
            val = getattr(ucc, attr)
            if callable(val):
                val = val()
                print(f"{attr}() = {val}")
            else:
                print(f"{attr} = {val}")
        except Exception as e:
            print(f"{attr} error: {e}")

# Try to get ex_ops via get_ex_ops
print("\n4. Getting excitation operators")
if hasattr(ucc, 'get_ex_ops'):
    try:
        ex_ops = ucc.get_ex_ops()
        print(f"ex_ops type: {type(ex_ops)}")
        if isinstance(ex_ops, list):
            print(f"Number of ex_ops: {len(ex_ops)}")
            if len(ex_ops) > 0:
                print(f"First ex_op: {ex_ops[0]}")
                print(f"Structure: {ex_ops[0]}")
                # If tuple, print each element
                if isinstance(ex_ops[0], tuple):
                    for i, item in enumerate(ex_ops[0]):
                        print(f"  [{i}] {item} type {type(item)}")
    except Exception as e:
        print(f"get_ex_ops error: {e}")

# Try to create UCC with custom ex_ops (subset)
print("\n5. Creating UCC with subset of ex_ops")
if hasattr(ucc, 'get_ex_ops'):
    try:
        all_ops = ucc.get_ex_ops()
        if isinstance(all_ops, list) and len(all_ops) >= 2:
            subset = all_ops[:2]
            print(f"Subset: {subset}")
            ucc2 = UCC(mol, ex_ops=subset)
            print(f"UCC2 created, n_params = {ucc2.n_params}")
            # Try to get circuit
            if hasattr(ucc2, 'get_circuit'):
                circuit = ucc2.get_circuit()
                print(f"Circuit type: {type(circuit)}")
                # Evaluate energy with zero parameters
                params = np.zeros(ucc2.n_params)
                energy = ucc2.energy(params)
                print(f"Energy with zero params: {energy}")
                # Random params
                params_rand = np.random.randn(ucc2.n_params) * 0.1
                energy_rand = ucc2.energy(params_rand)
                print(f"Energy with random params: {energy_rand}")
    except Exception as e:
        print(f"Error creating subset UCC: {e}")
        import traceback
        traceback.print_exc()

# Check mapping between excitation tuples and ex_ops
print("\n6. Excitation pairs")
if hasattr(ucc, 'exc_pairs'):
    try:
        exc_pairs = ucc.exc_pairs
        print(f"exc_pairs type: {type(exc_pairs)}")
        if isinstance(exc_pairs, list) and len(exc_pairs) > 0:
            print(f"First exc_pair: {exc_pairs[0]}")
    except Exception as e:
        print(f"exc_pairs error: {e}")

# Check param_to_ex_ops
print("\n7. param_to_ex_ops")
if hasattr(ucc, 'param_to_ex_ops'):
    try:
        mapping = ucc.param_to_ex_ops
        print(f"param_to_ex_ops type: {type(mapping)}")
        if isinstance(mapping, list) and len(mapping) > 0:
            print(f"First mapping: {mapping[0]}")
            print(f"Length: {len(mapping)}")
    except Exception as e:
        print(f"param_to_ex_ops error: {e}")

print("\n--- Research complete ---")