#!/usr/bin/env python3

import sys
sys.path.append("../001")
sys.path.append("../002")

from tencirchem import UCCSD
from pyscf import gto, scf

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
hf = scf.RHF(mol).run()
ucc = UCCSD(mol)

print(f"n_params: {ucc.n_params}")
print(f"ex_ops: {ucc.ex_ops}")
print(f"len(ex_ops): {len(ucc.ex_ops)}")

if hasattr(ucc, 'param_ids'):
    print(f"param_ids: {ucc.param_ids}")
if hasattr(ucc, 'param_to_ex_ops'):
    print(f"param_to_ex_ops: {ucc.param_to_ex_ops}")
    if ucc.param_to_ex_ops is not None:
        for i, mapping in enumerate(ucc.param_to_ex_ops):
            print(f"  param {i} -> ex_op {mapping}")

# Check get_ex_ops output
if hasattr(ucc, 'get_ex_ops'):
    ops, types, guesses = ucc.get_ex_ops()
    print(f"get_ex_ops:")
    print(f"  ops: {ops}")
    print(f"  types: {types}")
    print(f"  guesses: {guesses}")
    print(f"  length: {len(ops)}")

# Evaluate energy with parameter vector
import numpy as np
params = np.zeros(ucc.n_params)
print(f"\nZero params energy: {ucc.energy(params)}")
params2 = np.random.randn(ucc.n_params) * 0.1
print(f"Random params energy: {ucc.energy(params2)}")

# Check if we can set parameters for specific excitation only
# Map excitation index to parameter index
# param_to_ex_ops maps param index to ex_op index? Let's see.
# If we want to activate only ex_op 0, which param index?
# Let's assume param_to_ex_ops[0] = 0? Let's print.
print("\nMapping:")
for param_idx, ex_op_idx in enumerate(ucc.param_to_ex_ops):
    print(f"Parameter {param_idx} maps to excitation operator {ex_op_idx}: {ucc.ex_ops[ex_op_idx]}")