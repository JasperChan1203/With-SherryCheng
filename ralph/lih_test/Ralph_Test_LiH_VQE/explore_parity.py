#!/usr/bin/env python3
import tencirchem
from tencirchem import parity
import inspect
print("parity function signature:", inspect.signature(parity))
print("parity docstring:", parity.__doc__[:200] if parity.__doc__ else None)

# Try to call parity with sample qubit operator
from openfermion import QubitOperator
op = QubitOperator('X0 Y1', 1.0)
print(f"\nSample operator: {op}")
# Need n_modes and n_elec
# n_modes likely number of spin orbitals? n_elec number of electrons
# Let's guess n_modes = 6 (spin orbitals), n_elec = 2
try:
    op_parity = parity(op, n_modes=6, n_elec=2)
    print(f"Parity transformed: {op_parity}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Check if there is a 'parity' module
import tencirchem.parity as parity_mod
print(f"\nparity module contents: {dir(parity_mod)}")

# Look for other mapping functions
for name in dir(tencirchem):
    if 'map' in name.lower() or 'transform' in name.lower():
        print(f"{name}: {getattr(tencirchem, name)}")

# Search for 'parity' in tencirchem source using inspect
import tencirchem.ucc
print("\nUCC module attributes containing parity:")
for attr in dir(tencirchem.ucc):
    if 'par' in attr.lower():
        print(f"  {attr}")