#!/usr/bin/env python3
import numpy as np
from tencirchem import UCC

int1e = np.array([[0, 0.1], [0.1, 0]])
int2e = np.zeros((2,2,2,2))
n_elec = 2

print("Testing mapping parameter:")
try:
    ucc = UCC.from_integral(int1e, int2e, n_elec, mapping='parity')
    print(f"  Success, n_qubits: {ucc.n_qubits}")
except Exception as e:
    print(f"  Error: {e}")

print("\nTesting mapping='jordan-wigner':")
try:
    ucc = UCC.from_integral(int1e, int2e, n_elec, mapping='jordan-wigner')
    print(f"  Success, n_qubits: {ucc.n_qubits}")
except Exception as e:
    print(f"  Error: {e}")

print("\nTesting mapping='bravyi-kitaev':")
try:
    ucc = UCC.from_integral(int1e, int2e, n_elec, mapping='bravyi-kitaev')
    print(f"  Success, n_qubits: {ucc.n_qubits}")
except Exception as e:
    print(f"  Error: {e}")

# Check what mapping attribute exists
print("\nInspecting UCC object attributes:")
ucc_default = UCC.from_integral(int1e, int2e, n_elec)
for attr in dir(ucc_default):
    if not attr.startswith('_'):
        val = getattr(ucc_default, attr)
        if isinstance(val, str) and 'map' in attr.lower():
            print(f"  {attr}: {val}")

# Look for mapping property
if hasattr(ucc_default, 'mapping'):
    print(f"  mapping: {ucc_default.mapping}")
if hasattr(ucc_default, 'hcb'):
    print(f"  hcb: {ucc_default.hcb}")

# Check parity function
from tencirchem import parity
print(f"\nparity function: {parity}")
import inspect
try:
    sig = inspect.signature(parity)
    print(f"  signature: {sig}")
except:
    pass