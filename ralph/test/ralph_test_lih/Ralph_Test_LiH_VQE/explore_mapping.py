#!/usr/bin/env python3
import tencirchem
from tencirchem import UCC
import inspect

print("UCC class signature:")
print(inspect.signature(UCC.__init__))
print("\nUCC.from_integral signature:")
try:
    print(inspect.signature(UCC.from_integral))
except:
    pass

# Check attributes of UCC class
print("\nUCC attributes containing 'map':")
for attr in dir(UCC):
    if 'map' in attr.lower():
        print(f"  {attr}")

# Create a dummy UCC object with simple integrals to inspect
import numpy as np
int1e = np.array([[0, 0.1], [0.1, 0]])
int2e = np.zeros((2,2,2,2))
n_elec = 2
print("\nTrying UCC.from_integral with default parameters:")
ucc = UCC.from_integral(int1e, int2e, n_elec)
print(f"  n_qubits: {ucc.n_qubits}")
print(f"  mapping: {getattr(ucc, 'mapping', 'not found')}")
print(f"  hcb: {getattr(ucc, 'hcb', 'not found')}")

print("\nTrying with hcb=True:")
ucc2 = UCC.from_integral(int1e, int2e, n_elec, hcb=True)
print(f"  n_qubits: {ucc2.n_qubits}")
print(f"  mapping: {getattr(ucc2, 'mapping', 'not found')}")

# Look for mapping parameter
print("\nChecking UCC.from_integral source:")
try:
    import inspect
    source = inspect.getsource(UCC.from_integral)
    lines = source.split('\n')
    for i, line in enumerate(lines[:30]):
        if 'mapping' in line:
            print(f"  {i}: {line}")
except:
    print("  Could not get source")

# Check if there is a parity mapping constant
print("\nSearching for parity in tencirchem constants:")
for name in dir(tencirchem):
    if 'par' in name.lower():
        print(f"  {name}: {getattr(tencirchem, name)}")