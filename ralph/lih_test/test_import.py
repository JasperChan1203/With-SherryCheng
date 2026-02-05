#!/usr/bin/env python3
"""
Test script to check tencirchem imports and dependencies
"""

import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import qiskit
    print(f"qiskit version: {qiskit.__version__}")
    from qiskit.quantum_info import SparsePauliOp
    print("✓ SparsePauliOp import successful")
except ImportError as e:
    print(f"✗ qiskit import error: {e}")

try:
    import tencirchem
    print(f"tencirchem version: {tencirchem.__version__}")
    from tencirchem import UCC, parity
    print("✓ tencirchem import successful")
except ImportError as e:
    print(f"✗ tencirchem import error: {e}")

try:
    import pyscf
    print(f"pyscf version: {pyscf.__version__}")
except ImportError as e:
    print(f"✗ pyscf import error: {e}")

try:
    import openfermion
    print(f"openfermion version: openfermion.__version__")
except ImportError as e:
    print(f"✗ openfermion import error: {e}")

# Test actual parity transformation
print("\n--- Testing parity transformation ---")
try:
    from openfermion import FermionOperator
    # Create a simple fermionic operator
    fo = FermionOperator("0^ 0", 1.0)
    print(f"Created FermionOperator: {fo}")
    n_modes = 2
    n_elec = 1
    qo = parity(fo, n_modes=n_modes, n_elec=n_elec)
    print(f"✓ Parity transformation successful: {qo}")
except Exception as e:
    print(f"✗ Parity transformation failed: {e}")
    import traceback
    traceback.print_exc()