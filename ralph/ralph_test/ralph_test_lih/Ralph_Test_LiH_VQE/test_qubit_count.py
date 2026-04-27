#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
import numpy as np
from tencirchem import UCC

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)
print(f"Active integrals shape: int1e {int1e.shape}, int2e {int2e.shape}")
print(f"Number of spatial orbitals: {int1e.shape[0]}")
print(f"Active electrons: 2")

# Test default mapping
ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
print(f"\nDefault mapping:")
print(f"  n_qubits: {ucc.n_qubits}")
print(f"  hcb: {getattr(ucc, 'hcb', 'N/A')}")
if hasattr(ucc, 'mapping'):
    print(f"  mapping: {ucc.mapping}")

# Test hcb=True
ucc_hcb = UCC.from_integral(int1e, int2e, 2, e_core=e_core, hcb=True)
print(f"\nHcb=True mapping:")
print(f"  n_qubits: {ucc_hcb.n_qubits}")
print(f"  hcb: {getattr(ucc_hcb, 'hcb', 'N/A')}")

# Test with hcb=False but maybe parity mapping via kwargs
print("\nTrying to find mapping parameter:")
import inspect
sig = inspect.signature(UCC.from_integral)
print(f"  Signature: {sig}")
# Look for mapping in kwargs
if 'kwargs' in sig.parameters:
    print("  **kwargs present")

# Try to pass mapping='parity' via **kwargs
try:
    ucc_parity = UCC.from_integral(int1e, int2e, 2, e_core=e_core, mapping='parity')
    print(f"  mapping='parity' succeeded, n_qubits: {ucc_parity.n_qubits}")
except Exception as e:
    print(f"  mapping='parity' failed: {e}")

# Check if there is a parity transformation function in tencirchem
import tencirchem
if hasattr(tencirchem, 'parity'):
    print(f"\ntenCirchem parity function: {tencirchem.parity}")
    # Maybe it's a transformation function that can be applied to qubit operator
    # Let's try to apply to Hamiltonian
    # Get qubit operator
    h_qubit_op = ucc.h_qubit_op
    print(f"Qubit operator type: {type(h_qubit_op)}")
    # Try to call parity function
    try:
        parity_op = tencirchem.parity(h_qubit_op)
        print(f"Parity transformed operator: {parity_op}")
    except Exception as e:
        print(f"Could not apply parity function: {e}")

# Check if there is a 'parity' method in UCC class
if hasattr(ucc, 'parity'):
    print(f"UCC.parity method: {ucc.parity}")

# Print benchmark expected qubits: 4
print(f"\nBenchmark expects n_qubits_parity = 4")
print(f"Our default mapping gives {ucc.n_qubits} qubits")
print(f"Our hcb mapping gives {ucc_hcb.n_qubits} qubits")

# Compute number of spin orbitals = 2 * spatial = 6
# Jordan-Wigner -> 6 qubits
# Parity -> 5 qubits? Actually parity mapping reduces by 1? Not sure.
# But benchmark says 4 qubits parity. That suggests maybe using hard-core boson mapping? hcb gave 3 qubits.
# Something else.

# Let's examine the Hamiltonian terms to see Pauli weight
print("\nFirst few Hamiltonian terms:")
count = 0
for term, coeff in ucc.h_qubit_op.terms.items():
    print(f"  {term}: {coeff:.6f}")
    count += 1
    if count > 5:
        break