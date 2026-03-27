#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC
from openfermion.transforms.opconversions.binary_code_transform import binary_code_transform
from openfermion.transforms.opconversions.parity_code import parity_code
import numpy as np

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
fermion_op = ucc.h_fermion_op

# Convert coefficients to Python float
from openfermion import FermionOperator
new_terms = {}
for term, coeff in fermion_op.terms.items():
    new_terms[term] = float(coeff)
fermion_op2 = FermionOperator()
for term, coeff in new_terms.items():
    fermion_op2 += FermionOperator(term, coeff)

n_modes = 6
# Get parity code
code = parity_code(n_modes)
print(f"Parity code shape: {code.shape}")
print(f"Code: {code}")

# Apply binary code transform
try:
    parity_op = binary_code_transform(fermion_op2, code)
    print(f"Success! Parity operator length: {len(parity_op.terms)}")
    # Determine qubit count
    max_idx = 0
    for term in parity_op.terms:
        for idx, _ in term:
            if idx > max_idx:
                max_idx = idx
    n_qubits = max_idx + 1
    print(f"Number of qubits: {n_qubits}")
    # Print constant term
    const = parity_op.terms.get((), 0)
    print(f"Constant term: {const}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    # Try to convert updated_parity to int before raising?
    # Let's monkey-patch the function
    import openfermion.transforms.opconversions.binary_code_transform as bct
    original = bct.binary_code_transform
    def patched_transform(fermion_operator, code):
        # Convert code to int? Not needed.
        # We'll patch inside the function where (-1)**updated_parity
        # This is hacky.
        pass