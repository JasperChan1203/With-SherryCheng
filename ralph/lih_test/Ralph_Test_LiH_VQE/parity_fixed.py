#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC
from openfermion import FermionOperator
from openfermion.transforms import parity_code
from openfermion.transforms.opconversions.binary_code_transform import binary_code_transform
import numpy as np

# Monkey-patch the problematic line
import openfermion.transforms.opconversions.binary_code_transform as bct
original_binary_code_transform = bct.binary_code_transform

def patched_binary_code_transform(fermion_operator, code):
    # We'll copy the function but replace the line where (-1)**updated_parity
    # Instead we'll call original but ensure updated_parity is int.
    # Let's import the needed internal functions
    from openfermion.transforms.opconversions.binary_code_transform import (
        _convert_fermion_operator_to_binary_code, _binary_code_transform)
    # Actually the function is _binary_code_transform?
    # Let's just copy the source from openfermion (simplified)
    # This is messy.
    pass

# Instead, let's use openfermion's jordan_wigner then parity reduction?
from openfermion.transforms import jordan_wigner
jw_op = jordan_wigner(fermion_op)
print(f"JW operator qubits: {max(idx for term in jw_op.terms for idx, _ in term) + 1 if jw_op.terms else 0}")

# But we need parity mapping directly.
# Let's try to call tencirchem.parity with converted fermion operator where coefficients are python float.
# The error is inside binary_code_transform; we can try to convert updated_parity before exponentiation by patching.
# Let's write a custom binary_code_transform that replicates the original but with type conversion.
# I'll copy the source from openfermion (approx).
# Let's search for source file location.
import inspect
print(inspect.getfile(bct))

# Let's read the source and modify.
with open(inspect.getfile(bct), 'r') as f:
    source = f.read()
# Too heavy.

# Let's try a different approach: use openfermion's `parity` function from openfermion.transforms?
# It seems not present.
# Let's implement parity transformation manually using binary codes.
# Use parity_code matrix to map fermion operators to qubit operators.
# We'll implement using openfermion's `binary_code_transform` but we need to fix the coefficient type.
# Let's wrap the function and intercept the problematic term.
def safe_binary_code_transform(fermion_operator, code):
    from openfermion.ops import QubitOperator
    from openfermion.transforms.opconversions.binary_code_transform import _convert_fermion_operator_to_binary_code
    # Convert fermion operator to binary representation
    binary_operator = _convert_fermion_operator_to_binary_code(fermion_operator, code)
    qubit_operator = QubitOperator()
    for binary_term, coefficient in binary_operator.terms.items():
        # binary_term is tuple of (index, bit)
        # Convert to Pauli operators
        # This is complex. Let's just call original and catch exception, then fix.
        pass

# Let's just try to convert all numpy ints in the fermion operator terms to int.
# Actually the issue is in binary_code_transform internal variable updated_parity which is numpy.int64.
# We can monkey-patch the pow function? Not.
# Let's see if we can convert all numpy ints in the fermion operator to int.
# The fermion operator uses tuples of (index, action) where index is int.
# That's fine.

# Let's try to use tencirchem.parity with a different n_elec? Maybe n_elec must be total electrons (including frozen)?
# Try n_elec = 4 (total electrons)
print("\nTrying parity with n_elec = 4")
from tencirchem import parity
try:
    parity_op = parity(fermion_op, n_modes=6, n_elec=4)
    print(f"Success with n_elec=4")
except Exception as e:
    print(f"Error: {e}")

# Try n_elec = 2 but convert fermion operator coefficients to complex?
# Let's convert coefficients to Python complex.
fermion_op3 = FermionOperator()
for term, coeff in fermion_op.terms.items():
    fermion_op3 += FermionOperator(term, complex(coeff))
print(f"Converted to complex coefficients.")
try:
    parity_op = parity(fermion_op3, n_modes=6, n_elec=2)
    print(f"Success with complex coefficients")
except Exception as e:
    print(f"Error: {e}")