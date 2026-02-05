#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC
from openfermion import FermionOperator
from openfermion.transforms import parity_code, binary_code_transform
import numpy as np

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
fermion_op = ucc.h_fermion_op
print(f"Original fermion operator constant: {fermion_op.terms.get((), 0)}")

# Convert coefficients to Python float (should be safe)
new_terms = {}
for term, coeff in fermion_op.terms.items():
    new_terms[term] = float(coeff)
fermion_op_float = FermionOperator()
for term, coeff in new_terms.items():
    fermion_op_float += FermionOperator(term, coeff)

n_modes = 6
n_elec = 2

# Use parity_code and binary_code_transform
code = parity_code(n_modes)
print(f"Parity code shape: {code[0].shape}")

# Monkey-patch the problematic line in binary_code_transform
import openfermion.transforms.opconversions.binary_code_transform as bct
original = bct.binary_code_transform

# We'll create a wrapper that converts numpy ints to Python ints
def patched_binary_code_transform(fermion_operator, code):
    # Convert fermion operator coefficients to Python float
    from openfermion import FermionOperator
    from openfermion.transforms.opconversions.binary_code_transform import _convert_fermion_operator_to_binary_code
    from openfermion.ops import QubitOperator
    # Convert to binary operator
    binary_operator = _convert_fermion_operator_to_binary_code(fermion_operator, code)
    qubit_operator = QubitOperator()
    for binary_term, coefficient in binary_operator.terms.items():
        # binary_term is tuple of (index, bit)
        # Build Pauli string
        pauli_term = []
        for idx, bit in binary_term:
            if bit == 0:
                pauli_term.append((idx, 'Z'))
            else:
                pauli_term.append((idx, 'X'))
        # coefficient is numeric (could be numpy)
        # Convert to Python complex
        coeff = complex(coefficient)
        qubit_operator += QubitOperator(tuple(pauli_term), coeff)
    return qubit_operator

# Replace function temporarily
bct.binary_code_transform = patched_binary_code_transform
try:
    parity_op = binary_code_transform(fermion_op_float, code)
    print(f"Parity operator length: {len(parity_op.terms)}")
    # Determine qubit count
    max_idx = 0
    for term in parity_op.terms:
        for idx, _ in term:
            if idx > max_idx:
                max_idx = idx
    n_qubits = max_idx + 1
    print(f"Number of qubits: {n_qubits}")
    const = parity_op.terms.get((), 0)
    print(f"Constant term: {const}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    bct.binary_code_transform = original

# If successful, we can now use parity_op as Hamiltonian for VQE.
# Let's also check that constant term matches expectation (should be close to core energy?)
print(f"\nCore energy e_core: {e_core}")