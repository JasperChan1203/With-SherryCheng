#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC
from openfermion import FermionOperator
import numpy as np

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
fermion_op = ucc.h_fermion_op
print(f"Type of fermion_op: {type(fermion_op)}")
print(f"Number of terms: {len(fermion_op.terms)}")
# Inspect first term coefficient type
for term, coeff in fermion_op.terms.items():
    print(f"Term {term}, coeff type {type(coeff)}, value {coeff}")
    break

# Try to convert coefficients to Python float
new_terms = {}
for term, coeff in fermion_op.terms.items():
    new_terms[term] = float(coeff)
fermion_op2 = FermionOperator()
for term, coeff in new_terms.items():
    fermion_op2 += FermionOperator(term, coeff)
print(f"\nConverted fermion operator constant term: {fermion_op2.terms.get((), 0)}")

# Apply parity transformation
from tencirchem import parity
n_modes = 6
n_elec = 2
try:
    parity_op = parity(fermion_op2, n_modes=n_modes, n_elec=n_elec)
    print(f"Parity operator obtained")
except Exception as e:
    print(f"Error after conversion: {e}")
    import traceback
    traceback.print_exc()

# Try openfermion's parity transformation
from openfermion.transforms import parity as parity_of
try:
    parity_op2 = parity_of(fermion_op2, n_modes, n_elec)
    print(f"OpenFermion parity operator length: {len(parity_op2.terms)}")
except Exception as e:
    print(f"OpenFermion parity error: {e}")