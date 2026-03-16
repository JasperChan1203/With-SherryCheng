#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import patch_openfermion

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
from openfermion import QubitOperator

bond_length = 2.0
mol = gto.M(
    atom=[["H", 0, 0, 0], ["Li", bond_length, 0, 0]],
    basis='sto-3g',
    symmetry=True,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
cas = mcscf.CASCI(hf, 3, 2)
mo_coeff = cas.sort_mo([2, 3, 6])
cas.kernel(mo_coeff)
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
n_orb = int1e.shape[0]
int2e = ao2mo.restore(1, int2e, n_orb)
ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core, hcb=False)
fermion_op = ucc.h_fermion_op
n_modes = 2 * int1e.shape[0]
h_qubit_op = parity(fermion_op, n_modes=n_modes, n_elec=2)

print(f"Total terms: {len(h_qubit_op.terms)}")
print("Constant term:", h_qubit_op.terms.get((), 0))

# Group terms by number of qubits
term_by_len = {1:[],2:[],3:[],4:[]}
for term, coeff in h_qubit_op.terms.items():
    if term == ():
        continue
    l = len(term)
    if l in term_by_len:
        term_by_len[l].append((term, coeff))
    else:
        term_by_len[l] = [(term, coeff)]

for l in range(1,5):
    terms = term_by_len[l]
    print(f"\nLength {l} terms: {len(terms)}")
    # sort by absolute coefficient
    terms_sorted = sorted(terms, key=lambda x: abs(x[1]), reverse=True)
    for term, coeff in terms_sorted[:10]:
        print(f"  {term}: {coeff}")
    if len(terms) > 10:
        print(f"  ... and {len(terms)-10} more")

# Also list terms with X or Y (off-diagonal)
print("\nOff-diagonal terms (contain X or Y):")
off_terms = []
for term, coeff in h_qubit_op.terms.items():
    if term == ():
        continue
    has_x_or_y = any(p in ('X','Y') for _, p in term)
    if has_x_or_y:
        off_terms.append((term, coeff))
off_sorted = sorted(off_terms, key=lambda x: abs(x[1]), reverse=True)
for term, coeff in off_sorted[:20]:
    print(f"  {term}: {coeff}")

# Find four-qubit terms that are off-diagonal (likely double excitation)
print("\nFour-qubit off-diagonal terms:")
four_off = [(t,c) for t,c in off_terms if len(t)==4]
for term, coeff in four_off:
    print(f"  {term}: {coeff}")