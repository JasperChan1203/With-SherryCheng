#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC, binary
from openfermion import FermionOperator

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)
ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
fermion_op = ucc.h_fermion_op
print(f"Fermion operator constant: {fermion_op.terms.get((), 0)}")

n_modes = 6
n_elec = 2
print(f"Trying binary transformation...")
try:
    bin_op = binary(fermion_op, n_modes=n_modes, n_elec=n_elec)
    print(f"Binary operator length: {len(bin_op.terms)}")
    max_idx = 0
    for term in bin_op.terms:
        for idx, _ in term:
            if idx > max_idx:
                max_idx = idx
    print(f"Number of qubits: {max_idx+1}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()