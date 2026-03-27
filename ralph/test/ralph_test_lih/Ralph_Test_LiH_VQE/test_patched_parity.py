#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
# Patch before importing anything that uses QubitOperator
import patch_openfermion
# Now import other modules
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC, parity

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
fermion_op = ucc.h_fermion_op

n_modes = 6
n_elec = 2
print(f"Applying parity transformation with patched QubitOperator...")
try:
    parity_op = parity(fermion_op, n_modes=n_modes, n_elec=n_elec)
    print(f"Success! Parity operator length: {len(parity_op.terms)}")
    max_idx = 0
    for term in parity_op.terms:
        for idx, _ in term:
            if idx > max_idx:
                max_idx = idx
    n_qubits = max_idx + 1
    print(f"Number of qubits: {n_qubits}")
    const = parity_op.terms.get((), 0)
    print(f"Constant term: {const}")
    # Print first few terms
    count = 0
    for term, coeff in parity_op.terms.items():
        print(f"  {term}: {coeff}")
        count += 1
        if count >= 5:
            break
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()