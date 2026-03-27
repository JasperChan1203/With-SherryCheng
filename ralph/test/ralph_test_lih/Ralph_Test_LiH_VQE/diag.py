#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
from openfermion.linalg import get_sparse_operator

# Patch QubitOperator
import numpy as np
from openfermion.ops.operators.qubit_operator import QubitOperator as OriginalQubitOperator
_original_init = OriginalQubitOperator.__init__
def _patched_init(self, term=None, coefficient=1.0):
    if isinstance(coefficient, np.integer):
        coefficient = int(coefficient)
    elif isinstance(coefficient, np.floating):
        coefficient = float(coefficient)
    elif isinstance(coefficient, np.complexfloating):
        coefficient = complex(coefficient)
    _original_init(self, term, coefficient)
OriginalQubitOperator.__init__ = _patched_init

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

H = get_sparse_operator(h_qubit_op, n_qubits=4).toarray()
print("Diagonal elements:")
for i in range(16):
    bits = format(i, '04b')
    print(f"|{bits}⟩: {H[i,i].real:.8f}")
print("\nLowest diagonal energies:")
diag = [(H[i,i].real, i) for i in range(16)]
diag.sort()
for energy, idx in diag[:5]:
    bits = format(idx, '04b')
    print(f"|{bits}⟩: {energy:.8f}")