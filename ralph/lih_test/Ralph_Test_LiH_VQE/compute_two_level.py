#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity

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

# Build matrix for two states |0011⟩ and |1100⟩
def state_index(bits):
    # bits string like '0011'
    idx = 0
    for i, b in enumerate(bits):
        if b == '1':
            idx |= 1 << i
    return idx

idx0011 = state_index('0011')
idx1100 = state_index('1100')
print(f"Index of |0011⟩: {idx0011}")
print(f"Index of |1100⟩: {idx1100}")

# Compute matrix elements using Pauli expansion
def matrix_element(i, j):
    # Compute <i|H|j> using Pauli operators
    # For simplicity, construct full Hamiltonian matrix (small)
    from openfermion.linalg import get_sparse_operator
    H = get_sparse_operator(h_qubit_op, n_qubits=4).toarray()
    return H[i, j]

H = np.zeros((2,2), dtype=complex)
H[0,0] = matrix_element(idx0011, idx0011)
H[1,1] = matrix_element(idx1100, idx1100)
H[0,1] = matrix_element(idx0011, idx1100)
H[1,0] = H[0,1].conjugate()
print(f"H = [[{H[0,0].real:.8f}, {H[0,1].real:.8f}],")
print(f"     [{H[1,0].real:.8f}, {H[1,1].real:.8f}]]")

# Diagonalize two-level system
eigvals, eigvecs = np.linalg.eigh(H)
print(f"Eigenvalues: {eigvals[0].real:.8f}, {eigvals[1].real:.8f}")
print(f"Eigenvectors:")
print(f"  Ground: {eigvecs[:,0]}")
print(f"  Excited: {eigvecs[:,1]}")

# Compute mixing angle such that ground state = cos(θ)|0011⟩ + sin(θ)|1100⟩
# Assuming real coefficients
c1, c2 = eigvecs[0,0].real, eigvecs[1,0].real
theta = np.arctan2(c2, c1)
print(f"Mixing angle θ = {theta:.6f} rad")
print(f"cos(θ) = {c1:.6f}, sin(θ) = {c2:.6f}")

# Off-diagonal element V = H[0,1]
V = H[0,1].real
Δ = H[1,1].real - H[0,0].real
print(f"Δ = E_1100 - E_0011 = {Δ:.8f}")
print(f"V = {V:.8f}")
print(f"Ratio V/Δ = {V/Δ:.8f}")

# Exact ground state energy from full diagonalization
from openfermion.linalg import get_sparse_operator
H_full = get_sparse_operator(h_qubit_op, n_qubits=4).toarray()
eigvals_full = np.linalg.eigvalsh(H_full)
print(f"\nFull ground state energy: {eigvals_full[0]:.8f}")
print(f"Two-level approximation energy: {eigvals[0]:.8f}")
print(f"Difference: {eigvals_full[0] - eigvals[0]:.8f}")