#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
import tensorcircuit as tc

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

# Determine number of qubits
max_idx = 0
for term in h_qubit_op.terms:
    for idx, _ in term:
        if idx > max_idx:
            max_idx = idx
n_qubits = max_idx + 1
print(f"n_qubits = {n_qubits}")

def energy_function(params, h_qubit_op, n_qubits):
    """Compute expectation value of Hamiltonian for given parameters."""
    c = tc.Circuit(n_qubits)
    # Prepare HF state |0011⟩ (qubits 2,3 in |1⟩)
    c.x(2)
    c.x(3)
    # Double excitation block: exp(iθ XXXX)
    for i in range(n_qubits):
        c.h(i)
    for i in range(n_qubits - 1):
        c.cnot(i, i+1)
    c.rz(n_qubits - 1, theta=params[0])
    for i in range(n_qubits - 2, -1, -1):
        c.cnot(i, i+1)
    for i in range(n_qubits):
        c.h(i)
    # Compute expectation
    energy = 0.0
    for term, coeff in h_qubit_op.terms.items():
        x_list = []
        y_list = []
        z_list = []
        for idx, pauli in term:
            if pauli == 'X':
                x_list.append(idx)
            elif pauli == 'Y':
                y_list.append(idx)
            elif pauli == 'Z':
                z_list.append(idx)
            else:
                raise ValueError(f"Unknown Pauli {pauli}")
        exp_val = c.expectation_ps(x=x_list, y=y_list, z=z_list)
        energy += coeff * exp_val
    return energy.real

# Scan theta from -0.5 to 0.5
thetas = np.linspace(-0.5, 0.5, 101)
energies = []
for theta in thetas:
    e = energy_function([theta], h_qubit_op, n_qubits)
    energies.append(e)
    print(f"theta={theta:.3f}, energy={e:.8f}")

# Find minimum
min_idx = np.argmin(energies)
print(f"\nMinimum at theta={thetas[min_idx]:.6f}, energy={energies[min_idx]:.8f}")
print(f"HF energy (theta=0): {energies[50]:.8f}")
print(f"Difference: {energies[min_idx] - energies[50]:.8f}")

# Compute exact ground state energy via diagonalization
from openfermion.linalg import get_sparse_operator
H = get_sparse_operator(h_qubit_op, n_qubits=n_qubits).toarray()
eigvals = np.linalg.eigvalsh(H)
print(f"\nExact ground state energy: {eigvals[0]:.8f}")
print(f"Correlation energy: {eigvals[0] - energies[50]:.8f}")