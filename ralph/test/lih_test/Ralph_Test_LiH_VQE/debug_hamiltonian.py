#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
from openfermion import QubitOperator
import tensorcircuit as tc

# Patch QubitOperator to accept numpy numeric types
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
print(f"HF energy: {hf.e_tot:.8f}")

# Select active orbitals via sort_mo
cas = mcscf.CASCI(hf, 3, 2)
mo_coeff = cas.sort_mo([2, 3, 6])
cas.kernel(mo_coeff)
active_orbitals = [1, 2, 5]
print(f"Active orbitals (0-based): {active_orbitals}")

# Get integrals
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
from pyscf import ao2mo
n_orb = int1e.shape[0]
int2e = ao2mo.restore(1, int2e, n_orb)

# Build Hamiltonian
ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core, hcb=False)
fermion_op = ucc.h_fermion_op
n_modes = 2 * int1e.shape[0]
h_qubit_op = parity(fermion_op, n_modes=n_modes, n_elec=2)
print(f"Number of qubits: {max(idx for term in h_qubit_op.terms for idx,_ in term) + 1}")

# Print some terms
print("\nFirst 10 Hamiltonian terms:")
count = 0
for term, coeff in h_qubit_op.terms.items():
    print(f"  {term}: {coeff}")
    count += 1
    if count >= 10:
        break

# Compute Hartree-Fock state expectation manually
# Hartree-Fock state in parity mapping? Let's compute expectation using tensorcircuit
# First, need to know the HF state in qubit basis.
# Use tencirchem to get HF state? We can compute using UCC's get_init_state method
init_state = ucc.get_init_state()
print(f"\nInitial state (occupation vector): {init_state}")
# This is occupation vector in spin orbitals (length 6). Let's map to qubits.
# For parity mapping, the state is something like |1100> maybe.
# Let's compute expectation using tencirchem's energy evaluation
from tencirchem import get_ps
from tencirchem.ci import get_hf_energy
hf_energy_active = get_hf_energy(int1e, int2e, 2)
print(f"HF energy from active integrals: {hf_energy_active}")
print(f"Total HF energy (core + active): {e_core + hf_energy_active}")

# Build a simple circuit that prepares HF state and compute expectation
# We'll use tensorcircuit to compute expectation of Hamiltonian
def expectation_hf_state():
    # Create circuit that prepares HF state in parity mapping
    # Need to know mapping. Let's assume HF state corresponds to |1100> (first two qubits 1, next two 0)
    # Actually parity mapping reduces qubits by 1? We have 4 qubits.
    # We'll try all possible computational basis states and compute expectation
    # There are only 2^4=16 states, we can brute force
    n_qubits = 4
    min_energy = float('inf')
    best_state = None
    for i in range(2**n_qubits):
        # create computational basis state
        c = tc.Circuit(n_qubits)
        for q in range(n_qubits):
            if (i >> q) & 1:
                c.x(q)
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
            exp_val = c.expectation_ps(x=x_list, y=y_list, z=z_list)
            energy += coeff * exp_val
        energy = energy.real
        if energy < min_energy:
            min_energy = energy
            best_state = i
    print(f"Brute force search: minimal classical energy = {min_energy:.8f}")
    print(f"State (binary): {format(best_state, '04b')}")
    return min_energy, best_state

min_e, best = expectation_hf_state()
print(f"Total energy including core: {min_e + e_core:.8f}")
print(f"Expected HF total energy: {hf.e_tot:.8f}")