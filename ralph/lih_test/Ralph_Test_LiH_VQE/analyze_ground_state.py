#!/usr/bin/env python3
"""
Analyze exact ground state of LiH Hamiltonian.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
import tensorcircuit as tc
from openfermion.linalg import get_sparse_operator

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

def build_hamiltonian():
    """Build the 4-qubit Hamiltonian for LiH."""
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
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of Pauli terms: {len(h_qubit_op.terms)}")
    const = h_qubit_op.terms.get((), 0)
    print(f"Constant term: {const}")
    return h_qubit_op, n_qubits, const

def hamiltonian_matrix(h_qubit_op, n_qubits):
    """Construct 2^n x 2^n Hamiltonian matrix using openfermion."""
    from openfermion.linalg import get_sparse_operator
    H_sparse = get_sparse_operator(h_qubit_op, n_qubits=n_qubits)
    return H_sparse.toarray()

def analyze_ground_state(H):
    """Diagonalize and analyze ground state."""
    eigvals, eigvecs = np.linalg.eigh(H)
    gs_energy = eigvals[0]
    gs_vec = eigvecs[:,0]
    print(f"Exact ground state energy: {gs_energy.real:.8f}")
    print(f"First excited state energy: {eigvals[1].real:.8f}")
    print(f"Energy gap: {eigvals[1].real - gs_energy.real:.8f}")

    # Find largest amplitude computational basis states
    dim = len(gs_vec)
    amplitudes = [(i, abs(gs_vec[i])**2) for i in range(dim)]
    amplitudes.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 computational basis states in ground state:")
    for i, prob in amplitudes[:5]:
        state_bin = format(i, f'0{int(np.log2(dim))}b')
        print(f"  |{state_bin}⟩: amplitude {gs_vec[i]:.4f}, prob {prob:.4f}")

    # Print HF state (likely |1100⟩?)
    # Determine HF state by checking single-particle energies?
    # For 2 electrons in 4 spin orbitals (0↑,0↓,1↑,1↓) where orbitals 0 and 1 are occupied.
    # In parity mapping, need to know mapping. Let's compute expectation of number operator?
    # Instead, let's just list all states with significant amplitude.
    return gs_energy, gs_vec

def main():
    h_qubit_op, n_qubits, const = build_hamiltonian()
    H = hamiltonian_matrix(h_qubit_op, n_qubits)
    gs_energy, gs_vec = analyze_ground_state(H)

    # Verify constant term: ground state expectation of non-constant part
    H_no_const = H - const * np.eye(H.shape[0])
    exp_val = gs_vec.conj().T @ H_no_const @ gs_vec
    print(f"\nExpectation of non-constant part: {exp_val.real:.8f}")
    print(f"Constant term: {const:.8f}")
    print(f"Sum: {(exp_val + const).real:.8f}")
    print(f"Ground state energy: {gs_energy.real:.8f}")

    # Compute HF state energy (brute force)
    dim = H.shape[0]
    hf_energy = min(H[i,i].real for i in range(dim))
    hf_state = np.argmin([H[i,i].real for i in range(dim)])
    print(f"\nHF state (diagonal minimum): |{format(hf_state, f'0{n_qubits}b')}⟩, energy {hf_energy:.8f}")
    print(f"Correlation energy: {gs_energy.real - hf_energy:.8f} Ha")
    print(f"Correlation energy in mHa: {(gs_energy.real - hf_energy)*1000:.3f}")

if __name__ == "__main__":
    main()