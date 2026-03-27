"""
Fermion-to-qubit transformation utilities for RLQAS.

This module provides helper functions for working with different fermion-to-qubit
transformations (parity, Jordan-Wigner, Bravyi-Kitaev) and computing reference states.
"""

import numpy as np
from openfermion import QubitOperator


def get_hartree_fock_bitstring(n_qubits: int, n_electrons: int, transform: str) -> int:
    """Compute Hartree-Fock bitstring for given transformation.

    Args:
        n_qubits: Number of qubits after transformation.
        n_electrons: Number of electrons (spin orbitals occupied in HF state).
        transform: Fermion-to-qubit transformation ('parity', 'jordan_wigner', 'bravyi_kitaev').

    Returns:
        Integer representing computational basis state (bitstring) where
        least significant bit is qubit 0.

    Raises:
        ValueError: If transform not supported or parameters invalid.
    """
    if transform == "jordan_wigner":
        # Jordan-Wigner: each spin orbital maps to a qubit
        # RHF: occupy first n_electrons spin orbitals (assuming alpha then beta ordering)
        # For unrestricted? Assume RHF with paired electrons.
        # We'll assume spin orbital ordering: alpha then beta for each spatial orbital.
        # So first n_electrons//2 spatial orbitals fully occupied (both alpha and beta).
        hf_bitstring = 0
        for i in range(n_electrons // 2):
            # alpha qubit index = 2*i
            hf_bitstring |= 1 << (2 * i)
            # beta qubit index = 2*i + 1
            hf_bitstring |= 1 << (2 * i + 1)
        # For odd n_electrons (unrestricted) not handled; fallback brute-force
        if n_electrons % 2 != 0:
            raise ValueError("Unrestricted HF (odd n_electrons) not supported for Jordan-Wigner mapping")
        return hf_bitstring

    elif transform == "parity":
        # Parity transformation reduces qubit count by 2 (conserves particle number and parity)
        # For N spin orbitals, parity mapping uses N-1 qubits.
        # The HF occupation corresponds to |1100...0⟩ (first two qubits = 1, rest = 0)
        # This is based on typical mapping where first two qubits encode parity information.
        # Need to verify with openfermion's parity mapping.
        # For now, we'll implement brute-force search but with optimization.
        # Return None to indicate need for brute-force.
        return None

    elif transform == "bravyi_kitaev":
        # Bravyi-Kitaev mapping is more complex; fallback to brute-force.
        return None

    else:
        raise ValueError(f"Unsupported transform: {transform}")


def compute_reference_state_diagonal(hamiltonian: QubitOperator, n_qubits: int) -> np.ndarray:
    """Compute reference state by finding computational basis state with minimal diagonal energy.

    This function efficiently computes diagonal energies for all computational basis states
    by evaluating only the diagonal terms (Z and I) of the Hamiltonian.

    Args:
        hamiltonian: QubitOperator representing the Hamiltonian.
        n_qubits: Number of qubits.

    Returns:
        One-hot vector (complex) representing the reference state.
    """
    # Precompute diagonal contributions for each qubit position
    # For each qubit i, we need to know the coefficient sum of terms where only Z appears at i.
    # Actually we can compute energy for each bitstring using vectorized approach.
    # Since n_qubits may be up to ~12, we can still brute-force but more efficiently.
    # We'll implement the same brute-force as before but with optimization:
    # - Pre-filter diagonal terms
    # - Compute energy using bitwise operations

    diagonal_terms = []
    diagonal_coeffs = []
    for term, coeff in hamiltonian.terms.items():
        diag = True
        for idx, pauli in term:
            if pauli not in ('Z', 'I'):
                diag = False
                break
        if diag:
            diagonal_terms.append(term)
            diagonal_coeffs.append(coeff)

    min_energy = float('inf')
    best_state_idx = 0

    # Iterate over all computational basis states
    for i in range(2 ** n_qubits):
        energy = 0.0
        for term, coeff in zip(diagonal_terms, diagonal_coeffs):
            sign = 1.0
            for idx, pauli in term:
                if pauli == 'Z':
                    bit = (i >> idx) & 1
                    sign *= (1 - 2 * bit)
                # I does nothing
            energy += coeff * sign
        if energy < min_energy:
            min_energy = energy
            best_state_idx = i

    # Create one-hot vector
    reference_state = np.zeros(2 ** n_qubits, dtype=complex)
    reference_state[best_state_idx] = 1.0
    return reference_state


def compute_reference_state(hamiltonian: QubitOperator, n_qubits: int,
                            transform: str, n_electrons: int) -> np.ndarray:
    """Compute reference state efficiently using Hartree-Fock mapping when possible.

    Args:
        hamiltonian: QubitOperator representing the Hamiltonian.
        n_qubits: Number of qubits after transformation.
        transform: Fermion-to-qubit transformation.
        n_electrons: Number of electrons (for HF mapping).

    Returns:
        One-hot vector (complex) representing the reference state.
    """
    try:
        hf_bitstring = get_hartree_fock_bitstring(n_qubits, n_electrons, transform)
        if hf_bitstring is not None:
            # Verify that the HF bitstring yields minimal energy (optional)
            # For now, accept it as reference state
            reference_state = np.zeros(2 ** n_qubits, dtype=complex)
            reference_state[hf_bitstring] = 1.0
            return reference_state
    except (ValueError, NotImplementedError):
        pass

    # Fallback to diagonal search
    return compute_reference_state_diagonal(hamiltonian, n_qubits)