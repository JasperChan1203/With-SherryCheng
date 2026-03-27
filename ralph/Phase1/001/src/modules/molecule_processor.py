"""
Molecule Processing Module for RLQAS.

This module converts molecular information into quantum computation inputs
using Tencirchem-ng 2024.10 and OpenFermion.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from openfermion import QubitOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from pyscf import gto, scf, fci, ao2mo, mcscf
import tencirchem
from tencirchem import parity, UCC


@dataclass
class MoleculeData:
    """Container for molecule processing results."""
    hamiltonian: QubitOperator      # Qubit Hamiltonian
    n_qubits: int                   # Number of qubits
    reference_state: np.ndarray     # Reference state (Hartree-Fock)
    fci_energy: float               # Exact FCI energy
    molecular_info: Dict            # Original molecular information


def process_molecule(
    molecule: str,
    bond_length: float,
    ansatz_type: str,
    active_space: Optional[Tuple[int, int]] = None,
    basis_set: str = "sto-3g",
    transform: str = "parity"
) -> MoleculeData:
    """
    Process a molecule and generate quantum computation inputs.

    Args:
        molecule: Molecular formula, e.g., 'LiH', 'BeH2', 'H4'
        bond_length: Bond length (Å)
        ansatz_type: Ansatz type: 'UCC', 'HEA', 'MIXED'
        active_space: Optional (number of active electrons, number of active orbitals)
        basis_set: Basis set
        transform: Fermion-to-qubit transformation: 'parity', 'jordan_wigner', 'bravyi_kitaev'

    Returns:
        MoleculeData object containing Hamiltonian, qubit count, reference state,
        FCI energy, and molecular information.

    Raises:
        ValueError: If inputs are invalid or processing fails.
        ImportError: If required dependencies are not available.
    """
    # Validate inputs
    if bond_length <= 0:
        raise ValueError("bond_length must be positive")
    if ansatz_type not in ("UCC", "HEA", "MIXED"):
        raise ValueError("ansatz_type must be 'UCC', 'HEA', or 'MIXED'")
    if transform not in ("parity", "jordan_wigner", "bravyi_kitaev"):
        raise ValueError("transform must be 'parity', 'jordan_wigner', or 'bravyi_kitaev'")

    # Parse molecule string into atom list
    # Simple support for H2, LiH, BeH2
    molecule = molecule.strip()
    if molecule == "H2":
        atoms = [("H", 0.0, 0.0, 0.0), ("H", bond_length, 0.0, 0.0)]
    elif molecule == "LiH":
        atoms = [("Li", 0.0, 0.0, 0.0), ("H", bond_length, 0.0, 0.0)]
    elif molecule == "BeH2":
        # Linear geometry: H-Be-H, bond length is Be-H distance
        atoms = [
            ("H", -bond_length, 0.0, 0.0),
            ("Be", 0.0, 0.0, 0.0),
            ("H", bond_length, 0.0, 0.0)
        ]
    else:
        raise ValueError(f"Unsupported molecule: {molecule}. Supported: H2, LiH, BeH2")

    # Build PySCF molecule
    mol = gto.M(
        atom=atoms,
        basis=basis_set,
        unit="angstrom",
        symmetry=False,
        verbose=0
    )

    # Hartree-Fock calculation with robust settings
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-8
    hf.conv_tol_grad = 1e-8
    hf.max_cycle = 1000
    hf.init_guess = 'atom'
    hf_energy = hf.kernel()
    if not hf.converged:
        raise RuntimeError("Hartree-Fock calculation did not converge")

    # Full CI energy
    fci_solver = fci.FCI(hf)
    fci_energy, _ = fci_solver.kernel()

    # Determine active space (full space if None)
    n_elec = mol.nelectron
    n_orb = mol.nao_nr()
    if active_space is None:
        active_space = (n_elec, n_orb)
    n_elec_active, n_orb_active = active_space

    # Get integrals in MO basis for active space
    mo_coeff = hf.mo_coeff
    mo_energy = hf.mo_energy

    # Determine active orbital indices
    if active_space == (n_elec, n_orb):
        # Full space: all orbitals
        active_indices = list(range(n_orb_active))
        # Transform integrals directly
        h1_ao = hf.get_hcore()
        h1_mo = np.einsum("pi,pq,qj->ij", mo_coeff, h1_ao, mo_coeff)
        h1_active = h1_mo[np.ix_(active_indices, active_indices)]
        h2_mo = ao2mo.full(mol, mo_coeff)
        h2_mo = ao2mo.restore(1, h2_mo, n_orb)
        h2_active = h2_mo[np.ix_(active_indices, active_indices, active_indices, active_indices)]
        e_core = 0.0
    else:
        # Active space smaller than full space: use CASCI to get integrals
        # Select active orbitals as highest-energy orbitals (valence)
        # Sort orbitals by energy descending (higher energy first)
        sorted_indices = np.argsort(mo_energy)[::-1]  # descending
        # Take the first n_orb_active highest-energy orbitals
        active_indices = sorted_indices[:n_orb_active].tolist()
        active_indices.sort()  # keep original order for consistency
        # Create CASCI object
        cas = mcscf.CASCI(hf, n_orb_active, n_elec_active)
        # Use sort_mo to reorder orbitals (1-indexed)
        cas.sort_mo([i+1 for i in active_indices])
        cas.kernel()
        # Get integrals and core energy from CASCI
        h1_active, e_core = cas.get_h1eff()
        h2_active = cas.get_h2eff()
        # h2_active is in chemists' notation with symmetry, restore full 4-index array
        h2_active = ao2mo.restore(1, h2_active, n_orb_active)

    # Build UCC from integrals
    ucc = UCC.from_integral(h1_active, h2_active, n_elec_active, e_core=e_core, hcb=False)

    # Get fermion operator
    fermion_op = ucc.h_fermion_op

    # Transform to qubit operator
    if transform == "parity":
        n_modes = 2 * n_orb_active  # spin orbitals
        qubit_op = parity(fermion_op, n_modes=n_modes, n_elec=n_elec_active)
    elif transform == "jordan_wigner":
        qubit_op = jordan_wigner(fermion_op)
    elif transform == "bravyi_kitaev":
        qubit_op = bravyi_kitaev(fermion_op)
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    # Determine number of qubits from qubit operator
    max_idx = 0
    for term in qubit_op.terms:
        for idx, _ in term:
            if idx > max_idx:
                max_idx = idx
    n_qubits = max_idx + 1

    # Compute reference state (Hartree-Fock) as one-hot vector
    # HF state corresponds to occupying first n_elec_active spin orbitals
    # Assuming RHF: each spatial orbital doubly occupied
    # For unrestricted? We'll assume RHF.
    # Build bitstring: first n_elec_active spin orbitals are 1, rest 0
    # Spin orbital ordering: alpha then beta? In Tencirchem, mapping?
    # We'll approximate: reference state is computational basis state with
    # minimal energy expectation (brute force over all basis states)
    # For small n_qubits (<= 6), this is feasible.
    # For larger, need better method.
    # We'll implement brute force for now.
    min_energy = float('inf')
    best_state_idx = 0
    for i in range(2 ** n_qubits):
        # Compute expectation value of Hamiltonian for computational basis state i
        energy = 0.0
        for term, coeff in qubit_op.terms.items():
            # For computational basis state, expectation is product of ±1
            # depending on Pauli Z eigenvalues
            # For term with X or Y, expectation is 0
            # So we only need terms with only Z and identity
            # We'll compute full expectation using simple method:
            # Convert term to diagonal check
            diag = True
            sign = 1.0
            for idx, pauli in term:
                if pauli == 'Z':
                    # eigenvalue +1 if qubit i has 0 at idx, -1 if 1
                    bit = (i >> idx) & 1
                    sign *= (1 - 2*bit)  # +1 for 0, -1 for 1
                elif pauli == 'I':
                    continue
                else:
                    diag = False
                    break
            if diag:
                energy += coeff * sign
        if energy < min_energy:
            min_energy = energy
            best_state_idx = i

    # Create one-hot vector
    reference_state = np.zeros(2 ** n_qubits, dtype=complex)
    reference_state[best_state_idx] = 1.0

    # Molecular information dictionary
    molecular_info = {
        "formula": molecule,
        "bond_length_angstrom": bond_length,
        "basis_set": basis_set,
        "active_space": active_space,
        "transform": transform,
        "ansatz_type": ansatz_type,
        "hf_energy": hf_energy,
        "n_electrons": n_elec,
        "n_orbitals": n_orb,
        "n_qubits": n_qubits,
    }

    return MoleculeData(
        hamiltonian=qubit_op,
        n_qubits=n_qubits,
        reference_state=reference_state,
        fci_energy=fci_energy,
        molecular_info=molecular_info,
    )