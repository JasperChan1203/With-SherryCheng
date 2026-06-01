"""
Molecule Processing Module for RLQAS.

This module converts molecular information into quantum computation inputs
using Tencirchem-ng 2024.10 and OpenFermion.

Copied from Task 001, with minimal adaptations for integration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import warnings
from rlqas_chem.utils.logger import get_logger
from rlqas_chem.utils.transforms import compute_reference_state
from openfermion import QubitOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.linalg import get_ground_state, get_sparse_operator
from pyscf import gto, scf
import tencirchem
from tencirchem import parity, UCC, UCCSD
from tencirchem.static.hamiltonian import get_integral_from_hf

logger = get_logger(__name__)

@dataclass
class MoleculeData:
    """Container for molecule processing results."""
    hamiltonian: QubitOperator      # Qubit Hamiltonian (electronic only, no nuclear repulsion)
    n_qubits: int                   # Number of qubits
    reference_state: np.ndarray     # Reference state (Hartree-Fock)
    fci_energy: float               # Exact FCI energy (electronic + nuclear repulsion)
    molecular_info: Dict            # Original molecular information
    ucc_object: Any = None          # Tencirchem UCC object (for consistency)
    ucc_sd_object: Any = None       # Tencirchem UCCSD object (for circuit building)
    nuclear_repulsion: float = 0.0  # Always 0.0: nuclear repulsion is embedded in Hamiltonian's constant term (e_core)


def process_molecule(
    molecule: str,
    bond_length: float,
    ansatz_type: str,
    active_space: Optional[Tuple[int, int]] = None,
    basis_set: str = "sto-3g",
    transform: str = "jordan_wigner",
    aslst: Optional[List[int]] = None,
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
        aslst: Optional list of active orbital indices (0-indexed, tencirchem-ng convention) for
               active space orbital selection.  E.g. [1, 2, 5] for LiH (2,3).  Ignored for
               full-space calculations.  When None, default near-HOMO/LUMO selection is used.

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
    if ansatz_type not in ("UCC", "HEA", "MIXED", "HYBRID"):
        raise ValueError("ansatz_type must be 'UCC', 'HEA', 'HYBRID', or 'MIXED'")
    if transform not in ("parity", "jordan_wigner", "bravyi_kitaev"):
        raise ValueError("transform must be 'parity', 'jordan_wigner', or 'bravyi_kitaev'")
    # Warning about potential inconsistency with circuit builder (uses Jordan-Wigner internally)
    if transform != "jordan_wigner":
        logger.warning(
            f"Transform '{transform}' may cause inconsistency with circuit builder "
            "which uses Jordan-Wigner mapping internally. Chemical accuracy may be affected."
        )

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
    elif molecule == "H4":
        # Linear H4 chain: equally-spaced H atoms
        atoms = [
            ("H", 0.0, 0.0, 0.0),
            ("H", bond_length, 0.0, 0.0),
            ("H", 2.0 * bond_length, 0.0, 0.0),
            ("H", 3.0 * bond_length, 0.0, 0.0),
        ]
    elif molecule == "H6":
        # Linear H6 chain: equally-spaced H atoms
        atoms = [
            ("H", 0.0, 0.0, 0.0),
            ("H", bond_length, 0.0, 0.0),
            ("H", 2.0 * bond_length, 0.0, 0.0),
            ("H", 3.0 * bond_length, 0.0, 0.0),
            ("H", 4.0 * bond_length, 0.0, 0.0),
            ("H", 5.0 * bond_length, 0.0, 0.0),
        ]
    else:
        raise ValueError(f"Unsupported molecule: {molecule}. Supported: H2, LiH, BeH2, H4, H6")

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

    # Determine active space (full space if None)
    n_elec = mol.nelectron
    n_orb = mol.nao_nr()
    if active_space is None:
        active_space = (n_elec, n_orb)
    n_elec_active, n_orb_active = active_space

    is_full_space = (active_space == (n_elec, n_orb))

    if is_full_space:
        # Full space: use tencirchem-ng's get_integral_from_hf (no aslst)
        h1_active, h2_active, e_core = get_integral_from_hf(hf)
    else:
        # Active space: use tencirchem-ng's get_integral_from_hf with aslst (0-indexed)
        h1_active, h2_active, e_core = get_integral_from_hf(hf, active_space, aslst)

    # Build UCC from integrals (e_core embeds nuclear repulsion so Hamiltonian is self-contained)
    ucc = UCC.from_integral(h1_active, h2_active, n_elec_active, e_core=e_core)
    # Also create UCCSD object for circuit builder (uses Jordan-Wigner mapping)
    ucc_sd = UCCSD(mol, active_space=active_space, init_method='mp2')

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

    # FCI energy = exact lowest eigenvalue of the qubit Hamiltonian.
    # This is self-consistent: it is the true ground state of the Hamiltonian
    # being optimized by VQE, regardless of how the active space was constructed.
    fci_energy, _ = get_ground_state(get_sparse_operator(qubit_op))

    # Compute reference state using optimized utility function
    reference_state = compute_reference_state(
        qubit_op, n_qubits, transform, n_elec_active
    )

    # Molecular information dictionary
    molecular_info = {
        "formula": molecule,
        "bond_length_angstrom": bond_length,
        "basis_set": basis_set,
        "active_space": active_space,
        "aslst": aslst,
        "transform": transform,
        "ansatz_type": ansatz_type,
        "hf_energy": hf_energy,
        "n_electrons": n_elec,
        "n_orbitals": n_orb,
        "n_qubits": n_qubits,
    }

    logger.info(f"Processed molecule {molecule} with active space {active_space}, "
                f"aslst={aslst}, transform {transform}, qubits={n_qubits}, FCI energy={fci_energy:.6f}")

    return MoleculeData(
        hamiltonian=qubit_op,
        n_qubits=n_qubits,
        reference_state=reference_state,
        fci_energy=fci_energy,
        molecular_info=molecular_info,
        ucc_object=ucc,
        ucc_sd_object=ucc_sd,
        nuclear_repulsion=0.0,  # Already included in Hamiltonian's constant term (e_core)
    )
