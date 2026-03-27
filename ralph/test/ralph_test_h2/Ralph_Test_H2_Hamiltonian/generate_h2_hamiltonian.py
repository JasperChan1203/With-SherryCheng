#!/usr/bin/env python
"""
Generate H2 molecule Hamiltonian at 0.5 Å bond length using PySCF.
Compute Hartree-Fock and Full Configuration Interaction energies.
Extract Hamiltonian information for quantum simulation.

This script implements the H2 Hamiltonian generation objective for the RLQAS project.
"""

import json
import sys
import numpy as np
import pyscf
from pyscf import gto, scf, fci

def main():
    print("H2 Hamiltonian Generation with PySCF")
    print("====================================\n")

    # 1. Define molecule: H2 at 0.5 Å bond length, STO-3G basis
    bond_length = 0.5  # Ångstrom
    print(f"Defining H2 molecule with bond length {bond_length} Å")
    print("Basis set: STO-3G")

    mol = gto.M(
        atom=[["H", 0, 0, 0], ["H", bond_length, 0, 0]],
        basis="sto-3g",
        unit="angstrom",
        symmetry=False,
        verbose=0
    )

    # Basic molecular information
    n_elec = mol.nelectron  # Should be 2 for H2
    n_orbs = mol.nao_nr()   # Number of atomic orbitals
    print(f"Number of electrons: {n_elec}")
    print(f"Number of atomic orbitals: {n_orbs}")

    # 2. Hartree-Fock calculation (Restricted HF)
    print("\nPerforming Hartree-Fock calculation...")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    hf_energy = mf.kernel()

    if not mf.converged:
        print("WARNING: Hartree-Fock calculation did not converge!")
        # Continue anyway for this small system

    print(f"Hartree-Fock energy: {hf_energy:.10f} Hartree")

    # 3. Full Configuration Interaction using built-in FCI from RHF object
    print("\nPerforming Full Configuration Interaction (FCI) calculation...")
    fci_solver = fci.FCI(mf)  # Use RHF object to ensure correct Hamiltonian
    fci_energy, fci_vec = fci_solver.kernel()
    print(f"FCI energy: {fci_energy:.10f} Hartree")
    print(f"Correlation energy (FCI - HF): {fci_energy - hf_energy:.10f} Hartree")

    # 4. Extract Hamiltonian information for quantum simulation
    print("\nExtracting Hamiltonian information...")

    # Get core Hamiltonian in atomic orbital basis
    h1_ao = mf.get_hcore()  # Core Hamiltonian (kinetic + nuclear attraction)

    # Get two-electron integrals in atomic orbital basis
    # Use pyscf's integral generation with 8-fold symmetry
    h2_ao = mol.intor("int2e", aosym="s8")

    # Transform to molecular orbital basis using HF coefficients
    mo_coeff = mf.mo_coeff  # MO coefficients
    n_spatial_orbs = mo_coeff.shape[1]  # Number of molecular orbitals

    # Transform one-electron integrals
    h1_mo = np.einsum("pi,pq,qj->ij", mo_coeff, h1_ao, mo_coeff)

    # Transform two-electron integrals using pyscf's ao2mo module
    from pyscf import ao2mo
    # Get two-electron integrals in MO basis (full 4-index tensor)
    h2_mo = ao2mo.full(mol, mo_coeff)
    # Convert to numpy array with proper shape
    h2_mo = ao2mo.restore(1, h2_mo, n_spatial_orbs)  # 1 = no symmetry

    # 5. Hamiltonian properties
    print("\nAnalyzing Hamiltonian properties...")
    # Number of qubits = number of spin orbitals = 2 * number of spatial orbitals
    n_spin_orbitals = 2 * n_spatial_orbs
    print(f"Number of spatial orbitals: {n_spatial_orbs}")
    print(f"Number of spin orbitals (qubits): {n_spin_orbitals}")

    # Check Hermiticity of one-electron Hamiltonian
    h1_hermitian = np.allclose(h1_mo, h1_mo.T.conj(), atol=1e-12)
    print(f"One-electron Hamiltonian is Hermitian: {h1_hermitian}")
    if not h1_hermitian:
        print("WARNING: One-electron Hamiltonian is not Hermitian within tolerance 1e-12")

    # 6. Prepare results dictionary matching PRD and validation requirements
    print("\nPreparing results for h2_results.json...")
    results = {
        "molecule": {
            "formula": "H2",
            "bond_length_angstrom": bond_length,
            "basis_set": "sto-3g",
            "atoms": [
                {"symbol": "H", "position": [0.0, 0.0, 0.0]},
                {"symbol": "H", "position": [bond_length, 0.0, 0.0]}
            ]
        },
        "energies": {
            "hf_hartree": float(hf_energy),
            "fci_hartree": float(fci_energy),
            "correlation_energy": float(fci_energy - hf_energy)
        },
        "hamiltonian": {
            "n_qubits": n_spin_orbitals,  # Validation expects this field
            "h1_hermitian": bool(h1_hermitian),  # Validation expects this field
            "n_spin_orbitals": n_spin_orbitals,
            "n_spatial_orbitals": n_spatial_orbs,
            "one_electron_integrals_shape": list(h1_mo.shape),
            "two_electron_integrals_shape": [n_spin_orbitals] * 4
        },
        "implementation_details": {
            "method": "PySCF RHF + FCI",
            "pyscf_version": pyscf.__version__,
            "script_path": __file__,
            "converged": bool(mf.converged)
        }
    }

    # 7. Write results to JSON file
    output_file = "h2_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to {output_file}")

    # 8. Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Molecule: H2, bond length = {bond_length} Å, basis = sto-3g")
    print(f"HF energy: {hf_energy:.8f} Hartree")
    print(f"FCI energy: {fci_energy:.8f} Hartree")
    print(f"Correlation energy: {fci_energy - hf_energy:.8f} Hartree")
    print(f"Number of qubits: {n_spin_orbitals}")
    print(f"Hamiltonian Hermitian: {h1_hermitian}")
    print(f"Output file: {output_file}")
    print("="*50)

    return 0

if __name__ == "__main__":
    sys.exit(main())