#!/usr/bin/env python
"""
Generate benchmark values for H2 molecule at 0.5 Å bond length using PySCF.
This script produces reference values that can be used to validate Ralph's implementation.
"""
import json
import numpy as np
from pyscf import gto, scf, fci

def generate_h2_benchmark():
    """Generate benchmark values for H2 at 0.5 Å bond length."""

    # Define H2 molecule with 0.5 Å bond length
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.5',  # 0.5 Å along z-axis
        basis='sto-3g',             # Minimal basis set
        unit='angstrom',            # Explicitly specify angstrom units
        verbose=0                   # Suppress output
    )

    # Hartree-Fock calculation
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()

    # Full Configuration Interaction (FCI) calculation
    cisolver = fci.FCI(mf)
    fci_energy, fci_vec = cisolver.kernel()

    # Get Hamiltonian information
    h1 = mf.get_hcore()  # One-electron integrals
    eri = mf._eri        # Two-electron integrals

    # Calculate Hamiltonian properties
    # For H2 with STO-3G basis: 4 spin orbitals -> 2-4 qubits depending on mapping
    n_qubits_min = 2   # Minimal mapping (e.g., parity with symmetry)
    n_qubits_max = 4   # Jordan-Wigner mapping

    # Check Hermiticity
    h1_hermitian = np.allclose(h1, h1.conj().T)

    # Prepare benchmark data
    benchmark = {
        "molecule": {
            "formula": "H2",
            "atoms": [
                {"symbol": "H", "position": [0.0, 0.0, 0.0]},
                {"symbol": "H", "position": [0.0, 0.0, 0.5]}
            ],
            "bond_length_angstrom": 0.5,
            "basis_set": "sto-3g"
        },
        "energies": {
            "hf_hartree": float(hf_energy),
            "fci_hartree": float(fci_energy),
            "correlation_energy": float(fci_energy - hf_energy)
        },
        "hamiltonian": {
            "n_spin_orbitals": 4,
            "n_qubits_min": n_qubits_min,
            "n_qubits_max": n_qubits_max,
            "one_electron_integrals_shape": list(h1.shape),
            "two_electron_integrals_shape": [4, 4, 4, 4],  # STO-3G basis shape
            "h1_hermitian": h1_hermitian
        },
        "verification_tolerances": {
            "hf_energy_tolerance_hartree": 0.001,
            "fci_energy_tolerance_hartree": 0.001,
            "bond_length_tolerance_angstrom": 0.01,
            "basis_set_match": True
        },
        "computation_details": {
            "method": "PySCF RHF + FCI",
            "pyscf_version": "2.12.0",
            "converged": mf.converged
        }
    }

    return benchmark

def main():
    """Generate and save benchmark data."""
    print("Generating H2 benchmark values using PySCF...")

    try:
        benchmark = generate_h2_benchmark()

        # Save to JSON file
        output_file = "h2_benchmark.json"
        with open(output_file, 'w') as f:
            json.dump(benchmark, f, indent=2)

        print(f"\n✓ Benchmark generated successfully!")
        print(f"  Output file: {output_file}")
        print(f"\nKey values:")
        print(f"  HF energy: {benchmark['energies']['hf_hartree']:.6f} Hartree")
        print(f"  FCI energy: {benchmark['energies']['fci_hartree']:.6f} Hartree")
        print(f"  Correlation energy: {benchmark['energies']['correlation_energy']:.6f} Hartree")
        print(f"  Bond length: {benchmark['molecule']['bond_length_angstrom']} Å")
        print(f"  Basis set: {benchmark['molecule']['basis_set']}")
        print(f"  Qubit range: {benchmark['hamiltonian']['n_qubits_min']}-{benchmark['hamiltonian']['n_qubits_max']}")

        # Verification notes
        print(f"\nVerification criteria:")
        print(f"  HF energy tolerance: ±{benchmark['verification_tolerances']['hf_energy_tolerance_hartree']} Hartree")
        print(f"  FCI energy tolerance: ±{benchmark['verification_tolerances']['fci_energy_tolerance_hartree']} Hartree")
        print(f"  Hamiltonian Hermitian: {benchmark['hamiltonian']['h1_hermitian']}")

    except Exception as e:
        print(f"✗ Error generating benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())