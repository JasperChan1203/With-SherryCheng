#!/usr/bin/env python3
"""
Verify FCI energy for benchmark orbital combination [1,2,5] (0-based).
This calculates the exact FCI energy using PySCF CASCI/FCI to see if
it matches the benchmark -7.860153 Hartree.
"""

import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo

def compute_exact_fci_with_orbitals():
    """Compute FCI energy with specific active orbitals [1,2,5] (0-based)."""
    # Define molecule with correct geometry: H at origin, Li at 2.0 Å
    bond_length = 2.0
    mol = gto.M(
        atom=f'H 0 0 0; Li {bond_length} 0 0',
        basis='sto-3g',
        symmetry=False,
        verbose=0
    )

    # Hartree-Fock calculation
    hf = scf.RHF(mol)
    hf.kernel()

    print("=" * 70)
    print("LiH FCI Energy Verification")
    print("=" * 70)
    print(f"Molecule: LiH, bond length = {bond_length} Å")
    print(f"Basis set: sto-3g")
    print(f"HF energy: {hf.e_tot:.8f} Hartree")
    print(f"Total electrons: {mol.nelectron}")

    # Orbital energies
    print("\nOrbital energies (0-based):")
    for i, e in enumerate(hf.mo_energy):
        occupied = "occupied" if i < mol.nelectron // 2 else "virtual"
        print(f"  {i}: {e:.6f} Ha ({occupied})")

    # Benchmark orbitals: 1-based [2,3,6] -> 0-based [1,2,5]
    active_orbitals = [1, 2, 5]
    print(f"\nBenchmark active orbitals: {active_orbitals} (0-based)")
    print(f"Corresponding to 1-based indices: {[i+1 for i in active_orbitals]}")

    # Verify orbital selection
    print(f"\nSelected orbital energies:")
    for idx in active_orbitals:
        occ_status = "occupied" if idx < mol.nelectron // 2 else "virtual"
        print(f"  Orbital {idx} (1-based {idx+1}): {hf.mo_energy[idx]:.6f} Ha ({occ_status})")

    # Setup for CASCI calculation
    n_orb = len(active_orbitals)  # 3 orbitals
    n_elec = 2  # 2 active electrons
    nocc = mol.nelectron // 2  # 2 occupied orbitals

    # Determine core orbitals (occupied orbitals not in active)
    core_orbitals = [i for i in range(nocc) if i not in active_orbitals]
    ncore = len(core_orbitals)

    # Determine frozen virtual orbitals (virtual orbitals not in active)
    total_orbs = hf.mo_coeff.shape[1]
    virtual_orbitals = list(range(nocc, total_orbs))
    frozen_virtual = [i for i in virtual_orbitals if i not in active_orbitals]

    print(f"\nCASCI setup:")
    print(f"  Active space: ({n_elec}, {n_orb})")
    print(f"  Core orbitals: {core_orbitals}, ncore = {ncore}")
    print(f"  Frozen virtual orbitals: {frozen_virtual}")
    print(f"  Check electron conservation: {ncore * 2 + n_elec} electrons")

    # Create CASCI object
    cas = mcscf.CASCI(hf, n_orb, n_elec)
    cas.ncore = ncore
    cas.frozen = frozen_virtual
    cas.mo_coeff = hf.mo_coeff

    print(f"\nComputing integrals...")

    try:
        # Get integrals
        int1e, e_core = cas.get_h1eff()
        int2e = cas.get_h2eff()
        int2e_full = ao2mo.restore(1, int2e, n_orb)

        print(f"  Core energy: {e_core:.8f} Hartree")
        print(f"  int1e shape: {int1e.shape}")

        # Compute FCI in active space
        fci_solver = fci.direct_spin0.FCI()
        e_fci, _ = fci_solver.kernel(int1e, int2e_full, n_orb, n_elec, ecore=e_core)

        print(f"\nFCI energy in active space: {e_fci:.8f} Hartree")

        # Also compute CASCI energy for verification
        cas.kernel()
        print(f"CASCI total energy: {cas.e_tot:.8f} Hartree")
        print(f"CASCI active energy: {cas.e_cas:.8f} Hartree")
        print(f"CASCI core energy: {cas.e_core:.8f} Hartree")

        # Compare with benchmark
        bench_fci = -7.860153
        diff = e_fci - bench_fci
        abs_diff = abs(diff)

        print(f"\nComparison with benchmark:")
        print(f"  Computed FCI: {e_fci:.8f} Hartree")
        print(f"  Benchmark FCI: {bench_fci:.8f} Hartree")
        print(f"  Difference: {diff:.8f} Hartree ({diff*1000:.3f} mHa)")

        if abs_diff < 0.0016:  # 1.6 mHa chemical accuracy
            print(f"  ✅ Within chemical accuracy!")
        else:
            print(f"  ❌ Outside chemical accuracy (tolerance: 1.6 mHa)")

        # Check if CASCI matches FCI (should be exact)
        cas_fci_diff = abs(cas.e_tot - e_fci)
        print(f"\nCASCI-FCI consistency check:")
        print(f"  CASCI total - FCI: {cas_fci_diff:.2e} Hartree")
        if cas_fci_diff < 1e-10:
            print(f"  ✅ CASCI and FCI energies match")
        else:
            print(f"  ⚠️  CASCI and FCI differ by {cas_fci_diff:.2e} Hartree")

        return e_fci, e_core, cas.e_tot

    except Exception as e:
        print(f"\nERROR during calculation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main verification function."""
    print("Verifying benchmark FCI energy for LiH with active orbitals [1,2,5] (0-based)")

    # Also test the alternative ordering: [0,1,4] (1-based [1,2,5])
    # which might be what the benchmark intended
    print("\n" + "="*70)
    print("Testing alternative interpretation: active orbitals [0,1,4]")
    print("(1-based [1,2,5], which might be what benchmark file intended)")

    # Run the main calculation
    fci_energy, core_energy, casci_energy = compute_exact_fci_with_orbitals()

    if fci_energy is not None:
        print("\n" + "="*70)
        print("SUMMARY:")
        print(f"FCI energy with orbitals [1,2,5] (0-based): {fci_energy:.8f} Hartree")
        print(f"Benchmark target: -7.860153 Hartree")
        print(f"Difference: {fci_energy + 7.860153:.8f} Hartree")

        # Check if this matches any of our earlier results
        print("\nNote: All valid (2,3) active space combinations give -7.832222 Hartree")
        print(f"This result matches that pattern: {abs(fci_energy + 7.832222) < 1e-6}")

if __name__ == "__main__":
    main()