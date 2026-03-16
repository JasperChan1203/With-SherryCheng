#!/usr/bin/env python3
"""
Analyze the tencirchem benchmark to understand FCI energy difference.
"""

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci

def run_benchmark_calculation():
    """Run the exact calculation from tencirchem_benchmark.py"""

    d = 2.0

    print("=" * 70)
    print("Analyzing tencirchem benchmark calculation")
    print("=" * 70)

    # Method 1: Exact reproduction from benchmark file
    print("\nMethod 1: Exact reproduction (with symmetry=True, no basis specified)")
    mol1 = gto.M(atom=[["H", 0, 0, 0], ["Li", d, 0, 0]], charge=0, symmetry=True)
    print(f"  Basis set: {mol1.basis}")
    print(f"  Symmetry: {mol1.symmetry}")
    print(f"  Atom: {mol1.atom}")
    print(f"  Number of basis functions: {mol1.nao}")
    print(f"  AO labels: {mol1.ao_labels()}")

    hf1 = scf.RHF(mol1)
    hf1.kernel()
    print(f"  HF energy: {hf1.e_tot:.8f}")
    print(f"  Number of spatial orbitals: {hf1.mo_coeff.shape[1]}")

    # Orbital energies
    print(f"\n  Orbital energies (0-based):")
    for i, e in enumerate(hf1.mo_energy):
        occupied = "occupied" if i < mol1.nelectron // 2 else "virtual"
        print(f"    {i}: {e:.6f} Ha ({occupied})")

    # CASCI with sort_mo([2,3,6]) - 1-based indices
    mycas = mcscf.CASCI(hf1, 3, 2)  # 3 orbitals, 2 electrons
    mo = mycas.sort_mo([2, 3, 6])  # 1-based indices
    mycas.kernel(mo)
    print(f"\n  CASCI total energy (from benchmark): {mycas.e_tot:.8f}")
    print(f"  CASCI active energy: {mycas.e_cas:.8f}")

    # Method 2: Our previous setup (no symmetry, explicit basis)
    print("\n" + "=" * 70)
    print("Method 2: Our previous setup (symmetry=False, basis='sto-3g')")
    mol2 = gto.M(
        atom=f'H 0 0 0; Li {d} 0 0',
        basis='sto-3g',
        symmetry=False,
        verbose=0
    )
    print(f"  Basis set: {mol2.basis}")
    print(f"  Symmetry: {mol2.symmetry}")

    hf2 = scf.RHF(mol2)
    hf2.kernel()
    print(f"  HF energy: {hf2.e_tot:.8f}")
    print(f"  Number of spatial orbitals: {hf2.mo_coeff.shape[1]}")

    print(f"\n  Orbital energies (0-based):")
    for i, e in enumerate(hf2.mo_energy):
        occupied = "occupied" if i < mol2.nelectron // 2 else "virtual"
        print(f"    {i}: {e:.6f} Ha ({occupied})")

    # CASCI with same active orbitals [1,2,5] (0-based)
    n_orb = 3
    n_elec = 2
    nocc = mol2.nelectron // 2
    active_orbitals = [1, 2, 5]  # 0-based

    core_orbitals = [i for i in range(nocc) if i not in active_orbitals]
    ncore = len(core_orbitals)

    total_orbs = hf2.mo_coeff.shape[1]
    virtual_orbitals = list(range(nocc, total_orbs))
    frozen_virtual = [i for i in virtual_orbitals if i not in active_orbitals]

    cas2 = mcscf.CASCI(hf2, n_orb, n_elec)
    cas2.ncore = ncore
    cas2.frozen = frozen_virtual
    cas2.mo_coeff = hf2.mo_coeff

    # Get integrals and compute FCI
    int1e, e_core = cas2.get_h1eff()
    int2e = cas2.get_h2eff()
    int2e_full = ao2mo.restore(1, int2e, n_orb)

    fci_solver = fci.direct_spin0.FCI()
    e_fci, _ = fci_solver.kernel(int1e, int2e_full, n_orb, n_elec, ecore=e_core)

    print(f"\n  FCI energy (our calculation): {e_fci:.8f}")

    # Also compute CASCI
    cas2.kernel()
    print(f"  CASCI total energy: {cas2.e_tot:.8f}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print(f"  Benchmark CASCI energy: {mycas.e_tot:.8f}")
    print(f"  Our FCI energy: {e_fci:.8f}")
    print(f"  Difference: {mycas.e_tot - e_fci:.8f} Ha ({abs(mycas.e_tot - e_fci)*1000:.3f} mHa)")
    print(f"  Target benchmark FCI: -7.860153 Ha")

    # Check if benchmark matches target
    bench_diff = abs(mycas.e_tot + 7.860153)
    print(f"\n  Benchmark vs target -7.860153:")
    print(f"    Benchmark CASCI: {mycas.e_tot:.8f}")
    print(f"    Difference: {bench_diff:.8f} Ha ({bench_diff*1000:.3f} mHa)")

    # Additional investigation: check basis set used in benchmark
    print("\n" + "=" * 70)
    print("BASIS SET INVESTIGATION:")
    print("PySCF default basis (when not specified):")
    # Create a test molecule without basis
    mol_test = gto.M(atom='H 0 0 0; Li 2.0 0 0')
    print(f"  Default basis: {mol_test.basis}")
    print(f"  Basis dictionary: {mol_test._basis}")

    # Check if symmetry affects orbital ordering
    print("\nSYMMETRY EFFECT:")
    mol_sym = gto.M(atom='H 0 0 0; Li 2.0 0 0', symmetry=True)
    mol_nosym = gto.M(atom='H 0 0 0; Li 2.0 0 0', symmetry=False)
    print(f"  With symmetry: {mol_sym.symmetry}")
    print(f"  Without symmetry: {mol_nosym.symmetry}")

    return mycas.e_tot, e_fci

if __name__ == "__main__":
    benchmark_energy, our_energy = run_benchmark_calculation()

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if abs(benchmark_energy + 7.860153) < 0.0016:
        print("✅ Benchmark calculation produces energy close to -7.860153 Ha")
    else:
        print("❌ Benchmark calculation does not match -7.860153 Ha")

    print(f"\nKey factors that might cause differences:")
    print("1. Basis set (default vs explicit 'sto-3g')")
    print("2. Symmetry settings (True vs False)")
    print("3. Orbital ordering due to symmetry")
    print("4. Active space definition (sort_mo vs manual core/frozen)")
    print("5. Integral transformation method")