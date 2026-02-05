#!/usr/bin/env python3
"""
Compare benchmark method vs current method for FCI energy calculation.
"""

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci

def benchmark_method():
    """Benchmark method with symmetry=True and sort_mo([2,3,6])."""
    d = 2.0
    mol = gto.M(atom=[["H", 0, 0, 0], ["Li", d, 0, 0]], basis='sto-3g', symmetry=True)
    hf = scf.RHF(mol)
    hf.kernel()

    mycas = mcscf.CASCI(hf, 3, 2)
    mo = mycas.sort_mo([2,3,6])  # 1-based indices
    mycas.kernel(mo)

    # Get integrals
    int1e, e_core = mycas.get_h1eff()
    int2e = mycas.get_h2eff()
    int2e_full = ao2mo.restore(1, int2e, 3)

    # Compute FCI
    fci_solver = fci.direct_spin0.FCI()
    e_fci, _ = fci_solver.kernel(int1e, int2e_full, 3, 2, ecore=e_core)

    return {
        'hf_energy': hf.e_tot,
        'casci_total': mycas.e_tot,
        'casci_active': mycas.e_cas,
        'e_core': e_core,
        'fci': e_fci,
        'int1e': int1e,
        'int2e_shape': int2e.shape,
        'method': 'benchmark (symmetry=True, sort_mo)'
    }

def current_method():
    """Current method with symmetry=False and manual core/frozen."""
    d = 2.0
    mol = gto.M(
        atom=f'H 0 0 0; Li {d} 0 0',
        basis='sto-3g',
        symmetry=False,
        verbose=0
    )
    hf = scf.RHF(mol)
    hf.kernel()

    # Active orbitals [1,2,5] 0-based
    active_orbitals = [1, 2, 5]
    n_orb = 3
    n_elec = 2
    nocc = mol.nelectron // 2

    core_orbitals = [i for i in range(nocc) if i not in active_orbitals]
    ncore = len(core_orbitals)

    total_orbs = hf.mo_coeff.shape[1]
    virtual_orbitals = list(range(nocc, total_orbs))
    frozen_virtual = [i for i in virtual_orbitals if i not in active_orbitals]

    cas = mcscf.CASCI(hf, n_orb, n_elec)
    cas.ncore = ncore
    cas.frozen = frozen_virtual
    cas.mo_coeff = hf.mo_coeff

    # Get integrals
    int1e, e_core = cas.get_h1eff()
    int2e = cas.get_h2eff()
    int2e_full = ao2mo.restore(1, int2e, n_orb)

    # Compute FCI
    fci_solver = fci.direct_spin0.FCI()
    e_fci, _ = fci_solver.kernel(int1e, int2e_full, n_orb, n_elec, ecore=e_core)

    # Also compute CASCI total
    cas.kernel()

    return {
        'hf_energy': hf.e_tot,
        'casci_total': cas.e_tot,
        'casci_active': cas.e_cas,
        'e_core': e_core,
        'fci': e_fci,
        'int1e': int1e,
        'int2e_shape': int2e.shape,
        'method': 'current (symmetry=False, manual core/frozen)'
    }

def compare_integrals(bench, curr):
    """Compare integrals between methods."""
    print("\n=== Integral Comparison ===")
    print(f"Core energy diff: {bench['e_core'] - curr['e_core']:.8f}")
    print(f"int1e shape: bench {bench['int1e'].shape}, curr {curr['int1e'].shape}")
    print(f"int1e max abs diff: {np.max(np.abs(bench['int1e'] - curr['int1e'])):.6e}")
    print(f"int2e shape: bench {bench['int2e_shape']}, curr {curr['int2e_shape']}")
    # Compare orbital energies?

def main():
    print("Comparing FCI calculation methods for LiH (2,3) active space")
    print("=" * 70)

    bench = benchmark_method()
    curr = current_method()

    print("\n=== Benchmark Method ===")
    print(f"HF energy: {bench['hf_energy']:.8f}")
    print(f"CASCI total: {bench['casci_total']:.8f}")
    print(f"CASCI active: {bench['casci_active']:.8f}")
    print(f"Core energy: {bench['e_core']:.8f}")
    print(f"FCI energy: {bench['fci']:.8f}")

    print("\n=== Current Method ===")
    print(f"HF energy: {curr['hf_energy']:.8f}")
    print(f"CASCI total: {curr['casci_total']:.8f}")
    print(f"CASCI active: {curr['casci_active']:.8f}")
    print(f"Core energy: {curr['e_core']:.8f}")
    print(f"FCI energy: {curr['fci']:.8f}")

    print("\n=== Differences ===")
    print(f"HF diff: {bench['hf_energy'] - curr['hf_energy']:.8e}")
    print(f"CASCI total diff: {bench['casci_total'] - curr['casci_total']:.8f} Ha")
    print(f"  ({abs(bench['casci_total'] - curr['casci_total'])*1000:.3f} mHa)")
    print(f"FCI diff: {bench['fci'] - curr['fci']:.8f} Ha")
    print(f"Core energy diff: {bench['e_core'] - curr['e_core']:.8f} Ha")

    compare_integrals(bench, curr)

    # Check against benchmark target
    target = -7.860153
    print(f"\n=== Comparison with target -7.860153 ===")
    print(f"Benchmark method diff: {bench['casci_total'] - target:.8f} Ha ({abs(bench['casci_total'] - target)*1000:.3f} mHa)")
    print(f"Current method diff: {curr['casci_total'] - target:.8f} Ha ({abs(curr['casci_total'] - target)*1000:.3f} mHa)")

    # Orbital energies
    print("\n=== Orbital energies (0-based) ===")
    # Need HF objects, but we can compute again
    d = 2.0
    mol_bench = gto.M(atom=[["H", 0, 0, 0], ["Li", d, 0, 0]], basis='sto-3g', symmetry=True)
    hf_bench = scf.RHF(mol_bench)
    hf_bench.kernel()
    print("Benchmark (symmetry=True):")
    for i, e in enumerate(hf_bench.mo_energy):
        print(f"  {i}: {e:.6f}")

    mol_curr = gto.M(atom=f'H 0 0 0; Li {d} 0 0', basis='sto-3g', symmetry=False, verbose=0)
    hf_curr = scf.RHF(mol_curr)
    hf_curr.kernel()
    print("\nCurrent (symmetry=False):")
    for i, e in enumerate(hf_curr.mo_energy):
        print(f"  {i}: {e:.6f}")

if __name__ == "__main__":
    main()