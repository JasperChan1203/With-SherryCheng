#!/usr/bin/env python3
"""
Search for active orbital combinations that yield FCI energy close to -7.86 Hartree.
Examines all possible 3-orbital combinations from 6 spatial orbitals.
"""

import itertools
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
import json

def compute_fci_for_combination(hf, active_orbitals):
    """Compute FCI energy for given active orbital indices (0-based)."""
    n_orb = len(active_orbitals)
    n_elec = 2  # active electrons
    nocc = hf.mol.nelectron // 2

    # Determine core orbitals (occupied orbitals not in active)
    core_orbitals = [i for i in range(nocc) if i not in active_orbitals]
    ncore = len(core_orbitals)

    # Determine frozen virtual orbitals (virtual orbitals not in active)
    total_orbs = hf.mo_coeff.shape[1]
    virtual_orbitals = list(range(nocc, total_orbs))
    frozen_virtual = [i for i in virtual_orbitals if i not in active_orbitals]

    # Create CASCI object
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

    # Also compute CASCI energy for verification
    cas.kernel()

    return e_fci, cas.e_tot, e_core

def main():
    print("Searching for optimal active orbital combinations for LiH")
    print("Target FCI energy: ~-7.860153 Hartree (benchmark)")
    print("=" * 70)

    # Define molecule (H at origin, Li at 2.0 Å)
    bond_length = 2.0
    mol = gto.M(
        atom=f'H 0 0 0; Li {bond_length} 0 0',
        basis='sto-3g',
        symmetry=False,
        verbose=0
    )
    hf = scf.RHF(mol)
    hf.kernel()

    print(f"HF energy: {hf.e_tot:.8f} Hartree")
    print(f"Number of spatial orbitals: {hf.mo_coeff.shape[1]}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Occupied orbitals (0-based): {list(range(mol.nelectron // 2))}")

    print("\nOrbital energies (0-based):")
    for i, e in enumerate(hf.mo_energy):
        print(f"  {i}: {e:.6f} Ha")

    # Generate all combinations of 3 orbitals from 6
    total_orbs = hf.mo_coeff.shape[1]
    all_combinations = list(itertools.combinations(range(total_orbs), 3))

    print(f"\nTotal combinations to test: {len(all_combinations)}")
    print("=" * 70)

    results = []
    target_energy = -7.860153  # Benchmark value

    for i, combo in enumerate(all_combinations):
        active_orbitals = list(combo)

        try:
            e_fci, casci_total, e_core = compute_fci_for_combination(hf, active_orbitals)
            diff = abs(e_fci - target_energy)

            result = {
                "combo_0based": active_orbitals,
                "combo_1based": [idx+1 for idx in active_orbitals],
                "fci_energy": e_fci,
                "casci_total": casci_total,
                "core_energy": e_core,
                "diff_from_target": diff,
                "orbitals": {
                    idx: {
                        "energy": hf.mo_energy[idx],
                        "occupied": idx < (mol.nelectron // 2)
                    }
                    for idx in active_orbitals
                }
            }
            results.append(result)

            energies = [hf.mo_energy[idx] for idx in active_orbitals]
            print(f"Combination {i+1:2d}: {active_orbitals} (1-based {[idx+1 for idx in active_orbitals]})")
            print(f"  FCI energy: {e_fci:.8f} Ha, Diff from target: {diff:.8f} Ha ({diff*1000:.3f} mHa)")
            print(f"  Orbital energies: {[f'{e:.6f}' for e in energies]}")
            print()

        except Exception as e:
            print(f"Combination {i+1:2d}: {active_orbitals} - ERROR: {e}")
            print()

    # Sort by closeness to target
    results.sort(key=lambda x: x["diff_from_target"])

    print("=" * 70)
    print("TOP COMBINATIONS (closest to target -7.860153 Hartree):")
    print("=" * 70)

    for i, result in enumerate(results[:10]):
        energies = [hf.mo_energy[idx] for idx in result['combo_0based']]
        print(f"{i+1:2d}. Orbitals (0-based): {result['combo_0based']}")
        print(f"    Orbitals (1-based): {result['combo_1based']}")
        print(f"    FCI energy: {result['fci_energy']:.8f} Ha")
        print(f"    Diff from target: {result['diff_from_target']:.8f} Ha ({result['diff_from_target']*1000:.3f} mHa)")
        print(f"    Orbital energies: {energies}")
        print()

    # Save detailed results to JSON
    output_data = {
        "molecule": {
            "formula": "LiH",
            "bond_length_angstrom": bond_length,
            "basis_set": "sto-3g",
            "hf_energy": hf.e_tot,
            "orbital_energies": hf.mo_energy.tolist()
        },
        "target_fci_energy": target_energy,
        "results": results
    }

    with open("orbital_search_results.json", "w") as f:
        json.dump(output_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float64) else x)

    print(f"\nDetailed results saved to: orbital_search_results.json")

    # Best combination
    best = results[0]
    print("\n" + "=" * 70)
    print("BEST COMBINATION FOUND:")
    print(f"  Orbitals (0-based): {best['combo_0based']}")
    print(f"  Orbitals (1-based): {best['combo_1based']}")
    print(f"  FCI energy: {best['fci_energy']:.8f} Ha")
    print(f"  Difference from target: {best['diff_from_target']:.8f} Ha ({best['diff_from_target']*1000:.3f} mHa)")

    # Compare with current selection
    current = [1, 2, 5]  # 0-based [1,2,5] -> 1-based [2,3,6]
    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT SELECTION:")
    print(f"  Current orbitals (0-based): {current}")
    print(f"  Current orbitals (1-based): {[idx+1 for idx in current]}")

    # Find current in results
    current_result = None
    for res in results:
        if res["combo_0based"] == current:
            current_result = res
            break

    if current_result:
        print(f"  Current FCI energy: {current_result['fci_energy']:.8f} Ha")
        print(f"  Current diff from target: {current_result['diff_from_target']:.8f} Ha ({current_result['diff_from_target']*1000:.3f} mHa)")
        print(f"  Improvement with best: {current_result['diff_from_target'] - best['diff_from_target']:.8f} Ha")
    else:
        print("  Current combination not found in results (should not happen)")

if __name__ == "__main__":
    main()