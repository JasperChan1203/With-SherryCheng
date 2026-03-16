#!/usr/bin/env python3
"""
Debug specific orbital combinations to understand errors.
"""

import traceback
from pyscf import gto, scf, mcscf, fci, ao2mo

def test_combination(active_orbitals):
    """Test a specific combination and print details."""
    print(f"\nTesting combination: {active_orbitals} (1-based {[idx+1 for idx in active_orbitals]})")

    # Define molecule
    bond_length = 2.0
    mol = gto.M(
        atom=f'H 0 0 0; Li {bond_length} 0 0',
        basis='sto-3g',
        symmetry=False,
        verbose=0
    )
    hf = scf.RHF(mol)
    hf.kernel()

    n_orb = len(active_orbitals)
    n_elec = 2  # active electrons
    nocc = hf.mol.nelectron // 2
    total_orbs = hf.mo_coeff.shape[1]

    print(f"  HF energy: {hf.e_tot:.8f}")
    print(f"  Occupied orbitals: {list(range(nocc))}")
    print(f"  Active orbitals: {active_orbitals}")

    # Determine core orbitals (occupied orbitals not in active)
    core_orbitals = [i for i in range(nocc) if i not in active_orbitals]
    ncore = len(core_orbitals)

    # Determine frozen virtual orbitals (virtual orbitals not in active)
    virtual_orbitals = list(range(nocc, total_orbs))
    frozen_virtual = [i for i in virtual_orbitals if i not in active_orbitals]

    print(f"  Core orbitals (frozen occupied): {core_orbitals}, ncore = {ncore}")
    print(f"  Frozen virtual orbitals: {frozen_virtual}")

    # Check if combination is valid
    if ncore < 0:
        print("  ERROR: ncore < 0")
        return
    if ncore > nocc:
        print(f"  ERROR: ncore ({ncore}) > nocc ({nocc})")
        return

    # Create CASCI object
    try:
        cas = mcscf.CASCI(hf, n_orb, n_elec)
        cas.ncore = ncore
        cas.frozen = frozen_virtual
        cas.mo_coeff = hf.mo_coeff

        print(f"  CASCI created: ncore={cas.ncore}, frozen={cas.frozen}")

        # Try to get integrals
        int1e, e_core = cas.get_h1eff()
        int2e = cas.get_h2eff()
        int2e_full = ao2mo.restore(1, int2e, n_orb)

        print(f"  Core energy: {e_core:.8f}")
        print(f"  int1e shape: {int1e.shape}")

        # Compute FCI
        fci_solver = fci.direct_spin0.FCI()
        e_fci, _ = fci_solver.kernel(int1e, int2e_full, n_orb, n_elec, ecore=e_core)

        print(f"  FCI energy: {e_fci:.8f}")

        # Also compute CASCI energy
        cas.kernel()
        print(f"  CASCI total energy: {cas.e_tot:.8f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

# Test problematic combinations
print("Testing problematic orbital combinations")
print("="*60)

# Combinations that failed in the search
problematic = [
    [0, 1, 2],  # Two occupied + one virtual
    [0, 1, 3],
    [0, 1, 4],  # Benchmark combination [0,1,4] -> 1-based [1,2,5]
    [0, 1, 5],
    [2, 3, 4],  # All virtual
    [2, 3, 5],
    [2, 4, 5],
    [3, 4, 5]
]

for combo in problematic:
    test_combination(combo)

# Also test a working combination for comparison
print("\n" + "="*60)
print("Testing a working combination for comparison")
test_combination([0, 2, 3])  # Should work