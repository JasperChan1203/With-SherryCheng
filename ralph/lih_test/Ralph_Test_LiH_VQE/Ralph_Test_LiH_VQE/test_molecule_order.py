#!/usr/bin/env python3
from pyscf import gto, scf
import numpy as np

bond_length = 2.0

print("Testing two different atom orders for LiH (bond length 2.0 Å)")
print("="*60)

# Order 1: Li at (0,0,0), H at (0,0,bond_length) - current implementation
print("\nOrder 1: Li (0,0,0), H (0,0,2.0)")
mol1 = gto.M(atom=f'Li 0 0 0; H 0 0 {bond_length}', basis='sto-3g', symmetry=False, verbose=0)
hf1 = scf.RHF(mol1)
hf1.kernel()
print(f"HF energy: {hf1.e_tot:.8f} Hartree")
print("Orbital energies (0-based indices):")
for i, e in enumerate(hf1.mo_energy):
    print(f"  {i}: {e:.6f} Ha")

# Order 2: H at (0,0,0), Li at (bond_length,0,0) - user suggested
print("\nOrder 2: H (0,0,0), Li (2.0,0,0)")
mol2 = gto.M(atom=f'H 0 0 0; Li {bond_length} 0 0', basis='sto-3g', symmetry=False, verbose=0)
hf2 = scf.RHF(mol2)
hf2.kernel()
print(f"HF energy: {hf2.e_tot:.8f} Hartree")
print("Orbital energies (0-based indices):")
for i, e in enumerate(hf2.mo_energy):
    print(f"  {i}: {e:.6f} Ha")

# Compare
print("\n" + "="*60)
print("Comparison:")
print(f"HF energy difference: {abs(hf1.e_tot - hf2.e_tot):.10f} Ha")
print("Orbital energy differences (absolute max):")
energy_diffs = [abs(hf1.mo_energy[i] - hf2.mo_energy[i]) for i in range(len(hf1.mo_energy))]
print(f"  Max diff: {max(energy_diffs):.10f} Ha")

# Check if orbital ordering might be different
print("\nNote: Even if energies match, orbital ordering/characteristics might differ.")
print("This could affect active orbital selection [2,3,6] (1-based).")

# Test with sort_mo
from pyscf.mcscf import sort_mo, CASCI
print("\n" + "="*60)
print("Testing sort_mo([2,3,6]) for both orders:")
for idx, (hf, label) in enumerate([(hf1, "Order 1"), (hf2, "Order 2")], 1):
    print(f"\n{label}:")
    cas = CASCI(hf, 3, 2)  # 3 orbitals, 2 electrons
    sorted_mo = sort_mo(cas, hf.mo_coeff, [2,3,6])  # 1-based indices
    # Check if the sorted orbitals correspond to expected original indices
    # This is complex; just note that sort_mo works
    print("  sort_mo([2,3,6]) applied successfully")
    print("  (Actual orbital reordering depends on implementation)")