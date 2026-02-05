#!/usr/bin/env python3
from pyscf import gto, scf, mcscf
import numpy as np

bond_length = 2.0
mol = gto.M(atom=f'Li 0 0 0; H 0 0 {bond_length}', basis='sto-3g', symmetry=False, verbose=0)
hf = scf.RHF(mol)
hf.kernel()

print("Original orbital energies:")
for i, e in enumerate(hf.mo_energy):
    print(f"  {i} (1-based {i+1}): {e:.6f}")

# Active orbitals: 0-based [1,2,5] -> 1-based [2,3,6]
active_orbitals = [1, 2, 5]
sort_list = [idx+1 for idx in active_orbitals]  # [2,3,6]

n_orb = len(active_orbitals)
n_elec = 2

# Create CASCI
cas = mcscf.CASCI(hf, n_orb, n_elec)

# Apply sort_mo
from pyscf.mcscf import sort_mo
sorted_mo = sort_mo(cas, hf.mo_coeff, sort_list)
cas.mo_coeff = sorted_mo

# Check order after sort_mo
print("\nAfter sort_mo([2,3,6]):")
print("Assuming specified orbitals are moved to appropriate positions.")
print("Typically, sort_mo places specified orbitals after core orbitals.")

# Try to determine positions
# Simple approach: compare columns (not perfect but informative)
print("\nChecking column matching (may have sign differences):")
for j in range(hf.mo_coeff.shape[1]):
    found = False
    for i in range(hf.mo_coeff.shape[1]):
        if np.allclose(np.abs(sorted_mo[:, j]), np.abs(hf.mo_coeff[:, i])):
            print(f"  Sorted column {j} matches original orbital {i} (1-based {i+1})")
            found = True
            break
    if not found:
        print(f"  Sorted column {j} no exact match")

# Try with ncore=1, frozen=[]
cas.ncore = 1
cas.frozen = []
cas.kernel()
print(f"\nCASCI with ncore=1, frozen=[]: total energy = {cas.e_tot:.8f}")
print(f"Core energy from get_h1eff: {cas.get_h1eff()[1]:.8f}")

# Try with ncore=0, frozen=[3,4,5] (assuming positions)
cas2 = mcscf.CASCI(hf, n_orb, n_elec)
cas2.mo_coeff = sorted_mo
cas2.ncore = 0
cas2.frozen = [3, 4, 5]  # guess
cas2.kernel()
print(f"\nCASCI with ncore=0, frozen=[3,4,5]: total energy = {cas2.e_tot:.8f}")
print(f"Core energy from get_h1eff: {cas2.get_h1eff()[1]:.8f}")