#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci

# Define LiH molecule
bond_length = 2.0
mol = gto.M(
    atom=f'Li 0 0 0; H 0 0 {bond_length}',
    basis='sto-3g',
    symmetry=False,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
print(f"HF energy: {hf.e_tot:.8f} Hartree")
print(f"Nuclear repulsion: {mol.energy_nuc()} Hartree")
print(f"Number of electrons: {mol.nelectron}")
print(f"Number of spatial orbitals: {hf.mo_coeff.shape[1]}")

# Orbital energies
mo_energy = hf.mo_energy
for i, e in enumerate(mo_energy):
    print(f"  Orbital {i}: {e:.6f}")
nocc = mol.nelectron // 2
print(f"Occupied orbitals: {list(range(nocc))}")

# Select active orbitals (HOMO, LUMO, LUMO+1)
homo_idx = nocc - 1
lumo_idx = nocc
active_orbitals = [homo_idx, lumo_idx, lumo_idx + 1]
print(f"Selected active orbitals (0-indexed): {active_orbitals}")
print(f"Orbital energies: {mo_energy[active_orbitals]}")

# Compute CASCI for (2,3) active space using PySCF's CASCI
print("\n--- CASCI calculation ---")
cas = mcscf.CASCI(hf, 3, 2)
cas.frozen = list(range(hf.mo_coeff.shape[1]))  # freeze all
cas.frozen = [i for i in range(hf.mo_coeff.shape[1]) if i not in active_orbitals]
cas.ncore = 0
cas.mo_coeff = hf.mo_coeff
cas.kernel()
print(f"CASCI total energy: {cas.e_tot:.8f}")
print(f"CASCI CI energy (within active space): {cas.e_cas:.8f}")
print(f"Core energy (e_core): {cas.e_core:.8f}")

# Compute integrals for active space using get_h1eff and get_h2eff
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
print(f"\nIntegrals from CASCI:")
print(f"  int1e shape: {int1e.shape}")
print(f"  e_core: {e_core}")
print(f"  int2e shape: {int2e.shape}")

# Compute FCI using pyscf.fci directly
print("\n--- FCI using pyscf.fci ---")
int2e_full = ao2mo.restore(1, int2e, 3)  # no symmetry
fci_solver = fci.direct_spin0.FCI()
e_fci, fcivec = fci_solver.kernel(int1e, int2e_full, 3, 2, ecore=e_core)
print(f"FCI energy (with ecore): {e_fci:.8f}")
print(f"FCI energy (without ecore): {fci_solver.kernel(int1e, int2e_full, 3, 2, ecore=0.0)[0]:.8f}")

# Compute total energy by adding core energy manually
print("\n--- Manual calculation ---")
# Compute Hamiltonian matrix elements?
# Let's compute CASCI energy again with full configuration
# Use pyscf.mcscf.CASCI to compute full CI within active space (already done)
print(f"CASCI total energy (should match FCI+core): {cas.e_tot:.8f}")
print(f"Difference: {cas.e_tot - e_fci:.8f}")

# Compare with benchmark
bench_fci = -7.860153
print(f"\nBenchmark FCI: {bench_fci:.8f}")
print(f"Difference from benchmark: {e_fci - bench_fci:.8f} Hartree, {abs(e_fci - bench_fci)*1000:.3f} mHa")

# Check orbital indices: benchmark expects orbitals [1,2,5] (1-indexed)
# Our selected orbitals (0-indexed) are [1,2,3] -> 1-indexed [2,3,4]
# Let's see orbital energies for orbitals 0-5
print("\nOrbital energies (0-indexed):")
for i in range(6):
    print(f"  {i}: {mo_energy[i]:.6f}")
print("Orbitals sorted by energy:", np.argsort(mo_energy))
# Maybe natural orbitals differ. Let's compute MP2 natural orbitals? Not needed.
# But we need to justify selection chemically.
# For LiH, bonding/antibonding orbitals maybe different ordering.
# Let's compute Mulliken population to see character?
print("\n--- Mulliken population analysis ---")
from pyscf import lo
pop = lo.orth_ao(mol, 'mulliken')
print("Mulliken population not computed.")