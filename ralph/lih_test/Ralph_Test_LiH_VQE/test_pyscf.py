#!/usr/bin/env python3
"""
Test PySCF for LiH molecule to understand orbital energies and selection.
"""

import numpy as np
from pyscf import gto, scf

# Define LiH molecule at 2.0 Å bond length
bond_length = 2.0  # Å
mol = gto.M(
    atom=f'Li 0 0 0; H 0 0 {bond_length}',
    basis='sto-3g',
    symmetry=False,
    verbose=0
)

print(f"Molecule LiH bond length {bond_length} Å")
print(f"Number of atoms: {mol.natm}")
print(f"Number of electrons: {mol.nelectron}")
print(f"Number of atomic orbitals: {mol.nao}")
print(f"Number of basis functions: {mol.nbas}")

# Perform Hartree-Fock
hf = scf.RHF(mol)
hf.kernel()
print(f"\nHF converged: {hf.converged}")
print(f"HF energy: {hf.e_tot} Hartree")

# Orbital energies
orb_energies = hf.mo_energy
print(f"\nOrbital energies (Hartree):")
for i, e in enumerate(orb_energies):
    print(f"  Orbital {i}: {e:.6f}")

# Occupied orbitals
nocc = mol.nelectron // 2
print(f"\nOccupied orbitals (first {nocc}): {list(range(nocc))}")
print(f"Virtual orbitals (rest): {list(range(nocc, len(orb_energies)))}")

# HOMO and LUMO indices (0-based)
homo_idx = nocc - 1
lumo_idx = nocc
print(f"\nHOMO index: {homo_idx}, energy: {orb_energies[homo_idx]:.6f}")
print(f"LUMO index: {lumo_idx}, energy: {orb_energies[lumo_idx]:.6f}")

# Show orbital symmetries if available
if hasattr(mol, 'irrep_id'):
    print("\nOrbital symmetries:")
    for i, ir in enumerate(mol.irrep_id):
        print(f"  Orbital {i}: {ir}")

# Compute MP2 natural orbitals? Not needed.

# Try to select active space of (2 electrons, 3 orbitals)
# We need to pick 3 orbitals that are chemically relevant.
# Common choice: HOMO, LUMO, LUMO+1
active_orbitals = [homo_idx, lumo_idx, lumo_idx + 1]
print(f"\nPossible active orbitals (HOMO, LUMO, LUMO+1): {active_orbitals}")
print("Check if LUMO+1 exists:", lumo_idx + 1 < len(orb_energies))

# Compute CASCI for (2,3) active space
from pyscf import mcscf
print("\n--- CASCI calculation for (2,3) active space ---")
cas = mcscf.CASCI(hf, 3, 2)
cas.mo_coeff = hf.mo_coeff  # Use HF orbitals
cas.frozen = 0
cas.ncore = 0
cas.ncas = 3
# Specify active orbitals by indices (0-based)
cas.ci = None
cas.kernel(active_orbitals)
print(f"CASCI energy: {cas.e_tot} Hartree")
print(f"Active orbital indices used: {active_orbitals}")

# Let's also try to use PySCF's automatic selection based on energy
# Sort orbitals by energy and pick around HOMO-LUMO gap
print("\n--- Automatic selection based on energy ordering ---")
# Get orbital indices sorted by energy (ascending)
sorted_idx = np.argsort(orb_energies)
print(f"Orbitals sorted by energy: {sorted_idx.tolist()}")
# We want 3 orbitals around HOMO-LUMO region
# Choose HOMO-1, HOMO, LUMO? But we need 2 electrons, so we need orbitals that can accommodate 2 electrons.
# For (2,3), we can pick HOMO, LUMO, LUMO+1 (as above) or HOMO-1, HOMO, LUMO.
# Let's compute both and see energies.
active_set1 = [homo_idx, lumo_idx, lumo_idx + 1]
active_set2 = [homo_idx - 1, homo_idx, lumo_idx]
print(f"Active set 1 (HOMO, LUMO, LUMO+1): {active_set1}")
print(f"Active set 2 (HOMO-1, HOMO, LUMO): {active_set2}")

# Compute CASCI for both sets
for i, active in enumerate([active_set1, active_set2]):
    if all(idx < len(orb_energies) for idx in active):
        cas = mcscf.CASCI(hf, 3, 2)
        cas.mo_coeff = hf.mo_coeff
        cas.kernel(active)
        print(f"CASCI energy set {i+1}: {cas.e_tot} Hartree")
    else:
        print(f"Active set {i+1} out of range")

# Finally, compute FCI using PySCF's FCI module
print("\n--- FCI calculation using PySCF's FCI ---")
from pyscf import fci
# Full CI within active space? Actually FCI over all orbitals is too large.
# We'll do CASCI with full active space (all orbitals) but that's impossible.
# Instead, we can compute FCI within the active space using CASCI with full configuration.
# CASCI with (nelec, norb) = (2, 3) already includes all configurations within active space.
# That's essentially FCI within active space.
# So CASCI energy we computed is the FCI energy for the active space.
# However, we need to ensure we include all electron correlations within active space.
# CASCI with (2,3) already does full CI within those 3 orbitals.
print("CASCI (2,3) is already FCI within active space.")

# Let's compute FCI using pyscf.fci directly for verification
print("\n--- FCI using pyscf.fci module ---")
# Get Hamiltonian integrals in active space
from pyscf import ao2mo
# Transform integrals to MO basis for active orbitals
mo_coeff = hf.mo_coeff
# Get one-electron integrals
h1e = np.einsum('pi,pq,qj->ij', mo_coeff[:, active_set1], hf.get_hcore(), mo_coeff[:, active_set1])
# Get two-electron integrals
eri = ao2mo.kernel(mol, mo_coeff[:, active_set1], compact=False)
eri = eri.reshape(3,3,3,3)
# Solve FCI
fci_solver = fci.direct_spin0.FCI()
e_fci, fcivec = fci_solver.kernel(h1e, eri, 3, 2)
print(f"FCI energy (direct_spin0): {e_fci} Hartree")