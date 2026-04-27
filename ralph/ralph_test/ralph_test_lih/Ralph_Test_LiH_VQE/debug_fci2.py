#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci

bond_length = 2.0
mol = gto.M(
    atom=f'Li 0 0 0; H 0 0 {bond_length}',
    basis='sto-3g',
    symmetry=False,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
print(f"HF energy: {hf.e_tot:.8f}")
print(f"Nuclear repulsion: {mol.energy_nuc():.8f}")
print(f"Electron count: {mol.nelectron}")

mo_energy = hf.mo_energy
nocc = mol.nelectron // 2
print(f"Occupied orbitals: {list(range(nocc))}")
print("Orbital energies:")
for i, e in enumerate(mo_energy):
    print(f"  {i}: {e:.6f}")

# Active orbitals: HOMO, LUMO, LUMO+1
active_orbitals = [nocc-1, nocc, nocc+1]
print(f"Active orbitals (0-indexed): {active_orbitals}")

# Compute CASCI with proper freezing
# Core orbitals: orbital 0 (doubly occupied)
# Virtual orbitals to freeze: orbitals 4,5
core_orbitals = [0]
virtual_frozen = [i for i in range(mo_energy.size) if i not in active_orbitals and i not in core_orbitals]
print(f"Core orbitals: {core_orbitals}")
print(f"Virtual frozen: {virtual_frozen}")

# Method 1: set ncore=1, frozen=virtual_frozen
cas = mcscf.CASCI(hf, 3, 2)
cas.ncore = 1
cas.frozen = virtual_frozen
cas.mo_coeff = hf.mo_coeff
cas.kernel()
print(f"\nCASCI total energy (ncore=1, frozen virtual): {cas.e_tot:.8f}")
print(f"CASCI core energy: {cas.e_core:.8f}")
print(f"CASCI active space energy: {cas.e_cas:.8f}")

# Get integrals for active space
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
int2e_full = ao2mo.restore(1, int2e, 3)
print(f"\nIntegrals from CASCI:")
print(f"  e_core: {e_core:.8f}")
print(f"  int1e shape: {int1e.shape}")
print(f"  int2e shape: {int2e.shape}")

# Compute FCI within active space using fci module
fci_solver = fci.direct_spin0.FCI()
e_fci, _ = fci_solver.kernel(int1e, int2e_full, 3, 2, ecore=e_core)
print(f"FCI energy (with ecore): {e_fci:.8f}")
print(f"Should match CASCI total: {cas.e_tot:.8f}, diff = {e_fci - cas.e_tot:.8f}")

# Method 2: use frozen list (all orbitals except active) and ncore=0 (but need to account core electrons)
# Let's compute total energy using full CASCI with frozen list
cas2 = mcscf.CASCI(hf, 3, 2)
cas2.frozen = core_orbitals + virtual_frozen
cas2.ncore = 0
cas2.mo_coeff = hf.mo_coeff
cas2.kernel()
print(f"\nCASCI total energy (frozen all except active): {cas2.e_tot:.8f}")
print(f"CASCI core energy: {cas2.e_core:.8f}")

# Compare with benchmark
bench = -7.860153
print(f"\nBenchmark FCI: {bench:.8f}")
print(f"Difference: {e_fci - bench:.8f} Ha, {abs(e_fci - bench)*1000:.3f} mHa")

# Compute HF energy decomposition
print(f"\nHF energy decomposition:")
print(f"  Nuclear repulsion: {mol.energy_nuc():.8f}")
print(f"  Electronic energy: {hf.e_tot - mol.energy_nuc():.8f}")

# Compute energy of frozen core orbital (orbital 0) contribution
# Use one-electron and two-electron integrals
# Not necessary now.