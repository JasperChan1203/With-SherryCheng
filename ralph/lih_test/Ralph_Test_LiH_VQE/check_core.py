#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals, compute_fci_energy
from pyscf import mcscf, fci, ao2mo
import numpy as np

mol, hf = define_molecule()
print(f"HF total energy: {hf.e_tot:.8f}")
active_orbitals, _, _ = select_active_orbitals(hf)
print(f"Active orbitals: {active_orbitals}")

# Compute CASCI total energy using PySCF CASCI with proper ncore and frozen
n_orb = len(active_orbitals)
n_elec = 2
cas = mcscf.CASCI(hf, n_orb, n_elec)
nocc = mol.nelectron // 2
core_orbitals = [i for i in range(nocc) if i not in active_orbitals]
virtual_orbitals = list(range(nocc, hf.mo_coeff.shape[1]))
frozen_virtual = [i for i in virtual_orbitals if i not in active_orbitals]
cas.ncore = len(core_orbitals)
cas.frozen = frozen_virtual
cas.mo_coeff = hf.mo_coeff
cas.kernel()
print(f"CASCI total energy: {cas.e_tot:.8f}")
# Get integrals from CASCI
int1e, e_core_cas = cas.get_h1eff()
int2e = cas.get_h2eff()
int2e_full = ao2mo.restore(1, int2e, n_orb)
print(f"CASCI core energy: {e_core_cas:.8f}")
print(f"CASCI active energy (e_cas): {cas.e_cas:.8f}")
print(f"Sum core + active: {e_core_cas + cas.e_cas:.8f}")

# Compute FCI using fci module with CASCI integrals
fci_solver = fci.direct_spin0.FCI()
e_fci_with_core, _ = fci_solver.kernel(int1e, int2e_full, n_orb, n_elec, ecore=e_core_cas)
print(f"FCI energy (with ecore): {e_fci_with_core:.8f}")
print(f"Difference from CASCI total: {e_fci_with_core - cas.e_tot:.8f}")

# Now use our get_active_integrals
int1e_our, int2e_our, e_core_our = get_active_integrals(hf, active_orbitals)
int2e_full_our = ao2mo.restore(1, int2e_our, n_orb)
print(f"\nOur get_active_integrals:")
print(f"  e_core: {e_core_our:.8f}")
print(f"  Difference from CASCI core: {e_core_our - e_core_cas:.8f}")
# Compute FCI with our integrals
e_fci_our, _ = fci_solver.kernel(int1e_our, int2e_full_our, n_orb, n_elec, ecore=e_core_our)
print(f"  FCI energy with our integrals: {e_fci_our:.8f}")
print(f"  Difference from CASCI total: {e_fci_our - cas.e_tot:.8f}")

# Compute compute_fci_energy result
e_fci_func = compute_fci_energy(hf, active_orbitals)
print(f"\ncompute_fci_energy result: {e_fci_func:.8f}")
print(f"Difference from CASCI total: {e_fci_func - cas.e_tot:.8f}")

# Check if our integrals match CASCI integrals
print(f"\nIntegral diff max: {np.max(np.abs(int1e_our - int1e))}")
print(f"int2e diff max: {np.max(np.abs(int2e_full_our - int2e_full))}")