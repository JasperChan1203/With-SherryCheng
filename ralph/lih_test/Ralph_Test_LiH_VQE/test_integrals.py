#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals, compute_fci_energy

mol, hf = define_molecule()
print(f"HF energy: {hf.e_tot:.8f}")
active_orbitals, active_energies, just = select_active_orbitals(hf)
print(f"Active orbitals (0-idx): {active_orbitals}")
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)
print(f"int1e shape: {int1e.shape}")
print(f"int2e shape: {int2e.shape}")
print(f"e_core: {e_core:.8f}")
print(f"Sum of int1e diagonal: {int1e.trace():.8f}")
print(f"Total HF energy? {hf.e_tot:.8f}")

# Compute FCI
fci_energy = compute_fci_energy(hf, active_orbitals)
print(f"FCI energy from compute_fci_energy: {fci_energy:.8f}")

# Compute CASCI total energy using PySCF CASCI
from pyscf import mcscf
cas = mcscf.CASCI(hf, len(active_orbitals), 2)
cas.frozen = [i for i in range(hf.mo_coeff.shape[1]) if i not in active_orbitals]
cas.ncore = 0
cas.mo_coeff = hf.mo_coeff
cas.kernel()
print(f"CASCI total energy: {cas.e_tot:.8f}")
print(f"CASCI core energy: {cas.e_core:.8f}")
print(f"CASCI active space energy: {cas.e_cas:.8f}")

# Compute total energy = e_core + fci_energy (should match CASCI total)
print(f"e_core + fci_energy: {e_core + fci_energy:.8f}")
print(f"Difference: {cas.e_tot - (e_core + fci_energy):.8f}")