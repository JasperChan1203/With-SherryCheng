#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals, compute_fci_energy

print("Testing with new molecular order: H at (0,0,0), Li at (2.0,0,0)")
print("="*70)

mol, hf = define_molecule()
print(f"HF energy: {hf.e_tot:.8f} Hartree")
print(f"Number of electrons: {mol.nelectron}")
print(f"Number of spatial orbitals: {hf.mo_coeff.shape[1]}")

print("\nOrbital energies (0-based indices):")
for i, e in enumerate(hf.mo_energy):
    print(f"  {i} (1-based {i+1}): {e:.6f} Ha")

active_orbitals, active_energies, justification = select_active_orbitals(hf)
print(f"\nSelected active orbitals (0-based): {active_orbitals}")
print(f"Active orbitals (1-based): {[idx+1 for idx in active_orbitals]}")
print(f"Active orbital energies: {active_energies}")
print(f"Justification: {justification}")

# Check which orbitals these are
nocc = mol.nelectron // 2
print("\nAnalysis of selected orbitals:")
for idx in active_orbitals:
    if idx < nocc:
        occ_status = "occupied (HOMO)" if idx == nocc-1 else "occupied"
    else:
        occ_status = "virtual"
    print(f"  Orbital {idx} (1-based {idx+1}): energy {hf.mo_energy[idx]:.6f}, {occ_status}")

# Compute FCI
print("\n" + "="*70)
print("Computing FCI energy...")
try:
    fci_energy = compute_fci_energy(hf, active_orbitals)
    print(f"FCI energy: {fci_energy:.8f} Hartree")
    print(f"Difference from benchmark (-7.860153): {fci_energy - (-7.860153):.8f} Ha")
except Exception as e:
    print(f"Error computing FCI: {e}")
    import traceback
    traceback.print_exc()

# Test integrals
print("\n" + "="*70)
print("Testing integral computation...")
try:
    int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)
    print(f"Core energy: {e_core:.8f}")
    print(f"int1e shape: {int1e.shape}")
    print(f"int2e shape: {int2e.shape}")
except Exception as e:
    print(f"Error computing integrals: {e}")
    import traceback
    traceback.print_exc()