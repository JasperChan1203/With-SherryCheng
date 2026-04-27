#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf

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
mo_energy = hf.mo_energy
print("Orbital energies:")
for i, e in enumerate(mo_energy):
    print(f"  {i}: {e:.6f}")

# Benchmark orbitals (1-indexed) [1,2,5] -> 0-indexed [0,1,4]
bench_active = [0,1,4]
print(f"\nBenchmark active orbitals (0-indexed): {bench_active}")

# Compute CASCI with active_space=(2,3) and these orbitals
# PySCF's CASCI can accept list of active orbitals
cas = mcscf.CASCI(hf, 3, 2)  # 3 orbitals, 2 electrons
cas.mo_coeff = hf.mo_coeff
# No frozen orbitals? Let's see what happens
cas.kernel(bench_active)
print(f"CASCI total energy with benchmark orbitals: {cas.e_tot:.8f}")
print(f"CASCI active space energy: {cas.e_cas:.8f}")
print(f"CASCI core energy: {cas.e_core:.8f}")

# Now compute with our selection (HOMO, LUMO, LUMO+1)
our_active = [1,2,3]
cas2 = mcscf.CASCI(hf, 3, 2)
cas2.mo_coeff = hf.mo_coeff
cas2.kernel(our_active)
print(f"\nOur selection active orbitals: {our_active}")
print(f"CASCI total energy: {cas2.e_tot:.8f}")
print(f"CASCI active space energy: {cas2.e_cas:.8f}")
print(f"CASCI core energy: {cas2.e_core:.8f}")

# Compare with benchmark FCI energy
bench_fci = -7.860153
print(f"\nBenchmark FCI: {bench_fci:.8f}")
print(f"Difference (benchmark orbitals): {cas.e_tot - bench_fci:.8f} Ha")
print(f"Difference (our orbitals): {cas2.e_tot - bench_fci:.8f} Ha")

# Let's also compute FCI within active space using fci module
from pyscf import ao2mo, fci
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
int2e_full = ao2mo.restore(1, int2e, 3)
fci_solver = fci.direct_spin0.FCI()
e_fci, _ = fci_solver.kernel(int1e, int2e_full, 3, 2, ecore=e_core)
print(f"\nFCI energy with benchmark orbitals (should match CASCI total): {e_fci:.8f}")