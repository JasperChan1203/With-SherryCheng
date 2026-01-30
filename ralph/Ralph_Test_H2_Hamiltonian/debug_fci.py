#!/usr/bin/env python
import numpy as np
import pyscf
from pyscf import gto, scf, fci, ci

# Define molecule
mol = gto.M(
    atom=[["H", 0, 0, 0], ["H", 0.5, 0, 0]],
    basis="sto-3g",
    unit="angstrom",
    symmetry=False,
    verbose=0
)

# RHF
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
hf_energy = mf.kernel()
print(f"HF energy: {hf_energy:.10f}")

# Method 1: FCI using fci.FCI(mf)
print("\nMethod 1: fci.FCI(mf)")
fci_solver = fci.FCI(mf)
fci_energy1, fci_vec1 = fci_solver.kernel()
print(f"FCI energy: {fci_energy1:.10f}")

# Method 2: fci.FCI(mol) with integrals
print("\nMethod 2: fci.FCI(mol) with integrals")
h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
h2_ao = mol.intor("int2e", aosym="s8")
mo_coeff = mf.mo_coeff
h1_mo = np.einsum("pi,pq,qj->ij", mo_coeff, h1_ao, mo_coeff)
from pyscf import ao2mo
h2_mo = ao2mo.full(mol, mo_coeff)
norb = mo_coeff.shape[1]
nelec = mol.nelectron
fci_solver2 = fci.FCI(mol)
fci_energy2, fci_vec2 = fci_solver2.kernel(h1_mo, h2_mo, norb, nelec)
print(f"FCI energy: {fci_energy2:.10f}")

# Method 3: CISD (should be exact for H2/sto-3g)
print("\nMethod 3: CISD")
cisolver = ci.CISD(mf)
cisolver.kernel()
cisd_energy = mf.e_tot + cisolver.e_corr
print(f"CISD total energy: {cisd_energy:.10f}")
print(f"CISD correlation energy: {cisolver.e_corr:.10f}")

# Method 4: Direct FCI using pyscf.fci module with correct nelec tuple
print("\nMethod 4: FCI with nelec=(1,1)")
fci_energy4, fci_vec4 = fci.FCI(mol).kernel(h1_mo, h2_mo, norb, (1,1))
print(f"FCI energy (nelec=(1,1)): {fci_energy4:.10f}")

# Compare with benchmark
print("\nBenchmark FCI energy: -1.0551597944706257")
print(f"Difference method1: {fci_energy1 - (-1.0551597944706257):.2e}")
print(f"Difference method2: {fci_energy2 - (-1.0551597944706257):.2e}")
print(f"Difference method3: {cisd_energy - (-1.0551597944706257):.2e}")
print(f"Difference method4: {fci_energy4 - (-1.0551597944706257):.2e}")