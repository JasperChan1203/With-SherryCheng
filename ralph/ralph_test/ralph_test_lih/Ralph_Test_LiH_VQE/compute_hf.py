#!/usr/bin/env python3
from pyscf import gto, scf

bond_length = 2.0
mol = gto.M(
    atom=[["H", 0, 0, 0], ["Li", bond_length, 0, 0]],
    basis='sto-3g',
    symmetry=True,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
print(f"HF energy: {hf.e_tot:.8f} Hartree")
print(f"MO energies: {hf.mo_energy}")
print(f"MO occupations: {hf.mo_occ}")
print(f"Number of spatial orbitals: {hf.mo_energy.shape[0]}")