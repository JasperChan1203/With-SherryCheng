#!/usr/bin/env python3
"""Test parity mapping consistency."""

import sys
sys.path.append("../001")
sys.path.append("../002")

from src.modules.molecule_processor import process_molecule
import tencirchem
from tencirchem import UCCSD
from pyscf import gto, scf

print("Processing H2 molecule...")
molecule_data = process_molecule("H2", 0.74, "UCC")
print(f"Transform in molecular_info: {molecule_data.molecular_info['transform']}")

# Build PySCF molecule as in circuit builder
mol = gto.M(
    atom=[('H', 0.0, 0.0, 0.0), ('H', 0.74, 0.0, 0.0)],
    basis='sto-3g',
    unit='angstrom',
    symmetry=False,
    verbose=0,
)
hf = scf.RHF(mol)
hf.kernel()

# Create UCCSD with parity transform
print("\nCreating UCCSD with parity transform...")
ucc_parity = UCCSD(mol, transform='parity')
print(f"Number of parameters: {ucc_parity.n_params}")
print(f"Excitation ops: {ucc_parity.ex_ops}")
print(f"Hamiltonian constant term? {ucc_parity.hamiltonian.terms.get((), 0)}")

# Create UCCSD with default (jordan-wigner)
print("\nCreating UCCSD with jordan-wigner transform...")
ucc_jw = UCCSD(mol, transform='jordan-wigner')
print(f"Number of parameters: {ucc_jw.n_params}")
print(f"Excitation ops: {ucc_jw.ex_ops}")
print(f"Hamiltonian constant term? {ucc_jw.hamiltonian.terms.get((), 0)}")

# Compare Hamiltonian constant term with molecule_data.hamiltonian constant term
const_molecule = molecule_data.hamiltonian.terms.get((), 0)
print(f"\nMolecule data Hamiltonian constant term: {const_molecule}")
print(f"Parity Hamiltonian constant term: {ucc_parity.hamiltonian.terms.get((), 0)}")
print(f"JW Hamiltonian constant term: {ucc_jw.hamiltonian.terms.get((), 0)}")

# Compute energy with zero parameters
energy_parity_zero = ucc_parity.energy(np.zeros(ucc_parity.n_params))
energy_jw_zero = ucc_jw.energy(np.zeros(ucc_jw.n_params))
print(f"\nZero-parameter energy parity: {energy_parity_zero}")
print(f"Zero-parameter energy JW: {energy_jw_zero}")
print(f"HF energy from molecular_info: {molecule_data.molecular_info['hf_energy']}")

# Compare excitation operators
print("\nExcitation operators parity vs JW:")
print(f"Same? {ucc_parity.ex_ops == ucc_jw.ex_ops}")

# Check if parity transform matches molecule_data.hamiltonian mapping
print("\nChecking if parity Hamiltonian matches molecule_data.hamiltonian...")
ham_parity = ucc_parity.hamiltonian
ham_molecule = molecule_data.hamiltonian
print(f"Number of terms parity: {len(ham_parity.terms)}")
print(f"Number of terms molecule: {len(ham_molecule.terms)}")
# Compare first few terms
print("First 5 terms parity:", list(ham_parity.terms.items())[:5])
print("First 5 terms molecule:", list(ham_molecule.terms.items())[:5])