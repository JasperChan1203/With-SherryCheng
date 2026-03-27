#!/usr/bin/env python3
"""Test FCI fix in molecule processor."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.molecule.processor import process_molecule
import tencirchem
from openfermion.linalg import get_sparse_operator
import numpy as np

print("Processing LiH with active_space=(2,3), transform='jordan_wigner'")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data.n_qubits}")
print(f"FCI energy from processor: {data.fci_energy}")
print(f"HF energy: {data.molecular_info.get('hf_energy')}")

# Get UCCSD object
ucc = data.ucc_sd_object
print(f"\nUCCSD e_fci: {ucc.e_fci}")
print(f"Difference: {data.fci_energy - ucc.e_fci}")

# Diagonalize qubit Hamiltonian
ham = data.hamiltonian
H = get_sparse_operator(ham, n_qubits=data.n_qubits).toarray()
eigs = np.linalg.eigvalsh(H)
ground = eigs[0]
print(f"\nExact ground from qubit Hamiltonian: {ground}")
print(f"Difference with processor FCI: {ground - data.fci_energy}")
print(f"Difference with UCCSD e_fci: {ground - ucc.e_fci}")

# Compute error in mHa
error_mha = (data.fci_energy - ground) * 1000
print(f"Error (mHa): {error_mha}")

# Check if chemical accuracy target is now achievable
# HF - FCI difference
hf = data.molecular_info.get('hf_energy')
print(f"\nHF - FCI: {hf - data.fci_energy} Hartree")
print(f"HF - FCI (mHa): {(hf - data.fci_energy) * 1000} mHa")
print(f"Chemical accuracy target 1.6 mHa is {'achievable' if (hf - data.fci_energy)*1000 > 1.6 else 'not achievable'} with perfect optimization")

# Test with parity transformation (4 qubits)
print("\n--- Parity transformation ---")
data_parity = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='parity')
print(f"n_qubits: {data_parity.n_qubits}")
print(f"FCI energy: {data_parity.fci_energy}")
print(f"HF energy: {data_parity.molecular_info.get('hf_energy')}")
print(f"HF - FCI (mHa): {(data_parity.molecular_info.get('hf_energy') - data_parity.fci_energy) * 1000} mHa")