#!/usr/bin/env python3
"""Check UCCSD energies for active space."""

import sys
import warnings
warnings.filterwarnings('ignore')
import tencirchem
from tencirchem import UCCSD
from pyscf import gto, scf

# Build molecule LiH bond length 1.6 Å
atoms = [("Li", 0.0, 0.0, 0.0), ("H", 1.6, 0.0, 0.0)]
mol = gto.M(
    atom=atoms,
    basis='sto-3g',
    unit='angstrom',
    symmetry=False,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()

# Create UCCSD with active space (2,3)
ucc = UCCSD(mol, active_space=(2,3), init_method='mp2')
print(f"UCCSD n_elec: {ucc.n_elec}, n_orb: {ucc.n_orb}")
print(f"UCCSD n_qubits: {ucc.n_qubits}")
print(f"UCCSD e_hf: {ucc.e_hf}")
print(f"UCCSD e_fci: {ucc.e_fci}")
print(f"UCCSD e_mp2: {ucc.e_mp2}")
print(f"UCCSD ex_ops: {ucc.ex_ops}")
print(f"UCCSD param_ids: {ucc.param_ids}")

# Compute HF energy of full system
print(f"\nFull system HF energy: {hf.e_tot}")

# Compute active-space FCI using PySCF CASCI
from pyscf import mcscf, fci
cas = mcscf.CASCI(hf, 3, 2)
cas.kernel()
print(f"CASCI energy (active space): {cas.e_tot}")
print(f"Difference with UCCSD e_fci: {cas.e_tot - ucc.e_fci}")

# Compute qubit Hamiltonian from UCCSD
ham = ucc.h_fermion_op
print(f"\nFermion operator n terms: {len(ham.terms)}")
# Transform to qubit operator (Jordan-Wigner)
from openfermion import jordan_wigner
qubit_op = jordan_wigner(ham)
print(f"Qubit operator n terms: {len(qubit_op.terms)}")

# Diagonalize
from openfermion.linalg import get_sparse_operator
H = get_sparse_operator(qubit_op, n_qubits=ucc.n_qubits).toarray()
eigs = np.linalg.eigvalsh(H)
print(f"Exact ground from qubit Hamiltonian: {eigs[0]}")
print(f"Difference with UCCSD e_fci: {eigs[0] - ucc.e_fci}")

# Check UCCSD kernel optimization
ucc.kernel()
print(f"\nUCCSD optimized energy: {ucc.energy()}")
print(f"Difference from e_fci: {ucc.energy() - ucc.e_fci}")

import numpy as np