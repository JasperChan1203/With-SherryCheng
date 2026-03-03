#!/usr/bin/env python3
"""Examine LiH molecule processing details."""

import sys
sys.path.insert(0, 'src')
import os
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'

from rlqas.phase1.molecule.processor import process_molecule
import numpy as np

print("Processing LiH with active_space=(2,3), parity transform")
data = process_molecule("LiH", 1.6, "UCC", active_space=(2,3), transform="parity")
print(f"n_qubits: {data.n_qubits}")
print(f"FCI energy: {data.fci_energy}")
print(f"HF energy from molecular_info: {data.molecular_info.get('hf_energy')}")
print(f"Reference state shape: {data.reference_state.shape}")
# Find index where reference state is 1
idx = np.argmax(np.abs(data.reference_state))
print(f"Reference state index: {idx} (binary {bin(idx)})")
# Compute expectation value using Hamiltonian diagonal terms (brute force)
ham = data.hamiltonian
n_qubits = data.n_qubits
# Compute expectation for this computational basis state
energy = 0.0
for term, coeff in ham.terms.items():
    diag = True
    sign = 1.0
    for q, pauli in term:
        if pauli == 'Z':
            bit = (idx >> q) & 1
            sign *= (1 - 2*bit)  # +1 for 0, -1 for 1
        elif pauli == 'I':
            continue
        else:
            diag = False
            break
    if diag:
        energy += coeff * sign
print(f"Reference state energy (diagonal): {energy}")
print(f"Difference with HF: {energy - data.molecular_info.get('hf_energy')}")

# Also compute expectation using all terms (including off-diagonal) via matrix representation? Too heavy.
# Let's compute using simulator
from rlqas.phase1.simulator.factory import SimulatorFactory
sim = SimulatorFactory.create_simulator(data.n_qubits)
# Need a circuit that prepares reference state (just identity)
# For now, trust diagonal energy.

# Check available excitations from circuit builder
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
builder = UCCCircuitBuilder(data)
print(f"\nCircuit builder n_params: {builder.n_params}")
print(f"Available excitations count: {len(builder.get_available_excitations())}")
print("First few excitations:", builder.get_available_excitations()[:5])

# Compute HF occupation pattern for parity transformation
# Parity transformation: qubits represent occupation differences.
# For RHF with 2 electrons in 3 orbitals? Need to understand mapping.
# Let's compute using tencirchem parity mapping.
import tencirchem
from tencirchem import parity
# We can get the UCC object from builder
ucc = builder.ucc
print(f"\nUCC n_elec: {ucc.n_elec}, n_qubits: {ucc.n_qubits}")
print(f"UCC hf_energy: {ucc.hf_energy}")
print(f"UCC e_hf: {ucc.e_hf}")
print(f"UCC e_fci: {ucc.e_fci}")
print(f"UCC e_mp2: {ucc.e_mp2}")
# The reference state used by tencirchem might be different.
# Let's compute the HF bitstring for parity transformation.
# According to tencirchem documentation, parity mapping uses occupation differences.
# Hard to deduce. We'll assume the brute-force reference state is correct.
print("\nDone.")