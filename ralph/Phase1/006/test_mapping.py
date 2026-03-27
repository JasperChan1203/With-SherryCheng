#!/usr/bin/env python3
"""Test mapping consistency between molecule processor and circuit builder."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder

print("Testing LiH with active_space=(2,3), transform='jordan_wigner'")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"MoleculeData n_qubits: {data.n_qubits}")
print(f"Transform: {data.molecular_info['transform']}")
print(f"Has ucc_object: {hasattr(data, 'ucc_object') and data.ucc_object is not None}")

builder = UCCCircuitBuilder(data)
print(f"Circuit builder n_params: {builder.n_params}")
print(f"UCC object type: {type(builder.ucc)}")
print(f"UCC n_qubits (if attribute): {getattr(builder.ucc, 'n_qubits', 'N/A')}")
print(f"UCC n_elec (if attribute): {getattr(builder.ucc, 'n_elec', 'N/A')}")

# Build empty circuit (no excitations)
circuit = builder.build_circuit([])
print(f"Circuit built: {circuit}")
# Evaluate energy with zero parameters (should be HF energy?)
params = builder.initialize_parameters(builder.n_params, strategy='zeros')
energy = builder.evaluate_energy(circuit, params)
print(f"Energy with zero parameters: {energy}")
print(f"HF energy from molecule data: {data.molecular_info['hf_energy']}")
print(f"Difference: {energy - data.molecular_info['hf_energy']}")

# Also compute expectation using simulator with Hamiltonian
from rlqas.phase1.simulator.factory import SimulatorFactory
sim = SimulatorFactory.create_simulator(data.n_qubits)
hamiltonian = data.hamiltonian
# Need to convert circuit to tensorcircuit? Simulator expects circuit.
# For now skip.

print("\nTesting parity transform (warning expected)")
data2 = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='parity')
print(f"Parity n_qubits: {data2.n_qubits}")
builder2 = UCCCircuitBuilder(data2)
print(f"Builder2 ucc type: {type(builder2.ucc)}")
print(f"Builder2 n_params: {builder2.n_params}")
# Expect inconsistency because circuit uses JW mapping but Hamiltonian uses parity.
# Energy evaluation may be wrong.
circuit2 = builder2.build_circuit([])
params2 = builder2.initialize_parameters(builder2.n_params, strategy='zeros')
energy2 = builder2.evaluate_energy(circuit2, params2)
print(f"Parity Hamiltonian energy (should be HF): {energy2}")
print(f"HF energy: {data2.molecular_info['hf_energy']}")
print(f"Difference: {energy2 - data2.molecular_info['hf_energy']}")