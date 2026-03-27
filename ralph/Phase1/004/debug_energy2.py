#!/usr/bin/env python3
"""Detailed energy comparison."""

import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

import numpy as np
import tensorcircuit as tc
from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder
from src.modules.quantum_simulator import SimulatorFactory

np.random.seed(42)

print("Processing H2 molecule...")
molecule_data = process_molecule("H2", 0.74, "UCC")
print(f"MoleculeData: n_qubits={molecule_data.n_qubits}")
print(f"FCI energy: {molecule_data.fci_energy}")
print(f"HF energy (from molecular_info): {molecule_data.molecular_info.get('hf_energy', 'not found')}")

# Create circuit builder
builder = UCCCircuitBuilder(molecule_data)
print(f"\nBuilder n_params: {builder.n_params}")

# Get Hamiltonian
hamiltonian = molecule_data.hamiltonian
print(f"Hamiltonian type: {type(hamiltonian)}")
print(f"Number of terms: {len(hamiltonian.terms)}")

# Compute HF energy using tensorcircuit: expectation of Hamiltonian on |00> state
print("\n=== Computing HF energy via tensorcircuit expectation on |00> ===")
n_qubits = molecule_data.n_qubits
circuit_hf = tc.Circuit(n_qubits)  # empty circuit, state = |00>
energy_hf = 0.0
for term, coeff in hamiltonian.terms.items():
    x_list = []
    y_list = []
    z_list = []
    for idx, pauli in term:
        if pauli == 'X':
            x_list.append(idx)
        elif pauli == 'Y':
            y_list.append(idx)
        elif pauli == 'Z':
            z_list.append(idx)
        elif pauli == 'I':
            pass
    exp_val = circuit_hf.expectation_ps(x=x_list, y=y_list, z=z_list)
    energy_hf += coeff * exp_val
print(f"HF energy via tensorcircuit expectation: {energy_hf.real}")

# Compute HF energy using builder's ucc.energy with zero parameters
energy_ucc_zero = builder.ucc.energy(np.zeros(builder.n_params))
print(f"UCCSD energy zero params: {energy_ucc_zero}")

# Build circuit with zero parameters, first excitation (should not affect energy)
circuit = builder.build_circuit([builder.available_excitations[0]], params=np.zeros(builder.n_params))
print(f"Circuit type: {type(circuit)}")
print(f"Circuit has set_params? {hasattr(circuit, 'set_params')}")
print(f"Circuit params attribute? {hasattr(circuit, 'params')}")

# Evaluate energy using builder.evaluate_energy
energy_builder = builder.evaluate_energy(circuit, np.zeros(builder.n_params))
print(f"Builder evaluate_energy: {energy_builder}")

# Evaluate energy using simulator
simulator = SimulatorFactory.create_simulator(n_qubits)
print(f"\n=== Simulator compute_energy ===")
energy_sim = simulator.compute_energy(circuit, hamiltonian, initial_state=molecule_data.reference_state)
print(f"Simulator compute_energy: {energy_sim}")

# Let's also compute expectation using circuit.expectation_ps directly (like simulator does)
print("\n=== Direct expectation using circuit.expectation_ps ===")
energy_direct = 0.0
for term, coeff in hamiltonian.terms.items():
    x_list = []
    y_list = []
    z_list = []
    for idx, pauli in term:
        if pauli == 'X':
            x_list.append(idx)
        elif pauli == 'Y':
            y_list.append(idx)
        elif pauli == 'Z':
            z_list.append(idx)
        elif pauli == 'I':
            pass
    exp_val = circuit.expectation_ps(x=x_list, y=y_list, z=z_list)
    energy_direct += coeff * exp_val
print(f"Direct expectation via circuit.expectation_ps: {energy_direct.real}")

# Compare with circuit's expectation_ps with initial_state? Actually circuit already includes reference.
# Let's also compute expectation using tensorcircuit's expectation with initial_state = |00>
print("\n=== Expectation with initial_state = |00> (should be same) ===")
# Create a new circuit that does nothing, but compute expectation with initial_state = |00>
# Actually expectation_ps doesn't accept initial_state. We need to compute state vector.
# Let's compute state vector of circuit: circuit.state()
state = circuit.state()
print(f"State shape: {state.shape}")
# Compute expectation manually using numpy
energy_state = 0.0
for term, coeff in hamiltonian.terms.items():
    op = tc.quantum.PauliString(term)
    exp_val = tc.backend.conj(state) @ (op @ state)
    energy_state += coeff * exp_val
print(f"Energy via state vector: {energy_state.real}")

# Now compute expectation of Hamiltonian using tensorcircuit's expectation
# Use tc.expectation
print("\n=== Using tc.expectation ===")
energy_tc = tc.expectation(circuit, hamiltonian)
print(f"tc.expectation result: {energy_tc}")

# Let's also compute expectation of Hamiltonian using tencirchem's internal function
print("\n=== Using tencirchem's expectation ===")
import tencirchem
# tencirchem.expectation?
# Not sure.

print("\n=== Summary ===")
print(f"HF energy (PySCF): {molecule_data.molecular_info.get('hf_energy')}")
print(f"HF energy (tensorcircuit): {energy_hf.real}")
print(f"UCCSD zero params: {energy_ucc_zero}")
print(f"Builder evaluate_energy: {energy_builder}")
print(f"Simulator compute_energy: {energy_sim}")
print(f"Direct expectation: {energy_direct.real}")
print(f"State vector expectation: {energy_state.real}")
print(f"tc.expectation: {energy_tc}")