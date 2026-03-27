#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import numpy as np
import tensorcircuit as tc
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder

data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
builder = UCCCircuitBuilder(data)
circuit = builder.ucc.get_circuit()
print("Circuit class:", circuit.__class__)
print("Circuit attributes:", [a for a in dir(circuit) if not a.startswith('_')])
# Check for set_state, set_inputs, set_initial_state, etc.
for attr in ['set_state', 'set_inputs', 'set_initial_state', 'initial_state', 'init_state', 'input_state']:
    if hasattr(circuit, attr):
        print(f"circuit.{attr} = {getattr(circuit, attr)}")
# Check if circuit has gates attribute
if hasattr(circuit, 'gates'):
    print(f"Number of gates: {len(circuit.gates)}")
    for i, gate in enumerate(circuit.gates[:10]):
        print(f"  gate {i}: {gate}")
# Check if circuit has _ops
if hasattr(circuit, '_ops'):
    print(f"_ops length: {len(circuit._ops)}")
# Create a new circuit with initial_state
n = circuit.n_qubits
psi = data.reference_state
print(f"Reference state shape {psi.shape}")
# Try to create circuit with initial_state
try:
    c2 = tc.Circuit(n, initial_state=psi)
    print("Created circuit with initial_state")
except Exception as e:
    print(f"Error creating circuit with initial_state: {e}")
# Try set_state
if hasattr(circuit, 'set_state'):
    circuit.set_state(psi)
    print("set_state called")
    print(circuit.state()[:4])
# Try to get the gate list and append to new circuit
if hasattr(circuit, 'gates'):
    c3 = tc.Circuit(n, initial_state=psi)
    for gate in circuit.gates:
        # gate is a tuple? need to extract name and qubits
        pass
# Compute expectation with initial_state using tensorcircuit expectation
# Use expectation_ps with circuit that has initial_state
c_test = tc.Circuit(n, initial_state=psi)
# Apply no gates, expectation of Hamiltonian
# We need to compute expectation term by term
from openfermion import QubitOperator
ham = data.hamiltonian
energy = 0.0
for term, coeff in ham.terms.items():
    x_list, y_list, z_list = [], [], []
    for idx, pauli in term:
        if pauli == 'X':
            x_list.append(idx)
        elif pauli == 'Y':
            y_list.append(idx)
        elif pauli == 'Z':
            z_list.append(idx)
    exp_val = c_test.expectation_ps(x=x_list, y=y_list, z=z_list)
    energy += coeff * exp_val
print(f"Expectation with reference state via tensorcircuit: {energy.real}")