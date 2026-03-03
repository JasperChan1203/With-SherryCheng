#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import numpy as np
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder

data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
builder = UCCCircuitBuilder(data)
print(f"builder.ucc type = {type(builder.ucc)}")
print(f"builder.ucc.__class__.__name__ = {builder.ucc.__class__.__name__}")
circuit = builder.ucc.get_circuit()
print(f"circuit type = {type(circuit)}")
print(f"circuit methods = [m for m in dir(circuit) if not m.startswith('_')]")
for attr in ['n_qubits', 'inputs', 'outputs', 'gates', 'init_state', 'initial_state']:
    if hasattr(circuit, attr):
        print(f"circuit.{attr} = {getattr(circuit, attr)}")
if hasattr(circuit, 'state'):
    state = circuit.state()
    print(f"circuit.state() shape = {state.shape}")
    print(f"first few amplitudes = {state[:4]}")
    # compute probability distribution
    prob = np.abs(state)**2
    print(f"max prob index = {np.argmax(prob)}")
    print(f"max prob value = {prob.max()}")
# Check if circuit has set_params
if hasattr(circuit, 'set_params'):
    print(f"circuit.set_params exists")
    # check default parameters
    if hasattr(circuit, 'params'):
        print(f"circuit.params = {circuit.params}")
# List gates
if hasattr(circuit, 'gates'):
    for i, gate in enumerate(circuit.gates):
        print(f"gate {i}: {gate}")
# Check if circuit has initial_state attribute
if hasattr(circuit, 'init_state'):
    print(f"circuit.init_state = {circuit.init_state}")
if hasattr(circuit, 'initial_state'):
    print(f"circuit.initial_state = {circuit.initial_state}")
# Compute expectation of Hamiltonian with this circuit (zero params)
params = np.zeros(builder.n_params)
print(f"params shape = {params.shape}")
if hasattr(circuit, 'set_params'):
    circuit.set_params(params)
energy = builder.ucc.energy(params)
print(f"UCCSD.energy(params) = {energy}")
# Also compute energy via circuit expectation using simulator
from rlqas.phase1.simulator.factory import SimulatorFactory
sim = SimulatorFactory.create_simulator(data.n_qubits)
print(f"sim type = {type(sim)}")
energy_sim = sim.compute_energy(circuit, data.hamiltonian, data.reference_state)
print(f"sim.compute_energy = {energy_sim}")
# Compute expectation using openfermion with reference state
from openfermion import expectation
ham = data.hamiltonian
psi = data.reference_state
exp = expectation(ham, psi)
print(f"openfermion expectation with reference state = {exp}")