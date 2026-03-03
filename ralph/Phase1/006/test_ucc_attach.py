#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder

data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
builder = UCCCircuitBuilder(data)
circuit = builder.build_circuit([])
print("circuit has ucc?", hasattr(circuit, 'ucc'))
print("circuit has params?", hasattr(circuit, 'params'))
if hasattr(circuit, 'ucc'):
    print("ucc type:", type(circuit.ucc))
    print("params shape:", circuit.params.shape)
    energy = circuit.ucc.energy(circuit.params)
    print("energy via ucc:", energy)
    print("HF energy:", data.molecular_info['hf_energy'])
    # Compare with simulator
from rlqas.phase1.simulator.factory import SimulatorFactory
sim = SimulatorFactory.create_simulator(data.n_qubits)
energy_sim = sim.compute_energy(circuit, data.hamiltonian, data.reference_state)
print("energy via simulator:", energy_sim)