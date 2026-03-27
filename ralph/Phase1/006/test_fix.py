#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder

print("Testing parity transform mapping fix")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='parity')
print(f"n_qubits: {data.n_qubits}")
print(f"transform: {data.molecular_info['transform']}")
print(f"ucc_sd_object is None: {data.ucc_sd_object is None}")
print(f"ucc_object type: {type(data.ucc_object)}")

builder = UCCCircuitBuilder(data)
print(f"builder.ucc type: {type(builder.ucc)}")
print(f"builder.ucc n_qubits: {getattr(builder.ucc, 'n_qubits', 'N/A')}")
print(f"builder.n_params: {builder.n_params}")
print(f"builder.ucc ex_ops: {builder.ucc.ex_ops[:5] if hasattr(builder.ucc, 'ex_ops') else 'N/A'}")

# Build empty circuit
circuit = builder.build_circuit([])
params = builder.initialize_parameters(builder.n_params, strategy='zeros')
energy = builder.evaluate_energy(circuit, params)
print(f"Energy with zero params: {energy}")
print(f"HF energy: {data.molecular_info['hf_energy']}")
print(f"Difference: {energy - data.molecular_info['hf_energy']}")

# Check qubit count consistency
if hasattr(builder.ucc, 'n_qubits'):
    if builder.ucc.n_qubits == data.n_qubits:
        print("✓ Qubit count consistent")
    else:
        print(f"✗ Qubit count mismatch: UCC {builder.ucc.n_qubits} vs molecule {data.n_qubits}")

print("\nTesting Jordan-Wigner transform (should use UCCSD)")
data2 = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data2.n_qubits}")
print(f"ucc_sd_object type: {type(data2.ucc_sd_object)}")
builder2 = UCCCircuitBuilder(data2)
print(f"builder2.ucc type: {type(builder2.ucc)}")
print(f"builder2.ucc n_qubits: {getattr(builder2.ucc, 'n_qubits', 'N/A')}")