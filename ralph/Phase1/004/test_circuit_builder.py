#!/usr/bin/env python3
"""Test UCCCircuitBuilder implementation."""

import sys
sys.path.append("../001")
sys.path.append("../002")

from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder

# Process H2 molecule
print("Processing H2 molecule...")
molecule_data = process_molecule("H2", 0.74, "UCC")
print(f"MoleculeData: n_qubits={molecule_data.n_qubits}, fci_energy={molecule_data.fci_energy}")

# Create circuit builder
print("\nCreating UCCCircuitBuilder...")
builder = UCCCircuitBuilder(molecule_data)
print(f"Available excitations: {builder.get_available_excitations()}")
print(f"Number of parameters: {builder.n_params}")

# Test parameter initialization
print("\nTesting parameter initialization...")
params_random = builder.initialize_parameters(builder.n_params, "random")
print(f"Random params: {params_random}")
params_zeros = builder.initialize_parameters(builder.n_params, "zeros")
print(f"Zero params: {params_zeros}")

# Test building circuit with subset of excitations
print("\nBuilding circuit with subset of excitations...")
excitations = builder.get_available_excitations()[:2]  # first two excitations
print(f"Selected excitations: {excitations}")
circuit = builder.build_circuit(excitations, params=params_zeros)
print(f"Circuit type: {type(circuit)}")
print(f"Circuit attributes: {dir(circuit)[:10]}")

# Evaluate energy using builder's evaluate_energy
print("\nEvaluating energy with zero parameters...")
energy_zero = builder.evaluate_energy(circuit, params_zeros)
print(f"Energy (zero params): {energy_zero}")

# Evaluate with random parameters
energy_random = builder.evaluate_energy(circuit, params_random)
print(f"Energy (random params): {energy_random}")

# Test mapping
print("\nExcitation to parameter mapping:")
for exc in builder.get_available_excitations():
    param_indices = builder.get_parameter_indices_for_excitation(exc)
    print(f"  {exc} -> param indices {param_indices}")

print("\nAll tests passed!")