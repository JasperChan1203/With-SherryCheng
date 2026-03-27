#!/usr/bin/env python3
"""Debug energy discrepancy in environment."""

import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

import numpy as np
from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search.environment import UCCSearchEnv
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder

np.random.seed(42)

print("Processing H2 molecule...")
molecule_data = process_molecule("H2", 0.74, "UCC")
print(f"MoleculeData: n_qubits={molecule_data.n_qubits}")
print(f"FCI energy: {molecule_data.fci_energy}")
print(f"HF energy (from molecular_info): {molecule_data.molecular_info.get('hf_energy', 'not found')}")

# Create environment
env = UCCSearchEnv(molecule_data)
print(f"\nEnvironment simulator type: {type(env.simulator)}")

# Create circuit builder
builder = UCCCircuitBuilder(molecule_data)

# Get available excitations
excitations = builder.get_available_excitations()
print(f"\nAvailable excitations: {excitations}")

# Build circuit with first excitation, zero parameters
params_zero = np.zeros(builder.n_params)
circuit = builder.build_circuit([excitations[0]], params=params_zero)
print(f"\nCircuit built with excitation {excitations[0]}")

# Evaluate energy using builder's evaluate_energy
energy_builder = builder.evaluate_energy(circuit, params_zero)
print(f"Builder evaluate_energy: {energy_builder}")

# Evaluate energy using environment's simulator
print("\nEvaluating with environment simulator...")
# Need to get Hamiltonian from molecule_data
hamiltonian = molecule_data.hamiltonian
print(f"Hamiltonian type: {type(hamiltonian)}")
print(f"Hamiltonian (first 5 terms): {list(hamiltonian.terms.items())[:5] if hasattr(hamiltonian, 'terms') else 'unknown'}")

# Simulator compute_energy
try:
    energy_sim = env.simulator.compute_energy(circuit, hamiltonian, initial_state=molecule_data.reference_state)
    print(f"Simulator compute_energy: {energy_sim}")
except Exception as e:
    print(f"Simulator error: {e}")
    import traceback
    traceback.print_exc()

# Compare with builder's evaluate_energy
print(f"\nDifference: {energy_sim - energy_builder}")

# Also check that circuit builder's ucc.energy gives same as evaluate_energy
energy_ucc = builder.ucc.energy(params_zero)
print(f"UCCSD energy (zero params): {energy_ucc}")

# Check if parameters mapping is correct
print(f"\nBuilder n_params: {builder.n_params}")
print(f"Builder param_to_ex_ops: {builder.param_to_ex_ops}")

# Let's also compute HF energy via PySCF
print("\n---")
print("Recomputing HF energy via PySCF:")
from pyscf import gto, scf
mol = gto.M(atom=[('H', 0, 0, 0), ('H', 0.74, 0, 0)], basis='sto-3g')
mf = scf.RHF(mol).run()
print(f"PySCF HF energy: {mf.e_tot}")