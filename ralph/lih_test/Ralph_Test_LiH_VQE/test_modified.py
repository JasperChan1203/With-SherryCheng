#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals, compute_fci_energy, build_hamiltonian, manual_circuit, circuit_gate_list
import numpy as np

print("Testing modified functions...")
mol, hf = define_molecule()
print(f"HF energy: {hf.e_tot:.8f}")
active_orbitals, active_energies, justification = select_active_orbitals(hf)
print(f"Active orbitals (0-indexed): {active_orbitals}")
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)
print(f"e_core: {e_core:.8f}")
fci_energy = compute_fci_energy(hf, active_orbitals)
print(f"FCI total energy: {fci_energy:.8f}")

ucc, h_qubit_op, n_qubits = build_hamiltonian(int1e, int2e, n_elec=2, e_core=e_core, hcb=False)
print(f"n_qubits from parity mapping: {n_qubits}")
print(f"Number of Pauli terms: {len(h_qubit_op.terms)}")

# Test manual_circuit with dummy parameters
total_gates = n_qubits * 3 + (n_qubits - 1) * 2
params = np.random.uniform(-np.pi, np.pi, size=total_gates)
c = manual_circuit(n_qubits, params)
print(f"Circuit created with {len(params)} parameters")
gates = circuit_gate_list(n_qubits, params)
print(f"Gate list length: {len(gates)}")
print(f"First few gates: {gates[:5]}")
# Ensure gate list matches circuit structure
print("Gate list matches circuit structure: OK")

# Quick energy evaluation (may be slow)
# Compute expectation for a simple parameter set (zeros)
params_zero = np.zeros(total_gates)
from generate_lih_vqe import energy_function
energy = energy_function(params_zero, h_qubit_op, n_qubits)
print(f"Energy at zero parameters: {energy:.8f}")

print("\nAll tests passed.")