#!/usr/bin/env python3
"""Debug Hamiltonian and UCCSD energy discrepancy."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule
import tencirchem
from openfermion.linalg import get_sparse_operator

print("Processing LiH with active_space=(2,3), transform='jordan_wigner'")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data.n_qubits}")
print(f"FCI energy (PySCF): {data.fci_energy}")

# Get qubit Hamiltonian
ham = data.hamiltonian
print(f"Number of terms: {len(ham.terms)}")

# Compute matrix representation
H_matrix = get_sparse_operator(ham, n_qubits=data.n_qubits).toarray()
print(f"Hamiltonian shape: {H_matrix.shape}")

# Compute eigenvalues (full diagonalization for small system)
eigs = np.linalg.eigvalsh(H_matrix)
ground = eigs[0]
print(f"Exact ground state energy (diagonalization): {ground}")
print(f"Difference with PySCF FCI: {ground - data.fci_energy}")

# Compare with HF energy
hf = data.molecular_info.get('hf_energy')
print(f"HF energy: {hf}")
print(f"HF - ground: {hf - ground}")

# Get UCCSD object
ucc = data.ucc_sd_object
print(f"\nUCCSD n_params: {ucc.n_params}")
print(f"UCCSD ex_ops: {ucc.ex_ops}")
print(f"UCCSD param_ids: {ucc.param_ids}")

# Get optimized parameters from kernel
ucc.kernel()
opt_params = ucc.params
opt_energy = ucc.energy()
print(f"UCCSD optimized energy: {opt_energy}")
print(f"Difference from exact ground: {opt_energy - ground}")
print(f"Error mHa: {(opt_energy - ground)*1000}")

# Compute UCCSD state vector
circuit = ucc.get_circuit()
# Check if circuit can produce state vector
import tensorcircuit as tc
if hasattr(circuit, 'state_vector'):
    state = circuit.state_vector()
else:
    # Evaluate state via simulation
    # We'll use tensorcircuit directly
    c = tc.Circuit(data.n_qubits)
    # Apply UCCSD unitary via circuit? Need to get unitary matrix from tencirchem
    # For simplicity, compute expectation via energy function
    pass

# Compute expectation via simulator
from rlqas.phase1.simulator.factory import SimulatorFactory
sim = SimulatorFactory.create_simulator(data.n_qubits)
# Build circuit with optimized parameters
circuit.set_params(opt_params)
# Compute expectation
energy_sim = sim.compute_energy(circuit, ham, initial_state=data.reference_state)
print(f"Simulator energy with optimized params: {energy_sim}")
print(f"Difference from ucc.energy(): {energy_sim - opt_energy}")

# Check if UCCSD state spans full Hilbert space
# Compute overlap of UCCSD state with exact ground state
# Get exact ground eigenvector
_, evecs = np.linalg.eigh(H_matrix)
exact_gs = evecs[:,0]
# Need UCCSD state vector. Let's compute using tencirchem's get_circuit and tensorcircuit
# tencirchem's circuit is a tensorcircuit.Circuit object
if hasattr(circuit, 'get_state_vector'):
    ucc_state = circuit.get_state_vector()
elif hasattr(circuit, 'state_vector'):
    ucc_state = circuit.state_vector()
else:
    # Use tensorcircuit to simulate circuit with initial HF state
    # Initial HF state is reference_state (one-hot)
    init_state = data.reference_state
    # circuit is already parameterized, we can compute state via tc
    import tensorcircuit as tc
    c = tc.Circuit.from_qir(circuit.to_qir())  # maybe not
    print("Cannot compute state vector easily")
    sys.exit(0)

print(f"UCCSD state vector shape: {ucc_state.shape}")
overlap = np.abs(np.vdot(ucc_state, exact_gs))**2
print(f"Overlap with exact ground: {overlap}")
print(f"Fidelity: {overlap}")

# Compute energy expectation via state vector
energy_state = np.vdot(ucc_state, H_matrix @ ucc_state).real
print(f"Energy via state vector: {energy_state}")

# Check if there are missing excitations in UCCSD
# List all possible single and double excitations for 2 electrons in 6 spin orbitals
# Generate all possible (i,j) and (i,j,k,l) where i,j,k,l are spin orbital indices
# and excitation preserves spin? Not needed.
# Compare with ucc.ex_ops
print("\nAll possible single excitations (spin orbitals 0-5):")
singles = []
for i in range(6):
    for j in range(6):
        if i != j:
            singles.append((i,j))
print(f"Number of possible singles: {len(singles)}")
print("UCCSD singles:", [ex for ex in ucc.ex_ops if len(ex)==2])

print("\nAll possible double excitations:")
doubles = []
for i in range(6):
    for j in range(6):
        for k in range(6):
            for l in range(6):
                if i != j and k != l and len(set([i,j,k,l])) == 4:
                    doubles.append((i,j,k,l))
print(f"Number of possible doubles: {len(doubles)}")
print("UCCSD doubles:", [ex for ex in ucc.ex_ops if len(ex)==4])