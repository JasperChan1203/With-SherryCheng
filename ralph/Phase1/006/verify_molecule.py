#!/usr/bin/env python3
"""Verify molecule processing for LiH (2,3) with Jordan-Wigner transformation."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.molecule.processor import process_molecule

print("Processing LiH with active_space=(2,3), transform='jordan_wigner'")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data.n_qubits}")
print(f"FCI energy: {data.fci_energy}")
print(f"HF energy: {data.molecular_info.get('hf_energy')}")
print(f"Transform: {data.molecular_info['transform']}")
print(f"Active space: {data.molecular_info['active_space']}")
print(f"Reference state shape: {data.reference_state.shape}")

# Compute expected qubit count: with Jordan-Wigner, each spin orbital maps to a qubit.
# Active space (2 electrons, 3 orbitals) => 3 spatial orbitals => 6 spin orbitals.
# However, Jordan-Wigner mapping uses 2*n_orb_active qubits? Actually yes, each spin orbital maps to a qubit.
# So n_qubits should be 2*3 = 6.
expected_qubits = 2 * data.molecular_info['active_space'][1]
if data.n_qubits == expected_qubits:
    print(f"✓ Qubit count matches expectation: {expected_qubits}")
else:
    print(f"✗ Qubit count mismatch: expected {expected_qubits}, got {data.n_qubits}")

# Check available excitations via circuit builder
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
builder = UCCCircuitBuilder(data)
print(f"\nCircuit builder n_params: {builder.n_params}")
print(f"Available excitations count: {len(builder.get_available_excitations())}")
print("First few excitations:", builder.get_available_excitations()[:5])

# Verify environment creation
from rlqas.phase1.search.environment import UCCSearchEnv
env = UCCSearchEnv(data, config={})
print(f"\nEnvironment action space: {env.action_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Number of actions: {env.n_actions}")
print(f"Available excitations count: {len(env.available_excitations)}")

# Test reset and step
obs, info = env.reset()
print(f"Reset observation shape: {obs.shape}")
print(f"Initial energy: {env.current_energy}")
print(f"Hartree-Fock energy: {env._get_hf_energy()}")

if env.n_actions > 0:
    # Take first action
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nStep result:")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Energy after step: {env.current_energy}")
    print(f"  Excitations: {env.current_excitations}")
else:
    print("No available excitations - check UCCSD generation.")

print("\nVerification complete.")