#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import numpy as np
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.environment import UCCSearchEnv

print("1. Processing LiH with Jordan-Wigner...")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"   n_qubits={data.n_qubits}, FCI={data.fci_energy}, HF={data.molecular_info['hf_energy']}")

print("\n2. Creating environment...")
env = UCCSearchEnv(data, config={
    "environment": {"max_depth": 12, "max_excitations": 15, "param_init_strategy": "random"}
})
print(f"   available_excitations count = {len(env.available_excitations)}")
print(f"   action space size = {env.n_actions}")
print(f"   observation space shape = {env.observation_space.shape}")
print(f"   initial current_energy = {env.current_energy}")
print(f"   initial best_energy = {env.best_energy}")

print("\n3. Resetting environment...")
obs, info = env.reset()
print(f"   observation shape = {obs.shape}")
print(f"   current_energy = {env.current_energy}")
print(f"   best_energy = {env.best_energy}")

print("\n4. Simulating one random action...")
action = np.random.randint(env.n_actions)
print(f"   random action = {action}")
print(f"   corresponding excitation = {env.available_excitations[action]}")

obs, reward, terminated, truncated, info = env.step(action)
print(f"   reward = {reward}")
print(f"   terminated = {terminated}, truncated = {truncated}")
print(f"   info keys = {info.keys()}")
print(f"   info energy = {info.get('energy')}")
print(f"   info best_energy = {info.get('best_energy')}")
print(f"   info excitations = {info.get('excitations')}")
print(f"   info params = {info.get('params')}")
print(f"   current_energy = {env.current_energy}")
print(f"   best_energy = {env.best_energy}")
print(f"   global_best_energy = {env.global_best_energy}")

print("\n5. Checking circuit builder...")
print(f"   circuit_builder.available_excitations count = {len(env.circuit_builder.available_excitations)}")
print(f"   circuit_builder.n_params = {env.circuit_builder.n_params}")
print(f"   current_excitations = {env.current_excitations}")
print(f"   active_parameters = {env.active_parameters}")

print("\n6. Simulator check...")
print(f"   simulator type = {type(env.simulator)}")
# Compute energy directly using simulator with circuit
circuit = env.circuit_builder.build_circuit(env.current_excitations, env.current_params)
print(f"   circuit type = {type(circuit)}")
if hasattr(circuit, 'set_params'):
    print(f"   circuit has set_params")
    if hasattr(circuit, 'params'):
        print(f"   circuit.params = {circuit.params}")
energy_direct = env.simulator.compute_energy(circuit, data.hamiltonian, data.reference_state)
print(f"   direct compute_energy = {energy_direct}")

print("\n7. Compare with UCCSD energy...")
if data.ucc_sd_object:
    params = env.current_params
    print(f"   params shape = {params.shape}")
    # UCCSD energy with all parameters (including zeros)
    ucc_energy = data.ucc_sd_object.energy(params)
    print(f"   UCCSD.energy(params) = {ucc_energy}")

print("\n8. Hamiltonian expectation of reference state...")
# compute expectation of Hamiltonian with reference state using simulator
# Use circuit with no gates (identity)
zero_circuit = env.circuit_builder.build_circuit([], np.zeros(env.circuit_builder.n_params))
energy_zero = env.simulator.compute_energy(zero_circuit, data.hamiltonian, data.reference_state)
print(f"   energy zero circuit = {energy_zero}")