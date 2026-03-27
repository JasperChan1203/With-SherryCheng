#!/usr/bin/env python3
"""Test UCCSearchEnv implementation."""

import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

import numpy as np
from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search.environment import UCCSearchEnv

# Set random seed for reproducibility
np.random.seed(42)

# Process H2 molecule
print("Processing H2 molecule...")
molecule_data = process_molecule("H2", 0.74, "UCC")
print(f"MoleculeData: n_qubits={molecule_data.n_qubits}, fci_energy={molecule_data.fci_energy}")

# Create environment
print("\nCreating UCCSearchEnv...")
env = UCCSearchEnv(molecule_data)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Number of actions: {env.n_actions}")
print(f"Number of parameters: {env.n_params}")

# Reset environment
print("\nResetting environment...")
obs = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial observation first 5 values: {obs[:5]}")

# Take a few random actions
print("\nTaking random actions...")
for step in range(5):
    action = env.action_space.sample()
    print(f"\nStep {step}: action={action}")
    obs, reward, done, info = env.step(action)
    print(f"  Reward: {reward:.6f}")
    print(f"  Done: {done}")
    print(f"  Current energy: {info['energy']:.6f}")
    print(f"  Best energy: {info['best_energy']:.6f}")
    print(f"  Excitations: {info['excitations']}")
    if done:
        print("  Episode terminated.")
        break

# Render final state
print("\nFinal state:")
env.render()

print("\nTest completed successfully!")