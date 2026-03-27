#!/usr/bin/env python3
"""Test environment after fix."""

import sys
sys.path.insert(0, 'src')
import os
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.environment import UCCSearchEnv
import numpy as np

print("Processing LiH parity")
data = process_molecule("LiH", 1.6, "UCC", active_space=(2,3), transform="parity")
print(f"n_qubits: {data.n_qubits}")
print(f"Available excitations count: {len(data.molecular_info.get('available_excitations', []))}")

# Create environment with default config
env = UCCSearchEnv(data)
print(f"Action space: {env.action_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"n_available: {env.n_available}")
print(f"max_excitations config: {env.config.get('max_excitations')}")

# Run a few random steps
obs, info = env.reset()
print(f"Reset obs shape: {obs.shape}")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: action {action}, reward {reward:.3f}, terminated {terminated}, truncated {truncated}")
    if terminated:
        break

print("Test completed.")