#!/usr/bin/env python3
"""Test environment for observation space and step errors."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.environment import UCCSearchEnv

print("Processing LiH with Jordan-Wigner transformation")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data.n_qubits}")
print(f"Available excitations: {len(data.molecular_info.get('available_excitations', []))}")

# Create environment with default config
env = UCCSearchEnv(data, config={})
print(f"Action space: {env.action_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Number of actions: {env.n_actions}")

# Test 1: Reset and step through a few actions
print("\n=== Test 1: Sequential unique actions ===")
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial energy: {env.current_energy}")

for i in range(min(5, env.n_actions)):
    action = i
    print(f"\nStep {i}: action {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Energy: {env.current_energy}")
    print(f"  Excitations: {env.current_excitations}")
    if terminated:
        print("  Episode terminated early")
        break

# Test 2: Duplicate action handling
print("\n=== Test 2: Duplicate action ===")
env.reset()
if env.n_actions > 0:
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"First action {action}: reward {reward}, terminated {terminated}")
    # Try same action again
    obs2, reward2, terminated2, truncated2, info2 = env.step(action)
    print(f"Duplicate action {action}: reward {reward2}, terminated {terminated2}")
    print(f"Error in info: {info2.get('error')}")
    # Should not be terminated (unless max steps reached)
    if not terminated2:
        print("Duplicate did not terminate episode (good)")
    else:
        print("Duplicate terminated episode (unexpected)")

# Test 3: Observation space consistency
print("\n=== Test 3: Observation space consistency ===")
env.reset()
obs = env._get_observation()
expected_len = env.observation_space.shape[0]
actual_len = len(obs)
if expected_len == actual_len:
    print(f"✓ Observation length matches: {actual_len}")
else:
    print(f"✗ Observation length mismatch: expected {expected_len}, got {actual_len}")
    # Compute components
    max_depth = env.config.get("max_depth", 10)
    n_available = env.n_available
    print(f"  max_depth={max_depth}, n_available={n_available}")
    print(f"  expected length = 1 + {max_depth} + {n_available} + 1 = {1+max_depth+n_available+1}")

# Test 4: Random episode simulation
print("\n=== Test 4: Random episode simulation (max 10 steps) ===")
env.reset()
step = 0
while step < 10:
    # Choose random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    print(f"Step {step}: action {action}, reward {reward:.3f}, energy {env.current_energy:.6f}, excitations {len(env.current_excitations)}")
    if terminated:
        print(f"Episode terminated: {info.get('termination_reason')}")
        break

print("\nAll tests completed.")