#!/usr/bin/env python3
"""Debug environment step."""

import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

import numpy as np
from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search.environment import UCCSearchEnv

np.random.seed(42)

print("Processing H2 molecule...")
molecule_data = process_molecule("H2", 0.74, "UCC")

# Create environment
env = UCCSearchEnv(molecule_data)
print(f"Action space size: {env.n_actions}")
print(f"Available excitations: {env.available_excitations}")

# Reset
obs = env.reset()
print(f"Reset done: {env.done}")
print(f"Current excitations: {env.current_excitations}")
print(f"Current energy: {env.current_energy}")
print(f"Best energy: {env.best_energy}")

# Take action 1
print("\n--- Taking action 1 ---")
action = 1
print(f"Action index: {action}")
print(f"Excitation: {env.available_excitations[action]}")

# Manually step through logic
if env.done:
    print("Environment already done")
else:
    if not env.action_space.contains(action):
        print("Invalid action")
    excitation = env.available_excitations[action]
    print(f"Excitation: {excitation}")
    if excitation in env.current_excitations:
        print("Duplicate excitation")
    else:
        print("Not duplicate")
    max_depth = env.config.get("max_depth", 10)
    max_excitations = env.config.get("max_excitations", 20)
    print(f"Current excitations length: {len(env.current_excitations)}")
    print(f"max_depth: {max_depth}, max_excitations: {max_excitations}")
    if len(env.current_excitations) >= max_depth or len(env.current_excitations) >= max_excitations:
        print("Max depth/excitations exceeded")
    else:
        print("Proceeding to add excitation")

# Now call step
obs, reward, done, info = env.step(action)
print(f"\nStep result:")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info keys: {info.keys()}")
if 'termination_reason' in info:
    print(f"Termination reason: {info['termination_reason']}")
print(f"Current energy: {info.get('energy')}")
print(f"Best energy: {info.get('best_energy')}")
print(f"Excitations: {info.get('excitations')}")
print(f"Step count: {info.get('step')}")

# Check if reward came from reward function
print("\nChecking reward function...")
print(f"Reward function best_energy: {env.reward_function.best_energy}")
print(f"Circuit complexity: {len(env.current_excitations)}")
# Compute reward manually
if env.reward_function.best_energy is None:
    print("Best energy is None")
else:
    energy_improvement = env.reward_function.best_energy - env.current_energy
    complexity_penalty = env.reward_function.config.get('complexity_penalty', 0.01) * len(env.current_excitations)
    manual_reward = energy_improvement - complexity_penalty
    print(f"Manual reward: {manual_reward}")

# Check if duplicate excitation detection triggered
print("\nChecking duplicate excitation after step...")
print(f"Current excitations: {env.current_excitations}")
print(f"Excitation in current_excitations? {excitation in env.current_excitations}")