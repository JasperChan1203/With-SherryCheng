#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.environment import UCCSearchEnv

print("Testing environment with Jordan-Wigner transformation")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data.n_qubits}")
print(f"transform: {data.molecular_info['transform']}")

env = UCCSearchEnv(data, config={})
print(f"action space: {env.action_space}")
print(f"observation space shape: {env.observation_space.shape}")

# Reset environment
obs, info = env.reset()
print(f"reset obs shape: {obs.shape}")
print(f"initial energy: {env.current_energy}")

# Take a random action (if actions available)
if env.n_actions > 0:
    action = 0  # first excitation
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"energy after step: {env.current_energy}")
    print(f"excitations: {env.current_excitations}")
else:
    print("No available excitations")

print("Test passed.")