#!/usr/bin/env python3
"""Test import of dependencies from Phase 1 Tasks 001, 002, 003."""

import sys
import os

print("Testing imports for RLQAS Phase 1 Task 004 (UCC Search Module)")
print("=" * 60)

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

print("\n1. Testing Task 001 imports...")
try:
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print("   SUCCESS: imported MoleculeData, process_molecule")
except ImportError as e:
    print(f"   FAILED: {e}")

print("\n2. Testing Task 002 imports...")
try:
    from src.modules.quantum_simulator import QuantumSimulator, TencirchemCISimulator, SimulatorFactory
    print("   SUCCESS: imported QuantumSimulator, TencirchemCISimulator, SimulatorFactory")
except ImportError as e:
    print(f"   FAILED: {e}")

print("\n3. Testing Task 003 imports...")
try:
    from src.modules.rl_agents import RLAgent, PPOAgent
    print("   SUCCESS: imported RLAgent, PPOAgent")
except ImportError as e:
    print(f"   FAILED: {e}")

print("\n4. Testing additional required packages...")
required_packages = [
    ("numpy", "np"),
    ("tencirchem", "tencirchem"),
    ("openfermion", "openfermion"),
    ("pyscf", "pyscf"),
    ("gym", "gym"),
    ("stable_baselines3", "stable_baselines3"),
    ("torch", "torch"),
]

for package_name, import_name in required_packages:
    try:
        __import__(import_name)
        print(f"   {package_name}: OK")
    except ImportError as e:
        print(f"   {package_name}: MISSING - {e}")

print("\n" + "=" * 60)
print("Import test complete.")