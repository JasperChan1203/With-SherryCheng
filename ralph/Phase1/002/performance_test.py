#!/usr/bin/env python3
"""
Quick performance test for 8-qubit energy evaluation.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import time
from openfermion import QubitOperator
import tensorcircuit as tc
from src.modules.quantum_simulator import TencirchemCISimulator

print("Performance test: 8-qubit circuit energy evaluation")

# Create random 8-qubit Hamiltonian (simple diagonal for speed)
hamiltonian = QubitOperator()
for i in range(8):
    hamiltonian += QubitOperator(f"Z{i}", np.random.randn())
# Add a few off-diagonal terms
hamiltonian += QubitOperator("X0 X1", 0.1)
hamiltonian += QubitOperator("Y2 Y3", 0.1)

# Create random parameterized circuit with 8 qubits, depth 5
n_qubits = 8
depth = 5
c = tc.Circuit(n_qubits)
for d in range(depth):
    for q in range(n_qubits):
        c.rx(q, theta=np.random.randn() * 0.5)
    for q in range(0, n_qubits-1, 2):
        c.cnot(q, q+1)

# Create simulator with default configuration
simulator = TencirchemCISimulator()

# Warm-up (first call may have overhead)
energy1 = simulator.compute_energy(c, hamiltonian)
print(f"First energy: {energy1}")

# Time measurement
times = []
for _ in range(10):
    start = time.perf_counter()
    energy = simulator.compute_energy(c, hamiltonian)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"  Energy: {energy:.6f}, time: {elapsed*1000:.2f} ms")

avg_time = np.mean(times) * 1000  # ms
std_time = np.std(times) * 1000
print(f"\nAverage time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"Target: <500 ms")
if avg_time < 500:
    print("✓ Performance target MET")
else:
    print("✗ Performance target NOT MET")

# Also test memory estimation
mem_est = simulator.estimate_memory(8)
print(f"\nMemory estimate for 8 qubits: {mem_est:.2f} GB")