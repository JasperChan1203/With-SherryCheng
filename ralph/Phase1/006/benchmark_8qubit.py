#!/usr/bin/env python3
"""
Benchmark script for 8-qubit simulation performance target.

Validates that 8-qubit circuit energy evaluation completes in <500ms.
"""

import sys
import time
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')

from openfermion import QubitOperator
import tensorcircuit as tc
from rlqas.phase1.simulator.factory import SimulatorFactory


def create_random_hamiltonian(n_qubits: int, n_terms: int = 10) -> QubitOperator:
    """Create a random Hamiltonian for benchmarking.

    Generates random Pauli terms with random coefficients.
    """
    import random
    hamiltonian = QubitOperator()
    paulis = ['X', 'Y', 'Z']
    for _ in range(n_terms):
        # Random number of qubits in term (1 to 3)
        term_len = random.randint(1, min(3, n_qubits))
        indices = random.sample(range(n_qubits), term_len)
        term = tuple((idx, random.choice(paulis)) for idx in indices)
        coeff = random.uniform(-1.0, 1.0)
        hamiltonian += QubitOperator(term, coeff)
    return hamiltonian


def create_random_circuit(n_qubits: int, depth: int = 3) -> tc.Circuit:
    """Create a random parameterized circuit using tensorcircuit."""
    circuit = tc.Circuit(n_qubits)
    # Add random rotation gates
    for d in range(depth):
        for q in range(n_qubits):
            angle = np.random.random() * 2 * np.pi
            circuit.rx(q, theta=angle)
        # Add entangling layer
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)
    return circuit


def benchmark_8qubit(n_trials: int = 5, warmup: int = 2) -> dict:
    """Run benchmark for 8-qubit simulation.

    Args:
        n_trials: Number of measurement trials.
        warmup: Number of warmup runs before timing.

    Returns:
        Dictionary with benchmark results.
    """
    n_qubits = 8
    print(f"Benchmarking {n_qubits}-qubit simulation performance...")
    print(f"Target: <500 ms per energy evaluation")

    # Create simulator
    simulator = SimulatorFactory.create_simulator(n_qubits)
    print(f"Simulator created: {simulator.__class__.__name__}")
    print(f"Max qubits supported: {simulator.get_max_qubits()}")

    # Create random Hamiltonian and circuit
    hamiltonian = create_random_hamiltonian(n_qubits, n_terms=20)
    circuit = create_random_circuit(n_qubits, depth=2)

    # Warmup runs
    print(f"Performing {warmup} warmup runs...")
    for _ in range(warmup):
        simulator.compute_energy(circuit, hamiltonian)

    # Timed runs
    times = []
    for i in range(n_trials):
        start = time.perf_counter()
        energy = simulator.compute_energy(circuit, hamiltonian)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed*1000:.2f} ms, energy = {energy:.6f}")

    # Statistics
    avg_time = np.mean(times) * 1000  # ms
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    std_time = np.std(times) * 1000

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Minimum time: {min_time:.2f} ms")
    print(f"  Maximum time: {max_time:.2f} ms")
    print(f"  Std dev:      {std_time:.2f} ms")

    # Check target
    target_met = avg_time < 500.0
    if target_met:
        print(f"  ✓ Performance target met: {avg_time:.2f} ms < 500 ms")
    else:
        print(f"  ✗ Performance target NOT met: {avg_time:.2f} ms >= 500 ms")

    # Memory estimation
    memory_gb = simulator.estimate_memory(n_qubits)
    memory_mb = memory_gb * 1024
    print(f"  Estimated memory usage: {memory_gb:.6f} GB ({memory_mb:.2f} MB)")

    return {
        'n_qubits': n_qubits,
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time,
        'target_met': target_met,
        'memory_gb': memory_gb,
        'n_trials': n_trials,
        'warmup': warmup,
    }


if __name__ == "__main__":
    results = benchmark_8qubit(n_trials=5, warmup=2)
    sys.exit(0 if results['target_met'] else 1)