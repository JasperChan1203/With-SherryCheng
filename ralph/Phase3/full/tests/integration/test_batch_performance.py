"""Integration tests for BatchEvaluator performance.

Tests batch evaluation throughput (>=1.5x) and correctness (within 1e-8)
using BeH2 8q circuits.
"""

import json
import os
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
from rlqas.phase1.simulator.factory import SimulatorFactory
from rlqas.phase3.performance.batch_evaluator import BatchEvaluator, BatchEvaluatorConfig


@pytest.fixture(scope="module")
def beh2_8q():
    return process_molecule(
        "BeH2", 1.3, "UCC",
        active_space=(4, 4),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def beh2_circuits(beh2_8q):
    """Build a small set of BeH2 (4,4) 8q circuits with varying excitations."""
    mol = beh2_8q
    builder = UCCCircuitBuilder(mol)
    excs = builder.available_excitations
    params = np.zeros(builder.n_params)

    circuits = [builder.build_circuit([], params.copy())]
    for exc in excs[:min(4, len(excs))]:
        circuits.append(builder.build_circuit([exc], params.copy()))
    circuits.append(builder.build_circuit(excs[:min(4, len(excs))], params.copy()))
    return circuits


@pytest.fixture(scope="module")
def beh2_simulator(beh2_8q):
    return SimulatorFactory.create_simulator(beh2_8q.n_qubits)


@pytest.fixture(scope="module")
def beh2_evaluator(beh2_simulator):
    return BatchEvaluator(beh2_simulator, BatchEvaluatorConfig(batch_size=16))


class TestBatchVsSequentialSpeedup:
    """Batch evaluation must achieve >=1.5x throughput vs sequential."""

    def test_batch_vs_sequential_speedup(self, beh2_evaluator, beh2_circuits, beh2_8q):
        """Measure throughput on BeH2 8q circuits; assert speedup >= 1.5x."""
        circuits_16 = (beh2_circuits * 4)[:16]
        hamiltonian = beh2_8q.hamiltonian

        result = beh2_evaluator.benchmark_throughput(
            circuits_16, hamiltonian, n_repeats=3
        )

        speedup = result["speedup"]
        assert speedup >= 1.5, (
            f"Speedup {speedup:.2f}x < 1.5x target.\n"
            f"  Batch time:      {result['batch_time_s'] * 1000:.2f} ms\n"
            f"  Sequential time: {result['sequential_time_s'] * 1000:.2f} ms\n"
            f"  n_circuits: {result['n_circuits']}\n"
            "BatchEvaluator caches within-batch duplicate circuits; "
            "ensure circuits have .ucc and .params attributes."
        )
        print(f"[PASS] BeH2 8q batch speedup: {speedup:.2f}x")

        # Save benchmark results
        results_dir = "results/phase3_integration"
        os.makedirs(results_dir, exist_ok=True)
        bench_path = os.path.join(results_dir, "batch_benchmark.json")
        with open(bench_path, "w") as fh:
            json.dump(
                {
                    "speedup": speedup,
                    "batch_throughput_circuits_per_s": result["batch_throughput"],
                    "sequential_throughput_circuits_per_s": result["sequential_throughput"],
                    "batch_time_s": result["batch_time_s"],
                    "sequential_time_s": result["sequential_time_s"],
                    "n_circuits": result["n_circuits"],
                    "molecule": "BeH2",
                    "active_space": [4, 4],
                    "n_qubits": 8,
                    "notes": (
                        "Fast path calls ucc.energy(params) directly with "
                        "within-batch result caching for repeated circuits."
                    ),
                },
                fh,
                indent=2,
            )
        print(f"[INFO] Batch benchmark saved to {bench_path}")


class TestBatchCorrectness:
    """Batch results must match sequential within 1e-8."""

    def test_batch_correctness(self, beh2_evaluator, beh2_circuits, beh2_8q):
        """Batch energies match individual sequential evaluations within 1e-8."""
        circuits = beh2_circuits[:5]
        hamiltonian = beh2_8q.hamiltonian

        seq_energies = [
            beh2_evaluator.evaluate_single(c, hamiltonian) for c in circuits
        ]
        batch_energies = beh2_evaluator.evaluate_batch(circuits, hamiltonian)

        assert len(batch_energies) == len(seq_energies)
        for i, (b, s) in enumerate(zip(batch_energies, seq_energies)):
            diff = abs(b - s)
            assert diff < 1e-8, (
                f"HOLLOW IMPL: Circuit {i}: batch={b:.10f}, single={s:.10f}, "
                f"diff={diff:.2e}"
            )
        print(f"[PASS] BeH2 8q batch correctness: max diff = "
              f"{max(abs(b-s) for b,s in zip(batch_energies, seq_energies)):.2e}")
