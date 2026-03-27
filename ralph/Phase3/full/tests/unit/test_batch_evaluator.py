"""Unit tests for BatchEvaluator, CIVectorBenchmark, MemoryManager, and checkpoint utils.

These tests cover Task 004 acceptance criteria:
  - Batch results match sequential within 1e-8 (correctness)
  - Batch throughput >= 1.5x sequential (performance)
  - MemoryManager returns sensible values
  - CIVectorBenchmark runs and saves/loads results
  - Checkpoint save/load round-trips (including numpy arrays)
"""

import json
import os

import numpy as np
import pytest

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
from rlqas.phase1.simulator.factory import SimulatorFactory
from rlqas.phase3.performance.batch_evaluator import BatchEvaluator, BatchEvaluatorConfig
from rlqas.phase3.performance.benchmarking import CIVectorBenchmark
from rlqas.phase3.performance.checkpoint import load_checkpoint, save_checkpoint
from rlqas.phase3.performance.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_mol():
    """H2 molecule with 4 qubits (active_space=(2,2))."""
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def h2_simulator(h2_mol):
    return SimulatorFactory.create_simulator(h2_mol.n_qubits)


@pytest.fixture(scope="module")
def h2_circuits(h2_mol):
    """Build a small set of H2 circuits with varying excitation subsets."""
    builder = UCCCircuitBuilder(h2_mol)
    excs = builder.available_excitations
    circuits = []

    # Build circuits with 0, 1, 2, ... available excitations
    params = np.zeros(builder.n_params)
    circuits.append(builder.build_circuit([], params.copy()))
    for i, exc in enumerate(excs):
        circuits.append(builder.build_circuit([exc], params.copy()))
    circuits.append(builder.build_circuit(excs, params.copy()))
    return circuits


@pytest.fixture(scope="module")
def evaluator(h2_simulator):
    return BatchEvaluator(h2_simulator, BatchEvaluatorConfig(batch_size=16))


# ---------------------------------------------------------------------------
# BatchEvaluator — correctness tests
# ---------------------------------------------------------------------------


class TestBatchEvaluatorCorrectness:
    """Batch energies must match sequential within 1e-8."""

    def test_batch_matches_sequential(self, evaluator, h2_circuits, h2_mol):
        circuits = h2_circuits
        hamiltonian = h2_mol.hamiltonian

        seq_energies = [evaluator.evaluate_single(c, hamiltonian) for c in circuits]
        batch_energies = evaluator.evaluate_batch(circuits, hamiltonian)

        assert len(batch_energies) == len(seq_energies)
        for i, (b, s) in enumerate(zip(batch_energies, seq_energies)):
            diff = abs(b - s)
            assert diff < 1e-8, (
                f"HOLLOW IMPL: Circuit {i}: batch={b:.10f}, single={s:.10f}, "
                f"diff={diff:.2e} — batch evaluator not calling real simulator"
            )

    def test_empty_batch_returns_empty(self, evaluator, h2_mol):
        result = evaluator.evaluate_batch([], h2_mol.hamiltonian)
        assert result == []

    def test_single_circuit_batch_matches_single(self, evaluator, h2_circuits, h2_mol):
        circ = h2_circuits[0]
        batch = evaluator.evaluate_batch([circ], h2_mol.hamiltonian)
        single = evaluator.evaluate_single(circ, h2_mol.hamiltonian)
        assert len(batch) == 1
        assert abs(batch[0] - single) < 1e-8

    def test_batch_returns_finite_floats(self, evaluator, h2_circuits, h2_mol):
        energies = evaluator.evaluate_batch(h2_circuits, h2_mol.hamiltonian)
        for e in energies:
            assert isinstance(e, float)
            assert np.isfinite(e), f"Non-finite energy: {e}"

    def test_evaluate_single_returns_float(self, evaluator, h2_circuits, h2_mol):
        e = evaluator.evaluate_single(h2_circuits[0], h2_mol.hamiltonian)
        assert isinstance(e, float)
        assert np.isfinite(e)


# ---------------------------------------------------------------------------
# BatchEvaluator — throughput / speedup test
# ---------------------------------------------------------------------------


class TestBatchEvaluatorThroughput:
    """Batch must achieve >=1.5x throughput vs sequential on a 16-circuit batch."""

    def test_speedup_gte_1_5x(self, evaluator, h2_circuits, h2_mol):
        """Replicate circuits to batch_size=16 and measure speedup."""
        circuits_16 = (h2_circuits * 6)[:16]  # repeat to fill batch
        hamiltonian = h2_mol.hamiltonian

        result = evaluator.benchmark_throughput(circuits_16, hamiltonian, n_repeats=3)

        speedup = result["speedup"]
        assert speedup >= 1.5, (
            f"Speedup {speedup:.2f}x < 1.5x target.\n"
            f"  Batch time:      {result['batch_time_s']*1000:.2f} ms\n"
            f"  Sequential time: {result['sequential_time_s']*1000:.2f} ms\n"
            f"  n_circuits: {result['n_circuits']}\n"
            "Fast path bypasses simulator.compute_energy() overhead; check "
            "that circuits have .ucc and .params attributes."
        )
        print(f"[PASS] Batch speedup: {speedup:.2f}x")

        # Save benchmark results as required by PRD
        results_dir = "results/phase3_integration"
        os.makedirs(results_dir, exist_ok=True)
        bench_path = os.path.join(results_dir, "batch_benchmark.json")
        with open(bench_path, "w") as fh:
            json.dump({
                "speedup": speedup,
                "batch_throughput_circuits_per_s": result["batch_throughput"],
                "sequential_throughput_circuits_per_s": result["sequential_throughput"],
                "batch_time_s": result["batch_time_s"],
                "sequential_time_s": result["sequential_time_s"],
                "n_circuits": result["n_circuits"],
                "molecule": "H2",
                "n_qubits": 4,
                "notes": (
                    "Fast path calls ucc.energy(params) directly, bypassing "
                    "TencirchemCISimulator overhead (~10x per-circuit speedup)."
                ),
            }, fh, indent=2)
        print(f"[INFO] Benchmark results saved to {bench_path}")


# ---------------------------------------------------------------------------
# CIVectorBenchmark tests
# ---------------------------------------------------------------------------


class TestCIVectorBenchmark:
    def test_benchmark_runs_returns_dict(self):
        bench = CIVectorBenchmark()
        results = bench.run_benchmark([4], n_trials=2)
        assert 4 in results
        info = results[4]
        assert "mean_ms" in info
        assert "std_ms" in info
        assert "max_ms" in info
        assert "min_ms" in info
        assert info["mean_ms"] >= 0.0

    def test_benchmark_multiple_qubit_counts(self):
        bench = CIVectorBenchmark()
        results = bench.run_benchmark([4, 8], n_trials=2)
        assert 4 in results
        assert 8 in results

    def test_save_load_roundtrip(self, tmp_path):
        bench = CIVectorBenchmark()
        results = bench.run_benchmark([4], n_trials=2)
        path = str(tmp_path / "bench.json")
        bench.save_results(path)
        assert os.path.exists(path)

        loaded = bench.load_results(path)
        assert 4 in loaded
        # Values should be preserved
        assert abs(loaded[4]["mean_ms"] - results[4]["mean_ms"]) < 1e-6

    def test_save_without_run_raises(self):
        bench = CIVectorBenchmark()
        with pytest.raises(RuntimeError, match="run_benchmark"):
            bench.save_results("/tmp/should_not_exist.json")


# ---------------------------------------------------------------------------
# MemoryManager tests
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def test_check_memory_returns_required_keys(self):
        mm = MemoryManager(max_memory_gb=32.0)
        mem = mm.check_memory()
        for key in ("used_gb", "available_gb", "percent", "max_allowed_gb"):
            assert key in mem, f"Missing key: {key}"
        assert mem["used_gb"] >= 0.0
        assert mem["max_allowed_gb"] == 32.0

    def test_adapt_batch_size_returns_positive_int(self):
        mm = MemoryManager(max_memory_gb=32.0)
        result = mm.adapt_batch_size(16)
        assert isinstance(result, int)
        assert result >= 1

    def test_adapt_batch_size_unchanged_when_memory_low(self):
        # With a generous memory limit, batch size should be unchanged
        mm = MemoryManager(max_memory_gb=1000.0)
        assert mm.adapt_batch_size(16) == 16

    def test_adapt_batch_size_reduced_when_memory_tight(self):
        # Force threshold to 0% so any memory usage triggers reduction
        mm = MemoryManager(max_memory_gb=0.001)
        mm._threshold_fraction = 0.0
        result = mm.adapt_batch_size(16)
        assert result < 16

    def test_release_intermediate_state_does_not_raise(self):
        mm = MemoryManager()
        mm.release_intermediate_state()  # must not raise


# ---------------------------------------------------------------------------
# Checkpoint save/load tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_basic_scalars(self, tmp_path):
        path = str(tmp_path / "ckpt.json")
        state = {"episode": 42, "best_energy": -7.882, "name": "test"}
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["episode"] == 42
        assert abs(loaded["best_energy"] - (-7.882)) < 1e-10
        assert loaded["name"] == "test"

    def test_save_load_numpy_1d(self, tmp_path):
        path = str(tmp_path / "ckpt_np.json")
        arr = np.array([1.0, 2.0, 3.0])
        save_checkpoint({"params": arr, "step": 10}, path)
        loaded = load_checkpoint(path)
        assert isinstance(loaded["params"], np.ndarray)
        assert np.allclose(loaded["params"], arr)
        assert loaded["step"] == 10

    def test_save_load_numpy_2d(self, tmp_path):
        path = str(tmp_path / "ckpt_2d.json")
        arr = np.eye(3, dtype=np.float32)
        save_checkpoint({"matrix": arr}, path)
        loaded = load_checkpoint(path)
        restored = loaded["matrix"]
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (3, 3)
        assert np.allclose(restored, arr)

    def test_save_load_nested(self, tmp_path):
        path = str(tmp_path / "ckpt_nested.json")
        state = {
            "episode": 5,
            "metrics": {"energy": -7.5, "depth": 3},
            "history": [1.0, 2.0, 3.0],
        }
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["metrics"]["energy"] == -7.5
        assert loaded["history"] == [1.0, 2.0, 3.0]

    def test_checkpoint_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "subdir" / "nested" / "ckpt.json")
        save_checkpoint({"x": 1}, path)
        assert os.path.exists(path)

    def test_checkpoint_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "valid.json")
        save_checkpoint({"a": 1, "b": [1, 2, 3]}, path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["a"] == 1

    def test_numpy_integer_and_float_scalars(self, tmp_path):
        path = str(tmp_path / "np_scalars.json")
        state = {
            "int_val": np.int64(99),
            "float_val": np.float32(3.14),
        }
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["int_val"] == 99
        assert abs(loaded["float_val"] - 3.14) < 1e-5
