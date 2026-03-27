"""Tests for circuit encoders (Task 005)."""
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
from rlqas.phase3.encoding import (
    MatrixEncoder, SparseEncoder, OneHotEncoder, EncoderFactory, EncodingBenchmark,
    CircuitEncoder,
)
from rlqas.phase3.encoding.base_encoder import CircuitEncoder as CircuitEncoderBase
from rlqas.phase3.hybrid_search.circuit_builder import (
    HybridFusionStrategy, HybridCircuitBuilder,
)


@pytest.fixture(scope="module")
def h2_mol():
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def h2_circuits(h2_mol):
    builder = UCCCircuitBuilder(h2_mol)
    params = np.zeros(builder.n_params)
    c0 = builder.build_circuit([], params.copy())
    c1 = builder.build_circuit(
        [builder.available_excitations[0]], params.copy()
    )
    return [c0, c1]


@pytest.fixture(scope="module")
def hybrid_circuit(h2_mol):
    strategy = HybridFusionStrategy()
    builder = HybridCircuitBuilder(h2_mol, strategy)
    return builder.build_hybrid_circuit(
        ["HEA", "UCC"], [{}, {"excitations": [0]}]
    )


# ─────────────────────────────────────────────
# MatrixEncoder tests
# ─────────────────────────────────────────────

class TestMatrixEncoder:
    def test_output_dim_matches_encode(self, h2_circuits):
        enc = MatrixEncoder()
        for circ in h2_circuits:
            vec = enc.encode(circ, 4, 10)
            assert len(vec) == enc.output_dim(4, 10), (
                f"len={len(vec)} != output_dim={enc.output_dim(4, 10)}"
            )

    def test_output_dim_formula(self):
        enc = MatrixEncoder()
        assert enc.output_dim(4, 10) == 40
        assert enc.output_dim(10, 15) == 150

    def test_determinism(self, h2_circuits):
        enc = MatrixEncoder()
        v1 = enc.encode(h2_circuits[0], 4, 10)
        v2 = enc.encode(h2_circuits[0], 4, 10)
        np.testing.assert_array_equal(v1, v2)

    def test_none_circuit(self):
        enc = MatrixEncoder()
        vec = enc.encode(None, 4, 10)
        assert len(vec) == 40
        assert np.all(vec == 0)

    def test_hybrid_circuit(self, hybrid_circuit):
        enc = MatrixEncoder()
        vec = enc.encode(hybrid_circuit, 4, 10)
        assert len(vec) == 40
        assert vec.dtype == np.float32

    def test_non_empty_for_non_trivial_circuit(self, h2_circuits):
        """A circuit with gates should produce a non-all-zero vector."""
        enc = MatrixEncoder()
        # c1 has gates; one of the two circuits should be non-zero
        vecs = [enc.encode(c, 4, 10) for c in h2_circuits]
        # At least one should be non-zero (both actually have gates)
        assert any(np.any(v != 0) for v in vecs)

    def test_dtype_float32(self, h2_circuits):
        enc = MatrixEncoder()
        vec = enc.encode(h2_circuits[0], 4, 10)
        assert vec.dtype == np.float32


# ─────────────────────────────────────────────
# SparseEncoder tests
# ─────────────────────────────────────────────

class TestSparseEncoder:
    def test_output_dim_matches_encode(self, h2_circuits):
        enc = SparseEncoder()
        for circ in h2_circuits:
            vec = enc.encode(circ, 4, 10)
            assert len(vec) == enc.output_dim(4, 10)

    def test_determinism(self, h2_circuits):
        enc = SparseEncoder()
        v1 = enc.encode(h2_circuits[1], 4, 10)
        v2 = enc.encode(h2_circuits[1], 4, 10)
        np.testing.assert_array_equal(v1, v2)

    def test_custom_max_gates(self, h2_circuits):
        enc = SparseEncoder(max_gates=5)
        vec = enc.encode(h2_circuits[0], 4, 10)
        assert len(vec) == 15  # 5 * 3

    def test_output_dim_default(self):
        enc = SparseEncoder()
        # default max_gates = n_qubits * max_depth
        assert enc.output_dim(4, 10) == 4 * 10 * 3  # 120


# ─────────────────────────────────────────────
# OneHotEncoder tests
# ─────────────────────────────────────────────

class TestOneHotEncoder:
    def test_output_dim_matches_encode(self, h2_circuits):
        enc = OneHotEncoder()
        for circ in h2_circuits:
            vec = enc.encode(circ, 4, 10)
            assert len(vec) == enc.output_dim(4, 10)

    def test_output_dim_formula(self):
        enc = OneHotEncoder()
        n_types = CircuitEncoderBase.N_GATE_TYPES
        assert enc.output_dim(4, 10) == 4 * 10 * n_types

    def test_determinism(self, h2_circuits):
        enc = OneHotEncoder()
        v1 = enc.encode(h2_circuits[0], 4, 10)
        v2 = enc.encode(h2_circuits[0], 4, 10)
        np.testing.assert_array_equal(v1, v2)

    def test_one_hot_valid(self, h2_circuits):
        """Each position should have exactly one 1.0 (identity or specific gate)."""
        enc = OneHotEncoder()
        vec = enc.encode(h2_circuits[0], 4, 10)
        n_types = CircuitEncoderBase.N_GATE_TYPES
        n_positions = 4 * 10
        vec_2d = vec.reshape(n_positions, n_types)
        # Each row should sum to 1.0 (exactly one hot bit)
        row_sums = vec_2d.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(n_positions))


# ─────────────────────────────────────────────
# EncoderFactory tests
# ─────────────────────────────────────────────

class TestEncoderFactory:
    def test_creates_matrix(self):
        enc = EncoderFactory.create("matrix")
        assert isinstance(enc, MatrixEncoder)

    def test_creates_sparse(self):
        enc = EncoderFactory.create("sparse")
        assert isinstance(enc, SparseEncoder)

    def test_creates_one_hot(self):
        enc = EncoderFactory.create("one_hot")
        assert isinstance(enc, OneHotEncoder)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            EncoderFactory.create("invalid_encoder")

    def test_sparse_config_max_gates(self, h2_circuits):
        enc = EncoderFactory.create("sparse", {"max_gates": 20})
        vec = enc.encode(h2_circuits[0], 4, 10)
        assert len(vec) == 60  # 20 * 3

    def test_all_types_are_base_class(self):
        for method in ("matrix", "sparse", "one_hot"):
            enc = EncoderFactory.create(method)
            assert isinstance(enc, CircuitEncoderBase)


# ─────────────────────────────────────────────
# EncodingBenchmark tests
# ─────────────────────────────────────────────

class TestEncodingBenchmark:
    def test_benchmark_runs(self, h2_circuits):
        bench = EncodingBenchmark()
        results = bench.run(h2_circuits, 4, 10, n_repeats=3)
        assert "matrix" in results
        assert "sparse" in results
        assert "one_hot" in results

    def test_benchmark_fields(self, h2_circuits):
        bench = EncodingBenchmark()
        results = bench.run(h2_circuits, 4, 10, n_repeats=3)
        for method, data in results.items():
            assert "mean_us" in data, f"Missing mean_us for {method}"
            assert "output_dim" in data, f"Missing output_dim for {method}"
            assert data["mean_us"] >= 0

    def test_benchmark_output_dims_correct(self, h2_circuits):
        bench = EncodingBenchmark()
        results = bench.run(h2_circuits, 4, 10, n_repeats=2)
        assert results["matrix"]["output_dim"] == 4 * 10
        assert results["sparse"]["output_dim"] == 4 * 10 * 3
        n_types = CircuitEncoderBase.N_GATE_TYPES
        assert results["one_hot"]["output_dim"] == 4 * 10 * n_types

    def test_benchmark_save(self, h2_circuits, tmp_path):
        bench = EncodingBenchmark()
        results = bench.run(h2_circuits, 4, 10, n_repeats=2)
        out_path = str(tmp_path / "bench.json")
        bench.save_results(results, out_path)
        import json, os
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert "matrix" in loaded
