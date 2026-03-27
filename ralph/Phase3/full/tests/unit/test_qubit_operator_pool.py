"""Tests for QubitOperatorPool and QubitUCCSearchController (Task 007)."""
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase3.qubit_ops.operator_pool import QubitOperatorPool
from rlqas.phase3.qubit_ops.controller import QubitUCCSearchController
from rlqas.phase3.qubit_ops.adapter import circuit_to_str


@pytest.fixture(scope="module")
def h2_mol():
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def lih_10q_mol():
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 5),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


# ─────────────────────────────────────────────
# QubitOperatorPool tests
# ─────────────────────────────────────────────

class TestQubitOperatorPool:
    def test_pool_non_empty_h2(self, h2_mol):
        """H2 qubit operator pool must be non-empty."""
        pool = QubitOperatorPool(h2_mol)
        assert pool.get_pool_size() >= 1, (
            f"H2 qubit operator pool is empty. "
            f"Hamiltonian has {len(h2_mol.hamiltonian.terms)} terms."
        )

    def test_pool_non_empty_lih(self, lih_10q_mol):
        """LiH qubit operator pool must have multiple operators."""
        pool = QubitOperatorPool(lih_10q_mol)
        assert pool.get_pool_size() >= 5, (
            f"LiH qubit operator pool has only {pool.get_pool_size()} operators."
        )

    def test_get_pool_returns_list(self, h2_mol):
        pool = QubitOperatorPool(h2_mol)
        operators = pool.get_pool()
        assert isinstance(operators, list)
        assert len(operators) == pool.get_pool_size()

    def test_pool_size_consistent(self, h2_mol):
        """Pool size should be consistent across calls."""
        pool = QubitOperatorPool(h2_mol)
        size1 = pool.get_pool_size()
        size2 = pool.get_pool_size()
        assert size1 == size2

    def test_operator_to_circuit_returns_tc_circuit(self, h2_mol):
        """operator_to_circuit should return a tensorcircuit.Circuit."""
        import tensorcircuit as tc
        pool = QubitOperatorPool(h2_mol)
        circ = pool.operator_to_circuit(0, h2_mol.n_qubits)
        assert isinstance(circ, tc.Circuit)

    def test_operator_to_circuit_non_trivial(self, lih_10q_mol):
        """Anti-hollow check: operators must produce non-trivial circuits."""
        pool = QubitOperatorPool(lih_10q_mol)
        for i in range(min(3, pool.get_pool_size())):
            circ = pool.operator_to_circuit(i, lih_10q_mol.n_qubits)
            circ_str = circuit_to_str(circ)
            assert len(circ_str) > 20, (
                f"Operator {i} produces trivial/empty circuit: {circ_str!r}"
            )

    def test_operator_to_circuit_non_trivial_h2(self, h2_mol):
        """H2: at least one operator should produce a multi-gate circuit."""
        import tensorcircuit as tc
        pool = QubitOperatorPool(h2_mol)
        found_nontrivial = False
        for i in range(pool.get_pool_size()):
            circ = pool.operator_to_circuit(i, h2_mol.n_qubits)
            if len(circ._qir) > 1:
                found_nontrivial = True
                break
        assert found_nontrivial, "No non-trivial circuits found for H2 pool"

    def test_get_operator_string(self, h2_mol):
        pool = QubitOperatorPool(h2_mol)
        op_str = pool.get_operator_string(0)
        assert isinstance(op_str, str)
        assert len(op_str) > 0

    def test_max_operators_config(self, lih_10q_mol):
        """max_operators config should limit pool size."""
        pool = QubitOperatorPool(lih_10q_mol, {"max_operators": 3})
        assert pool.get_pool_size() <= 3

    def test_excitation_level_single(self, lih_10q_mol):
        """Single excitation level should include only 1-2 qubit Pauli terms."""
        pool = QubitOperatorPool(lih_10q_mol, {"excitation_level": "s"})
        for term in pool.get_pool():
            assert len(term) <= 2, f"Single excitation pool has {len(term)}-qubit term"

    def test_excitation_level_double(self, lih_10q_mol):
        """Double excitation level should include only 3+ qubit Pauli terms."""
        pool = QubitOperatorPool(lih_10q_mol, {"excitation_level": "d"})
        for term in pool.get_pool():
            assert len(term) > 2, f"Double excitation pool has {len(term)}-qubit term"


# ─────────────────────────────────────────────
# QubitUCCSearchController tests
# ─────────────────────────────────────────────

class TestQubitUCCSearchController:
    def test_instantiation(self, h2_mol):
        ctrl = QubitUCCSearchController(h2_mol, "ppo", {"max_depth": 5})
        assert ctrl is not None
        assert ctrl.qubit_pool.get_pool_size() >= 1

    def test_search_returns_result(self, h2_mol):
        """Search should return a SearchResult with valid fields."""
        ctrl = QubitUCCSearchController(h2_mol, "ppo", {"max_depth": 5})
        result = ctrl.search(n_episodes=5)
        assert result is not None
        assert result.best_energy is not None
        assert isinstance(result.best_energy, float)

    def test_pool_size_in_metrics(self, h2_mol):
        """qubit_pool_size must be recorded in performance_metrics."""
        ctrl = QubitUCCSearchController(h2_mol, "ppo", {"max_depth": 5})
        result = ctrl.search(n_episodes=3)
        assert "qubit_pool_size" in result.performance_metrics
        assert result.performance_metrics["qubit_pool_size"] >= 1

    def test_training_history_populated(self, h2_mol):
        """Training history should have entries after search."""
        ctrl = QubitUCCSearchController(h2_mol, "ppo", {"max_depth": 5})
        result = ctrl.search(n_episodes=5)
        assert len(result.training_history) >= 1

    def test_fusion_template_set(self, h2_mol):
        """fusion_template should be non-empty."""
        ctrl = QubitUCCSearchController(h2_mol, "ppo", {"max_depth": 5})
        result = ctrl.search(n_episodes=3)
        assert isinstance(result.fusion_template, list)
        assert len(result.fusion_template) > 0

    def test_env_energy_not_hollow(self, h2_mol):
        """Anti-hollow: energy must be physically meaningful (not just 0 or trivial)."""
        ctrl = QubitUCCSearchController(h2_mol, "ppo", {"max_depth": 5})
        result = ctrl.search(n_episodes=10)
        # H2 FCI energy is around -1.137 Ha; must be negative and in physical range
        assert result.best_energy < 0, (
            f"Best energy {result.best_energy} is not negative — hollow implementation"
        )
        assert result.best_energy > -10.0, (
            f"Best energy {result.best_energy} is unreasonably low"
        )
