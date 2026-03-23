"""Integration tests for Qubit UCC Search (Task 007).

Tests:
1. QubitOperatorPool non-empty for H2 and LiH
2. QubitUCCSearchController.search() returns SearchResult with real energy
3. qubit_vs_fermion_lih_10q.json exists and has valid schema
"""

import os
import json
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_mol():
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def lih_4q_mol():
    """LiH minimal active space (2,2) — 4 qubits, fast."""
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def lih_10q_mol():
    """LiH (2,5) active space — 10 qubits."""
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 5),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


# ---------------------------------------------------------------------------
# Pool non-empty tests
# ---------------------------------------------------------------------------

class TestQubitPoolNonEmpty:
    def test_qubit_pool_nonempty_h2(self, h2_mol):
        """QubitOperatorPool generates at least 1 operator for H2."""
        from rlqas.phase3.qubit_ops.operator_pool import QubitOperatorPool
        pool = QubitOperatorPool(h2_mol)
        assert pool.get_pool_size() >= 1, (
            f"H2 qubit pool is empty: get_pool_size() = {pool.get_pool_size()}"
        )
        ops = pool.get_pool()
        assert len(ops) >= 1

    def test_qubit_pool_nonempty_lih(self, lih_10q_mol):
        """QubitOperatorPool generates at least 5 operators for LiH (2,5)."""
        from rlqas.phase3.qubit_ops.operator_pool import QubitOperatorPool
        pool = QubitOperatorPool(lih_10q_mol)
        assert pool.get_pool_size() >= 5, (
            f"LiH qubit pool too small: {pool.get_pool_size()} < 5"
        )

    def test_qubit_pool_operator_to_circuit(self, h2_mol):
        """operator_to_circuit() returns non-trivial circuit for H2."""
        from rlqas.phase3.qubit_ops.operator_pool import QubitOperatorPool
        pool = QubitOperatorPool(h2_mol)
        assert pool.get_pool_size() >= 1
        circ = pool.operator_to_circuit(0, h2_mol.n_qubits)
        circ_str = str(circ)
        assert len(circ_str) > 20, (
            f"Operator 0 produces trivial/empty circuit: {circ_str!r}"
        )


# ---------------------------------------------------------------------------
# QubitUCCSearchController basic test
# ---------------------------------------------------------------------------

class TestQubitControllerSearch:
    def test_qubit_controller_search(self, lih_4q_mol):
        """QubitUCCSearchController.search() returns SearchResult with real energy."""
        from rlqas.phase3.qubit_ops.controller import QubitUCCSearchController

        ctrl = QubitUCCSearchController(
            lih_4q_mol,
            agent_type="ppo",
            config={
                "environment": {
                    "run_classical_opt": True,
                    "complexity_penalty": 0.0,
                    "max_depth": 8,
                    "param_init_strategy": "zeros",
                }
            },
        )
        result = ctrl.search(n_episodes=30, early_stop_threshold=1.6e-3)

        assert result is not None
        assert result.best_energy is not None, "best_energy is None"
        assert np.isfinite(result.best_energy), f"best_energy is not finite: {result.best_energy}"
        assert len(result.training_history) >= 1, "training_history is empty"

        energy_error = abs(result.best_energy - lih_4q_mol.fci_energy)
        # LiH (2,2) has very few excitations; chemical accuracy is easily achievable
        print(
            f"QubitUCCSearchController LiH (2,2): "
            f"best={result.best_energy:.6f} Ha, error={energy_error * 1000:.4f} mHa"
        )
        # Energy must be physical (below -7.0 Ha for LiH)
        assert result.best_energy < -7.0, (
            f"Energy {result.best_energy:.6f} is not physical (above -7.0 Ha)"
        )


# ---------------------------------------------------------------------------
# Comparison JSON exists and is valid
# ---------------------------------------------------------------------------

class TestQubitVsFermionJson:
    def test_qubit_vs_fermion_json_saved(self):
        """qubit_vs_fermion_lih_10q.json exists at expected path."""
        path = "results/phase3_integration/qubit_vs_fermion_lih_10q.json"
        assert os.path.exists(path), (
            f"Missing: {path}. Run Task 007 comparison script first."
        )

    def test_qubit_vs_fermion_json_schema(self):
        """qubit_vs_fermion_lih_10q.json has required schema fields."""
        path = "results/phase3_integration/qubit_vs_fermion_lih_10q.json"
        if not os.path.exists(path):
            pytest.skip("Comparison JSON not generated yet")

        with open(path) as f:
            data = json.load(f)

        assert "fermion" in data, "Missing 'fermion' key"
        assert "qubit" in data, "Missing 'qubit' key"
        assert "comparison" in data, "Missing 'comparison' key"

        for algo in ("fermion", "qubit"):
            entry = data[algo]
            assert "best_energy" in entry, f"Missing {algo}.best_energy"
            assert "energy_error_mha" in entry, f"Missing {algo}.energy_error_mha"
            assert "chemical_accuracy_reached" in entry
            assert "operator_count" in entry

        assert data["fermion"]["best_energy"] < -7.0, (
            f"Fermion energy {data['fermion']['best_energy']:.6f} not physical"
        )
        assert data["qubit"]["best_energy"] < -7.0, (
            f"Qubit energy {data['qubit']['best_energy']:.6f} not physical"
        )

        print(
            f"Fermion: {data['fermion']['best_energy']:.6f} Ha "
            f"({data['fermion']['energy_error_mha']:.4f} mHa) "
            f"CA={data['fermion']['chemical_accuracy_reached']}"
        )
        print(
            f"Qubit: {data['qubit']['best_energy']:.6f} Ha "
            f"({data['qubit']['energy_error_mha']:.4f} mHa) "
            f"CA={data['qubit']['chemical_accuracy_reached']}"
        )
        print(f"Comparison: {data['comparison']}")

    def test_fermion_achieves_chemical_accuracy(self):
        """Fermion operator search must achieve chemical accuracy on LiH 10q."""
        path = "results/phase3_integration/qubit_vs_fermion_lih_10q.json"
        if not os.path.exists(path):
            pytest.skip("Comparison JSON not generated yet")

        with open(path) as f:
            data = json.load(f)

        fermion_error_mha = data["fermion"]["energy_error_mha"]
        assert fermion_error_mha < 1.6, (
            f"Fermion operator search did NOT achieve chemical accuracy on LiH 10q. "
            f"Error = {fermion_error_mha:.4f} mHa (threshold = 1.6 mHa). "
            "UCCSearchController with PPO should reach chemical accuracy within 200 episodes."
        )
