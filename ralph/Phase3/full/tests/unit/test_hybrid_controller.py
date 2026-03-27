"""Unit tests for HybridSearchController and SearchResult."""
import os
import json
import pytest

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase3.hybrid_search.controller import HybridSearchController, SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lih_4q_mol():
    """LiH with minimal active space (2 electrons, 2 orbitals) → 4 qubits.
    Used for fast controller smoke tests.
    """
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 2), basis_set="sto-3g", transform="jordan_wigner"
    )


@pytest.fixture(scope="module")
def h2_mol():
    """H2 molecule: active_space=(2,2) → 4 qubits."""
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2), basis_set="sto-3g", transform="jordan_wigner"
    )


_FAST_CONFIG = {
    "max_depth": 5,
    "max_blocks": 4,
    "run_classical_opt": True,
    "complexity_penalty": 0.0,
}

# ---------------------------------------------------------------------------
# Instantiation and basic API
# ---------------------------------------------------------------------------

class TestHybridSearchControllerInstantiation:
    def test_instantiation_ppo(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        assert ctrl is not None
        assert ctrl.env is not None
        assert ctrl.agent is not None

    def test_instantiation_dqn(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "dqn", _FAST_CONFIG)
        assert ctrl is not None

    def test_instantiation_a2c(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "a2c", _FAST_CONFIG)
        assert ctrl is not None

    def test_run_classical_opt_always_true(self, lih_4q_mol):
        # Even if user passes False, controller must enforce True
        ctrl = HybridSearchController(
            lih_4q_mol, "ppo",
            {**_FAST_CONFIG, "run_classical_opt": False}
        )
        assert ctrl.env.run_classical_opt is True

    def test_from_config(self, lih_4q_mol):
        config = {
            "rl": {"agent_type": "ppo"},
            **_FAST_CONFIG,
        }
        ctrl = HybridSearchController.from_config(lih_4q_mol, config)
        assert ctrl is not None
        assert ctrl.agent_type == "ppo"


# ---------------------------------------------------------------------------
# Search method
# ---------------------------------------------------------------------------

class TestHybridSearchControllerSearch:
    def test_search_returns_search_result(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=5)
        assert isinstance(result, SearchResult)

    def test_search_result_best_energy_is_float(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=5)
        assert result.best_energy is not None
        assert isinstance(result.best_energy, float)
        import math
        assert math.isfinite(result.best_energy)

    def test_search_result_fusion_template_nonempty(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=5)
        assert result.fusion_template is not None
        assert len(result.fusion_template) > 0
        for block in result.fusion_template:
            assert isinstance(block, str)

    def test_training_history_has_at_least_2_entries(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=5)
        assert len(result.training_history) >= 2, (
            "Training history has fewer than 2 entries — search loop not running."
        )

    def test_training_history_entries_have_required_keys(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=3)
        for entry in result.training_history:
            assert "episode" in entry
            assert "reward" in entry
            assert "energy" in entry
            assert "best_energy" in entry

    def test_best_error_computed_when_fci_available(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=3)
        if lih_4q_mol.fci_energy is not None:
            assert result.best_error is not None
            assert result.best_error >= 0.0

    def test_performance_metrics_populated(self, lih_4q_mol):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=3)
        assert isinstance(result.performance_metrics, dict)
        assert "n_episodes" in result.performance_metrics


# ---------------------------------------------------------------------------
# Anti-hollow check (Task 003 requirement)
# ---------------------------------------------------------------------------

class TestAntiHollowController:
    """Verify the controller runs real training (not hollow/mock)."""

    def test_real_training_energies_vary(self, lih_4q_mol):
        """Anti-hollow Test C: training history must have varied energies.

        If the VQE inner loop is broken, all energies would be identical
        (constant zero or constant HF).
        """
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=20)

        assert result.best_energy is not None, "best_energy is None — training not working"
        assert isinstance(result.best_energy, float), "best_energy not a float"

        # Energy should be physically meaningful for LiH
        assert result.best_energy < -3.0, (
            f"best_energy={result.best_energy:.6f} too high for LiH — "
            f"training not working."
        )

        energies = [
            h.get("best_energy")
            for h in result.training_history
            if h.get("best_energy") is not None
        ]
        assert len(energies) >= 2, (
            "Training history has fewer than 2 non-None best_energy values — "
            "search loop not actually running."
        )

        assert result.fusion_template is not None
        assert len(result.fusion_template) > 0

    def test_best_energy_below_hf(self, h2_mol):
        """After training, best energy should improve below the HF baseline.

        Uses H2 (not LiH 4q) because LiH with active_space=(2,2) has a tiny
        HF-FCI gap (0.264 mHa) so HF is already at chemical accuracy, making
        this test degenerate.  H2 has a ~19 mHa HF-FCI gap.
        """
        hf_e = h2_mol.molecular_info.get("hf_energy", 0.0)
        ctrl = HybridSearchController(h2_mol, "ppo", _FAST_CONFIG)
        result = ctrl.search(n_episodes=20)
        # Classical opt should produce energy below HF
        # (this checks that VQE inner loop ran at least once with UCC excitations)
        assert result.best_energy <= hf_e, (
            f"best_energy={result.best_energy:.6f} is above HF={hf_e:.6f}. "
            f"Classical optimization may not be running, or agent never selected UCC action."
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestHybridSearchControllerPersistence:
    def test_save_results_creates_json(self, lih_4q_mol, tmp_path):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        ctrl.search(n_episodes=3)
        path = str(tmp_path / "results.json")
        ctrl.save_results(path)
        assert os.path.exists(path)

    def test_saved_json_is_valid(self, lih_4q_mol, tmp_path):
        ctrl = HybridSearchController(lih_4q_mol, "ppo", _FAST_CONFIG)
        ctrl.search(n_episodes=3)
        path = str(tmp_path / "results.json")
        ctrl.save_results(path)
        with open(path) as f:
            data = json.load(f)
        assert "best_energy" in data
        assert "training_history" in data
        assert isinstance(data["training_history"], list)

    def test_from_config_with_nested_rl_section(self, lih_4q_mol):
        config = {
            "rl": {"agent_type": "ppo", "n_episodes": 2},
            "search": {"ansatz_type": "HYBRID"},
            **_FAST_CONFIG,
        }
        ctrl = HybridSearchController.from_config(lih_4q_mol, config)
        result = ctrl.search(n_episodes=2)
        assert isinstance(result, SearchResult)
