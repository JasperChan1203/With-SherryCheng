"""Integration tests for Hybrid Architecture Search.

Tests PPO and DQN on BeH2 (4,4) 8q and verifies fusion template recording.
These tests are marked 'medium' — run in CI but take 2-5 minutes each.
"""

import json
import os
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule


@pytest.fixture(scope="module")
def beh2_8q():
    return process_molecule(
        "BeH2", 1.3, "UCC",
        active_space=(4, 4),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


@pytest.fixture(scope="module")
def lih_4q():
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


class TestHybridControllerAlgorithms:
    """PPO and DQN hybrid search on BeH2 8q must achieve chemical accuracy."""

    def test_hybrid_controller_ppo_beh2_8q(self, beh2_8q):
        """PPO hybrid search on BeH2 (4,4) 8q; assert energy_error < 1.6e-3."""
        from rlqas.phase3.hybrid_search.controller import HybridSearchController

        mol = beh2_8q
        controller = HybridSearchController(
            mol,
            agent_type="ppo",
            config={
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "early_stop_threshold": 1.6e-3,
                "max_depth": 12,
                "max_blocks": 6,
            },
        )
        result = controller.search(n_episodes=200, early_stop_threshold=1.6e-3)

        assert result.best_energy is not None
        energy_error = abs(result.best_energy - mol.fci_energy)
        assert energy_error < 1.6e-3, (
            f"BeH2 (4,4) 8q PPO: chemical accuracy NOT achieved. "
            f"Error = {energy_error * 1000:.4f} mHa >= 1.6 mHa. "
            f"best_energy={result.best_energy:.6f}, fci={mol.fci_energy:.6f}"
        )

        # Save PPO result for comparison
        results_dir = "results/phase3_integration"
        os.makedirs(results_dir, exist_ok=True)
        ppo_result = {
            "molecule": "BeH2",
            "active_space": [4, 4],
            "n_qubits": mol.n_qubits,
            "agent": "ppo",
            "best_energy": result.best_energy,
            "fci_energy": mol.fci_energy,
            "energy_error_mha": energy_error * 1000,
            "chemical_accuracy_reached": bool(energy_error < 1.6e-3),
            "n_episodes": len(result.training_history),
        }
        # Store for comparison with DQN
        TestHybridControllerAlgorithms._ppo_result = ppo_result

    def test_hybrid_controller_dqn_beh2_8q(self, beh2_8q):
        """DQN hybrid search on BeH2 (4,4) 8q; assert energy_error < 1.6e-3."""
        from rlqas.phase3.hybrid_search.controller import HybridSearchController

        mol = beh2_8q
        controller = HybridSearchController(
            mol,
            agent_type="dqn",
            config={
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "early_stop_threshold": 1.6e-3,
                "max_depth": 12,
                "max_blocks": 6,
            },
        )
        result = controller.search(n_episodes=200, early_stop_threshold=1.6e-3)

        assert result.best_energy is not None
        energy_error = abs(result.best_energy - mol.fci_energy)
        assert energy_error < 1.6e-3, (
            f"BeH2 (4,4) 8q DQN: chemical accuracy NOT achieved. "
            f"Error = {energy_error * 1000:.4f} mHa >= 1.6 mHa. "
            f"best_energy={result.best_energy:.6f}, fci={mol.fci_energy:.6f}"
        )

        # Save algorithm comparison to results file
        results_dir = "results/phase3_integration"
        os.makedirs(results_dir, exist_ok=True)
        dqn_result = {
            "molecule": "BeH2",
            "active_space": [4, 4],
            "n_qubits": mol.n_qubits,
            "agent": "dqn",
            "best_energy": result.best_energy,
            "fci_energy": mol.fci_energy,
            "energy_error_mha": energy_error * 1000,
            "chemical_accuracy_reached": bool(energy_error < 1.6e-3),
            "n_episodes": len(result.training_history),
        }

        ppo_result = getattr(
            TestHybridControllerAlgorithms, "_ppo_result", None
        )
        comparison = {
            "ppo": ppo_result,
            "dqn": dqn_result,
            "notes": "Both PPO and DQN on BeH2 (4,4) 8q hybrid search",
        }
        comp_path = os.path.join(results_dir, "algorithm_comparison.json")
        with open(comp_path, "w") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"[INFO] Algorithm comparison saved to {comp_path}")


class TestFusionTemplateRecording:
    def test_fusion_template_recorded(self, lih_4q):
        """SearchResult.fusion_template must be a non-empty list of strings."""
        from rlqas.phase3.hybrid_search.controller import HybridSearchController

        mol = lih_4q
        controller = HybridSearchController(
            mol,
            agent_type="ppo",
            config={
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "max_depth": 8,
                "max_blocks": 4,
            },
        )
        result = controller.search(n_episodes=50, early_stop_threshold=1.6e-3)

        assert result.fusion_template is not None
        assert isinstance(result.fusion_template, list)
        assert len(result.fusion_template) >= 1, (
            "fusion_template is empty — block selection not tracked"
        )
        assert all(isinstance(b, str) for b in result.fusion_template), (
            "fusion_template must be list of strings"
        )
        print(f"[PASS] fusion_template = {result.fusion_template}")
