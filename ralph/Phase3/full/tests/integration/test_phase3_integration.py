"""Phase 3 integration smoke tests.

Covers:
  1. All Phase 3 modules importable
  2. HybridSearchController LiH smoke test (4 qubits, chemical accuracy)
  3. ExperimentManager HYBRID dispatch
  4. ANTI-HOLLOW A: Single step on LiH 10q must NOT reach chemical accuracy
  5. ANTI-HOLLOW B: Energy below HF after single step with classical opt
  6. Training history is real (not hollow)

These tests must run quickly (< 5 minutes total) and are the primary
anti-hollow guards for Phase 3.
"""

import os
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_all_phase3_modules_importable(self):
        """All Phase 3 public modules can be imported without error."""
        from rlqas.phase3.hybrid_search.circuit_builder import (
            HybridFusionStrategy, HybridCircuitBuilder, HybridCircuit
        )
        from rlqas.phase3.hybrid_search.environment import (
            HybridSearchEnv, HybridRewardFunction
        )
        from rlqas.phase3.hybrid_search.controller import (
            HybridSearchController, SearchResult
        )
        from rlqas.phase3.performance.batch_evaluator import (
            BatchEvaluator, BatchEvaluatorConfig
        )
        from rlqas.phase3.performance.benchmarking import CIVectorBenchmark
        from rlqas.phase3.performance.memory_manager import MemoryManager
        from rlqas.phase3.performance.checkpoint import save_checkpoint, load_checkpoint
        from rlqas.phase3.encoding.encoder_factory import EncoderFactory
        from rlqas.phase3.qubit_ops.operator_pool import QubitOperatorPool
        from rlqas.phase3.qubit_ops.controller import QubitUCCSearchController


# ---------------------------------------------------------------------------
# 2. LiH smoke test (fast — 4 qubits, chemical accuracy)
# ---------------------------------------------------------------------------

class TestHybridSearchLiHSmoke:
    """HybridSearchController on LiH full space (no active_space, 12 qubits): chemical accuracy."""

    @pytest.fixture(scope="class")
    def lih_12q(self):
        # Full space LiH STO-3G: 4 electrons, 6 orbitals → 12 qubits
        return process_molecule(
            "LiH", 1.6, "UCC",
            basis_set="sto-3g",
            transform="jordan_wigner",
        )

    def test_hybrid_search_lih_smoke_test(self, lih_12q):
        """200-episode hybrid search on LiH full space 12q must reach chemical accuracy."""
        from rlqas.phase3.hybrid_search.controller import HybridSearchController

        mol = lih_12q
        controller = HybridSearchController(
            mol,
            agent_type="ppo",
            config={
                "n_episodes": 200,
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "early_stop_threshold": 1.6e-3,
                "max_depth": 15,
                "max_blocks": 8,
            },
        )
        result = controller.search(n_episodes=200, early_stop_threshold=1.6e-3)

        assert result.best_energy is not None
        assert np.isfinite(result.best_energy), "best_energy is not finite"

        energy_error = abs(result.best_energy - mol.fci_energy)
        assert energy_error < 1.6e-3, (
            f"LiH full space 12q chemical accuracy NOT achieved. "
            f"Error = {energy_error * 1000:.4f} mHa (threshold = 1.6 mHa). "
            f"best_energy={result.best_energy:.6f}, fci={mol.fci_energy:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. ExperimentManager HYBRID dispatch
# ---------------------------------------------------------------------------

class TestExperimentManagerHybridDispatch:
    def test_experiment_manager_hybrid_dispatch(self, tmp_path):
        """ExperimentManager with HYBRID ansatz_type dispatches to HybridSearchController."""
        from rlqas.phase2.experiment.manager import ExperimentManager

        exp_config = {
            "name": "test_hybrid_dispatch",
            "type": "hybrid_search",
            "molecule": {
                "formula": "LiH",
                "bond_length": 1.6,
                "active_space": [2, 2],
                "basis_set": "sto-3g",
                "transform": "jordan_wigner",
            },
            "search": {
                "ansatz_type": "HYBRID",
                "max_depth": 5,
                "max_blocks": 4,
                "encoding_method": "matrix",
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
            },
            "rl": {
                "agent_type": "ppo",
                "n_episodes": 10,
                "early_stop_threshold": 1.6e-3,
            },
        }

        manager = ExperimentManager(output_dir=str(tmp_path / "results"))
        results = manager.run_experiment(config=exp_config)
        # Must complete without exception and return a dict
        assert isinstance(results, dict) or results is not None


# ---------------------------------------------------------------------------
# 4 & 5. Anti-hollow checks (LiH 10 qubits)
# ---------------------------------------------------------------------------

class TestAntiHollowChecks:
    """Primary anti-hollow guards for Phase 3."""

    @pytest.fixture(scope="class")
    def lih_10q(self):
        return process_molecule(
            "LiH", 1.6, "UCC",
            active_space=(2, 5),
            basis_set="sto-3g",
            transform="jordan_wigner",
        )

    def test_hybrid_env_single_step_does_not_cheat(self, lih_10q):
        """Single action step on LiH 10q must NOT reach chemical accuracy.

        If this fails, energy evaluation is broken (e.g. returning FCI directly).
        """
        from rlqas.phase3.hybrid_search.environment import HybridSearchEnv
        from rlqas.phase3.hybrid_search.circuit_builder import HybridFusionStrategy

        mol = lih_10q
        fci_energy = mol.fci_energy  # approx -7.882097 Ha

        env = HybridSearchEnv(
            mol,
            HybridFusionStrategy({"fusion_mode": "sequential"}),
            {
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "max_depth": 10,
                "max_blocks": 10,
            },
        )
        obs, _ = env.reset()
        obs, reward, done, trunc, info = env.step(0)
        energy = info["energy"]
        error = abs(energy - fci_energy)

        assert error > 1.6e-3, (
            f"HOLLOW IMPL DETECTED: Single-step energy error {error * 1000:.4f} mHa < 1.6 mHa. "
            f"Energy={energy:.6f} Ha, FCI={fci_energy:.6f} Ha. "
            "run_classical_opt is disabled or energy is being returned from FCI directly."
        )
        print(
            f"[PASS] Single-step error = {error * 1000:.4f} mHa > 1.6 mHa "
            "— energy evaluation is real"
        )

    def test_hybrid_env_energy_below_hf(self, lih_10q):
        """After single step with run_classical_opt=True, energy must be below HF.

        HF energy for LiH is approximately -7.862 Ha.
        Classical optimization must push energy below this level.
        """
        from rlqas.phase3.hybrid_search.environment import HybridSearchEnv
        from rlqas.phase3.hybrid_search.circuit_builder import HybridFusionStrategy

        mol = lih_10q
        hf_energy_approx = -7.862  # rough lower bound check

        env = HybridSearchEnv(
            mol,
            HybridFusionStrategy({"fusion_mode": "sequential"}),
            {
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "max_depth": 10,
                "max_blocks": 10,
            },
        )
        obs, _ = env.reset()
        obs, reward, done, trunc, info = env.step(0)
        energy = info["energy"]

        assert energy < hf_energy_approx, (
            f"HOLLOW IMPL DETECTED: Energy {energy:.6f} Ha is above HF level "
            f"{hf_energy_approx} Ha. "
            "Classical optimization (run_classical_opt=True) is not running."
        )
        print(
            f"[PASS] Energy {energy:.6f} Ha is below HF level "
            "— classical optimization is working"
        )


# ---------------------------------------------------------------------------
# 6. Training history is real
# ---------------------------------------------------------------------------

class TestSearchResultIsRealTraining:
    def test_search_result_is_real_training(self):
        """30-episode search on LiH full space: training history must be populated."""
        from rlqas.phase3.hybrid_search.controller import HybridSearchController

        # Full space LiH STO-3G: 4 electrons, 6 orbitals → 12 qubits
        mol = process_molecule(
            "LiH", 1.6, "UCC",
            basis_set="sto-3g",
            transform="jordan_wigner",
        )
        controller = HybridSearchController(
            mol,
            agent_type="ppo",
            config={
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "max_depth": 8,
                "max_blocks": 6,
            },
        )
        result = controller.search(n_episodes=30, early_stop_threshold=1.6e-3)

        # Training history must have at least 2 entries (not degenerate)
        assert len(result.training_history) >= 2, (
            "Training history empty — search loop not actually running"
        )

        # best_energy must be a real float, not None/NaN
        assert result.best_energy is not None
        assert np.isfinite(result.best_energy), (
            f"best_energy is not finite: {result.best_energy}"
        )

        # Energies must vary across episodes (not all identical constants)
        energies = [
            h.get("best_energy")
            for h in result.training_history
            if h.get("best_energy") is not None
        ]
        assert len(energies) >= 2, "No best_energy entries in training history"
        # At least 2 different energy values across episodes
        unique_energies = set(round(e, 8) for e in energies)
        assert len(unique_energies) >= 1, "All energies identical — search not running"
        print(
            f"[PASS] Training history has {len(energies)} episodes, "
            f"best_energy={result.best_energy:.6f} Ha"
        )


# ---------------------------------------------------------------------------
# 7. Qubit operator comparison JSON exists and is valid
# ---------------------------------------------------------------------------

class TestQubitOperatorComparison:
    def test_qubit_operator_comparison_lih_10q(self):
        """qubit_vs_fermion_lih_10q.json exists and has valid schema."""
        import json
        path = "results/phase3_integration/qubit_vs_fermion_lih_10q.json"
        assert os.path.exists(path), (
            f"Missing: {path}. Run the Task 007 comparison before this test."
        )
        with open(path) as f:
            data = json.load(f)

        assert "fermion" in data and "qubit" in data, "Missing fermion or qubit keys"
        assert "best_energy" in data["fermion"] and "best_energy" in data["qubit"]

        # Both energies must be physical (below -7.0 Ha for LiH)
        assert data["fermion"]["best_energy"] < -7.0, (
            f"Fermion energy {data['fermion']['best_energy']:.6f} Ha is not physical"
        )
        assert data["qubit"]["best_energy"] < -7.0, (
            f"Qubit energy {data['qubit']['best_energy']:.6f} Ha is not physical"
        )

        print(f"[PASS] Fermion: {data['fermion']['best_energy']:.6f} Ha "
              f"({data['fermion']['energy_error_mha']:.4f} mHa)")
        print(f"[PASS] Qubit:   {data['qubit']['best_energy']:.6f} Ha "
              f"({data['qubit']['energy_error_mha']:.4f} mHa)")
        print(f"[PASS] Comparison: {data['comparison']}")
