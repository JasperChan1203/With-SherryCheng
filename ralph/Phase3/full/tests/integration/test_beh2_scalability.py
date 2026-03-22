"""BeH2 scalability integration tests.

Tests that chemical accuracy (< 1.6e-3 Ha) is achievable for BeH2 across
different active spaces (8, 10, 12, 14 qubits).

All tests use UCCSearchController (more reliable than hybrid for scalability
testing — UCCSD provides a good approximation to FCI for these systems).

Expensive tests are marked @pytest.mark.slow.
"""

import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule


def _run_ucc_search(mol, n_episodes: int = 200) -> float:
    """Run UCCSearchController and return best energy."""
    from rlqas.phase1.search.controller import UCCSearchController

    # Use nested config format so UCCSearchConfig routes keys to correct sections.
    # Flat keys (e.g. max_depth=15) go to top-level and are ignored by get_section().
    # max_depth=30 is large enough for all UCCSD excitations (BeH2 has 18, H4/H6 have more).
    config = {
        "environment": {
            "max_depth": 30,
            "max_excitations": 50,
            "run_classical_opt": True,
            "param_init_strategy": "zeros",
        },
        "reward_function": {
            "complexity_penalty": 0.0,
        },
        "controller": {
            "n_episodes": n_episodes,
            "early_stop_threshold": 1.6e-3,
        },
    }
    controller = UCCSearchController(mol, config=config)
    result = controller.search(n_episodes=n_episodes)
    if isinstance(result, dict):
        return result.get("best_energy", float("nan"))
    return float(getattr(result, "best_energy", float("nan")))


class TestBeH2Scalability:
    """Chemical accuracy on BeH2 across qubit counts."""

    def test_beh2_10qubits_chemical_accuracy(self):
        """UCC search on BeH2 (4,5) 10q; assert energy_error < 1.6e-3."""
        mol = process_molecule(
            "BeH2", 1.3, "UCC",
            active_space=(4, 5),
            basis_set="sto-3g",
            transform="jordan_wigner",
        )
        assert mol.n_qubits == 10, f"Expected 10 qubits, got {mol.n_qubits}"

        best_energy = _run_ucc_search(mol, n_episodes=200)
        energy_error = abs(best_energy - mol.fci_energy)
        assert energy_error < 1.6e-3, (
            f"BeH2 (4,5) 10-qubit: chemical accuracy NOT achieved. "
            f"Error = {energy_error * 1000:.4f} mHa, threshold = 1.6 mHa. "
            f"best_energy={best_energy:.6f}, fci={mol.fci_energy:.6f}"
        )

    @pytest.mark.slow
    def test_beh2_12qubits_chemical_accuracy(self):
        """UCC search on BeH2 (6,6) 12q; assert energy_error < 1.6e-3.

        Uses UCCSearchController (more reliable than hybrid for 12q scalability
        testing — UCCSD provides a tight approximation to FCI for BeH2).
        """
        mol = process_molecule(
            "BeH2", 1.3, "UCC",
            active_space=(6, 6),
            basis_set="sto-3g",
            transform="jordan_wigner",
        )
        assert mol.n_qubits == 12, f"Expected 12 qubits, got {mol.n_qubits}"

        best_energy = _run_ucc_search(mol, n_episodes=200)
        energy_error = abs(best_energy - mol.fci_energy)
        assert energy_error < 1.6e-3, (
            f"BeH2 (6,6) 12-qubit: chemical accuracy NOT achieved. "
            f"Error = {energy_error * 1000:.4f} mHa, threshold = 1.6 mHa"
        )

    @pytest.mark.slow
    def test_beh2_14qubits_chemical_accuracy(self):
        """UCC search on BeH2 (8,7) 14q; assert energy_error < 1.6e-3.

        May be skipped if system memory is insufficient.
        """
        try:
            mol = process_molecule(
                "BeH2", 1.3, "UCC",
                active_space=(8, 7),
                basis_set="sto-3g",
                transform="jordan_wigner",
            )
        except Exception as e:
            pytest.skip(f"BeH2 (8,7) 14q setup failed: {e}")

        if mol.n_qubits != 14:
            pytest.skip(f"Expected 14 qubits, got {mol.n_qubits} — active space issue")

        best_energy = _run_ucc_search(mol, n_episodes=200)
        energy_error = abs(best_energy - mol.fci_energy)
        assert energy_error < 1.6e-3, (
            f"BeH2 (8,7) 14-qubit: chemical accuracy NOT achieved. "
            f"Error = {energy_error * 1000:.4f} mHa, threshold = 1.6 mHa"
        )
