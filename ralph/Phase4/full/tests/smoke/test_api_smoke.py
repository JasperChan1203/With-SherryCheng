"""Smoke tests for rlqas top-level Python API (Task 001)."""
import json
import tempfile
import pytest
import rlqas


def test_imports():
    """All phases are importable after pip install -e ."""
    import rlqas.phase1
    import rlqas.phase2
    import rlqas.phase3


def test_search_returns_required_keys():
    result = rlqas.search("H2", 0.74, ansatz_type="UCC", agent_type="ppo", n_episodes=50)
    required_keys = [
        "best_energy", "fci_energy", "energy_error_mha",
        "chemical_accuracy", "n_operators", "fusion_template",
        "molecule", "ansatz_type", "agent_type", "n_episodes_run", "n_qubits",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_search_anti_hollow():
    """Anti-hollow: energy_error_mha must be > 0 and < 50 mHa."""
    result = rlqas.search("H2", 0.74, ansatz_type="UCC", agent_type="ppo", n_episodes=50)
    assert isinstance(result["energy_error_mha"], float)
    assert result["energy_error_mha"] > 0, "energy_error_mha == 0: delegation is broken"
    assert result["energy_error_mha"] < 50, f"unreasonably large: {result['energy_error_mha']}"
    assert result["chemical_accuracy"] in (True, False)
    assert result["n_qubits"] == 4  # H2 with active_space=(1,2)
    print(f"[PASS] H2 UCC: {result['energy_error_mha']:.3f} mHa | {result['n_qubits']} qubits")


def test_search_dispatches_ucc():
    result = rlqas.search("H2", 0.74, ansatz_type="UCC", agent_type="ppo", n_episodes=20)
    assert result["ansatz_type"] == "UCC"


def test_search_invalid_ansatz_raises():
    with pytest.raises(ValueError, match="Invalid ansatz_type"):
        rlqas.search("H2", 0.74, ansatz_type="INVALID")


def test_search_invalid_agent_raises():
    with pytest.raises(ValueError, match="Invalid agent_type"):
        rlqas.search("H2", 0.74, agent_type="INVALID")


def test_experiment_run():
    exp = rlqas.Experiment(
        molecule_config={"formula": "H2", "bond_length": 0.74},
        search_config={"ansatz_type": "UCC"},
        rl_config={"agent_type": "ppo", "n_episodes": 20},
    )
    result = exp.run()
    assert "energy_error_mha" in result
    assert result["molecule"] == "H2"


def test_experiment_save_load():
    exp = rlqas.Experiment(
        molecule_config={"formula": "H2", "bond_length": 0.74},
        search_config={"ansatz_type": "UCC"},
        rl_config={"agent_type": "ppo", "n_episodes": 20},
    )
    exp.run()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    exp.save(path)
    loaded = exp.load(path)
    assert loaded["molecule"] == "H2"
    assert "energy_error_mha" in loaded


def test_experiment_save_before_run_raises():
    exp = rlqas.Experiment(
        molecule_config={"formula": "H2", "bond_length": 0.74},
        search_config={},
        rl_config={},
    )
    with pytest.raises(RuntimeError, match="Call run\\(\\) before save"):
        exp.save("/tmp/test_never_created.json")
