"""Tests for rlqas_chem.search() top-level API."""
import pytest
import rlqas_chem


def test_search_returns_required_keys():
    r = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type='ppo', n_episodes=20)
    assert 'best_energy' in r
    assert 'energy_error_mha' in r
    assert 'chemical_accuracy' in r
    assert 'fci_energy' in r
    assert 'n_qubits' in r


def test_h2_energy_below_minus_one():
    r = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type='ppo', n_episodes=20)
    assert r['best_energy'] < -1.0, f"H2 energy should be < -1.0, got {r['best_energy']}"


def test_version():
    assert rlqas_chem.__version__ == '1.0.0'


def test_experiment_class():
    exp = rlqas_chem.Experiment(
        molecule_config={'formula': 'H2', 'bond_length': 0.74},
        search_config={'ansatz_type': 'UCC'},
        rl_config={'agent_type': 'ppo', 'n_episodes': 20},
    )
    r = exp.run()
    assert 'best_energy' in r
