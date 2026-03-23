"""Smoke tests verifying encoding_method integration in UCCSearchEnv and HEASearchEnv.

Task 005 acceptance: both environments accept encoding_method without error.
"""

import pytest
from rlqas.phase1.molecule.processor import process_molecule


@pytest.fixture(scope="module")
def h2_mol():
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


class TestUCCSearchEnvEncoding:
    """UCCSearchEnv accepts encoding_method config and uses EncoderFactory."""

    def test_default_matrix_encoding_unchanged(self, h2_mol):
        """Default (matrix) encoding does not change existing behavior."""
        from rlqas.phase1.search.environment import UCCSearchEnv
        env = UCCSearchEnv(h2_mol, {"run_classical_opt": True, "complexity_penalty": 0.0})
        assert env.encoding_method == "matrix"
        assert env._encoder is None  # default: use legacy one-hot arch
        obs, _ = env.reset()
        obs, r, done, trunc, info = env.step(0)
        assert obs.shape == env.observation_space.shape

    def test_sparse_encoding_runs_2_episodes(self, h2_mol):
        """UCCSearchEnv with encoding_method='sparse' runs 2 episodes without error."""
        from rlqas.phase1.search.environment import UCCSearchEnv
        env = UCCSearchEnv(
            h2_mol,
            {"encoding_method": "sparse", "run_classical_opt": True, "complexity_penalty": 0.0},
        )
        assert env.encoding_method == "sparse"
        assert hasattr(env, "circuit_enc_dim")

        for _ in range(2):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 5:
                obs, r, done, trunc, info = env.step(env.action_space.sample())
                assert obs.shape == env.observation_space.shape
                steps += 1

    def test_one_hot_encoding_runs_2_episodes(self, h2_mol):
        """UCCSearchEnv with encoding_method='one_hot' runs 2 episodes without error."""
        from rlqas.phase1.search.environment import UCCSearchEnv
        env = UCCSearchEnv(
            h2_mol,
            {"encoding_method": "one_hot", "run_classical_opt": True, "complexity_penalty": 0.0},
        )
        assert env.encoding_method == "one_hot"

        for _ in range(2):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 5:
                obs, r, done, trunc, info = env.step(env.action_space.sample())
                assert obs.shape == env.observation_space.shape
                steps += 1

    def test_obs_shape_consistent_across_steps(self, h2_mol):
        """Observation shape stays consistent across all steps in an episode."""
        from rlqas.phase1.search.environment import UCCSearchEnv
        env = UCCSearchEnv(
            h2_mol,
            {"encoding_method": "sparse", "run_classical_opt": True, "complexity_penalty": 0.0},
        )
        expected_shape = env.observation_space.shape
        obs, _ = env.reset()
        assert obs.shape == expected_shape
        for _ in range(3):
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            assert obs.shape == expected_shape
            if done:
                break


class TestHEASearchEnvEncoding:
    """HEASearchEnv accepts encoding_method (new-style constructor with mol + config dict)."""

    def test_new_style_constructor_one_hot(self, h2_mol):
        """HEASearchEnv(mol, config_dict) works with encoding_method='one_hot'."""
        from rlqas.phase2.hea_search.environment import HEASearchEnv
        env = HEASearchEnv(h2_mol, {"encoding_method": "one_hot", "max_layers": 3})
        assert env.encoding_method == "one_hot"
        assert env.n_qubits == h2_mol.n_qubits
        assert env.max_layers == 3
        obs, _ = env.reset()
        for _ in range(2):
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            if done:
                break

    def test_new_style_constructor_sparse(self, h2_mol):
        """HEASearchEnv(mol, config_dict) works with encoding_method='sparse'."""
        from rlqas.phase2.hea_search.environment import HEASearchEnv
        env = HEASearchEnv(h2_mol, {"encoding_method": "sparse", "max_layers": 2})
        assert env.encoding_method == "sparse"
        assert env.n_qubits == h2_mol.n_qubits
        obs, _ = env.reset()
        obs, r, done, trunc, info = env.step(0)
        assert obs.shape == env.observation_space.shape

    def test_legacy_constructor_unchanged(self):
        """Legacy HEASearchEnv(n_qubits=4) still works."""
        from rlqas.phase2.hea_search.environment import HEASearchEnv
        env = HEASearchEnv(n_qubits=4, max_layers=3)
        assert env.n_qubits == 4
        assert env.encoding_method == "matrix"
        obs, _ = env.reset()
        obs, r, done, trunc, info = env.step(0)
        assert obs.shape == env.observation_space.shape
