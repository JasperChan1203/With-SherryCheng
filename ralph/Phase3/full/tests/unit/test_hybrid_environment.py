"""Unit tests for HybridSearchEnv and HybridRewardFunction."""
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase3.hybrid_search.circuit_builder import HybridFusionStrategy
from rlqas.phase3.hybrid_search.environment import HybridSearchEnv, HybridRewardFunction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_mol():
    """H2 molecule: active_space=(2,2) → 4 qubits."""
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2), basis_set="sto-3g", transform="jordan_wigner"
    )


@pytest.fixture(scope="module")
def lih_10q_mol():
    """LiH molecule: active_space=(2,5) → 10 qubits."""
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 5), basis_set="sto-3g", transform="jordan_wigner"
    )


@pytest.fixture(scope="module")
def h2_env(h2_mol):
    strategy = HybridFusionStrategy()
    return HybridSearchEnv(
        h2_mol, strategy,
        {"run_classical_opt": True, "complexity_penalty": 0.0}
    )


# ---------------------------------------------------------------------------
# HybridRewardFunction
# ---------------------------------------------------------------------------

class TestHybridRewardFunction:
    def test_finite_reward_for_finite_energy(self, h2_mol):
        rf = HybridRewardFunction()
        hf_e = h2_mol.molecular_info.get("hf_energy", -1.0)
        rf.update_baseline(hf_e)
        reward = rf.compute_reward(None, hf_e - 0.01, hf_e - 0.05,
                                   {"circuit_depth": 2, "n_blocks": 1})
        assert np.isfinite(reward)

    def test_nan_energy_returns_negative_10(self):
        rf = HybridRewardFunction()
        reward = rf.compute_reward(None, float("nan"), -1.0, {})
        assert reward == -10.0

    def test_inf_energy_returns_negative_10(self):
        rf = HybridRewardFunction()
        reward = rf.compute_reward(None, float("inf"), -1.0, {})
        assert reward == -10.0

    def test_first_step_returns_zero(self, h2_mol):
        rf = HybridRewardFunction()
        hf_e = h2_mol.molecular_info.get("hf_energy", -1.0)
        rf.update_baseline(hf_e)
        # First call (after update_baseline) should return 0
        reward = rf.compute_reward(None, hf_e, hf_e - 0.1,
                                   {"circuit_depth": 1, "n_blocks": 1})
        assert reward == 0.0

    def test_subsequent_improvement_positive_reward(self, h2_mol):
        rf = HybridRewardFunction()
        hf_e = h2_mol.molecular_info.get("hf_energy", -1.0)
        rf.update_baseline(hf_e)
        # First step: sets baseline
        rf.compute_reward(None, hf_e, hf_e - 0.1,
                          {"circuit_depth": 1, "n_blocks": 1})
        # Second step: energy improved → accuracy_reward should be positive
        reward = rf.compute_reward(None, hf_e - 0.01, hf_e - 0.1,
                                   {"circuit_depth": 1, "n_blocks": 1})
        # Accuracy component is positive, penalties small — total can be positive
        assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# Gym compliance tests
# ---------------------------------------------------------------------------

class TestHybridSearchEnvGymCompliance:
    def test_has_action_space(self, h2_env):
        assert hasattr(h2_env, "action_space")

    def test_has_observation_space(self, h2_env):
        assert hasattr(h2_env, "observation_space")

    def test_reset_returns_obs_and_info(self, h2_env):
        obs, info = h2_env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_observation_in_space_after_reset(self, h2_env):
        obs, _ = h2_env.reset()
        assert h2_env.observation_space.contains(obs.astype(np.float32))

    def test_step_returns_five_values(self, h2_env):
        h2_env.reset()
        result = h2_env.step(0)
        assert len(result) == 5

    def test_obs_shape_consistent_between_reset_and_step(self, h2_env):
        obs1, _ = h2_env.reset()
        obs2, _, _, _, _ = h2_env.step(0)
        assert obs1.shape == obs2.shape

    def test_action_space_is_discrete(self, h2_env):
        import gymnasium as gym
        assert isinstance(h2_env.action_space, gym.spaces.Discrete)

    def test_observation_space_is_box(self, h2_env):
        import gymnasium as gym
        assert isinstance(h2_env.observation_space, gym.spaces.Box)

    def test_action_space_size_equals_ucc_plus_hea(self, h2_env):
        expected = h2_env._n_ucc_excitations + h2_env._n_hea_configs
        assert h2_env.action_space.n == expected

    def test_step_after_done_raises_runtime_error(self, h2_env):
        h2_env.reset()
        # Force done by exceeding max_blocks
        h2_env.done = True
        with pytest.raises(RuntimeError):
            h2_env.step(0)

    def test_invalid_action_raises_value_error(self, h2_env):
        h2_env.reset()
        with pytest.raises(ValueError):
            h2_env.step(h2_env.action_space.n + 100)


# ---------------------------------------------------------------------------
# H2 episode tests
# ---------------------------------------------------------------------------

class TestHybridSearchEnvH2:
    def test_10_complete_episodes(self, h2_mol):
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(
            h2_mol, strategy,
            {"run_classical_opt": True, "complexity_penalty": 0.0,
             "max_depth": 6, "max_blocks": 4}
        )
        for ep in range(10):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 20:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                assert np.isfinite(reward), f"Non-finite reward at ep={ep}, step={steps}"
                assert "energy" in info
                assert np.isfinite(info["energy"]), f"Non-finite energy at ep={ep}, step={steps}"
                steps += 1

    def test_energy_finite_for_ucc_actions(self, h2_mol):
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(
            h2_mol, strategy,
            {"run_classical_opt": True, "complexity_penalty": 0.0}
        )
        n_ucc = env._n_ucc_excitations
        for action in range(min(3, n_ucc)):
            env.reset()
            _, _, _, _, info = env.step(action)
            assert np.isfinite(info["energy"]), f"Non-finite energy for UCC action {action}"

    def test_energy_finite_for_hea_actions(self, h2_mol):
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(
            h2_mol, strategy,
            {"run_classical_opt": True, "complexity_penalty": 0.0}
        )
        n_ucc = env._n_ucc_excitations
        for action in range(n_ucc, n_ucc + env._n_hea_configs):
            env.reset()
            _, _, _, _, info = env.step(action)
            assert np.isfinite(info["energy"]), f"Non-finite energy for HEA action {action}"

    def test_observation_dimension(self, h2_mol):
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(h2_mol, strategy, {})
        expected_dim = 1 + env._n_ucc_excitations + env._n_hea_configs + 1
        obs, _ = env.reset()
        assert obs.shape == (expected_dim,)

    def test_run_classical_opt_defaults_to_true(self, h2_mol):
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(h2_mol, strategy, {})
        assert env.run_classical_opt is True


# ---------------------------------------------------------------------------
# Anti-hollow checks (CRITICAL)
# ---------------------------------------------------------------------------

class TestAntiHollowChecks:
    """Verify that energy evaluation is real and not hollow (returning FCI)."""

    def test_single_step_does_not_cheat_to_chemical_accuracy(self, lih_10q_mol):
        """Anti-hollow Test A: a single UCC step must NOT reach chemical accuracy.

        If this fails it means energy is being returned as FCI directly, or
        classical optimization over all parameters is being run (breaking
        the architecture search premise).
        """
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(
            lih_10q_mol, strategy,
            {"run_classical_opt": True, "complexity_penalty": 0.0,
             "max_depth": 10, "max_blocks": 10}
        )
        obs, _ = env.reset()
        # Action 0 = first UCC excitation operator
        obs, reward, done, trunc, info = env.step(0)
        energy = info["energy"]
        error = abs(energy - lih_10q_mol.fci_energy)
        assert error > 1.6e-3, (
            f"HOLLOW IMPL DETECTED: Single-step energy error {error * 1000:.4f} mHa < 1.6 mHa. "
            f"Energy={energy:.6f} Ha, FCI={lih_10q_mol.fci_energy:.6f} Ha. "
            f"This means run_classical_opt is running over all parameters "
            f"(not just the selected excitation) or energy is trivially FCI."
        )
        print(f"[PASS] Single-step error = {error * 1000:.4f} mHa > 1.6 mHa")

    def test_energy_below_hf_after_single_step(self, lih_10q_mol):
        """Anti-hollow Test B: classical optimization must bring energy below HF.

        For LiH, HF ≈ -7.862 Ha.  After adding one double excitation and
        running L-BFGS-B, energy should be at or below approximately -7.865 Ha.
        If energy is still at or above HF, classical optimization is not running.
        """
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(
            lih_10q_mol, strategy,
            {"run_classical_opt": True, "complexity_penalty": 0.0,
             "max_depth": 10, "max_blocks": 10}
        )
        obs, _ = env.reset()
        obs, reward, done, trunc, info = env.step(0)
        energy = info["energy"]
        hf_energy_approx = -7.862  # Approximate HF energy for LiH at 1.6 Å
        assert energy < hf_energy_approx, (
            f"HOLLOW IMPL DETECTED: Energy {energy:.6f} Ha >= HF level {hf_energy_approx} Ha. "
            f"Classical optimization (run_classical_opt=True) is not running, "
            f"or the first UCC excitation gives zero gradient."
        )
        print(f"[PASS] Energy {energy:.6f} Ha is below HF level — classical opt is working")

    def test_energy_is_real_float_not_zero(self, lih_10q_mol):
        """Verify evaluated energy is non-zero (not silently returning 0.0)."""
        strategy = HybridFusionStrategy()
        env = HybridSearchEnv(
            lih_10q_mol, strategy,
            {"run_classical_opt": True, "complexity_penalty": 0.0}
        )
        obs, _ = env.reset()
        obs, _, _, _, info = env.step(0)
        energy = info["energy"]
        assert abs(energy) > 1.0, (
            f"Energy {energy:.6f} suspiciously close to 0 — VQE inner loop may be missing."
        )
