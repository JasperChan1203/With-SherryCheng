#!/usr/bin/env python3
"""Unit tests for UCCRewardFunction."""

import sys
import unittest
import numpy as np

# Add Task directories to Python path (optional, not needed for reward function)
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.ucc_search.reward_function import UCCRewardFunction


class TestUCCRewardFunction(unittest.TestCase):
    """Test UCCRewardFunction class."""

    def test_default_config(self):
        """Test reward function with default configuration."""
        rf = UCCRewardFunction()
        # Default baseline_type should be "hartree_fock" per config defaults
        self.assertEqual(rf.baseline_type, "hartree_fock")
        self.assertEqual(rf.energy_weight, 1.0)
        self.assertEqual(rf.complexity_penalty, 0.01)
        self.assertFalse(rf.use_shaping)

    def test_compute_reward_first_evaluation(self):
        """Test first reward computation (baseline initialization)."""
        # Use current_best baseline for simpler test
        config = {"reward_function": {"baseline_type": "current_best"}}
        rf = UCCRewardFunction(config)
        # First call, should return 0.0 and set baseline
        reward = rf.compute_reward(-1.0, 1)
        self.assertEqual(reward, 0.0)
        # Baseline should now be set
        self.assertIsNotNone(rf.best_energy)
        self.assertEqual(rf.best_energy, -1.0)

    def test_compute_reward_energy_improvement(self):
        """Test reward for energy improvement."""
        config = {"reward_function": {"baseline_type": "current_best"}}
        rf = UCCRewardFunction(config)
        # First evaluation sets baseline
        rf.compute_reward(-1.0, 1)  # baseline = -1.0, reward 0.0
        # Second evaluation with better energy
        reward = rf.compute_reward(-1.1, 2)
        # Expected: improvement = (-1.0) - (-1.1) = 0.1
        # complexity penalty = 0.01 * 2 = 0.02
        # reward = 0.1 - 0.02 = 0.08
        self.assertAlmostEqual(reward, 0.08, places=6)
        # Best energy updated
        self.assertAlmostEqual(rf.best_energy, -1.1, places=6)

    def test_compute_reward_energy_worsening(self):
        """Test reward when energy worsens."""
        config = {"reward_function": {"baseline_type": "current_best"}}
        rf = UCCRewardFunction(config)
        rf.compute_reward(-1.1, 1)
        reward = rf.compute_reward(-1.0, 2)  # higher energy (worse)
        # improvement = (-1.1) - (-1.0) = -0.1
        # complexity penalty = 0.01 * 2 = 0.02
        # reward = -0.1 - 0.02 = -0.12
        self.assertAlmostEqual(reward, -0.12, places=6)
        # Best energy stays at -1.1
        self.assertAlmostEqual(rf.best_energy, -1.1, places=6)

    def test_baseline_type_hartree_fock(self):
        """Test hartree_fock baseline type."""
        config = {
            "reward_function": {
                "baseline_type": "hartree_fock",
                "energy_weight": 1.0,
                "complexity_penalty": 0.01,
            }
        }
        rf = UCCRewardFunction(config)
        # Set HF energy via update_baseline before first compute
        rf.update_baseline(-1.0)
        # First evaluation returns 0.0 (initialization)
        reward0 = rf.compute_reward(-1.0, 1)
        self.assertEqual(reward0, 0.0)
        # Compute reward relative to HF baseline (second evaluation)
        reward = rf.compute_reward(-1.1, 1)
        # improvement = (-1.0) - (-1.1) = 0.1
        # penalty = 0.01 * 1 = 0.01
        # reward = 0.1 - 0.01 = 0.09
        self.assertAlmostEqual(reward, 0.09, places=6)
        # HF baseline unchanged
        self.assertAlmostEqual(rf.hf_energy, -1.0, places=6)
        # Best energy also updated (backward compatibility)
        self.assertAlmostEqual(rf.best_energy, -1.0, places=6)

    def test_baseline_type_rolling_average(self):
        """Test rolling_average baseline type."""
        config = {
            "reward_function": {
                "baseline_type": "rolling_average",
                "rolling_window_size": 3,
            }
        }
        rf = UCCRewardFunction(config)
        # First evaluation initializes rolling average with current energy, reward 0.0
        reward1 = rf.compute_reward(-1.0, 1)
        self.assertEqual(reward1, 0.0)
        self.assertAlmostEqual(rf.rolling_energy, -1.0, places=6)
        # Second evaluation: baseline is still -1.0 (window size 1)
        reward2 = rf.compute_reward(-1.1, 1)
        # improvement = (-1.0) - (-1.1) = 0.1, penalty 0.01
        self.assertAlmostEqual(reward2, 0.09, places=6)
        # Rolling window now contains [-1.0, -1.1], average = -1.05
        self.assertAlmostEqual(rf.rolling_energy, -1.05, places=6)
        # Third evaluation: baseline = -1.05
        reward3 = rf.compute_reward(-1.2, 1)
        # improvement = (-1.05) - (-1.2) = 0.15, penalty 0.01
        self.assertAlmostEqual(reward3, 0.14, places=6)
        # Window now [-1.0, -1.1, -1.2], average = -1.1
        self.assertAlmostEqual(rf.rolling_energy, -1.1, places=6)
        # Fourth evaluation: window size limit 3, oldest drops
        reward4 = rf.compute_reward(-1.3, 1)
        # improvement = (-1.1) - (-1.3) = 0.2, penalty 0.01
        self.assertAlmostEqual(reward4, 0.19, places=6)
        # Window now [-1.1, -1.2, -1.3], average = -1.2
        self.assertAlmostEqual(rf.rolling_energy, -1.2, places=6)

    def test_energy_weight(self):
        """Test energy_weight parameter."""
        config = {
            "reward_function": {
                "baseline_type": "current_best",
                "energy_weight": 2.0,
                "complexity_penalty": 0.01,
            }
        }
        rf = UCCRewardFunction(config)
        rf.compute_reward(-1.0, 1)
        reward = rf.compute_reward(-1.1, 2)
        # improvement = 0.1, weighted = 0.2
        # penalty = 0.01 * 2 = 0.02
        # reward = 0.2 - 0.02 = 0.18
        self.assertAlmostEqual(reward, 0.18, places=6)

    def test_complexity_penalty(self):
        """Test complexity_penalty parameter."""
        config = {
            "reward_function": {
                "baseline_type": "current_best",
                "complexity_penalty": 0.05,
            }
        }
        rf = UCCRewardFunction(config)
        rf.compute_reward(-1.0, 1)
        reward = rf.compute_reward(-1.1, 5)  # circuit complexity 5
        # improvement = 0.1
        # penalty = 0.05 * 5 = 0.25
        # reward = 0.1 - 0.25 = -0.15
        self.assertAlmostEqual(reward, -0.15, places=6)

    def test_shaping_rewards(self):
        """Test shaping rewards when enabled."""
        config = {
            "reward_function": {
                "baseline_type": "current_best",
                "shaping_rewards": True,
            }
        }
        rf = UCCRewardFunction(config)
        # First evaluation (no shaping, reward 0.0)
        rf.compute_reward(-1.0, 1)
        # Second evaluation with improvement
        reward = rf.compute_reward(-1.01, 1)
        # improvement = 0.01, penalty = 0.01
        # shaping: consecutive improvements = 1 -> +0.01
        # reward = 0.01 - 0.01 + 0.01 = 0.01
        self.assertAlmostEqual(reward, 0.01, places=6)
        # Third improvement
        reward2 = rf.compute_reward(-1.02, 1)
        # improvement = 0.01, penalty = 0.01
        # shaping: consecutive improvements = 2 -> +0.02
        # reward = 0.01 - 0.01 + 0.02 = 0.02
        self.assertAlmostEqual(reward2, 0.02, places=6)
        # Fourth evaluation with worsening (energy increase)
        reward3 = rf.compute_reward(-1.01, 1)
        # improvement = -0.01, penalty = 0.01
        # shaping: consecutive improvements reset to 0, penalty for increase -0.005
        # reward = -0.01 - 0.01 - 0.005 = -0.025
        self.assertAlmostEqual(reward3, -0.025, places=6)

    def test_update_baseline(self):
        """Test update_baseline method."""
        rf = UCCRewardFunction()
        rf.update_baseline(-2.0)
        self.assertEqual(rf.best_energy, -2.0)
        # With hartree_fock baseline type
        config = {"reward_function": {"baseline_type": "hartree_fock"}}
        rf2 = UCCRewardFunction(config)
        rf2.update_baseline(-2.0)
        self.assertEqual(rf2.hf_energy, -2.0)
        self.assertEqual(rf2.best_energy, -2.0)  # backward compatibility

    def test_edge_cases(self):
        """Test edge cases: zero improvement, negative improvement."""
        config = {"reward_function": {"baseline_type": "current_best"}}
        rf = UCCRewardFunction(config)
        rf.compute_reward(-1.0, 1)
        # Zero improvement (same energy)
        reward = rf.compute_reward(-1.0, 2)
        # improvement = 0, penalty = 0.02, reward = -0.02
        self.assertAlmostEqual(reward, -0.02, places=6)
        # Negative improvement (worse energy)
        reward2 = rf.compute_reward(-0.9, 1)
        self.assertAlmostEqual(reward2, -0.11, places=6)


if __name__ == '__main__':
    unittest.main()