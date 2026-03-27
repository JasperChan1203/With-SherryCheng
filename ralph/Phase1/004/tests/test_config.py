#!/usr/bin/env python3
"""Unit tests for UCCSearchConfig."""

import sys
import unittest
import numpy as np

sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.ucc_search.config import UCCSearchConfig


class TestUCCSearchConfig(unittest.TestCase):
    """Test UCCSearchConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UCCSearchConfig()
        defaults = config.DEFAULT_CONFIG

        # Check environment defaults
        env_config = config.get_section("environment")
        self.assertEqual(env_config.get("max_depth"), 10)
        self.assertEqual(env_config.get("max_excitations"), 20)
        self.assertTrue(env_config.get("use_sqeb"))
        self.assertEqual(env_config.get("param_init_strategy"), "random")
        self.assertTrue(env_config.get("observation_normalization"))

        # Check reward defaults
        reward_config = config.get_section("reward_function")
        self.assertEqual(reward_config.get("energy_weight"), 1.0)
        self.assertEqual(reward_config.get("complexity_penalty"), 0.01)
        self.assertEqual(reward_config.get("baseline_type"), "hartree_fock")
        self.assertFalse(reward_config.get("shaping_rewards"))

        # Check controller defaults
        controller_config = config.get_section("controller")
        self.assertEqual(controller_config.get("agent_type"), "ppo")
        self.assertEqual(controller_config.get("n_episodes"), 1000)
        self.assertAlmostEqual(controller_config.get("early_stop_threshold"), 1.6e-3)

    def test_custom_config(self):
        """Test configuration with custom values."""
        custom_config = {
            "environment": {
                "max_depth": 5,
                "max_excitations": 10,
                "use_sqeb": False,
            },
            "reward_function": {
                "energy_weight": 2.0,
                "complexity_penalty": 0.05,
                "baseline_type": "current_best",
            },
            "controller": {
                "n_episodes": 500,
                "early_stop_threshold": 1e-3,
            }
        }

        config = UCCSearchConfig(custom_config)

        # Check overridden values
        env_config = config.get_section("environment")
        self.assertEqual(env_config.get("max_depth"), 5)
        self.assertEqual(env_config.get("max_excitations"), 10)
        self.assertFalse(env_config.get("use_sqeb"))
        # Check that other defaults are preserved
        self.assertEqual(env_config.get("param_init_strategy"), "random")

        reward_config = config.get_section("reward_function")
        self.assertEqual(reward_config.get("energy_weight"), 2.0)
        self.assertEqual(reward_config.get("complexity_penalty"), 0.05)
        self.assertEqual(reward_config.get("baseline_type"), "current_best")

        controller_config = config.get_section("controller")
        self.assertEqual(controller_config.get("n_episodes"), 500)
        self.assertAlmostEqual(controller_config.get("early_stop_threshold"), 1e-3)

    def test_get_method(self):
        """Test get method for configuration values."""
        config = UCCSearchConfig()

        # Get existing values
        max_depth = config.get("environment", "max_depth")
        self.assertEqual(max_depth, 10)

        # Get non-existent value with default
        non_existent = config.get("environment", "non_existent", "default_value")
        self.assertEqual(non_existent, "default_value")

        # Get from non-existent section
        non_existent_section = config.get("non_existent_section", "key", "default")
        self.assertEqual(non_existent_section, "default")

    def test_set_method(self):
        """Test set method for configuration values."""
        config = UCCSearchConfig()

        # Set existing value
        config.set("environment", "max_depth", 15)
        self.assertEqual(config.get("environment", "max_depth"), 15)

        # Set new value in existing section
        config.set("environment", "new_param", "value")
        self.assertEqual(config.get("environment", "new_param"), "value")

        # Set value in new section
        config.set("new_section", "key", "value")
        self.assertEqual(config.get("new_section", "key"), "value")

    def test_get_section(self):
        """Test get_section method."""
        config = UCCSearchConfig()

        env_section = config.get_section("environment")
        self.assertIsInstance(env_section, dict)
        self.assertIn("max_depth", env_section)
        self.assertIn("max_excitations", env_section)

        # Non-existent section returns empty dict
        non_existent = config.get_section("non_existent")
        self.assertEqual(non_existent, {})

    def test_validation(self):
        """Test configuration validation."""
        config = UCCSearchConfig()
        # Default config should be valid
        self.assertTrue(config.validate())

        # Invalid config (negative max_depth)
        invalid_config = {
            "environment": {
                "max_depth": -1,
                "max_excitations": 20,
            }
        }
        config2 = UCCSearchConfig(invalid_config)
        with self.assertRaises(ValueError):
            config2.validate()

        # Invalid config (negative energy_weight)
        invalid_config2 = {
            "reward_function": {
                "energy_weight": -1.0,
                "complexity_penalty": 0.01,
            }
        }
        config3 = UCCSearchConfig(invalid_config2)
        with self.assertRaises(ValueError):
            config3.validate()

        # Invalid config (negative complexity_penalty)
        invalid_config3 = {
            "reward_function": {
                "energy_weight": 1.0,
                "complexity_penalty": -0.01,
            }
        }
        config4 = UCCSearchConfig(invalid_config3)
        with self.assertRaises(ValueError):
            config4.validate()

    def test_config_property(self):
        """Test config property returns deep copy."""
        config = UCCSearchConfig()
        config_dict = config.config

        # Should be a dictionary
        self.assertIsInstance(config_dict, dict)

        # Modifying returned dict should not affect original
        config_dict["environment"]["max_depth"] = 999
        self.assertNotEqual(config.get("environment", "max_depth"), 999)
        self.assertEqual(config.get("environment", "max_depth"), 10)

    def test_partial_update(self):
        """Test updating only part of configuration."""
        config = UCCSearchConfig()
        # Update only environment section
        update_config = {
            "environment": {
                "max_depth": 15,
            }
        }
        config2 = UCCSearchConfig(update_config)

        # Check updated value
        self.assertEqual(config2.get("environment", "max_depth"), 15)
        # Check that other values remain default
        self.assertEqual(config2.get("environment", "max_excitations"), 20)
        self.assertEqual(config2.get("reward_function", "energy_weight"), 1.0)


if __name__ == '__main__':
    unittest.main()