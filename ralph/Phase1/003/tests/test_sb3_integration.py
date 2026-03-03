"""
Integration tests for SB3 adapter and configuration.
"""

import os
import tempfile
import pytest
import gym
import numpy as np
from src.modules.rl_agents.config import AgentConfig, validate_config
from src.modules.rl_agents.sb3_adapter import (
    create_vectorized_env,
    set_sb3_seed,
    get_sb3_policy_class,
    load_sb3_model,
)


class TestAgentConfig:
    """Test AgentConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        defaults = config.to_dict()

        # Check core PPO hyperparameters from RLQAS spec
        assert defaults["learning_rate"] == 3e-4
        assert defaults["gamma"] == 0.99
        assert defaults["gae_lambda"] == 0.95
        assert defaults["clip_range"] == 0.2
        assert defaults["ent_coef"] == 0.01
        assert defaults["vf_coef"] == 0.5
        assert defaults["max_grad_norm"] == 0.5
        assert defaults["n_steps"] == 2048
        assert defaults["batch_size"] == 64
        assert defaults["n_epochs"] == 10

    def test_custom_config(self):
        """Test custom configuration merging."""
        custom = {"learning_rate": 1e-3, "gamma": 0.95}
        config = AgentConfig(custom)
        values = config.to_dict()

        assert values["learning_rate"] == 1e-3
        assert values["gamma"] == 0.95
        # Other values should remain default
        assert values["n_steps"] == 2048
        assert values["policy_type"] == "MlpPolicy"

    def test_config_validation_valid(self):
        """Test validation with valid configuration."""
        valid_config = {
            "learning_rate": 1e-4,
            "gamma": 0.9,
            "n_steps": 1024,
            "batch_size": 32,
            "seed": 12345,
        }
        # Should not raise
        config = AgentConfig(valid_config)
        assert config.get("learning_rate") == 1e-4

    def test_config_validation_invalid_type(self):
        """Test validation with invalid type."""
        invalid = {"learning_rate": "not_a_float"}
        with pytest.raises(TypeError):
            AgentConfig(invalid)

    def test_config_validation_invalid_value(self):
        """Test validation with invalid value."""
        invalid = {"gamma": 1.5}  # gamma > 1
        with pytest.raises(ValueError):
            AgentConfig(invalid)

    def test_config_validation_unknown_param(self):
        """Test validation with unknown parameter."""
        invalid = {"unknown_param": 123}
        with pytest.raises(KeyError):
            AgentConfig(invalid)

    def test_config_update(self):
        """Test updating configuration."""
        config = AgentConfig()
        config.update({"learning_rate": 2e-4})
        assert config.get("learning_rate") == 2e-4

    def test_config_update_invalid(self):
        """Test updating with invalid configuration."""
        config = AgentConfig()
        with pytest.raises(ValueError):
            config.update({"learning_rate": -0.1})  # negative learning rate

    def test_batch_size_gt_n_steps_validation(self):
        """Test cross-parameter validation: batch_size > n_steps."""
        invalid = {"n_steps": 256, "batch_size": 512}
        with pytest.raises(ValueError):
            AgentConfig(invalid)

    def test_validate_config_function(self):
        """Test validate_config utility function."""
        valid = {"learning_rate": 1e-3}
        assert validate_config(valid) is True

        invalid = {"learning_rate": -0.1}
        with pytest.raises(ValueError):
            validate_config(invalid)


class TestSB3Adapter:
    """Test SB3 adapter functions."""

    def test_create_vectorized_env_from_callable(self):
        """Test creating vectorized environment from callable."""
        env_fn = lambda: gym.make("CartPole-v1")
        vec_env = create_vectorized_env(env_fn, n_envs=2, seed=42)

        # Check it's a vectorized environment
        from stable_baselines3.common.vec_env import VecEnv
        assert isinstance(vec_env, VecEnv)
        assert vec_env.num_envs == 2

        # Test stepping
        obs = vec_env.reset()
        assert obs.shape == (2, 4)  # 2 environments, 4 observations each

        vec_env.close()

    def test_create_vectorized_env_from_single_env(self):
        """Test creating vectorized environment from single environment."""
        env = gym.make("CartPole-v1")
        vec_env = create_vectorized_env(env)

        from stable_baselines3.common.vec_env import DummyVecEnv
        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 1

        vec_env.close()
        env.close()

    def test_create_vectorized_env_already_vectorized(self):
        """Test passing already vectorized environment."""
        env_fn = lambda: gym.make("CartPole-v1")
        from stable_baselines3.common.vec_env import DummyVecEnv
        existing_vec_env = DummyVecEnv([env_fn])
        vec_env = create_vectorized_env(existing_vec_env)

        assert vec_env is existing_vec_env
        existing_vec_env.close()

    def test_set_sb3_seed(self):
        """Test setting random seeds."""
        # Should run without errors
        set_sb3_seed(42)
        set_sb3_seed(12345)

    def test_get_sb3_policy_class_valid(self):
        """Test getting valid SB3 policy classes."""
        from stable_baselines3.common.policies import (
            ActorCriticPolicy,
            ActorCriticCnnPolicy,
            MultiInputActorCriticPolicy,
        )

        assert get_sb3_policy_class("MlpPolicy") == ActorCriticPolicy
        assert get_sb3_policy_class("CnnPolicy") == ActorCriticCnnPolicy
        assert get_sb3_policy_class("MultiInputPolicy") == MultiInputActorCriticPolicy

    def test_get_sb3_policy_class_invalid(self):
        """Test getting invalid SB3 policy class."""
        with pytest.raises(ValueError):
            get_sb3_policy_class("InvalidPolicy")

    def test_load_sb3_model_invalid_path(self):
        """Test loading SB3 model with invalid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "nonexistent.zip")
            with pytest.raises(FileNotFoundError):
                load_sb3_model(invalid_path)

    @pytest.mark.skip(reason="Requires actual saved model; tested in PPOAgent tests")
    def test_load_sb3_model_valid(self):
        """Test loading valid SB3 model."""
        # This is tested in PPOAgent save/load tests
        pass