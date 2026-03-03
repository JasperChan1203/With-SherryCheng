"""
PPOAgent: Proximal Policy Optimization agent using Stable-Baselines3.

This module implements the PPOAgent class, which wraps the Stable-Baselines3
PPO implementation and conforms to the RLAgent abstract interface.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gym

from .base_agent import RLAgent
from .config import AgentConfig


def get_device(use_gpu: bool = True) -> str:
    """Get appropriate device for PyTorch/SB3.

    Args:
        use_gpu: Whether to try using GPU

    Returns:
        Device string ("cuda" or "cpu")
    """
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PPOAgent(RLAgent):
    """Proximal Policy Optimization agent using Stable-Baselines3.

    This agent wraps the Stable-Baselines3 PPO implementation and provides
    the RLAgent interface. It supports automatic GPU detection, configuration
    management, and UCC compatibility helpers.

    Args:
        config: Configuration dictionary. If None, defaults are used.
        env: Gym environment (or callable that returns an environment).
            If provided, the agent will be initialized with this environment.
            If None, environment must be provided before training via set_env()
            or passed to learn().
    """

    # Default hyperparameters from RLQAS specification section 3.3
    DEFAULT_CONFIG = AgentConfig.DEFAULT_CONFIG

    def __init__(self, config: Optional[Dict] = None, env=None):
        """Initialize PPO agent with configuration."""
        super().__init__()
        # Use AgentConfig for validation and defaults
        self._config_obj = AgentConfig(config)
        self.env = env
        self.model = None
        self._device = get_device(self._config_obj.get("use_gpu"))

        # Set random seeds if specified
        seed = self._config_obj.get("seed")
        if seed is not None:
            self._set_seed(seed)

        # Initialize model if environment is provided
        if env is not None:
            self._init_model(env)


    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @property
    def config(self):
        """Get configuration as dictionary (for backward compatibility)."""
        return self._config_obj.to_dict()

    def _init_model(self, env):
        """Initialize SB3 PPO model with environment."""
        # Convert to vectorized environment if needed
        if not isinstance(env, DummyVecEnv):
            if callable(env):
                # env is a function that creates a Gym environment
                env = make_vec_env(
                    env,
                    n_envs=self.config["n_envs"],
                    seed=self.config["seed"],
                    monitor_dir=self.config["monitor_dir"],
                    wrapper_class=self.config["wrapper_class"],
                )
            else:
                # Single environment
                env = DummyVecEnv([lambda: env])

        # Create model
        self.model = SB3_PPO(
            policy=self.config["policy_type"],
            env=env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            tensorboard_log=self.config["tensorboard_log"],
            verbose=self.config["verbose"],
            device=self._device,
            seed=self.config["seed"],
        )
        self.env = env

    def set_env(self, env):
        """Set or change the environment for the agent.

        Args:
            env: Gym environment (or callable that returns an environment).
        """
        self.env = env
        # Note: SB3 doesn't allow changing environment after model creation
        # We need to create a new model
        self._init_model(env)

    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Select action given current state.

        Args:
            state: Current observation from environment.

        Returns:
            Tuple of (action, info_dict). The action is an integer,
            info_dict contains action probabilities and value function estimate.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call set_env() first.")

        # SB3 expects vectorized environments, so we need to add batch dimension
        # and handle the vectorized output
        action, state_ = self.model.predict(state, deterministic=False)
        # SB3 returns numpy array, we need to convert to int for discrete action spaces
        action = int(action) if action.size == 1 else action

        # Get additional info (probabilities, value estimate)
        # This is a simplified implementation; SB3 doesn't expose these directly
        # We'll return basic info for now
        info = {
            "action_probabilities": None,  # SB3 doesn't provide this directly
            "value_estimate": None,
            "state": state_,
        }
        return action, info

    def learn(self, experience: Optional[Dict] = None, total_timesteps: int = 10000) -> Dict:
        """Learn from experience batch or train on current environment.

        This method supports two modes:
        1. If experience dict is provided, learn from that batch (not implemented yet).
        2. If no experience dict, train on the current environment for given timesteps.

        Args:
            experience: Optional experience dictionary (not yet implemented).
            total_timesteps: Number of timesteps to train on environment.

        Returns:
            Dictionary containing learning metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call set_env() first.")

        if experience is not None:
            # TODO: Implement learning from experience batch
            raise NotImplementedError(
                "Learning from experience batch not yet implemented. "
                "Use environment training mode (pass total_timesteps)."
            )

        # Train on environment
        self.model.learn(total_timesteps=total_timesteps)

        # Extract training metrics (simplified)
        metrics = {
            "total_timesteps": total_timesteps,
            "learning_rate": self.model.learning_rate,
        }
        return metrics

    def save(self, path: str) -> None:
        """Save agent to disk.

        Args:
            path: File path to save agent (SB3 .zip format).
        """
        if self.model is None:
            raise RuntimeError("Cannot save uninitialized model.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save SB3 model
        self.model.save(path)

        # Note: Configuration and other state should also be saved
        # For simplicity, we rely on SB3's save mechanism which includes policy parameters
        # but not our custom config. In a production system we'd save config separately.

    def load(self, path: str) -> None:
        """Load agent from disk.

        Args:
            path: File path to load agent from (SB3 .zip format).
        """
        # Load SB3 model
        self.model = SB3_PPO.load(path)

        # Extract environment from loaded model
        self.env = self.model.env

        # Update config from model (approximate)
        # Note: SB3 doesn't expose all hyperparameters after loading

        # Map policy class name to SB3 policy type alias
        policy_class_to_type = {
            "ActorCriticPolicy": "MlpPolicy",
            "ActorCriticCnnPolicy": "CnnPolicy",
            "MultiInputActorCriticPolicy": "MultiInputPolicy",
        }
        policy_class_name = self.model.policy.__class__.__name__
        policy_type = policy_class_to_type.get(policy_class_name, policy_class_name)

        self._config_obj.update({
            "policy_type": policy_type,
            "learning_rate": self.model.learning_rate,
        })

    def get_config(self) -> Dict:
        """Get current configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()