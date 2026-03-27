"""
DQNAgent: Deep Q-Network agent using Stable-Baselines3.

This module implements the DQNAgent class, which wraps the Stable-Baselines3
DQN implementation and conforms to the RLAgent abstract interface from Phase 1.

DQN is particularly well-suited for discrete action spaces in quantum
architecture search, where the agent selects excitation operators.
"""

import os
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

from rlqas_chem.rl.base_agent import RLAgent


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


class DQNConfig:
    """Configuration manager for DQN agent.

    This class handles configuration validation, defaults, and parameter
    management for DQN agents, following Phase 1 patterns.

    Attributes:
        config: The validated configuration dictionary.
    """

    # Default hyperparameters optimized for RLQAS quantum architecture search
    DEFAULT_CONFIG = {
        # Core DQN hyperparameters
        "learning_rate": 1e-3,
        "gamma": 0.99,
        # Epsilon-greedy exploration
        "exploration_fraction": 0.1,  # Fraction of training for epsilon decay
        "exploration_initial_eps": 1.0,  # Initial epsilon value
        "exploration_final_eps": 0.01,  # Final epsilon value after decay
        # Replay buffer
        "buffer_size": 10000,  # Maximum replay buffer size
        "batch_size": 64,  # Batch size for learning
        # Target network update (SB3 uses target_update_interval)
        "target_update_interval": 1000,  # Steps between target network updates
        # Training frequency
        "train_freq": 4,  # Steps between training updates
        "gradient_steps": 1,  # Number of gradient steps per training
        # Additional configuration
        "policy_type": "MlpPolicy",
        "verbose": 1,
        "seed": 42,
        "use_gpu": True,
        "tensorboard_log": None,
        # Environment configuration
        "n_envs": 1,
        "monitor_dir": None,
        "wrapper_class": None,
        # Learning starts after this many steps
        "learning_starts": 100,
        # Tau for soft target network updates
        "tau": 1.0,
        # Max grad norm
        "max_grad_norm": 10.0,
    }

    # Validation rules for parameters
    VALIDATION_RULES = {
        "learning_rate": (float, lambda x: x > 0, "must be positive float"),
        "gamma": (float, lambda x: 0 < x <= 1, "must be in (0, 1]"),
        "exploration_fraction": (float, lambda x: 0 <= x <= 1, "must be in [0, 1]"),
        "exploration_initial_eps": (float, lambda x: 0 <= x <= 1, "must be in [0, 1]"),
        "exploration_final_eps": (float, lambda x: 0 <= x <= 1, "must be in [0, 1]"),
        "buffer_size": (int, lambda x: x > 0, "must be positive integer"),
        "batch_size": (int, lambda x: x > 0, "must be positive integer"),
        "target_update_interval": (int, lambda x: x > 0, "must be positive integer"),
        "train_freq": (int, lambda x: x > 0, "must be positive integer"),
        "gradient_steps": (int, lambda x: x >= 0, "must be non-negative integer"),
        "policy_type": (str, lambda x: isinstance(x, str) and len(x) > 0,
                       "must be non-empty string"),
        "verbose": (int, lambda x: x in [0, 1, 2], "must be 0, 1, or 2"),
        "seed": (int, lambda x: True, "any integer allowed"),
        "use_gpu": (bool, lambda x: True, "boolean"),
        "tensorboard_log": (type(None), lambda x: True, "must be None or string"),
        "n_envs": (int, lambda x: x > 0, "must be positive integer"),
        "monitor_dir": (type(None), lambda x: True, "must be None or string"),
        "wrapper_class": (type(None), lambda x: True, "must be None or callable"),
        "learning_starts": (int, lambda x: x >= 0, "must be non-negative integer"),
        "tau": (float, lambda x: x > 0, "must be positive float"),
        "max_grad_norm": (float, lambda x: x > 0, "must be positive float"),
    }

    def __init__(self, user_config: Optional[Dict] = None):
        """Initialize configuration with user values merged with defaults.

        Args:
            user_config: User configuration dictionary (optional).
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if user_config:
            self._merge_and_validate(user_config)

    def _merge_and_validate(self, user_config: Dict):
        """Merge user configuration and validate all parameters.

        Args:
            user_config: User configuration dictionary.

        Raises:
            ValueError: If any parameter fails validation.
            KeyError: If user config contains unknown parameters.
        """
        # Check for unknown parameters
        for key in user_config:
            if key not in self.DEFAULT_CONFIG:
                raise KeyError(f"Unknown configuration parameter: {key}")

        # Merge
        self.config.update(user_config)

        # Validate each parameter
        for key, value in self.config.items():
            self._validate_param(key, value)

    def _validate_param(self, key: str, value: Any):
        """Validate a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value.

        Raises:
            ValueError: If validation fails.
            TypeError: If type is incorrect.
        """
        if key not in self.VALIDATION_RULES:
            return  # No validation rule for this parameter

        expected_type, validator, error_msg = self.VALIDATION_RULES[key]

        # Type check
        if not isinstance(value, expected_type):
            # Handle None type specially
            if expected_type is type(None) and value is None:
                pass
            else:
                raise TypeError(
                    f"Parameter {key} must be of type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        # Value validation
        if not validator(value):
            raise ValueError(f"Parameter {key} {error_msg}, got {value}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Parameter name.
            default: Default value if key not found.

        Returns:
            Parameter value or default.
        """
        return self.config.get(key, default)

    def update(self, updates: Dict):
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates.

        Raises:
            ValueError: If any update fails validation.
        """
        self._merge_and_validate(updates)

    def to_dict(self) -> Dict:
        """Get configuration as dictionary.

        Returns:
            Copy of configuration dictionary.
        """
        return self.config.copy()


class DQNAgent(RLAgent):
    """Deep Q-Network agent using Stable-Baselines3.

    This agent wraps the Stable-Baselines3 DQN implementation and provides
    the RLAgent interface from Phase 1. It supports automatic GPU detection,
    configuration management, and UCC compatibility helpers.

    DQN is particularly well-suited for:
    - Discrete action spaces (quantum excitation operator selection)
    - Environments with sparse rewards (quantum architecture search)
    - Problems requiring exploration-exploitation balance

    Args:
        config: Configuration dictionary. If None, defaults are used.
        env: Gym environment (or callable that returns an environment).
            If provided, the agent will be initialized with this environment.
            If None, environment must be provided before training via set_env()
            or passed to learn().
    """

    # Default hyperparameters
    DEFAULT_CONFIG = DQNConfig.DEFAULT_CONFIG

    def __init__(self, config: Optional[Dict] = None, env=None):
        """Initialize DQN agent with configuration."""
        super().__init__()
        # Use DQNConfig for validation and defaults
        self._config_obj = DQNConfig(config)
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
        """Initialize SB3 DQN model with environment."""
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

        # Create DQN model
        self.model = SB3_DQN(
            policy=self.config["policy_type"],
            env=env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            batch_size=self.config["batch_size"],
            gamma=self.config["gamma"],
            exploration_fraction=self.config["exploration_fraction"],
            exploration_initial_eps=self.config["exploration_initial_eps"],
            exploration_final_eps=self.config["exploration_final_eps"],
            target_update_interval=self.config["target_update_interval"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            learning_starts=self.config["learning_starts"],
            tau=self.config["tau"],
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
        action, _ = self.model.predict(state, deterministic=False)
        # SB3 returns numpy array, we need to convert to int for discrete action spaces
        action = int(action) if np.ndim(action) == 0 else int(action[0])

        # Get additional info
        # Note: SB3 DQN doesn't expose action probabilities directly
        info = {
            "action_probabilities": None,  # DQN doesn't provide this directly
            "value_estimate": None,  # Q-values could be extracted if needed
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
            # This would require manual DQN implementation or custom training loop
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
            "exploration_rate": self.model.exploration_rate if hasattr(self.model, 'exploration_rate') else None,
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
        os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)

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
        self.model = SB3_DQN.load(path, device=self._device)

        # Extract environment from loaded model
        self.env = self.model.env

        # Update config from model (approximate)
        # Note: SB3 doesn't expose all hyperparameters after loading
        self._config_obj.update({
            "policy_type": "MlpPolicy",  # Default for loaded models
            "learning_rate": self.model.learning_rate,
        })

    def get_config(self) -> Dict:
        """Get current configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
