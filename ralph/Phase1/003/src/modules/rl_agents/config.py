"""
Configuration management for RL agents.

This module provides configuration handling, validation, and default values
for RL agents, following the RLQAS specification section 3.3.
"""

from typing import Dict, Any, Optional


class AgentConfig:
    """Configuration manager for RL agents.

    This class handles configuration validation, defaults, and parameter
    management for RL agents. It ensures all parameters are within valid
    ranges and provides sensible defaults from the RLQAS specification.

    Attributes:
        config: The validated configuration dictionary.
    """

    # Default hyperparameters from RLQAS specification section 3.3
    DEFAULT_CONFIG = {
        # Core PPO hyperparameters
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
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
    }

    # Validation rules for parameters
    VALIDATION_RULES = {
        "learning_rate": (float, lambda x: x > 0, "must be positive float"),
        "gamma": (float, lambda x: 0 < x <= 1, "must be in (0, 1]"),
        "gae_lambda": (float, lambda x: 0 <= x <= 1, "must be in [0, 1]"),
        "clip_range": (float, lambda x: 0 <= x <= 1, "must be in [0, 1]"),
        "ent_coef": (float, lambda x: x >= 0, "must be non-negative"),
        "vf_coef": (float, lambda x: x >= 0, "must be non-negative"),
        "max_grad_norm": (float, lambda x: x > 0, "must be positive"),
        "n_steps": (int, lambda x: x > 0, "must be positive integer"),
        "batch_size": (int, lambda x: x > 0, "must be positive integer"),
        "n_epochs": (int, lambda x: x > 0, "must be positive integer"),
        "policy_type": (str, lambda x: isinstance(x, str) and len(x) > 0,
                       "must be non-empty string"),
        "verbose": (int, lambda x: x in [0, 1, 2], "must be 0, 1, or 2"),
        "seed": (int, lambda x: True, "any integer allowed"),
        "use_gpu": (bool, lambda x: True, "boolean"),
        "tensorboard_log": (type(None), lambda x: True, "must be None or string"),
        "n_envs": (int, lambda x: x > 0, "must be positive integer"),
        "monitor_dir": (type(None), lambda x: True, "must be None or string"),
        "wrapper_class": (type(None), lambda x: True, "must be None or callable"),
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

        # Additional cross-parameter validation
        if self.config["batch_size"] > self.config["n_steps"]:
            raise ValueError(
                f"batch_size ({self.config['batch_size']}) must be <= "
                f"n_steps ({self.config['n_steps']})"
            )

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


def validate_config(config: Dict) -> bool:
    """Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If configuration is invalid.
    """
    try:
        AgentConfig(config)
        return True
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(f"Configuration validation failed: {e}")