"""Configuration management for UCC search module.

Copied from Task 004, with minimal adaptations for integration.
"""

from typing import Dict, Any
import copy


class UCCSearchConfig:
    """Configuration for UCC search module.

    Attributes:
        environment: Environment configuration
        circuit_builder: Circuit builder configuration
        reward_function: Reward function configuration
        controller: Controller configuration
    """

    DEFAULT_CONFIG = {
        "environment": {
            "max_depth": 10,
            "max_excitations": 20,
            "use_sqeb": True,
            "param_init_strategy": "random",
            "observation_normalization": True,
            "action_masking": True,
        },
        "circuit_builder": {
            "excitation_types": ["single", "double"],
            "param_init_strategies": ["random", "zeros", "normal"],
            "default_param_strategy": "random",
            "param_init_range": (-0.1, 0.1),
        },
        "reward_function": {
            "energy_weight": 1.0,
            "complexity_penalty": 0.01,
            "baseline_type": "hartree_fock",  # or "current_best", "rolling_average"
            "shaping_rewards": False,
        },
        "controller": {
            "agent_type": "ppo",
            "n_episodes": 1000,
            "early_stop_threshold": 1.6e-3,
            "checkpoint_frequency": 100,
            "log_frequency": 10,
        },
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize configuration.

        Args:
            config: Configuration dictionary to override defaults.
        """
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config:
            self._update_config(config)

    def _update_config(self, config: Dict[str, Any]):
        """Update configuration recursively.

        Args:
            config: Configuration dictionary to merge.
        """
        for key, value in config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            section: Configuration section (e.g., "environment")
            key: Configuration key within section
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if section in self._config:
            return self._config[section].get(key, default)
        return default

    def set(self, section: str, key: str, value: Any):
        """Set configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Configuration section name

        Returns:
            Dictionary of configuration values for the section
        """
        return self._config.get(section, {})

    @property
    def config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return copy.deepcopy(self._config)

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid
        """
        # Basic validation
        env_config = self.get_section("environment")
        if env_config.get("max_depth", 0) < 0:
            raise ValueError("max_depth must be non-negative")
        if env_config.get("max_excitations", 0) < 0:
            raise ValueError("max_excitations must be non-negative")

        reward_config = self.get_section("reward_function")
        if reward_config.get("energy_weight", 0) < 0:
            raise ValueError("energy_weight must be non-negative")
        if reward_config.get("complexity_penalty", 0) < 0:
            raise ValueError("complexity_penalty must be non-negative")

        return True