"""
Configuration Loader for RLQAS Experiment Management.

This module provides configuration loading and validation utilities
for YAML and JSON configuration files.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Experiment configuration dataclass.

    Attributes:
        name: Experiment name
        type: Experiment type ("sequential_test", "hea_search", "custom")
        description: Optional description
        agents: List of agent configurations (for sequential_test)
        environment: Environment configuration
        training: Training configuration
        output_dir: Output directory for results
        verbose: Verbosity level
    """

    name: str
    type: str
    description: str = ""
    agents: List[Dict] = field(default_factory=list)
    environment: Dict = field(default_factory=dict)
    training: Dict = field(default_factory=dict)
    output_dir: str = "results/experiments"
    verbose: int = 1

    VALID_TYPES = ["sequential_test", "hea_search", "custom"]

    def __post_init__(self):
        """Validate configuration."""
        if self.type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid experiment type: {self.type}. "
                f"Must be one of {self.VALID_TYPES}"
            )

        if not self.name:
            raise ValueError("Experiment name cannot be empty")

        if self.verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1, or 2")


class ConfigLoader:
    """Configuration loader for experiment management.

    This class provides utilities for loading, validating, and parsing
    experiment configurations from YAML and JSON files.
    """

    def __init__(self):
        """Initialize configuration loader."""
        self._schema: Dict[str, Any] = self._get_default_schema()

    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default configuration schema.

        Returns:
            Schema dictionary
        """
        return {
            "required": ["name", "type"],
            "optional": [
                "description",
                "agents",
                "environment",
                "training",
                "hea",
                "agent",
                "output_dir",
                "verbose",
            ],
            "types": {
                "name": str,
                "type": str,
                "description": str,
                "agents": list,
                "environment": dict,
                "training": dict,
                "hea": dict,
                "agent": dict,
                "output_dir": str,
                "verbose": int,
            },
        }

    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        ext = os.path.splitext(config_path)[1].lower()

        with open(config_path, "r") as f:
            if ext in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif ext == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {ext}")

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        for field in self._schema["required"]:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check types
        for field, expected_type in self._schema["types"].items():
            if field in config:
                if not isinstance(config[field], expected_type):
                    errors.append(
                        f"Field '{field}' must be of type {expected_type.__name__}, "
                        f"got {type(config[field]).__name__}"
                    )

        # Check for unknown fields
        known_fields = set(self._schema["required"] + self._schema["optional"])
        for field in config:
            if field not in known_fields:
                errors.append(f"Unknown field: {field}")

        return errors

    def parse(self, config: Dict[str, Any]) -> ExperimentConfig:
        """Parse configuration into ExperimentConfig dataclass.

        Args:
            config: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        return ExperimentConfig(
            name=config.get("name", "unnamed"),
            type=config.get("type", "custom"),
            description=config.get("description", ""),
            agents=config.get("agents", []),
            environment=config.get("environment", {}),
            training=config.get("training", {}),
            output_dir=config.get("output_dir", "results/experiments"),
            verbose=config.get("verbose", 1),
        )

    def load_and_validate(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Validated configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        config = self.load(config_path)
        errors = self.validate(config)

        if errors:
            raise ValueError(
                f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return config

    def save(self, config: Dict[str, Any], output_path: str) -> str:
        """Save configuration to file.

        Args:
            config: Configuration dictionary
            output_path: Path to save configuration

        Returns:
            Path to saved file
        """
        ext = os.path.splitext(output_path)[1].lower()

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, "w") as f:
            if ext in [".yaml", ".yml"]:
                yaml.dump(config, f, default_flow_style=False)
            elif ext == ".json":
                json.dump(config, f, indent=2)
            else:
                # Default to YAML
                yaml.dump(config, f, default_flow_style=False)

        return output_path

    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge two configurations.

        Args:
            base_config: Base configuration
            override_config: Configuration to override with

        Returns:
            Merged configuration
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value

        return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_and_validate(config_path)


def save_config(config: Dict[str, Any], output_path: str) -> str:
    """Convenience function to save configuration.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration

    Returns:
        Path to saved file
    """
    loader = ConfigLoader()
    return loader.save(config, output_path)


def create_template_config(
    experiment_type: str = "sequential_test",
    name: str = "my_experiment",
) -> Dict[str, Any]:
    """Create a template configuration.

    Args:
        experiment_type: Type of experiment
        name: Experiment name

    Returns:
        Template configuration dictionary
    """
    if experiment_type == "sequential_test":
        return {
            "name": name,
            "type": "sequential_test",
            "description": "Sequential RL algorithm comparison",
            "agents": [
                {
                    "agent_type": "ppo",
                    "name": "ppo_agent",
                    "config": {
                        "learning_rate": 3e-4,
                        "verbose": 0,
                    },
                },
                {
                    "agent_type": "dqn",
                    "name": "dqn_agent",
                    "config": {
                        "learning_rate": 1e-3,
                        "verbose": 0,
                    },
                },
            ],
            "environment": {
                "type": "quantum",
                "n_qubits": 4,
            },
            "training": {
                "total_timesteps": 10000,
                "n_seeds": 3,
            },
            "output_dir": "results/sequential",
            "verbose": 1,
        }

    elif experiment_type == "hea_search":
        return {
            "name": name,
            "type": "hea_search",
            "description": "HEA architecture search",
            "hea": {
                "n_qubits": 4,
                "max_layers": 4,
                "entanglement_patterns": ["linear", "circular"],
                "rotation_gates": ["rx", "ry", "rz"],
                "parameter_sharing": "layer_wise",
            },
            "agent": {
                "type": "ppo",
                "config": {
                    "learning_rate": 3e-4,
                    "verbose": 0,
                },
            },
            "training": {
                "total_timesteps": 10000,
            },
            "output_dir": "results/hea",
            "verbose": 1,
        }

    else:
        return {
            "name": name,
            "type": "custom",
            "description": "Custom experiment",
        }
