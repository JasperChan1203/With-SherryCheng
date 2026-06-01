"""
HEA Search Configuration for RLQAS Phase 2.

This module provides configuration management for HEA search.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class HEAConfig:
    """Configuration for HEA search.

    This class manages all configuration parameters for HEA architecture
    search, including circuit parameters, RL agent settings, and training
    options.

    Attributes:
        n_qubits: Number of qubits in the HEA
        max_layers: Maximum number of layers
        entanglement_patterns: List of allowed entanglement patterns
        rotation_gates: List of allowed rotation gate types
        parameter_sharing: Parameter sharing strategy
        agent_type: RL agent type ("ppo" or "dqn")
        agent_config: RL agent configuration
        total_timesteps: Total training timesteps
        target_energy: Optional target energy
        output_dir: Output directory for results
        verbose: Verbosity level
    """

    # Circuit configuration
    n_qubits: int = 4
    max_layers: int = 4
    entanglement_patterns: List[str] = field(default_factory=lambda: ["linear", "circular", "full"])
    rotation_gates: List[str] = field(default_factory=lambda: ["rx", "ry", "rz"])
    parameter_sharing: str = "layer_wise"

    # RL agent configuration
    agent_type: str = "ppo"
    agent_config: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    total_timesteps: int = 10000
    n_episodes: int = 100
    target_energy: Optional[float] = None
    run_classical_opt: bool = True

    # Output configuration
    output_dir: str = "results/hea_search"
    verbose: int = 1

    # Validation rules
    VALID_ENTANGLEMENT_PATTERNS = ["linear", "circular", "full", "fully_connected"]
    VALID_ROTATION_GATES = ["rx", "ry", "rz"]
    VALID_PARAMETER_SHARING = ["none", "layer_wise", "global"]
    VALID_AGENT_TYPES = ["ppo", "dqn"]

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate all configuration parameters."""
        # Validate entanglement patterns
        for pattern in self.entanglement_patterns:
            if pattern not in self.VALID_ENTANGLEMENT_PATTERNS:
                raise ValueError(
                    f"Invalid entanglement pattern: {pattern}. "
                    f"Must be one of {self.VALID_ENTANGLEMENT_PATTERNS}"
                )

        # Validate rotation gates
        for gate in self.rotation_gates:
            if gate not in self.VALID_ROTATION_GATES:
                raise ValueError(
                    f"Invalid rotation gate: {gate}. "
                    f"Must be one of {self.VALID_ROTATION_GATES}"
                )

        # Validate parameter sharing
        if self.parameter_sharing not in self.VALID_PARAMETER_SHARING:
            raise ValueError(
                f"Invalid parameter sharing: {self.parameter_sharing}. "
                f"Must be one of {self.VALID_PARAMETER_SHARING}"
            )

        # Validate agent type
        if self.agent_type not in self.VALID_AGENT_TYPES:
            raise ValueError(
                f"Invalid agent type: {self.agent_type}. "
                f"Must be one of {self.VALID_AGENT_TYPES}"
            )

        # Validate numeric parameters
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.max_layers <= 0:
            raise ValueError("max_layers must be positive")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1, or 2")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "n_qubits": self.n_qubits,
            "max_layers": self.max_layers,
            "entanglement_patterns": self.entanglement_patterns,
            "rotation_gates": self.rotation_gates,
            "parameter_sharing": self.parameter_sharing,
            "agent_type": self.agent_type,
            "agent_config": self.agent_config,
            "total_timesteps": self.total_timesteps,
            "n_episodes": self.n_episodes,
            "target_energy": self.target_energy,
            "run_classical_opt": self.run_classical_opt,
            "output_dir": self.output_dir,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HEAConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            HEAConfig instance
        """
        return cls(**config_dict)

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates

        Raises:
            ValueError: If any update fails validation
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Unknown configuration parameter: {key}")

        # Re-validate after updates
        self._validate()

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration with defaults.

        Returns:
            Agent configuration dictionary
        """
        defaults = {
            "ppo": {"verbose": self.verbose, "n_steps": 128},
            "dqn": {"verbose": self.verbose, "buffer_size": 1000},
        }

        agent_defaults = defaults.get(self.agent_type, {})
        agent_defaults.update(self.agent_config)
        return agent_defaults


def get_default_config(
    n_qubits: int = 4,
    max_layers: int = 4,
    agent_type: str = "ppo",
) -> HEAConfig:
    """Get default HEA search configuration.

    Args:
        n_qubits: Number of qubits
        max_layers: Maximum number of layers
        agent_type: RL agent type

    Returns:
        HEAConfig instance with default values
    """
    return HEAConfig(
        n_qubits=n_qubits,
        max_layers=max_layers,
        agent_type=agent_type,
    )


def get_lih_config(agent_type: str = "ppo") -> HEAConfig:
    """Get configuration optimized for LiH molecule.

    Args:
        agent_type: RL agent type

    Returns:
        HEAConfig instance for LiH
    """
    return HEAConfig(
        n_qubits=10,  # LiH (2,5) with Jordan-Wigner
        max_layers=6,
        entanglement_patterns=["linear", "circular"],
        rotation_gates=["rx", "ry", "rz"],
        agent_type=agent_type,
        total_timesteps=20000,
        output_dir="results/hea_lih",
    )
