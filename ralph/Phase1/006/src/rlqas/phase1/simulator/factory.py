"""
Simulator factory for RLQAS.

Factory for creating appropriate quantum simulators based on system scale.
"""

from typing import Optional, Dict
from .base import QuantumSimulator
from .tencirchem import TencirchemCISimulator


class SimulatorFactory:
    """Factory for creating appropriate quantum simulators based on system scale."""

    @staticmethod
    def create_simulator(
        n_qubits: int,
        config: Optional[Dict] = None
    ) -> QuantumSimulator:
        """Create appropriate simulator based on qubit count and configuration.

        Decision logic:
        - n_qubits < 8: statevector simulator (fast, memory permitting)
        - 8 ≤ n_qubits ≤ 20: CI vector simulator (default, memory efficient)
        - n_qubits > 20: CI vector with memory check, fallback to approximate methods
        - Config parameter overrides: user can specify engine directly

        Args:
            n_qubits: Number of qubits in the system
            config: Configuration dictionary (optional)

        Returns:
            QuantumSimulator instance
        """
        if config is None:
            config = {}

        # Check if engine is explicitly specified
        if "engine" in config:
            # Use specified engine
            simulator_config = config.copy()
        else:
            # Apply default decision logic
            simulator_config = config.copy()
            if n_qubits < 8:
                simulator_config["engine"] = "statevector"
            elif n_qubits <= 20:
                simulator_config["engine"] = "ci_vector"
            else:
                simulator_config["engine"] = "ci_vector"
                # For large systems, enable conservative memory settings
                if "max_memory_gb" not in simulator_config:
                    simulator_config["max_memory_gb"] = 16.0
                if "fallback_method" not in simulator_config:
                    simulator_config["fallback_method"] = "mps"

        # Create simulator with configuration
        return TencirchemCISimulator(simulator_config)