"""
Quantum Simulator abstract base class for RLQAS.

This module defines the abstract base class QuantumSimulator, which provides
a generic interface for quantum circuit simulation compatible with Tencirchem.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from openfermion import QubitOperator


class QuantumSimulator(ABC):
    """Abstract base class for quantum simulators.

    Defines the interface for quantum circuit simulation compatible with Tencirchem.
    """

    @abstractmethod
    def compute_energy(
        self,
        circuit: Any,  # Tencirchem-compatible circuit (e.g., tensorcircuit.Circuit)
        hamiltonian: QubitOperator,
        initial_state: Optional[np.ndarray] = None
    ) -> float:
        """Compute energy expectation value for given circuit and Hamiltonian.

        Args:
            circuit: Quantum circuit in Tencirchem-compatible format
            hamiltonian: Qubit Hamiltonian from Phase 1 Task 001
            initial_state: Optional initial state vector (default is reference state)

        Returns:
            Energy expectation value in Hartree
        """
        pass

    @abstractmethod
    def get_max_qubits(self) -> int:
        """Get maximum number of qubits supported by this simulator."""
        pass

    @abstractmethod
    def estimate_memory(self, n_qubits: int) -> float:
        """Estimate memory requirement in GB for given number of qubits."""
        pass