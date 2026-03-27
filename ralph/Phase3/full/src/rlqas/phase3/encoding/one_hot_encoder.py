"""One-hot encoder: one-hot gate type per (qubit, time_step) position."""
import numpy as np
from .base_encoder import CircuitEncoder


class OneHotEncoder(CircuitEncoder):
    """One-hot circuit encoder.

    For each (qubit, time_step) position, creates a one-hot vector of length
    n_gate_types. Total output: n_qubits * max_depth * n_gate_types.
    """

    def encode(self, circuit: any, n_qubits: int, max_depth: int) -> np.ndarray:
        """Encode circuit as one-hot gate type per position.

        Args:
            circuit: Circuit object
            n_qubits: Number of qubits
            max_depth: Maximum circuit depth

        Returns:
            1D float32 array of length n_qubits * max_depth * N_GATE_TYPES
        """
        matrix = self._circuit_to_gate_matrix(circuit, n_qubits, max_depth)
        n_types = self.N_GATE_TYPES
        result = np.zeros(n_qubits * max_depth * n_types, dtype=np.float32)

        for q in range(n_qubits):
            for t in range(max_depth):
                gate_type = int(matrix[q, t])
                if 0 <= gate_type < n_types:
                    idx = (q * max_depth + t) * n_types + gate_type
                    result[idx] = 1.0

        return result

    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        """Return n_qubits * max_depth * N_GATE_TYPES."""
        return n_qubits * max_depth * self.N_GATE_TYPES
