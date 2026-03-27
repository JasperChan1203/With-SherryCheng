"""Sparse encoder: non-identity gates as (qubit, time, gate_type) triples."""
import numpy as np
from typing import Optional
from .base_encoder import CircuitEncoder


class SparseEncoder(CircuitEncoder):
    """Sparse circuit encoder.

    Encodes only non-identity gates as (qubit_idx, time_step, gate_type) triples,
    padded to max_gates * 3 for a fixed output dimension.
    """

    def __init__(self, max_gates: Optional[int] = None):
        """Initialize sparse encoder.

        Args:
            max_gates: Max number of gates to encode. If None, uses n_qubits * max_depth.
        """
        self._max_gates = max_gates

    def _get_max_gates(self, n_qubits: int, max_depth: int) -> int:
        if self._max_gates is not None:
            return self._max_gates
        return n_qubits * max_depth

    def encode(self, circuit: any, n_qubits: int, max_depth: int) -> np.ndarray:
        """Encode circuit as sparse (qubit, time, gate_type) triples.

        Args:
            circuit: Circuit object
            n_qubits: Number of qubits
            max_depth: Maximum circuit depth

        Returns:
            1D float32 array of length max_gates * 3
        """
        matrix = self._circuit_to_gate_matrix(circuit, n_qubits, max_depth)
        max_gates = self._get_max_gates(n_qubits, max_depth)

        # Collect non-zero positions
        triples = []
        for q in range(n_qubits):
            for t in range(max_depth):
                if matrix[q, t] != 0:
                    triples.append([float(q), float(t), float(matrix[q, t])])
                    if len(triples) >= max_gates:
                        break
            if len(triples) >= max_gates:
                break

        # Pad to max_gates triples
        result = np.zeros(max_gates * 3, dtype=np.float32)
        for i, triple in enumerate(triples[:max_gates]):
            result[i * 3: i * 3 + 3] = triple

        return result

    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        """Return max_gates * 3."""
        return self._get_max_gates(n_qubits, max_depth) * 3
