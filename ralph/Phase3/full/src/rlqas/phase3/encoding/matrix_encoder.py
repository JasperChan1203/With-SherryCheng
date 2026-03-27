"""Matrix encoder: gate-type matrix encoding."""
import numpy as np
from .base_encoder import CircuitEncoder


class MatrixEncoder(CircuitEncoder):
    """Gate-type matrix encoder.

    Encodes circuit as a (n_qubits x max_depth) matrix where each cell
    contains an integer gate type code. Flattened row-major to 1D vector.
    """

    def encode(self, circuit: any, n_qubits: int, max_depth: int) -> np.ndarray:
        """Encode circuit as flattened gate-type matrix.

        Args:
            circuit: Circuit object
            n_qubits: Number of qubits
            max_depth: Maximum circuit depth

        Returns:
            1D float32 array of length n_qubits * max_depth
        """
        matrix = self._circuit_to_gate_matrix(circuit, n_qubits, max_depth)
        return matrix.flatten().astype(np.float32)

    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        """Return n_qubits * max_depth."""
        return n_qubits * max_depth
