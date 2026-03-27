"""Factory for creating CircuitEncoder instances."""
from typing import Dict, Optional
from .base_encoder import CircuitEncoder
from .matrix_encoder import MatrixEncoder
from .sparse_encoder import SparseEncoder
from .one_hot_encoder import OneHotEncoder


class EncoderFactory:
    """Factory for creating circuit encoders by name."""

    @staticmethod
    def create(encoding_method: str, config: Dict = None) -> CircuitEncoder:
        """Create a circuit encoder.

        Args:
            encoding_method: 'matrix' | 'sparse' | 'one_hot'
            config: Optional encoder configuration

        Returns:
            CircuitEncoder instance

        Raises:
            ValueError: If encoding_method is not recognized
        """
        config = config or {}
        if encoding_method == "matrix":
            return MatrixEncoder()
        elif encoding_method == "sparse":
            max_gates = config.get("max_gates")
            return SparseEncoder(max_gates=max_gates)
        elif encoding_method == "one_hot":
            return OneHotEncoder()
        else:
            raise ValueError(
                f"Unknown encoding_method: {encoding_method!r}. "
                f"Must be one of: 'matrix', 'sparse', 'one_hot'"
            )
