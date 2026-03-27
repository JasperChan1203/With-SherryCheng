"""Memory management for large quantum circuit evaluations.

Monitors process memory usage via psutil and adapts batch size to prevent
OOM errors during long-running Phase 3 experiments.
"""

import gc
import os
from typing import Any, Dict

import psutil


class MemoryManager:
    """Manages memory usage during batch circuit evaluation.

    Monitors current process memory and reduces batch size when usage
    approaches ``max_memory_gb``.

    Args:
        max_memory_gb: Upper limit on allowed memory usage in GB.
            Batch size is reduced when usage exceeds
            ``_threshold_fraction * max_memory_gb``.
    """

    def __init__(self, max_memory_gb: float = 32.0) -> None:
        self.max_memory_gb = max_memory_gb
        self._threshold_fraction: float = 0.85

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_memory(self) -> Dict[str, float]:
        """Return current memory usage statistics.

        Returns:
            Dict with keys:
              - ``used_gb``: RSS memory of current process in GB.
              - ``available_gb``: Available system memory in GB.
              - ``percent``: Current process RSS as percentage of total RAM.
              - ``max_allowed_gb``: The configured maximum (``max_memory_gb``).
        """
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            used_gb = mem_info.rss / (1024 ** 3)

            virtual_mem = psutil.virtual_memory()
            available_gb = virtual_mem.available / (1024 ** 3)
            total_gb = virtual_mem.total / (1024 ** 3)
            percent = (used_gb / total_gb) * 100.0

            return {
                "used_gb": used_gb,
                "available_gb": available_gb,
                "percent": percent,
                "max_allowed_gb": self.max_memory_gb,
            }
        except Exception:
            # Fallback if psutil is unavailable or fails
            return {
                "used_gb": 0.0,
                "available_gb": self.max_memory_gb,
                "percent": 0.0,
                "max_allowed_gb": self.max_memory_gb,
            }

    def adapt_batch_size(
        self,
        current_batch_size: int,
        evaluator: Any = None,
    ) -> int:
        """Reduce batch size if memory threshold is approached.

        If current process RSS exceeds
        ``_threshold_fraction * max_memory_gb``, halves the batch size.
        Otherwise returns ``current_batch_size`` unchanged.

        Args:
            current_batch_size: The current batch size.
            evaluator: Unused; kept for interface compatibility.

        Returns:
            New (possibly reduced) batch size (minimum 1).
        """
        mem = self.check_memory()
        used_fraction = mem["used_gb"] / max(self.max_memory_gb, 1e-9)

        if used_fraction > self._threshold_fraction:
            new_size = max(1, current_batch_size // 2)
            return new_size
        return current_batch_size

    def release_intermediate_state(self) -> None:
        """Hint the garbage collector to release intermediate computation state.

        Call this between batch evaluations in memory-constrained scenarios.
        """
        gc.collect()
