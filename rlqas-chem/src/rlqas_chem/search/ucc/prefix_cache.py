"""PrefixCache: operator prefix → VQE energy cache for Tree-GRPO."""
from typing import Dict, Optional


class PrefixCache:
    """Cache mapping operator prefix tuples to VQE energies.

    Enables Tree-GRPO to detect and reuse previously computed circuit
    energies when multiple episodes share the same operator sequence prefix.
    Different prefix orderings (e.g., (0,1) vs (1,0)) are stored separately.
    """

    def __init__(self):
        self._cache: Dict[tuple, float] = {}
        self._hits: int = 0

    def get(self, prefix: tuple) -> Optional[float]:
        """Look up energy for prefix. Increments hit counter if found."""
        energy = self._cache.get(prefix, None)
        if energy is not None:
            self._hits += 1
        return energy

    def put(self, prefix: tuple, energy: float) -> None:
        """Store energy for prefix."""
        self._cache[prefix] = energy

    def hits(self) -> int:
        """Return cumulative number of cache hits."""
        return self._hits

    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached entries and reset hit counter."""
        self._cache.clear()
        self._hits = 0
