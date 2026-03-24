"""
RLQAS: Reinforcement Learning Quantum Architecture Search
Unified top-level package (Phase 4).

Quick start:
    import rlqas
    result = rlqas.search("LiH", 1.6, ansatz_type="UCC", agent_type="ppo")
    print(f"Error: {result['energy_error_mha']:.3f} mHa")
"""
import pkgutil
# Extend __path__ so rlqas.phase1, rlqas.phase2, rlqas.phase3 remain accessible
# even though this __init__.py makes rlqas a regular (non-namespace) package.
__path__ = pkgutil.extend_path(__path__, __name__)

from .api import search, Experiment

__all__ = ["search", "Experiment"]
__version__ = "4.0.0"
