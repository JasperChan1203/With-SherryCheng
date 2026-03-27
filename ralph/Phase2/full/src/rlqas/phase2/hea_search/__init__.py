"""
HEA (Hardware Efficient Ansatz) Search Module for RLQAS Phase 2.

This module provides HEA architecture search capabilities with:
- Configurable entanglement patterns (linear, circular, fully connected)
- Multiple rotation gate types (rx, ry, rz)
- Parameter sharing strategies (none, layer-wise, global)
- RL-based architecture optimization
"""

from .environment import HEASearchEnv
from .circuit_builder import HEACircuitBuilder, create_hea_circuit
from .controller import HEASearchController, run_hea_search
from .config import HEAConfig, get_default_config, get_lih_config

__all__ = [
    # Environment
    "HEASearchEnv",
    # Circuit builder
    "HEACircuitBuilder",
    "create_hea_circuit",
    # Controller
    "HEASearchController",
    "run_hea_search",
    # Configuration
    "HEAConfig",
    "get_default_config",
    "get_lih_config",
]
