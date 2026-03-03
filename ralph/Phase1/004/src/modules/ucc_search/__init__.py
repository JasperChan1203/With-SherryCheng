"""UCC Search Module for RLQAS Phase 1.

This module implements the UCC architecture search environment, circuit builder,
reward function, and controller for quantum chemistry simulation.

Classes:
    UCCSearchEnv: Gym-compatible environment for UCC architecture search.
    UCCCircuitBuilder: Builds UCC quantum circuits from excitation sequences.
    UCCRewardFunction: Computes rewards based on energy improvement.
    UCCSearchController: Manages the complete UCC search process.
    UCCSearchConfig: Configuration management for UCC search.
"""

from .environment import UCCSearchEnv
from .circuit_builder import UCCCircuitBuilder
from .reward_function import UCCRewardFunction
from .controller import UCCSearchController
from .config import UCCSearchConfig

__all__ = [
    "UCCSearchEnv",
    "UCCCircuitBuilder",
    "UCCRewardFunction",
    "UCCSearchController",
    "UCCSearchConfig",
]