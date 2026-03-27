"""RLQAS Phase 1: Reinforcement Learning for Quantum Architecture Search.

This package integrates all modules from Phase 1 Tasks 001-005 into a unified
Python package for quantum chemistry architecture search using reinforcement
learning.
"""

# Patch gym to gymnasium before any imports to suppress deprecation warnings
import sys
import os
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'
os.environ['GYM_SUPPRESS_ENV_MESSAGE'] = '1'

# Attempt to replace gym with gymnasium in sys.modules before any imports
try:
    import gymnasium
    # Replace gym with gymnasium to prevent stable-baselines3 from importing gym
    sys.modules['gym'] = gymnasium
except ImportError:
    pass

# Molecule processing
from .molecule import MoleculeData, process_molecule

# Quantum simulation
from .simulator import QuantumSimulator, TencirchemCISimulator, SimulatorFactory

# Reinforcement learning
from .rl import RLAgent, PPOAgent, AgentConfig

# UCC architecture search
from .search import (
    UCCSearchEnv,
    UCCCircuitBuilder,
    UCCRewardFunction,
    UCCSearchController,
    UCCSearchConfig,
)

# Validation and evaluation
from .validation import run_lih_validation, MetricsCollector, ReportGenerator

__all__ = [
    # Molecule processing
    "MoleculeData",
    "process_molecule",
    # Quantum simulation
    "QuantumSimulator",
    "TencirchemCISimulator",
    "SimulatorFactory",
    # Reinforcement learning
    "RLAgent",
    "PPOAgent",
    "AgentConfig",
    # UCC architecture search
    "UCCSearchEnv",
    "UCCCircuitBuilder",
    "UCCRewardFunction",
    "UCCSearchController",
    "UCCSearchConfig",
    # Validation and evaluation
    "run_lih_validation",
    "MetricsCollector",
    "ReportGenerator",
]

__version__ = "1.0.0"