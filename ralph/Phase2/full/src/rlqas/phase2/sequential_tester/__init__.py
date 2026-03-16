"""
Sequential Testing Framework for RLQAS Phase 2.

This module provides utilities for sequential testing and comparison
of multiple RL algorithms on quantum architecture search problems.
"""

from .sequential_tester import SequentialRLTester
from .comparison import (
    ComparisonUtilities,
    compare_energy_convergence,
    compare_training_efficiency,
    generate_summary_report,
)
from .metrics import (
    MetricsCollector,
    create_metrics_collector,
    compare_excitation_efficiency,
)

__all__ = [
    # Main tester
    "SequentialRLTester",
    # Comparison utilities
    "ComparisonUtilities",
    "compare_energy_convergence",
    "compare_training_efficiency",
    "generate_summary_report",
    # Metrics collection
    "MetricsCollector",
    "create_metrics_collector",
    "compare_excitation_efficiency",
]
