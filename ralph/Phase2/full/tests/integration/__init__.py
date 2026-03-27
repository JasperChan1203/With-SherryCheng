"""
Integration test package for RLQAS Phase 2.

This package contains comprehensive integration tests for all Phase 2 components.
"""

from .test_phase2_integration import (
    TestPhase2Integration,
)
from .test_multi_algorithm import (
    TestMultiAlgorithmComparison,
    TestAlgorithmPerformance,
)
from .test_hea_integration import (
    TestHEAIntegration,
    TestHEAWithMolecules,
)
from .test_lih_molecule import (
    TestLiH10Qubits,
    TestLiH12Qubits,
    TestChemicalAccuracy,
)
from .test_runtime_adaptation import (
    TestCapabilityDetection,
    TestFeatureImplementation,
    TestAdaptiveExecution,
    TestCapabilityRegistry,
    TestRuntimeAdaptationEndToEnd,
)

__all__ = [
    # Phase 2 integration
    'TestPhase2Integration',
    # Multi-algorithm comparison
    'TestMultiAlgorithmComparison',
    'TestAlgorithmPerformance',
    # HEA integration
    'TestHEAIntegration',
    'TestHEAWithMolecules',
    # LiH molecule tests
    'TestLiH10Qubits',
    'TestLiH12Qubits',
    'TestChemicalAccuracy',
    # Runtime adaptation
    'TestCapabilityDetection',
    'TestFeatureImplementation',
    'TestAdaptiveExecution',
    'TestCapabilityRegistry',
    'TestRuntimeAdaptationEndToEnd',
]
