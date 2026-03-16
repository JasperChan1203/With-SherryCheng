"""
Adaptation Module for RLQAS Phase 2.

This module provides autonomous RL exploration and runtime environment
adaptation capabilities including:
- RL Algorithm Exploration Framework
- Capability Detection System
- Feature Implementer with templates
- Adaptive Execution Flow
- Capability Registry
"""

from .exploration_framework import (
    ExplorationFramework,
    create_exploration_framework,
)
from .capability_detector import (
    CapabilityDetector,
    CapabilityStatus,
    create_detector,
    detect_all_capabilities,
)
from .feature_implementer import (
    FeatureImplementer,
    create_implementer,
)
from .adaptive_executor import (
    AdaptiveExecutor,
    create_adaptive_executor,
)
from .capability_registry import (
    CapabilityRegistry,
    create_registry,
)

__all__ = [
    # Exploration framework
    "ExplorationFramework",
    "create_exploration_framework",
    # Capability detection
    "CapabilityDetector",
    "CapabilityStatus",
    "create_detector",
    "detect_all_capabilities",
    # Feature implementation
    "FeatureImplementer",
    "create_implementer",
    # Adaptive execution
    "AdaptiveExecutor",
    "create_adaptive_executor",
    # Capability registry
    "CapabilityRegistry",
    "create_registry",
]
