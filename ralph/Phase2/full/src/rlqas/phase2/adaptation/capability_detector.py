"""
Capability Detection System for RLQAS Phase 2.

This module provides runtime capability detection for external dependencies
and identifies missing functionality that requires adaptive implementation.
"""

import os
import importlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CapabilityStatus:
    """Status of a detected capability.

    Attributes:
        name: Capability name
        available: Whether capability is available
        source: Source module/package
        version: Version information if available
        missing_features: List of missing features
        error_message: Error message if detection failed
        detected_at: Detection timestamp
    """

    name: str
    available: bool
    source: str = ""
    version: str = ""
    missing_features: List[str] = field(default_factory=list)
    error_message: str = ""
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


class CapabilityDetector:
    """Runtime capability detection system.

    This class provides utilities for:
    - Detecting available functionality in external libraries
    - Identifying missing features required by RLQAS
    - Checking module attributes and methods
    - Caching detection results

    Args:
        cache_results: Whether to cache detection results
    """

    def __init__(self, cache_results: bool = True):
        """Initialize capability detector."""
        self.cache_results = cache_results
        self._cache: Dict[str, CapabilityStatus] = {}
        self._module_cache: Dict[str, Any] = {}

    def detect_module(
        self,
        module_name: str,
        required_attrs: Optional[List[str]] = None,
        required_methods: Optional[List[str]] = None,
    ) -> CapabilityStatus:
        """Detect module availability and capabilities.

        Args:
            module_name: Name of module to detect
            required_attrs: Optional list of required attributes
            required_methods: Optional list of required methods

        Returns:
            CapabilityStatus object
        """
        # Check cache
        cache_key = f"{module_name}_{tuple(required_attrs or [])}_{tuple(required_methods or [])}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        status = CapabilityStatus(
            name=module_name,
            available=False,
        )

        try:
            # Try to import module
            module = importlib.import_module(module_name)
            self._module_cache[module_name] = module
            status.available = True
            status.source = module_name

            # Get version if available
            if hasattr(module, "__version__"):
                status.version = str(module.__version__)

            # Check required attributes
            if required_attrs:
                for attr in required_attrs:
                    if not hasattr(module, attr):
                        status.missing_features.append(f"attr:{attr}")

            # Check required methods
            if required_methods:
                for method in required_methods:
                    if not hasattr(module, method):
                        status.missing_features.append(f"method:{method}")
                    elif not callable(getattr(module, method)):
                        status.missing_features.append(f"callable:{method}")

        except ImportError as e:
            status.error_message = str(e)
        except Exception as e:
            status.error_message = f"Detection error: {str(e)}"

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = status

        return status

    def detect_class_capability(
        self,
        module_name: str,
        class_name: str,
        required_methods: Optional[List[str]] = None,
        required_attrs: Optional[List[str]] = None,
    ) -> CapabilityStatus:
        """Detect class-level capabilities within a module.

        Args:
            module_name: Module containing the class
            class_name: Name of class to check
            required_methods: Required methods
            required_attrs: Required attributes

        Returns:
            CapabilityStatus object
        """
        cache_key = f"{module_name}.{class_name}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        status = CapabilityStatus(
            name=f"{module_name}.{class_name}",
            available=False,
        )

        try:
            # Import module
            module = importlib.import_module(module_name)
            status.source = module_name

            # Get class
            if not hasattr(module, class_name):
                status.error_message = f"Class {class_name} not found in {module_name}"
                if self.cache_results:
                    self._cache[cache_key] = status
                return status

            cls = getattr(module, class_name)

            # Check version
            if hasattr(module, "__version__"):
                status.version = str(module.__version__)

            status.available = True

            # Check required methods
            if required_methods:
                for method in required_methods:
                    if not hasattr(cls, method):
                        status.missing_features.append(f"method:{method}")
                    elif not callable(getattr(cls, method)):
                        status.missing_features.append(f"callable:{method}")

            # Check required attributes
            if required_attrs:
                for attr in required_attrs:
                    if not hasattr(cls, attr):
                        status.missing_features.append(f"attr:{attr}")

        except ImportError as e:
            status.error_message = str(e)
        except Exception as e:
            status.error_message = f"Detection error: {str(e)}"

        if self.cache_results:
            self._cache[cache_key] = status

        return status

    def detect_function_signature(
        self,
        module_name: str,
        function_name: str,
        required_params: Optional[List[str]] = None,
    ) -> CapabilityStatus:
        """Detect function signature compatibility.

        Args:
            module_name: Module containing the function
            function_name: Name of function to check
            required_params: Required parameter names

        Returns:
            CapabilityStatus object
        """
        import inspect

        cache_key = f"{module_name}.{function_name}_sig"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        status = CapabilityStatus(
            name=f"{module_name}.{function_name}",
            available=False,
        )

        try:
            module = importlib.import_module(module_name)
            status.source = module_name

            if not hasattr(module, function_name):
                status.error_message = f"Function {function_name} not found"
                if self.cache_results:
                    self._cache[cache_key] = status
                return status

            func = getattr(module, function_name)
            status.available = True

            if hasattr(module, "__version__"):
                status.version = str(module.__version__)

            # Check signature
            if required_params:
                sig = inspect.signature(func)
                sig_params = list(sig.parameters.keys())

                for param in required_params:
                    if param not in sig_params:
                        status.missing_features.append(f"param:{param}")

        except ImportError as e:
            status.error_message = str(e)
        except Exception as e:
            status.error_message = f"Detection error: {str(e)}"

        if self.cache_results:
            self._cache[cache_key] = status

        return status

    def detect_tencirchem_capabilities(self) -> Dict[str, CapabilityStatus]:
        """Detect Tencirchem library capabilities.

        Returns:
            Dictionary of capability statuses
        """
        capabilities = {}

        # Check main module
        capabilities["tencirchem"] = self.detect_module(
            "tencirchem",
            required_attrs=["ucc"],
        )

        # Check UCC class capabilities
        capabilities["tencirchem.ucc"] = self.detect_class_capability(
            "tencirchem",
            "UCC",
            required_methods=["get_circuit", "run_vqe"],
            required_attrs=["n_qubits", "n_electrons"],
        )

        # Check for parity transformation support
        capabilities["tencirchem.parity"] = self.detect_function_signature(
            "tencirchem",
            "parity_transform",
            required_params=["hamiltonian"],
        )

        return capabilities

    def detect_qiskit_capabilities(self) -> Dict[str, CapabilityStatus]:
        """Detect Qiskit library capabilities.

        Returns:
            Dictionary of capability statuses
        """
        capabilities = {}

        # Check main module
        capabilities["qiskit"] = self.detect_module(
            "qiskit",
            required_attrs=["QuantumCircuit", "Aer"],
        )

        # Check VQE
        capabilities["qiskit.algorithms.VQE"] = self.detect_class_capability(
            "qiskit.algorithms",
            "VQE",
            required_methods=["compute_eigenvalues"],
        )

        return capabilities

    def get_cached_status(self, key: str) -> Optional[CapabilityStatus]:
        """Get cached capability status.

        Args:
            key: Cache key

        Returns:
            CapabilityStatus or None
        """
        return self._cache.get(key)

    def clear_cache(self):
        """Clear detection cache."""
        self._cache.clear()
        self._module_cache.clear()

    def get_all_capabilities(self) -> Dict[str, CapabilityStatus]:
        """Get all detected capabilities.

        Returns:
            Dictionary of all capability statuses
        """
        return self._cache.copy()

    def is_capability_available(self, key: str) -> bool:
        """Check if a capability is available.

        Args:
            key: Capability key

        Returns:
            True if available, False otherwise
        """
        if key in self._cache:
            return self._cache[key].available

        # Try to detect
        if "." in key:
            parts = key.split(".")
            if len(parts) == 2:
                status = self.detect_class_capability(parts[0], parts[1])
                return status.available

        return False

    def get_missing_features(self, key: str) -> List[str]:
        """Get list of missing features for a capability.

        Args:
            key: Capability key

        Returns:
            List of missing feature names
        """
        if key in self._cache:
            return self._cache[key].missing_features.copy()
        return []


def create_detector(cache_results: bool = True) -> CapabilityDetector:
    """Create a capability detector instance.

    Args:
        cache_results: Whether to cache results

    Returns:
        CapabilityDetector instance
    """
    return CapabilityDetector(cache_results=cache_results)


def detect_all_capabilities() -> Dict[str, Dict[str, CapabilityStatus]]:
    """Detect all known library capabilities.

    Returns:
        Dictionary of library capabilities
    """
    detector = CapabilityDetector()

    return {
        "tencirchem": detector.detect_tencirchem_capabilities(),
        "qiskit": detector.detect_qiskit_capabilities(),
    }
