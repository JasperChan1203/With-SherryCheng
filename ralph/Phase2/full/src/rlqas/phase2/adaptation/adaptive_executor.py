"""
Adaptive Executor for RLQAS Phase 2.

This module provides runtime adaptive execution flow with capability
checking and on-demand feature implementation.
"""

import os
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime

from rlqas.phase2.adaptation.capability_detector import (
    CapabilityDetector,
    CapabilityStatus,
    create_detector,
)
from rlqas.phase2.adaptation.feature_implementer import (
    FeatureImplementer,
    create_implementer,
)
from rlqas.phase2.adaptation.capability_registry import (
    CapabilityRegistry,
    create_registry,
)


class AdaptiveExecutor:
    """Runtime adaptive execution system.

    This class provides:
    - Runtime capability checking before critical operations
    - On-demand feature implementation
    - Capability caching and reuse
    - Seamless integration with existing code

    Args:
        output_dir: Directory for generated implementations
        auto_implement: Whether to automatically implement missing features
        verbose: Verbosity level
    """

    def __init__(
        self,
        output_dir: str = "src/rlqas/phase2/adaptation/generated",
        auto_implement: bool = True,
        verbose: int = 1,
    ):
        """Initialize adaptive executor."""
        self.output_dir = output_dir
        self.auto_implement = auto_implement
        self.verbose = verbose

        # Initialize components
        self.detector = create_detector(cache_results=True)
        self.implementer = create_implementer(output_dir=output_dir)
        self.registry = create_registry()

        # Execution state
        self._capabilities_checked: Dict[str, bool] = {}
        self._features_implemented: List[str] = []
        self._execution_log: List[Dict] = []

    def check_capability(
        self,
        capability_key: str,
        auto_fix: Optional[bool] = None,
    ) -> bool:
        """Check if a capability is available.

        Args:
            capability_key: Capability identifier
            auto_fix: Whether to auto-implement if missing

        Returns:
            True if capability is available
        """
        if auto_fix is None:
            auto_fix = self.auto_implement

        # Check registry first
        if self.registry.is_registered(capability_key):
            record = self.registry.get_record(capability_key)
            if record and record.get("validated", False):
                self._capabilities_checked[capability_key] = True
                return True

        # Detect capability
        status = self._detect_capability(capability_key)

        if status.available:
            # Register in registry
            self.registry.register_capability(
                capability_key,
                status.source,
                status.version,
            )
            self._capabilities_checked[capability_key] = True
            return True

        # Capability missing
        if self.verbose >= 1:
            print(f"Missing capability: {capability_key}")
            print(f"  Missing features: {status.missing_features}")

        # Auto-implement if enabled
        if auto_fix:
            return self._implement_missing_capability(capability_key, status)

        self._capabilities_checked[capability_key] = False
        return False

    def _detect_capability(self, capability_key: str) -> CapabilityStatus:
        """Detect a specific capability.

        Args:
            capability_key: Capability identifier

        Returns:
            CapabilityStatus object
        """
        # Handle different capability types
        if capability_key.startswith("tencirchem"):
            if "parity" in capability_key:
                return self.detector.detect_function_signature(
                    "tencirchem",
                    "parity_transform",
                    required_params=["hamiltonian"],
                )
            elif "ucc" in capability_key:
                return self.detector.detect_class_capability(
                    "tencirchem",
                    "UCC",
                    required_methods=["get_circuit", "run_vqe"],
                )
            else:
                return self.detector.detect_module("tencirchem")

        elif capability_key.startswith("qiskit"):
            return self.detector.detect_module("qiskit")

        # Generic detection
        if "." in capability_key:
            parts = capability_key.split(".")
            if len(parts) == 2:
                return self.detector.detect_class_capability(parts[0], parts[1])

        return self.detector.detect_module(capability_key)

    def _implement_missing_capability(
        self,
        capability_key: str,
        status: CapabilityStatus,
    ) -> bool:
        """Implement a missing capability.

        Args:
            capability_key: Capability identifier
            status: Capability status

        Returns:
            True if implementation successful
        """
        if self.verbose >= 1:
            print(f"  Attempting to implement: {capability_key}")

        # Determine implementation strategy
        if "parity" in capability_key:
            return self._implement_parity_support(capability_key, status)

        elif "transform" in capability_key:
            return self._implement_transformer(capability_key, status)

        # Unknown capability type
        if self.verbose >= 1:
            print(f"  No implementation strategy for: {capability_key}")

        return False

    def _implement_parity_support(
        self,
        capability_key: str,
        status: CapabilityStatus,
    ) -> bool:
        """Implement parity transformation support.

        Args:
            capability_key: Capability identifier
            status: Capability status

        Returns:
            True if successful
        """
        try:
            # Generate parity adapter
            target = "tencirchem" if "tencirchem" in capability_key else "generic"
            output_path = self.implementer.generate_parity_adapter(
                target_library=target,
            )

            if self.verbose >= 1:
                print(f"  Generated parity adapter: {output_path}")

            # Verify implementation
            verification = self.implementer.verify_implementation(output_path)
            if not verification.get("importable", False):
                if self.verbose >= 1:
                    print(f"  Verification failed: {verification.get('errors', [])}")
                return False

            # Register in registry
            self.registry.register_capability(
                capability_key,
                "generated_parity_adapter",
                "auto-generated",
                validated=True,
                implementation_path=output_path,
            )

            self._features_implemented.append(capability_key)
            self._log_action("implement_parity", capability_key, True)

            return True

        except Exception as e:
            if self.verbose >= 1:
                print(f"  Implementation error: {e}")
            self._log_action("implement_parity", capability_key, False, str(e))
            return False

    def _implement_transformer(
        self,
        capability_key: str,
        status: CapabilityStatus,
    ) -> bool:
        """Implement transformation support.

        Args:
            capability_key: Capability identifier
            status: Capability status

        Returns:
            True if successful
        """
        try:
            # Extract transform type
            transform_type = capability_key.replace("transform_", "").replace(".adapter", "")

            # Generate transformer adapter
            output_path = self.implementer.generate_transformer_adapter(
                transform_type=transform_type,
            )

            if self.verbose >= 1:
                print(f"  Generated transformer adapter: {output_path}")

            # Register in registry
            self.registry.register_capability(
                capability_key,
                "generated_transformer_adapter",
                "auto-generated",
                validated=True,
                implementation_path=output_path,
            )

            self._features_implemented.append(capability_key)
            self._log_action("implement_transformer", capability_key, True)

            return True

        except Exception as e:
            if self.verbose >= 1:
                print(f"  Implementation error: {e}")
            self._log_action("implement_transformer", capability_key, False, str(e))
            return False

    def execute_with_adaptation(
        self,
        operation: Callable,
        required_capabilities: List[str],
        *args,
        **kwargs,
    ) -> Any:
        """Execute an operation with runtime adaptation.

        Args:
            operation: Operation to execute
            required_capabilities: List of required capabilities
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result
        """
        if self.verbose >= 2:
            print(f"Executing operation: {operation.__name__}")
            print(f"  Required capabilities: {required_capabilities}")

        # Check all required capabilities
        missing = []
        for cap in required_capabilities:
            if not self.check_capability(cap, auto_fix=True):
                missing.append(cap)

        if missing:
            if self.verbose >= 1:
                print(f"  Warning: Some capabilities could not be implemented: {missing}")

        # Execute operation
        try:
            result = operation(*args, **kwargs)
            self._log_action("execute", operation.__name__, True)
            return result
        except Exception as e:
            self._log_action("execute", operation.__name__, False, str(e))
            raise

    def process_molecule(
        self,
        molecule_processor: Callable,
        molecule_data: Dict[str, Any],
        use_parity: bool = True,
    ) -> Dict[str, Any]:
        """Process a molecule with adaptive capability support.

        This is the example use case from the PRD:
        1. Detection: Check if tencirchem.ucc supports parity mapping
        2. Implementation: Generate TencirchemParityAdapter if needed
        3. Integration: Use adapter for parity calls
        4. Caching: Store adapter in registry

        Args:
            molecule_processor: Molecule processing function
            molecule_data: Molecule data dictionary
            use_parity: Whether to use parity transformation

        Returns:
            Processing results
        """
        required_caps = ["tencirchem"]
        if use_parity:
            required_caps.append("tencirchem.parity")

        return self.execute_with_adaptation(
            operation=molecule_processor,
            required_capabilities=required_caps,
            molecule_data=molecule_data,
            use_parity=use_parity,
        )

    def _log_action(self, action_type: str, target: str, success: bool, error: str = ""):
        """Log an action.

        Args:
            action_type: Type of action
            target: Action target
            success: Whether action succeeded
            error: Optional error message
        """
        self._execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "target": target,
            "success": success,
            "error": error,
        })

    def get_execution_log(self) -> List[Dict]:
        """Get execution log.

        Returns:
            List of log entries
        """
        return self._execution_log.copy()

    def get_implemented_features(self) -> List[str]:
        """Get list of implemented features.

        Returns:
            List of feature names
        """
        return self._features_implemented.copy()

    def get_capability_status(self, capability_key: str) -> Optional[bool]:
        """Get cached capability status.

        Args:
            capability_key: Capability identifier

        Returns:
            Status or None if not checked
        """
        return self._capabilities_checked.get(capability_key)

    def save_execution_report(self, output_path: Optional[str] = None) -> str:
        """Save execution report.

        Args:
            output_path: Optional output path

        Returns:
            Path to saved report
        """
        import json

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"adaptation_report_{timestamp}.json"

        report = {
            "generated_at": datetime.now().isoformat(),
            "capabilities_checked": self._capabilities_checked,
            "features_implemented": self._features_implemented,
            "execution_log": self._execution_log,
            "registry_summary": self.registry.get_summary(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return output_path


def create_adaptive_executor(
    output_dir: str = "src/rlqas/phase2/adaptation/generated",
    auto_implement: bool = True,
    verbose: int = 1,
) -> AdaptiveExecutor:
    """Create an adaptive executor instance.

    Args:
        output_dir: Output directory
        auto_implement: Auto-implement missing features
        verbose: Verbosity level

    Returns:
        AdaptiveExecutor instance
    """
    return AdaptiveExecutor(
        output_dir=output_dir,
        auto_implement=auto_implement,
        verbose=verbose,
    )
