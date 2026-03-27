"""
Capability Registry for RLQAS Phase 2.

This module provides a registry for storing and managing implemented
capabilities and their validation status.
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime


class CapabilityRegistry:
    """Registry for storing and managing capabilities.

    This class provides:
    - Storage for implemented features and their validation status
    - Capability sharing across experiments
    - Capability evolution and improvement tracking
    - Persistence to disk

    Args:
        storage_path: Optional path for persistent storage
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize capability registry."""
        self.storage_path = storage_path
        self._capabilities: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
        }

        # Load from storage if provided
        if storage_path and os.path.exists(storage_path):
            self._load_from_storage()

    def register_capability(
        self,
        capability_key: str,
        source: str,
        version: str,
        validated: bool = False,
        implementation_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Register a capability in the registry.

        Args:
            capability_key: Capability identifier
            source: Source of the capability
            version: Version string
            validated: Whether capability has been validated
            implementation_path: Path to implementation file
            metadata: Additional metadata

        Returns:
            True if registration successful
        """
        self._capabilities[capability_key] = {
            "key": capability_key,
            "source": source,
            "version": version,
            "validated": validated,
            "implementation_path": implementation_path,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "last_used": None,
            "use_count": 0,
        }

        self._metadata["last_updated"] = datetime.now().isoformat()

        # Save if storage path is set
        if self.storage_path:
            self._save_to_storage()

        return True

    def get_record(self, capability_key: str) -> Optional[Dict]:
        """Get capability record.

        Args:
            capability_key: Capability identifier

        Returns:
            Capability record or None
        """
        return self._capabilities.get(capability_key)

    def is_registered(self, capability_key: str) -> bool:
        """Check if capability is registered.

        Args:
            capability_key: Capability identifier

        Returns:
            True if registered
        """
        return capability_key in self._capabilities

    def is_validated(self, capability_key: str) -> bool:
        """Check if capability is validated.

        Args:
            capability_key: Capability identifier

        Returns:
            True if validated
        """
        record = self._capabilities.get(capability_key)
        if record:
            return record.get("validated", False)
        return False

    def update_capability(
        self,
        capability_key: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update capability record.

        Args:
            capability_key: Capability identifier
            updates: Dictionary of updates

        Returns:
            True if update successful
        """
        if capability_key not in self._capabilities:
            return False

        record = self._capabilities[capability_key]
        record.update(updates)
        record["last_updated"] = datetime.now().isoformat()

        self._metadata["last_updated"] = datetime.now().isoformat()

        if self.storage_path:
            self._save_to_storage()

        return True

    def mark_used(self, capability_key: str):
        """Mark capability as used.

        Args:
            capability_key: Capability identifier
        """
        if capability_key in self._capabilities:
            record = self._capabilities[capability_key]
            record["last_used"] = datetime.now().isoformat()
            record["use_count"] = record.get("use_count", 0) + 1

    def unregister_capability(self, capability_key: str) -> bool:
        """Unregister a capability.

        Args:
            capability_key: Capability identifier

        Returns:
            True if unregistration successful
        """
        if capability_key in self._capabilities:
            del self._capabilities[capability_key]
            self._metadata["last_updated"] = datetime.now().isoformat()

            if self.storage_path:
                self._save_to_storage()

            return True
        return False

    def get_all_capabilities(self) -> Dict[str, Dict]:
        """Get all registered capabilities.

        Returns:
            Dictionary of all capabilities
        """
        return self._capabilities.copy()

    def get_validated_capabilities(self) -> Dict[str, Dict]:
        """Get all validated capabilities.

        Returns:
            Dictionary of validated capabilities
        """
        return {
            k: v for k, v in self._capabilities.items()
            if v.get("validated", False)
        }

    def get_capabilities_by_source(self, source: str) -> Dict[str, Dict]:
        """Get capabilities by source.

        Args:
            source: Source identifier

        Returns:
            Dictionary of capabilities from source
        """
        return {
            k: v for k, v in self._capabilities.items()
            if v.get("source") == source
        }

    def search_capabilities(self, query: str) -> List[Dict]:
        """Search capabilities by query.

        Args:
            query: Search query

        Returns:
            List of matching capability records
        """
        results = []
        query_lower = query.lower()

        for key, record in self._capabilities.items():
            # Search in key, source, and metadata
            if (query_lower in key.lower() or
                query_lower in record.get("source", "").lower() or
                any(query_lower in str(v).lower() for v in record.get("metadata", {}).values())):
                results.append(record)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary.

        Returns:
            Summary dictionary
        """
        total = len(self._capabilities)
        validated = sum(1 for v in self._capabilities.values() if v.get("validated", False))

        return {
            "total_capabilities": total,
            "validated_capabilities": validated,
            "unvalidated_capabilities": total - validated,
            "sources": list(set(v.get("source", "") for v in self._capabilities.values())),
            "metadata": self._metadata,
        }

    def export_capabilities(self, output_path: str) -> str:
        """Export capabilities to file.

        Args:
            output_path: Output file path

        Returns:
            Path to exported file
        """
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "metadata": self._metadata,
            "capabilities": self._capabilities,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return output_path

    def import_capabilities(
        self,
        import_path: str,
        merge: bool = True,
    ) -> int:
        """Import capabilities from file.

        Args:
            import_path: Input file path
            merge: Whether to merge with existing capabilities

        Returns:
            Number of capabilities imported
        """
        with open(import_path, "r") as f:
            import_data = json.load(f)

        capabilities = import_data.get("capabilities", {})
        count = 0

        if merge:
            for key, record in capabilities.items():
                if key not in self._capabilities:
                    self._capabilities[key] = record
                    count += 1
        else:
            self._capabilities = capabilities
            count = len(capabilities)

        self._metadata["last_updated"] = datetime.now().isoformat()

        if self.storage_path:
            self._save_to_storage()

        return count

    def _save_to_storage(self):
        """Save registry to storage."""
        if not self.storage_path:
            return

        os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else ".", exist_ok=True)

        data = {
            "metadata": self._metadata,
            "capabilities": self._capabilities,
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_storage(self):
        """Load registry from storage."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return

        with open(self.storage_path, "r") as f:
            data = json.load(f)

        self._metadata = data.get("metadata", self._metadata)
        self._capabilities = data.get("capabilities", {})


def create_registry(
    storage_path: Optional[str] = None,
) -> CapabilityRegistry:
    """Create a capability registry instance.

    Args:
        storage_path: Optional storage path

    Returns:
        CapabilityRegistry instance
    """
    return CapabilityRegistry(storage_path=storage_path)
