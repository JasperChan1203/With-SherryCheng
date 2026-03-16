"""
Unit tests for Adaptation Framework.

These tests verify ExplorationFramework, CapabilityDetector,
FeatureImplementer, AdaptiveExecutor, and CapabilityRegistry.
"""

import os
import sys
import tempfile
import pytest
import json

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

from rlqas.phase2.adaptation import (
    ExplorationFramework,
    create_exploration_framework,
    CapabilityDetector,
    create_detector,
    FeatureImplementer,
    create_implementer,
    AdaptiveExecutor,
    create_adaptive_executor,
    CapabilityRegistry,
    create_registry,
)


class TestExplorationFramework:
    """Tests for ExplorationFramework class."""

    def test_initialization(self):
        """Test framework initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ExplorationFramework(output_dir=tmpdir, verbose=0)
            assert "ppo" in framework.discovered_algorithms
            assert "dqn" in framework.discovered_algorithms

    def test_discover_algorithm(self):
        """Test discovering new algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ExplorationFramework(output_dir=tmpdir, verbose=0)

            algo_id = framework.discover_algorithm(
                name="SAC",
                description="Soft Actor-Critic",
                properties={
                    "type": "actor_critic",
                    "action_space": ["continuous"],
                    "sample_efficiency": "high",
                },
            )

            assert algo_id == "sac"
            assert "sac" in framework.discovered_algorithms

    def test_evaluate_compatibility(self):
        """Test algorithm compatibility evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ExplorationFramework(output_dir=tmpdir, verbose=0)

            # Evaluate baseline
            eval_result = framework.evaluate_compatibility("ppo")
            assert "overall_score" in eval_result
            assert "criteria" in eval_result
            assert "recommendation" in eval_result

    def test_compare_algorithms(self):
        """Test algorithm comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ExplorationFramework(output_dir=tmpdir, verbose=0)

            # Evaluate both baselines
            framework.evaluate_compatibility("ppo")
            framework.evaluate_compatibility("dqn")

            comparison = framework.compare_algorithms()
            assert "ranking" in comparison
            assert "best_overall" in comparison
            assert len(comparison["ranking"]) == 2

    def test_save_results(self):
        """Test saving exploration results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = ExplorationFramework(output_dir=tmpdir, verbose=0)
            framework.evaluate_compatibility("ppo")

            path = framework.save_exploration_results()
            assert os.path.exists(path)


class TestCapabilityDetector:
    """Tests for CapabilityDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = CapabilityDetector(cache_results=True)
        assert detector._cache == {}

    def test_detect_module_available(self):
        """Test detecting available module."""
        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_module("os")
        assert status.available is True
        assert status.source == "os"

    def test_detect_module_unavailable(self):
        """Test detecting unavailable module."""
        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_module("nonexistent_module_xyz")
        assert status.available is False
        assert "ImportError" in status.error_message or "No module" in status.error_message

    def test_detect_class_capability(self):
        """Test detecting class capabilities."""
        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_class_capability(
            "os",
            "PathLike",
            required_methods=["__fspath__"],
        )
        # os.PathLike exists
        assert status.available is True

    def test_caching(self):
        """Test result caching."""
        detector = CapabilityDetector(cache_results=True)

        # First detection
        status1 = detector.detect_module("json")

        # Second detection (should be cached)
        status2 = detector.detect_module("json")

        assert status1.available == status2.available
        assert len(detector._cache) > 0

    def test_clear_cache(self):
        """Test clearing cache."""
        detector = CapabilityDetector(cache_results=True)
        detector.detect_module("json")
        assert len(detector._cache) > 0

        detector.clear_cache()
        assert len(detector._cache) == 0


class TestFeatureImplementer:
    """Tests for FeatureImplementer class."""

    def test_initialization(self):
        """Test implementer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)
            assert "parity_adapter" in implementer._templates
            assert "transformer_adapter" in implementer._templates

    def test_get_available_templates(self):
        """Test getting available templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)
            templates = implementer.get_available_templates()
            assert "parity_adapter" in templates

    def test_generate_parity_adapter(self):
        """Test generating parity adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)

            output_path = implementer.generate_parity_adapter(
                target_library="tencirchem",
            )

            assert os.path.exists(output_path)
            assert "tencirchem" in output_path

    def test_generate_transformer_adapter(self):
        """Test generating transformer adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)

            output_path = implementer.generate_transformer_adapter(
                transform_type="jordan_wigner",
            )

            assert os.path.exists(output_path)

    def test_verify_implementation(self):
        """Test verifying generated implementation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)

            output_path = implementer.generate_parity_adapter()
            verification = implementer.verify_implementation(output_path)

            assert verification["exists"] is True
            assert verification["valid_syntax"] is True

    def test_verify_nonexistent(self):
        """Test verifying nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)

            verification = implementer.verify_implementation("/nonexistent/file.py")
            assert verification["exists"] is False


class TestAdaptiveExecutor:
    """Tests for AdaptiveExecutor class."""

    def test_initialization(self):
        """Test executor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)
            assert executor.detector is not None
            assert executor.implementer is not None
            assert executor.registry is not None

    def test_check_capability_available(self):
        """Test checking available capability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            # Check for os module (should be available)
            result = executor.check_capability("os", auto_fix=False)
            assert result is True

    def test_check_capability_missing(self):
        """Test checking missing capability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            # Check for nonexistent module
            result = executor.check_capability("nonexistent_module_xyz", auto_fix=False)
            assert result is False

    def test_execute_with_adaptation(self):
        """Test executing with adaptation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            def simple_operation(x, y):
                return x + y

            result = executor.execute_with_adaptation(
                operation=simple_operation,
                required_capabilities=["os"],
                x=1,
                y=2,
            )

            assert result == 3

    def test_get_execution_log(self):
        """Test getting execution log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0, auto_implement=False)

            # Execute with adaptation to generate log entry
            def simple_op(x, y):
                return x + y

            executor.execute_with_adaptation(
                operation=simple_op,
                required_capabilities=["os"],
                x=1,
                y=2,
            )
            log = executor.get_execution_log()

            assert len(log) > 0

    def test_save_execution_report(self):
        """Test saving execution report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)
            os.chdir(tmpdir)

            executor.check_capability("os")
            report_path = executor.save_execution_report()

            assert os.path.exists(report_path)


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry class."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = CapabilityRegistry()
        assert registry._capabilities == {}

    def test_register_capability(self):
        """Test registering capability."""
        registry = CapabilityRegistry()

        success = registry.register_capability(
            capability_key="test.capability",
            source="test_source",
            version="1.0.0",
            validated=True,
        )

        assert success is True
        assert registry.is_registered("test.capability")

    def test_is_validated(self):
        """Test validation check."""
        registry = CapabilityRegistry()

        registry.register_capability(
            "validated_cap",
            "source",
            "1.0",
            validated=True,
        )
        registry.register_capability(
            "unvalidated_cap",
            "source",
            "1.0",
            validated=False,
        )

        assert registry.is_validated("validated_cap") is True
        assert registry.is_validated("unvalidated_cap") is False

    def test_get_record(self):
        """Test getting capability record."""
        registry = CapabilityRegistry()

        registry.register_capability(
            "test_cap",
            "test_source",
            "1.0",
            metadata={"key": "value"},
        )

        record = registry.get_record("test_cap")
        assert record is not None
        assert record["source"] == "test_source"
        assert record["metadata"]["key"] == "value"

    def test_update_capability(self):
        """Test updating capability."""
        registry = CapabilityRegistry()

        registry.register_capability("test_cap", "source", "1.0")
        success = registry.update_capability(
            "test_cap",
            {"version": "2.0", "validated": True},
        )

        assert success is True
        record = registry.get_record("test_cap")
        assert record["version"] == "2.0"

    def test_unregister_capability(self):
        """Test unregistering capability."""
        registry = CapabilityRegistry()

        registry.register_capability("test_cap", "source", "1.0")
        assert registry.is_registered("test_cap")

        success = registry.unregister_capability("test_cap")
        assert success is True
        assert not registry.is_registered("test_cap")

    def test_get_summary(self):
        """Test getting registry summary."""
        registry = CapabilityRegistry()

        registry.register_capability("cap1", "source1", "1.0", validated=True)
        registry.register_capability("cap2", "source2", "1.0", validated=False)

        summary = registry.get_summary()
        assert summary["total_capabilities"] == 2
        assert summary["validated_capabilities"] == 1

    def test_export_import(self):
        """Test export and import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = CapabilityRegistry()
            registry.register_capability("test_cap", "source", "1.0", validated=True)

            # Export
            export_path = os.path.join(tmpdir, "capabilities.json")
            registry.export_capabilities(export_path)
            assert os.path.exists(export_path)

            # Import to new registry
            new_registry = CapabilityRegistry()
            count = new_registry.import_capabilities(export_path)
            assert count == 1
            assert new_registry.is_registered("test_cap")

    def test_persistent_storage(self):
        """Test persistent storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "registry.json")

            # Create and register
            registry1 = CapabilityRegistry(storage_path=storage_path)
            registry1.register_capability("persist_cap", "source", "1.0")

            # Load from storage
            registry2 = CapabilityRegistry(storage_path=storage_path)
            assert registry2.is_registered("persist_cap")


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_exploration_framework(self):
        """Test create_exploration_framework."""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = create_exploration_framework(output_dir=tmpdir)
            assert isinstance(framework, ExplorationFramework)

    def test_create_detector(self):
        """Test create_detector."""
        detector = create_detector()
        assert isinstance(detector, CapabilityDetector)

    def test_create_implementer(self):
        """Test create_implementer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = create_implementer(output_dir=tmpdir)
            assert isinstance(implementer, FeatureImplementer)

    def test_create_adaptive_executor(self):
        """Test create_adaptive_executor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = create_adaptive_executor(output_dir=tmpdir)
            assert isinstance(executor, AdaptiveExecutor)

    def test_create_registry(self):
        """Test create_registry."""
        registry = create_registry()
        assert isinstance(registry, CapabilityRegistry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
