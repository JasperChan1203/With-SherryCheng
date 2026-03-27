"""
Runtime Adaptation Integration Tests.

These tests verify the runtime adaptation framework including:
- Capability detection
- Dynamic feature implementation
- Adaptive execution
- Capability registry
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


class TestCapabilityDetection:
    """Tests for capability detection system."""

    def test_detect_available_module(self):
        """Test detecting an available module."""
        from rlqas.phase2.adaptation import CapabilityDetector

        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_module('os')

        assert status.available is True
        assert status.source == 'os'

    def test_detect_unavailable_module(self):
        """Test detecting an unavailable module."""
        from rlqas.phase2.adaptation import CapabilityDetector

        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_module('nonexistent_module_xyz123')

        assert status.available is False
        assert 'ImportError' in status.error_message or 'No module' in status.error_message

    def test_detect_class_capability(self):
        """Test detecting class-level capabilities."""
        from rlqas.phase2.adaptation import CapabilityDetector

        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_class_capability(
            'os',
            'PathLike',
            required_methods=['__fspath__'],
        )

        assert status.available is True

    def test_detect_function_signature(self):
        """Test detecting function signature capabilities."""
        from rlqas.phase2.adaptation import CapabilityDetector

        detector = CapabilityDetector(cache_results=False)
        status = detector.detect_function_signature(
            'json',
            'loads',
            required_params=['s'],
        )

        assert status.available is True

    def test_tencirchem_capability_detection(self):
        """Test Tencirchem-specific capability detection."""
        from rlqas.phase2.adaptation import CapabilityDetector

        detector = CapabilityDetector(cache_results=False)
        capabilities = detector.detect_tencirchem_capabilities()

        # Should return a dict of capabilities even if tencirchem not installed
        assert isinstance(capabilities, dict)


class TestFeatureImplementation:
    """Tests for dynamic feature implementation."""

    def test_generate_parity_adapter(self):
        """Test generating parity adapter."""
        from rlqas.phase2.adaptation import FeatureImplementer

        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)
            output_path = implementer.generate_parity_adapter(
                target_library='tencirchem',
            )

            assert os.path.exists(output_path)
            assert 'tencirchem' in output_path

            # Verify file content
            with open(output_path, 'r') as f:
                content = f.read()

            assert 'TencirchemParityAdapter' in content
            assert 'apply_parity_transformation' in content

    def test_generate_transformer_adapter(self):
        """Test generating transformer adapter."""
        from rlqas.phase2.adaptation import FeatureImplementer

        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)
            output_path = implementer.generate_transformer_adapter(
                transform_type='jordan_wigner',
            )

            assert os.path.exists(output_path)

            # Verify file content
            with open(output_path, 'r') as f:
                content = f.read()

            assert 'CustomTransformerAdapter' in content
            assert 'jordan_wigner' in content

    def test_verify_implementation(self):
        """Test verifying generated implementations."""
        from rlqas.phase2.adaptation import FeatureImplementer

        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)

            # Generate adapter
            output_path = implementer.generate_parity_adapter()

            # Verify
            verification = implementer.verify_implementation(output_path)

            assert verification['exists'] is True
            assert verification['valid_syntax'] is True

    def test_verify_nonexistent_file(self):
        """Test verifying nonexistent file."""
        from rlqas.phase2.adaptation import FeatureImplementer

        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)
            verification = implementer.verify_implementation('/nonexistent/path/file.py')

            assert verification['exists'] is False


class TestAdaptiveExecution:
    """Tests for adaptive execution flow."""

    def test_check_available_capability(self):
        """Test checking available capability."""
        from rlqas.phase2.adaptation import AdaptiveExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)
            result = executor.check_capability('os', auto_fix=False)

            assert result is True

    def test_check_unavailable_capability(self):
        """Test checking unavailable capability."""
        from rlqas.phase2.adaptation import AdaptiveExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0, auto_implement=False)
            result = executor.check_capability('nonexistent_xyz_module', auto_fix=False)

            assert result is False

    def test_execute_with_adaptation(self):
        """Test executing operation with adaptation."""
        from rlqas.phase2.adaptation import AdaptiveExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            def simple_add(x, y):
                return x + y

            result = executor.execute_with_adaptation(
                operation=simple_add,
                required_capabilities=['os'],
                x=5,
                y=3,
            )

            assert result == 8

    def test_process_molecule_workflow(self):
        """Test complete molecule processing workflow."""
        from rlqas.phase2.adaptation import AdaptiveExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            def mock_processor(molecule_data, use_parity=True):
                return {
                    'molecule': molecule_data.get('name', 'unknown'),
                    'parity_used': use_parity,
                    'energy': -1.0,
                }

            molecule_data = {
                'name': 'lih',
                'basis': 'sto-3g',
                'n_qubits': 10,
            }

            result = executor.process_molecule(
                molecule_processor=mock_processor,
                molecule_data=molecule_data,
                use_parity=True,
            )

            assert result['molecule'] == 'lih'
            assert result['parity_used'] is True

    def test_execution_log(self):
        """Test execution logging."""
        from rlqas.phase2.adaptation import AdaptiveExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            def simple_op(x, y):
                return x * y

            executor.execute_with_adaptation(
                operation=simple_op,
                required_capabilities=['json'],
                x=2,
                y=3,
            )

            log = executor.get_execution_log()

            assert len(log) > 0
            assert any(entry['action'] == 'execute' for entry in log)

    def test_save_execution_report(self):
        """Test saving execution report."""
        from rlqas.phase2.adaptation import AdaptiveExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            # Generate some log entries
            executor.check_capability('os')

            report_path = executor.save_execution_report(
                output_path=os.path.join(tmpdir, 'test_report.json'),
            )

            assert os.path.exists(report_path)

            with open(report_path, 'r') as f:
                report = json.load(f)

            assert 'generated_at' in report
            assert 'capabilities_checked' in report


class TestCapabilityRegistry:
    """Tests for capability registry."""

    def test_register_and_check_capability(self):
        """Test registering and checking capability."""
        from rlqas.phase2.adaptation import CapabilityRegistry

        registry = CapabilityRegistry()

        # Register
        success = registry.register_capability(
            'test.capability',
            'test_source',
            '1.0.0',
            validated=True,
        )

        assert success is True
        assert registry.is_registered('test.capability') is True
        assert registry.is_validated('test.capability') is True

    def test_unvalidated_capability(self):
        """Test unvalidated capability."""
        from rlqas.phase2.adaptation import CapabilityRegistry

        registry = CapabilityRegistry()

        registry.register_capability(
            'unvalidated_cap',
            'source',
            '1.0',
            validated=False,
        )

        assert registry.is_registered('unvalidated_cap') is True
        assert registry.is_validated('unvalidated_cap') is False

    def test_capability_search(self):
        """Test searching capabilities."""
        from rlqas.phase2.adaptation import CapabilityRegistry

        registry = CapabilityRegistry()

        registry.register_capability(
            'parity.adapter',
            'generated',
            '1.0',
            metadata={'type': 'adapter'},
        )

        results = registry.search_capabilities('parity')

        assert len(results) > 0
        assert results[0]['key'] == 'parity.adapter'

    def test_export_import_capabilities(self):
        """Test exporting and importing capabilities."""
        from rlqas.phase2.adaptation import CapabilityRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate registry
            registry1 = CapabilityRegistry()
            registry1.register_capability('cap1', 'source1', '1.0', validated=True)
            registry1.register_capability('cap2', 'source2', '1.0', validated=False)

            # Export
            export_path = os.path.join(tmpdir, 'capabilities.json')
            registry1.export_capabilities(export_path)

            # Import to new registry
            registry2 = CapabilityRegistry()
            count = registry2.import_capabilities(export_path)

            assert count == 2
            assert registry2.is_registered('cap1')
            assert registry2.is_registered('cap2')

    def test_persistent_storage(self):
        """Test persistent storage of capabilities."""
        from rlqas.phase2.adaptation import CapabilityRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, 'registry.json')

            # Create and register
            registry1 = CapabilityRegistry(storage_path=storage_path)
            registry1.register_capability('persist_cap', 'source', '1.0')

            # Load from storage
            registry2 = CapabilityRegistry(storage_path=storage_path)

            assert registry2.is_registered('persist_cap')


class TestRuntimeAdaptationEndToEnd:
    """End-to-end tests for runtime adaptation."""

    def test_full_adaptation_lifecycle(self):
        """Test complete adaptation lifecycle."""
        from rlqas.phase2.adaptation import (
            CapabilityDetector,
            FeatureImplementer,
            CapabilityRegistry,
            AdaptiveExecutor,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            detector = CapabilityDetector(cache_results=True)
            implementer = FeatureImplementer(output_dir=tmpdir)
            registry = CapabilityRegistry()
            executor = AdaptiveExecutor(output_dir=tmpdir, verbose=0)

            # Detect capability
            status = detector.detect_module('json')
            assert status.available

            # Register capability
            registry.register_capability(
                'json_module',
                'detected',
                '1.0',
                validated=True,
            )

            # Generate feature
            adapter_path = implementer.generate_parity_adapter()
            assert os.path.exists(adapter_path)

            # Execute with adaptation
            def test_op():
                return 'success'

            result = executor.execute_with_adaptation(
                operation=test_op,
                required_capabilities=['json_module'],
            )

            assert result == 'success'

    def test_adapter_generation_and_verification(self):
        """Test adapter generation and verification workflow."""
        from rlqas.phase2.adaptation import FeatureImplementer

        with tempfile.TemporaryDirectory() as tmpdir:
            implementer = FeatureImplementer(output_dir=tmpdir)

            # Generate parity adapter
            parity_path = implementer.generate_parity_adapter('tencirchem')

            # Generate transformer adapter
            transform_path = implementer.generate_transformer_adapter('jordan_wigner')

            # Verify both
            parity_verify = implementer.verify_implementation(parity_path)
            transform_verify = implementer.verify_implementation(transform_path)

            assert parity_verify['exists'] is True
            assert parity_verify['valid_syntax'] is True
            assert transform_verify['exists'] is True
            assert transform_verify['valid_syntax'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
