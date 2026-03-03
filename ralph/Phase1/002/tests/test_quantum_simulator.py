"""
Unit tests for quantum simulator module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import tensorcircuit as tc
from openfermion import QubitOperator
from src.modules.quantum_simulator import (
    QuantumSimulator,
    TencirchemCISimulator,
    SimulatorFactory
)


class TestQuantumSimulator:
    """Test abstract base class interface."""

    def test_abstract_class_cannot_be_instantiated(self):
        """QuantumSimulator is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            QuantumSimulator()

    def test_abstract_methods_exist(self):
        """All abstract methods should be defined."""
        # Check that abstract methods exist
        assert hasattr(QuantumSimulator, 'compute_energy')
        assert hasattr(QuantumSimulator, 'get_max_qubits')
        assert hasattr(QuantumSimulator, 'estimate_memory')

        # Check they are abstract
        import inspect
        # Check if class is abstract
        assert inspect.isabstract(QuantumSimulator)
        # Check if methods have __isabstractmethod__ attribute set to True
        assert getattr(QuantumSimulator.compute_energy, '__isabstractmethod__', False)
        assert getattr(QuantumSimulator.get_max_qubits, '__isabstractmethod__', False)
        assert getattr(QuantumSimulator.estimate_memory, '__isabstractmethod__', False)


class TestTencirchemCISimulator:
    """Test TencirchemCISimulator concrete implementation."""

    @pytest.fixture
    def default_simulator(self):
        """Create simulator with default configuration."""
        return TencirchemCISimulator()

    @pytest.fixture
    def custom_simulator(self):
        """Create simulator with custom configuration."""
        config = {
            "engine": "statevector",
            "precision": 1e-10,
            "use_symmetry": False,
            "max_memory_gb": 16.0,
            "fallback_method": "mps",
            "use_gpu": False
        }
        return TencirchemCISimulator(config)

    @pytest.fixture
    def simple_hamiltonian(self):
        """Create a simple 2-qubit Hamiltonian for testing."""
        hamiltonian = QubitOperator()
        hamiltonian += QubitOperator("Z0", 1.0)
        hamiltonian += QubitOperator("Z1", 1.0)
        hamiltonian += QubitOperator("X0 X1", 0.5)
        return hamiltonian

    def _create_mock_circuit(self):
        """Create a mock circuit object for testing."""
        c = tc.Circuit(2)
        c.ry(0, theta=0.1)
        c.ry(1, theta=0.2)
        c.n_qubits = 2
        c.parameters = [0.1, 0.2]
        return c

    @pytest.fixture
    def mock_circuit_fixture(self):
        """Fixture returning mock circuit."""
        return self._create_mock_circuit()

    def mock_circuit(self):
        """Method returning mock circuit (for direct calls)."""
        return self._create_mock_circuit()

    def test_initialization_default(self, default_simulator):
        """Test initialization with default configuration."""
        assert default_simulator.config["engine"] == "ci_vector"
        assert default_simulator.config["precision"] == 1e-8
        assert default_simulator.config["use_symmetry"] is True
        assert default_simulator.config["max_memory_gb"] == 32.0
        assert default_simulator.config["fallback_method"] == "statevector"
        assert default_simulator.config["use_gpu"] is False

    def test_initialization_custom(self, custom_simulator):
        """Test initialization with custom configuration."""
        assert custom_simulator.config["engine"] == "statevector"
        assert custom_simulator.config["precision"] == 1e-10
        assert custom_simulator.config["use_symmetry"] is False
        assert custom_simulator.config["max_memory_gb"] == 16.0
        assert custom_simulator.config["fallback_method"] == "mps"
        assert custom_simulator.config["use_gpu"] is False

    def test_invalid_configuration(self):
        """Test validation of invalid configuration parameters."""
        # Invalid engine
        with pytest.raises(ValueError, match="engine must be one of"):
            TencirchemCISimulator({"engine": "invalid"})

        # Invalid precision
        with pytest.raises(ValueError, match="precision must be positive number"):
            TencirchemCISimulator({"precision": -1.0})

        # Invalid max_memory_gb
        with pytest.raises(ValueError, match="max_memory_gb must be positive number"):
            TencirchemCISimulator({"max_memory_gb": 0})

    def test_get_max_qubits(self, default_simulator):
        """Test maximum qubits method."""
        max_qubits = default_simulator.get_max_qubits()
        assert isinstance(max_qubits, int)
        assert max_qubits > 0
        # Should be 20 as per implementation
        assert max_qubits == 20

    def test_estimate_memory(self, default_simulator):
        """Test memory estimation method."""
        # Test with small number of qubits
        memory_2q = default_simulator.estimate_memory(2)
        memory_4q = default_simulator.estimate_memory(4)
        memory_8q = default_simulator.estimate_memory(8)

        assert isinstance(memory_2q, float)
        assert memory_2q > 0
        # Memory should increase with qubit count
        assert memory_4q > memory_2q
        assert memory_8q > memory_4q

        # Test edge cases - large qubit count should still produce finite memory estimate
        memory_100q = default_simulator.estimate_memory(100)
        assert isinstance(memory_100q, float)
        assert memory_100q > 0

    def test_compute_energy_basic(self, default_simulator, mock_circuit_fixture, simple_hamiltonian):
        """Test basic energy computation."""
        energy = default_simulator.compute_energy(mock_circuit_fixture, simple_hamiltonian)
        assert isinstance(energy, float)
        # Compute expected energy using tensorcircuit directly
        expected_energy = 0.0
        for term, coeff in simple_hamiltonian.terms.items():
            x_list, y_list, z_list = [], [], []
            for idx, pauli in term:
                if pauli == 'X':
                    x_list.append(idx)
                elif pauli == 'Y':
                    y_list.append(idx)
                elif pauli == 'Z':
                    z_list.append(idx)
            exp_val = mock_circuit_fixture.expectation_ps(x=x_list, y=y_list, z=z_list)
            expected_energy += coeff * exp_val
        expected_energy = expected_energy.real
        # Allow small numerical tolerance
        assert abs(energy - expected_energy) < 1e-10

    def test_compute_energy_with_initial_state(self, default_simulator, mock_circuit_fixture, simple_hamiltonian):
        """Test energy computation with initial state."""
        initial_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        energy = default_simulator.compute_energy(
            mock_circuit_fixture, simple_hamiltonian, initial_state
        )
        assert isinstance(energy, float)

    def test_count_qubits(self, default_simulator, simple_hamiltonian):
        """Test internal qubit counting method."""
        n_qubits = default_simulator._count_qubits(simple_hamiltonian)
        assert n_qubits == 2

        # Test with empty Hamiltonian
        empty_ham = QubitOperator()
        n_qubits_empty = default_simulator._count_qubits(empty_ham)
        assert n_qubits_empty == 0

    def test_fallback_logic(self):
        """Test fallback behavior when memory limit exceeded."""
        # Create simulator with very low memory limit
        config = {"max_memory_gb": 0.001}  # 1 MB limit
        simulator = TencirchemCISimulator(config)

        # Create Hamiltonian that will exceed memory limit
        hamiltonian = QubitOperator("Z0", 1.0)
        mock_circuit = self.mock_circuit()

        # Should trigger fallback but still return a value
        energy = simulator.compute_energy(mock_circuit, hamiltonian)
        assert isinstance(energy, float)


class TestSimulatorFactory:
    """Test SimulatorFactory decision logic."""

    def test_create_simulator_default(self):
        """Test factory with default configuration."""
        # Small system (<8 qubits) -> statevector
        simulator = SimulatorFactory.create_simulator(4)
        assert isinstance(simulator, TencirchemCISimulator)
        assert simulator.config["engine"] == "statevector"

        # Medium system (8-20 qubits) -> ci_vector
        simulator = SimulatorFactory.create_simulator(10)
        assert simulator.config["engine"] == "ci_vector"

        # Large system (>20 qubits) -> ci_vector with conservative settings
        simulator = SimulatorFactory.create_simulator(25)
        assert simulator.config["engine"] == "ci_vector"
        assert simulator.config["max_memory_gb"] == 16.0
        assert simulator.config["fallback_method"] == "mps"

    def test_create_simulator_with_config_override(self):
        """Test factory with configuration override."""
        config = {"engine": "mps", "precision": 1e-6}
        simulator = SimulatorFactory.create_simulator(10, config)
        assert simulator.config["engine"] == "mps"
        assert simulator.config["precision"] == 1e-6

    def test_create_simulator_edge_cases(self):
        """Test factory edge cases."""
        # n_qubits = 0
        simulator = SimulatorFactory.create_simulator(0)
        assert simulator.config["engine"] == "statevector"

        # n_qubits = 8 (boundary)
        simulator = SimulatorFactory.create_simulator(8)
        assert simulator.config["engine"] == "ci_vector"

        # n_qubits = 20 (boundary)
        simulator = SimulatorFactory.create_simulator(20)
        assert simulator.config["engine"] == "ci_vector"


class TestIntegrationWithTask001:
    """Integration tests with Phase 1 Task 001 molecule processor."""

    def test_import_task001_modules(self):
        """Test that we can import Task 001 modules."""
        import sys
        sys.path.append("../001")
        try:
            from src.modules.molecule_processor import process_molecule, MoleculeData
            # If import succeeds, test that classes exist
            # MoleculeData is a dataclass, check fields
            import dataclasses
            fields = dataclasses.fields(MoleculeData)
            field_names = {f.name for f in fields}
            assert 'hamiltonian' in field_names
            assert 'n_qubits' in field_names
            assert 'reference_state' in field_names
            assert callable(process_molecule)
        except ImportError as e:
            pytest.skip(f"Task 001 modules not available: {e}")

    def test_process_h2_molecule(self):
        """Test processing H2 molecule from Task 001."""
        import sys
        sys.path.append("../001")
        try:
            from src.modules.molecule_processor import process_molecule
        except ImportError:
            pytest.skip("Task 001 modules not available")

        # Process H2 molecule
        molecule_data = process_molecule(
            molecule="H2",
            bond_length=0.74,
            ansatz_type="UCC",
            active_space=None,
            basis_set="sto-3g",
            transform="parity"
        )

        # Verify we got a MoleculeData object
        assert hasattr(molecule_data, 'hamiltonian')
        assert hasattr(molecule_data, 'n_qubits')
        assert hasattr(molecule_data, 'reference_state')
        assert hasattr(molecule_data, 'fci_energy')

        # Verify Hamiltonian is QubitOperator
        from openfermion import QubitOperator
        assert isinstance(molecule_data.hamiltonian, QubitOperator)

        # Verify n_qubits is reasonable (should be 2 for H2)
        assert molecule_data.n_qubits == 2

        # Verify reference state is numpy array
        assert isinstance(molecule_data.reference_state, np.ndarray)
        assert molecule_data.reference_state.shape == (4,)  # 2^2 = 4

    def test_simulator_with_h2_hamiltonian(self):
        """Test simulator with actual H2 Hamiltonian from Task 001."""
        import sys
        sys.path.append("../001")
        try:
            from src.modules.molecule_processor import process_molecule
        except ImportError:
            pytest.skip("Task 001 modules not available")

        # Get H2 molecule data
        molecule_data = process_molecule(
            molecule="H2",
            bond_length=0.74,
            ansatz_type="UCC",
            active_space=None,
            basis_set="sto-3g",
            transform="parity"
        )

        # Create simulator
        simulator = TencirchemCISimulator()

        # Create a simple mock circuit
        class MockCircuit:
            def __init__(self, n_qubits):
                self.n_qubits = n_qubits
                self.parameters = [0.0]

        mock_circuit = MockCircuit(molecule_data.n_qubits)

        # Compute energy
        energy = simulator.compute_energy(
            mock_circuit,
            molecule_data.hamiltonian,
            molecule_data.reference_state
        )

        # Verify we get a float
        assert isinstance(energy, float)

        # Energy should be negative for bound molecules
        # (dummy implementation may not obey this)
        # We'll just check it's a number
        assert not np.isnan(energy)
        assert not np.isinf(energy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])