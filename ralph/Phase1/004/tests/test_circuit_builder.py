#!/usr/bin/env python3
"""Unit tests for UCCCircuitBuilder."""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")

from src.modules.molecule_processor import MoleculeData
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder


class TestUCCCircuitBuilder(unittest.TestCase):
    """Test UCCCircuitBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock MoleculeData object
        self.molecule_data = Mock(spec=MoleculeData)
        self.molecule_data.n_qubits = 2
        self.molecule_data.molecular_info = {
            "formula": "H2",
            "bond_length_angstrom": 0.74,
            "basis_set": "sto-3g",
        }
        # Mock tencirchem.UCCSD
        self.mock_ucc = Mock()
        self.mock_ucc.ex_ops = [(3, 2), (1, 0), (1, 3, 2, 0)]
        self.mock_ucc.param_ids = [0, 1, 2]
        self.mock_ucc.param_to_ex_ops = {0: [(3, 2)], 1: [(1, 0)], 2: [(1, 3, 2, 0)]}
        self.mock_ucc.n_params = 3
        self.mock_ucc.get_circuit.return_value = Mock()
        self.mock_ucc.energy.return_value = -1.1

        # Mock pyscf objects
        self.mock_mol = Mock()
        self.mock_hf = Mock()
        self.mock_hf.kernel.return_value = -1.116759
        self.mock_hf.converged = True

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_initialization(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test circuit builder initialization."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)

        self.assertEqual(builder.n_params, 3)
        self.assertEqual(len(builder.available_excitations), 3)
        self.assertEqual(builder.available_excitations, self.mock_ucc.ex_ops)
        # Check mapping
        self.assertEqual(builder.ex_op_to_param[0], 0)
        self.assertEqual(builder.ex_op_to_param[1], 1)
        self.assertEqual(builder.ex_op_to_param[2], 2)

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_get_available_excitations(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test get_available_excitations method."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)
        excitations = builder.get_available_excitations()
        self.assertEqual(excitations, self.mock_ucc.ex_ops)

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_initialize_parameters(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test parameter initialization strategies."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)

        # Test random strategy
        np.random.seed(42)
        params = builder.initialize_parameters(3, strategy='random')
        self.assertEqual(params.shape, (3,))
        self.assertTrue(np.all(params >= -0.1) and np.all(params <= 0.1))

        # Test zeros strategy
        params = builder.initialize_parameters(3, strategy='zeros')
        self.assertTrue(np.all(params == 0.0))

        # Test normal strategy
        params = builder.initialize_parameters(3, strategy='normal')
        self.assertEqual(params.shape, (3,))

        # Test invalid strategy
        with self.assertRaises(ValueError):
            builder.initialize_parameters(3, strategy='invalid')

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_build_circuit(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test building a circuit with given excitations."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)

        # Build circuit with subset of excitations
        excitations = [(3, 2), (1, 0)]
        circuit = builder.build_circuit(excitations)

        # Verify that circuit is returned (mock)
        self.assertIsNotNone(circuit)
        # Verify that build_circuit was called on mock_ucc
        self.mock_ucc.get_circuit.assert_called_once()

        # Test with provided parameters
        params = np.array([0.1, 0.2, 0.3])
        circuit2 = builder.build_circuit(excitations, params)
        self.assertIsNotNone(circuit2)

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_build_circuit_invalid_excitations(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test building circuit with invalid excitation raises error."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)

        with self.assertRaises(ValueError):
            builder.build_circuit([(5, 4)])  # Not in available excitations

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_get_parameter_indices_for_excitation(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test mapping excitation to parameter indices."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)

        # Each excitation maps to a single parameter index
        idx = builder.get_parameter_indices_for_excitation((3, 2))
        self.assertEqual(idx, [0])

        idx = builder.get_parameter_indices_for_excitation((1, 0))
        self.assertEqual(idx, [1])

        idx = builder.get_parameter_indices_for_excitation((1, 3, 2, 0))
        self.assertEqual(idx, [2])

        # Invalid excitation
        with self.assertRaises(ValueError):
            builder.get_parameter_indices_for_excitation((5, 4))

    @patch('src.modules.ucc_search.circuit_builder.gto.M')
    @patch('src.modules.ucc_search.circuit_builder.scf.RHF')
    @patch('src.modules.ucc_search.circuit_builder.UCCSD')
    def test_evaluate_energy(self, mock_ucc_class, mock_rhf, mock_gto):
        """Test energy evaluation."""
        mock_gto.return_value = self.mock_mol
        mock_rhf.return_value = self.mock_hf
        mock_ucc_class.return_value = self.mock_ucc

        builder = UCCCircuitBuilder(self.molecule_data)

        # Mock circuit
        mock_circuit = Mock()
        params = np.array([0.1, 0.2, 0.3])

        energy = builder.evaluate_energy(mock_circuit, params)

        # Should call ucc.energy with params
        self.mock_ucc.energy.assert_called_once_with(params)
        self.assertEqual(energy, self.mock_ucc.energy.return_value)


if __name__ == '__main__':
    unittest.main()