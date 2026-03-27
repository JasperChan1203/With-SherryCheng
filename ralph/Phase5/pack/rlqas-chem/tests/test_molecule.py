"""Tests for molecule processing module."""
import pytest
from rlqas_chem.molecule import process_molecule, MoleculeData


def test_h2_ucc():
    mol = process_molecule('H2', 0.74, 'UCC')
    assert isinstance(mol, MoleculeData)
    assert mol.n_qubits == 4
    assert mol.fci_energy < -1.0
    assert mol.hamiltonian is not None


def test_lih_ucc_active_space():
    mol = process_molecule('LiH', 1.6, 'UCC', active_space=(2, 5))
    assert mol.n_qubits == 10
    assert mol.fci_energy < -7.0


def test_beh2_hea():
    mol = process_molecule('BeH2', 1.34, 'HEA')
    assert mol.n_qubits > 0
    assert mol.fci_energy < 0


def test_invalid_molecule():
    with pytest.raises(ValueError):
        process_molecule('XYZ', 1.0, 'UCC')


def test_invalid_bond_length():
    with pytest.raises(ValueError):
        process_molecule('H2', -1.0, 'UCC')
