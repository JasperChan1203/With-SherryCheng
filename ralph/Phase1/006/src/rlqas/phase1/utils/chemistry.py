"""
Chemistry utilities for RLQAS.

This module provides utility functions for quantum chemistry calculations,
including unit conversions, symmetry detection, and molecular geometry.
"""

import numpy as np
from typing import List, Tuple, Optional


def atoms_from_formula(formula: str, geometry: str = "linear") -> List[Tuple[str, float, float, float]]:
    """Generate atom coordinates from molecular formula and geometry.

    Args:
        formula: Molecular formula, e.g., 'H2', 'LiH', 'BeH2'.
        geometry: Geometry type: 'linear', 'square' (for H4).

    Returns:
        List of (element, x, y, z) tuples.

    Raises:
        ValueError: If formula not supported.
    """
    formula = formula.strip()
    if formula == "H2":
        return [("H", 0.0, 0.0, 0.0), ("H", 1.0, 0.0, 0.0)]
    elif formula == "LiH":
        return [("Li", 0.0, 0.0, 0.0), ("H", 1.0, 0.0, 0.0)]
    elif formula == "BeH2":
        # Linear H-Be-H
        return [
            ("H", -1.0, 0.0, 0.0),
            ("Be", 0.0, 0.0, 0.0),
            ("H", 1.0, 0.0, 0.0)
        ]
    elif formula == "H4":
        if geometry == "square":
            side = 1.0
            return [
                ("H", 0.0, 0.0, 0.0),
                ("H", side, 0.0, 0.0),
                ("H", side, side, 0.0),
                ("H", 0.0, side, 0.0)
            ]
        else:
            raise ValueError(f"Unsupported geometry for H4: {geometry}")
    else:
        raise ValueError(f"Unsupported formula: {formula}")


def hartree_to_mha(energy_hartree: float) -> float:
    """Convert energy from Hartree to milliHartree (mHa)."""
    return energy_hartree * 1000.0


def mha_to_hartree(energy_mha: float) -> float:
    """Convert energy from milliHartree (mHa) to Hartree."""
    return energy_mha / 1000.0


def is_chemical_accuracy_achieved(energy_error_hartree: float, threshold_mha: float = 1.6) -> bool:
    """Check if energy error meets chemical accuracy threshold.

    Args:
        energy_error_hartree: Energy difference in Hartree.
        threshold_mha: Chemical accuracy threshold in milliHartree.

    Returns:
        True if |error| < threshold.
    """
    error_mha = abs(hartree_to_mha(energy_error_hartree))
    return error_mha < threshold_mha


def estimate_memory_usage(n_qubits: int, method: str = "statevector") -> float:
    """Estimate memory usage for quantum simulation.

    Args:
        n_qubits: Number of qubits.
        method: Simulation method: 'statevector', 'ci_vector', 'mps'.

    Returns:
        Estimated memory usage in gigabytes (GB).

    Note:
        These are rough estimates; actual memory usage depends on implementation.
    """
    if method == "statevector":
        # Complex double array of size 2^n
        bytes_needed = 2 ** n_qubits * 16  # 16 bytes per complex double
    elif method == "ci_vector":
        # CI vector size depends on number of determinants
        # Rough estimate: full CI space size = C(n_qubits, n_elec) for half-filling
        # We'll use approximate 2^(n_qubits/2) * scaling factor
        bytes_needed = 2 ** (n_qubits // 2) * 100  # rough
    elif method == "mps":
        # MPS bond dimension grows with qubits; assume moderate bond dimension
        bytes_needed = n_qubits * 1000 * 1000  # 1MB per site
    else:
        raise ValueError(f"Unknown method: {method}")

    return bytes_needed / (1024 ** 3)  # Convert to GB