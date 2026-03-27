"""
Molecular constants for RLQAS.

This module provides constants for molecular calculations, including
atomic numbers, basis sets, and default parameters.
"""

# Atomic numbers for common elements
ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
}

# Common basis sets
BASIS_SETS = [
    "sto-3g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "def2-svp",
    "def2-tzvp",
    "def2-qzvp",
]

# Default bond lengths (Å) for common molecules
DEFAULT_BOND_LENGTHS = {
    "H2": 0.74,
    "LiH": 1.6,
    "BeH2": 1.34,
    "H4": 1.0,  # square geometry side length
}

# Conversion factors
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KJ_MOL = 2625.499639
BOHR_TO_ANGSTROM = 0.52917721067
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM