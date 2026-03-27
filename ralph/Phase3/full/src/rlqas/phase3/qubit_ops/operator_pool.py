"""Qubit-space excitation operator pool for UCC search extension.

API Investigation Results (2026-03-22):
- Tencirchem does NOT have a native QubitUCC class or Pauli string excitation pool.
- h_qubit_op attribute on UCCSD gives the molecular Hamiltonian as openfermion.QubitOperator.
- No 'qubit' or 'pauli' named attributes found in tencirchem namespace.
- Approach: Extract excitation operators from off-diagonal Hamiltonian terms (X/Y Paulis).
  These are the anti-Hermitian generators in qubit space (ADAPT-VQE qubit pool).
- Each operator is converted to a circuit via exp(-i*theta/2 * P) rotation.
"""
import numpy as np
from typing import List, Any, Dict, Optional, Tuple

from rlqas.phase1.molecule.processor import MoleculeData


class QubitOperatorPool:
    """Pool of qubit-space excitation operators (Pauli strings).

    Generates excitation operators in qubit space by extracting off-diagonal
    (X/Y-containing) terms from the Jordan-Wigner Hamiltonian. These serve as
    qubit-space excitation generators analogous to fermion-space UCCSD excitations.

    This approach follows the ADAPT-VQE qubit operator pool construction.
    """

    def __init__(self, molecule_data: MoleculeData, config: Dict = None):
        """Initialize qubit operator pool.

        Args:
            molecule_data: MoleculeData with hamiltonian (QubitOperator) and n_qubits.
            config: Configuration dict with:
                - excitation_level: 's' | 'd' | 'sd' (default: 'sd')
                - symmetry_filter: bool (default: True)
                - max_operators: int (default: 100)
        """
        self.molecule_data = molecule_data
        self.config = config or {}
        self.excitation_level = self.config.get("excitation_level", "sd")
        self.symmetry_filter = self.config.get("symmetry_filter", True)
        self.max_operators = self.config.get("max_operators", 100)
        self.n_qubits = molecule_data.n_qubits

        self._pool: List[Tuple] = []
        self._pool_coeffs: List[complex] = []
        self._build_pool()

    def _build_pool(self) -> None:
        """Build the qubit operator pool from the molecular Hamiltonian."""
        h_op = self.molecule_data.hamiltonian

        seen: set = set()
        for term, coeff in h_op.terms.items():
            if not term:
                continue  # Skip identity
            if abs(coeff) < 1e-10:
                continue

            # Only include terms with X/Y components (off-diagonal = excitations)
            pauli_chars = [p for _, p in term]
            n_xy = sum(1 for p in pauli_chars if p in ("X", "Y"))
            if n_xy == 0:
                continue

            n_paulis = len(term)

            # Filter by excitation level
            if self.excitation_level == "s" and n_paulis > 2:
                continue
            elif self.excitation_level == "d" and n_paulis <= 2:
                continue
            # "sd" accepts all

            # Deduplicate by canonical sorted key
            term_key = tuple(sorted(term, key=lambda x: x[0]))
            if term_key in seen:
                continue
            seen.add(term_key)

            self._pool.append(term)
            self._pool_coeffs.append(coeff)

            if len(self._pool) >= self.max_operators:
                break

        # Fallback: if hamiltonian has no X/Y terms, generate simple qubit excitations
        if not self._pool:
            for q in range(min(self.n_qubits, 4)):
                self._pool.append(((q, "X"),))
                self._pool_coeffs.append(1.0)
                self._pool.append(((q, "Y"),))
                self._pool_coeffs.append(1.0)

    def get_pool(self) -> List[Any]:
        """Return the list of qubit operator terms."""
        return list(self._pool)

    def get_pool_size(self) -> int:
        """Return the number of operators in the pool."""
        return len(self._pool)

    def operator_to_circuit(self, op_index: int, n_qubits: int) -> Any:
        """Convert pool operator to a parametric circuit block.

        Builds exp(-i*theta/2 * P) rotation circuit for Pauli string P.

        Args:
            op_index: Index into the pool (0 <= op_index < get_pool_size())
            n_qubits: Number of qubits in the circuit

        Returns:
            tensorcircuit.Circuit implementing the Pauli rotation
        """
        from .adapter import pauli_string_to_circuit
        term = self._pool[op_index]
        coeff = self._pool_coeffs[op_index]
        return pauli_string_to_circuit(term, coeff, n_qubits, theta=0.1)

    def get_operator_string(self, op_index: int) -> str:
        """Get human-readable string representation of an operator.

        Args:
            op_index: Index into the pool

        Returns:
            String like 'X0Y1Z2'
        """
        term = self._pool[op_index]
        parts = [f"{p}{idx}" for idx, p in term]
        return "".join(parts)
