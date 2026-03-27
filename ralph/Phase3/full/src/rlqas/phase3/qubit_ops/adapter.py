"""Adapter for qubit-space excitation operators.

Converts Pauli string excitation operators to tensorcircuit circuits via
the standard exp(-i*theta/2 * P) rotation decomposition.
"""
import numpy as np
from typing import Any, Tuple


def pauli_string_to_circuit(
    pauli_term: Tuple,
    coefficient: complex,
    n_qubits: int,
    theta: float = 0.1,
) -> Any:
    """Convert a Pauli string term to a parametric circuit block.

    Creates exp(-i*theta/2 * P) rotation where P is the Pauli string.
    Standard decomposition:
    1. Basis rotations: X->H, Y->Rx(pi/2)
    2. CNOT ladder to compute parity
    3. Rz(theta) on last qubit
    4. Reverse CNOT ladder
    5. Reverse basis rotations

    Args:
        pauli_term: Tuple of (qubit_idx, pauli_char) pairs, e.g. ((0,'X'),(1,'Y'))
        coefficient: Term coefficient (used for sign convention)
        n_qubits: Number of qubits in the circuit
        theta: Rotation angle parameter

    Returns:
        tensorcircuit.Circuit implementing the Pauli rotation
    """
    import tensorcircuit as tc
    c = tc.Circuit(n_qubits)

    if not pauli_term:
        return c

    # Sort by qubit index for canonical ordering
    sorted_term = sorted(pauli_term, key=lambda x: x[0])
    qubits = [idx for idx, _ in sorted_term]

    # Step 1: Basis change rotations
    for idx, pauli_char in sorted_term:
        if pauli_char == "X":
            c.h(idx)
        elif pauli_char == "Y":
            c.rx(idx, theta=np.pi / 2)
        # Z: no basis change needed

    # Step 2: CNOT ladder (XOR parity into last qubit)
    for i in range(len(qubits) - 1):
        c.cnot(qubits[i], qubits[i + 1])

    # Step 3: Rz rotation on last qubit
    if qubits:
        c.rz(qubits[-1], theta=theta)

    # Step 4: Reverse CNOT ladder
    for i in range(len(qubits) - 2, -1, -1):
        c.cnot(qubits[i], qubits[i + 1])

    # Step 5: Reverse basis change rotations
    for idx, pauli_char in sorted_term:
        if pauli_char == "X":
            c.h(idx)
        elif pauli_char == "Y":
            c.rx(idx, theta=-np.pi / 2)

    return c


def circuit_to_str(c: Any) -> str:
    """Get a meaningful string representation of a circuit.

    Args:
        c: tensorcircuit.Circuit object

    Returns:
        Descriptive string with gate counts
    """
    import tensorcircuit as tc

    if isinstance(c, tc.Circuit):
        try:
            summary = c.gate_summary()
            parts = [f"{g}:{n}" for g, n in summary.items()]
            return (
                f"Circuit(n_qubits={c._nqubits}, "
                f"n_gates={len(c._qir)}, "
                f"gates=[{', '.join(parts)}])"
            )
        except Exception:
            return f"Circuit(n_qubits={getattr(c, '_nqubits', '?')})"
    return str(c)
