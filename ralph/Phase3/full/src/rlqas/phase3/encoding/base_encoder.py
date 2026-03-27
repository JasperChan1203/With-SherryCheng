"""Abstract base class for circuit encoders."""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class CircuitEncoder(ABC):
    """Abstract base class for circuit state encoders.

    Converts a quantum circuit representation to a fixed-size 1D numpy array
    suitable for RL agent consumption.
    """

    @abstractmethod
    def encode(self, circuit: any, n_qubits: int, max_depth: int) -> np.ndarray:
        """Encode a circuit to a fixed-size vector.

        Args:
            circuit: Circuit object (HybridCircuit, tencirchem circuit, or HEA dict)
            n_qubits: Number of qubits
            max_depth: Maximum circuit depth

        Returns:
            Fixed-size 1D numpy float32 array of length output_dim(n_qubits, max_depth)
        """

    @abstractmethod
    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        """Return the length of the encoded vector.

        Args:
            n_qubits: Number of qubits
            max_depth: Maximum circuit depth

        Returns:
            Output dimension (must equal len(encode(...)))
        """

    # Gate type mapping shared across encoders
    GATE_TYPE_MAP = {
        "identity": 0,
        "rx": 1,
        "ry": 2,
        "rz": 3,
        "cx": 4,
        "cnot": 4,
        "h": 5,
        "ucc_excitation": 6,
        "ucc_single": 7,
        "ucc_double": 8,
        "hea_layer": 9,
        "x": 10,
        "y": 11,
        "z": 12,
    }
    N_GATE_TYPES = 13

    def _circuit_to_gate_matrix(self, circuit: any, n_qubits: int, max_depth: int) -> np.ndarray:
        """Convert circuit to a (n_qubits, max_depth) gate type matrix.

        Matrix cell (q, t) = integer gate type for gate on qubit q at time step t.
        0 = identity (empty).

        Returns:
            np.ndarray of shape (n_qubits, max_depth), dtype int32
        """
        matrix = np.zeros((n_qubits, max_depth), dtype=np.int32)

        if circuit is None:
            return matrix

        # Handle HybridCircuit objects (have .blocks and .fusion_template)
        if hasattr(circuit, "blocks") and hasattr(circuit, "fusion_template"):
            t_slot = 0
            for block in circuit.blocks:
                if t_slot >= max_depth:
                    break
                b_type = block.get("type", "UCC")
                if b_type == "UCC":
                    excitations = block.get("excitations", [])
                    for exc in excitations:
                        if t_slot >= max_depth:
                            break
                        gate_type = (
                            self.GATE_TYPE_MAP["ucc_double"]
                            if len(exc) == 4
                            else self.GATE_TYPE_MAP["ucc_single"]
                        )
                        for q in exc:
                            if q < n_qubits:
                                matrix[q, t_slot] = gate_type
                        t_slot += 1
                elif b_type == "HEA":
                    circuit_dict = block.get("circuit", {})
                    layers = circuit_dict.get("layers", [])
                    for layer in layers:
                        if t_slot >= max_depth:
                            break
                        for gate_info in layer.get("rotation_gates", []):
                            q = gate_info.get("qubit", 0)
                            gtype = gate_info.get("type", "ry")
                            if q < n_qubits:
                                matrix[q, t_slot] = self.GATE_TYPE_MAP.get(gtype, 1)
                        t_slot += 1
            return matrix

        # Handle HEA circuit dict
        if isinstance(circuit, dict) and "layers" in circuit:
            t_slot = 0
            for layer in circuit["layers"]:
                if t_slot >= max_depth:
                    break
                for gate_info in layer.get("rotation_gates", []):
                    q = gate_info.get("qubit", 0)
                    gtype = gate_info.get("type", "ry")
                    if q < n_qubits:
                        matrix[q, t_slot] = self.GATE_TYPE_MAP.get(gtype, 1)
                t_slot += 1
            return matrix

        # Handle tensorcircuit.Circuit objects (direct tc.Circuit)
        # These have a _qir attribute (list of gate dicts)
        if hasattr(circuit, "_qir") and hasattr(circuit, "_nqubits"):
            t_slot = 0
            for gate_dict in circuit._qir:
                if t_slot >= max_depth:
                    break
                name = gate_dict.get("name", "identity").lower()
                qubits = gate_dict.get("index", ())
                gate_type = self.GATE_TYPE_MAP.get(name, self.GATE_TYPE_MAP.get("ucc_excitation", 6))
                for q in qubits:
                    if isinstance(q, int) and q < n_qubits:
                        matrix[q, t_slot] = gate_type
                t_slot += 1
            return matrix

        # Handle list of excitations (raw circuit spec)
        if isinstance(circuit, list):
            for t_slot, item in enumerate(circuit[:max_depth]):
                if t_slot >= max_depth:
                    break
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    gate_type = (
                        self.GATE_TYPE_MAP["ucc_double"]
                        if len(item) == 4
                        else self.GATE_TYPE_MAP["ucc_single"]
                    )
                    for q in item:
                        if isinstance(q, int) and q < n_qubits:
                            matrix[q, t_slot] = gate_type

        return matrix
