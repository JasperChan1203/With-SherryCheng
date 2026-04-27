"""
HEA Circuit Builder for RLQAS Phase 2.

This module implements the HEACircuitBuilder class for constructing
Hardware Efficient Ansatz circuits with configurable entanglement patterns.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class HEACircuitBuilder:
    """Builder for Hardware Efficient Ansatz (HEA) circuits.

    This class provides utilities for constructing HEA circuits with
    configurable entanglement patterns, rotation gates, and parameter
    sharing strategies.

    Supported entanglement patterns:
    - linear: Nearest-neighbor entanglement (qubit i connected to i+1)
    - circular: Linear + connection between first and last qubit
    - fully_connected: All-to-all entanglement

    Supported rotation gates:
    - rx: Rotation around X axis
    - ry: Rotation around Y axis
    - rz: Rotation around Z axis

    Parameter sharing strategies:
    - none: Each gate has independent parameters
    - layer_wise: All gates in a layer share parameters
    - global: All gates share the same parameters
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        entanglement_pattern: str = "linear",
        rotation_gates: Optional[List[str]] = None,
        parameter_sharing: str = "layer_wise",
    ):
        """Initialize HEA circuit builder.

        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of layers in the HEA
            entanglement_pattern: Entanglement pattern ("linear", "circular", "fully_connected")
            rotation_gates: List of rotation gate types (default: ["rx", "ry", "rz"])
            parameter_sharing: Parameter sharing strategy ("none", "layer_wise", "global")
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement_pattern = entanglement_pattern
        self.rotation_gates = rotation_gates or ["rx", "ry", "rz"]
        self.parameter_sharing = parameter_sharing

        # Validate inputs
        self._validate_inputs()

        # Circuit representation
        self._layers: List[Dict] = []
        self._parameters: Optional[np.ndarray] = None
        self._entanglement_pairs: List[Tuple[int, int]] = []

    def _validate_inputs(self):
        """Validate input parameters."""
        valid_patterns = ["linear", "circular", "fully_connected"]
        if self.entanglement_pattern not in valid_patterns:
            raise ValueError(
                f"Invalid entanglement pattern: {self.entanglement_pattern}. "
                f"Must be one of {valid_patterns}"
            )

        valid_gates = ["rx", "ry", "rz"]
        for gate in self.rotation_gates:
            if gate not in valid_gates:
                raise ValueError(
                    f"Invalid rotation gate: {gate}. Must be one of {valid_gates}"
                )

        valid_sharing = ["none", "layer_wise", "global"]
        if self.parameter_sharing not in valid_sharing:
            raise ValueError(
                f"Invalid parameter sharing: {self.parameter_sharing}. "
                f"Must be one of {valid_sharing}"
            )

    def build(self, parameters: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Build the HEA circuit.

        Args:
            parameters: Optional parameter array. If None, random parameters
                are initialized.

        Returns:
            Dictionary containing circuit representation
        """
        self._layers = []
        self._entanglement_pairs = self._get_entanglement_pairs()

        if parameters is None:
            parameters = self._initialize_parameters()

        self._parameters = parameters

        # Build layers
        param_idx = 0
        for layer_idx in range(self.n_layers):
            layer = self._build_layer(layer_idx, param_idx)
            self._layers.append(layer)
            param_idx += layer["n_parameters"]

        return self.get_circuit_dict()

    def _get_entanglement_pairs(self) -> List[Tuple[int, int]]:
        """Get entanglement pairs based on pattern.

        Returns:
            List of (control, target) qubit pairs
        """
        pairs = []

        if self.entanglement_pattern == "linear":
            # Nearest neighbor: (0,1), (1,2), ..., (n-2, n-1)
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))

        elif self.entanglement_pattern == "circular":
            # Linear + (n-1, 0)
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))
            pairs.append((self.n_qubits - 1, 0))

        elif self.entanglement_pattern == "fully_connected":
            # All pairs
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    pairs.append((i, j))

        return pairs

    def _initialize_parameters(self) -> np.ndarray:
        """Initialize circuit parameters.

        Returns:
            Parameter array
        """
        n_params = self._count_total_parameters()
        return np.random.uniform(-np.pi, np.pi, size=n_params)

    def _count_total_parameters(self) -> int:
        """Count total number of parameters.

        Returns:
            Total parameter count
        """
        if self.parameter_sharing == "global":
            # Single parameter per gate type
            return len(self.rotation_gates)

        elif self.parameter_sharing == "layer_wise":
            # One parameter per gate type per layer
            return self.n_layers * len(self.rotation_gates)

        else:  # "none"
            # Full parameters: per qubit, per gate, per layer
            return self.n_layers * self.n_qubits * len(self.rotation_gates)

    def _build_layer(self, layer_idx: int, param_start: int) -> Dict[str, Any]:
        """Build a single layer of the HEA.

        Args:
            layer_idx: Index of the layer
            param_start: Starting index in parameter array

        Returns:
            Dictionary representing the layer
        """
        layer = {
            "index": layer_idx,
            "rotation_gates": [],
            "entanglement_gates": [],
            "parameters": [],
            "n_parameters": 0,
        }

        # Add rotation gates
        for qubit in range(self.n_qubits):
            for gate_idx, gate_type in enumerate(self.rotation_gates):
                param = self._get_parameter(layer_idx, qubit, gate_idx)
                layer["rotation_gates"].append({
                    "type": gate_type,
                    "qubit": qubit,
                    "parameter": float(param),
                })
                layer["parameters"].append(float(param))

        # Add entanglement gates
        for control, target in self._entanglement_pairs:
            layer["entanglement_gates"].append({
                "type": "CNOT",
                "control": control,
                "target": target,
            })

        layer["n_parameters"] = len(layer["parameters"])
        return layer

    def _get_parameter(
        self,
        layer_idx: int,
        qubit: int,
        gate_idx: int,
    ) -> float:
        """Get parameter value based on sharing strategy.

        Args:
            layer_idx: Layer index
            qubit: Qubit index
            gate_idx: Gate type index

        Returns:
            Parameter value
        """
        if self._parameters is None:
            return 0.0

        if self.parameter_sharing == "global":
            return float(self._parameters[gate_idx])

        elif self.parameter_sharing == "layer_wise":
            param_idx = layer_idx * len(self.rotation_gates) + gate_idx
            return float(self._parameters[param_idx])

        else:  # "none"
            param_idx = (
                layer_idx * self.n_qubits * len(self.rotation_gates)
                + qubit * len(self.rotation_gates)
                + gate_idx
            )
            return float(self._parameters[param_idx])

    def get_circuit_dict(self) -> Dict[str, Any]:
        """Get circuit as dictionary.

        Returns:
            Dictionary containing full circuit representation
        """
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "entanglement_pattern": self.entanglement_pattern,
            "rotation_gates": self.rotation_gates,
            "parameter_sharing": self.parameter_sharing,
            "entanglement_pairs": self._entanglement_pairs,
            "layers": self._layers,
            "total_parameters": len(self._parameters) if self._parameters is not None else 0,
        }

    def get_entanglement_depth(self) -> int:
        """Get the entanglement depth of the circuit.

        Returns:
            Number of entanglement pairs per layer
        """
        return len(self._entanglement_pairs)

    def get_circuit_depth(self) -> int:
        """Get the total circuit depth.

        Returns:
            Estimated circuit depth (gate layers)
        """
        if not self._layers:
            self.build()

        # Each layer has rotation gates (depth = n_rotation_gates)
        # plus entanglement gates (depth depends on pattern)
        rotation_depth = len(self.rotation_gates)
        entanglement_depth = len(self._entanglement_pairs)

        return self.n_layers * (rotation_depth + entanglement_depth)

    def set_parameters(self, parameters: np.ndarray):
        """Set circuit parameters.

        Args:
            parameters: New parameter array
        """
        expected_size = self._count_total_parameters()
        if len(parameters) != expected_size:
            raise ValueError(
                f"Expected {expected_size} parameters, got {len(parameters)}"
            )
        self._parameters = parameters.copy()
        # Rebuild circuit with new parameters
        if self._layers:
            self.build(self._parameters)

    def get_parameters(self) -> np.ndarray:
        """Get current circuit parameters.

        Returns:
            Parameter array
        """
        if self._parameters is None:
            self.build()
        return self._parameters.copy()

    def to_tensorcircuit(self) -> "tc.Circuit":
        """Convert built circuit dict to tensorcircuit.Circuit for simulation.

        Must call build() before calling this method.

        Returns:
            tensorcircuit.Circuit with all gates applied
        """
        import tensorcircuit as tc

        if not self._layers:
            self.build()

        c = tc.Circuit(self.n_qubits)

        for layer in self._layers:
            # Apply rotation gates
            for gate in layer["rotation_gates"]:
                qubit = gate["qubit"]
                angle = gate["parameter"]
                gate_type = gate["type"]
                if gate_type == "rx":
                    c.rx(qubit, theta=angle)
                elif gate_type == "ry":
                    c.ry(qubit, theta=angle)
                elif gate_type == "rz":
                    c.rz(qubit, theta=angle)

            # Apply entanglement (CNOT) gates
            for gate in layer["entanglement_gates"]:
                c.cnot(gate["control"], gate["target"])

        return c


def create_hea_circuit(
    n_qubits: int,
    n_layers: int,
    entanglement_pattern: str = "linear",
    rotation_gates: Optional[List[str]] = None,
    parameter_sharing: str = "layer_wise",
    parameters: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience function to create an HEA circuit.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        entanglement_pattern: Entanglement pattern
        rotation_gates: Rotation gate types
        parameter_sharing: Parameter sharing strategy
        parameters: Optional initial parameters

    Returns:
        Circuit dictionary
    """
    builder = HEACircuitBuilder(
        n_qubits=n_qubits,
        n_layers=n_layers,
        entanglement_pattern=entanglement_pattern,
        rotation_gates=rotation_gates,
        parameter_sharing=parameter_sharing,
    )
    return builder.build(parameters)
