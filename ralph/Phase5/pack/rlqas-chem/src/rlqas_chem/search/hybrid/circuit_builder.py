"""Hybrid circuit builder combining HEA and UCC sub-circuits."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import json

from rlqas_chem.molecule.processor import MoleculeData
from rlqas_chem.search.ucc.circuit_builder import UCCCircuitBuilder
from rlqas_chem.search.hea.circuit_builder import HEACircuitBuilder


@dataclass
class HybridCircuit:
    """Container for a hybrid HEA+UCC circuit."""
    blocks: List[Dict[str, Any]]
    n_qubits: int
    ucc: Any = None          # tencirchem UCCSD object
    ucc_params: Optional[np.ndarray] = None   # UCC parameters
    hea_blocks: List[Dict] = field(default_factory=list)  # HEA block metadata
    fusion_template: List[str] = field(default_factory=list)
    params: Optional[np.ndarray] = None  # combined params for simulator compatibility

    @property
    def num_qubits(self) -> int:
        return self.n_qubits

    def __str__(self) -> str:
        blocks_str = ", ".join(
            f"{b['type']}(depth={b.get('depth', '?')})" for b in self.blocks
        )
        n_p = len(self.params) if self.params is not None else 0
        return (
            f"HybridCircuit(n_qubits={self.n_qubits}, "
            f"n_blocks={len(self.blocks)}, "
            f"template={self.fusion_template}, "
            f"n_params={n_p}, "
            f"blocks=[{blocks_str}])"
        )

    def __repr__(self) -> str:
        return self.__str__()


class HybridFusionStrategy:
    """Strategy for fusing HEA and UCC sub-circuits.

    Supports three fusion modes:
    - sequential: HEA blocks alternate with UCC blocks
    - parallel: Combined HEA_UCC blocks run simultaneously
    - conditional: UCC block added only conditionally
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.fusion_mode = self.config.get("fusion_mode", "sequential")
        self.min_ucc_components = self.config.get("min_ucc_components", 1)
        self.max_ucc_components = self.config.get("max_ucc_components", 5)
        self.hea_layers_per_block = self.config.get("hea_layers_per_block", 2)

    def generate_fusion_template(self) -> List[str]:
        """Generate ordered list of block types.

        sequential: ["HEA", "UCC", "HEA"]
        parallel:   ["HEA_UCC", "HEA_UCC"]
        conditional: ["HEA", "UCC"]  (UCC block added only if energy improves)
        Respects min_ucc_components and max_ucc_components bounds.
        """
        n_ucc = max(self.min_ucc_components, 1)
        n_ucc = min(n_ucc, self.max_ucc_components)

        if self.fusion_mode == "sequential":
            template = ["HEA"]
            for _ in range(n_ucc):
                template.extend(["UCC", "HEA"])
            return template
        elif self.fusion_mode == "parallel":
            return ["HEA_UCC"] * n_ucc
        elif self.fusion_mode == "conditional":
            return ["HEA", "UCC"]
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode!r}")

    def fuse_circuits(self, hea_circuit: Dict, ucc_circuit: Any) -> Dict:
        """Compose HEA and UCC sub-circuits according to fusion template."""
        return {
            "type": "fused",
            "hea": hea_circuit,
            "ucc": ucc_circuit,
            "fusion_mode": self.fusion_mode,
        }


class HybridCircuitBuilder:
    """Builds hybrid HEA+UCC circuits for the hybrid architecture search."""

    def __init__(
        self,
        molecule_data: MoleculeData,
        fusion_strategy: HybridFusionStrategy,
        config: Dict = None,
    ):
        self.molecule_data = molecule_data
        self.fusion_strategy = fusion_strategy
        self.config = config or {}
        self.n_qubits = molecule_data.n_qubits

        # UCC circuit builder (handles excitation operators)
        self.ucc_builder = UCCCircuitBuilder(molecule_data, config)
        self._n_ucc_params = self.ucc_builder.n_params
        self._n_ucc_excitations = len(self.ucc_builder.available_excitations)

    def build_block(self, block_type: str, block_spec: Dict) -> Dict[str, Any]:
        """Build a single block of the given type.

        Args:
            block_type: 'HEA' | 'UCC' | 'HEA_UCC'
            block_spec: Configuration for this block

        Returns:
            Block dictionary with type, circuit data, and metadata
        """
        if block_type == "UCC":
            return self._build_ucc_block(block_spec)
        elif block_type == "HEA":
            return self._build_hea_block(block_spec)
        elif block_type == "HEA_UCC":
            hea_spec = block_spec.get("hea_spec", {})
            ucc_spec = block_spec.get("ucc_spec", {})
            hea_block = self._build_hea_block(hea_spec)
            ucc_block = self._build_ucc_block(ucc_spec)
            return {
                "type": "HEA_UCC",
                "hea": hea_block,
                "ucc": ucc_block,
                "depth": hea_block.get("depth", 0) + ucc_block.get("depth", 0),
            }
        else:
            raise ValueError(f"Unknown block_type: {block_type!r}")

    def _build_ucc_block(self, spec: Dict) -> Dict[str, Any]:
        """Build a UCC excitation block."""
        excitation_indices = spec.get("excitations", [])
        available = self.ucc_builder.available_excitations

        actual_excitations = []
        for idx in excitation_indices:
            if isinstance(idx, int) and 0 <= idx < len(available):
                actual_excitations.append(available[idx])
            elif isinstance(idx, (tuple, list)):
                t = tuple(idx)
                if t in available:
                    actual_excitations.append(t)

        params = np.zeros(self._n_ucc_params, dtype=np.float32)
        if actual_excitations:
            circuit = self.ucc_builder.build_circuit(actual_excitations, params)
        else:
            # Empty UCC block: build zero-excitation circuit
            circuit = self.ucc_builder.build_circuit([], params)

        return {
            "type": "UCC",
            "circuit": circuit,
            "excitations": actual_excitations,
            "params": params,
            "depth": len(actual_excitations),
        }

    def _build_hea_block(self, spec: Dict) -> Dict[str, Any]:
        """Build an HEA rotation block."""
        n_layers = spec.get(
            "n_layers", self.fusion_strategy.hea_layers_per_block
        )
        entanglement = spec.get("entanglement_pattern", "linear")
        rotation_gates = spec.get("rotation_gates", ["rx", "ry", "rz"])
        parameter_sharing = spec.get("parameter_sharing", "layer_wise")

        hea_builder = HEACircuitBuilder(
            n_qubits=self.n_qubits,
            n_layers=n_layers,
            entanglement_pattern=entanglement,
            rotation_gates=rotation_gates,
            parameter_sharing=parameter_sharing,
        )
        circuit_dict = hea_builder.build()

        return {
            "type": "HEA",
            "circuit": circuit_dict,
            "hea_builder": hea_builder,
            "n_layers": n_layers,
            "entanglement_pattern": entanglement,
            "rotation_gates": rotation_gates,
            "depth": hea_builder.get_circuit_depth(),
            "n_params": len(hea_builder.get_parameters()),
        }

    def build_hybrid_circuit(
        self, template: List[str], block_specs: List[Dict]
    ) -> HybridCircuit:
        """Build a complete hybrid circuit from a template and block specs.

        Args:
            template: List of block types, e.g. ['HEA', 'UCC', 'HEA']
            block_specs: List of block configuration dicts (one per template entry)

        Returns:
            HybridCircuit object
        """
        if len(template) != len(block_specs):
            raise ValueError(
                f"template length ({len(template)}) != block_specs length ({len(block_specs)})"
            )

        blocks = []
        ucc_object = None
        hea_block_list = []
        total_depth = 0

        for block_type, spec in zip(template, block_specs):
            block = self.build_block(block_type, spec)
            blocks.append(block)
            total_depth += block.get("depth", 0)

            if block_type == "UCC":
                if ucc_object is None:
                    circuit = block.get("circuit")
                    if circuit is not None and hasattr(circuit, "ucc"):
                        ucc_object = circuit.ucc
                hea_block_list.append(None)  # placeholder

            elif block_type == "HEA":
                hea_block_list.append(block)

            elif block_type == "HEA_UCC":
                ucc_sub = block.get("ucc", {})
                if ucc_object is None:
                    circuit = ucc_sub.get("circuit")
                    if circuit is not None and hasattr(circuit, "ucc"):
                        ucc_object = circuit.ucc
                hea_block_list.append(block.get("hea"))

        # Fall back to ucc_builder's UCCSD object if no UCC block added yet
        if ucc_object is None:
            ucc_object = self.ucc_builder.ucc

        # Build combined parameters: [ucc_params | hea_params_1 | hea_params_2 | ...]
        ucc_params = np.zeros(self._n_ucc_params, dtype=np.float32)
        hea_params_parts = []
        for hea_blk in hea_block_list:
            if hea_blk is not None and "hea_builder" in hea_blk:
                hea_params_parts.append(hea_blk["hea_builder"].get_parameters())

        if hea_params_parts:
            combined_hea = np.concatenate(hea_params_parts).astype(np.float32)
            combined_params = np.concatenate([ucc_params, combined_hea])
        else:
            combined_params = ucc_params

        return HybridCircuit(
            blocks=blocks,
            n_qubits=self.n_qubits,
            ucc=ucc_object,
            ucc_params=ucc_params.copy(),
            hea_blocks=[b for b in hea_block_list if b is not None],
            fusion_template=list(template),
            params=combined_params,
        )

    def save_fusion_config(
        self, path: str, template: List[str], block_specs: List[Dict]
    ) -> None:
        """Save fusion config to JSON for reproducibility."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            return obj

        data = {
            "template": template,
            "block_specs": _convert(block_specs),
            "n_qubits": self.n_qubits,
            "fusion_mode": self.fusion_strategy.fusion_mode,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_fusion_config(self, path: str) -> Tuple[List[str], List[Dict]]:
        """Load fusion config from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return data["template"], data["block_specs"]
