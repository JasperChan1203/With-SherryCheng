"""UCC circuit builder for constructing quantum circuits from excitation sequences."""

import sys
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")

from src.modules.molecule_processor import MoleculeData
import tencirchem
from tencirchem import UCCSD
from pyscf import gto, scf


class UCCCircuitBuilder:
    """Builds UCC quantum circuits from excitation sequences.

    This class uses tencirchem.UCCSD to generate circuits for a given molecule.
    It supports selecting subsets of excitation operators and parameter initialization.
    """

    def __init__(self, molecule_data: MoleculeData, config: Dict[str, Any] = None):
        """Initialize circuit builder.

        Args:
            molecule_data: MoleculeData object from Task 001
            config: Builder configuration
        """
        self.molecule_data = molecule_data
        self.config = config or {}

        # Reconstruct PySCF molecule from molecular_info
        # Note: This assumes molecular_info contains necessary data to reconstruct
        # the molecule. This is a simplification; in production, we would store
        # the PySCF molecule object or integrals directly.
        mol_info = molecule_data.molecular_info
        formula = mol_info.get("formula", "H2")
        bond_length = mol_info.get("bond_length_angstrom", 0.74)
        basis_set = mol_info.get("basis_set", "sto-3g")

        # Parse formula (simple support for H2, LiH, BeH2)
        if formula == "H2":
            atoms = [("H", 0.0, 0.0, 0.0), ("H", bond_length, 0.0, 0.0)]
        elif formula == "LiH":
            atoms = [("Li", 0.0, 0.0, 0.0), ("H", bond_length, 0.0, 0.0)]
        elif formula == "BeH2":
            atoms = [
                ("H", -bond_length, 0.0, 0.0),
                ("Be", 0.0, 0.0, 0.0),
                ("H", bond_length, 0.0, 0.0),
            ]
        else:
            raise ValueError(f"Unsupported molecule formula: {formula}")

        # Build PySCF molecule
        mol = gto.M(
            atom=atoms,
            basis=basis_set,
            unit="angstrom",
            symmetry=False,
            verbose=0,
        )

        # Perform Hartree-Fock calculation (should match Task 001)
        hf = scf.RHF(mol)
        hf.conv_tol = 1e-8
        hf.conv_tol_grad = 1e-8
        hf.max_cycle = 1000
        hf_energy = hf.kernel()
        if not hf.converged:
            raise RuntimeError("Hartree-Fock calculation did not converge")

        # Create UCCSD instance with full excitation operator set
        self.ucc = UCCSD(mol)

        # Precompute mappings
        self.ex_ops = self.ucc.ex_ops  # list of excitation tuples
        self.param_ids = self.ucc.param_ids  # parameter index for each excitation
        self.param_to_ex_ops = self.ucc.param_to_ex_ops  # mapping param -> list of ex ops

        # Build inverse mapping: excitation operator index -> parameter index
        self.ex_op_to_param = {}
        for param_idx, ex_op_list in self.param_to_ex_ops.items():
            for ex_op in ex_op_list:
                ex_op_idx = self.ex_ops.index(ex_op)
                self.ex_op_to_param[ex_op_idx] = param_idx

        # Number of parameters
        self.n_params = self.ucc.n_params

        # Available excitations as list of tuples (same as ex_ops)
        self.available_excitations = self.ex_ops

    def build_circuit(self, excitations: List[Tuple[int, int]],
                      params: Optional[np.ndarray] = None) -> Any:
        """Build UCC circuit from excitation sequence.

        Args:
            excitations: List of excitation tuples (i,j) or (i,j,k,l)
                         These must be subsets of self.available_excitations.
            params: Circuit parameters (if None, initialize based on config).
                    Length must equal self.n_params.

        Returns:
            Parameterized quantum circuit (tencirchem-compatible)
        """
        # Validate excitations are subset of available excitations
        for exc in excitations:
            if exc not in self.available_excitations:
                raise ValueError(f"Excitation {exc} not in available excitations")

        # Determine which parameters are active based on excitations
        active_param_indices = set()
        for exc in excitations:
            ex_op_idx = self.available_excitations.index(exc)
            param_idx = self.ex_op_to_param[ex_op_idx]
            active_param_indices.add(param_idx)

        # Initialize parameters if not provided
        if params is None:
            init_strategy = self.config.get("param_init_strategy", "random")
            params = self.initialize_parameters(self.n_params, strategy=init_strategy)

            # Set parameters for inactive excitations to zero
            for param_idx in range(self.n_params):
                if param_idx not in active_param_indices:
                    params[param_idx] = 0.0
        else:
            if len(params) != self.n_params:
                raise ValueError(
                    f"Params length {len(params)} does not match n_params {self.n_params}"
                )

        # Build circuit with current parameters
        # Note: The UCCSD circuit uses all excitation operators, but parameters
        # for inactive excitations are set to zero, effectively removing them.
        circuit = self.ucc.get_circuit()
        # The circuit is parameterized; we need to bind parameters.
        # In tencirchem, the circuit is a tensorcircuit.Circuit that expects
        # parameters via `circuit.set_params(params)`.
        # Let's check if circuit has set_params method.
        if hasattr(circuit, "set_params"):
            circuit.set_params(params)
        else:
            # If not, we assume the circuit is already parameterized and
            # the energy evaluation will use params via ucc.energy(params)
            # For now, we store params in circuit object for later use.
            circuit.params = params

        return circuit

    def get_available_excitations(self) -> List[Tuple[int, int]]:
        """Get list of available excitation operators.

        Returns:
            List of excitation tuples
        """
        return self.available_excitations

    def initialize_parameters(self, n_params: int, strategy: str = 'random') -> np.ndarray:
        """Initialize circuit parameters.

        Args:
            n_params: Number of parameters needed (must equal self.n_params)
            strategy: Initialization strategy ('random', 'zeros', 'normal')

        Returns:
            Initial parameter values
        """
        # Allow any n_params; useful for initializing subsets of parameters

        if strategy == 'random':
            return np.random.uniform(-0.1, 0.1, size=n_params)
        elif strategy == 'zeros':
            return np.zeros(n_params)
        elif strategy == 'normal':
            return np.random.normal(0, 0.1, size=n_params)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def get_parameter_indices_for_excitation(self, excitation: Tuple[int, int]) -> List[int]:
        """Get parameter indices associated with an excitation operator.

        Args:
            excitation: Excitation tuple

        Returns:
            List of parameter indices (usually one)
        """
        if excitation not in self.available_excitations:
            raise ValueError(f"Excitation {excitation} not available")
        ex_op_idx = self.available_excitations.index(excitation)
        param_idx = self.ex_op_to_param[ex_op_idx]
        return [param_idx]

    def evaluate_energy(self, circuit: Any, params: np.ndarray) -> float:
        """Evaluate energy for a circuit with given parameters.

        Args:
            circuit: Circuit object (from build_circuit)
            params: Parameter values

        Returns:
            Energy in Hartree
        """
        # Use the underlying UCCSD object to compute energy
        return self.ucc.energy(params)