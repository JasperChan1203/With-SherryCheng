"""UCC search environment for quantum circuit architecture search."""

import sys
import numpy as np
import gym
from typing import Tuple, Dict, Any, List, Optional

# Add Task directories to Python path
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")

from src.modules.molecule_processor import MoleculeData, process_molecule
from src.modules.quantum_simulator import SimulatorFactory
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder
from src.modules.ucc_search.reward_function import UCCRewardFunction
from src.modules.ucc_search.config import UCCSearchConfig


class UCCSearchEnv(gym.Env):
    """UCC architecture search environment.

    This environment implements a Gym-compatible interface for searching
    UCC quantum circuit architectures. The agent selects excitation operators
    to add to the circuit, and receives rewards based on energy improvement.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, molecule_data: MoleculeData, config: Dict[str, Any] = None):
        """Initialize environment with molecule data and configuration.

        Args:
            molecule_data: MoleculeData object from Task 001
            config: Environment configuration dictionary
        """
        super().__init__()

        self.molecule_data = molecule_data
        self.config = UCCSearchConfig(config).get_section("environment")

        # Initialize components
        self.circuit_builder = UCCCircuitBuilder(molecule_data, config)
        self.reward_function = UCCRewardFunction(config)
        self.simulator = SimulatorFactory.create_simulator(molecule_data.n_qubits)

        # Get available excitations
        self.available_excitations = self.circuit_builder.get_available_excitations()
        self.n_actions = len(self.available_excitations)

        # Get total number of parameters and mapping from excitation to parameter index
        self.n_params = self.circuit_builder.n_params
        self.excitation_to_param_idx = {}
        for exc in self.available_excitations:
            param_indices = self.circuit_builder.get_parameter_indices_for_excitation(exc)
            self.excitation_to_param_idx[exc] = param_indices[0]  # take first parameter index

        # Active parameters mask
        self.active_parameters = np.zeros(self.n_params, dtype=bool)

        # Environment state
        self.current_excitations: List[Tuple[int, int]] = []
        self.current_params: Optional[np.ndarray] = None
        self.current_energy: Optional[float] = None
        self.best_energy: Optional[float] = None
        self.global_best_energy: Optional[float] = None
        self.global_best_excitations: List[Tuple[int, int]] = []
        self.global_best_params: Optional[np.ndarray] = None
        self.step_count = 0
        self.done = False

        # Set up action and observation spaces
        self._setup_spaces()

        # Initialize state
        self.reset()

    def _get_hf_energy(self) -> float:
        """Get Hartree-Fock energy from molecule data."""
        return self.molecule_data.molecular_info.get("hf_energy", 0.0)

    def _setup_spaces(self):
        """Set up action and observation spaces."""
        # Action space: discrete actions corresponding to excitation operators
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # Observation space: vector with components:
        # 1. Current energy (normalized relative to Hartree-Fock)
        # 2. Circuit parameters (normalized rotation angles, up to max_depth)
        # 3. Circuit architecture (one-hot encoding of excitation operators)
        # 4. Step count (normalized)
        max_depth = self.config.get("max_depth", 10)
        max_excitations = self.config.get("max_excitations", 20)

        # Energy component (1)
        # Assuming energy range: Hartree-Fock to FCI (or lower)
        energy_low = np.array([-10.0], dtype=np.float32)  # Conservative lower bound
        energy_high = np.array([10.0], dtype=np.float32)   # Conservative upper bound

        # Circuit parameters (max_depth)
        param_low = np.full(max_depth, -np.pi, dtype=np.float32)
        param_high = np.full(max_depth, np.pi, dtype=np.float32)

        # Architecture encoding (max_excitations one-hot)
        arch_low = np.zeros(max_excitations, dtype=np.float32)
        arch_high = np.ones(max_excitations, dtype=np.float32)

        # Step count (normalized)
        step_low = np.array([0.0], dtype=np.float32)
        step_high = np.array([1.0], dtype=np.float32)

        # Combine all components
        low = np.concatenate([energy_low, param_low, arch_low, step_low])
        high = np.concatenate([energy_high, param_high, arch_high, step_high])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action in environment.

        Args:
            action: Action index corresponding to excitation operator

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode has already terminated")

        # Check if action is valid
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Convert action to excitation operator
        excitation = self.available_excitations[action]

        # Check for duplicate excitation (invalid action)
        if excitation in self.current_excitations:
            # Duplicate excitation: treat as invalid action, give negative reward and terminate
            self.done = True
            info = {
                "error": "duplicate_excitation",
                "termination_reason": "invalid_action",
                "energy": self.current_energy,
                "best_energy": self.best_energy,
                "excitations": self.current_excitations.copy(),
                "params": self.current_params.copy() if self.current_params is not None else None,
                "step": self.step_count,
            }
            return self._get_observation(), -10.0, True, info

        # Check max depth and max excitations
        max_depth = self.config.get("max_depth", 10)
        max_excitations = self.config.get("max_excitations", 20)
        if len(self.current_excitations) >= max_depth or len(self.current_excitations) >= max_excitations:
            self.done = True
            info = {
                "error": "max_depth_exceeded",
                "termination_reason": "max_depth_exceeded",
                "energy": self.current_energy,
                "best_energy": self.best_energy,
                "excitations": self.current_excitations.copy(),
                "params": self.current_params.copy() if self.current_params is not None else None,
                "step": self.step_count,
            }
            return self._get_observation(), -10.0, True, info

        # Add excitation to current circuit
        self.current_excitations.append(excitation)

        # Get parameter index for this excitation
        param_idx = self.excitation_to_param_idx[excitation]

        # If parameter not already active, activate it and initialize with random value
        if not self.active_parameters[param_idx]:
            init_strategy = self.config.get("param_init_strategy", "random")
            random_val = self.circuit_builder.initialize_parameters(1, strategy=init_strategy)[0]
            self.current_params[param_idx] = random_val
            self.active_parameters[param_idx] = True

        # Build circuit
        circuit = self.circuit_builder.build_circuit(
            self.current_excitations, self.current_params
        )

        # Evaluate energy using circuit builder (consistent mapping)
        try:
            self.current_energy = self.circuit_builder.evaluate_energy(
                circuit, self.current_params
            )
        except Exception as e:
            # Energy evaluation failure: treat as terminal state with negative reward
            self.done = True
            info = {
                "error": str(e),
                "termination_reason": "simulator_failure",
                "energy": self.current_energy,
                "best_energy": self.best_energy,
                "excitations": self.current_excitations.copy(),
                "params": self.current_params.copy() if self.current_params is not None else None,
                "step": self.step_count,
            }
            return self._get_observation(), -10.0, True, info

        # Compute reward
        circuit_complexity = len(self.current_excitations)
        reward = self.reward_function.compute_reward(self.current_energy, circuit_complexity)

        # Update best energy
        if self.best_energy is None or self.current_energy < self.best_energy:
            self.best_energy = self.current_energy
            # Update global best if improved
            if self.global_best_energy is None or self.current_energy < self.global_best_energy:
                self.global_best_energy = self.current_energy
                self.global_best_excitations = self.current_excitations.copy()
                self.global_best_params = self.current_params.copy() if self.current_params is not None else None

        # Increment step count
        self.step_count += 1

        # Check termination conditions
        self.done = self._check_termination()

        # Prepare info dictionary
        info = {
            "excitations": self.current_excitations.copy(),
            "params": self.current_params.copy() if self.current_params is not None else None,
            "energy": self.current_energy,
            "best_energy": self.best_energy,
            "step": self.step_count,
            "termination_reason": self._get_termination_reason() if self.done else None,
        }

        return self._get_observation(), reward, self.done, info

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        # Reset circuit state
        self.current_excitations = []
        self.current_params = np.zeros(self.n_params, dtype=np.float32)
        self.active_parameters = np.zeros(self.n_params, dtype=bool)
        self.current_energy = self._get_hf_energy()
        self.best_energy = self._get_hf_energy()
        self.global_best_energy = self._get_hf_energy()
        self.global_best_excitations = []
        self.global_best_params = np.zeros(self.n_params, dtype=np.float32)
        self.step_count = 0
        self.done = False

        # Reset reward function baseline
        self.reward_function.update_baseline(self._get_hf_energy())

        return self._get_observation()

    def render(self, mode: str = 'human'):
        """Render environment state.

        Args:
            mode: Rendering mode ('human' or 'ansi')
        """
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Circuit depth: {len(self.current_excitations)}")
            print(f"Current energy: {self.current_energy:.6f} Hartree")
            print(f"Best energy: {self.best_energy:.6f} Hartree")
            print(f"Excitations: {self.current_excitations}")
            if self.current_params is not None:
                print(f"Parameters: {self.current_params}")
        elif mode == 'ansi':
            # Return string representation
            lines = [
                f"Step: {self.step_count}",
                f"Circuit depth: {len(self.current_excitations)}",
                f"Current energy: {self.current_energy:.6f} Hartree",
                f"Best energy: {self.best_energy:.6f} Hartree",
                f"Excitations: {self.current_excitations}",
            ]
            if self.current_params is not None:
                lines.append(f"Parameters: {self.current_params}")
            return "\n".join(lines)

    def close(self):
        """Clean up environment resources."""
        # Clean up simulator resources if needed
        if hasattr(self.simulator, 'close'):
            self.simulator.close()

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector.

        Returns:
            Observation vector
        """
        max_depth = self.config.get("max_depth", 10)
        max_excitations = self.config.get("max_excitations", 20)

        # Energy component (normalized relative to Hartree-Fock)
        # Normalize: (energy - HF) / |HF|, with clipping
        hf_energy = self._get_hf_energy()
        if hf_energy != 0:
            energy_norm = (self.current_energy - hf_energy) / abs(hf_energy)
        else:
            energy_norm = 0.0
        energy_norm = np.clip(energy_norm, -10.0, 10.0)

        # Circuit parameters (pad to max_depth, truncate if params longer than max_depth)
        if self.current_params is not None:
            params = self.current_params.copy()
        else:
            params = np.array([], dtype=np.float32)
        params_padded = np.zeros(max_depth, dtype=np.float32)
        n_copy = min(len(params), max_depth)
        params_padded[:n_copy] = params[:n_copy]

        # Architecture encoding (one-hot of excitation indices)
        arch = np.zeros(max_excitations, dtype=np.float32)
        for exc in self.current_excitations:
            try:
                idx = self.available_excitations.index(exc)
                arch[idx] = 1.0
            except ValueError:
                # Excitation not in available list (should not happen)
                pass

        # Step count (normalized by max steps)
        max_steps = max_excitations  # Could be different
        step_norm = self.step_count / max_steps if max_steps > 0 else 0.0

        # Combine components
        observation = np.concatenate([
            np.array([energy_norm], dtype=np.float32),
            params_padded,
            arch,
            np.array([step_norm], dtype=np.float32)
        ])

        return observation

    def _check_termination(self) -> bool:
        """Check termination conditions.

        Returns:
            True if episode should terminate
        """
        max_depth = self.config.get("max_depth", 10)
        max_excitations = self.config.get("max_excitations", 20)

        # Max circuit depth reached
        if len(self.current_excitations) >= max_depth:
            return True

        # Max excitations reached
        if len(self.current_excitations) >= max_excitations:
            return True

        # Energy convergence (close to FCI)
        if self.molecule_data.fci_energy is not None:
            energy_error = abs(self.current_energy - self.molecule_data.fci_energy)
            if energy_error < self.config.get("convergence_threshold", 1.6e-3):
                return True

        # Max steps reached
        if self.step_count >= max_excitations:
            return True

        return False

    def _get_termination_reason(self) -> str:
        """Get reason for termination.

        Returns:
            String describing termination reason
        """
        max_depth = self.config.get("max_depth", 10)
        max_excitations = self.config.get("max_excitations", 20)

        if len(self.current_excitations) >= max_depth:
            return "max_depth_reached"
        elif len(self.current_excitations) >= max_excitations:
            return "max_excitations_reached"
        elif self.molecule_data.fci_energy is not None:
            energy_error = abs(self.current_energy - self.molecule_data.fci_energy)
            if energy_error < self.config.get("convergence_threshold", 1.6e-3):
                return "energy_convergence"
        elif self.step_count >= max_excitations:
            return "max_steps_reached"
        else:
            return "unknown"