"""
HEA (Hardware Efficient Ansatz) Search Module for RLQAS Phase 2.

This module implements the HEASearchEnv class for Hardware Efficient Ansatz
architecture search with configurable entanglement patterns.
"""

import os
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HEASearchEnv(gym.Env):
    """Gym environment for Hardware Efficient Ansatz (HEA) search.

    This environment allows an RL agent to search for optimal HEA circuit
    architectures by selecting entanglement patterns, rotation gates, and
    parameter sharing strategies.

    The environment supports:
    - Multiple entanglement patterns (linear, circular, fully connected)
    - Configurable rotation gate types (rx, ry, rz)
    - Parameter sharing strategies (layer-wise, global, none)
    - Variable circuit depth

    Args:
        n_qubits: Number of qubits in the circuit
        max_layers: Maximum number of layers in the HEA
        entanglement_patterns: List of allowed entanglement patterns
        rotation_gates: List of allowed rotation gate types
        parameter_sharing: Parameter sharing strategy
    """

    # Entanglement pattern constants
    ENTANGLEMENT_LINEAR = 0
    ENTANGLEMENT_CIRCULAR = 1
    ENTANGLEMENT_FULL = 2

    # Rotation gate constants
    ROTATION_RX = 0
    ROTATION_RY = 1
    ROTATION_RZ = 2

    # Parameter sharing constants
    SHARING_NONE = 0
    SHARING_LAYER_WISE = 1
    SHARING_GLOBAL = 2

    def __init__(
        self,
        n_qubits: int = 4,
        max_layers: int = 4,
        entanglement_patterns: Optional[List[str]] = None,
        rotation_gates: Optional[List[str]] = None,
        parameter_sharing: str = "layer_wise",
        target_energy: Optional[float] = None,
    ):
        """Initialize HEA search environment."""
        super().__init__()

        self.n_qubits = n_qubits
        self.max_layers = max_layers
        self.target_energy = target_energy

        # Configuration
        self.entanglement_patterns = entanglement_patterns or ["linear", "circular", "full"]
        self.rotation_gates = rotation_gates or ["rx", "ry", "rz"]
        self.parameter_sharing = parameter_sharing

        # Action space: select layer configuration
        # Each action selects: entanglement pattern + rotation gate type
        n_entanglement = len(self.entanglement_patterns)
        n_rotation = len(self.rotation_gates)
        self.action_space = spaces.Discrete(n_entanglement * n_rotation)

        # State space: current circuit configuration
        # [layer_idx, energy, params...]
        state_dim = 3 + (max_layers * n_qubits)  # layer, energy, params
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Environment state
        self._current_layer = 0
        self._circuit_params: Optional[np.ndarray] = None
        self._current_energy: float = 0.0
        self._episode_reward: float = 0.0
        self._entanglement_history: List[int] = []
        self._rotation_history: List[int] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed
            options: Optional configuration options

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        self._current_layer = 0
        self._circuit_params = self._initialize_params()
        self._current_energy = self._compute_energy()
        self._episode_reward = 0.0
        self._entanglement_history = []
        self._rotation_history = []

        observation = self._get_observation()
        info = {
            "layer": self._current_layer,
            "energy": self._current_energy,
        }

        return observation.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Action index selecting entanglement and rotation

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # Parse action
        n_rotation = len(self.rotation_gates)
        entanglement_idx = action // n_rotation
        rotation_idx = action % n_rotation

        # Validate indices
        entanglement_idx = min(entanglement_idx, len(self.entanglement_patterns) - 1)
        rotation_idx = min(rotation_idx, len(self.rotation_gates) - 1)

        # Apply action to circuit
        self._apply_action(entanglement_idx, rotation_idx)

        # Compute new energy
        new_energy = self._compute_energy()

        # Compute reward (energy improvement)
        energy_delta = self._current_energy - new_energy
        reward = energy_delta

        # Update state
        self._current_energy = new_energy
        self._episode_reward += reward
        self._current_layer += 1

        # Check if episode is done
        done = self._current_layer >= self.max_layers
        truncated = False

        observation = self._get_observation()
        info = {
            "layer": self._current_layer,
            "energy": self._current_energy,
            "entanglement": self.entanglement_patterns[entanglement_idx],
            "rotation": self.rotation_gates[rotation_idx],
            "energy_delta": energy_delta,
            "episode_reward": self._episode_reward,
        }

        return observation.astype(np.float32), reward, done, truncated, info

    def _initialize_params(self) -> np.ndarray:
        """Initialize circuit parameters randomly.

        Returns:
            Initial parameter array
        """
        n_params = self.max_layers * self.n_qubits * 3  # 3 rotation types per qubit
        return self.np_random.uniform(-np.pi, np.pi, size=n_params)

    def _apply_action(self, entanglement_idx: int, rotation_idx: int):
        """Apply action to modify circuit parameters.

        Args:
            entanglement_idx: Index of entanglement pattern
            rotation_idx: Index of rotation gate type
        """
        self._entanglement_history.append(entanglement_idx)
        self._rotation_history.append(rotation_idx)

        # Modify parameters based on action
        layer_start = self._current_layer * self.n_qubits * 3
        for q in range(self.n_qubits):
            param_idx = layer_start + q * 3 + rotation_idx
            if param_idx < len(self._circuit_params):
                # Small perturbation based on entanglement pattern
                scale = 0.1 * (entanglement_idx + 1)
                self._circuit_params[param_idx] += self.np_random.uniform(-scale, scale)

    def _compute_energy(self) -> float:
        """Compute circuit energy (simulated).

        In a real implementation, this would call a quantum simulator
        or use pre-computed integrals.

        Returns:
            Simulated energy value
        """
        if self._circuit_params is None:
            return 0.0

        # Simplified energy computation based on parameters
        # In practice, this would be <psi|H|psi> for the Hamiltonian H
        param_mean = np.mean(np.abs(self._circuit_params))
        base_energy = -1.0  # Reference energy

        # Energy depends on parameter values and entanglement history
        entanglement_bonus = 0.0
        if self._entanglement_history:
            # Favor certain entanglement patterns
            pattern_counts = {}
            for p in self._entanglement_history:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1
            # Bonus for diversity
            entanglement_bonus = 0.01 * len(pattern_counts)

        # Energy improvement with more layers (up to a point)
        layer_factor = min(1.0, self._current_layer / self.max_layers)

        energy = base_energy - (param_mean * 0.01) - entanglement_bonus - (layer_factor * 0.1)

        # Add target energy offset if specified
        if self.target_energy is not None:
            energy += self.target_energy

        return energy

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation array
        """
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        obs[0] = self._current_layer / self.max_layers  # Normalized layer
        obs[1] = self._current_energy  # Current energy

        # Entanglement pattern encoding
        if self._entanglement_history:
            obs[2] = self._entanglement_history[-1] / len(self.entanglement_patterns)
        else:
            obs[2] = 0.0

        # Circuit parameters (flattened, limited size)
        if self._circuit_params is not None:
            n_param_slots = len(obs) - 3
            params = self._circuit_params[:n_param_slots]
            obs[3:3+len(params)] = params

        return obs

    def get_circuit_config(self) -> Dict[str, Any]:
        """Get current circuit configuration.

        Returns:
            Dictionary containing circuit configuration
        """
        return {
            "n_qubits": self.n_qubits,
            "max_layers": self.max_layers,
            "current_layer": self._current_layer,
            "entanglement_history": [
                self.entanglement_patterns[i] for i in self._entanglement_history
            ],
            "rotation_history": [
                self.rotation_gates[i] for i in self._rotation_history
            ],
            "parameter_sharing": self.parameter_sharing,
            "current_energy": self._current_energy,
            "n_parameters": len(self._circuit_params) if self._circuit_params is not None else 0,
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment.

        Args:
            mode: Render mode ("human" or "ansi")

        Returns:
            Rendered string if mode is "ansi"
        """
        config = self.get_circuit_config()
        output = f"HEA Circuit (Layer {self._current_layer}/{self.max_layers})\n"
        output += f"  Qubits: {self.n_qubits}\n"
        output += f"  Energy: {self._current_energy:.6f}\n"
        output += f"  Entanglement: {config['entanglement_history']}\n"
        output += f"  Rotations: {config['rotation_history']}\n"

        if mode == "human":
            print(output)
        return output
