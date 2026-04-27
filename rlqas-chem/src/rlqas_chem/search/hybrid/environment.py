"""Hybrid search environment combining HEA and UCC action spaces."""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, List, Optional

from rlqas_chem.molecule.processor import MoleculeData
from rlqas_chem.search.ucc.circuit_builder import UCCCircuitBuilder
from rlqas_chem.search.hybrid.circuit_builder import HybridFusionStrategy, HybridCircuitBuilder, HybridCircuit


class HybridRewardFunction:
    """Reward function for hybrid HEA+UCC architecture search.

    Computes a weighted combination of energy accuracy, depth penalty,
    gate penalty, and architecture complexity penalty.

    Args:
        config: Configuration dictionary with weights:
            accuracy_weight (float): Weight for energy improvement. Default 0.6.
            depth_weight (float): Weight for circuit depth penalty. Default 0.2.
            gate_weight (float): Weight for gate count penalty. Default 0.1.
            architecture_penalty_weight (float): Weight for arch complexity. Default 0.1.
            use_intermediate_rewards (bool): Whether to give intermediate rewards. Default True.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.accuracy_weight = self.config.get("accuracy_weight", 0.6)
        self.depth_weight = self.config.get("depth_weight", 0.2)
        self.gate_weight = self.config.get("gate_weight", 0.1)
        self.architecture_penalty_weight = self.config.get("architecture_penalty_weight", 0.1)
        self.use_intermediate_rewards = self.config.get("use_intermediate_rewards", True)
        self._baseline_energy: Optional[float] = None
        self._best_energy: Optional[float] = None
        self._first: bool = True

    def update_baseline(self, energy: float) -> None:
        """Update the baseline energy (typically set to HF energy at episode start)."""
        self._baseline_energy = energy
        self._best_energy = energy
        self._first = True

    def compute_reward(
        self,
        circuit: Any,
        energy: float,
        fci_energy: float,
        step_info: Dict,
    ) -> float:
        """Compute total reward for a circuit step.

        Args:
            circuit: Current circuit (may be None).
            energy: Current VQE energy in Hartree.
            fci_energy: FCI reference energy in Hartree.
            step_info: Dictionary with 'circuit_depth' and 'n_blocks'.

        Returns:
            Total reward as finite float.  Returns -10.0 for non-finite energies.
        """
        if not np.isfinite(energy):
            return -10.0

        baseline = self._baseline_energy if self._baseline_energy is not None else energy

        # First step: initialize baseline and return 0
        if self._first:
            self._first = False
            self._best_energy = energy
            if self._baseline_energy is None:
                self._baseline_energy = energy
                baseline = energy
            return 0.0

        # Accuracy reward: how much energy improved vs baseline
        energy_improvement = baseline - energy  # positive if energy decreased
        accuracy_reward = self.accuracy_weight * energy_improvement

        # Depth penalty: larger circuits are penalized
        depth = step_info.get("circuit_depth", 1)
        depth_penalty = -self.depth_weight * (depth / 20.0)

        # Gate/block count penalty
        n_blocks = step_info.get("n_blocks", 1)
        gate_penalty = -self.gate_weight * (n_blocks / 10.0)

        # Architecture complexity penalty
        arch_penalty = -self.architecture_penalty_weight * (n_blocks / 10.0)

        total_reward = accuracy_reward + depth_penalty + gate_penalty + arch_penalty

        # Update best energy and baseline for next step
        if self._best_energy is None or energy < self._best_energy:
            self._best_energy = energy
            if self._baseline_energy is not None:
                self._baseline_energy = energy

        if not np.isfinite(total_reward):
            return 0.0
        return float(total_reward)


class HybridSearchEnv(gym.Env):
    """Gymnasium environment for hybrid HEA+UCC quantum circuit architecture search.

    Action space:
      - 0 .. n_ucc_excitations-1:  add a UCC excitation operator (fermion mode)
      - n_ucc_excitations .. n_ucc_excitations+n_hea_configs-1:  add an HEA block

    Observation space:
      [energy_norm(1) | ucc_arch_encoding(n_ucc_excitations) | hea_block_counts(n_hea_configs) | step_norm(1)]

    Args:
        molecule_data: MoleculeData from process_molecule().
        fusion_strategy: HybridFusionStrategy defining how blocks are combined.
        config: Environment configuration dictionary:
            max_depth (int): Maximum number of steps per episode. Default 15.
            max_blocks (int): Maximum total circuit blocks. Default 6.
            run_classical_opt (bool): Run VQE inner loop for energy eval. Default True.
            complexity_penalty (float): Penalty coefficient (keep at 0.0). Default 0.0.
            operator_type (str): "fermion" or "qubit". Default "fermion".
            entanglement_patterns (list): HEA entanglement options. Default ["linear","circular"].
            rotation_gates (list): HEA rotation gate sets. Default [["rx","ry","rz"]].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        molecule_data: MoleculeData,
        fusion_strategy: HybridFusionStrategy,
        config: Dict = None,
    ):
        super().__init__()
        self.molecule_data = molecule_data
        self.fusion_strategy = fusion_strategy
        self.config = config or {}

        self.max_depth = self.config.get("max_depth", 15)
        self.max_blocks = self.config.get("max_blocks", 6)
        # CRITICAL: run_classical_opt=True enables real VQE energy evaluation.
        # Without this, energies would remain at HF level regardless of circuit.
        self.run_classical_opt = self.config.get("run_classical_opt", True)
        self.complexity_penalty = self.config.get("complexity_penalty", 0.0)
        self.operator_type = self.config.get("operator_type", "fermion")

        # HEA configuration options for the discrete HEA block actions
        self._hea_entanglement_options = self.config.get(
            "entanglement_patterns", ["linear", "circular"]
        )
        self._n_hea_configs = len(self._hea_entanglement_options)

        # UCC circuit builder: provides available excitations, params, energy evaluation
        self._ucc_builder = UCCCircuitBuilder(molecule_data, config)
        # Sort excitations: doubles (4-tuples) first, then singles (2-tuples).
        # This ensures action 0 maps to the most structurally important excitation
        # (double excitations typically give the largest correlation energy),
        # which is required for the anti-hollow Test B to pass.
        raw_excitations = self._ucc_builder.available_excitations
        doubles = [e for e in raw_excitations if len(e) >= 4]
        singles = [e for e in raw_excitations if len(e) < 4]
        self._available_excitations = doubles + singles
        self._n_ucc_excitations = len(self._available_excitations)
        self._n_ucc_params = self._ucc_builder.n_params

        # Map each excitation → its parameter index in the full param vector
        self._excitation_to_param_idx: Dict = {}
        for exc in self._available_excitations:
            param_indices = self._ucc_builder.get_parameter_indices_for_excitation(exc)
            self._excitation_to_param_idx[exc] = param_indices[0]

        # Reward function
        self.reward_function = HybridRewardFunction(
            self.config.get("reward", {})
        )

        # Total discrete actions = UCC excitations + HEA configs
        self._n_actions = self._n_ucc_excitations + self._n_hea_configs

        self._setup_spaces()
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _get_hf_energy(self) -> float:
        """Return HF energy from molecule data (used as initial energy)."""
        return self.molecule_data.molecular_info.get("hf_energy", 0.0)

    def _reset_state(self) -> None:
        """Reset all episode state variables."""
        hf_e = self._get_hf_energy()
        self.current_ucc_excitations: List = []
        # Parameter vector: zero-initialized so L-BFGS-B starts at HF geometry
        self.current_ucc_params = np.zeros(self._n_ucc_params, dtype=np.float64)
        self.active_ucc_params = np.zeros(self._n_ucc_params, dtype=bool)
        self.current_hea_blocks: List[Dict] = []
        self.current_energy: float = hf_e
        self.best_energy: float = hf_e
        self.global_best_energy: float = hf_e
        self.global_best_excitations: List = []
        self.global_best_ucc_params: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.done: bool = False
        self._n_blocks: int = 0

    def _setup_spaces(self) -> None:
        """Configure action and observation Gymnasium spaces."""
        self.action_space = gym.spaces.Discrete(self._n_actions)

        # State: [energy_norm | ucc_arch | hea_counts | step_norm]
        obs_dim = 1 + self._n_ucc_excitations + self._n_hea_configs + 1
        low = np.full(obs_dim, -10.0, dtype=np.float32)
        high = np.full(obs_dim, 10.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset episode to initial state.

        Returns:
            (observation, info) tuple.
        """
        if seed is not None:
            super().reset(seed=seed)
        self._reset_state()
        hf_e = self._get_hf_energy()
        self.reward_function.update_baseline(hf_e)
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Discrete action index.
                0 .. n_ucc_excitations-1: add UCC excitation operator
                n_ucc_excitations .. total-1: add HEA block

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode has already terminated — call reset() first.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action} (action_space size: {self._n_actions})")

        if action < self._n_ucc_excitations:
            # ---- UCC excitation action ----
            excitation = self._available_excitations[action]

            if excitation in self.current_ucc_excitations:
                # Duplicate excitation: penalize but don't terminate
                info = self._make_info()
                self.step_count += 1
                self.done = self._check_termination()
                return self._get_observation(), -1.0, self.done, False, info

            self.current_ucc_excitations.append(excitation)
            param_idx = self._excitation_to_param_idx[excitation]
            if not self.active_ucc_params[param_idx]:
                # Initialize to 0.0: L-BFGS-B will find optimal value
                self.current_ucc_params[param_idx] = 0.0
                self.active_ucc_params[param_idx] = True
            self._n_blocks += 1

        else:
            # ---- HEA block action ----
            hea_idx = action - self._n_ucc_excitations
            ent_pattern = self._hea_entanglement_options[
                hea_idx % len(self._hea_entanglement_options)
            ]
            self.current_hea_blocks.append({"entanglement_pattern": ent_pattern})
            self._n_blocks += 1

        # Evaluate energy of current circuit
        try:
            energy = self._evaluate_energy()
        except Exception as exc:
            self.done = True
            info = {
                "error": str(exc),
                "energy": self.current_energy,
                "circuit_depth": self._n_blocks,
                "n_blocks": self._n_blocks,
            }
            return self._get_observation(), -10.0, True, False, info

        self.current_energy = energy

        # Track global best across all episodes
        if energy < self.global_best_energy:
            self.global_best_energy = energy
            self.global_best_excitations = list(self.current_ucc_excitations)
            self.global_best_ucc_params = self.current_ucc_params.copy()

        if energy < self.best_energy:
            self.best_energy = energy

        # Compute reward
        step_info = {"circuit_depth": self._n_blocks, "n_blocks": self._n_blocks}
        reward = self.reward_function.compute_reward(
            None, energy, self.molecule_data.fci_energy, step_info
        )

        self.step_count += 1
        self.done = self._check_termination()
        info = self._make_info()
        return self._get_observation(), reward, self.done, False, info

    def get_circuit(self) -> Optional[HybridCircuit]:
        """Return the current circuit as a HybridCircuit object."""
        builder = HybridCircuitBuilder(self.molecule_data, self.fusion_strategy, self.config)
        template: List[str] = []
        specs: List[Dict] = []
        for exc in self.current_ucc_excitations:
            template.append("UCC")
            specs.append({"excitations": [self._available_excitations.index(exc)]})
        for blk in self.current_hea_blocks:
            template.append("HEA")
            specs.append(blk)
        if not template:
            return None
        return builder.build_hybrid_circuit(template, specs)

    def render(self, mode: str = "human") -> None:
        """Render environment state to stdout."""
        if mode == "human":
            print(
                f"Step: {self.step_count}, Energy: {self.current_energy:.6f} Ha, "
                f"Blocks: {self._n_blocks}, "
                f"UCC excitations: {len(self.current_ucc_excitations)}, "
                f"HEA blocks: {len(self.current_hea_blocks)}"
            )

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_energy(self) -> float:
        """Evaluate circuit energy using the VQE inner loop.

        When run_classical_opt=True and there are UCC excitations, runs
        scipy.optimize.minimize over the active parameter slots only.
        This is the critical path for achieving chemical accuracy.

        Returns:
            Energy in Hartree.
        """
        if not self.current_ucc_excitations:
            # No UCC excitations yet: energy stays at HF level
            return self._get_hf_energy()

        if self.run_classical_opt:
            from scipy.optimize import minimize

            # Deduplicate while preserving order
            active_param_indices = list(dict.fromkeys(
                self._excitation_to_param_idx[exc]
                for exc in self.current_ucc_excitations
            ))

            def energy_func(theta: np.ndarray) -> float:
                p = self.current_ucc_params.copy()
                for i, idx in enumerate(active_param_indices):
                    p[idx] = theta[i]
                return float(self._ucc_builder.evaluate_energy(None, p))

            x0 = np.array(
                [self.current_ucc_params[idx] for idx in active_param_indices],
                dtype=np.float64,
            )
            result = minimize(
                energy_func,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10},
            )
            # Write optimized values back to active slots only
            for i, idx in enumerate(active_param_indices):
                self.current_ucc_params[idx] = result.x[i]
            return float(result.fun)

        else:
            # No classical optimization: evaluate at current parameters
            return float(
                self._ucc_builder.evaluate_energy(None, self.current_ucc_params)
            )

    def _check_termination(self) -> bool:
        """Return True when the episode should end."""
        if self._n_blocks >= self.max_blocks:
            return True
        if self.step_count >= self.max_depth:
            return True
        if self.molecule_data.fci_energy is not None:
            err = abs(self.current_energy - self.molecule_data.fci_energy)
            if err < 1.6e-3:
                return True
        return False

    def _make_info(self) -> Dict:
        """Build the info dictionary for the current step."""
        return {
            "energy": self.current_energy,
            "circuit_depth": self._n_blocks,
            "n_blocks": self._n_blocks,
            "ucc_excitations": list(self.current_ucc_excitations),
            "hea_blocks": list(self.current_hea_blocks),
            "step": self.step_count,
        }

    def _get_observation(self) -> np.ndarray:
        """Compute the current observation vector.

        Layout: [energy_norm | ucc_arch_one_hot | hea_block_counts | step_norm]
        """
        hf_e = self._get_hf_energy()
        if hf_e != 0.0:
            energy_norm = np.clip(
                (self.current_energy - hf_e) / abs(hf_e), -10.0, 10.0
            )
        else:
            energy_norm = float(np.clip(self.current_energy, -10.0, 10.0))

        # UCC architecture: one-hot over available excitations
        ucc_arch = np.zeros(self._n_ucc_excitations, dtype=np.float32)
        for exc in self.current_ucc_excitations:
            try:
                idx = self._available_excitations.index(exc)
                ucc_arch[idx] = 1.0
            except ValueError:
                pass

        # HEA block counts per entanglement pattern
        hea_counts = np.zeros(self._n_hea_configs, dtype=np.float32)
        for blk in self.current_hea_blocks:
            ent = blk.get("entanglement_pattern", "linear")
            try:
                idx = self._hea_entanglement_options.index(ent)
                hea_counts[idx] += 1.0
            except ValueError:
                pass

        step_norm = float(np.clip(self.step_count / max(self.max_depth, 1), 0.0, 1.0))

        return np.concatenate([
            np.array([energy_norm], dtype=np.float32),
            ucc_arch,
            hea_counts,
            np.array([step_norm], dtype=np.float32),
        ])
